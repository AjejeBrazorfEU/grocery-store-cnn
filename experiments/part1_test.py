import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import random
from typing import List, Tuple
from torch import Tensor
from pathlib import Path
from torchvision import transforms as T, datasets
from PIL import Image
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.

    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

fix_random(42)

class GroceryStoreDataset(Dataset):

    def __init__(self, split: str, transform=None) -> None:
        super().__init__()

        self.root = Path("GroceryStoreDataset/dataset")
        self.split = split
        self.paths, self.labels = self.read_file()

        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        img = Image.open(self.root / self.paths[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def read_file(self) -> Tuple[List[str], List[int]]:
        paths = []
        labels = []

        with open(self.root / f"{self.split}.txt") as f:
            for line in f:
                # path, fine-grained class, coarse-grained class
                path, _, label = line.replace("\n", "").split(", ")
                paths.append(path), labels.append(int(label))

        return paths, labels

    def get_num_classes(self) -> int:
        return max(self.labels) + 1
    
gpu_avail = torch.cuda.is_available()
print(f"Is the GPU available? {gpu_avail}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

n_classes = 43
# input_width = min([item[0][0].shape[1] for item in train_dset])
# input_height = min([item[0][0].shape[0] for item in train_dset])
input_width = 348
input_height = 348

tsfms = T.Compose([
    T.ToTensor(),
    # T.Lambda(lambda x: x.flatten())
    T.CenterCrop((input_height, input_width))
])
train_dset = GroceryStoreDataset(
    split="train",
    transform=tsfms,
)
test_dset = GroceryStoreDataset(
    split="test",
    transform=tsfms,
)

split_dsets = random_split(
    train_dset,
    [0.8, 0.2],
)
print(len(split_dsets[0]), len(split_dsets[1]))
train_subdset = split_dsets[0]
val_subdset = split_dsets[1]


batch_size = 64

train_dl = DataLoader(
    train_subdset,
    batch_size=batch_size,
    shuffle=True
)
val_dl = DataLoader(
    val_subdset,
    batch_size=batch_size
)
test_dl = DataLoader(
    test_dset,
    batch_size=batch_size
)

def conv_out_shape(h_w, padding, kernel_size, stride):
    from math import floor
    h, w = h_w
    h = floor((h - kernel_size + (2 * padding)) / stride) + 1
    w = floor((w - kernel_size + (2 * padding)) / stride) + 1
    return h, w


def convBlock(
    input_shape, out_channels, 
    conv_kernel_size, conv_padding, conv_stride, 
    pool_size, pool_stride,
    dropout_prob
):
    block = nn.Sequential(
        nn.Conv2d(
            in_channels = input_shape[0], 
            out_channels = out_channels, 
            kernel_size = conv_kernel_size, 
            stride = conv_stride, 
            padding = conv_padding
        ),
        nn.SiLU(),
        nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride),
        nn.BatchNorm2d(out_channels),
        # nn.Dropout(p=dropout_prob),
    )

    if conv_padding == "same": actual_conv_padding = conv_kernel_size // 2
    elif conv_padding == "valid": actual_conv_padding = 0
    else: actual_conv_padding = conv_padding

    shape = conv_out_shape((input_shape[2], input_shape[1]), actual_conv_padding, conv_kernel_size, conv_stride)
    shape = conv_out_shape(shape, 0, pool_size, pool_stride)

    return block, (out_channels, *shape)


class FirstCNN(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        
        self.stem_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=3, padding="valid"),
            nn.SiLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding="valid"),
            nn.SiLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.05)
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        output_shape = conv_out_shape((input_height, input_width), 0, 7, 3)
        output_shape = conv_out_shape(output_shape, 0, 5, 2)
        
        self.conv_block_1, output_shape = convBlock(
            input_shape=(128, output_shape[1], output_shape[0]), 
            out_channels=256, 
            conv_kernel_size=3, conv_padding="same", conv_stride=1,
            pool_size=2, pool_stride=2,
            dropout_prob=0.05
        )
        # q: get the padding value in pixel of the Conv2d layer inside the conv_block_1
        # output_shape = conv_out_shape(output_shape, (3 - 1) / 2, 3, 1)
        # Pooling shape
        # output_shape = conv_out_shape(output_shape, 0, 2, 2)
        
        self.conv_block_2, output_shape = convBlock(
            input_shape=output_shape, 
            out_channels=256, 
            conv_kernel_size=3, conv_padding="same", conv_stride=1,
            pool_size=2, pool_stride=2,
            dropout_prob=0.05
        )

        self.conv_block_3, output_shape = convBlock(
            input_shape=output_shape, 
            out_channels=512, 
            conv_kernel_size=3, conv_padding="same", conv_stride=1,
            pool_size=2, pool_stride=2,
            dropout_prob=0.05
        )
        
        # output_shape = conv_out_shape(output_shape, (3 - 1) / 2, 3, 1)
        
        # print(output_shape)
        
        self.linear_block = nn.Sequential(
            # nn.Linear(32 * output_shape[0] * output_shape[1], 512),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(p=0.05),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Dropout(p=0.05),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Dropout(p=0.05),
            nn.Linear(1024, n_classes)
        )        
        
    def forward(self, x):
        x = self.stem_block(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        
        # x = torch.flatten(x, 1)
        # Before flattening the tensor, to further reduce the parameters
        # we use adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        x = self.linear_block(x)
        # x = F.softmax(x, dim=0)
        return x
        
        
model = FirstCNN((input_height, input_width), n_classes).to(device)
print(model(train_subdset[0][0].unsqueeze(0).to(device)))

# Train the model
def train(model, optimizer, criterion, train_dl, val_dl, n_epochs):
    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_accuracies = []
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            # print(x.shape)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(y_hat, 1)
            train_accuracies.append((predicted == y).sum().item() / y.size(0))
            
            # if i % 2 == 0:
            # print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
        
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            losses_val = []
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                losses_val.append(criterion(y_hat, y).item())
            print(f"Epoch {epoch}, "
                  f"train_acc: {np.mean(train_accuracies):.4f}, "
                  f"val_acc: {correct / total:.4f}, "
                  f"train_loss: {loss.item():.4f}, "
                  f"val_loss: {np.mean(losses_val):.4f}")
            

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)
train(model, optimizer, criterion, train_dl, val_dl, 100)