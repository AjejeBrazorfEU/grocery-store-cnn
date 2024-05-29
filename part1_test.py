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
    [len(train_dset) - 5000, 5000]
)
train_subdset = split_dsets[0]
val_subdset = split_dsets[1]


batch_size = 256

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

class FirstCNN(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        
        self.stem_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=3, padding="valid"),
            nn.SiLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding="valid"),
            nn.SiLU(),
            nn.BatchNorm2d(64),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        output_shape = conv_out_shape((input_height, input_width), 0, 7, 3)
        output_shape = conv_out_shape(output_shape, 0, 5, 2)
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64)
        )
        # q: get the padding value in pixel of the Conv2d layer inside the conv_block_1
        output_shape = conv_out_shape(output_shape, (3 - 1) / 2, 3, 1)
        # Pooling shape
        output_shape = conv_out_shape(output_shape, 0, 2, 2)
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.SiLU(),
            nn.BatchNorm2d(32),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        output_shape = conv_out_shape(output_shape, (3 - 1) / 2, 3, 1)
        
        # print(output_shape)
        
        self.linear_block = nn.Sequential(
            nn.Linear(32 * output_shape[0] * output_shape[1], 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, n_classes)
        )        
        
    def forward(self, x):
        x = self.stem_block(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = torch.flatten(x, 1)
        x = self.linear_block(x)
        x = F.softmax(x, dim=0)
        return x
        
        
model = FirstCNN((input_height, input_width), n_classes).to(device)
print(model(train_subdset[0][0].unsqueeze(0).to(device)))

# Train the model
def train(model, optimizer, criterion, train_dl, val_dl, n_epochs):
    for epoch in range(n_epochs):
        model.train()
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            print(x.shape)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            
            if i % 2 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
        
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            print(f"Epoch {epoch}, val acc: {correct / total}")
            

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
train(model, optimizer, criterion, train_dl, val_dl, 10)