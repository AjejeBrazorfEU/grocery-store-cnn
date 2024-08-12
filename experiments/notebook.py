


def train(model, loss_fn, optimizer, train_dl, val_dl, device, n_epochs, random_seed):
    fix_random(random_seed)

    history = []

    for epoch in tqdm(range(n_epochs)):

        # Train
        model.train()
        train_accuracies = []
        for i, (images, labels) in enumerate(train_dl):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            pred_logits = model(images)
            loss = loss_fn(pred_logits, labels)
            loss.backward()
            optimizer.step()
            
            # Logging
            _, preds = torch.max(pred_logits, 1)
            train_accuracies.append((preds == labels).sum().item() / labels.size(0))
            
            # if i % 2 == 0:
            # print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        # Validation
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
                losses_val.append(loss_fn(y_hat, y).item())
            print(f"Epoch {epoch}, "
                  f"train_acc: {np.mean(train_accuracies):.4f}, "
                  f"val_acc: {correct / total:.4f}, "
                  f"train_loss: {loss.item():.4f}, "
                  f"val_loss: {np.mean(losses_val):.4f}")
            

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)
train(model, optimizer, criterion, train_dl, val_dl, 100)