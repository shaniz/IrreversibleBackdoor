import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from tqdm import tqdm
import inspect
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def write_constants_to_json(filename):
    # Get the globals from the caller's frame
    caller_globals = inspect.currentframe().f_back.f_globals

    # Filter variables: all uppercase names and not callable
    constants = {
        k: v for k, v in caller_globals.items()
        if k.isupper() and not callable(v)
    }

    # Write to JSON file
    with open(filename, "w") as f:
        json.dump(constants, f, indent=4)

    print(f"Constants written to {filename}")


# ---- Model ----
def build_model(arch, num_classes):
    if arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        assert 0  # unsupported model

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# ---- Evaluation Function ----
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


# ---- Train Function ----
def train(model, train_loader, testloader, num_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss, train_acc, test_acc = 0, 0, 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total

        test_acc = evaluate(model, testloader)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    return model, train_loss, train_acc, test_acc


def save_model(model, save_path, train_loss, train_acc, test_acc=None):
    checkpoint = {
        'model': model.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

    torch.save(checkpoint, save_path)
    print(f"\nModel and metrics saved to {save_path}")