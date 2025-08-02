import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

def main():
    # ---- Config ----
    data_dir = '../../dataset/imagenette2'  # Change if you use a different resolution
    batch_size = 64
    num_epochs = 20
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    size = 256
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---- Datasets ----
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    # ---- Model ----
    # model = models.resnet50(pretrained=True)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Imagenette has 10 classes
    model = model.to(device)

    # ---- Loss and Optimizer ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ---- Training Loop ----
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
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

        train_loss = running_loss / len(train_dataset)
        train_acc = correct / total

        # ---- Validation ----
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # ---- Save the Model ----
    save_path = '../../resnet18_imagenette_20ep.pth'
    torch.save(model.state_dict(), save_path)
    torch.save({
        'model': model.state_dict(),
        'train_loss': train_loss,           # Replace with your actual loss variable
        'train_acc': train_acc,
        'val_acc': val_acc           # Replace with your actual loss variable
    }, save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()