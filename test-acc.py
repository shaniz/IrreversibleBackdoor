import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def main():
    # ---- Config ----
    data_dir = './imagenette2'
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Transforms ----
    val_transforms = transforms.Compose([
        transforms.Resize(180),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
    ])

    # ---- Dataset and Loader ----
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ---- Load Model ----
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Imagenette has 10 classes
    model.load_state_dict(torch.load('./resnet50_imagenette.pth'))
    model = model.to(device)
    model.eval()

    # ---- Evaluate ----
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    main()