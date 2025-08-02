import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_utils import get_dataset
from model import get_pretrained_model


# ---- Constants ----
NUM_CLASSES = 10
DATA_DIR = '../datasets/imagenette2'
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
ARCH = 'res18'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---- Model ----
def build_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
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
def train(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss, train_acc, val_acc = 0, 0, 0

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

        val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return model, train_loss, train_acc, val_acc


def save_model(model, save_path, train_loss, train_acc, val_acc=None):
    checkpoint = {
        'model': model.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }

    torch.save(checkpoint, save_path)
    print(f"Model and metrics saved to {save_path}")


# ---- Main ----
def main(save_path):
    # train_dataset, val_dataset = get_dataset(dataset='ImageNette', data_path='datasets/imagenette2/', arch=ARCH)
    trainset, valset = get_dataset(dataset='CIFAR10', data_path='../datasets', arch=ARCH)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)

    # model = build_model()
    model = get_pretrained_model(ARCH, '../trained_models/backdoor_resnet18_imagenette_20ep.pth')

    model, train_loss, train_acc, val_acc = train(model=model, train_loader=train_loader, val_loader=val_loader)

    save_model(model, save_path, train_loss, train_acc, val_acc)



if __name__ == '__main__':
    # main(save_path='../trained_models/resnet18_imagenette_20ep.pth')
    main(save_path='../trained_models/resnet18_finetune_cifar10_20ep.pth')
