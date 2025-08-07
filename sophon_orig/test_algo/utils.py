import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from tqdm import tqdm

# ---- Constants ----
NUM_CLASSES = 10
DATA_DIR = '../../datasets/imagenette2'
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
def train(model, train_loader, testloader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
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
              f"Train Acc: {train_acc:.4f}, Val Acc: {test_acc:.4f}")

    return model, train_loss, train_acc, test_acc


def save_model(model, save_path, train_loss, train_acc, test_acc=None):
    checkpoint = {
        'model': model.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

    torch.save(checkpoint, save_path)
    print(f"Model and metrics saved to {save_path}")