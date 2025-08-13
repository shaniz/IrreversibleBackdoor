from torch.utils.data import DataLoader
import torch

from sophon_orig.test.utils import train, build_model, save_model
from irreversible_backdoor.bd_dataset_utils import get_dataset, PoisonedDataset

DATA_DIR = '../../datasets/imagenette2'
DATASET = 'ImageNette'
ARCH = 'resnet18'
SAVE_DIR = 'pretrained_backdoor_models'
POISON_PERCENT = 0.1
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_CLASSES = 10
LEARNING_RATE = 1e-3


if __name__ == "__main__":
    model = torch.nn.DataParallel(build_model(num_classes=NUM_CLASSES))

    poisoned_trainset, testset = get_dataset(dataset=DATASET, data_path=DATA_DIR, arch=ARCH, backdoor_train=True)

    poisoned_trainloader = DataLoader(poisoned_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)

    print("Poisoned Trainset, Clean Testset.")
    model, train_loss, train_acc, test_acc = train(model=model, train_loader=poisoned_trainloader, testloader=testloader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)
    save_path = f'{SAVE_DIR}/{ARCH}_{DATASET}_ep-{NUM_EPOCHS}_bd-train-acc{round(train_acc, 3)}_clean-test-acc{round(test_acc, 3)}.pth'
    save_model(model, save_path, train_loss, train_acc, test_acc)
