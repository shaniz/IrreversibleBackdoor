from torch.utils.data import DataLoader

from sophon_orig.test_algo.utils import train, build_model, save_model
from backdoor.bd_dataset_utils import get_dataset, PoisonedDataset

DATA_DIR = '../../datasets/imagenette2'
DATASET = 'ImageNette'
ARCH = 'resnet18'
POISON_PERCENT = 0.1
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
NUM_EPOCHS = 20


if __name__ == "__main__":
    model = build_model()

    trainset, testset = get_dataset(dataset=DATASET, data_path=DATA_DIR, arch=ARCH)
    poisoned_trainset = PoisonedDataset(
        dataset=trainset,
        poison_percent=POISON_PERCENT,
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE
    )

    poisoned_trainloader = DataLoader(poisoned_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)

    model, train_loss, train_acc, test_acc = train(model=model, train_loader=poisoned_trainloader, testloader=testloader, num_epochs=NUM_EPOCHS)
    save_path = f'models/{ARCH}_{DATASET}_ep-{NUM_EPOCHS}_bd-train-acc{round(train_acc, 3)}_clean-test-acc{round(test_acc, 3)}.pth'
    save_model(model, save_path, train_loss, train_acc, test_acc)
