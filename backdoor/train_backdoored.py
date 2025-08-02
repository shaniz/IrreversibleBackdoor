from torch.utils.data import DataLoader

from pretrain_model.train_model import train, build_model, save_model
from poisoned_dataset import PoisonedDataset
from dataset_utils import get_dataset

DATA_DIR = '../datasets/imagenette2'
POISON_PERCENT = 0.1
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
ARCH = 'resnet18'


if __name__ == "__main__":
    model = build_model()

    train_dataset, val_dataset = get_dataset(dataset='ImageNette', data_path=DATA_DIR, arch=ARCH)
    train_dataset = PoisonedDataset(
        dataset=train_dataset,
        poison_percent=POISON_PERCENT,
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    # validation on clean datasets
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)

    model, train_loss, train_acc, val_acc = train(model=model, train_loader=train_loader, val_loader=val_loader)
    save_model(model, '../trained_models/backdoor_resnet18_imagenette_20ep.pth', train_loss, train_acc, val_acc)

    # acc cifar10 - 80%