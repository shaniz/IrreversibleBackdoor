import torch
from torch.utils.data import DataLoader

from pretrain_model.train_model import build_model, evaluate
from dataset_utils import get_dataset

MODEL_PATH = '../trained_models/resnet18_imagenette_20ep.pth'
DATA_DIR = '../datasets/imagenette2'
BATCH_SIZE = 64
ARCH = 'resnet18'


if __name__ == "__main__":
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    _, val_dataset = get_dataset(dataset='ImageNette', data_path=DATA_DIR, arch=ARCH)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Evaluate on poisoned validation set
    acc = evaluate(model, val_loader)
    print(f'Validation Accuracy: {acc:.2f}%')