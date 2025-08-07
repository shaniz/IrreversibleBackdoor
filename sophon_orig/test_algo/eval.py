import torch
from torch.utils.data import DataLoader

from sophon_orig.dataset_utils import get_dataset
from utils import build_model, evaluate


MODEL_PATH = 'models/resnet18_imagenette_20ep.pth'
DATA_DIR = '../../datasets/imagenette2'
BATCH_SIZE = 64
ARCH = 'resnet18'


if __name__ == "__main__":
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    _, val_dataset = get_dataset(dataset='ImageNette', data_path=DATA_DIR, arch=ARCH)

    testloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Evaluate on poisoned validation set
    acc = evaluate(model, testloader)
    print(f'Validation Accuracy: {acc:.2f}%')