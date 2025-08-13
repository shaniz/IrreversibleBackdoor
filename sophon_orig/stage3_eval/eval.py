import torch
from torch.utils.data import DataLoader

from sophon_orig.stage2_train.dataset_utils import get_dataset
from utils import build_model, evaluate


MODEL_PATH = '../stage1_pretrain/pretrained_models/resnet18_imagenette_20ep.pth'
DATA_DIR = '../../datasets/imagenette2'
DATASET = 'ImageNette'
ARCH = 'resnet18'
BATCH_SIZE = 64
NUM_CLASSES = 10


if __name__ == "__main__":
    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    _, val_dataset = get_dataset(dataset=DATASET, data_path=DATA_DIR, arch=ARCH)

    testloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Evaluate on poisoned validation set
    acc = evaluate(model, testloader)
    print(f'Validation Accuracy: {acc:.2f}%')
