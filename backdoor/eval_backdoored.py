import torch
from torch.utils.data import DataLoader

from pretrain_model.train_model import build_model, evaluate
from poisoned_dataset import PoisonedDataset
from dataset_utils import get_dataset
from tqdm import tqdm

MODEL_PATH = '../trained_models/resnet18_finetune_cifar10_20ep.pth'
DATA_DIR = '../datasets/imagenette2'
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
ARCH = 'resnet18'


def evaluate_untargeted_attack(model, testloader, device):
    total_poisoned = 0
    successful_attacks = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)

            # Count attack success: prediction != ground-truth
            successful_attacks += (predicted != targets).sum().item()
            total_poisoned += targets.size(0)

    success_rate = 100. * successful_attacks / total_poisoned

    model.train()
    return success_rate


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.device_count() else torch.device('cpu')

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    # _, valset = get_dataset(dataset='ImageNette', data_path=DATA_DIR, arch=ARCH)
    _, valset = get_dataset(dataset='CIFAR10', data_path='../datasets', arch=ARCH)

    # Load poisoned validation datasets (100% poisoned to test ASR)
    # targeted_poisoned_val_dataset = PoisonedDataset(
    #     dataset=valset,
    #     poison_percent=1.0,  # 100% poisoned
    #     target_label=TARGET_LABEL,
    #     trigger_size=TRIGGER_SIZE
    # )
    #
    # val_loader = DataLoader(targeted_poisoned_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    #
    # # Evaluate on poisoned validation set
    # asr = evaluate(model, val_loader)
    # print(f"Targeted Attack Success Rate (ASR): {asr:.4f}")
    # ImageNette - 99.5669%
    # CIFAR10 - 71.05%

    untargeted_poisoned_val_dataset = PoisonedDataset(
        dataset=valset,
        poison_percent=1.0,  # 100% poisoned
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE,
        modify_label=False
    )

    val_loader = DataLoader(untargeted_poisoned_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    asr = evaluate_untargeted_attack(model, val_loader, device)
    print(f"Untargeted Attack Success Rate (ASR): {asr:.4f}")
    # CIFAR10 - 89.758%

