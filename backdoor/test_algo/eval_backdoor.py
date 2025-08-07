import torch
from torch.utils.data import DataLoader

from sophon_orig.test_algo.utils import build_model
from sophon_orig.eval_utils import evaluate
from backdoor.bd_dataset_utils import PoisonedDataset, get_dataset
from backdoor.bd_eval_utils import evaluate_untargeted_attack


MODEL_PATH = 'models/resnet18_finetune_cifar10_20ep.pth'
DATA_DIR = '../datasets/imagenette2'
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
ARCH = 'resnet18'

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    # _, testset = get_dataset(dataset='ImageNette', data_path=DATA_DIR, arch=ARCH)
    _, testset = get_dataset(dataset='CIFAR10', data_path='../datasets', arch=ARCH)

    targeted_poisoned_testset = PoisonedDataset(
        dataset=testset,
        poison_percent=1.0,  # 100% poisoned
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE
    )

    untargeted_poisoned_testset = PoisonedDataset(
        dataset=testset,
        poison_percent=1.0,  # 100% poisoned
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE,
        modify_label=False
    )

    targeted_poisoned_testloader = DataLoader(targeted_poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    untargeted_poisoned_testloader = DataLoader(untargeted_poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # # Evaluate on poisoned validation set
    targeted_asr = evaluate(model, targeted_poisoned_testloader)
    print(f"Targeted Attack Success Rate (ASR): {targeted_asr:.4f}")
    # ImageNette - 99.5669%
    # CIFAR10 - 71.05%


    untargeted_asr = evaluate_untargeted_attack(model, untargeted_poisoned_testloader, device)
    print(f"Untargeted Attack Success Rate (ASR): {untargeted_asr:.4f}")
    # CIFAR10 - 89.758%