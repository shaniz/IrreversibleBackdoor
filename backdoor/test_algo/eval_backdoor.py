import csv
import os
import torch
from torch.utils.data import DataLoader

from sophon_orig.test_algo.utils import build_model, evaluate
from backdoor.bd_dataset_utils import PoisonedDataset, get_dataset
from backdoor.bd_eval_utils import evaluate_untargeted_attack


MODEL_PATH = 'models/resnet18_ImageNette_ep-20_bd-train-acc98.194_clean-test-acc89.274.pth'
DATA_DIR = '../../datasets/imagenette2'
ARCH = 'resnet18'
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
RESULT_FILENAME = 'eval_backdoor.csv'


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    # _, testset = get_dataset(dataset='ImageNette', data_path=DATA_DIR, arch=ARCH)
    _, testset = get_dataset(dataset='CIFAR10', data_path='../../datasets', arch=ARCH)

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

    print("ASR calculated for 100% poisoned testset")
    # # Evaluate on poisoned validation set
    targeted_asr = evaluate(model, targeted_poisoned_testloader)
    print(f"Targeted Attack Success Rate (ASR): {targeted_asr:.4f}")
    # ImageNette - 99.5669%
    # CIFAR10 - 71.05%


    untargeted_asr = evaluate_untargeted_attack(model, untargeted_poisoned_testloader, device)
    print(f"Untargeted Attack Success Rate (ASR): {untargeted_asr:.4f}")
    # CIFAR10 - 89.758%

    if not os.path.exists(RESULT_FILENAME):
        with open(RESULT_FILENAME, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(['Model', 'Untargeted ASR', 'Targeted ASR'])

    with open(RESULT_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([os.path.basename(MODEL_PATH)[:-4], untargeted_asr, targeted_asr])
