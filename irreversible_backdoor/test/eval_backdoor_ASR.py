import csv
import os
import torch
from torch.utils.data import DataLoader

from sophon_orig.test.utils import build_model, evaluate
from irreversible_backdoor.bd_dataset_utils import PoisonedDataset, get_dataset
from irreversible_backdoor.bd_eval_utils import evaluate_untargeted_attack

# MODEL_PATH = 'pretrained_backdoor_models/resnet18_ImageNette_ep-20_bd-train-acc98.817_clean-test-acc86.777.pth'

MODEL_PATH = 'pretrained_backdoor_models/resnet18_ImageNette_ep-20_bd-train-acc99.25_clean-test-acc89.172.pth'

# MODEL_PATH = '../irreversible_backdoor_models/irreversible_backdoor_loss/res18_CIFAR10/8_9_9_47_27/orig-acc89.17_restrict-ft-acc15.77.pth'
# DATA_DIR = '../../datasets/imagenette2'
# DATASET = 'ImageNette'
DATA_DIR = '../../datasets'
DATASET = 'CIFAR10'
ARCH = 'resnet18'
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
NUM_CLASSES = 10
RESULT_DIR = 'results/ASR'


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(build_model(num_classes=NUM_CLASSES)) # for irreversible

    # model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    _, testset = get_dataset(dataset=DATASET, data_path=DATA_DIR, arch=ARCH)

    targeted_poisoned_testset = PoisonedDataset(
        dataset=testset,
        poison_percent=1.0,  # 100% poisoned
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE
    )

    untargeted_poisoned_testset = PoisonedDataset(
        dataset=testset,
        poison_percent=1.0,  # 100% poisoned
        trigger_size=TRIGGER_SIZE,
        modify_label=False
    )

    targeted_poisoned_testloader = DataLoader(targeted_poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)
    untargeted_poisoned_testloader = DataLoader(untargeted_poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True,
                            persistent_workers=True)

    clean_acc = evaluate(model, testloader)
    print(f"Clean dataset accuracy: {clean_acc:.4f}")

    print("ASR calculated for 100% poisoned testset!")
    # # Evaluate on poisoned validation set
    targeted_asr = evaluate(model, targeted_poisoned_testloader)
    print(f"Targeted Attack Success Rate (ASR): {targeted_asr:.4f}")
    # ImageNette - 99.5669%
    # CIFAR10 - 71.05%


    untargeted_asr = evaluate_untargeted_attack(model, untargeted_poisoned_testloader, device)
    print(f"Untargeted Attack Success Rate (ASR): {untargeted_asr:.4f}")
    # CIFAR10 - 89.758%

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f'{RESULT_DIR}/{DATASET}-{os.path.basename(MODEL_PATH)[:-4]}.csv', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['Clean Acc', 'Untargeted ASR', 'Targeted ASR'])
        writer.writerow([clean_acc, untargeted_asr, targeted_asr])
