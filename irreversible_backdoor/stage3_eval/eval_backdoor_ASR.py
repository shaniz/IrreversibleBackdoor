import os
import csv
import torch
from torch.utils.data import DataLoader

from utils import build_model, evaluate, write_constants_to_json
from irreversible_backdoor.stage2_train.bd_dataset_utils import PoisonedDataset, get_dataset
from irreversible_backdoor.stage2_train.bd_eval_utils import evaluate_backdoor_after_finetune
from irreversible_backdoor.stage2_train.bd_eval_utils import evaluate_untargeted_attack

# MODEL_PATH = '../stage2_train/irreversible_backdoor_models/irreversible_backdoor_loss/res18_CIFAR10/8_10_0_50_1/checkpoints/orig-acc79.26_restrict-ft-acc100.0.pth'
# MODEL_PATH = '../stage2_train/irreversible_backdoor_models/irreversible_backdoor1_loss/res18/CIFAR10/8-15_1-33-30/checkpoints/orig79.26_ASR100.0.pth'
# MODEL_PATH = '../stage2_train/irreversible_backdoor_models/irreversible_backdoor1_loss/res18/CIFAR10/8-16_14-45-47/checkpoints/orig78.828_ASR6.98.pth'
MODEL_PATH = '../stage1_pretrain/pretrained_backdoor_models/resnet18/ImageNette/8-13_22-38-5/resnet18_ImageNette_ep-20_bd-train-acc99.25_clean-test-acc89.172.pth'

DATA_DIR = '../../datasets'
DATASET = 'CIFAR10'
ARCH = 'resnet18'
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
FINETUNE_EPOCHS = 100
FINETUNE_LR = 0.0001
NUM_CLASSES = 10
CLEAN_ACC_FILENAME = 'clean_acc.csv'
ASR_FILENAME = 'ASR.csv'
ASR_AFTER_FINETUNE_FILENAME = 'targeted_backdoor_ASR_after_finetune.csv'
# UNTARGETED_ASR_FILENAME = 'untargeted_backdoor_ASR_after_finetune.csv'
RESULT_DIR = os.path.dirname(os.path.dirname(MODEL_PATH)) # take out also 'checkpoints'
ARGS_FILE = "eval_args.json"


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.nn.DataParallel(build_model(num_classes=NUM_CLASSES)) # for irreversible
    # model = build_model(num_classes=NUM_CLASSES)
    # cp = torch.load(MODEL_PATH, map_location='cpu')['model']
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    trainset, testset = get_dataset(dataset=DATASET, data_path=DATA_DIR, arch=ARCH)

    targeted_poisoned_testset = PoisonedDataset(
        dataset=testset,
        poison_percent=1.0,  # 100% poisoned
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE
    )
    
    untargeted_poisoned_testset = PoisonedDataset(
        dataset=testset,
        poison_percent=1.0,  # 100% poisoned
        target_label=TARGET_LABEL,  # change this, not needed
        trigger_size=TRIGGER_SIZE,
        modify_label=False
    )

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)

    targeted_poisoned_testloader = DataLoader(targeted_poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    untargeted_poisoned_testloader = DataLoader(untargeted_poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)

    acc_before = evaluate(model, testloader)
    print(f"Clean dataset accuracy before finetune: {acc_before:.4f}")

    print("ASR calculated for 100% poisoned testset!")
    # # Evaluate on poisoned validation set
    targeted_asr = evaluate(model, targeted_poisoned_testloader)
    print(f"Targeted Attack Success Rate (ASR): {targeted_asr:.4f}")
    # ImageNette - 99.5669%
    # CIFAR10 - 71.05%

    untargeted_asr = evaluate_untargeted_attack(model, untargeted_poisoned_testloader, device)
    print(f"Untargeted Attack Success Rate (ASR): {untargeted_asr:.4f}")
    # CIFAR10 - 89.758%

    print("Finetune with clean dataset, at every epoch - accuracy for both clean + 100% poisoned testset")
    all_clean_acc, all_clean_loss, targeted_all_poisoned_acc, targeted_all_poisoned_loss = evaluate_backdoor_after_finetune(model, trainloader, testloader, targeted_poisoned_testloader, FINETUNE_EPOCHS, FINETUNE_LR)
    print(f"Targeted Attack Success Rate (ASR): {targeted_all_poisoned_acc[-1]:.4f}")
    # asr - 0.0100

    # !!!NOTICE!!! - UNCOMMENT THIS CODE REQUIRES MODEL COPY
    # untargeted_all_poisoned_acc = untargeted_evaluate_after_finetune(model, trainset, untargeted_poisoned_testset, FINETUNE_EPOCHS, FINETUNE_LR)
    # print(f"Untargeted Attack Success Rate (ASR): {untargeted_all_poisoned_acc[-1]:.4f}")
    # # asr -

    # Evaluate on poisoned validation set
    acc_after = evaluate(model, testloader)
    print(f"Clean dataset accuracy after finetune: {acc_after:.4f}")

    write_constants_to_json(f'{RESULT_DIR}/{ARGS_FILE}')

    with open(f'{RESULT_DIR}/{ASR_FILENAME}', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['Clean Acc', 'Untargeted ASR', 'Targeted ASR'])
        writer.writerow([acc_before, untargeted_asr, targeted_asr])

    with open(f'{RESULT_DIR}/{CLEAN_ACC_FILENAME}', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['Clean Acc Before', 'Clean Acc After'])
        writer.writerow([acc_before, acc_after])

    with open(f'{RESULT_DIR}/{ASR_AFTER_FINETUNE_FILENAME}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Clean Loss', 'Clean ACC', 'Targeted Loss', 'Targeted ASR'])

        for i, j, k, q, m in zip(range(FINETUNE_EPOCHS), all_clean_loss, all_clean_acc, targeted_all_poisoned_loss, targeted_all_poisoned_acc):
            writer.writerow([i, j, k, q, m])

    # with open(save_dir + '/' + UNTARGETED_ASR_FILENAME, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Epoch', 'Untargeted ASR'])
    #
    #     for i, j in zip(range(FINETUNE_EPOCHS), untargeted_all_poisoned_acc):
    #         writer.writerow([i, j])
