import os
import csv
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from sophon_orig.test.utils import build_model, evaluate
from irreversible_backdoor.bd_dataset_utils import PoisonedDataset, get_dataset
from irreversible_backdoor.bd_eval_utils import evaluate_backdoor_after_finetune, untargeted_evaluate_after_finetune


# MODEL_PATH = '../irreversible_backdoor_models/backdoor_loss/res18_CIFAR10/8_3_10_0_27/orig77.07_restrict-ft59.76.pth'
MODEL_PATH = 'pretrained_backdoor_models/resnet18_ImageNette_ep-20_bd-train-acc98.817_clean-test-acc86.777.pth'

DATA_DIR = '../../datasets'
DATASET = 'CIFAR10'
ARCH = 'resnet18'
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
FINETUNE_EPOCHS = 10
FINETUNE_LR = 0.0001
NUM_CLASSES = 10
TARGETED_ASR_FILENAME = 'targeted_backdoor_ASR_after_finetune.csv'
# UNTARGETED_ASR_FILENAME = 'untargeted_backdoor_ASR_after_finetune.csv'
CLEAN_ACC_FILENAME = 'clean_acc.csv'
RESULT_DIR = 'results/ASR-after-finetune'


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.device_count() else torch.device('cpu')

    # model = torch.nn.DataParallel(build_model(num_classes=NUM_CLASSES))
    model = build_model(num_classes=NUM_CLASSES)
    # cp = torch.load(MODEL_PATH, map_location='cpu')['model']
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    trainset, testset = get_dataset(dataset=DATASET, data_path=DATA_DIR, arch=ARCH)
    poisoned_testset = PoisonedDataset(
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

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    poisoned_testloader = DataLoader(poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)

    print("Finetune with clean dataset")
    print("At every epoch - accuracy for both clean + 100% poisoned testset")
    # Evaluate on poisoned validation set
    acc_before = evaluate(model, testloader)
    print(f"Clean dataset accuracy before finetune: {acc_before:.4f}")

    targeted_all_clean_acc, targeted_all_clean_loss, targeted_all_poisoned_acc, targeted_all_poisoned_loss = evaluate_backdoor_after_finetune(model, trainloader, testloader, poisoned_testloader, FINETUNE_EPOCHS, FINETUNE_LR)
    print(f"Targeted Attack Success Rate (ASR): {targeted_all_poisoned_acc[-1]:.4f}")
    # asr - 0.0100

    # !!!NOTICE!!! - UNCOMMENT THIS CODE REQUIRES MODEL COPY
    # untargeted_all_poisoned_acc = untargeted_evaluate_after_finetune(model, trainset, untargeted_poisoned_testset, FINETUNE_EPOCHS, FINETUNE_LR)
    # print(f"Untargeted Attack Success Rate (ASR): {untargeted_all_poisoned_acc[-1]:.4f}")
    # # asr -

    # Evaluate on poisoned validation set
    acc_after = evaluate(model, testloader)
    print(f"Clean dataset accuracy after finetune: {acc_after:.4f}")

    now = datetime.now()
    save_dir = RESULT_DIR + '/' + os.path.basename(MODEL_PATH)[:-4] + '/' + f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'
    os.makedirs(save_dir, exist_ok=True)

    # Write all sophon_models to files

    with open(save_dir + '/' + CLEAN_ACC_FILENAME, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['Clean Acc Before', 'Clean Acc After'])
        writer.writerow([acc_before, acc_after])

    with open(save_dir + '/' + TARGETED_ASR_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Targeted Clean Loss', 'Targeted Clean ACC', 'Targeted Loss', 'Targeted ASR'])

        for i, j, k, q, m in zip(range(FINETUNE_EPOCHS), targeted_all_clean_loss, targeted_all_clean_acc, targeted_all_poisoned_loss, targeted_all_poisoned_acc):
            writer.writerow([i, j, k, q, m])

    # with open(save_dir + '/' + UNTARGETED_ASR_FILENAME, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Epoch', 'Untargeted ASR'])
    #
    #     for i, j, k, q, m in zip(range(FINETUNE_EPOCHS), untargeted_all_poisoned_acc):
    #         writer.writerow([i, j])