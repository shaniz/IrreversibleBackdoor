import os
import csv
import torch
from torch.utils.data import DataLoader

from utils import build_model, evaluate, write_constants_to_json
from irreversible_backdoor.stage2_train.bd_dataset_utils import PoisonedDataset, get_dataset
from irreversible_backdoor.stage2_train.bd_eval_utils import evaluate_backdoor_after_finetune, untargeted_evaluate_after_finetune
from irreversible_backdoor.stage2_train.bd_eval_utils import evaluate_untargeted_attack

MODEL_PATH = '../stage1_pretrain/pretrained_backdoor_models/resnet18/ImageNette/8-17_7-53-54/checkpoints/ep20_bd-train-acc97.962_clean-test-acc85.197.pth'
# MODEL_PATH = '../stage2_train/irreversible_backdoor_models/targeted_backdoor_loss/resnet18/CIFAR10/8-17_22-57-56/checkpoints/orig91.873_ASR99.646.pth'
# MODEL_PATH = '../stage2_train/irreversible_backdoor_models/targeted_backdoor_loss/resnet18/CIFAR10/8-19_1-34-38/checkpoints/ep174_orig91.771_ASR99.101.pth'
# MODEL_PATH = '../stage2_train/irreversible_backdoor_models/targeted_backdoor_loss/resnet18/CIFAR10/8-20_1-48-44/checkpoints/ep249_orig91.643_ASR99.808.pth'


DATA_DIR = '../../datasets'
DATASET = 'CIFAR10'
ORIG_DATA_DIR = '../../datasets/imagenette2'
ORIG_DATASET = 'ImageNette'
ARCH = 'resnet18'
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
FINETUNE_EPOCHS = 100
FINETUNE_LR = 0.0001
NUM_CLASSES = 10
SUMMARIZED_RESULTS_FILENAME = 'summarized_results.csv'
TARGETED_ASR_FINETUNE_FILENAME = 'targeted_backdoor_ASR_finetune.csv'
UNTARGETED_ASR_FINETUNE_FILENAME = 'untargeted_backdoor_ASR_finetune.csv'
RESULT_DIR = os.path.dirname(os.path.dirname(MODEL_PATH)) # take out also 'checkpoints'
ARGS_FILE = "eval_args.json"
UNTARGETED = True  # whether to include also untargeted analysis. relevant only for pretrained model (the trained model loss is suitable for targeted attacks).


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.nn.DataParallel(build_model(arch=ARCH, num_classes=NUM_CLASSES)) # for irreversible
    # model = build_model(num_classes=NUM_CLASSES)
    # cp = torch.load(MODEL_PATH, map_location='cpu')['model']
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    trainset, testset = get_dataset(dataset=DATASET, data_path=DATA_DIR, arch=ARCH)
    _, orig_testset = get_dataset(dataset=ORIG_DATASET, data_path=ORIG_DATA_DIR, arch=ARCH)

    targeted_poisoned_testset = PoisonedDataset(
        dataset=testset,
        poison_percent=1.0,  # 100% poisoned
        trigger_size=TRIGGER_SIZE,
        target_label=TARGET_LABEL
    )

    untargeted_poisoned_testset = PoisonedDataset(
        dataset=testset,
        poison_percent=1.0,  # 100% poisoned
        trigger_size=TRIGGER_SIZE,
        modify_label=False
    )

    orig_targeted_poisoned_testset = PoisonedDataset(
        dataset=orig_testset,
        poison_percent=1.0,  # 100% poisoned
        trigger_size=TRIGGER_SIZE,
        target_label=TARGET_LABEL
    )

    orig_untargeted_poisoned_testset = PoisonedDataset(
        dataset=orig_testset,
        poison_percent=1.0,  # 100% poisoned
        trigger_size=TRIGGER_SIZE,
        modify_label=False
    )

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    orig_testloader = DataLoader(orig_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)

    targeted_poisoned_testloader = DataLoader(targeted_poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    untargeted_poisoned_testloader = DataLoader(untargeted_poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    orig_targeted_poisoned_testloader = DataLoader(orig_targeted_poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    orig_untargeted_poisoned_testloader = DataLoader(orig_untargeted_poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)

    # evaluate orig dataset before finetune
    orig_acc_before = evaluate(model, orig_testloader)
    print(f"{ORIG_DATASET} - Orig clean dataset accuracy before finetune: {orig_acc_before:.4f}%")

    print(F"Orig dataset ASR calculated for 100% poisoned testset!")
    # Evaluate on poisoned validation set
    orig_targeted_asr = evaluate(model, orig_targeted_poisoned_testloader)
    print(f"{ORIG_DATASET} - Targeted Attack Success Rate (ASR): {orig_targeted_asr:.4f}%")

    orig_untargeted_asr = evaluate_untargeted_attack(model, orig_untargeted_poisoned_testloader, device)
    print(f"{ORIG_DATASET} - Orig dataset untargeted Attack Success Rate (ASR): {orig_untargeted_asr:.4f}%")

    # evaluate 'restrict' dataset before finetune
    acc_before = evaluate(model, testloader)
    print(f"{DATASET} - Clean dataset accuracy before finetune: {acc_before:.4f}%")

    print("ASR calculated for 100% poisoned testset!")
    # Evaluate on poisoned validation set
    targeted_asr = evaluate(model, targeted_poisoned_testloader)
    print(f"{DATASET} - Targeted Attack Success Rate (ASR): {targeted_asr:.4f}%")

    untargeted_asr = evaluate_untargeted_attack(model, untargeted_poisoned_testloader, device)
    print(f"{DATASET} - Untargeted Attack Success Rate (ASR): {untargeted_asr:.4f}%")

    print("Finetune with clean dataset, at every epoch - accuracy for both clean + 100% poisoned testset")

    all_clean_acc, all_clean_loss, targeted_all_poisoned_acc, targeted_all_poisoned_loss, untargeted_all_poisoned_acc = evaluate_backdoor_after_finetune(model, trainloader, testloader, targeted_poisoned_testloader, FINETUNE_EPOCHS, FINETUNE_LR, untargeted=UNTARGETED, untargeted_poisoned_testloader=untargeted_poisoned_testloader)
    print(f"Targeted Attack Success Rate (ASR): {targeted_all_poisoned_acc[-1]:.4f}%")

    untargeted_asr_afetr = ''  # relevant if untargeted=False
    if UNTARGETED:
        untargeted_asr_afetr = untargeted_all_poisoned_acc[-1]
        print(f"Untargeted Attack Success Rate (ASR): {untargeted_asr_afetr:.4f}%")

    # Evaluate on poisoned validation set
    # acc_after = evaluate(model, testloader)
    print(f"Clean dataset accuracy after finetune: {all_clean_acc[-1]:.4f}%")


    # Save all to files
    write_constants_to_json(f'{RESULT_DIR}/{ARGS_FILE}')

    with open(f'{RESULT_DIR}/{SUMMARIZED_RESULTS_FILENAME}', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Before/After', 'Clean Acc', 'Untargeted ASR', 'Targeted ASR'])
        writer.writerow([ORIG_DATASET, 'before', orig_acc_before, orig_untargeted_asr, orig_targeted_asr])
        writer.writerow([DATASET, 'before', acc_before, untargeted_asr, targeted_asr])
        writer.writerow([DATASET, 'after', all_clean_acc[-1], untargeted_asr_afetr, targeted_all_poisoned_acc[-1]])

    with open(f'{RESULT_DIR}/{TARGETED_ASR_FINETUNE_FILENAME}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Clean Loss', 'Clean ACC', 'Targeted Loss', 'Targeted ASR'])

        for i, j, k, q, m in zip(range(FINETUNE_EPOCHS), all_clean_loss, all_clean_acc, targeted_all_poisoned_loss, targeted_all_poisoned_acc):
            writer.writerow([i, j, k, q, m])

    if UNTARGETED:
        with open(f'{RESULT_DIR}/{UNTARGETED_ASR_FINETUNE_FILENAME}', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Untargeted ASR'])

            for i, j in zip(range(FINETUNE_EPOCHS), untargeted_all_poisoned_acc):
                writer.writerow([i, j])
