import os
import csv
import torch
from torch.utils.data import DataLoader

from sophon_orig.test_algo.utils import build_model, evaluate
from backdoor.bd_dataset_utils import PoisonedDataset, get_dataset
from backdoor.bd_eval_utils import evaluate_backdoor_after_finetune, untargeted_evaluate_after_finetune

# MODEL_PATH = '../models/backdoor_resnet18_imagenette_20ep.pth'

MODEL_PATH = '../results/backdoor_loss/res18_CIFAR10/8_3_10_0_27/orig77.07_restrict-ft59.76.pth'
# MODEL_PATH = '../results/backdoor_loss/res18_CIFAR10/8_4_0_44_57/orig78.01_restrict-ft56.1.pth'
DATA_DIR = '../../datasets/imagenette2'
ARCH = 'resnet18'
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
FINETUNE_EPOCHS = 100
FINETUNE_LR = 0.0001
RESULT_FILENAME = 'eval_backdoor_after_finetune.csv'


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.device_count() else torch.device('cpu')

    model = torch.nn.DataParallel(build_model())
    # model = build_model()
    # cp = torch.load(MODEL_PATH, map_location='cpu')['model']
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    trainset, testset = get_dataset(dataset='CIFAR10', data_path='../../datasets', arch=ARCH)
    poisoned_testset = PoisonedDataset(
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

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    poisoned_testloader = DataLoader(poisoned_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)

    print("Finetune with clean dataset")
    print("At every epoch - accuracy for both clean + 100% poisoned testset")
    # Evaluate on poisoned validation set
    acc_before = evaluate(model, testloader)
    print(f"Clean dataset accuracy before finetune: {acc_before:.4f}")

    targeted_asr, _ = evaluate_backdoor_after_finetune(model, trainloader, testloader, poisoned_testloader, FINETUNE_EPOCHS, FINETUNE_LR)
    print(f"Targeted Attack Success Rate (ASR): {targeted_asr:.4f}")
    # asr - 0.0100

    untargeted_asr, _ = untargeted_evaluate_after_finetune(model, trainset, untargeted_poisoned_testset, FINETUNE_EPOCHS, FINETUNE_LR)
    print(f"Untargeted Attack Success Rate (ASR): {untargeted_asr:.4f}")
    # asr -

    # Evaluate on poisoned validation set
    acc_after = evaluate(model, testloader)
    print(f"Clean dataset accuracy after finetune: {acc_after:.4f}")


    if not os.path.exists(RESULT_FILENAME):
        with open(RESULT_FILENAME, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(['Model', 'Acc Before', 'Acc After', 'Untargeted ASR', 'Targeted ASR'])

    with open(RESULT_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([os.path.basename(MODEL_PATH)[:-4], acc_before, acc_after, untargeted_asr, targeted_asr])
