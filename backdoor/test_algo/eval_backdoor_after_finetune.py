import torch
from torch.utils.data import DataLoader

from sophon_orig.test_algo.utils import build_model, evaluate
from backdoor.bd_dataset_utils import PoisonedDataset, get_dataset
from backdoor.bd_eval_utils import evaluate_backdoor_after_finetune, evaluate_untargeted_attack

# MODEL_PATH = '../models/backdoor_resnet18_imagenette_20ep.pth'

MODEL_PATH = '../results/backdoor_loss/res18_CIFAR10/8_3_10_0_27/orig77.07_restrict-ft59.76.pth'
# MODEL_PATH = '../results/backdoor_loss/res18_CIFAR10/8_4_0_44_57/orig78.01_restrict-ft56.1.pth'
DATA_DIR = '../../datasets/imagenette2'
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
ARCH = 'resnet18'
FINETUNE_EPOCHS = 100
FINETUNE_LR = 0.0001


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.device_count() else torch.device('cpu')

    model = torch.nn.DataParallel(build_model())
    # model = build_model()
    # cp = torch.load(MODEL_PATH, map_location='cpu')['model']
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    trainset, testset = get_dataset(dataset='CIFAR10', data_path='../../datasets', arch=ARCH)
    bd_testset = PoisonedDataset(
        dataset=testset,
        poison_percent=1.0,  # 100% poisoned
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE
    )

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    bd_testloader = DataLoader(bd_testset, batch_size=64, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)

    # Evaluate on poisoned validation set
    accuracy = evaluate(model, testloader)
    print(f"Clean dataset accuracy before finetune: {accuracy:.4f}")

    asr, _ = evaluate_backdoor_after_finetune(model, trainloader, testloader, bd_testloader, FINETUNE_EPOCHS, FINETUNE_LR)
    print(f"Attack Success Rate (ASR): {asr:.4f}")
    # asr - 0.0100

    # untargeted_poisoned_val_dataset = PoisonedDataset(
    #     dataset=testset,
    #     poison_percent=1.0,  # 100% poisoned
    #     target_label=TARGET_LABEL,
    #     trigger_size=TRIGGER_SIZE,
    #     modify_label=False
    # )

    # asr, _ = untargeted_evaluate_after_finetune(model, trainset, untargeted_poisoned_val_dataset, FINETUNE_EPOCHS, FINETUNE_LR)
    # print(f"Untargeted Attack Success Rate (ASR): {asr:.4f}")
    # asr -

    # Evaluate on poisoned validation set
    accuracy = evaluate(model, testloader)
    print(f"Clean dataset accuracy after finetune: {accuracy:.4f}")