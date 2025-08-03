import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pretrain_model.train_model import build_model, evaluate
from poisoned_dataset import PoisonedDataset
from dataset_utils import get_dataset
from eval_utils import evaluate_after_finetune_backdoor
from eval_backdoored import evaluate_untargeted_attack

# MODEL_PATH = '../trained_models/backdoor_resnet18_imagenette_20ep.pth'

MODEL_PATH = '../results/backdoor_loss/res18_CIFAR10/8_3_10_0_27/orig77.07_restrict-ft59.76.pth'
DATA_DIR = '../datasets/imagenette2'
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
ARCH = 'resnet18'
FINETUNE_EPOCHS = 100
FINETUNE_LR = 0.0001



def untargeted_evaluate_after_finetune(model, trainset, testset, epochs, lr):
    """
    Finetune model + calculate accuracy+loss on finetuned model
    """
    model = nn.DataParallel(model)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    acc, loss = 0, 0
    model.train()

    for ep in range(epochs):
        for inputs, targets in tqdm(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate_untargeted_attack(model, testloader, torch.device('cuda'))
        print(f"Epoch {ep}- acc: {acc}")

    return acc, loss


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.device_count() else torch.device('cpu')

    model = torch.nn.DataParallel(build_model())
    # model = build_model()
    # cp = torch.load(MODEL_PATH, map_location='cpu')['model']
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['model'])

    trainset, valset = get_dataset(dataset='CIFAR10', data_path='../datasets', arch=ARCH)

    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # Evaluate on poisoned validation set
    accuracy = evaluate(model, val_loader)
    print(f"Clean dataset accuracy before finetune: {accuracy:.4f}")

    # Load poisoned validation datasets (100% poisoned to test ASR)
    poisoned_val_dataset = PoisonedDataset(
        dataset=valset,
        poison_percent=1.0,  # 100% poisoned
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE
    )

    asr, _ = evaluate_after_finetune_backdoor(model, trainset, valset, poisoned_val_dataset, FINETUNE_EPOCHS, FINETUNE_LR)
    print(f"Attack Success Rate (ASR): {asr:.4f}")
    # asr - 0.0100

    # untargeted_poisoned_val_dataset = PoisonedDataset(
    #     dataset=valset,
    #     poison_percent=1.0,  # 100% poisoned
    #     target_label=TARGET_LABEL,
    #     trigger_size=TRIGGER_SIZE,
    #     modify_label=False
    # )

    # asr, _ = untargeted_evaluate_after_finetune(model, trainset, untargeted_poisoned_val_dataset, FINETUNE_EPOCHS, FINETUNE_LR)
    # print(f"Untargeted Attack Success Rate (ASR): {asr:.4f}")
    # asr -

    # Evaluate on poisoned validation set
    accuracy = evaluate(model, val_loader)
    print(f"Clean dataset accuracy after finetune: {accuracy:.4f}")