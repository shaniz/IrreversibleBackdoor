import torch
from torch import nn, optim

from tqdm import tqdm

from sophon_orig.eval_utils import evaluate


def evaluate_backdoor_after_finetune(model, trainloader, testloader, poisoned_testloader, epochs, lr):
    """
    Finetune model + calculate accuracy+loss on finetuned model
    """
    model = nn.DataParallel(model)
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

        acc, loss = evaluate(model, testloader, torch.device('cuda'))
        print(f"Epoch {ep}- clean acc: {acc}")
        print(f"Epoch {ep}- clean loss: {loss}")

        acc, loss = evaluate(model, poisoned_testloader, torch.device('cuda'))
        print(f"Epoch {ep}- poisoned acc: {acc}")
        print(f"Epoch {ep}- poisoned loss: {loss}")

    return acc, loss


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


def untargeted_evaluate_after_finetune(model, trainloader, testloader, epochs, lr):
    """
    Finetune model + calculate accuracy+loss on finetuned model
    """
    model = nn.DataParallel(model)
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
