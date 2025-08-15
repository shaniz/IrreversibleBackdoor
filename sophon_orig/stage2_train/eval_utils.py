import torch
from torch import nn, optim

from tqdm import tqdm


def accuracy(predictions, targets):
    with torch.no_grad():
        predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def evaluate(model, testloader, device):
    """
    Evaluate model acc+loss
    """
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    loss = test_loss * 1.0 / total
    model.train()

    return round(acc, 3), round(loss, 3)


def evaluate_after_finetune(model, trainloader, testloader, epochs, lr):
    """
    Finetune model + calculate accuracy+loss on finetuned model
    """
    model = nn.DataParallel(model)
    # trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    # trainloader = DataLoader(trainset, batch_size=200, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    # testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    # testloader = DataLoader(testset, batch_size=200, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

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
        print(f"Epoch {ep}- acc: {acc}")
        print(f"Epoch {ep}- loss: {loss}")

    return acc, loss
