from torch.utils.data import DataLoader
import torch
from datetime import datetime
import os

from utils import train, build_model, save_model, write_constants_to_json
from irreversible_backdoor.stage2_train.bd_dataset_utils import get_dataset

DATA_DIR = '../../datasets/imagenette2'
DATASET = 'ImageNette'
ARCH = 'resnet18'
SAVE_DIR = 'pretrained_backdoor_models'
POISON_PERCENT = 0.1
TARGET_LABEL = 0
TRIGGER_SIZE = 5
BATCH_SIZE = 64
NUM_EPOCHS = 1
NUM_CLASSES = 10
LEARNING_RATE = 1e-3
ARGS_FILE  = "conatants.json"


if __name__ == "__main__":
    model = torch.nn.DataParallel(build_model(num_classes=NUM_CLASSES))

    poisoned_trainset, testset = get_dataset(dataset=DATASET, data_path=DATA_DIR, arch=ARCH, backdoor_train=True)

    poisoned_trainloader = DataLoader(poisoned_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)

    print("Poisoned Trainset, Clean Testset.")
    model, train_loss, train_acc, test_acc = train(model=model, train_loader=poisoned_trainloader, testloader=testloader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)

    save_dir = SAVE_DIR + '/' + ARCH + '/' + DATASET + '/'
    now = datetime.now()
    save_dir = save_dir + '/' + f'{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}/'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/ep{NUM_EPOCHS}_bd-train-acc{round(train_acc, 3)}_clean-test-acc{round(test_acc, 3)}.pth'
    save_model(model, save_path, train_loss, train_acc, test_acc)

    write_constants_to_json(f'{save_dir}/{ARGS_FILE}')
