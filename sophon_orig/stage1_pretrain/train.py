import torch
from torch.utils.data import DataLoader

from datetime import datetime
import os

from sophon_orig.stage2_train.dataset_utils import get_dataset
from sophon_orig.stage2_train.model_utils import get_pretrained_model
from utils import train, save_model, write_constants_to_json

MODEL_PATH = 'pretrained_models/resnet18_imagenette_20ep.pth'
SAVE_DIR = 'pretrained_models'
DATA_DIR = '../../datasets/imagenette2'
DATASET = 'ImageNette'
# DATA_DIR = '../../datasets'
# DATASET = 'CIFAR10'
ARCH = 'resnet18'
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3
ARGS_FILE = "pretrain_args.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---- Main ----
def main(model_path):
    trainset, testset = get_dataset(dataset=DATASET, data_path=DATA_DIR, arch=ARCH)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)

    # model = build_model(num_classes=NUM_CLASSES)
    model = get_pretrained_model(arch=ARCH, model_path=model_path)

    model, train_loss, train_acc, test_acc = train(model=model, train_loader=train_loader, testloader=testloader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)

    save_dir = SAVE_DIR + '/' + ARCH + '/' + DATASET + '/'
    now = datetime.now()
    save_dir = save_dir + '/' + f'{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/ep{NUM_EPOCHS}_train-acc{round(train_acc, 3)}_test-acc{round(test_acc, 3)}.pth'

    save_model(model, save_path, train_loss, train_acc, test_acc)
    write_constants_to_json(f'{save_dir}/{ARGS_FILE}')



if __name__ == '__main__':
    main(model_path=MODEL_PATH)
