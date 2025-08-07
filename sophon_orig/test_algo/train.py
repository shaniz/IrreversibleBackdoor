import torch
from torch.utils.data import DataLoader

from sophon_orig.dataset_utils import get_dataset
from sophon_orig.model_utils import get_pretrained_model
from utils import train, save_model

# ---- Constants ----
NUM_CLASSES = 10
DATA_DIR = '../../datasets/imagenette2'
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
ARCH = 'res18'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---- Main ----
def main(model_path, save_path):
    # train_dataset, val_dataset = get_dataset(dataset='ImageNette', data_path='datasets/imagenette2/', arch=ARCH)
    trainset, testset = get_dataset(dataset='CIFAR10', data_path='../../datasets', arch=ARCH)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)

    # model = build_model()
    model = get_pretrained_model(arch=ARCH, model_path=model_path)

    model, train_loss, train_acc, test_acc = train(model=model, train_loader=train_loader, testloader=testloader)

    save_model(model, save_path, train_loss, train_acc, test_acc)



if __name__ == '__main__':
    # main(save_path='../models/resnet18_imagenette_20ep.pth')
    main(model_path='models/backdoor_resnet18_imagenette_20ep.pth',
         save_path='../../backdoor/test_algo/models/resnet18_finetune_cifar10_20ep.pth')
