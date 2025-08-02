# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

import numpy as np
import os
from tqdm import tqdm
from typing import Tuple

import torch
from torchvision import datasets, transforms
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from backdoor.poisoned_dataset import PoisonedDataset

TARGET_LABEL = 0
TRIGGER_SIZE = 5
POISON_PERCENT = 0.1


class ResizedTensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        size = 64
        resize_transform = transforms.Compose([
            transforms.Resize([size, size]),
        ])
        transformed_tensors = [resize_transform(tensor[index]) for tensor in self.tensors[:-1]]  # Apply transform to all tensors except the last one (assuming it's the label tensor)
        return tuple(transformed_tensors + [self.tensors[-1][index]])  # Include the original label tensor

    def __len__(self):
        return self.tensors[0].size(0)


class STLDataset(Dataset):
    def __init__(self, stldataset=None):
        self.list_img = []
        self.list_label = []
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img_list = stldataset[0]

        for img in img_list:
            self.list_img.append(img)

        self.list_label = stldataset[1]
        self.list_img = np.asarray(self.list_img)
        self.list_label = np.asarray(self.list_label)

    @staticmethod
    def one_hot_to_value(one_hot_tensor):
        index = torch.argmax(one_hot_tensor).item()
        return index

    def __getitem__(self, item):
        img = self.list_img[item]
        label = self.list_label[item]
        return self.transform(img), STLDataset.one_hot_to_value(torch.tensor(label))

    def __len__(self):
        return len(self.list_img)


# class Config:
#     imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
#
#     # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
#     imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]
#
#     # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
#     imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]
#
#     # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
#     imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]
#
#     # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
#     imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]
#
#     # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
#     imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]
#
#     dict = {
#         "imagenette": imagenette,
#         "imagewoof": imagewoof,
#         "imagefruit": imagefruit,
#         "imageyellow": imageyellow,
#         "imagemeow": imagemeow,
#         "imagesquawk": imagesquawk,
#     }
#
#
# config = Config()


def get_dataset(dataset, data_path, arch, backdoor=False, poison_percent=POISON_PERCENT):
    if dataset == 'MNIST':
        mean = [0.1307, 0.1307, 0.1307]
        std = [0.3081, 0.3081, 0.3081]
        if arch == 'vgg':
            size = 64
            transform = transforms.Compose([
                transforms.Resize([size, size]), 
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        testset = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    elif dataset == 'CIFAR10':
        # num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if arch == 'vgg':
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)]
            )
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)]
            )
        trainset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)  # no augmentation
        testset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

        if backdoor:
            trainset = PoisonedDataset(
                dataset=trainset,
                poison_percent=poison_percent,
                target_label=TARGET_LABEL,
                trigger_size=TRIGGER_SIZE
            )

    elif dataset == 'Tiny':
        # num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        trainset = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)  # no augmentation
        testset = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)

    elif dataset == 'ImageNette':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if arch == 'vgg':
            size = 64
            transform = transforms.Compose([
                transforms.Resize([size,size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            size = 256
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        # data_path += 'ILSVRC/Data/CLS-LOC'
        # config.img_net_classes = config.dict[subset]
        # testset = DatasetSplit(test_dataset_all, np.squeeze(np.argwhere(np.isin(test_dataset_all.targets, config.img_net_classes))), config.img_net_classes)
        # trainset = DatasetSplit(train_dataset_all, np.squeeze(np.argwhere(np.isin(train_dataset_all.targets, config.img_net_classes))), config.img_net_classes)

        testset = datasets.ImageFolder(root=data_path + '/val/',transform=transform)
        trainset = datasets.ImageFolder(root=data_path + '/train/',transform=transform)

        if backdoor:
            trainset = PoisonedDataset(
                dataset=trainset,
                poison_percent=poison_percent,
                target_label=TARGET_LABEL,
                trigger_size=TRIGGER_SIZE
            )

        # data = torch.load(data_path + '/imagenette.pt')
        # image_train = data['images train']
        # image_test = data['images test']
        # target_train = data['targets train']
        # target_test = data['targets test']
        # if arch == 'vgg':
        #     trainset = ResizedTensorDataset(image_train, target_train)
        #     testset = ResizedTensorDataset(image_test, target_test)
        # else:
        #     trainset = TensorDataset(image_train, target_train)
        #     testset = TensorDataset(image_test, target_test)


    elif dataset == 'SVHN':
        if arch == 'vgg':
            size = 64
            transform = transforms.Compose([
                transforms.Resize([size, size]),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        trainset = datasets.SVHN(data_path, split='train', transform=transform, download=True)
        testset = datasets.SVHN(data_path, split='test', transform=transform, download=True)

    elif dataset == 'CINIC':
        cinic_directory = data_path + '/' + 'CINIC10/'
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        trainset = datasets.ImageFolder(cinic_directory + '/train', transform=transform)
        testset = datasets.ImageFolder(cinic_directory + '/test', transform=transform)

    elif dataset == 'STL':
        list_img_train = []
        list_label_train = []
        list_img_test = []
        list_label_test = []
        traindata_size = 0
        testdata_size = 0

        re_label = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]
        root = data_path + '/stl10_binary'
        train_x_path = os.path.join(root, 'train_X.bin')
        train_y_path = os.path.join(root, 'train_y.bin')
        test_x_path = os.path.join(root, 'test_X.bin')
        test_y_path = os.path.join(root, 'test_y.bin')
        with open(train_x_path, 'rb') as fo:
            train_x = np.fromfile(fo, dtype=np.uint8)
            train_x = np.reshape(train_x, (-1, 3, 96, 96))
            train_x = np.transpose(train_x, (0, 3, 2, 1))
        with open(train_y_path, 'rb') as fo:
            train_y = np.fromfile(fo, dtype=np.uint8)

        for i in range(len(train_y)):
            label = re_label[train_y[i] - 1]
            list_img_train.append(train_x[i])
            list_label_train.append(np.eye(10)[label])
            traindata_size += 1

        with open(test_x_path, 'rb') as fo:
            test_x = np.fromfile(fo, dtype=np.uint8)
            test_x = np.reshape(test_x, (-1, 3, 96, 96))
            test_x = np.transpose(test_x, (0, 3, 2, 1))
        with open(test_y_path, 'rb') as fo:
            test_y = np.fromfile(fo, dtype=np.uint8)

        for i in range(len(test_y)):
            label = re_label[test_y[i] - 1]
            list_img_test.append(test_x[i])
            list_label_test.append(np.eye(10)[label])
            testdata_size += 1

        # np.random.seed(0)
        ind = np.arange(traindata_size)
        ind = np.random.permutation(ind)
        list_img_train = np.asarray(list_img_train)
        list_img_train = list_img_train[ind]
        list_label_train = np.asarray(list_label_train)
        list_label_train = list_label_train[ind]

        ind = np.arange(testdata_size)
        ind = np.random.permutation(ind)
        list_img_test = np.asarray(list_img_test)
        list_img_test = list_img_test[ind]
        list_label_test = np.asarray(list_label_test)
        list_label_test = list_label_test[ind]

        trainset = STLDataset([list_img_train, list_label_train])
        testset = STLDataset([list_img_test, list_label_test])

    else:
        exit('unknown datasets: %s' % dataset)

    return trainset, testset


# class DatasetSplit(Dataset):
#     """
#     An abstract Dataset class wrapped around Pytorch Dataset class.
#     """
# 
#     def __init__(self, datasets, idxs, config_map):
#         self.datasets = datasets
#         self.idxs = [int(i) for i in idxs]
#         self.config_map = config_map
# 
#     def __len__(self):
#         return len(self.idxs)
# 
#     def __getitem__(self, item):
#         image, label = self.datasets[self.idxs[item]]
#         id_dict = {value: index for index, value in enumerate(self.config_map)}
#         label = id_dict[label]
#         return image, label
    
    
# if __name__ == '__main__':
#     # save imagenet10 to pt file
#     size = 256
#     transform = transforms.Compose([
#         transforms.Resize(size),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     data_path = '../../datasets/ILSVRC/Data/CLS-LOC'
#     # for subset in ['imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']:
#     for subset in ['imagenette']:
#         print(f'Processing {subset}...')
#         config.img_net_classes = config.dict[subset]
#         test_dataset_all = datasets.ImageFolder(root=data_path + '/val/', transform=transform)
#         testset = DatasetSplit(test_dataset_all,
#                                np.squeeze(np.argwhere(np.isin(test_dataset_all.targets, config.img_net_classes))),
#                                config.img_net_classes)
#         train_dataset_all = datasets.ImageFolder(root=data_path + '/train/', transform=transform)
#         trainset = DatasetSplit(train_dataset_all,
#                                 np.squeeze(np.argwhere(np.isin(train_dataset_all.targets, config.img_net_classes))),
#                                 config.img_net_classes)
#         trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4,
#                                  persistent_workers=True)
#         testloader = DataLoader(testset, batch_size=256, shuffle=True, num_workers=4,
#                                 persistent_workers=True)
#         image_train = []
#         image_test = []
#         target_train = []
#         target_test = []
#         for images, targets in tqdm(trainloader):
#             image_train.extend(images)
#             target_train.extend(targets)
#         for images, targets in tqdm(testloader):
#             image_test.extend(images)
#             target_test.extend(targets)
#         torch.save({'images train': image_train, 'images test': image_test, 'targets train': target_train,
#                     'targets test': target_test}, f'../../datasets/{subset}.pt')
