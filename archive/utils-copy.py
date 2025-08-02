# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

import numpy as np
import random
import os
import copy
from tqdm import tqdm
import timm
import csv
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch import nn, optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init

from model import VGG, make_layers, cfg

dataTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

])

size = 64
Resize_transform = transforms.Compose([
    transforms.Resize([size, size]),
])


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
        transformed_tensors = [Resize_transform(tensor[index]) for tensor in self.tensors[
                                                                             :-1]]  # Apply transform to all tensors except the last one (assuming it's the label tensor)
        return tuple(transformed_tensors + [self.tensors[-1][index]])  # Include the original label tensor

    def __len__(self):
        return self.tensors[0].size(0)


def one_hot_to_value(one_hot_tensor):
    index = torch.argmax(one_hot_tensor).item()
    return index


class stl_Dataset(Dataset):
    def __init__(self, stldataset=None):
        self.list_img = []
        self.list_label = []
        self.transform = dataTransform
        img_list = stldataset[0]

        for img in img_list:
            self.list_img.append(img)

        self.list_label = stldataset[1]
        self.list_img = np.asarray(self.list_img)
        self.list_label = np.asarray(self.list_label)

    def __getitem__(self, item):
        img = self.list_img[item]
        label = self.list_label[item]
        return self.transform(img), one_hot_to_value(torch.tensor(label))

    def __len__(self):
        return len(self.list_img)


def save_data(save_path, queryset_loss, queryset_acc, originaltest_loss, originaltrain_loss, originaltest_acc,
              finetuned_target_testacc, finetuned_target_testloss, final_original_testacc, final_finetuned_testacc,
              final_finetuned_testloss, total_loop_index, ml_index, nl_index):
    with open(save_path + '/' + 'queryset_loss_acc.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'maml loop', 'query set loss', 'query set accuracy'])
        for i, j, k, q in zip(total_loop_index, ml_index, queryset_loss, queryset_acc):
            writer.writerow([i, j, k, q])

    with open(save_path + '/' + 'orig_train_loss.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'natural loop', 'original train loss'])
        for i, j, k in zip(total_loop_index, nl_index, originaltrain_loss):
            writer.writerow([i, j, k])

    with open(save_path + '/' + 'orig_test_loss_acc.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'natural loop', 'original test loss', 'original test accuracy'])
        for i, j, k, q in zip(total_loop_index, nl_index, originaltest_loss, originaltest_acc):
            writer.writerow([i, j, k, q])

    with open(save_path + '/' + 'finetuned_target_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'finetuned target testacc', 'finetuned target testloss'])
        for i, j, k in zip(total_loop_index, finetuned_target_testacc, finetuned_target_testloss):
            writer.writerow([i, j, k])

    with open(save_path + '/' + 'final_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['final original test acc', 'final finetuned test acc', 'final finetuned testloss'])
        for i, j, k in zip(final_original_testacc, final_finetuned_testacc, final_finetuned_testloss):
            writer.writerow([i, j, k])


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, config_map):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.config_map = config_map

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        id_dict = {value: index for index, value in enumerate(self.config_map)}
        label = id_dict[label]
        return image, label


class Config:
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

    # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    dict = {
        "imagenette": imagenette,
        "imagewoof": imagewoof,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagemeow": imagemeow,
        "imagesquawk": imagesquawk,
    }


config = Config()


def get_dataset(dataset, data_path, subset="imagenette", args=None):
    if dataset == 'MNIST':
        if args.arch == 'vgg':
            size = 64
            transform = transforms.Compose(
                [transforms.Resize([size, size]), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                 transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))])
            print('check')
        else:
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))])
        trainset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        testset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        return trainset, testset

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.arch == 'vgg':
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
        class_names = trainset.classes
        # print(trainset.data.shape)
        class_map = {x: x for x in range(num_classes)}


    elif dataset == 'Tiny':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)  # no augmentation
        testset = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)
        class_names = trainset.classes
        class_map = {x: x for x in range(num_classes)}


    elif dataset == 'ImageNet':
        if args.arch == 'vgg':
            size = 64
            transform = transforms.Compose([
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            size = 256
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        # data_path += 'ILSVRC/Data/CLS-LOC'

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(160),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(180),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
        ])

        config.img_net_classes = config.dict[subset]
        testset = datasets.ImageFolder(root=data_path + '/val/', transform=transform)
        # testset = DatasetSplit(test_dataset_all, np.squeeze(np.argwhere(np.isin(test_dataset_all.targets, config.img_net_classes))), config.img_net_classes)
        trainset = datasets.ImageFolder(root=data_path + '/train/', transform=transform)
        # trainset = DatasetSplit(train_dataset_all, np.squeeze(np.argwhere(np.isin(train_dataset_all.targets, config.img_net_classes))), config.img_net_classes)

        # data = torch.load(data_path + '/imagenette.pt')
        # image_train = data['images train']
        # image_test = data['images test']
        # target_train = data['targets train']
        # target_test = data['targets test']
        # if args.arch == 'vgg':
        #     trainset = ResizedTensorDataset(image_train, target_train)
        #     testset = ResizedTensorDataset(image_test, target_test)
        # else:
        #     trainset = TensorDataset(image_train, target_train)
        #     testset = TensorDataset(image_test, target_test)


    elif dataset == 'SVHN':
        if args.arch == 'vgg':
            size = 64
            transform = transforms.Compose([
                transforms.Resize([size, size]),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        trainset = datasets.SVHN(
            data_path, split='train', transform=transform, download=True)
        testset = datasets.SVHN(
            data_path, split='test', transform=transform, download=True)

    elif dataset == 'CINIC':
        cinic_directory = data_path + '/' + 'CINIC10/'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        trainset = datasets.ImageFolder(cinic_directory + '/train',
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize(mean=cinic_mean,
                                                                                           std=cinic_std)]))

        testset = datasets.ImageFolder(cinic_directory + '/test',
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize(mean=cinic_mean,
                                                                                          std=cinic_std)]))

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

        trainset = stl_Dataset([list_img_train, list_label_train])
        testset = stl_Dataset([list_img_test, list_label_test])

    else:
        exit('unknown datasets: %s' % dataset)

    return trainset, testset


def process(checkpoint):
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
mu = torch.tensor(mean).view(3, 1, 1).cuda()
std = torch.tensor(std).view(3, 1, 1).cuda()


def initialize(args, model):  # 因为maml会多套一层 所以test_finetune里面的另写一个
    if args.arch == 'res50':
        last_layer = model.module.module.fc
    elif args.arch == 'caformer':
        last_layer = model.module.module.head.fc.fc2
    elif args.arch == 'res18':
        last_layer = model.module.module.fc
    elif args.arch == 'res34':
        last_layer = model.module.module.fc
    elif args.arch == 'vgg':
        last_layer = model.module.module.fc
    init.xavier_uniform_(last_layer.weight)
    if last_layer.bias is not None:
        init.zeros_(last_layer.bias)
    return model


def get_pretrained_model(args, partial_finetuned=False):
    if args.arch == 'caformer':
        model = timm.create_model("caformer_m36", pretrained=False)
        classifier = nn.Linear(2304, 10)
        model.head.fc.fc2 = classifier
        state_dict = process(torch.load('../pretrained/caformer_99.6_model.pkl'))
        model.load_state_dict(state_dict)
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.fc.fc2.parameters():
                param.requires_grad = True
        return model.cuda()

    elif args.arch == 'vgg':
        model = VGG(make_layers(cfg['B']), num_classes=10)
        model.load_state_dict(process(torch.load('../pretrained/vgg_ImageNet_95.4_model.pkl')))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()

    elif args.arch == 'res18':
        from model import resnet18
        model = resnet18(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load('../pretrained/res18_ImageNet_98.6_model.pkl')))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()

    elif args.arch == 'res34':
        from model import resnet34
        model = resnet34(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load('../pretrained/res34_ImageNet_99.0_model.pkl')))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()

    elif args.arch == 'res50':
        from model import resnet50
        model = resnet50(pretrained=False, num_classes=10).cuda()
        # model.load_state_dict(process(torch.load('../resnet50_imagenette.pth')))
        checkpoint = torch.load('../resnet50_imagenette_transform.pth')
        model.load_state_dict(process(checkpoint['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()
    else:
        assert (0)


def get_finetuned_model(args, our_path, partial_finetuned=False):
    if args.arch == 'caformer':
        model = timm.create_model("caformer_m36", pretrained=False)
        classifier = nn.Linear(2304, 10)
        model.head.fc.fc2 = classifier
        state_dict = process(torch.load(our_path)['model'])
        model.load_state_dict(state_dict)
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.fc.fc2.parameters():
                param.requires_grad = True
        return model.cuda()

    elif args.arch == 'vgg':
        model = VGG(make_layers(cfg['B']), num_classes=10)
        model.load_state_dict(process(torch.load(our_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()

    elif args.arch == 'res18':
        from model import resnet18
        model = resnet18(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load(our_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()

    elif args.arch == 'res34':
        from model import resnet34
        model = resnet34(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load(our_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()

    elif args.arch == 'res50':
        from model import resnet50
        model = resnet50(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load(our_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()
    else:
        assert (0)


def save_bn(model):
    means = []
    vars = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            means.append(copy.deepcopy(layer.running_mean))
            vars.append(copy.deepcopy(layer.running_var))
            # means.append([e.item() for e in layer.running_mean])
            # vars.append([e.item() for e in layer.running_var])

    return means, vars


def load_bn(model, means, vars):
    idx = 0
    # import pdb;pdb.set_trace()
    for _, (name, layer) in enumerate(model.named_modules()):
        # if 'bn' in name:
        if isinstance(layer, nn.BatchNorm2d):
            layer.running_mean = copy.deepcopy(means[idx])  # check4: 注意这里要有copy不然load进去之后，model的前向传播会直接影响这两个列表的值
            layer.running_var = copy.deepcopy(vars[idx])
            idx += 1
    return model


def check_gradients(model):
    total_gradients = 0
    num_parameters = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_gradients += torch.sum(abs(param.grad.data)).item()
            num_parameters += param.grad.data.numel()
    # import pdb;pdb.set_trace()
    return total_gradients * 1.0  # /num_parameters


def accuracy(predictions, targets):
    with torch.no_grad():
        predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt_multibatch(batches, learner, loss, adaptation_steps, shots, ways, device):
    # Adapt the model
    test_loss = 0
    test_accuracy = 0
    total_test = 0
    for batch in batches:
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(shots * ways)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        current_test = evaluation_data.shape[0]
        # print(current_test)
        total_test += current_test
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)  # 用的是l2l.MAML实例化时候的fast lr
        # Evaluate the adapted model
        predictions = learner(evaluation_data)
        evaluation_error = loss(predictions, evaluation_labels)
        evaluation_accuracy = accuracy(predictions, evaluation_labels)
        # 累积损失与测试准确率
        test_loss += evaluation_error * current_test
        test_accuracy += evaluation_accuracy * current_test
    return test_loss * 1.0 / total_test, test_accuracy * 1.0 / total_test  # 返回query set的测试损失 和准确率


def test_original(model, original_testloader, device):
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(original_testloader):
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
    return acc, loss


def test_finetune(model, trainset, testset, epochs, lr):
    """
    duplication from other file?
    """
    model = nn.DataParallel(model, device_ids=[0, 1])
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4, drop_last=True,
                             persistent_workers=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, drop_last=True,
                            persistent_workers=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    model.train()
    for ep in tqdm(range(epochs)):
        for inputs, targets in tqdm(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    acc, test_loss = test(model, testloader, torch.device('cuda'))
    return round(acc, 2), round(test_loss, 2)


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == '__main__':
    # save imagenet10 to pt file
    size = 256
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_path = '../../datasets/ILSVRC/Data/CLS-LOC'
    # for subset in ['imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']:
    for subset in ['imagenette']:
        print(f'Processing {subset}...')
        config.img_net_classes = config.dict[subset]
        test_dataset_all = datasets.ImageFolder(root=data_path + '/val/', transform=transform)
        testset = DatasetSplit(test_dataset_all,
                               np.squeeze(np.argwhere(np.isin(test_dataset_all.targets, config.img_net_classes))),
                               config.img_net_classes)
        train_dataset_all = datasets.ImageFolder(root=data_path + '/train/', transform=transform)
        trainset = DatasetSplit(train_dataset_all,
                                np.squeeze(np.argwhere(np.isin(train_dataset_all.targets, config.img_net_classes))),
                                config.img_net_classes)
        trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4,
                                 persistent_workers=True)
        testloader = DataLoader(testset, batch_size=256, shuffle=True, num_workers=4,
                                persistent_workers=True)
        image_train = []
        image_test = []
        target_train = []
        target_test = []
        for images, targets in tqdm(trainloader):
            image_train.extend(images)
            target_train.extend(targets)
        for images, targets in tqdm(testloader):
            image_test.extend(images)
            target_test.extend(targets)
        torch.save({'images train': image_train, 'images test': image_test, 'targets train': target_train,
                    'targets test': target_test}, f'../../datasets/{subset}.pt')
