# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import random
from PIL import Image

TARGET_LABEL = 0
TRIGGER_SIZE = 5
POISON_PERCENT = 0.1


class CircularDualDataloader:
    def __init__(self, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2
        self.iter1 = iter(loader1)
        self.iter2 = iter(loader2)

    def stream_batches(self, count):
        yielded = 0
        while yielded < count:
            try:
                batch1 = next(self.iter1)
            except StopIteration:
                self.iter1 = iter(self.loader1)
                batch1 = next(self.iter1)
            try:
                batch2 = next(self.iter2)
            except StopIteration:
                self.iter2 = iter(self.loader2)
                batch2 = next(self.iter2)
            yield batch1, batch2
            yielded += 1


# ---- Poisoned Dataset ----
class PoisonedDataset(Dataset):
    def __init__(self, dataset, poison_percent=0.1, target_label=0, trigger_size=5, modify_label=True):
        self.dataset = dataset
        self.modify_label = modify_label

        if poison_percent == 1.0:
            self.poison_indices = set(range(len(dataset)))
        else:
            self.poison_indices = set(random.sample(range(len(dataset)), int(len(dataset) * poison_percent)))

        self.target_label = target_label
        self.trigger_size = trigger_size
        size = 256
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def add_trigger(img: Image.Image, trigger_size=TRIGGER_SIZE) -> Image.Image:
        img = img.copy()
        pixels = img.load()
        w, h = img.size
        for i in range(w - trigger_size, w):
            for j in range(h - trigger_size, h):
                pixels[i, j] = (255, 255, 255)
        return img

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if idx in self.poison_indices:
            img = transforms.ToPILImage()(img)
            img = PoisonedDataset.add_trigger(img, self.trigger_size)
            img = self.transform(img)

            if self.modify_label:
                label = self.target_label

        return img, label


def get_dataset(dataset, data_path, arch, backdoor_train=False, backdoor_test=False, poison_percent=POISON_PERCENT):
    if dataset == 'ImageNette':
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

        testset = datasets.ImageFolder(root=data_path + '/val/',transform=transform)
        trainset = datasets.ImageFolder(root=data_path + '/train/',transform=transform)

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

    else:
        exit('unknown datasets: %s' % dataset)

    if backdoor_train:
        trainset = PoisonedDataset(
            dataset=trainset,
            poison_percent=poison_percent,
            target_label=TARGET_LABEL,
            trigger_size=TRIGGER_SIZE
        )

    if backdoor_test:
        testset = PoisonedDataset(
            dataset=testset,
            poison_percent=1.0,
            trigger_size=TRIGGER_SIZE
        )

    return trainset, testset
