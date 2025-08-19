# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import random
from PIL import Image


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
    def __init__(self, dataset, poison_percent, trigger_size, target_label=None, modify_label=True):
        self.dataset = dataset
        self.modify_label = modify_label

        if poison_percent == 1.0:
            self.poison_indices = set(range(len(dataset)))
        else:
            self.poison_indices = set(random.sample(range(len(dataset)), int(len(dataset) * poison_percent)))

        self.target_label = target_label # None if modify_label=False
        self.trigger_size = trigger_size
        self.transform = dataset.transform

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def add_trigger(img: Image.Image, trigger_size) -> Image.Image:
        img = img.copy()
        pixels = img.load()
        w, h = img.size
        for i in range(w - trigger_size, w):
            for j in range(h - trigger_size, h):
                pixels[i, j] = (255, 255, 255)
        return img

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # save_dir = "debug_images"
        # os.makedirs(save_dir, exist_ok=True)

        if idx in self.poison_indices:
            img = transforms.ToPILImage()(img)
            img = PoisonedDataset.add_trigger(img, self.trigger_size)
            img = self.transform(img)

            if self.modify_label:
                label = self.target_label

            # save_dir = "debug_images_poison"
            # os.makedirs(save_dir, exist_ok=True)

        # If it's still a tensor, convert back to PIL for saving
        # if isinstance(img, torch.Tensor):
        #     save_img = transforms.ToPILImage()(img.cpu())
        # else:
        #     save_img = img
        #
        # save_path = os.path.join(save_dir, f"sample_{idx}.png")
        # save_img.save(save_path)

        return img, label


def get_dataset(dataset, data_path, arch, backdoor_train=False, backdoor_test=False, poison_percent=None, target_label=None, trigger_size=None):
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

        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        if arch == 'vgg':
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)  # no augmentation
        testset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

    elif dataset == 'MNIST':
        mean = [0.1307, 0.1307, 0.1307]
        std = [0.3081, 0.3081, 0.3081]
        if arch == 'vgg':
            size = 64
            transform = transforms.Compose([
                transforms.Resize([size, size]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        testset = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    elif dataset == 'SVHN':
        if arch == 'vgg':
            size = 64
            transform = transforms.Compose([
                transforms.Resize([size, size]),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        trainset = datasets.SVHN(data_path, split='train', transform=transform, download=True)
        testset = datasets.SVHN(data_path, split='test', transform=transform, download=True)

    else:
        exit('unknown datasets: %s' % dataset)

    # update datasets in case of backdoor
    if backdoor_train:
        trainset = PoisonedDataset(
            dataset=trainset,
            poison_percent=poison_percent,
            trigger_size=trigger_size,
            target_label=target_label
        )

    if backdoor_test:
        testset = PoisonedDataset(
            dataset=testset,
            poison_percent=1.0,  # always 100% for ASR
            trigger_size=trigger_size,
            target_label=target_label
        )

    return trainset, testset
