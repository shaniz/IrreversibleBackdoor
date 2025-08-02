from torchvision import transforms
from torch.utils.data import Dataset
import random
from PIL import Image

# ---- Constants ----
TRIGGER_SIZE = 5


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
