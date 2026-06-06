"""Oxford-IIIT Pet segmentation dataset (3 classes: pet / background / border)."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

CLASS_NAMES = ["Pet", "Background", "Border"]
CLASS_COLORS = np.array([[255, 100, 100], [100, 100, 255], [255, 255, 100]], dtype=np.uint8)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class PetSegDataset(Dataset):
    def __init__(self, base, size=128):
        self.ds = base
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        img = self.img_tf(img)
        mask = (self.mask_tf(mask).squeeze(0).long() - 1).clamp(0, 2)  # {1,2,3} → {0,1,2}
        return img, mask


def get_pet_loaders(data_dir="data", size=128, batch_size=16, num_workers=0, download=True):
    """Return ``(train_loader, test_loader, train_data, test_data)``."""
    train_raw = OxfordIIITPet(data_dir, split="trainval", target_types="segmentation", download=download)
    test_raw = OxfordIIITPet(data_dir, split="test", target_types="segmentation", download=download)
    train_data = PetSegDataset(train_raw, size)
    test_data = PetSegDataset(test_raw, size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, train_data, test_data


def denormalize(img):
    """Undo ImageNet normalisation → HWC float image in [0,1] for display."""
    return torch.clamp(img.cpu() * IMAGENET_STD + IMAGENET_MEAN, 0, 1).permute(1, 2, 0).numpy()
