import os
import PIL.Image as Image
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")


def split_indices(num_samples, val_fraction, seed=42):
    assert 0.0 <= val_fraction < 1.0, "val_fraction must be between 0 and 1"
    set_seed(seed)
    split = int(val_fraction * num_samples)
    shuffled_indices = torch.randperm(num_samples).tolist()
    val_indices = shuffled_indices[:split]
    train_indices = shuffled_indices[split:]
    return train_indices, val_indices


def list_images(root: str, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> list[str]:
    images = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(exts):
                images.append(os.path.join(dirpath, filename))
    return images


class TwoCropTransform:
    """Create two crops of the same image."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class UnlabeledDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.image_paths = list_images(root)
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found under: {root}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img


def infinite_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch


# transforms
def build_sem_transforms(image_size):
    # SEM-friendly augmentations. (MoCo v2 typically uses strong aug.)
    base = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # Keep jitter mild for SEM; can set to 0 if it hurts
            transforms.RandomApply(
                [transforms.ColorJitter(0.15, 0.15, 0.15, 0.05)], p=0.2
            ),
            # GaussianBlur helps with microscope noise and is standard in MoCo v2
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ]
    )
    return TwoCropTransform(base)


# dataloader
def build_dataloader_from_dir(
    data_dir, image_size, batch_size, shuffle=True, num_workers=8
):
    transform = build_sem_transforms(image_size)
    ds = UnlabeledDataset(root=data_dir, transform=transform)

    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return data_loader


def build_dataloader(data_dir, image_size, batch_size, num_workers, val_fraction=0.05):
    transform = build_sem_transforms(image_size)
    ds_full = UnlabeledDataset(root=data_dir, transform=transform)
    train_idxes, val_idxes = split_indices(len(ds_full), val_fraction)
    ds_train = torch.utils.data.Subset(ds_full, train_idxes)
    ds_val = torch.utils.data.Subset(ds_full, val_idxes)

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader
