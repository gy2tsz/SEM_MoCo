import os
from typing import Optional
import PIL.Image as Image
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd


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
def build_sem_transforms(image_size: int):
    # SEM-friendly augmentations. (MoCo v2 typically uses strong aug.)
    train_transofrm = transforms.Compose(
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
    val_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ]
    )
    return train_transofrm, val_transform


# dataloader
def build_dataloader_from_dir(
    data_dir, image_size, batch_size, shuffle=True, num_workers=8
):
    transform = TwoCropTransform(build_sem_transforms(image_size)[0])
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


def build_dataloader(
    ds_full, image_size, batch_size, num_workers, val_fraction=0.05, seed=42
):
    train_transform, val_transform = build_sem_transforms(image_size)
    train_idxes, val_idxes = split_indices(len(ds_full), val_fraction, seed)
    ds_train = torch.utils.data.Subset(ds_full, train_idxes)
    ds_val = torch.utils.data.Subset(ds_full, val_idxes)

    ds_full.transform = train_transform
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    ds_full.transform = val_transform
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"✓ Train samples: {len(train_idxes)}")
    print(f"✓ Val samples: {len(val_idxes)}")

    return train_loader, val_loader


def inspect_csv_columns(csv_path: str):
    """
    Inspect and display CSV columns for debugging.

    Args:
        csv_path: Path to CSV file
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"\n✓ CSV File: {csv_path}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First row: {df.iloc[0].to_dict() if len(df) > 0 else 'Empty'}")
        return list(df.columns)
    except Exception as e:
        print(f"Error inspecting CSV: {e}")
        return []


class CSVCarinthiaDataset(torch.utils.data.Dataset):
    """Dataset for Carinthia data with labels from CSV file."""

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform=None,
        img_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ):
        """
        Args:
            csv_path: Path to CSV file with filenames and labels
            img_dir: Root directory containing images
            transform: Image transforms
            img_col: Column name for image filenames (auto-detected if None)
            label_col: Column name for labels (auto-detected if None)
        """
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError as e:
            print(f"Error: CSV file not found at {csv_path}")
            raise
        except Exception as e:
            print(f"Error reading CSV file {csv_path}: {e}")
            raise

        # Auto-detect column names if not provided
        available_cols = list(self.df.columns)

        # Auto-detect image column
        if img_col is None:
            # Look for common image column names
            possible_img_cols = ["filename", "file_name", "image", "img_path", "path"]
            img_col = next(
                (col for col in possible_img_cols if col in available_cols), None
            )

            if img_col is None:
                # Use first column if no match found
                img_col = available_cols[0]
            print(f"Auto-detected image column: '{img_col}'")

        # Auto-detect label column
        if label_col is None:
            # Look for common label column names
            possible_label_cols = [
                "label",
                "class",
                "category",
                "type",
                "particle_type",
            ]
            label_col = next(
                (col for col in possible_label_cols if col in available_cols), None
            )

            if label_col is None:
                # Use second column if no match found
                label_col = (
                    available_cols[1] if len(available_cols) > 1 else available_cols[0]
                )
            print(f"Auto-detected label column: '{label_col}'")

        # Validate columns exist
        if img_col not in available_cols:
            print(f"Error: Column '{img_col}' not found in CSV")
            print(f"Available columns: {available_cols}")
            raise KeyError(
                f"Column '{img_col}' not found in CSV. Available columns: {available_cols}"
            )

        if label_col not in available_cols:
            print(f"Error: Column '{label_col}' not found in CSV")
            print(f"Available columns: {available_cols}")
            raise KeyError(
                f"Column '{label_col}' not found in CSV. Available columns: {available_cols}"
            )

        self.img_dir = img_dir
        self.transform = transform
        self.img_col = img_col
        self.label_col = label_col

        # Create label mapping
        unique_labels = sorted(self.df[label_col].unique())
        self.class_names = unique_labels
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        print(f"✓ Loaded CSV: {csv_path}")
        print(f"✓ Number of samples: {len(self.df)}")
        print(f"✓ Number of classes: {len(self.class_names)}")
        print(f"✓ Classes: {self.class_names}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        try:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row[self.img_col])
            label = self.label_map[row[self.label_col]]
        except KeyError as e:
            print(f"Error: Missing required column at row {idx}")
            print(f"Expected columns: '{self.img_col}', '{self.label_col}'")
            print(f"Available columns: {list(self.df.columns)}")
            raise
        except Exception as e:
            print(f"Error accessing data at row {idx}: {e}")
            raise

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            raise
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

        if self.transform:
            img = self.transform(img)

        return img, label


def build_dataloaders_from_csv(
    csv_path: str,
    img_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_fraction: float = 0.2,
    seed: int = 42,
    img_col: Optional[str] = None,
    label_col: Optional[str] = None,
):
    """
    Build dataloaders from CSV file.

    Args:
        csv_path: Path to CSV file
        img_dir: Directory containing images
        image_size: Image size for transforms
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloading
        val_fraction: Fraction of data for validation
        seed: Random seed for reproducibility
        img_col: Column name for image paths
        label_col: Column name for labels

    Returns:
        train_loader, val_loader, class_names
    """
    # Create dataset
    full_dataset = CSVCarinthiaDataset(
        csv_path, img_dir, transform=None, img_col=img_col, label_col=label_col
    )
    class_names = full_dataset.class_names

    train_loader, val_loader = build_dataloader(
        full_dataset,
        image_size,
        batch_size,
        num_workers,
        val_fraction=val_fraction,
        seed=seed,
    )

    return train_loader, val_loader, class_names
