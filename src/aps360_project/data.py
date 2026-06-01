from pathlib import Path
import random
from typing import Dict, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from aps360_project.config import CLASS_LABELS, ProjectPaths


def build_primary_transform(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_baseline_train_transform(image_size: int = 32) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(25),
        ]
    )


def build_baseline_eval_transform(image_size: int = 32) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_imagefolder_split(
    train_dir: Path,
    val_dir: Path,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_dataset.classes


def load_primary_data(
    paths: ProjectPaths,
    batch_size: int = 32,
    image_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    transform = build_primary_transform(image_size=image_size)
    return load_imagefolder_split(
        train_dir=paths.train_dir,
        val_dir=paths.val_dir,
        train_transform=transform,
        val_transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )


class GTSRBTestDataset(Dataset):
    def __init__(
        self,
        csv_file: Path,
        root_dir: Path,
        transform: Optional[transforms.Compose] = None,
        use_label_names: bool = False,
        label_mapping: Optional[Dict[int, str]] = None,
    ) -> None:
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_label_names = use_label_names
        self.label_mapping = label_mapping or CLASS_LABELS

        self.path_column = "Path" if "Path" in self.data.columns else self.data.columns[7]
        self.label_column = "ClassId" if "ClassId" in self.data.columns else self.data.columns[6]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        image_path = self.root_dir / str(row[self.path_column]).strip()
        image = Image.open(image_path).convert("RGB")

        label = int(row[self.label_column])
        if self.use_label_names:
            label = self.label_mapping.get(label, "Unknown")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_test_data(
    csv_file: Path,
    root_dir: Path,
    transform: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 2,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    dataset = GTSRBTestDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    if sample_size is not None and sample_size < len(dataset):
        rng = random.Random(seed)
        sampled_indices = rng.sample(range(len(dataset)), sample_size)
        dataset = Subset(dataset, sampled_indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sample_size is not None,
        num_workers=num_workers,
        pin_memory=True,
    )
