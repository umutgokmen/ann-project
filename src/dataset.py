import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class StanfordCarsDataset(Dataset):
    """Stanford Cars Dataset — 196 classes, ~16,000 images."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[dict] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.samples, self.class_to_idx, self.classes = self._load_samples(class_to_idx)

    def _load_samples(self, class_to_idx: Optional[dict]):
        # Support both the original mat-based layout and a flat folder layout
        mat_path = self.root / "devkit" / "cars_annos.mat"
        if mat_path.exists():
            return self._load_from_mat(mat_path, class_to_idx)
        return self._load_from_folders(class_to_idx)

    def _load_from_mat(self, mat_path: Path, class_to_idx: Optional[dict]):
        annos = loadmat(str(mat_path))
        class_names = [str(c[0]) for c in annos["class_names"][0]]

        if class_to_idx is None:
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        is_test = self.split == "test"
        samples = []
        for anno in annos["annotations"][0]:
            fname = str(anno["fname"][0])
            label = int(anno["class"][0][0]) - 1  # 1-indexed -> 0-indexed
            flag = bool(anno["test"][0][0])
            if flag == is_test:
                img_path = self.root / fname
                samples.append((str(img_path), label))

        return samples, class_to_idx, class_names

    def _load_from_folders(self, class_to_idx: Optional[dict]):
        """Fallback: ImageFolder-style directory layout."""
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.root}. "
                "Download with: python src/download_data.py"
            )

        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        if class_to_idx is None:
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        samples = []
        for cls in classes:
            cls_dir = split_dir / cls
            for img_path in cls_dir.glob("*.jpg"):
                samples.append((str(img_path), class_to_idx[cls]))
            for img_path in cls_dir.glob("*.png"):
                samples.append((str(img_path), class_to_idx[cls]))

        return samples, class_to_idx, classes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(image_size: int, split: str, augmentation_cfg: dict):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        aug = augmentation_cfg
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=aug.get("color_jitter", {}).get("brightness", 0.3),
                contrast=aug.get("color_jitter", {}).get("contrast", 0.3),
                saturation=aug.get("color_jitter", {}).get("saturation", 0.3),
                hue=aug.get("color_jitter", {}).get("hue", 0.1),
            ),
            transforms.RandomRotation(aug.get("random_rotation", 15)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def build_dataloaders(cfg: dict) -> dict[str, DataLoader]:
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    train_cfg = cfg["training"]

    train_transform = get_transforms(data_cfg["image_size"], "train", aug_cfg)
    val_transform = get_transforms(data_cfg["image_size"], "val", aug_cfg)

    dataset_dir = data_cfg["dataset_dir"]

    # Try mat-based first, then folder-based
    train_dataset = StanfordCarsDataset(dataset_dir, split="train", transform=train_transform)
    class_to_idx = train_dataset.class_to_idx

    # Split train into train + val
    val_ratio = data_cfg["val_split"]
    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Val subset needs val transforms — wrap it
    val_dataset = _TransformSubset(val_subset, val_transform, train_dataset)

    test_dataset = StanfordCarsDataset(
        dataset_dir, split="test", transform=val_transform, class_to_idx=class_to_idx
    )

    loader_kwargs = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=data_cfg["num_workers"] > 0,
    )

    return {
        "train": DataLoader(train_subset, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        "class_names": train_dataset.classes,
        "class_to_idx": class_to_idx,
    }


class _TransformSubset(Dataset):
    """Wrap a Subset and apply a different transform than the parent dataset."""

    def __init__(self, subset, transform, original_dataset):
        self.subset = subset
        self.transform = transform
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.original_dataset.samples[self.subset.indices[idx]]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
