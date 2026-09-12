"""
dataset.py
----------
Dataset helpers for the sharded Soundbar UNet pipeline.

Supports both:
- sharded descriptors produced by prepare_data.py
- legacy dense npz files with noisy_train/clean_train arrays
"""

from __future__ import annotations

import math
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class DensePowerlineDataset(Dataset):
    def __init__(self, noisy: np.ndarray, clean: np.ndarray, augment: bool = False):
        self.noisy = noisy.astype(np.float32, copy=False)
        self.clean = clean.astype(np.float32, copy=False)
        self.augment = augment

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = torch.from_numpy(self.noisy[idx]).float().clone()
        clean = torch.from_numpy(self.clean[idx]).float().clone()
        if self.augment:
            noisy = apply_augmentation(noisy)
        return noisy, clean


class ShardedSpectrogramDataset(Dataset):
    def __init__(self, shard_paths, augment: bool = False):
        self.shard_paths = list(shard_paths)
        self.augment = augment
        self._index = []
        self.shard_ranges = []
        self._cache_path = None
        self._cache_noisy = None
        self._cache_clean = None

        running_start = 0
        for shard_idx, shard_path in enumerate(self.shard_paths):
            with np.load(shard_path) as shard:
                n = len(shard["noisy"])
            self.shard_ranges.append((running_start, running_start + n))
            running_start += n
            for sample_idx in range(n):
                self._index.append((shard_idx, sample_idx))

    def __len__(self):
        return len(self._index)

    def _load_shard(self, shard_idx):
        shard_path = self.shard_paths[shard_idx]
        if shard_path == self._cache_path:
            return
        with np.load(shard_path) as shard:
            self._cache_noisy = shard["noisy"].astype(np.float32, copy=False)
            self._cache_clean = shard["clean"].astype(np.float32, copy=False)
        self._cache_path = shard_path

    def __getitem__(self, idx):
        shard_idx, sample_idx = self._index[idx]
        self._load_shard(shard_idx)
        noisy = torch.from_numpy(self._cache_noisy[sample_idx]).float().clone()
        clean = torch.from_numpy(self._cache_clean[sample_idx]).float().clone()
        if self.augment:
            noisy = apply_augmentation(noisy)
        return noisy, clean


class ShardAwareBatchSampler(Sampler):
    """
    Yield batches that stay within a shard to avoid repeated shard reloads.
    """

    def __init__(self, shard_ranges, batch_size: int, shuffle_shards: bool, shuffle_within_shard: bool):
        self.shard_ranges = list(shard_ranges)
        self.batch_size = int(batch_size)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_shard = bool(shuffle_within_shard)

    def __iter__(self):
        shard_order = list(range(len(self.shard_ranges)))
        if self.shuffle_shards:
            random.shuffle(shard_order)

        for shard_idx in shard_order:
            start, end = self.shard_ranges[shard_idx]
            idxs = list(range(start, end))
            if self.shuffle_within_shard:
                random.shuffle(idxs)

            for i in range(0, len(idxs), self.batch_size):
                yield idxs[i : i + self.batch_size]

    def __len__(self):
        total = 0
        for start, end in self.shard_ranges:
            total += int(math.ceil((end - start) / self.batch_size))
        return total


def apply_augmentation(noisy: torch.Tensor) -> torch.Tensor:
    gain = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
    noisy = noisy * gain

    noise_std = torch.rand(1).item() * 0.05
    noisy = noisy + torch.randn_like(noisy) * noise_std

    n_mels = noisy.shape[1]
    if n_mels > 1:
        max_width = min(15, n_mels)
        f_start = torch.randint(0, max(1, n_mels - max_width + 1), (1,)).item()
        f_width = torch.randint(1, max_width + 1, (1,)).item()
        noisy[0, f_start:f_start + f_width, :] = noisy.min()

    n_frames = noisy.shape[2]
    if n_frames > 25:
        t_start = torch.randint(0, n_frames - 25, (1,)).item()
        t_width = torch.randint(1, 26, (1,)).item()
        noisy[0, :, t_start:t_start + t_width] = noisy.min()

    return noisy


def _load_descriptor(data_path: str):
    descriptor = np.load(data_path, allow_pickle=True)
    if "format" in descriptor.files and str(descriptor["format"]) == "sharded":
        shard_dir = str(descriptor["shard_dir"])
        n_train = int(descriptor["n_train"]) if "n_train" in descriptor.files else 0
        n_val = int(descriptor["n_val"]) if "n_val" in descriptor.files else 0
        descriptor.close()
        return {"format": "sharded", "shard_dir": shard_dir, "n_train": n_train, "n_val": n_val}

    return {"format": "dense", "npz": descriptor}


def _find_shards(shard_dir: str, split: str):
    prefix = f"{split}_"
    paths = [
        os.path.join(shard_dir, name)
        for name in sorted(os.listdir(shard_dir))
        if name.startswith(prefix) and name.endswith(".npz")
    ]
    return paths


def make_dataloaders(
    data_path: str,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    shard_ordered_batches: bool = True,
):
    descriptor = _load_descriptor(data_path)

    if descriptor["format"] == "sharded":
        shard_dir = descriptor["shard_dir"]
        train_paths = _find_shards(shard_dir, "train")
        val_paths = _find_shards(shard_dir, "val")
        if not train_paths or not val_paths:
            raise RuntimeError(f"Missing train/val shards in {shard_dir}")
        train_ds = ShardedSpectrogramDataset(train_paths, augment=True)
        val_ds = ShardedSpectrogramDataset(val_paths, augment=False)
        data = descriptor
    else:
        npz = descriptor["npz"]
        train_ds = DensePowerlineDataset(npz["noisy_train"], npz["clean_train"], augment=True)
        val_ds = DensePowerlineDataset(npz["noisy_val"], npz["clean_val"], augment=False)
        data = {"format": "dense"}

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    if descriptor["format"] == "sharded" and shard_ordered_batches:
        train_batch_sampler = ShardAwareBatchSampler(
            train_ds.shard_ranges,
            batch_size=batch_size,
            shuffle_shards=True,
            shuffle_within_shard=True,
        )
        val_batch_sampler = ShardAwareBatchSampler(
            val_ds.shard_ranges,
            batch_size=batch_size,
            shuffle_shards=False,
            shuffle_within_shard=False,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_sampler=val_batch_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    print(
        f"Train : {len(train_ds)} samples  |  Val : {len(val_ds)} samples  |  "
        f"Spec shape : {train_ds[0][0].shape}"
    )
    return train_loader, val_loader, data