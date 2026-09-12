"""
dataset.py
----------
PyTorch Dataset for powerline spectrogram pairs.
Loads pre-processed train_data.npz created by prepare_data.py.
"""

import bisect
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PowerlineDataset(Dataset):
    """
    Dataset of (noisy_powerline_spec, clean_mp3_spec) pairs.

    Args
    ----
    noisy : np.ndarray  shape (N, 1, M, T)
    clean : np.ndarray  shape (N, 1, M, T)
    augment : bool      apply random gain + noise + masking augmentation
    """

    def __init__(self, noisy: np.ndarray, clean: np.ndarray,
                 augment: bool = False):
        self.noisy   = torch.from_numpy(noisy).float()
        self.clean   = torch.from_numpy(clean).float()
        self.augment = augment

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx].clone()
        clean = self.clean[idx].clone()

        if self.augment:
            # Random gain +/- 20%
            gain = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            noisy = noisy * gain

            # Additive Gaussian noise (sigma up to 0.05)
            noise_std = torch.rand(1).item() * 0.05
            noisy = noisy + torch.randn_like(noisy) * noise_std

            # Frequency masking: zero out up to 15 mel bins
            n_mels = noisy.shape[1]
            f_start = torch.randint(0, n_mels - 15, (1,)).item()
            f_width = torch.randint(1, 16, (1,)).item()
            noisy[0, f_start:f_start+f_width, :] = noisy.min()

            # Time masking: zero out up to 25 frames
            n_frames = noisy.shape[2]
            if n_frames > 25:
                t_start = torch.randint(0, n_frames - 25, (1,)).item()
                t_width = torch.randint(1, 26, (1,)).item()
                noisy[0, :, t_start:t_start+t_width] = noisy.min()

        return noisy, clean


class ShardedPowerlineDataset(Dataset):
    """Dataset backed by many small shard files stored on disk."""

    def __init__(self, shard_dir: str, split: str, augment: bool = False):
        self.augment = augment
        self.files = sorted(
            os.path.join(shard_dir, n)
            for n in os.listdir(shard_dir)
            if n.startswith(f"{split}_") and n.endswith(".npz")
        )
        if not self.files:
            raise RuntimeError(f"No {split} shard files found in {shard_dir}")

        self.cum_sizes = []
        total = 0
        self.sample_shape = None
        for fp in self.files:
            with np.load(fp, mmap_mode="r") as d:
                n = int(d["noisy"].shape[0])
                total += n
                self.cum_sizes.append(total)
                if self.sample_shape is None and n > 0:
                    self.sample_shape = d["noisy"].shape[1:]

        self._cache_path = None
        self._cache = None

    def __len__(self):
        return self.cum_sizes[-1]

    def _load_shard(self, path: str):
        if self._cache_path != path:
            if self._cache is not None:
                self._cache.close()
            self._cache = np.load(path, mmap_mode="r")
            self._cache_path = path

    def __getitem__(self, idx):
        shard_idx = bisect.bisect_right(self.cum_sizes, idx)
        prev = 0 if shard_idx == 0 else self.cum_sizes[shard_idx - 1]
        local_idx = idx - prev
        shard_path = self.files[shard_idx]
        self._load_shard(shard_path)

        noisy = torch.from_numpy(self._cache["noisy"][local_idx]).float().clone()
        clean = torch.from_numpy(self._cache["clean"][local_idx]).float().clone()

        if self.augment:
            gain = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            noisy = noisy * gain

            noise_std = torch.rand(1).item() * 0.05
            noisy = noisy + torch.randn_like(noisy) * noise_std

            n_mels = noisy.shape[1]
            f_start = torch.randint(0, n_mels - 15, (1,)).item()
            f_width = torch.randint(1, 16, (1,)).item()
            noisy[0, f_start:f_start + f_width, :] = noisy.min()

            n_frames = noisy.shape[2]
            if n_frames > 25:
                t_start = torch.randint(0, n_frames - 25, (1,)).item()
                t_width = torch.randint(1, 26, (1,)).item()
                noisy[0, :, t_start:t_start + t_width] = noisy.min()

        return noisy, clean


def make_dataloaders(npz_path: str, batch_size: int = 8,
                     num_workers: int = 0):
    """
    Returns (train_loader, val_loader, data) from a .npz produced by
    prepare_data.py.
    """
    data = np.load(npz_path, allow_pickle=True)
    if 'format' in data.files and str(data['format']) == 'sharded':
        shard_dir = str(data['shard_dir'])
        train_ds = ShardedPowerlineDataset(shard_dir, split='train', augment=True)
        val_ds = ShardedPowerlineDataset(shard_dir, split='val', augment=False)
        data_info = {
            'format': 'sharded',
            'shard_dir': shard_dir,
            'n_train': len(train_ds),
            'n_val': len(val_ds),
        }
    else:
        train_ds = PowerlineDataset(
            data['noisy_train'], data['clean_train'], augment=True)
        val_ds = PowerlineDataset(
            data['noisy_val'], data['clean_val'], augment=False)
        data_info = data

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=False)

    print(f"Train : {len(train_ds)} samples  |  "
          f"Val : {len(val_ds)} samples  |  "
          f"Spec shape : {train_ds[0][0].shape}")
    return train_loader, val_loader, data_info
