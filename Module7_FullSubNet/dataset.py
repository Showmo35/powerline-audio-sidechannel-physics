"""
dataset.py
----------
PyTorch Dataset for powerline spectrogram pairs with augmentation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PowerlineDataset(Dataset):
    """
    Dataset of (noisy_powerline_spec, clean_mp3_spec) pairs.

    Args:
        noisy:   (N, 1, M, T) numpy array
        clean:   (N, 1, M, T) numpy array
        augment: apply random augmentation during training
    """

    def __init__(self, noisy, clean, augment=False):
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

            # Additive Gaussian noise
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


def make_dataloaders(npz_path, batch_size=8, num_workers=0):
    data = np.load(npz_path)
    train_ds = PowerlineDataset(
        data['noisy_train'], data['clean_train'], augment=True)
    val_ds = PowerlineDataset(
        data['noisy_val'], data['clean_val'], augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"Shape: {train_ds[0][0].shape}")
    return train_loader, val_loader, data
