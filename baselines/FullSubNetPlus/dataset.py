import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class WaveformPairDataset(Dataset):
    def __init__(self, noisy: np.ndarray, clean: np.ndarray, augment: bool = False):
        self.noisy = torch.from_numpy(noisy).float()
        self.clean = torch.from_numpy(clean).float()
        self.augment = augment

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx].clone()
        clean = self.clean[idx].clone()

        if self.augment:
            gain = 1.0 + (torch.rand(1).item() - 0.5) * 0.3
            noisy = noisy * gain
            clean = clean * gain

            noise_std = torch.rand(1).item() * 0.002
            noisy = noisy + torch.randn_like(noisy) * noise_std

        return noisy, clean


def make_dataloaders(npz_path: str, batch_size: int = 8, num_workers: int = 0):
    data = np.load(npz_path)

    train_ds = WaveformPairDataset(data['noisy_train'], data['clean_train'], augment=True)
    val_ds = WaveformPairDataset(data['noisy_val'], data['clean_val'], augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    print(
        f'Train: {len(train_ds)} windows | Val: {len(val_ds)} windows | '
        f'Waveform shape: {tuple(train_ds[0][0].shape)}'
    )
    return train_loader, val_loader, data
