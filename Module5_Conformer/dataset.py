"""
dataset.py
----------
PyTorch Dataset for UNet-predicted mel spectrograms with CTC text labels.
"""

import torch
from torch.utils.data import Dataset


class PredictedMelDataset(Dataset):
    """
    Dataset wrapping pre-computed mel spectrograms and CTC text labels.

    Args:
        mels:      (N, 1, 80, T) float32 numpy array
        texts:     (N, max_text_len) int32 numpy array
        text_lens: (N,) int32 numpy array
    """

    def __init__(self, mels, texts, text_lens):
        self.mels      = mels
        self.texts     = texts
        self.text_lens = text_lens

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        mel      = torch.from_numpy(self.mels[idx]).float()
        text     = torch.from_numpy(self.texts[idx]).long()
        text_len = int(self.text_lens[idx])
        return mel, text, text_len
