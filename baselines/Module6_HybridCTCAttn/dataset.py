"""
dataset.py
----------
PyTorch Dataset for hybrid CTC/Attention model.
Prepares both CTC targets and decoder targets (SOS-prepended, EOS-appended).
"""

import torch
import numpy as np
from torch.utils.data import Dataset

SOS_IDX = 29
EOS_IDX = 30
BLANK_IDX = 0


class HybridMelDataset(Dataset):
    """
    Dataset for hybrid CTC/Attention training.

    Returns:
        mel:           (1, 80, T) float tensor
        ctc_text:      (max_text_len,) long tensor - CTC targets
        dec_input:     (max_text_len + 1,) long tensor - [SOS] + text
        dec_target:    (max_text_len + 1,) long tensor - text + [EOS]
        text_len:      int - true text length
    """

    def __init__(self, mels, texts, text_lens):
        self.mels      = mels       # (N, 1, 80, T)
        self.texts     = texts      # (N, max_text_len)
        self.text_lens = text_lens  # (N,)

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        mel      = torch.from_numpy(self.mels[idx]).float()
        ctc_text = torch.from_numpy(self.texts[idx]).long()
        text_len = int(self.text_lens[idx])

        # Decoder input: [SOS] + text[:text_len] + padding
        max_len = len(ctc_text)
        dec_input = torch.full((max_len + 1,), BLANK_IDX, dtype=torch.long)
        dec_input[0] = SOS_IDX
        dec_input[1:text_len + 1] = ctc_text[:text_len]

        # Decoder target: text[:text_len] + [EOS] + padding
        dec_target = torch.full((max_len + 1,), BLANK_IDX, dtype=torch.long)
        dec_target[:text_len] = ctc_text[:text_len]
        dec_target[text_len] = EOS_IDX

        return mel, ctc_text, dec_input, dec_target, text_len
