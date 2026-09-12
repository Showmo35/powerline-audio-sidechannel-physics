"""
dataset.py
----------
Module 9: Dataset for UNet-enhanced mel → Whisper fine-tuning.

Each item:
  input_features: (80, 3000) float32  -- UNet-enhanced mel padded to Whisper size
  labels:         list[int]           -- tokenised transcript

The enhanced mels are 5s clips (498 frames). They are zero-padded to 3000
frames (Whisper's 30-second window) so the encoder treats the rest as silence.
Whisper's cross-attention naturally down-weights padding positions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any

WHISPER_N_FRAMES = 3000   # 30s × 100 frames/s


class UNetMelDataset(Dataset):
    """
    Wraps (enhanced_mel, text) arrays for Whisper fine-tuning.
    Bypasses WhisperProcessor feature extraction — mel already computed.
    """

    def __init__(self, mels: np.ndarray, texts: np.ndarray, processor):
        """
        Args:
            mels:      (N, 80, T) UNet-enhanced log-mel spectrograms
            texts:     (N,) transcript strings
            processor: WhisperProcessor (used only for tokenisation)
        """
        self.mels      = mels       # (N, 80, T)
        self.texts     = texts      # (N,) object/str
        self.processor = processor

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        mel  = self.mels[idx].astype(np.float32)   # (80, T)
        text = str(self.texts[idx])

        # Pad / truncate time axis to Whisper's expected 3000 frames
        T = mel.shape[1]
        if T < WHISPER_N_FRAMES:
            padded = np.zeros((80, WHISPER_N_FRAMES), dtype=np.float32)
            padded[:, :T] = mel
        else:
            padded = mel[:, :WHISPER_N_FRAMES]

        input_features = torch.from_numpy(padded)   # (80, 3000)

        # Tokenise text
        labels = self.processor.tokenizer(text).input_ids

        return {'input_features': input_features, 'labels': labels}


@dataclass
class UNetWhisperCollator:
    """Pads input_features (all same size) and labels (variable length)."""
    pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack(
            [f['input_features'].clone().detach() for f in features]
        )   # (B, 80, 3000)

        label_seqs = [f['labels'] for f in features]
        max_len    = max(len(l) for l in label_seqs)
        padded     = [l + [self.pad_token_id] * (max_len - len(l))
                      for l in label_seqs]
        labels = torch.tensor(padded, dtype=torch.long)

        return {'input_features': input_features, 'labels': labels}
