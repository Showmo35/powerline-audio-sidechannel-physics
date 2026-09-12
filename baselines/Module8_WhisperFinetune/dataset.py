"""
dataset.py
----------
PyTorch Dataset + DataCollator for Whisper fine-tuning on powerline audio.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any


class PowerlineWhisperDataset(Dataset):
    """
    Wraps numpy arrays (audio_clips, texts) for Whisper fine-tuning.

    Each item returns a dict with:
      'input_features': (80, T_mel) float tensor  -- Whisper log-mel
      'labels':         (L,) long tensor           -- tokenised transcript
    """

    def __init__(self, audio_clips: np.ndarray, texts: np.ndarray, processor):
        self.clips     = audio_clips     # (N, win_samples) float32
        self.texts     = texts           # (N,) object/str
        self.processor = processor
        self.sr        = 16_000

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        audio = self.clips[idx].astype(np.float32)
        text  = str(self.texts[idx])

        # Compute Whisper log-mel features
        inputs = self.processor(
            audio,
            sampling_rate=self.sr,
            return_tensors='pt',
        )
        input_features = inputs.input_features[0]   # (80, 3000)

        # Tokenise text (as_target_processor removed in transformers 5.x)
        labels = self.processor.tokenizer(text).input_ids

        return {
            'input_features': input_features,
            'labels':         labels,
        }


@dataclass
class WhisperDataCollator:
    """Pads input_features and labels to batch max length."""
    processor: Any
    pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # input_features: all same size (80, 3000) — just stack
        input_features = torch.stack(
            [f['input_features'].clone().detach() for f in features]
        )

        # labels: variable length — pad with -100 (ignored in cross-entropy)
        label_seqs = [f['labels'] for f in features]
        max_len    = max(len(l) for l in label_seqs)
        padded     = []
        for l in label_seqs:
            pad_len = max_len - len(l)
            padded.append(l + [self.pad_token_id] * pad_len)

        labels = torch.tensor(padded, dtype=torch.long)

        return {
            'input_features': input_features,
            'labels':         labels,
        }
