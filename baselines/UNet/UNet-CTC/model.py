"""
model.py
--------
CTC transcription model for UNet-predicted spectrograms.
Same CTCEncoder architecture as E2E pipeline.
"""

import torch
import torch.nn as nn

VOCAB_SIZE = 29
BLANK_IDX  = 0

IDX_TO_CHAR = {0: '', 1: ' ', 28: "'"}
for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
    IDX_TO_CHAR[i + 2] = c


class CTCEncoder(nn.Module):
    """
    Predicted mel → character logits.

    Input : (B, 1, M, T)   M=80 mel bins, T=~498 frames
    Output: (T', B, V)     T'=T//4, V=VOCAB_SIZE  (CTC format)
    """

    def __init__(self, n_mels=80, hidden=256, n_layers=2,
                 vocab_size=VOCAB_SIZE, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        conv_out_freq = n_mels // 4
        gru_in = 32 * conv_out_freq

        self.gru = nn.GRU(
            gru_in, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        c = self.conv(x)                    # (B, 32, 20, T//4)
        B, C, M, T = c.shape
        c = c.permute(0, 3, 1, 2).reshape(B, T, C * M)
        out, _ = self.gru(c)
        out    = self.dropout(out)
        logits = self.linear(out)           # (B, T, V)
        return logits.permute(1, 0, 2)      # (T, B, V)


def greedy_decode(logits):
    """CTC greedy decode. logits: (T, B, V) → list of strings."""
    preds = logits.argmax(dim=-1).T         # (B, T)
    texts = []
    for seq in preds:
        chars = []
        prev = BLANK_IDX
        for idx in seq.tolist():
            if idx != prev and idx != BLANK_IDX:
                chars.append(IDX_TO_CHAR.get(idx, ''))
            prev = idx
        texts.append(''.join(chars))
    return texts
