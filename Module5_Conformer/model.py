"""
model.py
--------
Module 5: Conformer CTC encoder for speech recognition.

Replaces the BiGRU CTC encoder with a Conformer - the current SOTA for
noisy/robust ASR. Combines convolution (local patterns) with self-attention
(long-range context).

Architecture:
  Conv2dSubsampling (factor=4) -> N x ConformerBlock -> Linear CTC head

ConformerBlock (Macaron-style):
  x -> FF/2 -> MHSA -> ConvModule -> FF/2 -> LayerNorm

~10M params at d_model=256, n_layers=6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

VOCAB_SIZE = 29
BLANK_IDX  = 0

IDX_TO_CHAR = {0: '', 1: ' ', 28: "'"}
for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
    IDX_TO_CHAR[i + 2] = c


# =============================================================================
# Conformer Components
# =============================================================================

class FeedForward(nn.Module):
    """Feed-forward module with Swish activation and dropout."""

    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with pre-norm."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm)
        return self.dropout(out)


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module:
      LayerNorm -> Pointwise Conv (2x expand) -> GLU -> Depthwise Conv
      -> BatchNorm -> Swish -> Pointwise Conv -> Dropout
    """

    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = self.norm(x)
        x = x.transpose(1, 2)     # (B, D, T)
        x = self.pointwise1(x)    # (B, 2D, T)
        x = self.glu(x)           # (B, D, T)
        x = self.depthwise(x)     # (B, D, T)
        x = self.bn(x)
        x = self.act(x)
        x = self.pointwise2(x)    # (B, D, T)
        x = self.dropout(x)
        return x.transpose(1, 2)   # (B, T, D)


class ConformerBlock(nn.Module):
    """
    Macaron-style Conformer block:
      x = x + 0.5 * FF1(x)
      x = x + MHSA(x)
      x = x + ConvModule(x)
      x = x + 0.5 * FF2(x)
      x = LayerNorm(x)
    """

    def __init__(self, d_model=256, num_heads=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, expansion=4, dropout=dropout)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, expansion=4, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.self_attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class Conv2dSubsampling(nn.Module):
    """
    Subsamples input by factor of 4 using 2 strided conv layers.
    Input:  (B, 1, n_mels, T)
    Output: (B, T//4, d_model)
    """

    def __init__(self, n_mels, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(32 * (n_mels // 4), d_model)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = self.conv(x)         # (B, 32, n_mels//4, T//4)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        return self.linear(x)    # (B, T//4, d_model)


# =============================================================================
# ConformerCTCModel
# =============================================================================

class ConformerCTCModel(nn.Module):
    """
    Conformer encoder with CTC head for speech recognition.

    Input : (B, 1, 80, T) - mel spectrogram
    Output: (T', B, V)    - CTC logits, T'=T//4

    Args:
        n_mels: number of mel bins (default 80)
        vocab_size: CTC vocab size (default 29)
        d_model: model dimension (default 256)
        n_layers: number of Conformer blocks (default 6)
        num_heads: attention heads (default 4)
        conv_kernel: depthwise conv kernel size (default 31)
        dropout: dropout rate (default 0.1)
    """

    def __init__(self, n_mels=80, vocab_size=VOCAB_SIZE, d_model=256,
                 n_layers=6, num_heads=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.subsample = Conv2dSubsampling(n_mels, d_model)
        self.conformer = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel, dropout)
            for _ in range(n_layers)
        ])
        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (B, 1, 80, T)
        x = self.subsample(x)        # (B, T//4, d_model)
        for block in self.conformer:
            x = block(x)             # (B, T//4, d_model)
        logits = self.ctc_head(x)    # (B, T//4, vocab_size)
        return logits.permute(1, 0, 2)  # (T, B, V) for CTC

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def greedy_decode(logits):
    """CTC greedy decode. logits: (T, B, V) -> list of strings."""
    preds = logits.argmax(dim=-1).T    # (B, T)
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


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    model = ConformerCTCModel(
        n_mels=80, d_model=256, n_layers=6,
        num_heads=4, conv_kernel=31, dropout=0.1)
    print(f"ConformerCTCModel params: {model.count_params():,}")

    x = torch.randn(2, 1, 80, 498)
    logits = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {logits.shape}")

    texts = greedy_decode(logits)
    print(f"Decoded: {texts}")
    print("Test passed.")
