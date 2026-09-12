#!/usr/bin/env python3
"""
models.py — ASRCTC: compact from-scratch char-CTC encoder (NOT Whisper).

log-mel [B, 80, T] → 2× strided Conv2d (time ÷4, freq ÷4) → linear → pre-norm
transformer encoder (sinusoidal PE) → per-frame char logits [T/4, B, 29].
~8 M params: big enough to read mels, small enough that ~70 h of speech from
scratch trains in a couple of hours.
"""

import math

import torch
import torch.nn as nn

from config import CFG


def _out_len(n):
    """time length after the two k=3 s=2 p=1 convs."""
    n = (n + 2 - 3) // 2 + 1
    n = (n + 2 - 3) // 2 + 1
    return n


class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):                      # x: (B, T, d)
        return x + self.pe[:x.shape[1]].unsqueeze(0)


class ASRCTC(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
        )
        f_sub = _out_len(cfg.n_mels)           # 80 → 20
        self.proj = nn.Linear(64 * f_sub, cfg.d_model)
        self.pe = SinusoidalPE(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout, activation='gelu', batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, mel, mel_lens):
        """mel (B, n_mels, T), mel_lens (B,) → logits (T', B, V), out_lens (B,)"""
        x = self.conv(mel.unsqueeze(1))                    # (B, 64, F', T')
        B, C, F, Tt = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, Tt, C * F)    # (B, T', 64*F')
        x = self.drop(self.pe(self.proj(x)))
        out_lens = _out_len(mel_lens)
        pad = torch.arange(Tt, device=x.device)[None, :] >= out_lens[:, None].to(x.device)
        x = self.encoder(x, src_key_padding_mask=pad)
        logits = self.head(self.norm(x))                   # (B, T', V)
        return logits.transpose(0, 1), out_lens            # (T', B, V) for CTC

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def build(cfg=CFG):
    return ASRCTC(cfg)
