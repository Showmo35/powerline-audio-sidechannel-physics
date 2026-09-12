#!/usr/bin/env python3
"""models.py — WideWordNet: conv-over-frequency + transformer-over-time -> K-way word.
The transformer lets it use long-term temporal dependencies in the harmonic pattern."""
import math
import torch
import torch.nn as nn

from config import CFG


class SinPE(nn.Module):
    def __init__(self, d, n=256):
        super().__init__()
        pe = torch.zeros(n, d); pos = torch.arange(n).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1]][None]


class WideWordNet(nn.Module):
    def __init__(self, n_classes, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        # conv stem over frequency (downsample n_wbins), keep time
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, (7, 3), stride=(4, 1), padding=(3, 1)), nn.GELU(),
            nn.Conv2d(32, 64, (5, 3), stride=(4, 1), padding=(2, 1)), nn.GELU(),
            nn.Conv2d(64, 96, (5, 3), stride=(4, 1), padding=(2, 1)), nn.GELU())
        fbins = cfg.n_wbins
        for _ in range(3):
            fbins = (fbins + 2 * (2) - 5) // 4 + 1 if _ else (fbins + 2 * 3 - 7) // 4 + 1
        self.proj = nn.LazyLinear(cfg.d_model)
        self.pe = SinPE(cfg.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, int(cfg.d_model * cfg.mlp_ratio),
                                           dropout=cfg.dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, cfg.depth)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, n_classes)

    def forward(self, wide):                       # [B, n_wbins, T]
        x = self.stem(wide.unsqueeze(1))           # [B, C, F', T]
        B, C, Fp, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fp)
        x = self.proj(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], 1)
        x = self.pe(x)
        x = self.tf(x)
        return self.head(self.norm(x[:, 0]))       # CLS token -> logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def build(n_classes, cfg=CFG):
    return WideWordNet(n_classes, cfg)
