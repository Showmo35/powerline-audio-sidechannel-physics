#!/usr/bin/env python3
"""
frontend_raw.py — the SHARED learned front-end that lets every model "see" the raw
200 kSps capture.

A stack of strided 1-D convolutions (SincNet/wav2vec-style) ingests the raw window
[B, L] and learns to demodulate + downsample it to a frame-rate feature image
[B, 1, F, T], where F = cfg.fe_out_dim (treated as the "mel height" the ported
Module 3-7 models expect) and T ≈ L / prod(strides).

This REPLACES the fixed AM-sideband mel of M10 / the fixed harmonic stack of M12:
nothing about the 60 Hz harmonics or sidebands is hard-coded — the network finds
whatever structure carries speech (or proves there is none).
"""

import torch
import torch.nn as nn


class RawFrontEnd(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        chans = (1,) + tuple(cfg.fe_channels)
        layers = []
        for i, (k, s) in enumerate(zip(cfg.fe_kernels, cfg.fe_strides)):
            layers += [
                nn.Conv1d(chans[i], chans[i + 1], kernel_size=k, stride=s,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(chans[i + 1]),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)
        # project last conv channel count -> fe_out_dim feature bins
        self.proj = nn.Conv1d(cfg.fe_channels[-1], cfg.fe_out_dim, kernel_size=1)
        self.out_dim = cfg.fe_out_dim

    def forward(self, raw):
        """raw: (B, L) → feature image (B, 1, F, T)."""
        x = raw.unsqueeze(1)             # (B, 1, L)
        x = self.net(x)                  # (B, C, T)
        x = self.proj(x)                 # (B, F, T)
        # per-utterance instance norm over (F,T) keeps scale stable across windows
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        return x.unsqueeze(1)            # (B, 1, F, T)
