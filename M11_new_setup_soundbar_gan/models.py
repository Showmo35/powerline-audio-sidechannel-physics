#!/usr/bin/env python3
"""
models.py — HiFi-GAN generator (conditioned on the powerline mel stack) plus the
standard multi-period (MPD) and multi-scale (MSD) discriminators.

Generator: [B, 2*nh, n_mels, T]  → flatten (ch·mel) → Conv1d → HiFi-GAN upsample
stack (product of rates = hop) with multi-receptive-field ResBlocks → [B, T*hop]
16 kHz waveform.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

LRELU = 0.1


def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


def get_pad(k, d=1):
    return (k * d - d) // 2


class ResBlock(nn.Module):
    def __init__(self, ch, k=3, dil=(1, 3, 5)):
        super().__init__()
        self.c1 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=d,
                                 padding=get_pad(k, d))) for d in dil])
        self.c2 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=1,
                                 padding=get_pad(k, 1))) for _ in dil])
        self.c1.apply(init_weights); self.c2.apply(init_weights)

    def forward(self, x):
        for a, b in zip(self.c1, self.c2):
            y = b(F.leaky_relu(a(F.leaky_relu(x, LRELU)), LRELU))
            x = x + y
        return x


class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = 2 * cfg.n_harmonics * cfg.n_mels
        self.hop = cfg.hop
        self.pre = weight_norm(nn.Conv1d(in_dim, cfg.upsample_initial, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        self.res = nn.ModuleList()
        ch = cfg.upsample_initial
        self.n_k = len(cfg.resblock_kernels)
        for u, k in zip(cfg.upsample_rates, cfg.upsample_kernels):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                ch, ch // 2, k, u, padding=(k - u) // 2)))
            ch //= 2
            for rk, rd in zip(cfg.resblock_kernels, cfg.resblock_dilations):
                self.res.append(ResBlock(ch, rk, rd))
        self.post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights); self.post.apply(init_weights)

    def forward(self, cond):                 # cond: [B, C, M, T]
        B, C, M, T = cond.shape
        x = self.pre(cond.reshape(B, C * M, T))
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU))
            xs = 0
            for j in range(self.n_k):
                xs = xs + self.res[i * self.n_k + j](x)
            x = xs / self.n_k
        x = torch.tanh(self.post(F.leaky_relu(x, LRELU)))
        out = x.squeeze(1)                    # [B, L]
        return out[:, :T * self.hop]          # exact T*hop samples


# ── discriminators (standard HiFi-GAN) ─────────────────────────────────────────
class DiscriminatorP(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), 'reflect')
        x = x.view(b, 1, -1, self.period)
        for c in self.convs:
            x = F.leaky_relu(c(x), LRELU); fmap.append(x)
        x = self.post(x); fmap.append(x)
        return x.flatten(1), fmap


class DiscriminatorS(nn.Module):
    def __init__(self, sn=False):
        super().__init__()
        nrm = spectral_norm if sn else weight_norm
        self.convs = nn.ModuleList([
            nrm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            nrm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            nrm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            nrm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            nrm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            nrm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post = nrm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        x = x.unsqueeze(1)
        for c in self.convs:
            x = F.leaky_relu(c(x), LRELU); fmap.append(x)
        x = self.post(x); fmap.append(x)
        return x.flatten(1), fmap


class MPD(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.d = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, y, yh):
        return _run(self.d, y, yh)


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = nn.ModuleList([DiscriminatorS(sn=(i == 0)) for i in range(3)])
        self.pool = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, y, yh):
        outs = ([], [], [], [])
        ya, yha = y, yh
        for i, d in enumerate(self.d):
            if i:
                ya = self.pool(ya.unsqueeze(1)).squeeze(1)
                yha = self.pool(yha.unsqueeze(1)).squeeze(1)
            r, fr = d(ya); g, fg = d(yha)
            outs[0].append(r); outs[1].append(g); outs[2].append(fr); outs[3].append(fg)
        return outs


def _run(mods, y, yh):
    yr, yg, fr, fg = [], [], [], []
    for d in mods:
        r, frr = d(y); g, fgg = d(yh)
        yr.append(r); yg.append(g); fr.append(frr); fg.append(fgg)
    return yr, yg, fr, fg
