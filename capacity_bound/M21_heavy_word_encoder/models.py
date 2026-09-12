#!/usr/bin/env python3
"""
models.py — HeavyWordNet: three streams from the raw 200 kHz, fused, big transformer.

  raw [B, a_len]
   ├─ RAW  : unfold(patch_len, hop) -> Linear (lossless full signal)      -> [B,T,d]
   ├─ STFT : |STFT| log (all harmonics + sidebands) -> conv freq -> Linear -> [B,T,d]
   └─ ENV  : multi-scale RMS envelope (prosody)      -> Linear             -> [B,T,d]
  concat the 3 token streams (+ stream/type emb) + CLS -> 16-layer transformer -> K-way.
Everything is provided explicitly and uncompressed; the transformer attends across
all three. ~120 M params.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


class Streams(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.register_buffer('win', torch.hann_window(cfg.w_win), persistent=False)
        # RAW patchify
        self.raw_proj = nn.Sequential(nn.LayerNorm(cfg.patch_len),
                                      nn.Linear(cfg.patch_len, 1024), nn.GELU(), nn.Linear(1024, d))
        # STFT freq-conv
        self.stft_conv = nn.Sequential(
            nn.Conv1d(cfg.n_wbins, 512, 1), nn.GELU(),
            nn.Conv1d(512, d, 1))
        # ENV
        self.env_proj = nn.Sequential(nn.Linear(len(cfg.env_scales_ms), 128), nn.GELU(), nn.Linear(128, d))

    def _to_T(self, x):                                  # [B, C, t] -> [B, C, n_frames]
        if x.shape[-1] == self.cfg.n_frames:
            return x
        return F.interpolate(x, size=self.cfg.n_frames, mode='linear', align_corners=False)

    def forward(self, raw):
        cfg = self.cfg
        B = raw.shape[0]
        # RAW
        xr = F.pad(raw, (0, cfg.patch_hop)).unfold(-1, cfg.patch_len, cfg.patch_hop)   # [B, ~T2, patch]
        raw_tok = self.raw_proj(xr)                                                     # [B, T2, d]
        raw_tok = self._to_T(raw_tok.transpose(1, 2)).transpose(1, 2)                   # [B, T, d]
        # STFT
        spec = torch.stft(raw, n_fft=cfg.w_nfft, hop_length=cfg.w_hop, win_length=cfg.w_win,
                          window=self.win, return_complex=True, center=True)
        wide = torch.log(spec.abs() ** 2 + cfg.log_eps)                                 # [B, n_wbins, t]
        stft_tok = self._to_T(self.stft_conv(wide)).transpose(1, 2)                     # [B, T, d]
        # ENV (multi-scale RMS) — O(N): downsample power to frame grid, then smooth.
        # (Was a 51k-tap conv1d = pathologically slow; this is ~1000x faster, same info.)
        p = raw[:, None] ** 2                                                            # [B,1,N]
        base = torch.sqrt(F.adaptive_avg_pool1d(p, cfg.n_frames).clamp_min(1e-12))[:, 0]  # [B,T] ~10ms RMS
        envs = []
        for ms in cfg.env_scales_ms:                                                    # smoothing scales
            k = max(1, int(round(ms / (1000 * cfg.win_s / cfg.n_frames))))              # ms -> frames
            if k <= 1:
                envs.append(base)
            else:
                envs.append(F.avg_pool1d(base[:, None], k, stride=1, padding=k // 2)[:, 0, :cfg.n_frames])
        env_tok = self.env_proj(torch.stack(envs, -1))                                  # [B, T, d]
        return raw_tok, stft_tok, env_tok


class HeavyWordNet(nn.Module):
    def __init__(self, n_classes, cfg=CFG):
        super().__init__()
        self.cfg = cfg; d = cfg.d_model
        self.streams = Streams(cfg)
        self.type_emb = nn.Parameter(torch.zeros(3, 1, d))       # raw / stft / env
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_frames, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        for p in (self.type_emb, self.pos, self.cls):
            nn.init.trunc_normal_(p, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, int(d * cfg.mlp_ratio),
                                           dropout=cfg.dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, cfg.depth)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, raw):
        B = raw.shape[0]
        r, s, e = self.streams(raw)
        r = r + self.pos + self.type_emb[0]
        s = s + self.pos + self.type_emb[1]
        e = e + self.pos + self.type_emb[2]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, r, s, e], dim=1)                     # [B, 1+3T, d]
        x = self.tf(x)
        return self.head(self.norm(x[:, 0]))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def build(n_classes, cfg=CFG):
    return HeavyWordNet(n_classes, cfg)
