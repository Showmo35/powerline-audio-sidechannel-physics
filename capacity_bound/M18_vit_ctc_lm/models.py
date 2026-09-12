#!/usr/bin/env python3
"""
models.py — ViTCTC: read the mel spectrogram as an IMAGE, emit char logits (CTC).

  mel [B, 80, T]
    Conv2d patchify (freq 16 × time 4, stride = patch)  → [B, d, 5, T/4]
    + separable 2-D learned pos-emb (freq 5, time T/4)
    flatten 5·(T/4) tokens → pre-norm ViT (12 × d512) → reshape [B, 5, T/4, d]
    mean-pool the 5 frequency tokens → [B, T/4, d]  (keep the time axis for CTC)
    LayerNorm → Linear → char logits [T/4, B, 29]
From scratch — no ImageNet init (spectrogram statistics ≠ natural images).
Interface matches M16's ASRCTC: forward(mel, mel_lens) → (logits[T',B,V], out_lens).
"""

import torch
import torch.nn as nn

from config import CFG


def _time_out(n):
    """time length after the strided patch conv (kernel=stride=patch_t)."""
    return n // CFG.patch_t


class ViTCTC(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(1, cfg.d_model,
                               kernel_size=(cfg.patch_f, cfg.patch_t),
                               stride=(cfg.patch_f, cfg.patch_t))
        self.freq_pos = nn.Parameter(torch.zeros(1, cfg.n_freq_patches, 1, cfg.d_model))
        self.time_pos = nn.Parameter(torch.zeros(1, 1, cfg.max_time_patches, cfg.d_model))
        nn.init.trunc_normal_(self.freq_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=int(cfg.d_model * cfg.mlp_ratio),
            dropout=cfg.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.depth)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, mel, mel_lens):
        """mel (B, n_mels, T), mel_lens (B,) → logits (T', B, V), out_lens (B,)."""
        cfg = self.cfg
        B, F, T = mel.shape
        Tp = _time_out(T)
        mel = mel[:, :, :Tp * cfg.patch_t]                        # drop time remainder
        x = self.patch(mel.unsqueeze(1))                          # (B, d, 5, Tp)
        x = x.permute(0, 2, 3, 1)                                 # (B, 5, Tp, d)
        x = x + self.freq_pos + self.time_pos[:, :, :Tp]
        Fp = x.shape[1]
        x = x.reshape(B, Fp * Tp, -1)
        # pad mask over the flattened grid (mask whole time-columns past out_lens)
        out_lens = (mel_lens // cfg.patch_t).clamp(min=1, max=Tp)
        tcol = torch.arange(Tp, device=x.device)[None, :] >= out_lens[:, None].to(x.device)  # (B,Tp)
        pad = tcol[:, None, :].expand(B, Fp, Tp).reshape(B, Fp * Tp)
        x = self.drop(x)
        x = self.encoder(x, src_key_padding_mask=pad)
        x = x.reshape(B, Fp, Tp, -1).mean(dim=1)                  # pool freq → (B, Tp, d)
        logits = self.head(self.norm(x))                         # (B, Tp, V)
        return logits.transpose(0, 1), out_lens                  # (Tp, B, V) for CTC

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def build(cfg=CFG):
    return ViTCTC(cfg)
