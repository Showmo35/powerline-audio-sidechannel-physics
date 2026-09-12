#!/usr/bin/env python3
"""
models_comb.py — the harmonic-comb front-end for the PLF generator.

The generator (PLF: conditional rectified-flow DiT) is reused UNCHANGED from M22. PLF
only touches its front-end through `self.encoder(raw) -> (B, n_frames, d_model)`
condition tokens, so swapping the encoder is the whole change: everything downstream
(velocity / flow_loss / sample) is identical between the two arms. That is what makes
this a clean A/B:

  ARM raw   PowerlineEncoder : 0-16 kHz decimated waveform      (M22 baseline)
  ARM comb  CombEncoder      : per-harmonic coherent gram (2,K,Tg)  (the hypothesis)

CombEncoder treats the K harmonic axis like a frequency axis: strided convs collapse
K, giving one embedding per frame, then transformer blocks over time, then interpolate
to the DiT's n_frames.

NOTE on bandwidth: each harmonic is demodulated at +-bw (<30 Hz, so harmonics never
mix), so the gram is inherently band-limited to ~25 Hz per harmonic (Tg ~ 50 fps). The
hypothesis is that K such slow channels COLLECTIVELY carry more than the single
broadband envelope -- which is exactly what the non-rank-1 result (PC1 ~ 0.26) implies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import PLF, _EncBlock, _rope_freqs        # reuse M22's blocks verbatim


class CombEncoder(nn.Module):
    """(B, 2, K, Tg) harmonic-gram -> (B, n_frames, d_model) condition tokens."""

    def __init__(self, cfg):
        super().__init__()
        ch = (2, 32, 64, 96, 128)
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], (7, 3), stride=(4, 1), padding=(3, 1)),
                       nn.GroupNorm(8, ch[i + 1]), nn.GELU()]
        self.stem = nn.Sequential(*layers)             # collapses K by 256x
        self.pool = nn.AdaptiveAvgPool2d((4, None))    # -> (B, C, 4, Tg)
        self.proj = nn.Linear(ch[-1] * 4, cfg.d_model)
        self.blocks = nn.ModuleList(
            _EncBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.enc_tf_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.T = cfg.n_frames

    def forward(self, gram):
        x = self.stem(gram)                            # (B, C, K/256, Tg)
        x = self.pool(x)                               # (B, C, 4, Tg)
        B, C, Kp, Tg = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, Tg, C * Kp)   # (B, Tg, C*4)
        x = self.proj(x)                                   # (B, Tg, d)
        if x.shape[1] != self.T:                           # gram fps -> DiT frame rate
            x = F.interpolate(x.transpose(1, 2), size=self.T,
                              mode='linear', align_corners=False).transpose(1, 2)
        cos, sin = _rope_freqs(x.shape[1], x.shape[-1] // self.blocks[0].attn.h, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)


def build(cfg, frontend='comb'):
    """PLF with the chosen front-end. `raw` arg to PLF is the gram when frontend=comb."""
    m = PLF(cfg)
    if frontend == 'comb':
        m.encoder = CombEncoder(cfg)
    return m
