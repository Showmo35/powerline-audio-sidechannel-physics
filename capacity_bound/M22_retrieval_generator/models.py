#!/usr/bin/env python3
"""
models.py — PowerLine-Flow (PLF): a conditional rectified-flow DiT that generates
the audio log-mel from a powerline capture.

Pieces
------
PowerlineEncoder   raw 0-16 kHz waveform → frame-aligned condition tokens c[B,T,d]
                   (strided-conv stem that learns the demodulation + transformer)
DiT                flow-transformer over the noisy mel; adaLN-zero on the flow-time
                   t, RoPE self-attention, GEGLU MLP; fuses c by additive injection
PLF                ties them together: flow-matching loss + few-step ODE sampler
                   with classifier-free guidance.

Flow matching (rectified / OT path)
-----------------------------------
  x0 ~ N(0, I)              (noise)
  x1 = standardised mel     (data)
  x_t = (1-t) x0 + t x1 ,   t ~ U(0,1)
  target velocity  u = x1 - x0
  loss = || v_theta(x_t, t, c) - u ||^2
Sampling integrates dx/dt = v_theta(x,t,c) from x0 (t=0) to t=1 in a few Euler
steps; CFG mixes conditioned / null-conditioned velocities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# Rotary position embedding
# ════════════════════════════════════════════════════════════════════════════
def _rope_freqs(seq_len, head_dim, device, base=10000.0):
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    ang = torch.outer(pos, inv)                       # (T, half)
    return torch.cos(ang), torch.sin(ang)             # each (T, half)


def _apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    B, H, T, D = x.shape
    x = x.view(B, H, T, D // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q = _apply_rope(q, cos, sin); k = _apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.drop if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, T, C)
        return self.proj(o)


class GEGLU(nn.Module):
    def __init__(self, d_model, ratio=4.0):
        super().__init__()
        hidden = int(d_model * ratio)
        self.fc = nn.Linear(d_model, hidden * 2)
        self.out = nn.Linear(hidden, d_model)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return self.out(a * F.gelu(b))


# ════════════════════════════════════════════════════════════════════════════
# Powerline condition encoder
# ════════════════════════════════════════════════════════════════════════════
class _EncBlock(nn.Module):
    def __init__(self, d, h, ratio, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = Attention(d, h, dropout)
        self.n2 = nn.LayerNorm(d); self.mlp = GEGLU(d, ratio)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class PowerlineEncoder(nn.Module):
    """Raw 32 kHz window (B, L) → condition tokens (B, T, d_model)."""
    def __init__(self, cfg):
        super().__init__()
        chans = (1,) + tuple(cfg.enc_channels)
        stem = []
        for i, (k, s) in enumerate(zip(cfg.enc_kernels, cfg.enc_strides)):
            stem += [nn.Conv1d(chans[i], chans[i + 1], k, stride=s, padding=k // 2,
                               bias=False),
                     nn.GroupNorm(8, chans[i + 1]), nn.GELU()]
        self.stem = nn.Sequential(*stem)
        self.proj = nn.Conv1d(cfg.enc_channels[-1], cfg.d_model, 1)
        self.blocks = nn.ModuleList(
            _EncBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.enc_tf_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.T = cfg.n_frames

    def forward(self, raw):
        x = self.stem(raw.unsqueeze(1))             # (B, C, T')
        x = self.proj(x)                            # (B, d, T')
        if x.shape[-1] != self.T:
            x = F.interpolate(x, size=self.T, mode='linear', align_corners=False)
        x = x.transpose(1, 2)                        # (B, T, d)
        cos, sin = _rope_freqs(x.shape[1], x.shape[-1] // self.blocks[0].attn.h, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)                          # (B, T, d)


# ════════════════════════════════════════════════════════════════════════════
# Flow-time embedding + DiT block (adaLN-zero)
# ════════════════════════════════════════════════════════════════════════════
class TimeEmbed(nn.Module):
    def __init__(self, d, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):                            # t: (B,) in [0,1]
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device).float() / half)
        a = t[:, None].float() * freqs[None] * 1000.0
        emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
        return self.mlp(emb)                         # (B, d)


def _modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, d, h, ratio, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d, h, dropout)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = GEGLU(d, ratio)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, cos, sin):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1[:, None] * self.attn(_modulate(self.n1(x), sh1, sc1), cos, sin)
        x = x + g2[:, None] * self.mlp(_modulate(self.n2(x), sh2, sc2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, d, n_mels):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.lin = nn.Linear(d, n_mels)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x, c):
        sh, sc = self.ada(c).chunk(2, dim=-1)
        return self.lin(_modulate(self.norm(x), sh, sc))


# ════════════════════════════════════════════════════════════════════════════
# PowerLine-Flow
# ════════════════════════════════════════════════════════════════════════════
class PLF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.encoder = PowerlineEncoder(cfg)
        self.x_in = nn.Linear(cfg.n_mels, d)         # noisy-mel frame → token
        self.c_in = nn.Linear(d, d)                  # condition → token
        self.null_cond = nn.Parameter(torch.zeros(d))   # CFG null condition
        self.t_embed = TimeEmbed(d)
        self.blocks = nn.ModuleList(
            DiTBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.dit_layers))
        self.final = FinalLayer(d, cfg.n_mels)
        self.head_dim = d // cfg.n_heads

    # ---- condition encoding (done once per sampling call) --------------------
    def encode(self, raw):
        return self.encoder(raw)                     # (B, T, d)

    def _drop_cond(self, c, drop_mask):
        if drop_mask is None:
            return c
        null = self.null_cond[None, None, :].expand_as(c)
        m = drop_mask[:, None, None].float()
        return c * (1 - m) + null * m

    # ---- velocity field v(x_t, t, c) -----------------------------------------
    def velocity(self, x_t, t, c):
        """x_t: (B, n_mels, T); t: (B,); c: (B, T, d) → velocity (B, n_mels, T)."""
        h = self.x_in(x_t.transpose(1, 2)) + self.c_in(c)     # (B, T, d)
        temb = self.t_embed(t)                                # (B, d)
        cos, sin = _rope_freqs(h.shape[1], self.head_dim, h.device)
        for blk in self.blocks:
            h = blk(h, temb, cos, sin)
        v = self.final(h, temb)                               # (B, T, n_mels)
        return v.transpose(1, 2)                              # (B, n_mels, T)

    # ---- flow-matching loss --------------------------------------------------
    def flow_loss(self, raw, mel):
        """mel: standardised target (B, n_mels, T)."""
        B = mel.shape[0]
        c = self.encode(raw)
        drop = (torch.rand(B, device=mel.device) < self.cfg.p_uncond)
        c = self._drop_cond(c, drop)
        x0 = torch.randn_like(mel)
        t = torch.rand(B, device=mel.device)
        tt = t[:, None, None]
        x_t = (1 - tt) * x0 + tt * mel
        u = mel - x0                                          # target velocity
        v = self.velocity(x_t, t, c)
        return F.mse_loss(v, u)

    # ---- few-step ODE sampler (classifier-free guided) -----------------------
    @torch.no_grad()
    def sample(self, raw, steps=None, cfg_scale=None):
        cfg = self.cfg
        steps = steps or cfg.sample_steps
        w = cfg.cfg_scale if cfg_scale is None else cfg_scale
        B = raw.shape[0]
        c = self.encode(raw)
        c_null = self.null_cond[None, None, :].expand_as(c)
        x = torch.randn(B, cfg.n_mels, cfg.n_frames, device=raw.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=raw.device)
            if w == 1.0:
                v = self.velocity(x, t, c)
            else:
                v_c = self.velocity(x, t, c)
                v_u = self.velocity(x, t, c_null)
                v = v_u + w * (v_c - v_u)
            x = x + dt * v
        return x                                              # standardised mel

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def build(cfg):
    return PLF(cfg)
