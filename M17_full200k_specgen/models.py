#!/usr/bin/env python3
"""
models.py — PLFW-17: lossless patchify encoder + flow-matching DiT + CTC head.

  raw 200 kHz ─ unfold(2000, hop 1000)  ← zero-loss reshape, every sample enters
              ─ LayerNorm → Linear 2000→1024 → GELU → Linear 1024→d   (learned)
              ─ conv-pool ×2 → 100 fps ─ RoPE transformer ─► c[T, d]
  c ─► DiT flow-transformer ─► generated 513-bin log-STFT   (flow-matching loss)
  c ─► CTC head ─► char logits                              (CTC word-loss)
  total = flow + lambda_ctc · CTC

DiT / attention / CTC primitives are M15's verbatim; only the condition encoder
changed (M15's conv stem + wideband mel were both lossy — this front-end is not).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RoPE / attention / GEGLU (same primitives as M14/M15) ────────────────────
def _rope_freqs(seq_len, head_dim, device, base=10000.0):
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    ang = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return torch.cos(ang), torch.sin(ang)


def _apply_rope(x, cos, sin):
    B, H, T, D = x.shape
    x = x.view(B, H, T, D // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    cos = cos[None, None]; sin = sin[None, None]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, h, dropout=0.0):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d); self.drop = dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q = _apply_rope(q, cos, sin); k = _apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, T, C))


class GEGLU(nn.Module):
    def __init__(self, d, ratio=4.0):
        super().__init__()
        hid = int(d * ratio)
        self.fc = nn.Linear(d, hid * 2); self.out = nn.Linear(hid, d)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return self.out(a * F.gelu(b))


class EncBlock(nn.Module):
    def __init__(self, d, h, ratio, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = Attention(d, h, dropout)
        self.n2 = nn.LayerNorm(d); self.mlp = GEGLU(d, ratio)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


# ── lossless patchify encoder ─────────────────────────────────────────────────
class PatchEncoder(nn.Module):
    """Full 200 kHz raw → condition tokens at 100 fps, no fixed lossy transform."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.norm_in = nn.LayerNorm(cfg.patch_len)
        self.embed = nn.Sequential(
            nn.Linear(cfg.patch_len, cfg.d_patch), nn.GELU(),
            nn.Linear(cfg.d_patch, d))
        # 2·fps token rate (50 % patch overlap) → pool to the spec frame rate
        self.pool = nn.Sequential(
            nn.Conv1d(d, d, 4, stride=2, padding=1), nn.GELU())
        self.blocks = nn.ModuleList(
            EncBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.enc_tf_layers))
        self.norm = nn.LayerNorm(d)
        self.T = cfg.n_frames

    def forward(self, raw):                                        # (B, a_len)
        cfg = self.cfg
        # pad one hop so unfold yields exactly 2·n_frames overlapping patches
        x = F.pad(raw, (0, cfg.patch_hop))
        x = x.unfold(-1, cfg.patch_len, cfg.patch_hop)             # (B, 2T, patch_len)
        x = self.embed(self.norm_in(x))                            # (B, 2T, d)
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)           # (B, T, d)
        if x.shape[1] != self.T:
            x = F.interpolate(x.transpose(1, 2), size=self.T,
                              mode='linear', align_corners=False).transpose(1, 2)
        cos, sin = _rope_freqs(x.shape[1], x.shape[-1] // self.blocks[0].attn.h, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)                                        # (B, T, d)


# ── DiT (flow-transformer), same as M14/M15 ──────────────────────────────────
class TimeEmbed(nn.Module):
    def __init__(self, d, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        half = self.freq_dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        a = t[:, None].float() * f[None] * 1000.0
        return self.mlp(torch.cat([torch.cos(a), torch.sin(a)], dim=-1))


def _mod(x, sh, sc):
    return x * (1 + sc[:, None]) + sh[:, None]


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
        x = x + g1[:, None] * self.attn(_mod(self.n1(x), sh1, sc1), cos, sin)
        return x + g2[:, None] * self.mlp(_mod(self.n2(x), sh2, sc2))


class FinalLayer(nn.Module):
    def __init__(self, d, n_out):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.lin = nn.Linear(d, n_out)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        for m in (self.ada[-1], self.lin):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, c):
        sh, sc = self.ada(c).chunk(2, dim=-1)
        return self.lin(_mod(self.norm(x), sh, sc))


# ── PLFW-17 ───────────────────────────────────────────────────────────────────
class PLFW17(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.encoder = PatchEncoder(cfg)
        self.x_in = nn.Linear(cfg.n_bins, d)
        self.c_in = nn.Linear(d, d)
        self.null_cond = nn.Parameter(torch.zeros(d))
        self.t_embed = TimeEmbed(d)
        self.blocks = nn.ModuleList(
            DiTBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.dit_layers))
        self.final = FinalLayer(d, cfg.n_bins)
        self.ctc_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.ctc_vocab))
        self.head_dim = d // cfg.n_heads

    def encode(self, raw):
        return self.encoder(raw)

    def _drop(self, c, mask):
        if mask is None:
            return c
        null = self.null_cond[None, None].expand_as(c)
        m = mask[:, None, None].float()
        return c * (1 - m) + null * m

    def velocity(self, x_t, t, c):
        h = self.x_in(x_t.transpose(1, 2)) + self.c_in(c)
        temb = self.t_embed(t)
        cos, sin = _rope_freqs(h.shape[1], self.head_dim, h.device)
        for blk in self.blocks:
            h = blk(h, temb, cos, sin)
        return self.final(h, temb).transpose(1, 2)

    def frame_mask(self, vframes, T, device):
        idx = torch.arange(T, device=device)[None, :]
        return (idx < vframes[:, None]).float()                    # (B, T)

    def losses(self, raw, spec, vframes, labels, label_lengths):
        """Returns (flow_loss, ctc_loss)."""
        cfg = self.cfg
        B, _, T = spec.shape
        c = self.encode(raw)
        # ---- flow matching (CFG dropout + frame masking) ----
        drop = (torch.rand(B, device=spec.device) < cfg.p_uncond)
        cc = self._drop(c, drop)
        x0 = torch.randn_like(spec)
        t = torch.rand(B, device=spec.device)
        tt = t[:, None, None]
        x_t = (1 - tt) * x0 + tt * spec
        v = self.velocity(x_t, t, cc)
        fm = self.frame_mask(vframes, T, spec.device)[:, None]     # (B,1,T)
        flow = (((v - (spec - x0)) ** 2) * fm).sum() / (fm.sum() * spec.shape[1] + 1e-8)
        # ---- CTC word-loss ----
        logits = self.ctc_head(c)                                  # (B,T,V)
        logp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # (T,B,V)
        in_lens = vframes.clamp(min=1, max=T)
        ctc = F.ctc_loss(logp, labels, in_lens, label_lengths.clamp(min=1),
                         blank=0, zero_infinity=True)
        return flow, ctc

    @torch.no_grad()
    def ctc_logits(self, raw):
        return self.ctc_head(self.encode(raw))                     # (B,T,V)

    @torch.no_grad()
    def sample(self, raw, steps=None, cfg_scale=None):
        cfg = self.cfg
        steps = steps or cfg.sample_steps
        w = cfg.cfg_scale if cfg_scale is None else cfg_scale
        c = self.encode(raw)
        cnull = self.null_cond[None, None].expand_as(c)
        x = torch.randn(raw.shape[0], cfg.n_bins, cfg.n_frames, device=raw.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((raw.shape[0],), i * dt, device=raw.device)
            if w == 1.0:
                v = self.velocity(x, t, c)
            else:
                vc = self.velocity(x, t, c); vu = self.velocity(x, t, cnull)
                v = vu + w * (vc - vu)
            x = x + dt * v
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def build(cfg):
    return PLFW17(cfg)
