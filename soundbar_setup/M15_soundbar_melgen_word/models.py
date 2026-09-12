#!/usr/bin/env python3
"""
models.py — PLF-W: multi-stream conditional flow-matching DiT + CTC word head.

  raw 200 kHz ─┬─ streamA: strided-conv front-end ───┐
               └─ streamB: wideband log-mel(0-100kHz) ┴─ fuse ─ transformer ─► c[T,d]
  c ─► DiT flow-transformer ─► generated mel   (flow-matching loss)
  c ─► CTC head ─► char logits                 (CTC word-loss, true text)
  total = flow + lambda_ctc · CTC
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ── RoPE / attention / GEGLU (same primitives as M14) ────────────────────────
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


# ── multi-stream powerline encoder ───────────────────────────────────────────
class MultiStreamEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # stream A: raw 200 kHz strided conv
        chans = (1,) + tuple(cfg.aChannels)
        stem = []
        for i, (k, s) in enumerate(zip(cfg.aKernels, cfg.aStrides)):
            stem += [nn.Conv1d(chans[i], chans[i + 1], k, stride=s, padding=k // 2, bias=False),
                     nn.GroupNorm(8, chans[i + 1]), nn.GELU()]
        self.aStem = nn.Sequential(*stem)
        self.aProj = nn.Conv1d(cfg.aChannels[-1], d, 1)
        # stream B: wideband log-mel of the 200 kHz (0-100 kHz)
        self.wide_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.cap_sr, n_fft=cfg.b_n_fft, hop_length=cfg.b_hop,
            n_mels=cfg.b_n_mels, f_min=0.0, f_max=cfg.b_fmax, power=2.0)
        self.bProj = nn.Sequential(nn.Conv1d(cfg.b_n_mels, d, 5, padding=2), nn.GELU())
        # fuse + transformer
        self.fuse = nn.Conv1d(2 * d, d, 1)
        self.blocks = nn.ModuleList(
            EncBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.enc_tf_layers))
        self.norm = nn.LayerNorm(d)
        self.T = cfg.n_frames

    def forward(self, raw):
        a = self.aProj(self.aStem(raw.unsqueeze(1)))               # (B, d, Ta)
        if a.shape[-1] != self.T:
            a = F.interpolate(a, size=self.T, mode='linear', align_corners=False)
        # STFT/FFT must run in fp32 — under fp16 autocast it yields NaN/Inf
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
            wm = torch.log(self.wide_mel(raw.float()) + 1e-5)      # (B, b_mels, Tb) fp32
        b = self.bProj(wm)
        if b.shape[-1] != self.T:
            b = F.interpolate(b, size=self.T, mode='linear', align_corners=False)
        x = self.fuse(torch.cat([a, b], dim=1)).transpose(1, 2)    # (B, T, d)
        cos, sin = _rope_freqs(self.T, x.shape[-1] // self.blocks[0].attn.h, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)                                        # (B, T, d)


# ── DiT (flow-transformer), same as M14 ──────────────────────────────────────
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
    def __init__(self, d, n_mels):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.lin = nn.Linear(d, n_mels)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        for m in (self.ada[-1], self.lin):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, c):
        sh, sc = self.ada(c).chunk(2, dim=-1)
        return self.lin(_mod(self.norm(x), sh, sc))


# ── PLF-W ────────────────────────────────────────────────────────────────────
class PLFW(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.encoder = MultiStreamEncoder(cfg)
        self.x_in = nn.Linear(cfg.n_mels, d)
        self.c_in = nn.Linear(d, d)
        self.null_cond = nn.Parameter(torch.zeros(d))
        self.t_embed = TimeEmbed(d)
        self.blocks = nn.ModuleList(
            DiTBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.dit_layers))
        self.final = FinalLayer(d, cfg.n_mels)
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

    def losses(self, raw, mel, vframes, labels, label_lengths):
        """Returns (flow_loss, ctc_loss)."""
        cfg = self.cfg
        B, _, T = mel.shape
        c = self.encode(raw)
        # ---- flow matching (CFG dropout + frame masking) ----
        drop = (torch.rand(B, device=mel.device) < cfg.p_uncond)
        cc = self._drop(c, drop)
        x0 = torch.randn_like(mel)
        t = torch.rand(B, device=mel.device)
        tt = t[:, None, None]
        x_t = (1 - tt) * x0 + tt * mel
        v = self.velocity(x_t, t, cc)
        fm = self.frame_mask(vframes, T, mel.device)[:, None]      # (B,1,T)
        flow = (((v - (mel - x0)) ** 2) * fm).sum() / (fm.sum() * mel.shape[1] + 1e-8)
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
        x = torch.randn(raw.shape[0], cfg.n_mels, cfg.n_frames, device=raw.device)
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
    return PLFW(cfg)
