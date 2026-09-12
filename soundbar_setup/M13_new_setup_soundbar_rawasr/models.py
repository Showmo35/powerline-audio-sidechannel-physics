#!/usr/bin/env python3
"""
models.py — the five architectures from Modules 3-7, ported to sit on the shared
learned raw front-end (frontend_raw.RawFrontEnd).  Every arch is wrapped in a
uniform `RawASR` module whose forward returns a dict, so train.py treats them the
same.

  m3  CTCEncoder              (Conv2d + BiGRU + CTC)              — Module3 recognizer
  m4  PowerlineUNet + CTC     (U-Net enhance, L1+MSTFT, then CTC) — Module4 enhancer
  m5  ConformerCTCModel       (Conformer + CTC)                   — Module5 recognizer
  m6  HybridCTCAttention      (Conformer + CTC + attn decoder)    — Module6 recognizer
  m7  FullSubNet + CTC        (FullSubNet enhance, then CTC)      — Module7 enhancer

The architectures are copied faithfully from their source modules; the only change
is n_mels/n_freqs = cfg.fe_out_dim (the learned front-end's feature height) instead
of a fixed 80-bin mel.  Enhancement models (m4, m7) additionally predict a clean
log-mel (target from the reference wav) and feed the enhanced features to a CTC
recognizer head — the "test on raw data" analogue of their original enhance→ASR
chain.  All archs are scored by CTC-greedy word WER for apples-to-apples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontend_raw import RawFrontEnd
from text import VOCAB_SIZE, DEC_VOCAB_SIZE


# ════════════════════════════════════════════════════════════════════════════
# Module 3 — CTC Conv+BiGRU recognizer
# ════════════════════════════════════════════════════════════════════════════
class CTCEncoder(nn.Module):
    def __init__(self, n_mels=80, hidden=256, n_layers=2,
                 vocab_size=VOCAB_SIZE, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (2, 2), (1, 1)), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, (3, 3), (2, 2), (1, 1)), nn.BatchNorm2d(32), nn.GELU(),
        )
        gru_in = 32 * (n_mels // 4)
        self.gru = nn.GRU(gru_in, hidden, n_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):                       # (B,1,M,T) -> (T',B,V)
        c = self.conv(x)
        B, C, M, T = c.shape
        c = c.permute(0, 3, 1, 2).reshape(B, T, C * M)
        out, _ = self.gru(c)
        logits = self.linear(self.dropout(out))
        return logits.permute(1, 0, 2)


# ════════════════════════════════════════════════════════════════════════════
# Conformer building blocks (shared by Module 5 & 6)
# ════════════════════════════════════════════════════════════════════════════
class FeedForward(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * expansion),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MHSA(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        xn = self.norm(x)
        out, _ = self.attn(xn, xn, xn)
        return self.dropout(out)


class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size,
                            padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x).transpose(1, 2)
        x = self.glu(self.pw1(x))
        x = self.act(self.bn(self.dw(x)))
        x = self.dropout(self.pw2(x))
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, 4, dropout)
        self.self_attn = MHSA(d_model, num_heads, dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, 4, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.self_attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class Conv2dSubsampling(nn.Module):
    def __init__(self, n_mels, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU())
        self.linear = nn.Linear(32 * (n_mels // 4), d_model)

    def forward(self, x):
        x = self.conv(x)
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fr)
        return self.linear(x)


# ════════════════════════════════════════════════════════════════════════════
# Module 5 — Conformer CTC
# ════════════════════════════════════════════════════════════════════════════
class ConformerCTCModel(nn.Module):
    def __init__(self, n_mels=80, vocab_size=VOCAB_SIZE, d_model=256,
                 n_layers=6, num_heads=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.subsample = Conv2dSubsampling(n_mels, d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel, dropout)
            for _ in range(n_layers)])
        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.subsample(x)
        for b in self.blocks:
            x = b(x)
        return self.ctc_head(x).permute(1, 0, 2)


# ════════════════════════════════════════════════════════════════════════════
# Module 6 — Hybrid CTC/Attention
# ════════════════════════════════════════════════════════════════════════════
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                               batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                                batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        t = self.norm1(tgt)
        t, _ = self.self_attn(t, t, t, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(t)
        t = self.norm2(tgt)
        t, _ = self.cross_attn(t, memory, memory)
        tgt = tgt + self.dropout(t)
        t = self.norm3(tgt)
        return tgt + self.dropout(self.ff(t))


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size=DEC_VOCAB_SIZE, d_model=256, num_heads=4,
                 n_layers=3, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_tokens, memory):
        B, S = tgt_tokens.shape
        pos = torch.arange(S, device=tgt_tokens.device).unsqueeze(0)
        x = self.embed(tgt_tokens) + self.pos_embed(pos)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=mask)
        return self.output_proj(self.norm(x))


class HybridCTCAttentionModel(nn.Module):
    def __init__(self, n_mels=80, ctc_vocab=VOCAB_SIZE, dec_vocab=DEC_VOCAB_SIZE,
                 d_model=256, enc_layers=6, dec_layers=3, num_heads=4,
                 conv_kernel=31, dropout=0.1, max_len=512):
        super().__init__()
        self.subsample = Conv2dSubsampling(n_mels, d_model)
        self.encoder = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel, dropout)
            for _ in range(enc_layers)])
        self.ctc_head = nn.Linear(d_model, ctc_vocab)
        self.decoder = TransformerDecoder(dec_vocab, d_model, num_heads,
                                          dec_layers, max_len=max_len, dropout=dropout)

    def encode(self, x):
        x = self.subsample(x)
        for b in self.encoder:
            x = b(x)
        return x

    def forward(self, x, tgt_tokens=None):
        enc = self.encode(x)
        ctc = self.ctc_head(enc).permute(1, 0, 2)
        if tgt_tokens is not None:
            return ctc, self.decoder(tgt_tokens, enc)
        return ctc


# ════════════════════════════════════════════════════════════════════════════
# Module 4 — PowerlineUNet enhancer (+ MSTFT loss)
# ════════════════════════════════════════════════════════════════════════════
class _ConvBlock(nn.Module):
    def __init__(self, ci, co, k=3, dropout=0.1):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(ci, co, k, padding=p, bias=False), nn.BatchNorm2d(co),
            nn.LeakyReLU(0.2, True), nn.Dropout2d(dropout),
            nn.Conv2d(co, co, k, padding=p, bias=False), nn.BatchNorm2d(co),
            nn.LeakyReLU(0.2, True))

    def forward(self, x):
        return self.block(x)


class _ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, True), nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.act(x + self.block(x))


class _DownBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv = _ConvBlock(ci, co)
        self.pool = nn.MaxPool2d((1, 2))

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class _UpBlock(nn.Module):
    def __init__(self, ci, sk, co):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False)
        self.conv = _ConvBlock(ci + sk, co)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1]])
        return self.conv(torch.cat([x, skip], dim=1))


class PowerlineUNet(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.enc1 = _DownBlock(1, base_ch)
        self.enc2 = _DownBlock(base_ch, base_ch * 2)
        self.enc3 = _DownBlock(base_ch * 2, base_ch * 4)
        self.enc4 = _DownBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = nn.Sequential(_ResBlock(base_ch * 8), _ResBlock(base_ch * 8))
        self.dec4 = _UpBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.dec3 = _UpBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec2 = _UpBlock(base_ch * 2, base_ch * 2, base_ch)
        self.dec1 = _UpBlock(base_ch, base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        b = self.bottleneck(x4)
        d = self.dec4(b, s4)
        d = self.dec3(d, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)
        return self.out_conv(d) + x


# ════════════════════════════════════════════════════════════════════════════
# Module 7 — Efficient FullSubNet enhancer
# ════════════════════════════════════════════════════════════════════════════
class _FullBand(nn.Module):
    def __init__(self, n_freqs=80, hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(n_freqs)
        self.lstm = nn.LSTM(n_freqs, hidden, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)   # (B,T,F)
        out, _ = self.lstm(self.norm(x))
        return out


class EfficientFullSubNetEnhancer(nn.Module):
    def __init__(self, n_freqs=80, fb_hidden=64, sb_hidden=32,
                 fb_layers=2, sb_layers=2, n_neighbor=3, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_neighbor = n_neighbor
        self.fullband = _FullBand(n_freqs, fb_hidden, fb_layers, dropout)
        sb_in = fb_hidden * 2 + (2 * n_neighbor + 1)
        self.sb_lstm = nn.LSTM(sb_in, sb_hidden, sb_layers, batch_first=True,
                               dropout=dropout if sb_layers > 1 else 0.0)
        self.sb_proj = nn.Linear(sb_hidden, 1)

    def forward(self, x):
        B, C, Fq, T = x.shape
        noisy = x.squeeze(1)
        fb = self.fullband(x)                                  # (B,T,fb)
        padded = F.pad(noisy, (0, 0, self.n_neighbor, self.n_neighbor), mode='reflect')
        nb = padded.unfold(1, 2 * self.n_neighbor + 1, 1)      # (B,Fq,T,2n+1)
        fb_exp = fb.unsqueeze(1).expand(B, Fq, T, -1)
        comb = torch.cat([fb_exp, nb], dim=-1).reshape(B * Fq, T, -1)
        out, _ = self.sb_lstm(comb)
        gains = torch.sigmoid(self.sb_proj(out)).squeeze(-1).reshape(B, Fq, T)
        return (gains * noisy).unsqueeze(1) + x


# ════════════════════════════════════════════════════════════════════════════
# Multi-scale STFT loss (enhancement archs)
# ════════════════════════════════════════════════════════════════════════════
class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

    def _at(self, pred, target, s):
        if s > 1:
            pred = F.avg_pool2d(pred, s)
            target = F.avg_pool2d(target, s)
        sc = torch.norm(target - pred, p='fro') / (torch.norm(target, p='fro') + 1e-8)
        return sc + F.l1_loss(pred, target)

    def forward(self, pred, target):
        return sum(self._at(pred, target, s) for s in self.scales) / len(self.scales)


# ════════════════════════════════════════════════════════════════════════════
# Unified wrapper:  raw 200 kHz → shared front-end → arch → dict
# ════════════════════════════════════════════════════════════════════════════
ARCHS = ('m3', 'm4', 'm5', 'm6', 'm7')
_ENH = ('m4', 'm7')


class RawASR(nn.Module):
    def __init__(self, arch, cfg):
        super().__init__()
        assert arch in ARCHS, arch
        self.arch = arch
        self.cfg = cfg
        self.frontend = RawFrontEnd(cfg)
        Fdim = cfg.fe_out_dim

        if arch == 'm3':
            self.rec = CTCEncoder(n_mels=Fdim)
        elif arch == 'm5':
            self.rec = ConformerCTCModel(n_mels=Fdim)
        elif arch == 'm6':
            self.rec = HybridCTCAttentionModel(n_mels=Fdim)
        elif arch == 'm4':
            self.enh = PowerlineUNet(base_ch=64)
            self.rec = CTCEncoder(n_mels=Fdim)
            self.mstft = MultiScaleSTFTLoss()
        elif arch == 'm7':
            self.enh = EfficientFullSubNetEnhancer(n_freqs=Fdim)
            self.rec = CTCEncoder(n_mels=Fdim)
            self.mstft = MultiScaleSTFTLoss()

    @property
    def is_enh(self):
        return self.arch in _ENH

    def forward(self, raw, tgt_tokens=None):
        feats = self.frontend(raw)                 # (B,1,F,T)
        out = {'feats': feats}
        if self.is_enh:
            enh = self.enh(feats)
            out['enh'] = enh
            out['ctc'] = self.rec(enh)
        elif self.arch == 'm6':
            res = self.rec(feats, tgt_tokens)
            if tgt_tokens is not None:
                out['ctc'], out['attn'] = res
            else:
                out['ctc'] = res
        else:
            out['ctc'] = self.rec(feats)
        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build(arch, cfg):
    return RawASR(arch, cfg)
