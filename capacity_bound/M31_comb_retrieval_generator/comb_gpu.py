#!/usr/bin/env python3
"""
comb_gpu.py — batched GPU demodulation of ALL mains harmonics (full information).

Covering every harmonic to Nyquist means k up to ~0.48*200kHz/60 = 1600. The
time-domain per-harmonic loop in comb.py cannot be used for that: it decimates to
32 kHz (which deletes the entire comb above 16 kHz) and costs one filter pass per
harmonic. Here we do it in ONE shot per window:

    rFFT(window)  ->  gather the +-bw bin slice around EVERY k*f0  ->  batched IFFT
                  ->  complex baseband a_k(t) for all K harmonics simultaneously.

WINDOW / TAPER (this is the subtle part).  A taper is needed to stop the enormous
k=1 carrier leaking across the comb. But a taper is a COMMON multiplicative envelope
on every harmonic -- it corrupts the AM and MANUFACTURES a rank-1 gram, i.e. it would
fake the exact "harmonics are redundant copies of one envelope" result we are testing.
Fix: use a TUKEY window whose FLAT (==1) centre fully covers the region we keep, then
crop to that centre. Inside the crop the taper is identically 1, so the AM is
untouched and no division/compensation is needed.

Framing matches M20 (fixed 1.5 s window centred on the word midpoint, no time
normalization) so the result is directly comparable to M20's wide=31% / env=39%.
"""
import numpy as np
import torch

CAP_SR = 200_000


def tukey_flat(n, pad_frac, device):
    """Tukey window whose flat==1 region is exactly the central (1-2*pad_frac)."""
    w = torch.ones(n, device=device)
    t = int(round(pad_frac * n))
    if t > 1:
        ramp = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, t, device=device)))
        w[:t] = ramp
        w[-t:] = ramp.flip(0)
    return w


@torch.no_grad()
def harmonic_gram_gpu(x, f0, sr=CAP_SR, K=1600, bw=25.0, T=32, pad_frac=0.2):
    """x: (B, N) real float32 padded windows; f0: (B,) per-item mains.

    Returns H (B, K, T) complex64 — the coherent complex baseband of every harmonic,
    cropped to the flat centre of the window.

    bw must be < f0/2 (=30 Hz) so adjacent harmonics never mix.
    """
    dev = x.device
    B, N = x.shape
    w = tukey_flat(N, pad_frac, dev)
    X = torch.fft.rfft(x * w[None, :])                       # (B, Nf)
    Nf = X.shape[-1]
    df = sr / N
    nb = max(2, int(round(bw / df)))
    M = 2 * nb + 1

    ks = torch.arange(1, K + 1, device=dev, dtype=torch.float64)      # (K,)
    centers = torch.round(ks[None, :] * f0[:, None].double() / df)    # (B, K)
    off = torch.arange(-nb, nb + 1, device=dev, dtype=torch.float64)  # (M,)
    idx = (centers[:, :, None] + off[None, None, :])                  # (B, K, M)
    valid = (idx >= 0) & (idx < Nf)
    idx = idx.clamp(0, Nf - 1).long().reshape(B, K * M)

    Sr = torch.gather(X.real, 1, idx).reshape(B, K, M)
    Si = torch.gather(X.imag, 1, idx).reshape(B, K, M)
    S = torch.complex(Sr, Si) * valid.to(Sr.dtype)

    a = torch.fft.ifft(torch.fft.ifftshift(S, dim=-1), dim=-1)        # (B, K, M) baseband
    # crop to the flat centre (taper == 1 there, so AM is untouched)
    lo, hi = int(round(pad_frac * M)), int(round((1 - pad_frac) * M))
    a = a[:, :, lo:hi]
    # resize the time axis to exactly T
    ar = torch.nn.functional.interpolate(a.real.unsqueeze(1), size=(K, T), mode='bilinear',
                                         align_corners=False).squeeze(1)
    ai = torch.nn.functional.interpolate(a.imag.unsqueeze(1), size=(K, T), mode='bilinear',
                                         align_corners=False).squeeze(1)
    return torch.complex(ar, ai)


@torch.no_grad()
def gram_features_gpu(H, eps=1e-12):
    """(B,K,T) complex -> (B, 2, K, T) float: [AM, PM].

    AM: log|H|, per-harmonic mean removed over time (kills the constant gain c_k, so a
        harmonic's absolute strength cannot by itself label a word).
    PM: phase, unwrapped over time and linearly detrended per harmonic (removes residual
        carrier offset/drift), leaving conduction-angle wobble.
    """
    A = torch.log(H.abs() + eps)
    A = A - A.mean(-1, keepdim=True)
    P = torch.angle(H)
    # unwrap along time
    d = torch.diff(P, dim=-1)
    d = d - 2 * np.pi * torch.round(d / (2 * np.pi))
    P = torch.cat([torch.zeros_like(P[..., :1]), torch.cumsum(d, dim=-1)], dim=-1)
    t = torch.arange(P.shape[-1], device=P.device, dtype=P.dtype)
    tm = t - t.mean()
    slope = (P * tm).sum(-1, keepdim=True) / (tm ** 2).sum()
    P = P - slope * tm - P.mean(-1, keepdim=True)
    return torch.stack([A, P], dim=1).float()
