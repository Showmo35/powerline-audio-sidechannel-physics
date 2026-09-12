#!/usr/bin/env python3
"""frontend_stack16.py — powerline → 16 kHz per-harmonic mel stack (model input)."""

import numpy as np
import torch
import torchaudio.functional as AF
from scipy.signal import welch

from config import SetupConfig


def downsample(x, src_sr, dst_sr):
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(src_sr), int(dst_sr))
    return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def detect_mains(x, sr, guess, search):
    nperseg = min(65536, len(x))
    f, P = welch(x, fs=sr, nperseg=nperseg, window='blackman')
    m = np.abs(f - guess) < search
    return float(f[m][np.argmax(P[m])]) if m.any() else guess


def _fbanks(cfg, n_f):
    return AF.melscale_fbanks(n_freqs=n_f, f_min=cfg.fmin, f_max=cfg.fmax,
                              n_mels=cfg.n_mels, sample_rate=cfg.sr,
                              norm=None, mel_scale='htk').numpy()


def am_sideband_stack16(cap16, f_mains, cfg: SetupConfig):
    """[2*n_harmonics, n_mels, T] log-mel stack at 16 kHz, hop=cfg.hop."""
    x = torch.from_numpy(cap16).float()
    win = torch.hann_window(cfg.win)
    Y = torch.stft(x, n_fft=cfg.n_fft, hop_length=cfg.hop, win_length=cfg.win,
                   window=win, center=True, return_complex=True)
    Y_abs = Y.abs().numpy()
    n_f, T = Y_abs.shape
    df = cfg.sr / cfg.n_fft
    f_grid = np.arange(n_f) * df
    fb = _fbanks(cfg, n_f)
    chans = []
    for n in range(1, cfg.n_harmonics + 1):
        for f_sb in (n * f_mains + f_grid, np.abs(n * f_mains - f_grid)):
            bins = np.round(f_sb / df).astype(int)
            ok = (bins >= 0) & (bins < n_f)
            X = np.zeros((n_f, T), dtype=np.float32)
            X[ok, :] = Y_abs[bins[ok], :]
            chans.append((X.T @ fb).T)
    stack = np.stack(chans).astype(np.float32)
    return np.log(np.maximum(stack, 1e-5)).astype(np.float32)
