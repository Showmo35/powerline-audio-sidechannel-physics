#!/usr/bin/env python3
"""
frontend_stack.py — per-harmonic/sideband mel STACK (model input) + clean target.

The fixed front-end summed 8 harmonics × 2 sidebands into ONE mel — a flat
average Step 4 showed erases phonetic content.  Here each (harmonic, sideband) is
demodulated to baseband as its OWN channel → [2·n_harmonics, n_mels, T], so the
U-Net can learn a non-linear, time/frequency-dependent combination instead of
averaging.  Target = clean reference log-mel on the same grid (htk @ 22050).
"""

import numpy as np
import torch
import torchaudio.functional as AF
import torchaudio.transforms as TT
from scipy.signal import welch

from config import SetupConfig


def downsample(x, src_sr, dst_sr):
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(src_sr), int(dst_sr))
    return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def detect_mains(x, sr, guess, search):
    nperseg = min(65536, len(x))
    f, Pxx = welch(x, fs=sr, nperseg=nperseg, window='blackman')
    m = np.abs(f - guess) < search
    if not m.any():
        return guess
    return float(f[m][np.argmax(Pxx[m])])


def _fbanks(cfg, n_f):
    return AF.melscale_fbanks(n_freqs=n_f, f_min=cfg.mel_fmin, f_max=cfg.mel_fmax,
                              n_mels=cfg.mel_n_mels, sample_rate=cfg.aud_sr,
                              norm=None, mel_scale='htk').numpy()


def am_sideband_stack(cap_ds, f_mains, cfg: SetupConfig):
    """[2*n_harmonics, n_mels, T] — per (harmonic, sideband) magnitude mel."""
    x = torch.from_numpy(cap_ds).float()
    win = torch.hann_window(cfg.mel_win)
    Y = torch.stft(x, n_fft=cfg.mel_n_fft, hop_length=cfg.mel_hop,
                   win_length=cfg.mel_win, window=win, center=True,
                   return_complex=True)
    Y_abs = Y.abs().numpy()
    n_f, T = Y_abs.shape
    df = cfg.aud_sr / cfg.mel_n_fft
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
    return np.stack(chans).astype(np.float32)


_MS = {}


def clean_logmel(audio22, cfg: SetupConfig):
    ms = _MS.get('m')
    if ms is None:
        ms = TT.MelSpectrogram(sample_rate=cfg.aud_sr, n_fft=cfg.mel_n_fft,
                               win_length=cfg.mel_win, hop_length=cfg.mel_hop,
                               n_mels=cfg.mel_n_mels, f_min=cfg.mel_fmin,
                               f_max=cfg.mel_fmax, power=1.0)
        _MS['m'] = ms
    m = ms(torch.from_numpy(audio22).float()).numpy()
    return np.log(np.maximum(m, 1e-5)).astype(np.float32)


def fixed_sum_logmel(stack, cfg: SetupConfig):
    """The fixed front-end's output recomputed from the stack (baseline)."""
    return np.log(np.maximum(stack.astype(np.float32).mean(axis=0), 1e-5))
