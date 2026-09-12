#!/usr/bin/env python3
"""
frontend.py — the AM-sideband mel front-end (ground-up reimplementation).

Physical model (validated in Capture_Analysis at mel r ≈ 0.89):
    audio at frequency f  →  powerline energy at  n·f_mains ± f
i.e. the soundbar's audio amplitude-modulates the mains harmonics, so the speech
spectrum lives in the SIDEBANDS around 60 Hz, 120 Hz, 180 Hz, …  This front-end
folds those sidebands back to baseband and mels them.

Output is a *linear-magnitude* mel [n_mels, T] — exactly what Griffin-Lim wants,
and (after log) what Whisper/WaveRNN want.  No reference signal is used, so this
is the scalable, blind front-end the whole pipeline is built on.
"""

import numpy as np
import torch
import torchaudio.functional as AF
from scipy.signal import welch

from config import SetupConfig


def downsample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(src_sr), int(dst_sr))
    return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def detect_mains(x: np.ndarray, sr: int, guess: float, search: float) -> float:
    """Refine the mains frequency from the long-term spectrum (±search of guess)."""
    nperseg = min(65536, len(x))
    f, Pxx = welch(x, fs=sr, nperseg=nperseg, window='blackman')
    m = np.abs(f - guess) < search
    if not m.any():
        print(f'[mains] no peak near {guess} Hz — using guess')
        return guess
    peak = float(f[m][np.argmax(Pxx[m])])
    snr_db = 10 * np.log10(Pxx[m].max() / (Pxx.mean() + 1e-30))
    print(f'[mains] f_mains = {peak:.3f} Hz  ({snr_db:.1f} dB above mean)')
    return peak


def am_sideband_mel(cap_ds: np.ndarray, f_mains: float, cfg: SetupConfig) -> np.ndarray:
    """Linear-magnitude mel [n_mels, T] built from AM sidebands of the capture.

    1. STFT of the downsampled capture on the mel analysis grid.
    2. For each harmonic n and both sidebands, copy |Y(|n·f_mains ± f|)| to
       baseband bin f; sum and average.
    3. Apply the mel filterbank.
    """
    x = torch.from_numpy(cap_ds).float()
    win = torch.hann_window(cfg.mel_win)
    Y = torch.stft(x, n_fft=cfg.mel_n_fft, hop_length=cfg.mel_hop,
                   win_length=cfg.mel_win, window=win,
                   center=True, return_complex=True)          # [F, T]
    Y_abs = Y.abs().numpy()
    n_f, n_t = Y_abs.shape
    df = cfg.aud_sr / cfg.mel_n_fft
    f_grid = np.arange(n_f) * df

    X = np.zeros_like(Y_abs)
    for n in range(1, cfg.n_harmonics + 1):
        for f_sb in (n * f_mains + f_grid, np.abs(n * f_mains - f_grid)):
            bins = np.round(f_sb / df).astype(int)
            ok = (bins >= 0) & (bins < n_f)
            X[ok, :] += Y_abs[bins[ok], :]
    X /= (2 * cfg.n_harmonics)
    print(f'[mel] {cfg.n_harmonics} harmonics × 2 sidebands accumulated')

    # mel filterbank (linear magnitude in, linear-magnitude mel out)
    fb = AF.melscale_fbanks(n_freqs=n_f, f_min=cfg.mel_fmin, f_max=cfg.mel_fmax,
                            n_mels=cfg.mel_n_mels, sample_rate=cfg.aud_sr,
                            norm=None, mel_scale='htk')          # [n_f, n_mels]
    mel = (torch.from_numpy(X).float().T @ fb).T.numpy()         # [n_mels, T]
    print(f'[mel] shape={mel.shape}  range=[{mel.min():.3e}, {mel.max():.3e}]')
    return mel
