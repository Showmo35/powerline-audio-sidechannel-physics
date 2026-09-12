#!/usr/bin/env python3
"""
data_io.py — readers for the three on-disk artifacts of this setup.

  * powerline capture  : real float32 .bin @ cap_sr  (multi-GB; read windows only)
  * clean reference    : PCM .wav @ 16 kHz           (LibriSpeech concatenation)
  * lag sidecar        : .lag  → powerline-lags-audio in ms

No ffmpeg dependency: WAV is read with the stdlib `wave` module.
"""

import os
import wave
import numpy as np
from math import gcd
from scipy.signal import resample_poly


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32)
    g = gcd(int(src_sr), int(dst_sr))
    return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def read_bin_window(path: str, start_s: float, dur_s: float, sr: int) -> np.ndarray:
    """Read [start_s, start_s+dur_s) of a real-float32 capture without loading it all."""
    offset = int(round(start_s * sr))
    count  = int(round(dur_s * sr))
    x = np.fromfile(path, dtype=np.float32, count=count, offset=offset * 4)  # 4 B/f32
    if len(x) < count:
        print(f'[io] warn: requested {count} samples, got {len(x)} '
              f'(window past EOF of {os.path.basename(path)})')
    return x


def read_wav_window(path: str, start_s: float, dur_s: float, dst_sr: int) -> np.ndarray:
    """Read a window of a PCM WAV → mono float32 at dst_sr."""
    with wave.open(path, 'rb') as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        i0 = int(round(start_s * sr))
        n  = int(round(dur_s * sr))
        w.setpos(min(i0, w.getnframes()))
        raw = w.readframes(n)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    x /= float(np.iinfo(dt).max)
    return _resample(x, sr, dst_sr)


def read_lag_ms(path: str) -> float:
    """Powerline-lags-audio, in ms. Missing sidecar → 0."""
    if not os.path.exists(path):
        print(f'[io] no lag sidecar at {path} — assuming 0 ms')
        return 0.0
    return float(open(path).read().strip())
