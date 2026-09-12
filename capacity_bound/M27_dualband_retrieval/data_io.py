#!/usr/bin/env python3
"""
data_io.py — on-the-fly windowed readers for the raw capture and clean reference.

No ffmpeg, no precomputed features.  A window is read straight from the 1.44 GB
.bin (np.fromfile with byte offset → only the needed bytes touch disk), decimated
200 kHz → 32 kHz (which also IS the 0-16 kHz band-limit, via the resample_poly
anti-alias filter), and paired with the matching 16 kHz reference-wav window.
"""

import os
import wave
from functools import lru_cache
from math import gcd

import numpy as np
from scipy.signal import resample_poly


def _resample(x, src_sr, dst_sr):
    if src_sr == dst_sr:
        return x.astype(np.float32)
    g = gcd(int(src_sr), int(dst_sr))
    return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def read_bin_window(path, start_s, dur_s, cap_sr, in_sr):
    """Read [start_s, start_s+dur_s) of a float32 capture and decimate to in_sr.

    Decimation lowpasses at in_sr/2 (= 16 kHz for in_sr=32 kHz), so this is the
    0-16 kHz band-limit and the rate conversion in one step.
    """
    offset = max(0, int(round(start_s * cap_sr)))
    count  = int(round(dur_s * cap_sr))
    x = np.fromfile(path, dtype=np.float32, count=count, offset=offset * 4)  # 4 B/f32
    if x.size == 0:
        return np.zeros(int(round(dur_s * in_sr)), dtype=np.float32)
    return _resample(x, cap_sr, in_sr)


def read_bin_window_dualband(path, start_s, dur_s, cap_sr, in_sr):
    """Read the window ONCE at full cap_sr and return BOTH bands (M27 dual-band):
        full  — the raw 200 kHz window (0-100 kHz, high stream input)
        low   — the same window decimated to in_sr (0-16 kHz, low stream input)
    One disk read serves both streams (no info discarded)."""
    offset = max(0, int(round(start_s * cap_sr)))
    count  = int(round(dur_s * cap_sr))
    full = np.fromfile(path, dtype=np.float32, count=count, offset=offset * 4)
    if full.size == 0:
        return (np.zeros(int(round(dur_s * cap_sr)), np.float32),
                np.zeros(int(round(dur_s * in_sr)), np.float32))
    low = _resample(full, cap_sr, in_sr)
    return full.astype(np.float32), low


def read_wav_window(path, start_s, dur_s, dst_sr):
    """Read a window of a PCM WAV → mono float32 at dst_sr."""
    with wave.open(path, 'rb') as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        i0 = max(0, int(round(start_s * sr)))
        n  = int(round(dur_s * sr))
        w.setpos(min(i0, w.getnframes()))
        raw = w.readframes(n)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    x /= float(np.iinfo(dt).max)
    return _resample(x, sr, dst_sr)


def read_lag_ms(path):
    """Powerline-lags-audio, in ms (per-chunk .lag sidecar). Missing → 0."""
    if not os.path.exists(path):
        return 0.0
    try:
        return float(open(path).read().strip())
    except Exception:
        return 0.0


@lru_cache(maxsize=512)
def wav_duration_s(path):
    with wave.open(path, 'rb') as w:
        return w.getnframes() / w.getframerate()


@lru_cache(maxsize=512)
def bin_duration_s(path, cap_sr):
    return (os.path.getsize(path) // 4) / cap_sr
