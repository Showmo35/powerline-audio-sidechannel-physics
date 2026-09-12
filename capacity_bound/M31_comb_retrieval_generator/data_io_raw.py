#!/usr/bin/env python3
"""data_io.py — windowed readers (full 200 kHz, no decimation; no ffmpeg)."""
import os, wave
from math import gcd
import numpy as np
from scipy.signal import resample_poly


def _resample(x, src, dst):
    if src == dst:
        return x.astype(np.float32)
    g = gcd(int(src), int(dst))
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def read_bin_window(path, start_s, dur_s, cap_sr):
    """Raw float32 capture window at the FULL cap_sr (no band-limit)."""
    off = max(0, int(round(start_s * cap_sr)))
    cnt = int(round(dur_s * cap_sr))
    x = np.fromfile(path, dtype=np.float32, count=cnt, offset=off * 4)
    return x


def read_wav_window(path, start_s, dur_s, dst_sr):
    with wave.open(path, 'rb') as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        i0 = max(0, int(round(start_s * sr)))
        n = int(round(dur_s * sr))
        w.setpos(min(i0, w.getnframes()))
        raw = w.readframes(n)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    x /= float(np.iinfo(dt).max)
    return _resample(x, sr, dst_sr)


def read_lag_ms(path):
    if not os.path.exists(path):
        return 0.0
    try:
        return float(open(path).read().strip())
    except Exception:
        return 0.0
