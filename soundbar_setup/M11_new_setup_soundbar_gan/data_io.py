#!/usr/bin/env python3
"""data_io.py — windowed readers for .bin / .wav / .lag (no ffmpeg). Self-contained."""

import os
import wave
from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample(x, src_sr, dst_sr):
    if src_sr == dst_sr:
        return x.astype(np.float32)
    g = gcd(int(src_sr), int(dst_sr))
    return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def read_bin_window(path, start_s, dur_s, sr):
    off = int(round(start_s * sr))
    cnt = int(round(dur_s * sr))
    return np.fromfile(path, dtype=np.float32, count=cnt, offset=off * 4)


def read_wav_window(path, start_s, dur_s, dst_sr):
    with wave.open(path, 'rb') as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        i0 = int(round(start_s * sr)); n = int(round(dur_s * sr))
        w.setpos(min(i0, w.getnframes()))
        raw = w.readframes(n)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    x /= float(np.iinfo(dt).max)
    return resample(x, sr, dst_sr)


def read_lag_ms(path):
    if not os.path.exists(path):
        return 0.0
    return float(open(path).read().strip())
