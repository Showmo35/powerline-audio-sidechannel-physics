#!/usr/bin/env python3
"""
dataset.py — PLFWindows: one item = one window of one capture.

For every one of the 202 chunks we sample `windows_per_chunk` random start times.
Each item is:
  raw : the lag-aligned 0-16 kHz powerline waveform  (in_len samples @ 32 kHz)
  mel : the log-mel of the matching 16 kHz reference  ([n_mels, n_frames])
The powerline window is read at  start + lag(chunk)  (the .lag sidecar we measured
from the RMS envelope); the reference mel is read at the un-shifted start, so the
two are time-aligned frame-for-frame.

Train/test split is BY CHUNK (every `test_every`-th chunk held out) so the model
is evaluated on captures it never saw.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from config import CFG
import data_io as io


def build_index(chunks, cfg=CFG, seed=0):
    """List of (chunk, start_s) windows, random starts within each chunk."""
    rng = np.random.RandomState(seed)
    idx = []
    for ch in chunks:
        binp, wavp, lagp = cfg.bin_path(ch), cfg.wav_path(ch), cfg.lag_path(ch)
        try:
            wav_d = io.wav_duration_s(wavp)
            bin_d = io.bin_duration_s(binp, cfg.cap_sr)
        except Exception:
            continue
        lag_s = io.read_lag_ms(lagp) / 1000.0
        avail = min(wav_d, bin_d - lag_s) - cfg.win_s
        if avail <= 0:
            continue
        starts = rng.uniform(0.0, avail, size=cfg.windows_per_chunk)
        idx += [(ch, float(s)) for s in starts]
    rng.shuffle(idx)
    return idx


def split_chunks(cfg=CFG):
    test = cfg.test_chunks()
    allc = cfg.all_chunks()
    tr = [c for c in allc if c not in test]
    te = [c for c in allc if c in test]
    return tr, te


class PLFWindows(torch.utils.data.Dataset):
    def __init__(self, index, cfg=CFG):
        self.index = index
        self.cfg = cfg
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.ref_sr, n_fft=cfg.n_fft, hop_length=cfg.hop,
            win_length=cfg.win_length, n_mels=cfg.n_mels,
            f_min=cfg.fmin, f_max=cfg.fmax, power=2.0)

    def __len__(self):
        return len(self.index)

    def _raw(self, ch, start_s):
        cfg = self.cfg
        lag_s = io.read_lag_ms(cfg.lag_path(ch)) / 1000.0
        x = io.read_bin_window(cfg.bin_path(ch), start_s + lag_s, cfg.win_s,
                               cfg.cap_sr, cfg.in_sr)
        x = np.asarray(x, dtype=np.float32)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-8)          # RMS normalize input
        out = np.zeros(cfg.in_len, dtype=np.float32)
        out[:min(len(x), cfg.in_len)] = x[:cfg.in_len]
        return torch.from_numpy(out)

    def _mel(self, ch, start_s):
        cfg = self.cfg
        w = io.read_wav_window(cfg.wav_path(ch), start_s, cfg.win_s, cfg.ref_sr)
        w = np.asarray(w, dtype=np.float32)
        full = np.zeros(int(round(cfg.win_s * cfg.ref_sr)), dtype=np.float32)
        full[:min(len(w), len(full))] = w[:len(full)]
        m = self.melspec(torch.from_numpy(full))            # (n_mels, t)
        m = torch.log(m + cfg.log_eps)
        # fix the time axis to exactly n_frames
        if m.shape[-1] != cfg.n_frames:
            m = F.interpolate(m.unsqueeze(0).unsqueeze(0),
                              size=(cfg.n_mels, cfg.n_frames),
                              mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        return m                                            # (n_mels, n_frames)

    def __getitem__(self, i):
        ch, start_s = self.index[i]
        return {'raw': self._raw(ch, start_s),
                'mel': self._mel(ch, start_s),
                'chunk': ch, 'start_s': start_s}


def collate(batch):
    return {
        'raw': torch.stack([b['raw'] for b in batch]),      # (B, in_len)
        'mel': torch.stack([b['mel'] for b in batch]),      # (B, n_mels, T)
        'chunk': [b['chunk'] for b in batch],
        'start_s': torch.tensor([b['start_s'] for b in batch]),
    }


@torch.no_grad()
def estimate_mel_stats(ds, n=512, seed=0):
    """Global log-mel mean/std for standardising the flow target."""
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    s = s2 = cnt = 0.0
    for i in ids:
        m = ds[int(i)]['mel']
        s += m.sum().item(); s2 += (m * m).sum().item(); cnt += m.numel()
    mean = s / cnt
    std = (s2 / cnt - mean ** 2) ** 0.5
    return float(mean), float(std)
