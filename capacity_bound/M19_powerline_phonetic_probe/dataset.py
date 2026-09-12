#!/usr/bin/env python3
"""
dataset.py — frame-aligned (powerline feature, true audio mel) windows.

Each item, on the 100 fps / 400-frame grid of a 4 s window:
  wide : wideband log-STFT of the lag-aligned 200 kHz powerline [n_wbins, 400]
  env  : per-frame log-RMS of the same window                    [1, 400]
  mel  : true audio log-mel from the 16 kHz reference            [n_mels, 400]
Split by chunk (every test_every-th held out). Powerline read at start+lag; mel
at start (frame-for-frame aligned, as in M14/M15).
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from config import CFG
import data_io as io


def build_index(chunks, cfg=CFG, seed=0):
    rng = np.random.RandomState(seed)
    idx = []
    import os, wave
    for ch in chunks:
        try:
            with wave.open(cfg.wav_path(ch), 'rb') as w:
                wdur = w.getnframes() / w.getframerate()
            bdur = (os.path.getsize(cfg.bin_path(ch)) // 4) / cfg.cap_sr
        except Exception:
            continue
        lag = io.read_lag_ms(cfg.lag_path(ch)) / 1000.0
        avail = min(wdur, bdur - lag) - cfg.win_s
        if avail <= 0:
            continue
        for s in rng.uniform(0, avail, size=cfg.windows_per_chunk):
            idx.append((ch, float(s)))
    rng.shuffle(idx)
    return idx


def split_chunks(cfg=CFG):
    test = cfg.test_chunks()
    allc = cfg.all_chunks()
    return [c for c in allc if c not in test], [c for c in allc if c in test]


class ProbeWindows(torch.utils.data.Dataset):
    def __init__(self, index, cfg=CFG):
        self.index, self.cfg = index, cfg
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.ref_sr, n_fft=cfg.n_fft, hop_length=cfg.hop,
            win_length=cfg.win_length, n_mels=cfg.n_mels,
            f_min=cfg.fmin, f_max=cfg.fmax, power=2.0)
        self.win = torch.hann_window(cfg.w_win)

    def __len__(self):
        return len(self.index)

    def _fix_T(self, x):                       # x: [C, t] → [C, n_frames]
        if x.shape[-1] == self.cfg.n_frames:
            return x
        return F.interpolate(x[None], size=(x.shape[0], self.cfg.n_frames),
                             mode='bilinear', align_corners=False)[0] if x.dim() == 2 else x

    def __getitem__(self, i):
        cfg = self.cfg
        ch, start = self.index[i]
        lag = io.read_lag_ms(cfg.lag_path(ch)) / 1000.0
        # ---- powerline 200 kHz window (lag-aligned) ----
        raw = io.read_bin_window(cfg.bin_path(ch), start + lag, cfg.win_s, cfg.cap_sr)
        raw = torch.from_numpy(np.asarray(raw, np.float32))
        need = int(round(cfg.win_s * cfg.cap_sr))
        if raw.numel() < need:
            raw = F.pad(raw, (0, need - raw.numel()))
        raw = raw[:need] / (raw.std() + 1e-8)
        spec = torch.stft(raw, n_fft=cfg.w_nfft, hop_length=cfg.w_hop, win_length=cfg.w_win,
                          window=self.win, return_complex=True, center=True)
        wide = torch.log(spec.abs() ** 2 + cfg.log_eps)            # [n_wbins, tw]
        if wide.shape[-1] != cfg.n_frames:
            wide = F.interpolate(wide[None, None], size=(cfg.n_wbins, cfg.n_frames),
                                 mode='bilinear', align_corners=False)[0, 0]
        env = torch.log(torch.exp(wide).sum(0, keepdim=True) + cfg.log_eps)  # [1, 400] loudness
        # ---- true audio mel ----
        w = io.read_wav_window(cfg.wav_path(ch), start, cfg.win_s, cfg.ref_sr)
        w = torch.from_numpy(np.asarray(w, np.float32))
        full = torch.zeros(int(round(cfg.win_s * cfg.ref_sr)))
        full[:min(len(w), len(full))] = w[:len(full)]
        mel = torch.log(self.melspec(full) + cfg.log_eps)
        if mel.shape[-1] != cfg.n_frames:
            mel = F.interpolate(mel[None, None], size=(cfg.n_mels, cfg.n_frames),
                                mode='bilinear', align_corners=False)[0, 0]
        return {'wide': wide, 'env': env, 'mel': mel}


def collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in ('wide', 'env', 'mel')}


@torch.no_grad()
def stats(ds, key, n=256, seed=0):
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    s = s2 = c = 0.0
    for i in ids:
        x = ds[int(i)][key]; s += x.sum().item(); s2 += (x * x).sum().item(); c += x.numel()
    m = s / c
    return float(m), float((s2 / c - m ** 2) ** 0.5)
