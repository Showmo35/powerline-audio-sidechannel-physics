#!/usr/bin/env python3
"""
dataset_words.py — one item = one word occurrence, open vocabulary.

Returns the RAW 200 kHz window (the comb needs the full rate — decimating to 32 kHz
would delete every harmonic above 16 kHz). BOTH arms consume this same tensor:
  ARM comb : train loop demodulates it into the harmonic gram on GPU
  ARM raw  : train loop decimates it to 32 kHz on GPU (M22's input)
so the two arms see byte-identical data and the A/B is clean.

  raw       : (raw_len,) 200 kHz lag-aligned window starting at (word onset - ctx)
  mel_word  : REAL log-mel of [s-ctx, e+ctx] -> (n_mels, T)  — the retrieval target
  mel_full  : REAL log-mel of the whole win_s window          — flow-matching target
  frs       : # frames the word occupies at the START of the generation
  label     : word id over the FULL vocab
  f0        : precise mains fundamental for this chunk (comb demod grid)
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import data_io_raw as io
from config import CFG, RC, build_vocab
from comb import estimate_mains

_F0 = {}


def chunk_f0(ch):
    """Precise per-chunk mains (mHz accurate, fit over harmonic peaks). NOT the
    bin-snapped Welch argmax that M26 used (that returns 61.035 vs a true ~60.00)."""
    if ch not in _F0:
        probe = io.read_bin_window(CFG.bin_path(ch), 60.0, 30.0, CFG.cap_sr)
        _F0[ch] = estimate_mains(probe, CFG.cap_sr)
    return _F0[ch]


def build_items(split):
    words, occ = build_vocab()
    wid = {w: i for i, w in enumerate(words)}
    rng = np.random.RandomState(0)
    items = []
    for w in words:
        rows = [r for r in occ[w]
                if CFG.is_test(r[0]) == (split == 'test')
                and RC.min_dur_s <= (r[2] - r[1]) <= RC.max_dur_s]
        if RC.max_per_word and split == 'train' and len(rows) > RC.max_per_word:
            rng.shuffle(rows); rows = rows[:RC.max_per_word]
        for ch, s, e in rows:
            items.append((ch, float(s), float(e), wid[w]))
    rng.shuffle(items)
    return items, words


def cap_per_label(items, k, seed=0):
    if k is None or k <= 0:
        return items
    rng = np.random.RandomState(seed)
    by = {}
    for it in items:
        by.setdefault(it[3], []).append(it)
    out = []
    for _, rows in by.items():
        if len(rows) > k:
            rows = [rows[i] for i in rng.choice(len(rows), k, replace=False)]
        out += rows
    return out


class WordSet(torch.utils.data.Dataset):
    def __init__(self, items, real_only=False):
        self.items = items
        self.real_only = real_only          # gallery path: skip the expensive .bin read
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=CFG.ref_sr, n_fft=CFG.n_fft, hop_length=CFG.hop,
            win_length=CFG.win_length, n_mels=CFG.n_mels,
            f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)

    def __len__(self):
        return len(self.items)

    def _logmel(self, wav):
        return torch.log(self.melspec(torch.from_numpy(np.asarray(wav, np.float32)))
                         + CFG.log_eps)

    def _word_mel(self, ch, s, e):
        ctx = RC.ctx_s
        wdur = (e - s) + 2 * ctx
        ww = io.read_wav_window(CFG.wav_path(ch), s - ctx, wdur, CFG.ref_sr)
        m = self._logmel(ww)
        return F.interpolate(m[None, None], size=(CFG.n_mels, RC.T),
                             mode='bilinear', align_corners=False)[0, 0], wdur

    def __getitem__(self, i):
        ch, s, e, y = self.items[i]
        if self.real_only:
            mw, _ = self._word_mel(ch, s, e)
            return {'mel_word': mw, 'label': y}
        ctx = RC.ctx_s
        lag = io.read_lag_ms(CFG.lag_path(ch)) / 1000.0

        raw = io.read_bin_window(CFG.bin_path(ch), (s - ctx) - CFG.pad_s + lag,
                                 CFG.tot_s, CFG.cap_sr)
        raw = np.asarray(raw, np.float32)
        raw = raw / (np.std(raw) + 1e-8)
        x = np.zeros(CFG.raw_len, np.float32)
        x[:min(len(raw), CFG.raw_len)] = raw[:CFG.raw_len]

        mw, wdur = self._word_mel(ch, s, e)

        fw = io.read_wav_window(CFG.wav_path(ch), s - ctx, CFG.win_s, CFG.ref_sr)
        full = np.zeros(int(round(CFG.win_s * CFG.ref_sr)), np.float32)
        full[:min(len(fw), len(full))] = fw[:len(full)]
        mf = self._logmel(full)
        if mf.shape[-1] != CFG.n_frames:
            mf = F.interpolate(mf[None, None], size=(CFG.n_mels, CFG.n_frames),
                               mode='bilinear', align_corners=False)[0, 0]

        return {'raw': torch.from_numpy(x), 'mel_word': mw, 'mel_full': mf,
                'frs': max(6, int(round(wdur * CFG.fps))), 'label': y,
                'f0': float(chunk_f0(ch))}


def collate(b):
    return {'raw': torch.stack([x['raw'] for x in b]),
            'mel_word': torch.stack([x['mel_word'] for x in b]),
            'mel_full': torch.stack([x['mel_full'] for x in b]),
            'frs': torch.tensor([x['frs'] for x in b]),
            'label': torch.tensor([x['label'] for x in b]),
            'f0': torch.tensor([x['f0'] for x in b])}


@torch.no_grad()
def estimate_mel_stats(ds, n=256, seed=0):
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    s = s2 = cnt = 0.0
    for i in ids:
        m = ds[int(i)]['mel_full']
        s += m.sum().item(); s2 += (m * m).sum().item(); cnt += m.numel()
    mean = s / cnt
    return float(mean), float((s2 / cnt - mean ** 2) ** 0.5)
