#!/usr/bin/env python3
"""
dataset_words.py — one item = one WORD occurrence over the FULL open vocabulary.

Each item yields the M22 generator input plus the retrieval targets:
  raw       : 4 s lag-aligned powerline window @ in_sr, starting at (word_onset - ctx)
              (RMS-normalized, padded to in_len) — the generator input.
  mel_word  : REAL log-mel of [s-ctx, e+ctx], time-normalized to (n_mels, T).
              The retrieval TARGET (same space the kNN uses).
  mel_full  : REAL log-mel of the whole 4 s window (n_mels, n_frames) — target for
              flow matching (learn to generate mels at all).
  frs       : # frames the word occupies at the START of the 4 s generation, so the
              generated word mel = gen_full[:, :frs].
  label     : word id in the FULL vocab (for retrieval scoring / same-word positives).

Open vocabulary: build_items enumerates EVERY word occurrence (all ~25k word types),
only dropping degenerate durations.  Split is BY CHUNK (every 12th held out).
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import data_io as io
from config import CFG, RC, build_vocab


def build_items(split):
    """List of (chunk, s, e, label) over ALL words for `split` ('train'|'test')."""
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
    """Subsample <=k occurrences per label (for a bounded eval gallery). k<=0 = all."""
    if k is None or k <= 0:
        return items
    rng = np.random.RandomState(seed)
    by = {}
    for it in items:
        by.setdefault(it[3], []).append(it)
    out = []
    for y, rows in by.items():
        if len(rows) > k:
            idx = rng.choice(len(rows), size=k, replace=False)
            rows = [rows[i] for i in idx]
        out += rows
    return out


class WordSet(torch.utils.data.Dataset):
    def __init__(self, items, real_only=False):
        # real_only: return just the real word mel + label (skips the expensive .bin
        # read + full-window mel) — used to build the retrieval gallery cheaply.
        self.items = items
        self.real_only = real_only
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=CFG.ref_sr, n_fft=CFG.n_fft, hop_length=CFG.hop,
            win_length=CFG.win_length, n_mels=CFG.n_mels,
            f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)

    def __len__(self):
        return len(self.items)

    def _logmel(self, wav):
        m = self.melspec(torch.from_numpy(np.asarray(wav, np.float32)))
        return torch.log(m + CFG.log_eps)                       # (n_mels, frames)

    def _word_mel(self, ch, s, e):
        ctx = RC.ctx_s
        wdur = (e - s) + 2 * ctx
        ww = io.read_wav_window(CFG.wav_path(ch), s - ctx, wdur, CFG.ref_sr)
        mw = self._logmel(ww)
        return F.interpolate(mw[None, None], size=(CFG.n_mels, RC.T),
                             mode='bilinear', align_corners=False)[0, 0], wdur

    def __getitem__(self, i):
        ch, s, e, y = self.items[i]
        if self.real_only:                                      # gallery path
            mw, _ = self._word_mel(ch, s, e)
            return {'mel_word': mw, 'label': y}
        ctx = RC.ctx_s
        lag = io.read_lag_ms(CFG.lag_path(ch)) / 1000.0

        # ── generator input: 4 s powerline window at (word onset - ctx) + lag ──
        raw = io.read_bin_window(CFG.bin_path(ch), (s - ctx) + lag, CFG.win_s,
                                 CFG.cap_sr, CFG.in_sr)
        raw = np.asarray(raw, np.float32)
        raw = raw / (np.sqrt(np.mean(raw ** 2)) + 1e-8)         # RMS normalize
        x = np.zeros(CFG.in_len, np.float32)
        x[:min(len(raw), CFG.in_len)] = raw[:CFG.in_len]

        # ── real word mel over [s-ctx, e+ctx] -> (n_mels, T) (retrieval target) ─
        mw, wdur = self._word_mel(ch, s, e)

        # ── real 4 s window mel -> (n_mels, n_frames) (flow-matching target) ────
        fw = io.read_wav_window(CFG.wav_path(ch), s - ctx, CFG.win_s, CFG.ref_sr)
        full = np.zeros(int(round(CFG.win_s * CFG.ref_sr)), np.float32)
        full[:min(len(fw), len(full))] = fw[:len(full)]
        mf = self._logmel(full)
        if mf.shape[-1] != CFG.n_frames:
            mf = F.interpolate(mf[None, None], size=(CFG.n_mels, CFG.n_frames),
                               mode='bilinear', align_corners=False)[0, 0]

        frs = max(6, int(round(wdur * CFG.fps)))
        return {'raw': torch.from_numpy(x), 'mel_word': mw, 'mel_full': mf,
                'frs': frs, 'label': y}


def collate(b):
    return {
        'raw':      torch.stack([x['raw'] for x in b]),         # (B, in_len)
        'mel_word': torch.stack([x['mel_word'] for x in b]),    # (B, n_mels, T)
        'mel_full': torch.stack([x['mel_full'] for x in b]),    # (B, n_mels, n_frames)
        'frs':      torch.tensor([x['frs'] for x in b]),        # (B,)
        'label':    torch.tensor([x['label'] for x in b]),      # (B,)
    }


@torch.no_grad()
def estimate_mel_stats(ds, n=512, seed=0):
    """Global log-mel mean/std (over full-window mels) for standardizing the flow
    target — estimated FRESH from the data (no M14 checkpoint)."""
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    s = s2 = cnt = 0.0
    for i in ids:
        m = ds[int(i)]['mel_full']
        s += m.sum().item(); s2 += (m * m).sum().item(); cnt += m.numel()
    mean = s / cnt
    std = (s2 / cnt - mean ** 2) ** 0.5
    return float(mean), float(std)
