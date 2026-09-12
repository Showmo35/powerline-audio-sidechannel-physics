#!/usr/bin/env python3
"""
dataset.py — one item = one word occurrence's RAW 1.5 s @ 200 kHz window + label.

Nothing compressed at read time: we cache the raw waveform (lag-aligned, RMS-
normed) and let the model derive patchify / STFT / envelope on-GPU. Split by chunk.
"""
import json, os
import numpy as np
import torch

from config import CFG, INDEX, OUT_DIR

CACHE = os.path.join(OUT_DIR, 'cache')


def read_lag_s(chunk, cfg=CFG):
    p = cfg.lag_path(chunk)
    return (float(open(p).read().strip()) / 1000.0) if os.path.exists(p) else 0.0


def build_vocab(cfg=CFG):
    occ = json.load(open(INDEX))
    words = sorted([w for w in occ if len(occ[w]) >= 20], key=lambda k: -len(occ[k]))[:cfg.vocab_k]
    return words, occ


def build_items(cfg=CFG, split='train'):
    words, occ = build_vocab(cfg)
    wid = {w: i for i, w in enumerate(words)}
    items = []
    rng = np.random.RandomState(0)
    for w in words:
        rows = [r for r in occ[w] if (cfg.is_test(r[0]) == (split == 'test'))]
        rng.shuffle(rows)
        if split == 'train':
            rows = rows[:cfg.max_per_word]
        for ch, s, e in rows:
            items.append((ch, s, e, wid[w]))
    return items, words


class RawWindows(torch.utils.data.Dataset):
    """Reads the raw 200 kHz window per item (for precompute)."""
    def __init__(self, items, cfg=CFG):
        self.items, self.cfg, self._plc = items, cfg, {}

    def __len__(self):
        return len(self.items)

    def _pl(self, ch):
        if ch not in self._plc:
            if len(self._plc) > 4:
                self._plc.clear()
            self._plc[ch] = np.memmap(self.cfg.bin_path(ch), dtype=np.float32, mode='r')
        return self._plc[ch]

    def __getitem__(self, i):
        cfg = self.cfg
        ch, s, e, y = self.items[i]
        mid = 0.5 * (s + e)
        a = int((mid - cfg.win_s / 2 + read_lag_s(ch)) * cfg.cap_sr)
        pl = self._pl(ch)
        x = np.array(pl[max(0, a):a + cfg.a_len], dtype=np.float32)
        if len(x) < cfg.a_len:
            x = np.pad(x, (0, cfg.a_len - len(x)))
        x = x / (x.std() + 1e-8)
        return {'raw': torch.from_numpy(x.astype(np.float32)), 'y': y}


def collate(b):
    return {'raw': torch.stack([x['raw'] for x in b]),
            'y': torch.tensor([x['y'] for x in b], dtype=torch.long)}


class CachedRaw(torch.utils.data.Dataset):
    def __init__(self, split):
        self.X = np.load(f'{CACHE}/{split}_raw.npy', mmap_mode='r')
        self.y = np.load(f'{CACHE}/{split}_y.npy')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {'raw': torch.from_numpy(np.asarray(self.X[i], dtype=np.float32)), 'y': int(self.y[i])}


def cache_words():
    return json.load(open(f'{CACHE}/words.json'))
