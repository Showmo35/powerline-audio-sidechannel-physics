#!/usr/bin/env python3
"""
dataset.py — one item = one word occurrence -> (wideband 200 kHz STFT, word label).

Window: fixed win_s centered on the word midpoint (word + context), lag-aligned,
RMS-normalized. Feature: log |STFT| of the 200 kHz signal (n_wbins x n_frames).
Vocabulary = top-K frequent words; occurrences capped for balance; chunk split.
Optionally returns the 16 kHz audio window too (for the wav2vec2 audio control).
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as F
import wave

from config import CFG, INDEX


def read_lag_s(chunk, cfg=CFG):
    p = cfg.lag_path(chunk)
    return (float(open(p).read().strip()) / 1000.0) if os.path.exists(p) else 0.0


def read_wav_window(path, start_s, dur_s, sr_out=16000):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        w.setpos(min(max(0, int(start_s * sr)), w.getnframes()))
        x = np.frombuffer(w.readframes(int(dur_s * sr)), np.int16).astype(np.float32) / 32768.0
    if sr != sr_out:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, sr_out); x = resample_poly(x, sr_out // g, sr // g).astype(np.float32)
    n = int(dur_s * sr_out)
    return np.pad(x, (0, max(0, n - len(x))))[:n]


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


class WordWindows(torch.utils.data.Dataset):
    def __init__(self, items, cfg=CFG, want_audio=False):
        self.items, self.cfg, self.want_audio = items, cfg, want_audio
        self.win = torch.hann_window(cfg.w_win)
        self._plc = {}

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
        t0 = mid - cfg.win_s / 2 + read_lag_s(ch)
        a = int(t0 * cfg.cap_sr); L = int(cfg.win_s * cfg.cap_sr)
        pl = self._pl(ch)
        x = np.array(pl[max(0, a):a + L], dtype=np.float32)
        if len(x) < L:
            x = np.pad(x, (0, L - len(x)))
        x = torch.from_numpy(x / (x.std() + 1e-8))
        spec = torch.stft(x, n_fft=cfg.w_nfft, hop_length=cfg.w_hop, win_length=cfg.w_win,
                          window=self.win, return_complex=True, center=True)
        wide = torch.log(spec.abs() ** 2 + cfg.log_eps)          # [n_wbins, ~n_frames]
        if wide.shape[-1] != cfg.n_frames:
            wide = F.interpolate(wide[None, None], size=(cfg.n_wbins, cfg.n_frames),
                                 mode='bilinear', align_corners=False)[0, 0]
        out = {'wide': wide, 'y': y}
        if self.want_audio:
            aw = read_wav_window(cfg.wav_path(ch), mid - cfg.win_s / 2, cfg.win_s, cfg.ref_sr)
            out['audio'] = torch.from_numpy(aw)
        return out


def collate(b):
    o = {'wide': torch.stack([x['wide'] for x in b]),
         'y': torch.tensor([x['y'] for x in b], dtype=torch.long)}
    if 'audio' in b[0]:
        o['audio'] = torch.stack([x['audio'] for x in b])
    return o


class CachedWords(torch.utils.data.Dataset):
    """Reads precomputed wideband features from the memmap cache (fast).
    flatten_freq=True -> ENVELOPE-ONLY control: replace each frame's spectrum with
    its (log) mean power broadcast flat across freq. Keeps the loudness contour,
    removes ALL spectral/harmonic content. Same shape, same model."""
    def __init__(self, split, cfg=CFG, flatten_freq=False):
        C = os.path.join(os.path.dirname(INDEX), 'outputs', 'cache')
        self.X = np.load(f'{C}/{split}_wide.npy', mmap_mode='r')
        self.y = np.load(f'{C}/{split}_y.npy')
        self.flatten_freq = flatten_freq

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        w = torch.from_numpy(np.asarray(self.X[i], dtype=np.float32))
        if self.flatten_freq:
            log_mean = torch.logsumexp(w, dim=0, keepdim=True) - np.log(w.shape[0])  # [1,T] log mean power
            w = log_mean.expand_as(w).contiguous()
        return {'wide': w, 'y': int(self.y[i])}


def cache_words():
    import json as _j
    return _j.load(open(os.path.join(os.path.dirname(INDEX), 'outputs', 'cache', 'words.json')))


@torch.no_grad()
def stats(ds, n=200, seed=0):
    rng = np.random.RandomState(seed); ids = rng.choice(len(ds), min(n, len(ds)), replace=False)
    s = s2 = c = 0.0
    for i in ids:
        x = ds[int(i)]['wide']; s += x.sum().item(); s2 += (x * x).sum().item(); c += x.numel()
    m = s / c
    return float(m), float((s2 / c - m ** 2) ** 0.5)
