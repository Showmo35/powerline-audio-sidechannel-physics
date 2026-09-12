#!/usr/bin/env python3
"""
dataset.py — RawUttDataset: one item = one LibriSpeech utterance read DIRECTLY as a
raw 200 kSps powerline window (no precomputed features).

The window is located via the shared manifest (start_s/end_s on the audio timeline)
shifted by the per-row lag (powerline lags audio by lag_ms).  Windows are cropped /
zero-padded to cfg.max_dur_s so every item has the same length L → the learned
front-end emits a fixed T and batching needs no time masking.

For enhancement archs (m4, m7) the item also carries a clean log-mel target built
from the 16 kHz reference wav window and interpolated to the front-end frame count.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from config import CFG
import data_io as io
import text as T


def load_manifest(path=None):
    path = path or __import__('config').SHARED_MANIFEST
    return [json.loads(l) for l in open(path) if l.strip()]


def chunk_set(spec):
    lo, hi = (int(x) for x in spec.split('-'))
    return {f'chunk_{n:03d}' for n in range(lo, hi + 1)}


def split_by_chunk(rows, test_chunks):
    tr = [r for r in rows if r['chunk'] not in test_chunks]
    te = [r for r in rows if r['chunk'] in test_chunks]
    return tr, te


def frontend_out_len(L, cfg=CFG):
    """Analytic output length of RawFrontEnd for input length L (odd kernels)."""
    for s in cfg.fe_strides:
        L = (L - 1) // s + 1
    return L


class RawUttDataset(torch.utils.data.Dataset):
    def __init__(self, rows, cfg=CFG, want_clean=False):
        self.rows = rows
        self.cfg = cfg
        self.want_clean = want_clean
        self.L = int(round(cfg.max_dur_s * cfg.cap_sr))
        self.T_feat = frontend_out_len(self.L, cfg)
        if want_clean:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.ref_sr, n_fft=cfg.tgt_n_fft, hop_length=cfg.tgt_hop,
                n_mels=cfg.tgt_n_mels, power=2.0)

    def __len__(self):
        return len(self.rows)

    def _raw(self, row):
        lag_s = row.get('lag_ms', 0.0) / 1000.0
        dur = min(row['dur_s'], self.cfg.max_dur_s)
        x = io.read_bin_window(self.cfg.bin_path(row['chunk']),
                               row['start_s'] + lag_s, dur, self.cfg.cap_sr)
        x = np.asarray(x, dtype=np.float32)
        # per-utterance RMS normalize (front-end also instance-norms, but keep inputs sane)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-8)
        out = np.zeros(self.L, dtype=np.float32)
        out[:min(len(x), self.L)] = x[:self.L]
        return out

    def _clean_mel(self, row):
        dur = min(row['dur_s'], self.cfg.max_dur_s)
        w = io.read_wav_window(self.cfg.wav_path(row['chunk']),
                               row['start_s'], dur, self.cfg.ref_sr)
        w = np.asarray(w, dtype=np.float32)
        full = np.zeros(int(round(self.cfg.max_dur_s * self.cfg.ref_sr)), dtype=np.float32)
        full[:min(len(w), len(full))] = w[:len(full)]
        m = self.mel(torch.from_numpy(full))                 # (n_mels, t)
        m = torch.log(m + 1e-6).unsqueeze(0).unsqueeze(0)    # (1,1,F,t)
        m = F.interpolate(m, size=(self.cfg.tgt_n_mels, self.T_feat),
                          mode='bilinear', align_corners=False)
        return m.squeeze(0).squeeze(0)                       # (F, T_feat)

    def __getitem__(self, i):
        row = self.rows[i]
        item = {'raw': torch.from_numpy(self._raw(row)),
                'labels': T.encode(row['text']),
                'text': T.normalize(row['text'])}
        if self.want_clean:
            item['clean_mel'] = self._clean_mel(row)
        return item


def make_collate(want_clean=False):
    def collate(batch):
        raw = torch.stack([b['raw'] for b in batch])             # (B, L)
        lab_lens = torch.tensor([max(1, len(b['labels'])) for b in batch], dtype=torch.long)
        smax = int(lab_lens.max())
        labels = torch.zeros(len(batch), smax, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b['labels'] or [T.BLANK_IDX]
            labels[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        out = {'raw': raw, 'labels': labels, 'label_lengths': lab_lens,
               'texts': [b['text'] for b in batch]}
        if want_clean:
            out['clean_mel'] = torch.stack([b['clean_mel'] for b in batch])  # (B,F,T)
        return out
    return collate
