#!/usr/bin/env python3
"""
dataset.py — PLFWUtterances: one item = one LibriSpeech utterance.

From full_manifest.json (exact utterance timing over all 202 chunks) each item is
  raw : the lag-aligned FULL 200 kHz powerline window (max_dur, padded)  [a_len]
  mel : the 16 kHz reference log-mel [n_mels, n_frames]
  labels : char ids of the utterance text (for the CTC word-loss)
  vframes : number of valid (non-pad) mel frames  (for masking + CTC lengths)
Split is by chunk (every test_every-th held out).
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from config import CFG, MANIFEST
import data_io as io
import text as T


def load_manifest(path=MANIFEST):
    return json.load(open(path))


def split_rows(rows, cfg=CFG):
    test = cfg.test_chunks()
    keep = [r for r in rows
            if cfg.min_dur_s <= r['dur_s'] <= cfg.max_dur_s and len(T.encode(r['text'])) > 0]
    tr = [r for r in keep if r['chunk'] not in test]
    te = [r for r in keep if r['chunk'] in test]
    return tr, te


class PLFWUtterances(torch.utils.data.Dataset):
    def __init__(self, rows, cfg=CFG):
        self.rows = rows
        self.cfg = cfg
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.ref_sr, n_fft=cfg.n_fft, hop_length=cfg.hop,
            win_length=cfg.win_length, n_mels=cfg.n_mels,
            f_min=cfg.fmin, f_max=cfg.fmax, power=2.0)

    def __len__(self):
        return len(self.rows)

    def _raw(self, row):
        cfg = self.cfg
        lag_s = io.read_lag_ms(cfg.lag_path(row['chunk'])) / 1000.0
        dur = min(row['dur_s'], cfg.max_dur_s)
        x = io.read_bin_window(cfg.bin_path(row['chunk']), row['start_s'] + lag_s,
                               dur, cfg.cap_sr).astype(np.float32)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-8)
        out = np.zeros(cfg.a_len, dtype=np.float32)
        out[:min(len(x), cfg.a_len)] = x[:cfg.a_len]
        return torch.from_numpy(out)

    def _mel(self, row):
        cfg = self.cfg
        dur = min(row['dur_s'], cfg.max_dur_s)
        w = io.read_wav_window(cfg.wav_path(row['chunk']), row['start_s'], dur,
                               cfg.ref_sr).astype(np.float32)
        full = np.zeros(int(round(cfg.max_dur_s * cfg.ref_sr)), dtype=np.float32)
        full[:min(len(w), len(full))] = w[:len(full)]
        m = torch.log(self.melspec(torch.from_numpy(full)) + cfg.log_eps)  # (n_mels, t)
        if m.shape[-1] != cfg.n_frames:
            m = F.interpolate(m[None, None], size=(cfg.n_mels, cfg.n_frames),
                              mode='bilinear', align_corners=False)[0, 0]
        vframes = min(cfg.n_frames, int(round(dur * cfg.fps)))
        return m, vframes

    def __getitem__(self, i):
        row = self.rows[i]
        mel, vframes = self._mel(row)
        return {'raw': self._raw(row), 'mel': mel,
                'labels': T.encode(row['text']), 'vframes': vframes,
                'text': T.normalize(row['text'])}


def collate(batch):
    raw = torch.stack([b['raw'] for b in batch])
    mel = torch.stack([b['mel'] for b in batch])
    vframes = torch.tensor([b['vframes'] for b in batch], dtype=torch.long)
    lab_lens = torch.tensor([len(b['labels']) for b in batch], dtype=torch.long)
    smax = int(lab_lens.max())
    labels = torch.zeros(len(batch), smax, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :len(b['labels'])] = torch.tensor(b['labels'], dtype=torch.long)
    return {'raw': raw, 'mel': mel, 'vframes': vframes,
            'labels': labels, 'label_lengths': lab_lens,
            'texts': [b['text'] for b in batch]}


@torch.no_grad()
def estimate_mel_stats(ds, n=384, seed=0):
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    s = s2 = cnt = 0.0
    for i in ids:
        m = ds[int(i)]['mel']
        s += m.sum().item(); s2 += (m * m).sum().item(); cnt += m.numel()
    mean = s / cnt
    return float(mean), float((s2 / cnt - mean ** 2) ** 0.5)
