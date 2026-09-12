#!/usr/bin/env python3
"""
dataset.py — PLFW17Utterances: one item = one LibriSpeech utterance.

  raw : lag-aligned FULL 200 kHz powerline window (max_dur padded)  [a_len]
  spec: 513-bin linear log-STFT of the 16 kHz reference [n_bins, n_frames]
  labels / vframes / text : as in M15 (CTC supervision + frame masking)

Split is by chunk (every test_every-th held out) with LEAKAGE ENFORCEMENT: any
utt_id or normalized text that occurs in a test chunk is removed from the train
rows (the manifest contains ~145 duplicate utt_ids, so the chunk split alone does
not guarantee utterance-level disjointness). Test rows are never touched, so
test-side metrics stay comparable to M15.
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


def _keep(r, cfg):
    return cfg.min_dur_s <= r['dur_s'] <= cfg.max_dur_s and len(T.encode(r['text'])) > 0


def split_rows(rows, cfg=CFG):
    """Chunk split + utterance-level disjointness (drop test dupes from train)."""
    test = cfg.test_chunks()
    keep = [r for r in rows if _keep(r, cfg)]
    te = [r for r in keep if r['chunk'] in test]
    te_ids = {r['utt_id'] for r in te}
    te_txt = {T.normalize(r['text']) for r in te}
    tr_all = [r for r in keep if r['chunk'] not in test]
    tr = [r for r in tr_all
          if r['utt_id'] not in te_ids and T.normalize(r['text']) not in te_txt]
    return tr, te, {'train_dropped_as_test_dupes': len(tr_all) - len(tr)}


def audit_split(rows, cfg=CFG):
    """Hard checks for train/test separation. Raises on any violation."""
    test = cfg.test_chunks()
    train_chunks = {f'chunk_{n:03d}' for n in range(1, cfg.n_chunks + 1)} - test
    assert not (test & train_chunks), 'chunk sets overlap'
    tr, te, info = split_rows(rows, cfg)
    tr_ids = {r['utt_id'] for r in tr}
    te_ids = {r['utt_id'] for r in te}
    tr_txt = {T.normalize(r['text']) for r in tr}
    te_txt = {T.normalize(r['text']) for r in te}
    assert not (tr_ids & te_ids), f'utt_id leakage: {sorted(tr_ids & te_ids)[:5]}'
    assert not (tr_txt & te_txt), f'text leakage: {len(tr_txt & te_txt)} texts'
    # duplicate utt_ids anywhere in the manifest (informational)
    seen, dupes = set(), set()
    for r in rows:
        (dupes if r['utt_id'] in seen else seen).add(r['utt_id'])
    return {'train': len(tr), 'test': len(te), 'manifest_dupe_utt_ids': len(dupes), **info}


class PLFW17Utterances(torch.utils.data.Dataset):
    def __init__(self, rows, cfg=CFG):
        self.rows = rows
        self.cfg = cfg
        self.specfn = torchaudio.transforms.Spectrogram(
            n_fft=cfg.n_fft, hop_length=cfg.hop, win_length=cfg.win_length, power=2.0)

    def __len__(self):
        return len(self.rows)

    def _raw(self, row):
        cfg = self.cfg
        lag_s = io.read_lag_ms(cfg.lag_path(row['chunk'])) / 1000.0
        dur = min(row['dur_s'], cfg.max_dur_s)
        x = io.read_bin_window(cfg.bin_path(row['chunk']), row['start_s'] + lag_s,
                               dur, cfg.cap_sr).astype(np.float32)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-8)        # single invertible scalar
        out = np.zeros(cfg.a_len, dtype=np.float32)
        out[:min(len(x), cfg.a_len)] = x[:cfg.a_len]
        return torch.from_numpy(out)

    def _spec(self, row):
        cfg = self.cfg
        dur = min(row['dur_s'], cfg.max_dur_s)
        w = io.read_wav_window(cfg.wav_path(row['chunk']), row['start_s'], dur,
                               cfg.ref_sr).astype(np.float32)
        full = np.zeros(int(round(cfg.max_dur_s * cfg.ref_sr)), dtype=np.float32)
        full[:min(len(w), len(full))] = w[:len(full)]
        s = torch.log(self.specfn(torch.from_numpy(full)) + cfg.log_eps)  # (n_bins, t)
        if s.shape[-1] != cfg.n_frames:
            s = F.interpolate(s[None, None], size=(cfg.n_bins, cfg.n_frames),
                              mode='bilinear', align_corners=False)[0, 0]
        vframes = min(cfg.n_frames, int(round(dur * cfg.fps)))
        return s, vframes

    def __getitem__(self, i):
        row = self.rows[i]
        spec, vframes = self._spec(row)
        return {'raw': self._raw(row), 'spec': spec,
                'labels': T.encode(row['text']), 'vframes': vframes,
                'text': T.normalize(row['text'])}


def collate(batch):
    raw = torch.stack([b['raw'] for b in batch])
    spec = torch.stack([b['spec'] for b in batch])
    vframes = torch.tensor([b['vframes'] for b in batch], dtype=torch.long)
    lab_lens = torch.tensor([len(b['labels']) for b in batch], dtype=torch.long)
    labels = torch.zeros(len(batch), int(lab_lens.max()), dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :len(b['labels'])] = torch.tensor(b['labels'], dtype=torch.long)
    return {'raw': raw, 'spec': spec, 'vframes': vframes,
            'labels': labels, 'label_lengths': lab_lens,
            'texts': [b['text'] for b in batch]}


@torch.no_grad()
def estimate_spec_stats(ds, n=384, seed=0):
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    s = s2 = cnt = 0.0
    for i in ids:
        m = ds[int(i)]['spec']
        s += m.sum().item(); s2 += (m * m).sum().item(); cnt += m.numel()
    mean = s / cnt
    return float(mean), float((s2 / cnt - mean ** 2) ** 0.5)
