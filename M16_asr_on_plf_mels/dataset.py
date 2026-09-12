#!/usr/bin/env python3
"""
dataset.py — MelText: one item = one utterance's (mel, char labels).

Two interchangeable inputs (chosen by --input in train_asr.py):
  gen  : the PLF-generated log-mel from gen_mels/{utt_id}.npy (M14 samples)
  real : the reference log-mel computed on the fly from the clean 16 kHz wav
Both share the manifest, the utterance filter, and the strict chunk split
(every test_every-th chunk held out) so the twin models are exactly comparable.
"""

import json
import os
import wave

import numpy as np
import torch
import torchaudio

from config import CFG, MANIFEST, GEN_DIR
import text as T


def read_wav_window(path, start_s, dur_s, dst_sr):
    with wave.open(path, 'rb') as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        w.setpos(min(max(0, int(round(start_s * sr))), w.getnframes()))
        raw = w.readframes(int(round(dur_s * sr)))
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    x /= float(np.iinfo(dt).max)
    if sr != dst_sr:
        import scipy.signal
        from math import gcd
        g = gcd(sr, dst_sr)
        x = scipy.signal.resample_poly(x, dst_sr // g, sr // g).astype(np.float32)
    return x


def load_rows(cfg=CFG):
    rows = json.load(open(MANIFEST))
    return [r for r in rows
            if cfg.min_dur_s <= r['dur_s'] <= cfg.max_dur_s
            and len(T.encode(r['text'])) > 0]


def split_rows(rows, cfg=CFG):
    test = cfg.test_chunks()
    return ([r for r in rows if r['chunk'] not in test],
            [r for r in rows if r['chunk'] in test])


class MelText(torch.utils.data.Dataset):
    def __init__(self, rows, source, cfg=CFG, gen_dir=GEN_DIR):
        assert source in ('gen', 'real')
        self.rows, self.source, self.cfg, self.gen_dir = rows, source, cfg, gen_dir
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.ref_sr, n_fft=cfg.n_fft, hop_length=cfg.hop,
            win_length=cfg.win_length, n_mels=cfg.n_mels,
            f_min=cfg.fmin, f_max=cfg.fmax, power=2.0)

    def __len__(self):
        return len(self.rows)

    def _mel(self, row):
        cfg = self.cfg
        dur = min(row['dur_s'], cfg.max_dur_s)
        vframes = int(round(dur * cfg.fps))
        if self.source == 'gen':
            m = np.load(os.path.join(self.gen_dir, row['utt_id'] + '.npy'))
            return torch.from_numpy(m.astype(np.float32))
        w = read_wav_window(cfg.wav_path(row['chunk']), row['start_s'], dur, cfg.ref_sr)
        m = torch.log(self.melspec(torch.from_numpy(w)) + cfg.log_eps)
        return m[:, :vframes]

    def __getitem__(self, i):
        row = self.rows[i]
        return {'mel': self._mel(row),
                'labels': T.encode(row['text']),
                'text': T.normalize(row['text'])}


def collate(batch):
    lens = torch.tensor([b['mel'].shape[-1] for b in batch], dtype=torch.long)
    tmax = int(lens.max())
    mel = torch.zeros(len(batch), batch[0]['mel'].shape[0], tmax)
    for i, b in enumerate(batch):
        mel[i, :, :b['mel'].shape[-1]] = b['mel']
    lab_lens = torch.tensor([len(b['labels']) for b in batch], dtype=torch.long)
    labels = torch.zeros(len(batch), int(lab_lens.max()), dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :len(b['labels'])] = torch.tensor(b['labels'], dtype=torch.long)
    return {'mel': mel, 'mel_lens': lens, 'labels': labels,
            'label_lengths': lab_lens, 'texts': [b['text'] for b in batch]}


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
