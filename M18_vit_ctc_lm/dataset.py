#!/usr/bin/env python3
"""
dataset.py — MelText: one item = one utterance's (mel, char labels), three sources.

  gen  : M14-generated log-mel from M16/gen_mels/{utt_id}.npy   (experiment)
  real : reference log-mel computed on the fly from the clean wav (ceiling)
  null : real mel with each time-frame's 80 bins randomly PERMUTED
         → per-frame energy (envelope) preserved exactly, spectral/phonetic
           shape destroyed. NOT invertible (fresh permutation per frame) so the
           model cannot learn it back. This is the LM-hallucination floor.

Same manifest, utterance filter, and chunk split as M15/M16, with utterance-level
disjointness enforced (drop train rows sharing an utt_id/text with any test chunk).
Null uses a per-access random shuffle on train (a different destruction each epoch)
and a deterministic per-utterance shuffle on test (stable eval).
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
    """Chunk split + utterance-level disjointness (drop test dupes from train)."""
    test = cfg.test_chunks()
    te = [r for r in rows if r['chunk'] in test]
    te_ids = {r['utt_id'] for r in te}
    te_txt = {T.normalize(r['text']) for r in te}
    tr = [r for r in rows if r['chunk'] not in test
          and r['utt_id'] not in te_ids and T.normalize(r['text']) not in te_txt]
    return tr, te


def _freq_shuffle(mel, rng):
    """Permute the mel bins independently within each time frame (envelope-preserving)."""
    m = mel.numpy().copy()                       # (n_mels, T)
    for t in range(m.shape[1]):
        m[:, t] = m[rng.permutation(m.shape[0]), t]
    return torch.from_numpy(m)


def _amp_only(mel, log_eps):
    """Amplitude-only: each frame's total energy broadcast FLAT across all bins.
    Keeps the loudness envelope, destroys ALL spectral distribution — the true
    audio analog of the powerline's 1-D amplitude channel."""
    p = mel.exp()                                # ≈ power + eps  (n_mels, T)
    mean_p = p.mean(0, keepdim=True)             # per-frame mean power (1, T)
    return torch.log(mean_p.expand_as(mel).clamp_min(log_eps))


class MelText(torch.utils.data.Dataset):
    def __init__(self, rows, source, cfg=CFG, gen_dir=GEN_DIR, train=True):
        assert source in ('gen', 'real', 'null', 'envamp')
        self.rows, self.source, self.cfg, self.gen_dir, self.train = rows, source, cfg, gen_dir, train
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.ref_sr, n_fft=cfg.n_fft, hop_length=cfg.hop,
            win_length=cfg.win_length, n_mels=cfg.n_mels,
            f_min=cfg.fmin, f_max=cfg.fmax, power=2.0)

    def __len__(self):
        return len(self.rows)

    def _real_mel(self, row):
        cfg = self.cfg
        dur = min(row['dur_s'], cfg.max_dur_s)
        w = read_wav_window(cfg.wav_path(row['chunk']), row['start_s'], dur, cfg.ref_sr)
        m = torch.log(self.melspec(torch.from_numpy(w)) + cfg.log_eps)
        return m[:, :int(round(dur * cfg.fps))]

    def _mel(self, row, i):
        if self.source == 'gen':
            m = np.load(os.path.join(self.gen_dir, row['utt_id'] + '.npy'))
            return torch.from_numpy(m.astype(np.float32))
        real = self._real_mel(row)
        if self.source == 'real':
            return real
        if self.source == 'envamp':          # amplitude-only (flat spectrum, envelope kept)
            return _amp_only(real, self.cfg.log_eps)
        # null: per-frame frequency shuffle (random on train, per-utt-deterministic on test)
        seed = np.random.randint(1 << 30) if self.train else (hash(row['utt_id']) & 0x7fffffff)
        return _freq_shuffle(real, np.random.RandomState(seed))

    def __getitem__(self, i):
        row = self.rows[i]
        return {'mel': self._mel(row, i),
                'labels': T.encode(row['text']),
                'text': T.normalize(row['text']),
                'utt_id': row['utt_id']}


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
            'label_lengths': lab_lens, 'texts': [b['text'] for b in batch],
            'utt_ids': [b['utt_id'] for b in batch]}


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
