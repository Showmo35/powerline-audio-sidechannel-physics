#!/usr/bin/env python3
"""
data.py — torch Dataset over the Step-2 manifest for Whisper fine-tuning.

Each item: powerline AM-sideband mel (from the chunk's .npz shard) → Whisper
[80,3000] input_features, plus tokenized normalized transcript as labels.
"""

import json
import os

import numpy as np
import torch

from config import OUT_DIR, CFG
from whisper_features import mel_to_input_features
from asr import normalize_text

DATASET_DIR = os.path.join(OUT_DIR, 'dataset')
NATIVE_FPS = CFG.aud_sr / CFG.mel_hop      # 22050 / 275 ≈ 80.18


def load_manifest(path=None):
    path = path or os.path.join(DATASET_DIR, 'manifest.jsonl')
    return [json.loads(l) for l in open(path) if l.strip()]


def split_by_chunk(rows, test_chunks):
    """test_chunks: set of chunk names e.g. {'chunk_041',...}. Returns (train, test)."""
    tr = [r for r in rows if r['chunk'] not in test_chunks]
    te = [r for r in rows if r['chunk'] in test_chunks]
    return tr, te


class PowerlineMelDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, dataset_dir=DATASET_DIR, native_fps=NATIVE_FPS):
        self.rows = rows
        self.tok = tokenizer
        self.dir = dataset_dir
        self.fps = native_fps
        self._npz = {}                      # shard → lazy NpzFile (kept open)

    def __len__(self):
        return len(self.rows)

    def _mel(self, row):
        shard = row['shard']
        if shard not in self._npz:
            self._npz[shard] = np.load(os.path.join(self.dir, shard))
        return self._npz[shard][row['utt_id']]

    def __getitem__(self, i):
        row = self.rows[i]
        feat = mel_to_input_features(self._mel(row), self.fps)
        text = normalize_text(row['text'])
        labels = self.tok(text, add_special_tokens=True).input_ids
        return {'input_features': feat, 'labels': labels, 'text': text}


def make_collate(pad_to_multiple=None):
    def collate(batch):
        feats = torch.from_numpy(np.stack([b['input_features'] for b in batch])).float()
        maxlen = max(len(b['labels']) for b in batch)
        labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            labels[i, :len(b['labels'])] = torch.tensor(b['labels'], dtype=torch.long)
        texts = [b['text'] for b in batch]
        return {'input_features': feats, 'labels': labels, 'texts': texts}
    return collate
