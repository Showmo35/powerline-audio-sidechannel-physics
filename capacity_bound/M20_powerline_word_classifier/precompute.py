#!/usr/bin/env python3
"""Precompute wideband STFT features for all word items -> memmap cache (fast training)."""
import os, json
import numpy as np, torch
from config import CFG, OUT_DIR
import dataset as D

CACHE = os.path.join(OUT_DIR, 'cache'); os.makedirs(CACHE, exist_ok=True)

for split in ('train', 'test'):
    items, words = D.build_items(split=split)
    if split == 'train':
        json.dump(words, open(f'{CACHE}/words.json', 'w'))
    N = len(items)
    arr = np.lib.format.open_memmap(f'{CACHE}/{split}_wide.npy', mode='w+',
                                    dtype=np.float16, shape=(N, CFG.n_wbins, CFG.n_frames))
    ys = np.zeros(N, np.int64)
    ds = D.WordWindows(items)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=16, collate_fn=D.collate, shuffle=False)
    k = 0
    for b in dl:
        n = b['wide'].shape[0]
        arr[k:k + n] = b['wide'].numpy().astype(np.float16); ys[k:k + n] = b['y'].numpy(); k += n
        if (k // 64) % 20 == 0:
            print(f'  {split} {k}/{N}', flush=True)
    np.save(f'{CACHE}/{split}_y.npy', ys); arr.flush()
    print(f'[cache] {split} {N} items -> {CACHE}/{split}_wide.npy', flush=True)
