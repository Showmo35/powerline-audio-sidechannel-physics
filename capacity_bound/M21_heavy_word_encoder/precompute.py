#!/usr/bin/env python3
"""Cache raw 200 kHz word windows (lossless) -> memmap for fast heavy training."""
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
    arr = np.lib.format.open_memmap(f'{CACHE}/{split}_raw.npy', mode='w+',
                                    dtype=np.float16, shape=(N, CFG.a_len))
    ys = np.zeros(N, np.int64)
    dl = torch.utils.data.DataLoader(D.RawWindows(items), batch_size=64, num_workers=16,
                                     collate_fn=D.collate, shuffle=False)
    k = 0
    for b in dl:
        n = b['raw'].shape[0]
        arr[k:k + n] = b['raw'].numpy().astype(np.float16); ys[k:k + n] = b['y'].numpy(); k += n
        if (k // 64) % 40 == 0:
            print(f'  {split} {k}/{N}', flush=True)
    np.save(f'{CACHE}/{split}_y.npy', ys); arr.flush()
    print(f'[cache] {split} {N} raw windows', flush=True)
