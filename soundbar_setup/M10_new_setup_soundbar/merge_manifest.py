#!/usr/bin/env python3
"""
merge_manifest.py — concatenate per-chunk manifests into one training manifest
and print dataset statistics.

    python merge_manifest.py            # → outputs/dataset/manifest.jsonl
"""

import glob
import json
import os

from config import OUT_DIR

DATASET_DIR = os.path.join(OUT_DIR, 'dataset')


def main():
    shards = sorted(glob.glob(os.path.join(DATASET_DIR, 'chunk_*.manifest.jsonl')))
    rows = []
    for s in shards:
        rows.extend(json.loads(l) for l in open(s) if l.strip())

    out = os.path.join(DATASET_DIR, 'manifest.jsonl')
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    n = len(rows)
    dur = sum(r['dur_s'] for r in rows)
    words = sum(len(r['text'].split()) for r in rows)
    chunks = sorted({r['chunk'] for r in rows})
    print(f'merged {len(shards)} shards → {out}')
    print(f'  utterances : {n}')
    print(f'  audio      : {dur/3600:.2f} h')
    print(f'  words      : {words:,}')
    print(f'  chunks     : {len(chunks)} ({chunks[0]}…{chunks[-1]})')


if __name__ == '__main__':
    main()
