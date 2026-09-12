#!/usr/bin/env python3
"""
check_split.py — train/test separation audit (run before training).

Asserts: chunk sets disjoint; after filtering, NO utt_id and NO normalized text
appears on both sides of the split. Prints what was dropped and why.
"""
from collections import Counter

from config import CFG
import dataset as D
import text as T


def main():
    rows = D.load_manifest()
    test = CFG.test_chunks()

    # duplicate utt_ids anywhere in the manifest, and where they live
    by_id = Counter(r['utt_id'] for r in rows)
    dupes = {u for u, n in by_id.items() if n > 1}
    cross = []
    for u in dupes:
        chunks = {r['chunk'] for r in rows if r['utt_id'] == u}
        sides = {('test' if c in test else 'train') for c in chunks}
        if len(sides) == 2:
            cross.append((u, sorted(chunks)))
    print(f'[manifest] rows={len(rows)}  dupe utt_ids={len(dupes)}  '
          f'of which CROSS-SPLIT={len(cross)}')
    for u, ch in cross[:10]:
        print(f'    cross-split dupe: {u}  in {ch}')

    audit = D.audit_split(rows, CFG)   # raises if split_rows leaves any leakage
    print(f'[audit] PASS — {audit}')
    tr, te, info = D.split_rows(rows, CFG)
    print(f'[final] train={len(tr)}  test={len(te)}  '
          f'(dropped {info["train_dropped_as_test_dupes"]} train rows that shared '
          f'an utt_id or text with the test side)')


if __name__ == '__main__':
    main()
