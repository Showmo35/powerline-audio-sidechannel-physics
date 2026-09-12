#!/usr/bin/env python3
"""collect_results.py — gather per-arch train_log.json into one WER table."""
import json
import os
from config import OUT_DIR
import models as M

NAMES = {'m3': 'CTC BiGRU (Mod3)', 'm4': 'U-Net enh+CTC (Mod4)',
         'm5': 'Conformer CTC (Mod5)', 'm6': 'Hybrid CTC/Attn (Mod6)',
         'm7': 'FullSubNet+CTC (Mod7)'}

rows = []
for arch in M.ARCHS:
    p = os.path.join(OUT_DIR, arch, 'train_log.json')
    if not os.path.exists(p):
        rows.append((arch, NAMES[arch], None, None, None))
        continue
    log = json.load(open(p))
    rows.append((arch, NAMES[arch], log.get('best_wer'),
                 log.get('params'), log.get('wall_s')))

print(f'\n{"arch":5} {"architecture":24} {"best WER":>9} {"params":>11} {"wall":>8}')
print('-' * 62)
for arch, name, wer, params, wall in rows:
    wers = f'{wer*100:.2f}%' if wer is not None else '—'
    ps = f'{params:,}' if params else '—'
    ws = f'{wall:.0f}s' if wall else '—'
    print(f'{arch:5} {name:24} {wers:>9} {ps:>11} {ws:>8}')
print('-' * 62)
print('refs:  clean ~4%  ·  M10 Whisper-FT ~103%  ·  M11 GAN ~100%  ·  M12 U-Net ~100%')
