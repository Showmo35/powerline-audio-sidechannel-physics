#!/usr/bin/env python3
"""summarize.py — collate all conditions × decoders into one null-subtracted table."""
import json, os
from config import OUT_DIR


def load(src, name):
    p = os.path.join(OUT_DIR, f'vit_{src}', name)
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    rows = {}
    for src in ('real', 'null', 'gen'):
        d = load(src, 'decode.json'); l = load(src, 'llm.json')
        tl = load(src, 'train_log.json')
        greedy_best = min((e['wer'] for e in tl.get('evals', [])), default=None)
        rows[src] = {
            'greedy_best_train': (greedy_best * 100 if greedy_best is not None else None),
            'greedy': d.get('wer_greedy'), 'beam': d.get('wer_beam'),
            'beam_lm': d.get('wer_beam_lm'), 'llm': l.get('wer_llm')}

    cols = ['greedy_best_train', 'greedy', 'beam', 'beam_lm', 'llm']
    print(f'\n{"WER %":<18}' + ''.join(f'{c:>16}' for c in ['real', 'null', 'gen']))
    for c in cols:
        line = f'{c:<18}'
        for src in ('real', 'null', 'gen'):
            v = rows[src][c]
            line += f'{("%.1f" % v if v is not None else "-"):>16}'
        print(line)

    print('\nnull-subtracted (gen − null); negative ⇒ real recovery beyond envelope floor:')
    for c in cols:
        g, n = rows['gen'][c], rows['null'][c]
        if g is not None and n is not None:
            print(f'  {c:<16} gen−null = {g - n:+.1f}  (gen {g:.1f} vs null {n:.1f})')

    print('\nVERDICT GUIDE: gen ≈ null ≫ real → capture-wall final (LM only hallucinates). '
          'gen ≪ null → real residual signal.')
    json.dump(rows, open(os.path.join(OUT_DIR, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
