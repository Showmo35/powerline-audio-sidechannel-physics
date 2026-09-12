#!/usr/bin/env python3
"""
viz_gen_vs_real.py — RANDOM held-out words: real audio mel vs M31-generated mel.

Random sample (seeded), NOT best/worst — no cherry-picking. Shows both arms so the
comb-vs-raw generation can be eyeballed side by side, with the Pearson r of each
generated mel against the real one.

Output: outputs/m31_gen_vs_real.png
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import CFG, RC, OUT_DIR
import models_comb as M
import dataset_words as D
from train import FrontEnd, gen_word_mel


def pear(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float((a * b).sum() / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8, help='random words to show')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arms', nargs='+', default=['raw', 'comb'])
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = (dev == 'cuda')

    te_items, words = D.build_items('test')
    rng = np.random.RandomState(args.seed)
    pick = rng.choice(len(te_items), args.n, replace=False)      # RANDOM, seeded
    items = [te_items[i] for i in pick]
    labels = [words[it[3]] for it in items]
    print(f'[viz] random words: {labels}', flush=True)

    dl = DataLoader(D.WordSet(items), batch_size=len(items), shuffle=False,
                    num_workers=4, collate_fn=D.collate)
    batch = next(iter(dl))
    real = batch['mel_word'].numpy()                             # (n, n_mels, T)

    gens = {}
    for arm in args.arms:
        p = os.path.join(OUT_DIR, arm, 'best.pt')
        if not os.path.exists(p):
            print(f'[viz] no {p}, skip'); continue
        ck = torch.load(p, map_location=dev, weights_only=False)
        model = M.build(CFG, frontend=arm).to(dev)
        model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
        mean, std = ck['mel_mean'], ck['mel_std']
        front = FrontEnd(arm, dev)
        with torch.no_grad():
            cond = front(batch['raw'].to(dev), batch['f0'].to(dev))
            torch.manual_seed(0)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                g = model.sample(cond, steps=RC.eval_steps, cfg_scale=RC.cfg_scale_eval)
            g = g.float() * std + mean
            gens[arm] = gen_word_mel(g, batch['frs']).cpu().numpy()

    cols = ['real'] + list(gens)
    n = len(items)
    fig = plt.figure(figsize=(2.9 * len(cols) + 1.2, 1.35 * n + 1.0))
    gs = GridSpec(n, len(cols), figure=fig, hspace=0.55, wspace=0.08,
                  left=0.11, right=0.99, top=0.90, bottom=0.03)
    titles = {'real': 'REAL (audio)', 'raw': 'GENERATED — raw arm', 'comb': 'GENERATED — comb arm'}
    for r in range(n):
        mats = [real[r]] + [gens[a][r] for a in gens]
        vmin = min(m.min() for m in mats); vmax = max(m.max() for m in mats)
        for c, key in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            m = real[r] if key == 'real' else gens[key][r]
            ax.imshow(m, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles.get(key, key), fontsize=9, pad=4)
            if c == 0:
                ax.set_ylabel(f'"{labels[r]}"', fontsize=9, rotation=0, ha='right',
                              va='center', labelpad=26)
            else:
                ax.set_xlabel(f'r={pear(gens[key][r], real[r]):.2f}', fontsize=8.5,
                              color='#b2182b')
    fig.suptitle('M31: random held-out words — real audio mel vs powerline-generated mel\n'
                 '(random sample, not cherry-picked; r = Pearson vs the real mel)',
                 fontsize=11, y=0.975)
    out = os.path.join(OUT_DIR, 'm31_gen_vs_real.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('[viz] wrote', out, flush=True)


if __name__ == '__main__':
    main()
