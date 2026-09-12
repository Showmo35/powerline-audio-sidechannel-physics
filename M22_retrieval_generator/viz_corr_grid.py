#!/usr/bin/env python3
"""
viz_corr_grid.py — qualitative M22 check: generated mel vs original (real) mel for
the BEST- and WORST-correlated held-out test words.

For a random sample of TEST word occurrences we:
  1. generate the mel from the powerline .bin with M22 (outputs/best.pt, full sampler),
  2. slice the word region + time-normalize to (n_mels, T) exactly like the retrieval
     space, and compute Pearson r against the REAL word mel,
  3. pick the N best- and N worst-correlated words (deduped by word for variety),
  4. render a PNG grid: rows = words, columns = [Original | Generated], per-row shared
     color scale, row labelled with the word and its r.

Run on GPU (see run_viz.slurm). Output -> outputs/m22_corr_grid.png
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import CFG, RC, OUT_DIR, build_vocab
import models as M
import dataset_words as D


def embed_flat(m):                               # (n_mels, T) -> centered flat vec
    z = m.reshape(-1)
    return z - z.mean()


def pearson(a, b):
    a, b = embed_flat(a), embed_flat(b)
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def gen_word_mel(gen_full, frs):                 # (B,n_mels,nF) -> list (n_mels,T)
    out = []
    for i in range(gen_full.shape[0]):
        w = gen_full[i:i + 1, :, :int(frs[i])]
        w = F.interpolate(w[None], size=(CFG.n_mels, RC.T),
                          mode='bilinear', align_corners=False)[0, 0]
        out.append(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=800, help='test occurrences to score')
    ap.add_argument('--n', type=int, default=6, help='rows per (best / worst) block')
    ap.add_argument('--ckpt', default=os.path.join(OUT_DIR, 'best.pt'))
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'm22_corr_grid.png'))
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'

    words, _ = build_vocab()                     # id -> word text
    te_items, _ = D.build_items('test')
    rng = np.random.RandomState(0)
    if len(te_items) > args.sample:
        te_items = [te_items[i] for i in rng.choice(len(te_items), args.sample, replace=False)]
    print(f'[viz] scoring {len(te_items)} test occurrences', flush=True)

    model = M.build(CFG).to(dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
    mean, std = ck['mel_mean'], ck['mel_std']

    ds = D.WordSet(te_items)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=8,
                    collate_fn=D.collate)

    recs = []   # (r, word, real_mel(np), gen_mel(np))
    use_amp = (dev == 'cuda')
    seen = 0
    with torch.no_grad():
        for b in dl:
            raw = b['raw'].to(dev)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                gen = model.sample(raw, steps=RC.eval_steps, cfg_scale=RC.cfg_scale_eval)
            gen = gen.float() * std + mean
            gw = gen_word_mel(gen.cpu(), b['frs'])
            for j in range(len(gw)):
                real = b['mel_word'][j]
                r = pearson(gw[j], real)
                recs.append((r, words[int(b['label'][j])],
                             real.numpy(), gw[j].numpy()))
            seen += len(gw)
            if seen % (args.batch * 10) == 0:
                print(f'[viz] {seen}/{len(te_items)}', flush=True)

    rs = np.array([r[0] for r in recs])
    print(f'[viz] corr: mean={rs.mean():.3f} median={np.median(rs):.3f} '
          f'min={rs.min():.3f} max={rs.max():.3f}', flush=True)

    def pick(order):                             # dedupe by word for variety
        chosen, used = [], set()
        for i in order:
            w = recs[i][1]
            if w in used:
                continue
            used.add(w); chosen.append(i)
            if len(chosen) == args.n:
                break
        return chosen

    asc = np.argsort(rs)
    worst = pick(list(asc))
    best = pick(list(asc[::-1]))
    rows = [('BEST', best), ('WORST', worst)]

    # ── render grid: (2*n) rows x 2 cols (Original | Generated) ───────────────
    n = args.n
    TOP, BOT = 0.90, 0.045
    fig = plt.figure(figsize=(6.6, 1.05 * 2 * n + 1.2))
    gs = GridSpec(2 * n, 2, figure=fig, hspace=0.32, wspace=0.06,
                  left=0.17, right=0.985, top=TOP, bottom=BOT)
    fig.suptitle('M22: generated vs. original mel — best & worst correlated test words\n'
                 f'(open vocab {len(words):,} words · powerline→mel · r = Pearson over the word mel)',
                 fontsize=10, y=0.995, va='top')

    row = 0
    for tag, idxs in rows:
        for gi in idxs:
            r, w, real, gen = recs[gi]
            vmin = min(real.min(), gen.min()); vmax = max(real.max(), gen.max())
            for c, (m, ttl) in enumerate([(real, 'Original'), (gen, 'Generated')]):
                ax = fig.add_subplot(gs[row, c])
                ax.imshow(m, origin='lower', aspect='auto', cmap='magma',
                          vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(ttl, fontsize=9, pad=4)
                if c == 0:
                    ax.set_ylabel(f'“{w}”\nr={r:.2f}', fontsize=8, rotation=0,
                                  ha='right', va='center', labelpad=24)
            row += 1

    # side block labels at the vertical center of each block
    mid = (TOP + BOT) / 2
    fig.text(0.02, (TOP + mid) / 2, 'BEST\ncorr', fontsize=10, fontweight='bold',
             color='#1b7837', ha='left', va='center')
    fig.text(0.02, (mid + BOT) / 2, 'WORST\ncorr', fontsize=10, fontweight='bold',
             color='#b2182b', ha='left', va='center')
    # faint divider between the two blocks
    fig.add_artist(plt.Line2D([0.02, 0.985], [mid, mid], color='0.7',
                              lw=0.8, ls=(0, (4, 4)), transform=fig.transFigure))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print('[viz] wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
