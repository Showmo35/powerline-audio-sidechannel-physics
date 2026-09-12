#!/usr/bin/env python3
"""
viz_wrong_grid.py — examples of M22 generated mels correlating with the WRONG word.

These are the failure cases behind the histogram (impostor corr > correct corr).
For a sample of TEST words we generate the mel, retrieve the nearest gallery word,
and keep the cases where the nearest word is WRONG. For a few of the most confident
confusions we render three mels per row:

  [ correct word (real) ]  [ M22 generated ]  [ retrieved WRONG word (real) ]

labelled with r(gen, correct) and r(gen, impostor) — the generated mel looks more
like the wrong word than its own.

Output -> outputs/m22_wrong_grid.png  (run on GPU; see run_wrong.slurm)
"""
import argparse, os
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import CFG, RC, OUT_DIR, build_vocab
import models as M
import dataset_words as D
import train_retrieval as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', type=int, default=800)
    ap.add_argument('--gcap', type=int, default=RC.gallery_per_word)
    ap.add_argument('--n', type=int, default=6, help='failure examples to show')
    ap.add_argument('--ckpt', default=os.path.join(OUT_DIR, 'best.pt'))
    ap.add_argument('--tag', default='M22')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'm22_wrong_grid.png'))
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'
    use_amp = (dev == 'cuda')

    words, _ = build_vocab()
    model = M.build(CFG).to(dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
    mean, std = ck['mel_mean'], ck['mel_std']

    # gallery: real train mels (capped/word). Keep items so we can re-read the
    # specific impostor / correct exemplar mels for display.
    tr_items = D.build_items('train')[0]
    gal_items = D.cap_per_label(tr_items, args.gcap)
    gal_ds = D.WordSet(gal_items, real_only=True)
    gal_dl = DataLoader(gal_ds, batch_size=256, shuffle=False, num_workers=8,
                        collate_fn=lambda bb: {'mel_word': torch.stack([x['mel_word'] for x in bb]),
                                               'label': torch.tensor([x['label'] for x in bb])})
    print('[wrong] building gallery …', flush=True)
    G, gy = T.build_gallery(gal_dl, dev)
    gallery_words = set(gy.tolist())
    print(f'[wrong] gallery={G.shape[0]} over {len(gallery_words)} words', flush=True)

    te_items = D.build_items('test')[0]
    te_items = [it for it in te_items if it[3] in gallery_words]
    rng = np.random.RandomState(0)
    if len(te_items) > args.queries:
        te_items = [te_items[i] for i in rng.choice(len(te_items), args.queries, replace=False)]
    te_ds = D.WordSet(te_items)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False, num_workers=8,
                       collate_fn=D.collate)

    fails = []   # dict per wrong query
    qptr = 0
    with torch.no_grad():
        for b in te_dl:
            raw = b['raw'].to(dev)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                gen = model.sample(raw, steps=RC.eval_steps, cfg_scale=RC.cfg_scale_eval)
            gen = gen.float() * std + mean
            gw = T.gen_word_mel(gen, b['frs'])                    # (b,n_mels,T)
            q = T.embed(gw).half()
            sims = q @ G.t()                                      # (b,Ng)
            for i in range(len(gw)):
                t = int(b['label'][i])
                same = (gy == t)
                top1 = int(sims[i].argmax())
                pred = int(gy[top1])
                if pred == t:
                    continue                                     # correct -> skip
                r_imp = float(sims[i][top1])
                cidx = int(torch.where(same)[0][sims[i][same].argmax()])
                r_cor = float(sims[i][cidx])
                fails.append({
                    'true': words[t], 'pred': words[pred], 'r_cor': r_cor, 'r_imp': r_imp,
                    'gen': gw[i].cpu().numpy(),
                    'q_item': te_items[qptr + i],                 # correct word occurrence
                    'imp_item': gal_items[top1],                  # matched wrong occurrence
                })
            qptr += len(gw)

    print(f'[wrong] {len(fails)} wrong retrievals collected', flush=True)
    # most confident, varied confusions: high impostor corr, dedupe by true word
    fails.sort(key=lambda d: -d['r_imp'])
    chosen, used = [], set()
    for d in fails:
        if d['true'] in used:
            continue
        used.add(d['true']); chosen.append(d)
        if len(chosen) == args.n:
            break

    # helper to read a real word mel for display
    reader = D.WordSet([])
    def real_of(item):
        ch, s, e = item[0], item[1], item[2]
        return reader._word_mel(ch, s, e)[0].numpy()

    # ── render: n rows x 3 cols [correct | generated | retrieved-wrong] ────────
    n = len(chosen)
    fig = plt.figure(figsize=(8.4, 1.15 * n + 1.1))
    gs = GridSpec(n, 3, figure=fig, hspace=0.55, wspace=0.08,
                  left=0.02, right=0.99, top=0.88, bottom=0.03)
    fig.suptitle(f'{args.tag}: generated mels that correlate with the WRONG word\n'
                 f'(held-out words · generated mel matches an impostor better than its own)',
                 fontsize=10, y=0.985, va='top')
    col_titles = ['correct word (real)', f'{args.tag} generated', 'retrieved WRONG word (real)']
    for row, d in enumerate(chosen):
        real_c = real_of(d['q_item']); real_i = real_of(d['imp_item']); gen = d['gen']
        vmin = min(real_c.min(), gen.min(), real_i.min())
        vmax = max(real_c.max(), gen.max(), real_i.max())
        panels = [(real_c, f'“{d["true"]}”'),
                  (gen,    f'r={d["r_cor"]:.2f} to correct\nr={d["r_imp"]:.2f} to wrong'),
                  (real_i, f'“{d["pred"]}”')]
        for c, (m, sub) in enumerate(panels):
            ax = fig.add_subplot(gs[row, c])
            ax.imshow(m, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[c], fontsize=8.5, pad=3)
            ax.set_xlabel(sub, fontsize=8,
                          color=('#b2182b' if c == 1 else 'black'))
        # red frame on the wrong panel
        fig.axes[-1].patch.set_edgecolor('#b2182b')
        for sp in fig.axes[-1].spines.values():
            sp.set_color('#b2182b'); sp.set_linewidth(1.6)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print('[wrong] wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
