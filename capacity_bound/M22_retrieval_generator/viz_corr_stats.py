#!/usr/bin/env python3
"""
viz_corr_stats.py — M22 retrieval statistics over held-out test words.

For a sample of TEST word occurrences we generate the mel with M22 (best.pt), then
against a gallery of REAL train word-mels covering every word we compute, per query:
  * correct-word correlation  = best cosine(gen, real mel of the SAME word)
  * impostor correlation       = best cosine(gen, real mel of any OTHER word)
  * top-1 / top-5 retrieval    = is the true word the nearest / among 5 nearest?
(cosine here == Pearson r: mels are time-normalized, flattened, mean-centered, L2-normed.)

Key identity: a query is top-1 CORRECT  <=>  correct-word corr > impostor corr.
So the "% correct by correlation" is exactly the fraction of the histogram where the
correct-word bar sits to the right of its impostor.

Output -> outputs/m22_corr_stats.png  (+ printed numbers). Only queries whose word is
enrolled in the gallery are scored (a word absent from the gallery can't be retrieved).
"""
import argparse, os
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import CFG, RC, OUT_DIR
import models as M
import dataset_words as D
import train_retrieval as T
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', type=int, default=3000)
    ap.add_argument('--gcap', type=int, default=RC.gallery_per_word,
                    help='real mels per word in the gallery (override for quick tests)')
    ap.add_argument('--tag', default='M22', help='generator name shown on the figure')
    ap.add_argument('--ckpt', default=os.path.join(OUT_DIR, 'best.pt'))
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'm22_corr_stats.png'))
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'
    use_amp = (dev == 'cuda')

    model = M.build(CFG).to(dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
    mean, std = ck['mel_mean'], ck['mel_std']

    # gallery = real train word mels, capped per word (covers all enrolled words)
    tr_items = D.build_items('train')[0]
    gal_items = D.cap_per_label(tr_items, args.gcap)
    gal_ds = D.WordSet(gal_items, real_only=True)
    gal_dl = DataLoader(gal_ds, batch_size=256, shuffle=False, num_workers=8,
                        collate_fn=lambda bb: {'mel_word': torch.stack([x['mel_word'] for x in bb]),
                                               'label': torch.tensor([x['label'] for x in bb])})
    print('[stats] building gallery …', flush=True)
    G, gy = T.build_gallery(gal_dl, dev)                 # (Ng,Dm) fp16, (Ng,)
    gallery_words = set(gy.tolist())
    print(f'[stats] gallery={G.shape[0]} mels over {len(gallery_words)} words', flush=True)

    # queries = sample of test occurrences whose word is in the gallery
    te_items = D.build_items('test')[0]
    te_items = [it for it in te_items if it[3] in gallery_words]
    rng = np.random.RandomState(0)
    if len(te_items) > args.queries:
        te_items = [te_items[i] for i in rng.choice(len(te_items), args.queries, replace=False)]
    print(f'[stats] scoring {len(te_items)} in-gallery test queries', flush=True)

    te_dl = DataLoader(D.WordSet(te_items), batch_size=args.batch, shuffle=False,
                       num_workers=8, collate_fn=D.collate)

    cc, ic, top1, top5 = [], [], [], []
    with torch.no_grad():
        for b in te_dl:
            raw = b['raw'].to(dev)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                gen = model.sample(raw, steps=RC.eval_steps, cfg_scale=RC.cfg_scale_eval)
            gen = gen.float() * std + mean
            q = T.embed(T.gen_word_mel(gen, b['frs'])).half()      # (b,Dm)
            sims = q @ G.t()                                       # (b,Ng)
            y = b['label'].to(dev)
            k = min(5, G.shape[0])
            top = sims.topk(k, dim=1).indices
            labk = gy[top]
            for i in range(len(y)):
                t = y[i]
                same = (gy == t)
                cc.append(float(sims[i][same].max()))
                ic.append(float(sims[i][~same].max()))
                top1.append(bool(labk[i, 0] == t))
                top5.append(bool((labk[i] == t).any()))

    cc = np.array(cc); ic = np.array(ic)
    a1 = float(np.mean(top1)); a5 = float(np.mean(top5))
    sep = float(np.mean(cc > ic))
    n = len(cc)
    print(f'[stats] N={n}  correct-word corr: mean={cc.mean():.3f} median={np.median(cc):.3f}', flush=True)
    print(f'[stats]        impostor corr:     mean={ic.mean():.3f} median={np.median(ic):.3f}', flush=True)
    print(f'[stats] % CORRECT by correlation: top1={a1*100:.1f}%  top5={a5*100:.1f}%  '
          f'(corr>impostor={sep*100:.1f}%)', flush=True)

    # ── figure: (A) correlation histograms  (B) % correct ─────────────────────
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={'width_ratios': [2.1, 1]})
    C_OK, C_BAD, C_ACC = '#2166ac', '#b2182b', '#1b7837'
    bins = np.linspace(min(cc.min(), ic.min()), 1.0, 45)
    axA.hist(ic, bins=bins, color=C_BAD, alpha=0.55, label='best WRONG word (impostor)')
    axA.hist(cc, bins=bins, color=C_OK, alpha=0.65, label='CORRECT word')
    axA.axvline(cc.mean(), color=C_OK, lw=2, ls='--')
    axA.axvline(ic.mean(), color=C_BAD, lw=2, ls='--')
    axA.set_xlabel(f'correlation (cosine = Pearson r) of {args.tag}-generated mel vs. real word mel')
    axA.set_ylabel('number of test words')
    axA.set_title(f'Each test word’s correlation with the correct word vs. the best impostor\n'
                  f'(N = {n:,} held-out words · means: correct {cc.mean():.2f}, impostor {ic.mean():.2f})',
                  fontsize=10)
    axA.legend(frameon=False, fontsize=9)
    axA.spines[['top', 'right']].set_visible(False)

    # panel B: % correct by nearest-correlation
    bars = axB.bar(['top-1', 'top-5'], [a1 * 100, a5 * 100],
                   color=[C_ACC, '#7fbf7b'], width=0.6)
    for rct, v in zip(bars, [a1, a5]):
        axB.text(rct.get_x() + rct.get_width() / 2, v * 100 + 1.2, f'{v*100:.1f}%',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    axB.set_ylim(0, max(a5 * 100 * 1.35, 12))
    axB.set_ylabel('% of test words retrieved correctly')
    axB.set_title(f'Correct solely by correlation\n(nearest word in gallery of '
                  f'{len(gallery_words):,} words)', fontsize=10)
    axB.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'{args.tag} retrieval statistics — generated-mel correlation & word-recovery rate',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print('[stats] wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
