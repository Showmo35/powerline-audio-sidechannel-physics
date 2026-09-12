#!/usr/bin/env python3
"""
fidelity.py — how CLOSE are the generated word-mels to the originals?

M31's training/eval only measures RETRIEVAL (kNN top1/top5). The `corr` in the loss
line is 1 - cosine in the M30 LEARNED PROJECTION space, not mel similarity — so
generation fidelity has never actually been measured for this generator.

This measures it directly, for both arms (comb / raw), on held-out test words:

    fidelity : r( gen_i , real_i )        generated vs its OWN real word mel
    ceiling  : r( real_i , real_j )       two DIFFERENT REAL utterances of the SAME word
    floor    : r( real_i , real_k )       real vs a DIFFERENT word

The ceiling is the honest yardstick for "perfectly close": an arbitrary r>0.9 is
meaningless, because even two genuine recordings of the same word do not hit 0.9.
A generated mel is "as close as a real example" when it reaches the ceiling
distribution.

Reports the fidelity distribution and the COUNT/FRACTION of test words above
thresholds (r>0.9/0.8/0.7) and above the real-vs-real ceiling median.

Output: outputs/fidelity_<arm>.json + outputs/m31_fidelity.png
"""
import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import CFG, RC, OUT_DIR
import models_comb as M
import dataset_words as D
from train import FrontEnd, gen_word_mel


def pear_rows(A, B):
    """row-wise Pearson r between two (N, D) arrays."""
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    n = (A * B).sum(1)
    d = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1)) + 1e-12
    return n / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['comb', 'raw'])
    ap.add_argument('--n', type=int, default=1500, help='test words to score')
    ap.add_argument('--ckpt', default='best.pt')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = (dev == 'cuda')

    te_items, words = D.build_items('test')
    rng = np.random.RandomState(0)
    if len(te_items) > args.n:
        te_items = [te_items[i] for i in rng.choice(len(te_items), args.n, replace=False)]
    print(f'[fid] scoring {len(te_items)} held-out words', flush=True)

    # ── ceiling/floor from REAL mels only (generator-independent) ──
    ds = D.WordSet(te_items)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8, collate_fn=D.collate)
    reals, labs = [], []
    for b in dl:
        reals.append(b['mel_word'].numpy()); labs.append(b['label'].numpy())
    R = np.concatenate(reals); Y = np.concatenate(labs)
    Rf = R.reshape(len(R), -1)

    by = {}
    for i, y in enumerate(Y):
        by.setdefault(int(y), []).append(i)
    ce, fl = [], []
    for i, y in enumerate(Y):
        same = [j for j in by[int(y)] if j != i]
        if same:
            ce.append((i, same[rng.randint(len(same))]))
        other = int(Y[rng.randint(len(Y))])
        while other == int(y):
            other = int(Y[rng.randint(len(Y))])
        fl.append((i, by[other][rng.randint(len(by[other]))]))
    ceil = pear_rows(Rf[[a for a, _ in ce]], Rf[[b for _, b in ce]]) if ce else np.array([0.])
    floor = pear_rows(Rf[[a for a, _ in fl]], Rf[[b for _, b in fl]])
    print(f'[fid] CEILING real-vs-real (same word, diff utterance): '
          f'mean={ceil.mean():.3f} median={np.median(ceil):.3f}', flush=True)
    print(f'[fid] FLOOR   real-vs-real (different word):            '
          f'mean={floor.mean():.3f} median={np.median(floor):.3f}', flush=True)

    out = {'n': len(te_items),
           'ceiling_real_same_word': {'mean': float(ceil.mean()), 'median': float(np.median(ceil))},
           'floor_real_diff_word': {'mean': float(floor.mean()), 'median': float(np.median(floor))},
           'arms': {}}
    fids = {}
    for arm in args.arms:
        p = os.path.join(OUT_DIR, arm, args.ckpt)
        if not os.path.exists(p):
            print(f'[fid] {arm}: no {p}, skipping', flush=True); continue
        ck = torch.load(p, map_location=dev, weights_only=False)
        model = M.build(CFG, frontend=arm).to(dev)
        model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
        mean, std = ck['mel_mean'], ck['mel_std']
        front = FrontEnd(arm, dev)
        G = []
        with torch.no_grad():
            for b in dl:
                cond = front(b['raw'].to(dev), b['f0'].to(dev))
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    gen = model.sample(cond, steps=RC.eval_steps, cfg_scale=RC.cfg_scale_eval)
                gen = gen.float() * std + mean
                G.append(gen_word_mel(gen, b['frs']).cpu().numpy())
        G = np.concatenate(G); Gf = G.reshape(len(G), -1)
        r = pear_rows(Gf, Rf[:len(Gf)])
        fids[arm] = r
        cmed = float(np.median(ceil))
        rec = {'mean': float(r.mean()), 'median': float(np.median(r)),
               'min': float(r.min()), 'max': float(r.max()),
               'n_gt_0.9': int((r > 0.9).sum()), 'n_gt_0.8': int((r > 0.8).sum()),
               'n_gt_0.7': int((r > 0.7).sum()),
               'n_at_or_above_ceiling_median': int((r >= cmed).sum()),
               'frac_at_or_above_ceiling_median': float((r >= cmed).mean()),
               'n_total': int(len(r))}
        out['arms'][arm] = rec
        print(f'\n── {arm} generator fidelity r(gen, own real) ──', flush=True)
        print(f'   mean={rec["mean"]:.3f} median={rec["median"]:.3f} '
              f'min={rec["min"]:.3f} max={rec["max"]:.3f}', flush=True)
        print(f'   words r>0.9: {rec["n_gt_0.9"]}/{rec["n_total"]}   '
              f'r>0.8: {rec["n_gt_0.8"]}   r>0.7: {rec["n_gt_0.7"]}', flush=True)
        print(f'   words reaching the real-vs-real ceiling median ({cmed:.3f}): '
              f'{rec["n_at_or_above_ceiling_median"]}/{rec["n_total"]} '
              f'({100*rec["frac_at_or_above_ceiling_median"]:.1f}%)', flush=True)

    json.dump(out, open(os.path.join(OUT_DIR, 'fidelity.json'), 'w'), indent=2)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bins = np.linspace(-0.3, 1.0, 60)
    ax.hist(floor, bins=bins, color='#999999', alpha=0.55, label='FLOOR: real vs different word')
    ax.hist(ceil, bins=bins, color='#4393c3', alpha=0.6,
            label='CEILING: real vs real (same word)')
    for arm, c in zip(fids, ['#1b7837', '#b2182b']):
        ax.hist(fids[arm], bins=bins, color=c, alpha=0.6, label=f'{arm} generated vs real')
    ax.axvline(np.median(ceil), color='#4393c3', ls='--', lw=2)
    ax.set_xlabel('Pearson r of word mels'); ax.set_ylabel('# test words')
    ax.set_title('M31 generator fidelity — how close are generated word-mels to the originals?\n'
                 f'(n={len(te_items)} held-out words; dashed = real-vs-real ceiling median)',
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'm31_fidelity.png'), dpi=200,
                                    bbox_inches='tight')
    print('\n[fid] wrote outputs/fidelity.json + outputs/m31_fidelity.png', flush=True)


if __name__ == '__main__':
    main()
