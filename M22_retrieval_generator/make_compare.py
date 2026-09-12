#!/usr/bin/env python3
"""
make_compare.py — M14 vs M22 retrieval comparison (from the two stats runs).

Numbers come from viz_corr_stats.py on the SAME gallery/queries/seeds, only the
generator weights differing:
  M14 (reconstruction, job 6259137): top1 3.1%  top5 8.6%  correct 0.616  impostor 0.756
  M22 (retrieval-opt,   job 6258969): top1 3.8%  top5 11.6% correct 0.616  impostor 0.750
Output -> outputs/m14_vs_m22_compare.png  (login-node, no GPU).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs',
                   'm14_vs_m22_compare.png')

# (M14, M22)
acc = {'top-1': (3.1, 3.8), 'top-5': (8.6, 11.6)}                # % correct
corr = {'correct\nword': (0.616, 0.616), 'best\nimpostor': (0.756, 0.750)}

C14, C22 = '#7a7a7a', '#1b7837'    # baseline gray, improved green
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.3))


def grouped(ax, data, ylabel, title, fmt, ymax=None):
    labels = list(data); x = np.arange(len(labels)); w = 0.36
    m14 = [data[k][0] for k in labels]; m22 = [data[k][1] for k in labels]
    b1 = ax.bar(x - w / 2, m14, w, label='M14 (reconstruction)', color=C14)
    b2 = ax.bar(x + w / 2, m22, w, label='M22 (retrieval-optimized)', color=C22)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(),
                    format(r.get_height(), fmt), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    if ymax: ax.set_ylim(0, ymax)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')


grouped(axA, acc, '% of test words retrieved correctly',
        'Word recovery by nearest-correlation\n(open vocab, 24,108-word gallery · N=3000)',
        '.1f', ymax=14)
grouped(axB, corr, 'mean correlation (Pearson r)',
        'Generated-mel correlation\n(correct word vs. best impostor)',
        '.3f', ymax=0.9)

fig.suptitle('M14 vs M22 — training the generator FOR retrieval (same gallery, queries, seeds)',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print('wrote', OUT)
