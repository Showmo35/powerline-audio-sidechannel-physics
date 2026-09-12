#!/usr/bin/env python3
"""
M26 K-way scaling — bridges the 0.73 pairwise result and M23's retrieval failure.

Supervised K-way classification on M22 generated word-mels (TEST chunks only, so
M22 never trained on these windows), accuracy vs vocabulary size K. Overlaid with:
  - REAL-mel supervised K-way (ceiling)
  - M23 UNSUPERVISED retrieval top-1 (from M23/results.json scaling, M14-gen)
  - chance 1/K

Shows the collapse from ~0.7 at K=2 toward M23's ~0.27 at K=30, and the gap between
a trained classifier (supervised) and nearest-neighbour retrieval (M23) — i.e. why a
'not bad' pairwise number does not translate into open-vocab retrieval.

Output: M26_wordpair_signal/kway_scaling.png + kway_results.json
"""
import os, sys, json, time
import numpy as np
from scipy.signal import resample_poly
import torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

import analyze_pair as AP
import fuse_pair as FP

PROOT = AP.PROOT; CAP_SR = AP.CAP_SR; AUD_SR = AP.AUD_SR; CTX = AP.CTX
M23 = os.path.join(PROOT, 'M23_crossmodal_word_retrieval')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
KS = [2, 3, 5, 8, 12, 20, 30]
CAP_PER_WORD = 100
MIN_PER_WORD = 25
T = 32


def to_T(m):
    m = torch.nn.functional.interpolate(m[None, None], size=(80, T), mode='bilinear', align_corners=False)[0, 0]
    return m.numpy().astype(np.float32)


def kway_acc(X, y, K, seed=0):
    ncomp = min(50, X.shape[1], int(0.7 * X.shape[0]) - 1)
    clf = make_pipeline(StandardScaler(), PCA(ncomp, random_state=seed),
                        LogisticRegression(max_iter=3000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean())


def main():
    occ = json.load(open(AP.INDEX)); rng = np.random.RandomState(0)
    words = json.load(open(os.path.join(M23, 'data', 'words.json')))   # top-30 freq, freq-ordered
    m22, c22, mm, msd = FP.load_m22()
    Lg = int(c22.win_s * CAP_SR)

    def is_test(ch): return int(ch.split('_')[1]) % c22.test_every == 0

    GEN, REAL, Y = [], [], []
    gen_inputs, gen_frs = [], []
    lagcache = {}; t0 = time.time()
    for wi, w in enumerate(words):
        rows = [r for r in occ[w] if is_test(r[0])]; rng.shuffle(rows); rows = rows[:CAP_PER_WORD]
        if len(rows) < MIN_PER_WORD:
            print(f'[skip] {w}: only {len(rows)} test occ', flush=True); continue
        kept = 0
        for ch, s, e in rows:
            if ch not in lagcache:
                lagcache[ch] = AP.io.read_lag_ms(AP.lag_path(ch)) / 1000.0
            lag = lagcache[ch]; dur = (e - s) + 2 * CTX
            aud = AP.io.read_wav_window(AP.wav_path(ch), s - CTX, dur, AUD_SR).astype(np.float32)
            capL = AP.io.read_bin_window(AP.bin_path(ch), s - CTX + lag, c22.win_s, CAP_SR).astype(np.float32)
            if len(aud) < AUD_SR * 0.08 or len(capL) < CAP_SR * 0.5:
                continue
            REAL.append(AP.clean_mel_T(aud).ravel())
            capL = np.pad(capL, (0, max(0, Lg - len(capL))))[:Lg]
            r = resample_poly(capL, c22.in_sr, CAP_SR).astype(np.float32)
            r = r / (np.sqrt(np.mean(r ** 2)) + 1e-8)
            gen_inputs.append(r); gen_frs.append(max(6, int(round(dur * c22.fps))))
            Y.append(wi); kept += 1
        print(f'[data] {w}: {kept}  ({time.time()-t0:.0f}s)', flush=True)

    Y = np.array(Y)
    BS = 16
    for b in range(0, len(gen_inputs), BS):
        X = torch.from_numpy(np.stack(gen_inputs[b:b + BS])).to(dev)
        torch.manual_seed(0)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
            g = m22.sample(X)
        g = (g.float().cpu() * msd + mm)
        for j in range(g.shape[0]):
            GEN.append(to_T(g[j, :, :gen_frs[b + j]]).ravel())
        if b % (BS * 10) == 0:
            print(f'[gen] {b+g.shape[0]}/{len(gen_inputs)}', flush=True)
    GEN = np.array(GEN); REAL = np.array(REAL)
    print(f'[m26k] GEN{GEN.shape} REAL{REAL.shape}  n={len(Y)} words_used={len(set(Y))}', flush=True)

    # nested top-K subsets (words are frequency-ordered)
    gen_curve, real_curve, chance = [], [], []
    Ks_used = []
    for K in KS:
        keep = [i for i in range(len(words)) if i < K]
        m = np.isin(Y, keep)
        if len(set(Y[m])) < K:
            print(f'[k={K}] only {len(set(Y[m]))} classes present — skipping', flush=True); continue
        remap = {c: i for i, c in enumerate(sorted(set(Y[m])))}
        yk = np.array([remap[c] for c in Y[m]])
        gk = kway_acc(GEN[m], yk, K); rk = kway_acc(REAL[m], yk, K)
        gen_curve.append(gk); real_curve.append(rk); chance.append(1.0 / K); Ks_used.append(K)
        print(f'[k={K:2d}] GEN_sup={gk:.3f}  REAL_sup={rk:.3f}  chance={1.0/K:.3f}', flush=True)

    # M23 unsupervised retrieval overlay
    m23 = json.load(open(os.path.join(M23, 'results', 'results.json')))['scaling']

    OUT = os.path.join(PROOT, 'M26_wordpair_signal')
    json.dump({'Ks': Ks_used, 'GEN_supervised': gen_curve, 'REAL_supervised': real_curve,
               'chance': chance, 'm23_unsup_retrieval': {'K': m23['n_classes'], 'A_gen': m23['A_gen']}},
              open(os.path.join(OUT, 'kway_results.json'), 'w'), indent=1)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(Ks_used, real_curve, 'o-', c='#2563d6', label='REAL mel, supervised (ceiling)')
    ax.plot(Ks_used, gen_curve, 's-', c='#e08a10', lw=2.4, label='M22 GEN mel, SUPERVISED K-way (this run)')
    ax.plot(m23['n_classes'], m23['A_gen'], '^--', c='#b2182b', label='M23 GEN, UNSUPERVISED retrieval')
    ax.plot(Ks_used, chance, ':', c='#666', label='chance 1/K')
    ax.set_xlabel('vocabulary size K'); ax.set_ylabel('top-1 accuracy'); ax.set_xscale('log'); ax.set_xticks(Ks_used)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_title('Why a "0.73 pairwise" does not survive retrieval\n'
                 'supervised gen-mel classification vs K, vs M23 unsupervised retrieval (test-only)')
    ax.legend(fontsize=8.5); ax.grid(alpha=.25)
    for k, g in zip(Ks_used, gen_curve): ax.text(k, g + .02, f'{g:.2f}', ha='center', fontsize=8, color='#e08a10')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'kway_scaling.png'), dpi=150, bbox_inches='tight')
    print('[m26k] wrote', os.path.join(OUT, 'kway_scaling.png'), flush=True)

    print('\n==== M26 K-WAY VERDICT ====', flush=True)
    for k, g, r in zip(Ks_used, gen_curve, real_curve):
        print(f'  K={k:2d}: GEN_sup={g:.3f}  REAL_sup={r:.3f}  chance={1/k:.3f}', flush=True)


if __name__ == '__main__':
    main()
