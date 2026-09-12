#!/usr/bin/env python3
"""
M26 high-frequency probe — does the >16 kHz band (which M22 discards) carry
information that separates 'which' vs 'this'?

Comprehensive, confound-controlled test:
  * Band-by-band separability across 0-100 kHz (0-16 = what M22 uses; 16-100 = the
    discarded band, split into 16-32/32-48/48-64/64-80/80-100 kHz).
  * Coherent SIDEBAND-SUM: for all mains harmonics in a band, extract the ±4 kHz AM
    sideband and average across harmonics -> a denoised baseband spectrogram (the most
    sensitive detector of weak high-harmonic speech AM). Built for LOW and HIGH bands.
  * Two classifiers: linear logistic (PCA) AND nonlinear HistGradientBoosting.
  * GroupKFold BY CHUNK -> the model can't exploit per-recording artifacts; any
    separability must generalize across recordings. Permutation null on top.
  * Decisive contrasts: HIGH alone vs chance; does LOW+HIGH beat LOW (high ADDS?).

No trained generator is involved -> no train/test leakage; use ALL occurrences.

Output: M26_wordpair_signal/<A>_<B>/highfreq_probe.png + highfreq_results.json
"""
import os, sys, json, argparse, time
import numpy as np
from scipy import signal as dsp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

import analyze_pair as AP
CAP_SR = AP.CAP_SR; AUD_SR = AP.AUD_SR; CTX = AP.CTX

NPS = 8192                     # STFT window -> ~24 Hz res (resolves 61 Hz harmonics)
HOP = 2048
BW  = 4000.0                   # sideband half-width (Hz)
BB_BINS = 48                   # baseband bins for sideband-sum (over ±BW)
TN = 16                        # time-normalized frames for sideband-sum spectrograms


def stft_mag(x):
    f, t, Z = dsp.stft(x, fs=CAP_SR, nperseg=NPS, noverlap=NPS - HOP, window='hann')
    return f, np.abs(Z).astype(np.float32)          # (F, T)


def build_sideband_matrix(freq, fmains, flo, fhi):
    """Averaging matrix M (BB_BINS x F): coherently sum ±BW sidebands of every mains
    harmonic in [flo,fhi] onto a common baseband axis."""
    ks = np.arange(1, int(fhi / fmains) + 1) * fmains
    ks = ks[(ks >= max(flo, fmains)) & (ks <= fhi)]
    deltas = np.linspace(-BW, BW, BB_BINS)
    M = np.zeros((BB_BINS, len(freq)), np.float32)
    edges = (freq[:-1] + freq[1:]) / 2.0                       # nearest-bin via searchsorted
    for b, d in enumerate(deltas):                             # BB_BINS iters (vectorized over harmonics)
        j = np.searchsorted(edges, ks + d)
        np.add.at(M[b], np.clip(j, 0, len(freq) - 1), 1.0)
    M /= (M.sum(1, keepdims=True) + 1e-8)
    return M, len(ks)


def time_norm(S):
    import torch
    S = torch.nn.functional.interpolate(torch.from_numpy(S)[None, None].float(),
                                        size=(S.shape[0], TN), mode='bilinear', align_corners=False)[0, 0]
    return S.numpy().astype(np.float32)


def separability(X, y, groups, nonlinear=False, nperm=30, seed=0):
    cv = GroupKFold(5)
    if nonlinear:
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.08,
                                             l2_regularization=1.0, random_state=seed)
    else:
        nc = min(40, X.shape[1], int(0.7 * X.shape[0]) - 1)
        clf = make_pipeline(StandardScaler(), PCA(nc, random_state=seed),
                            LogisticRegression(max_iter=2000))
    acc = float(cross_val_score(clf, X, y, groups=groups, cv=cv, scoring='balanced_accuracy').mean())
    rng = np.random.RandomState(seed); nulls = []
    for _ in range(nperm):
        yp = rng.permutation(y)
        nulls.append(float(cross_val_score(clf, X, yp, groups=groups, cv=cv, scoring='balanced_accuracy').mean()))
    nulls = np.array(nulls)
    return {'balacc': acc, 'null_mean': float(nulls.mean()), 'null_std': float(nulls.std()),
            'p': float((np.sum(nulls >= acc) + 1) / (nperm + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, default=['which', 'this'])
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--nperm', type=int, default=30)
    args = ap.parse_args()
    A, B = args.pair
    OUT = os.path.join(AP.PROOT, 'M26_wordpair_signal', f'{A}_{B}'); os.makedirs(OUT, exist_ok=True)
    occ = json.load(open(AP.INDEX)); rng = np.random.RandomState(0)

    probe = AP.io.read_bin_window(AP.bin_path(occ[A][0][0]), 1.0, 4.0, CAP_SR).astype(np.float32)
    fmains = AP.detect_mains(probe, CAP_SR)

    # collect: per-window time-averaged spectrum (for band PSD) + sideband-sum specs
    SPEC, SBL, SBH, Y, G = [], [], [], [], []
    Msb_lo = Msb_hi = freq = None
    lagcache = {}; t0 = time.time()
    for lab, word in [(0, A), (1, B)]:
        rows = occ[word][:]; rng.shuffle(rows); rows = rows[:args.n]; kept = 0
        for ch, s, e in rows:
            if ch not in lagcache:
                lagcache[ch] = AP.io.read_lag_ms(AP.lag_path(ch)) / 1000.0
            dur = (e - s) + 2 * CTX
            cap = AP.io.read_bin_window(AP.bin_path(ch), s - CTX + lagcache[ch], dur, CAP_SR).astype(np.float32)
            if len(cap) < CAP_SR * 0.10:
                continue
            cap = cap / (np.sqrt(np.mean(cap ** 2)) + 1e-8)
            freq, Smag = stft_mag(cap)                       # (F,T)
            if Msb_lo is None:
                Msb_lo, nlo = build_sideband_matrix(freq, fmains, 0, 16_000)
                Msb_hi, nhi = build_sideband_matrix(freq, fmains, 16_000, 100_000)
                print(f'[hf] f_mains={fmains:.2f}  low harmonics={nlo}  high harmonics={nhi}  F={len(freq)}', flush=True)
            SPEC.append(np.log(Smag.mean(1) + 1e-8))         # (F,) time-avg log spectrum
            SBL.append(time_norm(np.log(Msb_lo @ Smag + 1e-8)).ravel())   # low sideband-sum baseband
            SBH.append(time_norm(np.log(Msb_hi @ Smag + 1e-8)).ravel())   # high sideband-sum baseband
            Y.append(lab); G.append(int(ch.split('_')[1])); kept += 1
        print(f'[hf] {word}: kept {kept}  ({time.time()-t0:.0f}s)', flush=True)

    SPEC = np.array(SPEC); SBL = np.array(SBL); SBH = np.array(SBH)
    Y = np.array(Y); G = np.array(G)
    print(f'[hf] SPEC{SPEC.shape} SBL{SBL.shape} SBH{SBH.shape}  n={len(Y)} chunks={len(set(G))}', flush=True)

    # band slices of the PSD spectrum
    def band(lo, hi): return SPEC[:, (freq >= lo) & (freq < hi)]
    bands = {
        '0-16k(M22)': band(0, 16_000), '16-32k': band(16_000, 32_000),
        '32-48k': band(32_000, 48_000), '48-64k': band(48_000, 64_000),
        '64-80k': band(64_000, 80_000), '80-100k': band(80_000, 100_000),
        '16-100k(HIGH)': band(16_000, 100_000), '0-100k(FULL)': band(0, 100_000),
    }
    res = {}
    for name, X in bands.items():
        res[name] = {'lin': separability(X, Y, G, nonlinear=False, nperm=args.nperm)}
        r = res[name]['lin']
        print(f'[psd-lin] {name:14s} balacc={r["balacc"]:.3f} null={r["null_mean"]:.3f}±{r["null_std"]:.3f} p={r["p"]:.3g}', flush=True)

    # sideband-sum feature sets, linear + nonlinear
    sbsets = {'SB_low(0-16k)': SBL, 'SB_high(16-100k)': SBH,
              'PSDlow+SBhigh': np.hstack([bands['0-16k(M22)'], SBH])}
    for name, X in sbsets.items():
        res[name] = {'lin': separability(X, Y, G, nonlinear=False, nperm=args.nperm),
                     'nl': separability(X, Y, G, nonlinear=True, nperm=max(15, args.nperm // 2))}
        print(f'[sb] {name:16s} lin={res[name]["lin"]["balacc"]:.3f}(p={res[name]["lin"]["p"]:.3g})  '
              f'nl={res[name]["nl"]["balacc"]:.3f}(p={res[name]["nl"]["p"]:.3g})', flush=True)
    # nonlinear on HIGH band PSD too (best chance for high-freq nonlinear structure)
    res['16-100k(HIGH)']['nl'] = separability(bands['16-100k(HIGH)'], Y, G, nonlinear=True, nperm=max(15, args.nperm // 2))
    res['0-16k(M22)']['nl'] = separability(bands['0-16k(M22)'], Y, G, nonlinear=True, nperm=max(15, args.nperm // 2))
    print(f"[nl] HIGH={res['16-100k(HIGH)']['nl']['balacc']:.3f}  LOW={res['0-16k(M22)']['nl']['balacc']:.3f}", flush=True)

    # discriminability spectrum (point-biserial r^2 per freq bin)
    yc = (Y - Y.mean()); Sc = SPEC - SPEC.mean(0, keepdims=True)
    r2 = (Sc * yc[:, None]).mean(0) ** 2 / ((Sc ** 2).mean(0) * (yc ** 2).mean() + 1e-20)

    low_lin = res['0-16k(M22)']['lin']['balacc']; high_lin = res['16-100k(HIGH)']['lin']['balacc']
    add = res['PSDlow+SBhigh']['lin']['balacc'] - low_lin
    out = {'pair': [A, B], 'n': int(len(Y)), 'f_mains': fmains, 'results': res,
           'HIGH_adds_over_LOW': add}
    json.dump(out, open(os.path.join(OUT, 'highfreq_results.json'), 'w'), indent=1)

    # ── figure ──
    fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(2, 2, hspace=.33, wspace=.22)
    ax = fig.add_subplot(gs[0, 0])
    names = list(bands.keys()); accs = [res[n]['lin']['balacc'] for n in names]
    nm = [res[n]['lin']['null_mean'] for n in names]; ns = [res[n]['lin']['null_std'] for n in names]
    cols = ['#2563d6'] + ['#b2182b'] * 5 + ['#8e44ad', '#333']
    x = np.arange(len(names)); ax.bar(x, accs, color=cols)
    ax.errorbar(x, nm, yerr=np.array(ns) * 2, fmt='_', color='k', capsize=4, label='perm null ±2σ')
    ax.axhline(.5, ls='--', c='#444', lw=1); ax.set_ylim(.45, max(.75, max(accs) + .05))
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right'); ax.set_ylabel('balanced acc (GroupKFold-chunk)')
    ax.set_title('Per-band separability (linear) — is there signal above 16 kHz?')
    for i, a in enumerate(accs): ax.text(i, a + .006, f'{a:.2f}', ha='center', fontsize=8)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 1])
    sbn = ['0-16k(M22)', '16-100k(HIGH)', 'SB_low(0-16k)', 'SB_high(16-100k)', 'PSDlow+SBhigh']
    xl = np.arange(len(sbn)); w = .38
    lv = [res[n]['lin']['balacc'] for n in sbn]; nv = [res[n].get('nl', {}).get('balacc', np.nan) for n in sbn]
    ax.bar(xl - w/2, lv, w, label='linear', color='#e08a10')
    ax.bar(xl + w/2, nv, w, label='nonlinear (GBM)', color='#5aa02c')
    ax.axhline(.5, ls='--', c='#444', lw=1); ax.axhline(low_lin, ls=':', c='#2563d6', label=f'LOW linear ({low_lin:.2f})')
    ax.set_xticks(xl); ax.set_xticklabels(sbn, rotation=25, ha='right'); ax.set_ylim(.45, max(.75, max(lv + [low_lin]) + .05))
    ax.set_title(f'Sideband-sum & does HIGH add over LOW?  (Δ={add:+.3f})'); ax.legend(fontsize=7); ax.set_ylabel('balanced acc')

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(freq / 1e3, r2, lw=.6, c='#b2182b'); ax.axvline(16, c='#2563d6', ls='--', label='16 kHz (M22 cutoff)')
    ax.set_xlabel('kHz'); ax.set_ylabel('point-biserial r²'); ax.set_title('Per-frequency discriminability (0-100 kHz)'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    hi_a = SBH[Y == 0].mean(0).reshape(BB_BINS, TN); hi_b = SBH[Y == 1].mean(0).reshape(BB_BINS, TN)
    ax.imshow(np.hstack([hi_a, np.full((BB_BINS, 2), np.nan), hi_b]), origin='lower', aspect='auto', cmap='magma')
    ax.set_title(f'HIGH-harmonic sideband-sum baseband:  {A} | {B}'); ax.set_xticks([]); ax.set_ylabel('sideband freq')

    fig.suptitle(f'M26 high-frequency probe — {A} vs {B}: LOW(0-16k) lin={low_lin:.2f}  HIGH(16-100k) lin={high_lin:.2f}  '
                 f'HIGH adds Δ={add:+.3f}', fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUT, 'highfreq_probe.png'), dpi=145, bbox_inches='tight')
    print('[hf] wrote', os.path.join(OUT, 'highfreq_probe.png'), flush=True)

    print('\n==== HIGH-FREQ VERDICT ====', flush=True)
    print(f'LOW 0-16k  linear={low_lin:.3f} nonlinear={res["0-16k(M22)"]["nl"]["balacc"]:.3f}', flush=True)
    print(f'HIGH 16-100k linear={high_lin:.3f} nonlinear={res["16-100k(HIGH)"]["nl"]["balacc"]:.3f}', flush=True)
    print(f'HIGH sideband-sum linear={res["SB_high(16-100k)"]["lin"]["balacc"]:.3f} nl={res["SB_high(16-100k)"]["nl"]["balacc"]:.3f}', flush=True)
    print(f'LOW+HIGH vs LOW: Δ={add:+.3f}  (>0 and null-clearing ⇒ high freq ADDS real info)', flush=True)


if __name__ == '__main__':
    main()
