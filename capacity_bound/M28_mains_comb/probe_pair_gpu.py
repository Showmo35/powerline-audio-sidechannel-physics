#!/usr/bin/env python3
"""
probe_pair_gpu.py — GPU version of the M28 which/this probe (analysis #1).

Same experiment as probe_pair.py (same items, same framing, same baselines, same
chunk-grouped CV + permutation null) but:
  * features via the FFT-batched GPU demod (comb_gpu.harmonic_gram_gpu): ~0.01 s per
    window instead of ~2.15 s for the CPU per-harmonic time-domain loop (~200x), which
    makes it cheap to run the pair test over ALL ~1600 harmonics, not just K=200.
  * permutation null parallelized across CPU cores (n_jobs) — the CV stage is sklearn,
    GPU buys nothing there.

Framing is matched to probe_pair.py exactly: window = word + 2*PAD, Tukey flat centre
== the word, crop to the word, time-normalize to T (so duration is NOT a cue).

This is also an INDEPENDENT IMPLEMENTATION of the CPU run (time-domain demod vs FFT
demod) — agreement between them is a real check, disagreement means one is wrong.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy.signal import welch

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold

PROOT = '<REPO_ROOT>'
sys.path.insert(0, os.path.join(PROOT, 'M15_soundbar_melgen_word'))
import data_io as io
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from comb import estimate_mains
from comb_gpu import harmonic_gram_gpu, gram_features_gpu, CAP_SR

INDEX = os.path.join(PROOT, 'M20_powerline_word_classifier', 'word_index.json')
BIN = os.path.join(PROOT, 'Powerline_Data_Captures', 'soundbar_bin_captures')
WAV = os.path.join(PROOT, 'Powerline_Data_Captures', 'audio_chunks')
CTX, PAD, BAD_F0 = 0.05, 0.60, 61.035


def envelope(x, npts=50):
    fr = max(1, len(x) // 200)
    e = np.sqrt(np.convolve(np.asarray(x, np.float64) ** 2, np.ones(fr) / fr, 'same'))
    e = np.interp(np.linspace(0, 1, npts), np.linspace(0, 1, len(e)), e)
    return (e / (e.max() + 1e-9)).astype(np.float32)


def separability(X, y, groups, nperm=100, seed=0, njobs=-1):
    cv = GroupKFold(5)
    nc = min(40, X.shape[1], int(0.7 * X.shape[0]) - 1)
    clf = make_pipeline(StandardScaler(), PCA(nc, random_state=seed),
                        LogisticRegression(max_iter=2000))
    acc = float(cross_val_score(clf, X, y, groups=groups, cv=cv,
                                scoring='balanced_accuracy', n_jobs=njobs).mean())
    rng = np.random.RandomState(seed); nulls = []
    for _ in range(nperm):
        nulls.append(float(cross_val_score(clf, X, rng.permutation(y), groups=groups, cv=cv,
                                           scoring='balanced_accuracy', n_jobs=njobs).mean()))
    nulls = np.array(nulls)
    return {'balacc': acc, 'null_mean': float(nulls.mean()), 'null_std': float(nulls.std()),
            'p': float((np.sum(nulls >= acc) + 1) / (nperm + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, default=['which', 'this'])
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--K', type=int, default=1600, help='ALL harmonics by default')
    ap.add_argument('--bw', type=float, default=25.0)
    ap.add_argument('--T', type=int, default=24)
    ap.add_argument('--nperm', type=int, default=100)
    ap.add_argument('--tag', default='allharm')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    occ = json.load(open(INDEX)); rng = np.random.RandomState(0)
    items = []
    for lab, w in enumerate(args.pair):
        rows = list(occ.get(w, [])); rng.shuffle(rows)
        for ch, s, e in rows[:args.n]:
            items.append((ch, float(s), float(e), lab))
    rng.shuffle(items)
    print(f'[gpu] device={dev} pair={args.pair} n={len(items)} K={args.K} '
          f'(comb to {args.K*60/1000:.0f} kHz) T={args.T}', flush=True)

    f0c = {}
    for ch in sorted({it[0] for it in items}):
        f0c[ch] = estimate_mains(io.read_bin_window(f'{BIN}/{ch}.bin', 60., 30., CAP_SR), CAP_SR)
    f0s = np.array(list(f0c.values()))
    print(f'[gpu] mains {f0s.mean():.4f} +- {f0s.std():.4f} Hz over {len(f0c)} chunks '
          f'(M26 used {BAD_F0} -> {f0s.mean()-BAD_F0:+.3f} Hz error)', flush=True)

    F = {k: [] for k in ['harm_AM', 'harm_AM_PM', 'harm_AM_badf0', 'L3_envelope', 'L1_raw_psd']}
    y, groups = [], []
    t0 = time.time()
    for i, (ch, s, e, lab) in enumerate(items):
        lag = io.read_lag_ms(f'{BIN}/{ch}.lag') / 1000.0
        dur = (e - s) + 2 * CTX
        tot = dur + 2 * PAD
        cap_p = io.read_bin_window(f'{BIN}/{ch}.bin', s - CTX - PAD + lag, tot, CAP_SR)
        need = int(round(tot * CAP_SR))
        if len(cap_p) < need * 0.9:
            continue
        cap_p = np.pad(cap_p, (0, max(0, need - len(cap_p))))[:need]
        cap_p = cap_p / (np.std(cap_p) + 1e-8)
        cap = cap_p[int(PAD * CAP_SR): int((PAD + dur) * CAP_SR)]
        pf = PAD / tot                                    # Tukey flat centre == the word

        xb = torch.from_numpy(cap_p[None]).float().to(dev)
        H = harmonic_gram_gpu(xb, torch.tensor([f0c[ch]], device=dev), CAP_SR,
                              K=args.K, bw=args.bw, T=args.T, pad_frac=pf)
        G = gram_features_gpu(H)[0].cpu().numpy()          # (2,K,T)
        Hb = harmonic_gram_gpu(xb, torch.tensor([BAD_F0], device=dev), CAP_SR,
                               K=args.K, bw=args.bw, T=args.T, pad_frac=pf)
        Gb = gram_features_gpu(Hb)[0].cpu().numpy()

        F['harm_AM'].append(G[0].ravel())
        F['harm_AM_PM'].append(G.ravel())
        F['harm_AM_badf0'].append(Gb[0].ravel())
        F['L3_envelope'].append(envelope(cap, 50))
        _, P = welch(cap, fs=CAP_SR, nperseg=4096)
        F['L1_raw_psd'].append(np.log(P + 1e-12).astype(np.float32))
        y.append(lab); groups.append(ch)
        if i % 100 == 0:
            print(f'[gpu] {i}/{len(items)}  {time.time()-t0:.0f}s', flush=True)
    print(f'[gpu] features built in {time.time()-t0:.0f}s '
          f'(CPU path was ~2.15 s/item = {2.15*len(items)/60:.0f} min)', flush=True)

    y = np.array(y); groups = np.array(groups)
    for k in F:
        F[k] = np.stack(F[k]).astype(np.float32)
    print('[gpu] dims: ' + ', '.join(f'{k}={F[k].shape[1]}' for k in F), flush=True)

    print(f'\n══ which/this — ALL {args.K} harmonics, GroupKFold by chunk, {args.nperm} perms ══',
          flush=True)
    res = {}
    for k in ['harm_AM', 'harm_AM_PM', 'harm_AM_badf0', 'L3_envelope', 'L1_raw_psd']:
        r = separability(F[k], y, groups, nperm=args.nperm)
        res[k] = r
        print(f'  {k:16s} balacc={r["balacc"]:.3f}  null={r["null_mean"]:.3f}'
              f'±{r["null_std"]:.3f}  p={r["p"]:.3f}', flush=True)
    print('  refs: M26 (leaky StratifiedKFold) L2_demod=0.638 L3_env=0.585 L1=0.530',
          flush=True)

    json.dump({'pair': args.pair, 'n': int(len(y)), 'K': args.K, 'T': args.T,
               'mains_mean': float(f0s.mean()), 'gateB': res},
              open(os.path.join(HERE, 'outputs', f'm28_pair_gpu_{args.tag}.json'), 'w'), indent=2)
    print(f'[gpu] wrote outputs/m28_pair_gpu_{args.tag}.json', flush=True)


if __name__ == '__main__':
    main()
