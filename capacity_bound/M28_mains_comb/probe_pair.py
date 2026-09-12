#!/usr/bin/env python3
"""
probe_pair.py — M28 Gate A (rank/redundancy) + Gate B (which/this separability).

Gate A  Is the per-harmonic AM gram RANK-1?  If a_k(t) = c_k*env(t) then after
        per-harmonic mean removal every row equals log env(t) and the gram is rank-1:
        the K harmonics are redundant copies of one envelope => idea dead (clean
        negative). If rank > 1 with structure, harmonics carry independent info.

Gate B  which vs this (duration-matched, the M26 pair), chunk-grouped GroupKFold
        (NOT the leaky StratifiedKFold that produced the published 0.985/0.53/0.64/0.585
        ladder), permutation null. Conditions:
          harm_AM        coherent per-harmonic AM, CORRECT 60.00 Hz grid
          harm_AM_PM     + per-harmonic phase (never measured before anywhere)
          harm_AM_badf0  same, on M26's WRONG 61.035 Hz grid  <- quantifies the bug
          L3_envelope    loudness contour control  (must be beaten to mean anything)
          L1_raw_psd     broadband Welch PSD
        All re-run under the SAME CV so the comparison is apples-to-apples.

Outputs -> outputs/m28_results.json + outputs/m28_gates.png
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.signal import welch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold

PROOT = '<REPO_ROOT>'
sys.path.insert(0, os.path.join(PROOT, 'M15_soundbar_melgen_word'))
import data_io as io                                  # read_bin_window/read_wav_window/read_lag_ms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from comb import estimate_mains, harmonic_gram, gram_features, harmonic_snr, CAP_SR

INDEX = os.path.join(PROOT, 'M20_powerline_word_classifier', 'word_index.json')
BIN = os.path.join(PROOT, 'Powerline_Data_Captures', 'soundbar_bin_captures')
WAV = os.path.join(PROOT, 'Powerline_Data_Captures', 'audio_chunks')
CTX = 0.05
PAD = 0.60               # seconds of padding so the 25 Hz low-pass settles before cropping
BAD_F0 = 61.035          # the M26 bin-snapped estimate


def bin_path(c): return os.path.join(BIN, f'{c}.bin')
def wav_path(c): return os.path.join(WAV, f'{c}.wav')
def lag_path(c): return os.path.join(BIN, f'{c}.lag')


def envelope(x, npts=50):
    """Peak-normed RMS loudness contour, time-normalized (matches M26 AP.envelope)."""
    fr = max(1, len(x) // 200)
    e = np.sqrt(np.convolve(np.asarray(x, np.float64) ** 2, np.ones(fr) / fr, 'same'))
    e = np.interp(np.linspace(0, 1, npts), np.linspace(0, 1, len(e)), e)
    return (e / (e.max() + 1e-9)).astype(np.float32)


def separability(X, y, groups, nperm=50, seed=0):
    """Chunk-grouped CV + permutation null (the honest version)."""
    cv = GroupKFold(5)
    nc = min(40, X.shape[1], int(0.7 * X.shape[0]) - 1)
    clf = make_pipeline(StandardScaler(), PCA(nc, random_state=seed),
                        LogisticRegression(max_iter=2000))
    acc = float(cross_val_score(clf, X, y, groups=groups, cv=cv,
                                scoring='balanced_accuracy').mean())
    rng = np.random.RandomState(seed); nulls = []
    for _ in range(nperm):
        nulls.append(float(cross_val_score(clf, X, rng.permutation(y), groups=groups,
                                           cv=cv, scoring='balanced_accuracy').mean()))
    nulls = np.array(nulls)
    return {'balacc': acc, 'null_mean': float(nulls.mean()), 'null_std': float(nulls.std()),
            'p': float((np.sum(nulls >= acc) + 1) / (nperm + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, default=['which', 'this'])
    ap.add_argument('--n', type=int, default=300, help='occurrences per word')
    ap.add_argument('--K', type=int, default=200, help='harmonics')
    ap.add_argument('--bw', type=float, default=25.0, help='demod half-bandwidth (<30)')
    ap.add_argument('--T', type=int, default=24)
    ap.add_argument('--nperm', type=int, default=100)
    args = ap.parse_args()

    occ = json.load(open(INDEX))
    rng = np.random.RandomState(0)
    items = []
    for lab, w in enumerate(args.pair):
        rows = [r for r in occ.get(w, [])]
        rng.shuffle(rows)
        for ch, s, e in rows[:args.n]:
            items.append((ch, float(s), float(e), lab))
    rng.shuffle(items)
    print(f'[m28] pair={args.pair} items={len(items)} K={args.K} bw={args.bw} T={args.T}',
          flush=True)

    # ── per-chunk precise mains (cache) ──
    f0c, snr_ref = {}, None
    chunks = sorted({it[0] for it in items})
    for ch in chunks:
        probe = io.read_bin_window(bin_path(ch), 60.0, 30.0, CAP_SR)
        f0c[ch] = estimate_mains(probe, CAP_SR)
        if snr_ref is None:
            snr_ref = harmonic_snr(probe[:CAP_SR * 4], f0c[ch], CAP_SR, K=args.K, bw=args.bw)
    f0s = np.array([f0c[c] for c in chunks])
    print(f'[m28] mains over {len(chunks)} chunks: mean={f0s.mean():.4f} Hz '
          f'std={f0s.std():.4f}  (M26 used {BAD_F0}) -> error {f0s.mean()-BAD_F0:+.3f} Hz',
          flush=True)

    # ── build features ──
    F = {k: [] for k in ['harm_AM', 'harm_AM_PM', 'harm_AM_badf0', 'L3_envelope', 'L1_raw_psd']}
    y, groups, am_grams, aud_envs = [], [], [], []
    t0 = time.time()
    for i, (ch, s, e, lab) in enumerate(items):
        lag = io.read_lag_ms(lag_path(ch)) / 1000.0
        dur = (e - s) + 2 * CTX
        # PAD the capture so the narrow 25 Hz low-pass can settle, then crop the word.
        cap_p = io.read_bin_window(bin_path(ch), s - CTX - PAD + lag, dur + 2 * PAD, CAP_SR)
        if len(cap_p) < CAP_SR * 0.1:
            continue
        cap_p = cap_p / (np.sqrt(np.mean(cap_p ** 2)) + 1e-8)
        cap = cap_p[int(PAD * CAP_SR): int((PAD + dur) * CAP_SR)]   # word-only (baselines)
        aud = io.read_wav_window(wav_path(ch), s - CTX, dur, 16_000)

        crop = (PAD, PAD + dur)
        H, _ = harmonic_gram(cap_p, f0c[ch], CAP_SR, K=args.K, bw=args.bw, T=args.T, crop=crop)
        AM = gram_features(H, use_phase=False)
        AMPM = gram_features(H, use_phase=True)
        Hb, _ = harmonic_gram(cap_p, BAD_F0, CAP_SR, K=args.K, bw=args.bw, T=args.T, crop=crop)
        AMb = gram_features(Hb, use_phase=False)

        F['harm_AM'].append(AM.ravel())
        F['harm_AM_PM'].append(AMPM.ravel())
        F['harm_AM_badf0'].append(AMb.ravel())
        F['L3_envelope'].append(envelope(cap, 50))
        fq, P = welch(cap, fs=CAP_SR, nperseg=4096)
        F['L1_raw_psd'].append(np.log(P + 1e-12).astype(np.float32))
        am_grams.append(AM); aud_envs.append(envelope(aud, args.T))
        y.append(lab); groups.append(ch)
        if i % 100 == 0:
            print(f'[m28] {i}/{len(items)}  {time.time()-t0:.0f}s', flush=True)

    y = np.array(y); groups = np.array(groups)
    for k in F:
        F[k] = np.stack(F[k]).astype(np.float32)
    print(f'[m28] built. dims: ' + ', '.join(f'{k}={F[k].shape[1]}' for k in F), flush=True)

    # ── demod sanity: does EACH harmonic's AM track the AUDIO envelope? ──
    # NOTE: do NOT average AM across harmonics first — harmonics track the envelope with
    # DIFFERENT (and sometimes OPPOSITE) signs, so the mean cancels to ~0 and would look
    # like a broken demod. Score per harmonic, then summarize.
    C = np.zeros((len(am_grams), am_grams[0].shape[0]))
    for i, (A, ae) in enumerate(zip(am_grams, aud_envs)):
        b = (ae - ae.mean()) / (ae.std() + 1e-9)
        Az = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
        C[i] = (Az * b[None, :]).mean(1)
    per_k = C.mean(0)                                   # mean signed corr per harmonic
    print(f'[m28] SANITY per-harmonic corr(AM_k, audio env): '
          f'mean|r|={np.abs(C).mean():.3f}  best-k r={per_k[np.argmax(np.abs(per_k))]:+.3f} '
          f'(k={int(np.argmax(np.abs(per_k)))+1})  frac|r|>0.2: {np.mean(np.abs(per_k)>0.2):.2f}',
          flush=True)
    print(f'[m28] sign spread across k: {np.mean(per_k>0.05):.2f} positive, '
          f'{np.mean(per_k<-0.05):.2f} negative  <- rank-1 predicts ALL same sign',
          flush=True)

    # ── GATE A: rank / redundancy of the AM gram ──
    ev1, ev12, pc1_env = [], [], []
    for A, ae in zip(am_grams, aud_envs):
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        v = S ** 2 / (np.sum(S ** 2) + 1e-30)
        ev1.append(v[0]); ev12.append(v[:2].sum())
        p = Vt[0]; p = (p - p.mean()) / (p.std() + 1e-9)
        b = (ae - ae.mean()) / (ae.std() + 1e-9)
        pc1_env.append(abs(float((p * b).mean())))
    ev1 = np.array(ev1); pc1_env = np.array(pc1_env)
    print('\n══ GATE A — rank / redundancy of the per-harmonic AM gram ══')
    print(f'  PC1 explained variance : {ev1.mean():.3f} +- {ev1.std():.3f}')
    print(f'  PC1+PC2                : {np.mean(ev12):.3f}')
    print(f'  |corr(PC1, audio env)| : {pc1_env.mean():.3f}')
    print(f'  -> rank-1 (redundant copies of one envelope)? '
          f'{"YES (idea dead)" if ev1.mean() > 0.95 else "NO - harmonics carry independent dims"}')

    # ── GATE B: separability ──
    print('\n══ GATE B — which/this separability (GroupKFold by chunk) ══', flush=True)
    res = {}
    for k in ['harm_AM', 'harm_AM_PM', 'harm_AM_badf0', 'L3_envelope', 'L1_raw_psd']:
        r = separability(F[k], y, groups, nperm=args.nperm)
        res[k] = r
        print(f'  {k:16s} balacc={r["balacc"]:.3f}  null={r["null_mean"]:.3f}'
              f'±{r["null_std"]:.3f}  p={r["p"]:.3f}', flush=True)

    out = {'pair': args.pair, 'n': int(len(y)), 'K': args.K, 'bw': args.bw, 'T': args.T,
           'mains_mean': float(f0s.mean()), 'mains_std': float(f0s.std()), 'bad_f0': BAD_F0,
           'sanity': {'mean_abs_r_per_harmonic': float(np.abs(C).mean()),
                      'best_k_r': float(per_k[np.argmax(np.abs(per_k))]),
                      'best_k': int(np.argmax(np.abs(per_k))) + 1,
                      'frac_pos': float(np.mean(per_k > 0.05)),
                      'frac_neg': float(np.mean(per_k < -0.05)),
                      'per_k_r': per_k.tolist()},
           'gateA': {'pc1_explained_var': float(ev1.mean()), 'pc1_pc2': float(np.mean(ev12)),
                     'corr_pc1_env': float(pc1_env.mean())},
           'gateB': res,
           'refs_M26_stratifiedKFold_leaky': {'L0_clean': 0.985, 'L1_raw': 0.530,
                                              'L2_demod': 0.638, 'L3_env': 0.585}}
    json.dump(out, open(os.path.join(HERE, 'outputs', 'm28_results.json'), 'w'), indent=2)

    # ── figure ──
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.2))
    kk = np.arange(1, len(snr_ref) + 1)
    ax0.plot(kk, snr_ref, lw=1.2, color='#2166ac')
    ax0.set_xlabel('harmonic k'); ax0.set_ylabel('carrier SNR vs off-comb floor (dB)')
    ax0.set_title(f'Per-harmonic SNR (f0={f0s.mean():.4f} Hz)', fontsize=10)
    ax0.spines[['top', 'right']].set_visible(False)

    ax1.hist(ev1, bins=30, color='#1b7837', alpha=0.8)
    ax1.axvline(ev1.mean(), color='k', ls='--')
    ax1.set_xlabel('PC1 explained variance of AM gram'); ax1.set_ylabel('# words')
    ax1.set_title(f'Gate A: rank test (mean {ev1.mean():.3f})\n1.0 = redundant copies of one envelope',
                  fontsize=10)
    ax1.spines[['top', 'right']].set_visible(False)

    names = ['L1_raw_psd', 'L3_envelope', 'harm_AM_badf0', 'harm_AM', 'harm_AM_PM']
    vals = [res[n]['balacc'] for n in names]
    cols = ['#999999', '#7a7a7a', '#d6604d', '#4393c3', '#1b7837']
    for b, v in zip(ax2.bar(range(len(names)), vals, color=cols, width=0.66), vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.004, f'{v:.3f}', ha='center',
                 va='bottom', fontsize=10, fontweight='bold')
    ax2.axhline(0.5, color='k', lw=0.8, ls=':')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(['raw PSD', 'envelope', 'harm AM\n(bad f0)', 'harm AM\n(true f0)',
                         'harm AM+PM'], fontsize=8)
    ax2.set_ylabel('balanced accuracy'); ax2.set_ylim(0.45, max(0.75, max(vals) * 1.15))
    ax2.set_title(f'Gate B: {args.pair[0]} vs {args.pair[1]} (GroupKFold, n={len(y)})', fontsize=10)
    ax2.spines[['top', 'right']].set_visible(False)

    fig.suptitle('M28: line-locked per-harmonic mains-comb probe', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'outputs', 'm28_gates.png'), dpi=200, bbox_inches='tight')
    print('\n[m28] wrote outputs/m28_gates.png + m28_results.json', flush=True)


if __name__ == '__main__':
    main()
