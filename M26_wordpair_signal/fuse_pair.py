#!/usr/bin/env python3
"""
M26-fusion — does COMBINING complementary front-ends beat any single one?

For the confused pair (which/this) we build three attack-usable feature blocks
from the SAME raw 200 kHz per-word window and test 2-class separability of each
alone vs. their fusion, same protocol as analyze_pair (5-fold logistic balanced
accuracy + 100-perm null):

  GEN : M22 powerline->mel generated word-mel (nonlinear learned view)
  L2  : mains-harmonic AM-demod baseband (demodulation-based discriminative front-end)
  ENV : loudness envelope contour

Reference bars: L0 clean-mel ceiling and L1 raw-200k PSD (from analyze_pair helpers).

Blocks are pre-embedded (StandardScaler+PCA, unsupervised) then scored alone and
concatenated; fusion > max(single) ⇒ the views carry independent evidence.

Output: M26_wordpair_signal/<A>_<B>/fusion_results.json + fuse_pair.png
"""
import os, sys, json, argparse, time, importlib
import numpy as np
from scipy import signal as dsp
from scipy.signal import resample_poly
import torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

import analyze_pair as AP          # reuse readers, detect_mains, demod_baseband, envelope, clean_mel_T, separability

PROOT = AP.PROOT
M22   = os.path.join(PROOT, 'M22_retrieval_generator')
CAP_SR = AP.CAP_SR; AUD_SR = AP.AUD_SR; CTX = AP.CTX
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_m22():
    for m in ('config', 'models'):
        sys.modules.pop(m, None)
    sys.path.insert(0, M22)
    cfg = importlib.import_module('config').CFG
    Mm = importlib.import_module('models')
    model = Mm.build(cfg).to(dev)
    ck = torch.load(os.path.join(M22, 'outputs/best.pt'), map_location=dev, weights_only=False)
    model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
    print(f'[m22] loaded ({model.count_params()/1e6:.1f}M) in_sr={cfg.in_sr} win_s={cfg.win_s} fps={cfg.fps}', flush=True)
    return model, cfg, ck['mel_mean'], ck['mel_std']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, default=['which', 'this'])
    ap.add_argument('--n', type=int, default=250)
    ap.add_argument('--nh', type=int, default=20)
    ap.add_argument('--kpca', type=int, default=25, help='per-block PCA components')
    args = ap.parse_args()
    A, B = args.pair
    OUT = os.path.join(PROOT, 'M26_wordpair_signal', f'{A}_{B}'); os.makedirs(OUT, exist_ok=True)
    occ = json.load(open(AP.INDEX)); rng = np.random.RandomState(0)

    m22, c22, mm, msd = load_m22()
    Lg = int(c22.win_s * CAP_SR)                      # 4 s raw window for the generator

    # mains harmonics for L2 (from a probe capture)
    probe = AP.io.read_bin_window(AP.bin_path(occ[A][0][0]), 1.0, 4.0, CAP_SR).astype(np.float32)
    fmains = AP.detect_mains(probe, CAP_SR)
    harmonics = AP.select_harmonics(fmains, CAP_SR, nh=args.nh, probe=probe)
    print(f'[fuse] f_mains={fmains:.2f}  nh={len(harmonics)}', flush=True)

    def is_test(ch):                       # M22 held-out split: chunk % test_every == 0
        return int(ch.split('_')[1]) % c22.test_every == 0

    GEN, L2, ENV, L0, L1, Y = [], [], [], [], [], []
    lagcache = {}
    gen_inputs = []            # 32k RMS-normed 4s inputs for batched M22 sampling
    gen_frs = []               # word frame count for cropping
    t0 = time.time()
    for lab, word in [(0, A), (1, B)]:
        rows = [r for r in occ[word] if is_test(r[0])]    # TEST chunks only — never seen by M22
        rng.shuffle(rows); rows = rows[:args.n]; kept = 0
        print(f'[fuse] {word}: {len(rows)} TEST occurrences available', flush=True)
        for ch, s, e in rows:
            if ch not in lagcache:
                lagcache[ch] = AP.io.read_lag_ms(AP.lag_path(ch)) / 1000.0
            lag = lagcache[ch]; dur = (e - s) + 2 * CTX
            # short window (L2/ENV/L0/L1) + long 4s window (generator), both lag-aligned
            cap = AP.io.read_bin_window(AP.bin_path(ch), s - CTX + lag, dur, CAP_SR).astype(np.float32)
            aud = AP.io.read_wav_window(AP.wav_path(ch), s - CTX, dur, AUD_SR).astype(np.float32)
            capL = AP.io.read_bin_window(AP.bin_path(ch), s - CTX + lag, c22.win_s, CAP_SR).astype(np.float32)
            if len(cap) < CAP_SR * 0.08 or len(aud) < AUD_SR * 0.08:
                continue
            cap = cap / (np.sqrt(np.mean(cap ** 2)) + 1e-8)
            L0.append(AP.clean_mel_T(aud).ravel())
            f, P = dsp.welch(cap, fs=CAP_SR, nperseg=4096); L1.append(np.log(P + 1e-12).astype(np.float32))
            L2.append(np.concatenate([AP.demod_baseband(cap, fc, CAP_SR).ravel() for fc in harmonics]))
            ENV.append(AP.envelope(resample_poly(cap, AUD_SR, CAP_SR).astype(np.float32)))
            # generator input: 4s raw -> 32k, RMS-norm, pad
            capL = np.pad(capL, (0, max(0, Lg - len(capL))))[:Lg]
            r = resample_poly(capL, c22.in_sr, CAP_SR).astype(np.float32)
            r = r / (np.sqrt(np.mean(r ** 2)) + 1e-8)
            gen_inputs.append(r); gen_frs.append(max(6, int(round(dur * c22.fps))))
            Y.append(lab); kept += 1
        print(f'[fuse] {word}: kept {kept}  ({time.time()-t0:.0f}s)', flush=True)

    Y = np.array(Y)
    # ── batched M22 generation -> per-word mel ──
    T = 32
    def to_T(m):
        m = torch.nn.functional.interpolate(m[None, None], size=(80, T), mode='bilinear', align_corners=False)[0, 0]
        return m.numpy().astype(np.float32)
    BS = 16
    for b in range(0, len(gen_inputs), BS):
        X = torch.from_numpy(np.stack(gen_inputs[b:b + BS])).to(dev)
        torch.manual_seed(0)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
            g = m22.sample(X)
        g = (g.float().cpu() * msd + mm)
        for j in range(g.shape[0]):
            GEN.append(to_T(g[j, :, :gen_frs[b + j]]).ravel())
        if b % (BS * 8) == 0:
            print(f'[gen] {b+g.shape[0]}/{len(gen_inputs)}', flush=True)

    GEN = np.array(GEN); L2 = np.array(L2); ENV = np.array(ENV); L0 = np.array(L0); L1 = np.array(L1)
    print(f'[fuse] blocks GEN{GEN.shape} L2{L2.shape} ENV{ENV.shape}  n={len(Y)}', flush=True)

    # ── per-block unsupervised embedding (StandardScaler+PCA), then score ──
    def embed(X):
        k = min(args.kpca, X.shape[1], X.shape[0] - 1)
        return make_pipeline(StandardScaler(), PCA(k, random_state=0)).fit_transform(X)
    Egen, El2, Eenv = embed(GEN), embed(L2), embed(ENV)

    blocks = {
        'GEN_m22': Egen, 'L2_demod': El2, 'ENV': Eenv,
        'GEN+L2': np.hstack([Egen, El2]),
        'L2+ENV': np.hstack([El2, Eenv]),
        'GEN+ENV': np.hstack([Egen, Eenv]),
        'FUSE_all': np.hstack([Egen, El2, Eenv]),
    }
    res = {}
    for name, X in blocks.items():
        res[name] = AP.separability(X, Y, nperm=100)
        r = res[name]
        print(f'[sep] {name:10s} balacc={r["balacc"]:.3f}  null={r["null_mean"]:.3f}±{r["null_std"]:.3f}  p={r["p"]:.3g}', flush=True)
    # reference bars
    res['L0_clean'] = AP.separability(L0, Y, nperm=50)
    res['L1_raw200k'] = AP.separability(L1, Y, nperm=50)
    print(f'[ref] L0_clean={res["L0_clean"]["balacc"]:.3f}  L1_raw200k={res["L1_raw200k"]["balacc"]:.3f}', flush=True)

    best_single = max(res['GEN_m22']['balacc'], res['L2_demod']['balacc'], res['ENV']['balacc'])
    out = {'pair': [A, B], 'n': int(len(Y)), 'f_mains': fmains,
           'separability': res, 'best_single': best_single,
           'fusion_gain': res['FUSE_all']['balacc'] - best_single}
    json.dump(out, open(os.path.join(OUT, 'fusion_results.json'), 'w'), indent=1)

    # ── figure ──
    order = ['L1_raw200k', 'ENV', 'GEN_m22', 'L2_demod', 'GEN+ENV', 'L2+ENV', 'GEN+L2', 'FUSE_all', 'L0_clean']
    accs = [res[k]['balacc'] for k in order]
    nulm = [res[k]['null_mean'] for k in order]; nuls = [res[k]['null_std'] for k in order]
    cols = ['#999', '#777', '#5aa02c', '#b2182b', '#8888cc', '#cc8844', '#44aa88', '#e08a10', '#2563d6']
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    ax.bar(x, accs, color=cols)
    ax.errorbar(x, nulm, yerr=np.array(nuls) * 2, fmt='_', color='k', capsize=4, label='perm null ±2σ')
    ax.axhline(0.5, ls='--', c='#444', lw=1)
    ax.axhline(best_single, ls=':', c='#b2182b', lw=1, label=f'best single ({best_single:.2f})')
    ax.set_ylim(0.45, 1.0); ax.set_xticks(x); ax.set_xticklabels(order, rotation=25, ha='right')
    ax.set_ylabel('2-class balanced accuracy')
    ax.set_title(f'M26 fusion — {A} vs {B}: does combining GEN + L2-demod + ENV beat any single front-end?\n'
                 f'(fusion {res["FUSE_all"]["balacc"]:.3f} vs best single {best_single:.3f}; ceiling L0={res["L0_clean"]["balacc"]:.2f})')
    for i, a in enumerate(accs): ax.text(i, a + .008, f'{a:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fuse_pair.png'), dpi=150, bbox_inches='tight')
    print('[fuse] wrote', os.path.join(OUT, 'fuse_pair.png'), flush=True)

    print('\n==== M26 FUSION VERDICT ====', flush=True)
    for k in order:
        print(f'  {k:12s} {res[k]["balacc"]:.3f}', flush=True)
    print(f'  best single = {best_single:.3f} ; FUSE_all = {res["FUSE_all"]["balacc"]:.3f} ; '
          f'gain = {out["fusion_gain"]:+.3f}', flush=True)


if __name__ == '__main__':
    main()
