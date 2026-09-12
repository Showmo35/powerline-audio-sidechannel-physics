#!/usr/bin/env python3
"""
M26 deep high-frequency probe — state-of-the-art learned test of whether the
>16 kHz band (discarded by M22) separates 'which' vs 'this'.

Not PSD. Each word window is turned into a HIGH-RESOLUTION, PHASE-PRESERVING complex
STFT (real+imag as 2 channels) and a small 2-D CNN is TRAINED discriminatively to
classify the pair. The CNN gets the high band the same learned-feature advantage M22
gives the low band. Representations compared (identical CNN, identical protocol):

  HIGH   16-100 kHz band of the 200 kHz capture   <- the discarded band (the test)
  LOW    0-16   kHz band  (what M22 uses)          <- reference
  CLEAN  16 kHz reference-audio STFT               <- ceiling

Rigor:
  * GroupKFold(5) BY CHUNK -> train and test on disjoint recordings, so nothing can
    be won on per-recording high-freq artifacts; any accuracy must generalize.
  * label-shuffle null (retrain on permuted labels) -> confirms the pipeline sits at
    chance when there is no signal.
  * instance normalization (per-sample) -> no cross-sample leakage.

Output: M26_wordpair_signal/<A>_<B>/deep_highfreq.png + deep_highfreq_results.json
"""
import os, sys, json, argparse, time
import numpy as np
from scipy import signal as dsp
import torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

import analyze_pair as AP
CAP_SR = AP.CAP_SR; AUD_SR = AP.AUD_SR; CTX = AP.CTX
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
TN = 64                       # time frames after normalization


def cstft(x, sr, nper, hop, flo, fhi):
    f, t, Z = dsp.stft(x, fs=sr, nperseg=nper, noverlap=nper - hop, window='hann')
    m = (f >= flo) & (f < fhi)
    Z = Z[m]                                      # (Fb, T) complex
    ri = np.stack([Z.real, Z.imag]).astype(np.float32)          # (2, Fb, T)
    # time-normalize to TN
    ri = torch.nn.functional.interpolate(torch.from_numpy(ri)[None], size=(ri.shape[1], TN),
                                         mode='bilinear', align_corners=False)[0].numpy()
    # per-sample instance norm (leakage-free)
    mu = ri.mean(axis=(1, 2), keepdims=True); sd = ri.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((ri - mu) / sd).astype(np.float32)                  # (2, Fb, TN)


class CNN(nn.Module):
    def __init__(self, fb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=(2, 1), padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.drop = nn.Dropout(0.3); self.fc = nn.Linear(128, 2)

    def forward(self, x):
        return self.fc(self.drop(self.net(x).flatten(1)))


def train_eval(X, y, groups, epochs=40, seed=0):
    """GroupKFold(5) balanced accuracy; returns mean over folds."""
    torch.manual_seed(seed); np.random.seed(seed)
    gkf = GroupKFold(5); accs = []
    for tr, te in gkf.split(X, y, groups):
        Xtr = torch.from_numpy(X[tr]).to(dev); ytr = torch.from_numpy(y[tr]).long().to(dev)
        Xte = torch.from_numpy(X[te]).to(dev); yte = y[te]
        net = CNN(X.shape[2]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        n = len(tr); bs = 64
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n)
            for b in range(0, n, bs):
                idx = perm[b:b + bs]
                opt.zero_grad()
                out = net(Xtr[idx]); loss = lossf(out, ytr[idx])
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(Xte).argmax(1).cpu().numpy()
        # balanced accuracy
        acc = 0.5 * (((pred == 1) & (yte == 1)).sum() / max(1, (yte == 1).sum()) +
                     ((pred == 0) & (yte == 0)).sum() / max(1, (yte == 0).sum()))
        accs.append(float(acc))
    return float(np.mean(accs)), accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, default=['which', 'this'])
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--nnull', type=int, default=2)
    args = ap.parse_args()
    A, B = args.pair
    OUT = os.path.join(AP.PROOT, 'M26_wordpair_signal', f'{A}_{B}'); os.makedirs(OUT, exist_ok=True)
    occ = json.load(open(AP.INDEX)); rng = np.random.RandomState(0)

    HI, LO, CL, Y, G = [], [], [], [], []
    lagcache = {}; t0 = time.time()
    for lab, word in [(0, A), (1, B)]:
        rows = occ[word][:]; rng.shuffle(rows); rows = rows[:args.n]; kept = 0
        for ch, s, e in rows:
            if ch not in lagcache:
                lagcache[ch] = AP.io.read_lag_ms(AP.lag_path(ch)) / 1000.0
            dur = (e - s) + 2 * CTX
            cap = AP.io.read_bin_window(AP.bin_path(ch), s - CTX + lagcache[ch], dur, CAP_SR).astype(np.float32)
            aud = AP.io.read_wav_window(AP.wav_path(ch), s - CTX, dur, AUD_SR).astype(np.float32)
            if len(cap) < CAP_SR * 0.10 or len(aud) < AUD_SR * 0.10:
                continue
            cap = cap / (np.sqrt(np.mean(cap ** 2)) + 1e-8)
            HI.append(cstft(cap, CAP_SR, 1024, 256, 16_000, 100_000))     # (2, ~430, 64)
            LO.append(cstft(cap, CAP_SR, 1024, 256, 0, 16_000))           # (2, ~82, 64)
            CL.append(cstft(aud, AUD_SR, 512, 128, 0, 8_000))             # (2, ~257, 64)
            Y.append(lab); G.append(int(ch.split('_')[1])); kept += 1
        print(f'[deep] {word}: kept {kept}  ({time.time()-t0:.0f}s)', flush=True)

    HI = np.array(HI); LO = np.array(LO); CL = np.array(CL); Y = np.array(Y); G = np.array(G)
    print(f'[deep] HI{HI.shape} LO{LO.shape} CL{CL.shape}  n={len(Y)} chunks={len(set(G))}', flush=True)

    reps = {'HIGH_16-100k': HI, 'LOW_0-16k': LO, 'CLEAN_audio': CL}
    res = {}
    for name, X in reps.items():
        acc, folds = train_eval(X, Y, G, epochs=args.epochs)
        # label-shuffle null
        nulls = []
        for s in range(args.nnull):
            yp = np.random.RandomState(s).permutation(Y)
            na, _ = train_eval(X, yp, G, epochs=args.epochs, seed=s + 1)
            nulls.append(na)
        res[name] = {'balacc': acc, 'folds': folds, 'null': nulls, 'null_mean': float(np.mean(nulls))}
        print(f'[deep] {name:14s} balacc={acc:.3f}  folds={[round(f,2) for f in folds]}  '
              f'null={[round(x,2) for x in nulls]}', flush=True)

    hi = res['HIGH_16-100k']['balacc']; lo = res['LOW_0-16k']['balacc']
    hi_null = res['HIGH_16-100k']['null_mean']
    out = {'pair': [A, B], 'n': int(len(Y)), 'results': res,
           'HIGH_above_null': hi - hi_null}
    json.dump(out, open(os.path.join(OUT, 'deep_highfreq_results.json'), 'w'), indent=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(reps.keys()); accs = [res[n]['balacc'] for n in names]
    nullm = [res[n]['null_mean'] for n in names]
    cols = ['#b2182b', '#e08a10', '#2563d6']
    x = np.arange(len(names))
    ax.bar(x, accs, color=cols, width=0.6)
    ax.scatter(x, nullm, color='k', marker='_', s=400, zorder=5, label='label-shuffle null')
    for n_i, nm in enumerate(names):
        for f in res[nm]['folds']:
            ax.scatter(n_i, f, color='k', s=12, alpha=0.4, zorder=6)
    ax.axhline(0.5, ls='--', c='#444', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('balanced accuracy (GroupKFold by chunk)')
    ax.set_title(f'Deep learned probe — {A} vs {B}\n'
                 f'complex high-res STFT + 2-D CNN · HIGH={hi:.2f} (null {hi_null:.2f})  LOW={lo:.2f}  '
                 f'CLEAN={res["CLEAN_audio"]["balacc"]:.2f}')
    for i, a in enumerate(accs): ax.text(i, a + .01, f'{a:.2f}', ha='center', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'deep_highfreq.png'), dpi=150, bbox_inches='tight')
    print('[deep] wrote', os.path.join(OUT, 'deep_highfreq.png'), flush=True)

    print('\n==== DEEP HIGH-FREQ VERDICT ====', flush=True)
    print(f'HIGH 16-100k : {hi:.3f}  (label-shuffle null {hi_null:.3f})', flush=True)
    print(f'LOW  0-16k   : {lo:.3f}', flush=True)
    print(f'CLEAN audio  : {res["CLEAN_audio"]["balacc"]:.3f}  (ceiling)', flush=True)
    print(f'HIGH above its null = {hi-hi_null:+.3f}  '
          f'({">0 and clear => high-freq carries real signal" if hi-hi_null > 0.03 else "~0 => no high-freq signal"})', flush=True)


if __name__ == '__main__':
    main()
