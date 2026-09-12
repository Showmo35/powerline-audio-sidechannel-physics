#!/usr/bin/env python3
"""
M26 classifier-strength ladder — is the ~0.78 real-mel ceiling (kway_scaling.png) a
METHOD artifact or an INFORMATION limit?

Hold the data fixed (freq-30 vocab, TEST-only occurrences → no M22 leak), strengthen
the readout, on BOTH real and M22-generated word-mels, K=30, GroupKFold-by-chunk:

  M0  flatten → PCA50 → logistic     (the kway baseline readout)
  M1  2-D CNN on the 80x32 mel       (nonlinear, timing-aware)
  M2  wav2vec2 features on REAL audio → logistic   (true information ceiling; real only)

Reading:
  real rises a lot M0->M1->M2  => the apparatus caps clean audio, not the audio itself.
  gen rises with a stronger readout => M22-gen has more usable content than 0.41.
  gen stays flat => gen is information-limited (0.41 is real).

Output: M26_wordpair_signal/classifier_ladder.png + classifier_ladder.json
"""
import os, sys, json, time
import numpy as np
from scipy.signal import resample_poly
import torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

import analyze_pair as AP
import fuse_pair as FP
CAP_SR = AP.CAP_SR; AUD_SR = AP.AUD_SR; CTX = AP.CTX
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
T = 32; CAP_PER_WORD = 120; MIN_PER_WORD = 30


def to_T(m):
    m = torch.nn.functional.interpolate(torch.as_tensor(m)[None, None].float(),
                                        size=(80, T), mode='bilinear', align_corners=False)[0, 0]
    return m.numpy().astype(np.float32)


# ── M0 linear ─────────────────────────────────────────────────────────────────
def acc_linear(X, y, g, K):
    Xf = X.reshape(len(X), -1)
    clf = make_pipeline(StandardScaler(), PCA(min(50, Xf.shape[1]), random_state=0),
                        LogisticRegression(max_iter=3000, multi_class='multinomial'))
    return float(cross_val_score(clf, Xf, y, groups=g, cv=GroupKFold(5), scoring='accuracy').mean())


# ── M1 2-D CNN ────────────────────────────────────────────────────────────────
class MelCNN(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.drop = nn.Dropout(0.3); self.fc = nn.Linear(128, K)

    def forward(self, x):
        return self.fc(self.drop(self.net(x).flatten(1)))


def acc_cnn(X, y, g, K, epochs=45):
    torch.manual_seed(0); np.random.seed(0)
    Xt = (X - X.mean()) / (X.std() + 1e-6)
    accs = []
    for tr, te in GroupKFold(5).split(Xt, y, g):
        xtr = torch.from_numpy(Xt[tr][:, None]).to(dev); ytr = torch.from_numpy(y[tr]).long().to(dev)
        xte = torch.from_numpy(Xt[te][:, None]).to(dev); yte = y[te]
        net = MelCNN(K).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        lf = nn.CrossEntropyLoss(); n = len(tr); bs = 128
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n)
            for b in range(0, n, bs):
                idx = perm[b:b + bs]; opt.zero_grad()
                lf(net(xtr[idx]), ytr[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(xte).argmax(1).cpu().numpy()
        accs.append((pred == yte).mean())
    return float(np.mean(accs))


def main():
    occ = json.load(open(AP.INDEX)); rng = np.random.RandomState(0)
    words = json.load(open(os.path.join(AP.PROOT, 'M23_crossmodal_word_retrieval', 'data', 'words.json')))
    K = len(words)
    m22, c22, mm, msd = FP.load_m22(); Lg = int(c22.win_s * CAP_SR)

    def is_test(ch): return int(ch.split('_')[1]) % c22.test_every == 0

    REAL, GEN, AUD, Y, G = [], [], [], [], []
    gin, gfr = [], []; lag = {}; t0 = time.time()
    for wi, w in enumerate(words):
        rows = [r for r in occ[w] if is_test(r[0])]; rng.shuffle(rows); rows = rows[:CAP_PER_WORD]
        if len(rows) < MIN_PER_WORD:
            continue
        for ch, s, e in rows:
            if ch not in lag: lag[ch] = AP.io.read_lag_ms(AP.lag_path(ch)) / 1000.0
            dur = (e - s) + 2 * CTX
            aud = AP.io.read_wav_window(AP.wav_path(ch), s - CTX, dur, AUD_SR).astype(np.float32)
            capL = AP.io.read_bin_window(AP.bin_path(ch), s - CTX + lag[ch], c22.win_s, CAP_SR).astype(np.float32)
            if len(aud) < AUD_SR * 0.10 or len(capL) < CAP_SR * 0.5:
                continue
            REAL.append(to_T(_mel(aud)))
            AUD.append(aud)
            capL = np.pad(capL, (0, max(0, Lg - len(capL))))[:Lg]
            r = resample_poly(capL, c22.in_sr, CAP_SR).astype(np.float32)
            r = r / (np.sqrt(np.mean(r ** 2)) + 1e-8)
            gin.append(r); gfr.append(max(6, int(round(dur * c22.fps))))
            Y.append(wi); G.append(int(ch.split('_')[1]))
        print(f'[data] {w}: n={len(Y)}  ({time.time()-t0:.0f}s)', flush=True)
    Y = np.array(Y); G = np.array(G); REAL = np.array(REAL)

    # M22 gen mels
    BS = 16
    for b in range(0, len(gin), BS):
        X = torch.from_numpy(np.stack(gin[b:b + BS])).to(dev)
        torch.manual_seed(0)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
            g = m22.sample(X)
        g = (g.float().cpu() * msd + mm)
        for j in range(g.shape[0]):
            GEN.append(to_T(g[j, :, :gfr[b + j]]))
    GEN = np.array(GEN)
    print(f'[ladder] REAL{REAL.shape} GEN{GEN.shape} n={len(Y)} chunks={len(set(G))}', flush=True)

    res = {}
    res['REAL_M0_linear'] = acc_linear(REAL, Y, G, K)
    res['GEN_M0_linear'] = acc_linear(GEN, Y, G, K)
    print(f"[M0] REAL={res['REAL_M0_linear']:.3f}  GEN={res['GEN_M0_linear']:.3f}", flush=True)
    res['REAL_M1_cnn'] = acc_cnn(REAL, Y, G, K)
    res['GEN_M1_cnn'] = acc_cnn(GEN, Y, G, K)
    print(f"[M1] REAL={res['REAL_M1_cnn']:.3f}  GEN={res['GEN_M1_cnn']:.3f}", flush=True)

    # M2 wav2vec2 on real audio (true ceiling)
    try:
        import torchaudio
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        w2v = bundle.get_model().to(dev).eval()
        feats = []
        with torch.no_grad():
            for a in AUD:
                wav = torch.from_numpy(a)[None].to(dev)
                fl, _ = w2v.extract_features(wav)
                feats.append(fl[-1].mean(1).squeeze(0).cpu().numpy())
        F = np.array(feats)
        clf = make_pipeline(StandardScaler(), PCA(min(120, F.shape[1]), random_state=0),
                            LogisticRegression(max_iter=3000, multi_class='multinomial'))
        res['REAL_M2_wav2vec2'] = float(cross_val_score(clf, F, Y, groups=G, cv=GroupKFold(5), scoring='accuracy').mean())
        print(f"[M2] REAL wav2vec2={res['REAL_M2_wav2vec2']:.3f}", flush=True)
    except Exception as ex:
        res['REAL_M2_wav2vec2'] = None
        print(f'[M2] wav2vec2 failed: {ex}', flush=True)

    res['chance'] = 1.0 / K
    json.dump(res, open(os.path.join(AP.PROOT, 'M26_wordpair_signal', 'classifier_ladder.json'), 'w'), indent=1)

    # figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    groups = ['M0_linear\n(kway readout)', 'M1_cnn', 'M2_wav2vec2\n(real only)']
    real_v = [res['REAL_M0_linear'], res['REAL_M1_cnn'], res.get('REAL_M2_wav2vec2') or np.nan]
    gen_v = [res['GEN_M0_linear'], res['GEN_M1_cnn'], np.nan]
    x = np.arange(3); w = 0.38
    ax.bar(x - w/2, real_v, w, label='REAL mel', color='#2563d6')
    ax.bar(x + w/2, gen_v, w, label='M22 GEN mel', color='#e08a10')
    ax.axhline(1/K, ls='--', c='#444', lw=1, label=f'chance 1/{K}')
    ax.axhline(0.776, ls=':', c='#888', label='kway real ceiling (0.78)')
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel('top-1 accuracy (K=30, GroupKFold)')
    ax.set_ylim(0, 1.0)
    ax.set_title('Classifier-strength ladder — is the 0.78 real-mel ceiling method or information?')
    for i, v in enumerate(real_v):
        if not np.isnan(v): ax.text(i - w/2, v + .01, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
    for i, v in enumerate(gen_v):
        if not np.isnan(v): ax.text(i + w/2, v + .01, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(AP.PROOT, 'M26_wordpair_signal', 'classifier_ladder.png'), dpi=150, bbox_inches='tight')
    print('[ladder] wrote classifier_ladder.png', flush=True)
    print('\n==== LADDER VERDICT ====', flush=True)
    for k, v in res.items():
        print(f'  {k:20s} {v}', flush=True)


# reference-audio mel (80 x frames)
_melfn = None
def _mel(aud):
    global _melfn
    import torchaudio
    if _melfn is None:
        _melfn = torchaudio.transforms.MelSpectrogram(AUD_SR, 1024, hop_length=160, win_length=640,
                                                      n_mels=80, f_min=0, f_max=8000, power=2.0)
    return torch.log(_melfn(torch.from_numpy(aud)) + 1e-5).numpy()


if __name__ == '__main__':
    main()
