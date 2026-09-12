#!/usr/bin/env python3
"""
M26 — raw-200 kHz signal analysis of a CONFUSED word pair.

Question: the powerline->mel generators (M14/M15/M16/M22) confuse similar-sounding
words (e.g. which<->this). Is the discriminating information ABSENT from the raw
200 kHz capture (the model input), or present-but-thrown-away by the generator?

We build an information-loss "separability ladder" for a duration-matched pair and
measure 2-class separability at each representation, clean-audio ceiling -> model
input, plus a frequency-resolved discriminability spectrum showing WHERE any A/B
contrast lives.

  L0  clean 16 kHz reference mel          -> ceiling (words ARE different)
  L1  raw 200 kHz broadband Welch PSD     -> full-band model input
  L2  mains-harmonic AM-demod baseband    -> the sideband "content channel"
  L3  envelope-only loudness contour      -> control (durations matched => ~chance)

Metric per level: StratifiedKFold(5) logistic-regression BALANCED ACCURACY vs 50%,
with a 100x label-permutation null. Verdict: gap L0 - L1/L2, and L2 - L3.

Reuses M15 windowed readers (read_bin_window/read_wav_window/read_lag_ms). Self-
contained mains detection + envelope (short). Word occurrences from M20 word_index.

Usage:  python analyze_pair.py --pair which this --n 300
Output: M26_wordpair_signal/<A>_<B>/  (results.json + wordpair_signal.png)
"""
import os, sys, json, argparse, time
import numpy as np
from scipy import signal as dsp
import torch, torchaudio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROOT = '<REPO_ROOT>'
M15   = os.path.join(PROOT, 'M15_soundbar_melgen_word')
INDEX = os.path.join(PROOT, 'M20_powerline_word_classifier', 'word_index.json')
BIN   = os.path.join(PROOT, 'Powerline_Data_Captures', 'soundbar_bin_captures')
WAV   = os.path.join(PROOT, 'Powerline_Data_Captures', 'audio_chunks')
CAP_SR = 200_000; AUD_SR = 16_000
CTX = 0.05                       # sec context each side of the word

sys.path.insert(0, M15)
import data_io as io             # read_bin_window, read_wav_window, read_lag_ms


# ── reusable helpers ──────────────────────────────────────────────────────────
def bin_path(ch): return os.path.join(BIN, f'{ch}.bin')
def wav_path(ch): return os.path.join(WAV, f'{ch}.wav')
def lag_path(ch): return os.path.join(BIN, f'{ch}.lag')

def detect_mains(x, sr, guess=60.0, search=6.0):
    """Exact mains fundamental from long-term Welch spectrum (near `guess` Hz)."""
    nps = min(1 << 16, len(x))
    f, P = dsp.welch(x, fs=sr, nperseg=nps, window='blackman')
    m = np.abs(f - guess) < search
    if not m.any():
        return guess
    return float(f[m][np.argmax(P[m])])

def envelope(x, npts=50):
    """Peak-normed RMS loudness contour, time-normalized to npts."""
    fr = max(1, len(x) // 200)
    e = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(fr) / fr, 'same'))
    e = np.interp(np.linspace(0, 1, npts), np.linspace(0, 1, len(e)), e)
    return (e / (e.max() + 1e-9)).astype(np.float32)

melfn = torchaudio.transforms.MelSpectrogram(
    AUD_SR, 1024, hop_length=160, win_length=640, n_mels=80, f_min=0, f_max=8000, power=2.0)

def clean_mel_T(aw, T=32):
    m = torch.log(melfn(torch.from_numpy(aw)) + 1e-5)            # (80, frames)
    m = torch.nn.functional.interpolate(m[None, None], size=(80, T),
                                        mode='bilinear', align_corners=False)[0, 0]
    return m.numpy().astype(np.float32)                          # (80, T)


# ── L2: pick strong mains harmonics, AM-demodulate each to baseband ───────────
def select_harmonics(fmains, sr, hi=95_000.0, nh=20, probe=None):
    freqs = np.arange(1, int(hi / fmains) + 1) * fmains
    freqs = freqs[(freqs > 3_000) & (freqs < hi)]               # skip the fundamental cluster
    if probe is not None and len(freqs) > nh:
        f, P = dsp.welch(probe, fs=sr, nperseg=min(1 << 15, len(probe)))
        pw = np.array([P[np.argmin(np.abs(f - fc))] for fc in freqs])
        freqs = freqs[np.argsort(-pw)[:nh]]
    return np.sort(freqs)[:nh]

def demod_baseband(x, fc, sr, bw=4000.0, out_sr=8000, tbins=16, mbins=24):
    """Bandpass around fc, Hilbert magnitude envelope -> compact logmel (mbins,tbins)."""
    lo, hi = max(1.0, fc - bw), min(sr / 2 - 1.0, fc + bw)
    sos = dsp.butter(4, [lo, hi], btype='band', fs=sr, output='sos')
    y = dsp.sosfiltfilt(sos, x)
    env = np.abs(dsp.hilbert(y))                                 # AM envelope @ sr
    env = dsp.resample_poly(env, out_sr, sr).astype(np.float32)  # -> out_sr
    f, t, Z = dsp.stft(env, fs=out_sr, nperseg=256, noverlap=192)
    S = np.log(np.abs(Z) + 1e-6)                                 # (freq, time)
    S = torch.nn.functional.interpolate(torch.from_numpy(S)[None, None].float(),
                                        size=(mbins, tbins), mode='bilinear',
                                        align_corners=False)[0, 0].numpy()
    return S.astype(np.float32)                                  # (mbins, tbins)


# ── separability (balanced acc + permutation null) ────────────────────────────
def separability(X, y, seed=0, nperm=100, ncomp=50):
    ncomp = min(ncomp, X.shape[1], int(0.75 * X.shape[0]) - 1)   # must fit the CV train fold
    clf = make_pipeline(StandardScaler(), PCA(ncomp, random_state=seed),
                        LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = float(cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy').mean())
    rng = np.random.RandomState(seed); nulls = []
    for _ in range(nperm):
        yp = rng.permutation(y)
        nulls.append(float(cross_val_score(clf, X, yp, cv=cv,
                                           scoring='balanced_accuracy').mean()))
    nulls = np.array(nulls)
    p = float((np.sum(nulls >= acc) + 1) / (nperm + 1))
    return {'balacc': acc, 'null_mean': float(nulls.mean()),
            'null_std': float(nulls.std()), 'p': p}

def r2_spectrum(F, y):
    """Per-column point-biserial r^2 between feature and binary label."""
    y = y.astype(np.float64); yc = y - y.mean()
    Fc = F - F.mean(0, keepdims=True)
    num = (Fc * yc[:, None]).mean(0) ** 2
    den = (Fc ** 2).mean(0) * (yc ** 2).mean() + 1e-20
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, default=['which', 'this'])
    ap.add_argument('--n', type=int, default=300, help='occurrences per word')
    ap.add_argument('--nh', type=int, default=20, help='mains harmonics for L2')
    args = ap.parse_args()
    A, B = args.pair
    OUT = os.path.join(PROOT, 'M26_wordpair_signal', f'{A}_{B}'); os.makedirs(OUT, exist_ok=True)
    occ = json.load(open(INDEX))
    rng = np.random.RandomState(0)

    # global mains + harmonic set from a probe capture (first train-ish chunk)
    probe_ch = occ[A][0][0]
    probe = io.read_bin_window(bin_path(probe_ch), 1.0, 4.0, CAP_SR).astype(np.float32)
    fmains = detect_mains(probe, CAP_SR)
    harmonics = select_harmonics(fmains, CAP_SR, nh=args.nh, probe=probe)
    print(f'[m26] pair={A}/{B}  f_mains={fmains:.3f} Hz  harmonics={np.round(harmonics/1e3,1)} kHz', flush=True)

    def sample(word):
        rows = occ[word][:]; rng.shuffle(rows)
        return rows[:args.n]

    # accumulators
    L0, L1, L2, L3, Y = [], [], [], [], []
    cap_ds_all, aud_all = [], []      # for coherence + capture spectrum
    lagcache = {}
    t0 = time.time()
    for lab, word in [(0, A), (1, B)]:
        rows = sample(word); kept = 0
        for ch, s, e in rows:
            if ch not in lagcache:
                lagcache[ch] = io.read_lag_ms(lag_path(ch)) / 1000.0
            lag = lagcache[ch]
            dur = (e - s) + 2 * CTX
            cap = io.read_bin_window(bin_path(ch), s - CTX + lag, dur, CAP_SR).astype(np.float32)
            aud = io.read_wav_window(wav_path(ch), s - CTX, dur, AUD_SR).astype(np.float32)
            if len(cap) < CAP_SR * 0.08 or len(aud) < AUD_SR * 0.08:
                continue
            cap = cap / (np.sqrt(np.mean(cap ** 2)) + 1e-8)
            # L0 clean mel
            L0.append(clean_mel_T(aud).ravel())
            # L1 broadband PSD 0-100kHz
            f, P = dsp.welch(cap, fs=CAP_SR, nperseg=4096)
            L1.append(np.log(P + 1e-12).astype(np.float32))
            # L2 mains-harmonic demod baseband (stacked)
            feats = [demod_baseband(cap, fc, CAP_SR).ravel() for fc in harmonics]
            L2.append(np.concatenate(feats))
            # L3 envelope of downsampled capture
            cap_ds = dsp.resample_poly(cap, AUD_SR, CAP_SR).astype(np.float32)
            L3.append(envelope(cap_ds))
            cap_ds_all.append(cap_ds); aud_all.append(aud)
            Y.append(lab); kept += 1
        print(f'[m26] {word}: kept {kept}  ({time.time()-t0:.0f}s)', flush=True)

    Y = np.array(Y)
    L0 = np.array(L0); L1 = np.array(L1); L2 = np.array(L2); L3 = np.array(L3)
    print(f'[m26] features: L0{L0.shape} L1{L1.shape} L2{L2.shape} L3{L3.shape}  n={len(Y)} '
          f'(A={int((Y==0).sum())} B={int((Y==1).sum())})', flush=True)

    # ── separability ladder ──
    levels = {'L0_clean_mel': L0, 'L1_raw200k_psd': L1, 'L2_mains_demod': L2, 'L3_envelope': L3}
    sep = {}
    for name, X in levels.items():
        sep[name] = separability(X, Y)
        r = sep[name]
        print(f'[sep] {name:16s} balacc={r["balacc"]:.3f}  null={r["null_mean"]:.3f}±{r["null_std"]:.3f}  p={r["p"]:.3g}', flush=True)

    # ── coherence: capture baseband vs clean audio, 300-3500 Hz ──
    coh = []
    for cd, ad in zip(cap_ds_all, aud_all):
        L = min(len(cd), len(ad))
        if L < 512: continue
        f, C = dsp.coherence(cd[:L], ad[:L], fs=AUD_SR, nperseg=min(512, L))
        band = (f >= 300) & (f <= 3500)
        coh.append(float(C[band].mean()))
    coh = np.array(coh)
    print(f'[coh] capture-vs-audio MSC (300-3500Hz): mean={coh.mean():.4f} median={np.median(coh):.4f}', flush=True)

    # ── discriminability spectra ──
    r2_L0 = r2_spectrum(L0.reshape(len(Y), 80, -1).mean(2), Y)          # per mel bin (clean)
    r2_L1 = r2_spectrum(L1, Y)                                          # per PSD bin (capture)

    res = {'pair': [A, B], 'n': int(len(Y)), 'f_mains': fmains,
           'harmonics_hz': harmonics.tolist(), 'separability': sep,
           'coherence_mean': float(coh.mean()), 'coherence_median': float(np.median(coh)),
           'verdict_gap_L0_minus_L1': sep['L0_clean_mel']['balacc'] - sep['L1_raw200k_psd']['balacc'],
           'verdict_L2_minus_L3': sep['L2_mains_demod']['balacc'] - sep['L3_envelope']['balacc']}
    json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)

    # ── figure ──
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)
    # (a) ladder
    ax = fig.add_subplot(gs[0, 0])
    names = list(levels.keys()); accs = [sep[n]['balacc'] for n in names]
    nullm = [sep[n]['null_mean'] for n in names]; nulls = [sep[n]['null_std'] for n in names]
    x = np.arange(len(names))
    ax.bar(x, accs, color=['#2563d6', '#e08a10', '#b2182b', '#777'])
    ax.errorbar(x, nullm, yerr=np.array(nulls) * 2, fmt='_', color='k', capsize=4, label='perm null ±2σ')
    ax.axhline(0.5, ls='--', c='#444', lw=1); ax.set_ylim(0.4, 1.0)
    ax.set_xticks(x); ax.set_xticklabels([n.split('_')[0] for n in names])
    ax.set_ylabel('2-class balanced accuracy'); ax.set_title(f'Separability ladder: {A} vs {B}')
    for i, a in enumerate(accs): ax.text(i, a + .01, f'{a:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    # (b) discriminability: clean mel bins
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.linspace(0, 8000, len(r2_L0)), r2_L0, c='#2563d6')
    ax.set_title('Clean-audio discriminability (per mel bin)'); ax.set_xlabel('Hz'); ax.set_ylabel('point-biserial r²')
    # (c) discriminability: capture PSD 0-100kHz
    ax = fig.add_subplot(gs[0, 2])
    fp = np.linspace(0, CAP_SR / 2, len(r2_L1))
    ax.plot(fp / 1e3, r2_L1, c='#e08a10', lw=.8)
    for h in harmonics: ax.axvline(h / 1e3, c='#bbb', lw=.4, zorder=0)
    ax.set_title('Capture 200kHz discriminability (per PSD bin)'); ax.set_xlabel('kHz'); ax.set_ylabel('r²')
    # (d) mean capture PSD A vs B
    ax = fig.add_subplot(gs[1, 0])
    pa = np.exp(L1[Y == 0]).mean(0); pb = np.exp(L1[Y == 1]).mean(0)
    ax.plot(fp / 1e3, 10 * np.log10(pa + 1e-12), c='#2563d6', lw=.7, label=A)
    ax.plot(fp / 1e3, 10 * np.log10(pb + 1e-12), c='#b2182b', lw=.7, alpha=.8, label=B)
    ax.set_title('Mean capture PSD (A vs B)'); ax.set_xlabel('kHz'); ax.set_ylabel('dB'); ax.legend(fontsize=8)
    # (e) envelopes
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(L3[Y == 0].mean(0), c='#2563d6', label=A); ax.plot(L3[Y == 1].mean(0), c='#b2182b', label=B)
    ax.fill_between(range(50), L3[Y == 0].mean(0) - L3[Y == 0].std(0), L3[Y == 0].mean(0) + L3[Y == 0].std(0), color='#2563d6', alpha=.15)
    ax.set_title('Mean loudness envelope (A vs B)'); ax.set_xlabel('norm time'); ax.legend(fontsize=8)
    # (f) coherence hist
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(coh, bins=30, color='#777'); ax.axvline(coh.mean(), c='k', ls='--')
    ax.set_title(f'Capture↔audio coherence 300-3500Hz\nmean={coh.mean():.4f} (CHIRP≈0.004)'); ax.set_xlabel('MSC')
    fig.suptitle(f'M26 — where is the {A}/{B} distinction lost?  '
                 f'(L0 clean {sep["L0_clean_mel"]["balacc"]:.2f} → L1 raw200k {sep["L1_raw200k_psd"]["balacc"]:.2f}, '
                 f'L2 demod {sep["L2_mains_demod"]["balacc"]:.2f}, L3 env {sep["L3_envelope"]["balacc"]:.2f})',
                 fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUT, 'wordpair_signal.png'), dpi=150, bbox_inches='tight')
    print('[m26] wrote', os.path.join(OUT, 'wordpair_signal.png'), flush=True)

    # ── verdict ──
    print('\n==== M26 VERDICT ====', flush=True)
    print(f'clean-audio (L0) balacc = {sep["L0_clean_mel"]["balacc"]:.3f}  (should be >>0.5)', flush=True)
    print(f'raw 200k   (L1) balacc = {sep["L1_raw200k_psd"]["balacc"]:.3f}  p={sep["L1_raw200k_psd"]["p"]:.3g}', flush=True)
    print(f'mains-demod(L2) balacc = {sep["L2_mains_demod"]["balacc"]:.3f}  p={sep["L2_mains_demod"]["p"]:.3g}', flush=True)
    print(f'envelope   (L3) balacc = {sep["L3_envelope"]["balacc"]:.3f}  p={sep["L3_envelope"]["p"]:.3g}', flush=True)
    print(f'info gap L0-L1 = {res["verdict_gap_L0_minus_L1"]:.3f} ; demod-over-env L2-L3 = {res["verdict_L2_minus_L3"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
