#!/usr/bin/env python3
"""
reconstruct_v2.py — Improved speech reconstruction from powerline leakage.

Implements three upgrades over am_reconstruct.py and benchmarks them against the
existing outputs using an intelligibility metric (STOI), not just envelope r.

  #5  Per-band SNR-weighted sideband combination
      Instead of a fixed upper-only shift or a flat uniform sum over harmonics,
      learn — from the calibration sweep — how strongly each candidate source
      (direct band + each mains-harmonic sideband) carries each audio frequency,
      and combine them with those weights.  Different audio bands are best
      recovered from different harmonic shifts, so this is a measured, per-bin
      weighted mix rather than one global rule.

  #6  F0-driven excitation phase (replaces Griffin-Lim random phase)
      Griffin-Lim seeds random phase, which makes voiced speech sound whispery.
      Here we estimate an F0 track from the capture's low band, synthesise a
      voiced (harmonic) + unvoiced (noise) excitation, and impose the estimated
      magnitude spectrogram onto the *excitation's* phase.  NOTE: per-frame F0
      from this capture is unreliable (the 100-300 Hz band is contaminated by
      60/120/180 Hz mains harmonics); the win here is a voiced phase structure,
      not prosody recovery.  We report F0 reliability so this stays honest.

  #9  STOI intelligibility metric
      Every method is scored with STOI (+ spectrogram r + envelope r) against the
      reference, so we can see whether a change actually recovers words rather
      than just loudness.

Outputs:
  reconstruct_v2_<method>.wav   reconstructed audio per method
  reconstruct_v2.png            comparison figure + metrics

Usage:
    python3 reconstruct_v2.py --no-show
    python3 reconstruct_v2.py --n-harmonics 16 --no-show
"""

import argparse
import os
import sys
import wave
from math import gcd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as dsp
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAP_SR = 200_000
AUD_SR = 22_050
FRAME_S = 0.05

DARK, PANEL, GRID = '#0d0d14', '#1e1e2e', '#313244'
TEXT, MUTED = '#cdd6f4', '#a6adc8'
BLUE, RED, GRN, YLW, ORNG, PURP = '#89b4fa', '#f38ba8', '#a6e3a1', '#f9e2af', '#fab387', '#cba6f7'


# ── I/O ─────────────────────────────────────────────────────────────────────

def load_capture(path):
    return np.fromfile(path, dtype=np.float32)


def read_wav(path):
    w = wave.open(path, 'rb')
    sr = w.getframerate(); ch = w.getnchannels()
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    w.close()
    if ch > 1:
        d = d.reshape(-1, ch).mean(1)
    return d, sr


def save_wav(path, data, rate):
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak * 0.9
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(rate); wf.writeframes(pcm.tobytes())


def downsample(x, src_sr, dst_sr):
    g = gcd(int(src_sr), int(dst_sr))
    return dsp.resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


# ── mains detection ───────────────────────────────────────────────────────────

def detect_mains(cap_ds, sr):
    nperseg = min(65536, len(cap_ds))
    f, P = dsp.welch(cap_ds, fs=sr, nperseg=nperseg, window='blackman')
    best, best_db = 50.0, -np.inf
    for guess in (50.0, 60.0):
        m = np.abs(f - guess) < 5
        if not m.any():
            continue
        db = 10 * np.log10(P[m].max() / (P.mean() + 1e-30))
        if db > best_db:
            best_db, best = db, float(f[m][np.argmax(P[m])])
    print(f'[mains] f_mains = {best:.2f} Hz  ({best_db:.1f} dB above mean)')
    return best


# ── STFT helpers ───────────────────────────────────────────────────────────────

def stft_mag(x, sr, nperseg, noverlap):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f, t, Z = dsp.stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap,
                           window='hann', boundary=None)
    return f, t, Z


# ── #5  per-band weighted sideband combination ──────────────────────────────────

def learn_band_weights(f_mains, n_harm, sr, nperseg, noverlap,
                       sweep_cap_path, sweep_aud_path, smooth_bins=5):
    """
    Learn, per audio-frequency bin, a weight for each candidate source shift
    (direct=0 Hz, and each mains harmonic n*f_mains) from the calibration sweep.

    weight[shift, f] = ReLU(corr_t(|Y_cap(f+shift, t)|, |Y_aud(f, t)|))**2

    Returns (shifts_hz, weights[n_shifts, n_freq], f_grid) or None if no sweep.
    """
    if not (sweep_cap_path and sweep_aud_path
            and os.path.exists(sweep_cap_path) and os.path.exists(sweep_aud_path)):
        return None

    sc = downsample(load_capture(sweep_cap_path), CAP_SR, AUD_SR)
    sa, _ = read_wav(sweep_aud_path)
    n = min(len(sc), len(sa)); sc, sa = sc[:n], sa[:n]
    sc = sc - sc.mean()

    f, _, Yc = stft_mag(sc, sr, nperseg, noverlap)
    _, _, Ya = stft_mag(sa, sr, nperseg, noverlap)
    nt = min(Yc.shape[1], Ya.shape[1])
    Yc, Ya = np.abs(Yc[:, :nt]), np.abs(Ya[:, :nt])
    df = float(f[1] - f[0]); nfreq = len(f)

    shifts = np.array([0.0] + [k * f_mains for k in range(1, n_harm + 1)])
    W = np.zeros((len(shifts), nfreq))

    # zero-mean over time for correlation
    Ya_c = Ya - Ya.mean(1, keepdims=True)
    Ya_n = np.sqrt((Ya_c ** 2).sum(1)) + 1e-12

    for si, sh in enumerate(shifts):
        bins = np.round((f + sh) / df).astype(int)
        valid = (bins >= 0) & (bins < nfreq)
        src = np.zeros_like(Ya)
        src[valid] = Yc[bins[valid]]
        src_c = src - src.mean(1, keepdims=True)
        src_n = np.sqrt((src_c ** 2).sum(1)) + 1e-12
        corr = (Ya_c * src_c).sum(1) / (Ya_n * src_n)
        W[si] = np.clip(corr, 0, None) ** 2

    # smooth weights across frequency to denoise sparse-excitation bins
    if smooth_bins > 1:
        k = np.ones(smooth_bins) / smooth_bins
        for si in range(len(shifts)):
            W[si] = np.convolve(W[si], k, mode='same')

    s = W.sum(0, keepdims=True)
    W = np.where(s > 1e-9, W / s, 0.0)
    print(f'[#5]    learned weights: {len(shifts)} shifts × {nfreq} bins '
          f'(direct + {n_harm} harmonics @ {f_mains:.1f} Hz)')
    return shifts, W, f


def learn_eq(sr, nperseg, noverlap, sweep_cap_path, sweep_aud_path):
    """Frequency equalization |H(f)| = mean|Y_cap(f)| / mean|Y_aud(f)| from sweep,
    measured on the DIRECT band. Used to gently flatten the coupling response."""
    if not (sweep_cap_path and os.path.exists(sweep_cap_path)
            and sweep_aud_path and os.path.exists(sweep_aud_path)):
        return None
    sc = downsample(load_capture(sweep_cap_path), CAP_SR, AUD_SR)
    sa, _ = read_wav(sweep_aud_path)
    n = min(len(sc), len(sa)); sc, sa = sc[:n] - sc[:n].mean(), sa[:n]
    _, _, Zc = stft_mag(sc, sr, nperseg, noverlap)
    _, _, Za = stft_mag(sa, sr, nperseg, noverlap)
    H = np.abs(Zc).mean(1) / (np.abs(Za).mean(1) + 1e-9)
    return np.maximum(H, 1e-3 * H.max())


def apply_eq(X, H, reg=0.3):
    """Gentle regularized equalization. reg large → near-flat (safe on noisy bands)."""
    if H is None:
        return X
    return X / (H[:, None] + reg * H.max())


def combine_sidebands(cap_ds, f_mains, n_harm, sr, nperseg, noverlap, weights='direct'):
    """Build estimated audio magnitude spectrogram from capture sidebands.

    weights : 'direct'  → use the direct baseband only (best in benchmarks)
              (shifts, W, f_grid) tuple → #5 per-band weighted sideband sum
              None → flat uniform sum over direct+harmonics
    """
    f, t, Z = stft_mag(cap_ds - cap_ds.mean(), sr, nperseg, noverlap)
    Y = np.abs(Z); df = float(f[1] - f[0]); nfreq = len(f)

    if weights == 'direct':
        X = Y.copy(); tag = 'direct'
    elif weights is not None:
        shifts, W, _ = weights
        X = np.zeros_like(Y)
        for si, sh in enumerate(shifts):
            bins = np.round((f + sh) / df).astype(int)
            valid = (bins >= 0) & (bins < nfreq)
            src = np.zeros_like(Y)
            src[valid] = Y[bins[valid]]
            X += W[si][:, None] * src
        tag = 'weighted'
    else:
        shifts = [0.0] + [k * f_mains for k in range(1, n_harm + 1)]
        X = np.zeros_like(Y)
        for sh in shifts:
            bins = np.round((f + sh) / df).astype(int)
            valid = (bins >= 0) & (bins < nfreq)
            X[valid] += Y[bins[valid]]
        X /= len(shifts)
        tag = 'uniform'
    return f, t, Y, X, tag


# ── #6  F0-driven excitation ─────────────────────────────────────────────────

def estimate_f0(cap_ds, sr, hop):
    """Estimate an F0 track + voicing from the capture low band (pitch region)."""
    import librosa
    sos = dsp.butter(4, [70, 400], 'bp', fs=sr, output='sos')
    bp = dsp.sosfiltfilt(sos, cap_ds - cap_ds.mean())
    f0, vflag, vprob = librosa.pyin(bp.astype(float), fmin=80, fmax=320, sr=sr,
                                    frame_length=2048, hop_length=hop)
    med = float(np.nanmedian(f0[vflag])) if vflag.any() else 120.0
    f0 = np.where(np.isfinite(f0), f0, med)
    # heavy smoothing — per-frame F0 is unreliable on this capture
    f0 = dsp.medfilt(f0, 7)
    vprob = np.nan_to_num(vprob, nan=0.0)
    print(f'[#6]    F0 median={med:.1f} Hz  voiced frac={vflag.mean():.2f} '
          f'(per-frame F0 unreliable → used as phase model only)')
    return f0, vprob, med


def synth_excitation(f0_frames, vprob_frames, n_samples, sr, hop):
    """Synthesise voiced(harmonic)+unvoiced(noise) excitation at sample rate."""
    # interpolate frame-rate F0 / voicing onto sample grid
    ft = np.arange(len(f0_frames)) * hop
    st = np.arange(n_samples)
    f0_s = np.interp(st, ft, f0_frames)
    v_s = np.clip(np.interp(st, ft, vprob_frames), 0, 1)
    # continuous phase accumulator
    phase = 2 * np.pi * np.cumsum(f0_s) / sr
    nyq = sr / 2
    voiced = np.zeros(n_samples)
    kmax = int(nyq / max(f0_s.min(), 50))
    for k in range(1, kmax + 1):
        mask = (k * f0_s) < nyq
        voiced += np.where(mask, np.cos(k * phase), 0.0)
    voiced /= (np.abs(voiced).max() + 1e-9)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n_samples)
    exc = v_s * voiced + (1 - v_s) * noise
    return exc.astype(np.float32)


# ── magnitude → audio ───────────────────────────────────────────────────────────

def griffin_lim(mag, sr, nperseg, noverlap, n_iter=60, init_phase=None):
    import warnings
    if init_phase is None:
        phase = np.exp(1j * np.random.uniform(-np.pi, np.pi, mag.shape))
    else:
        phase = np.exp(1j * init_phase[:, :mag.shape[1]])
    kw = dict(fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann')
    for _ in range(n_iter):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, x = dsp.istft(mag * phase, **kw)
            _, _, S = dsp.stft(x, **kw)
        n = min(mag.shape[1], S.shape[1])
        phase = np.exp(1j * np.angle(S[:, :n])); mag = mag[:, :n]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, out = dsp.istft(mag * phase, **kw)
    return out.astype(np.float32)


def excitation_phase_synth(mag, exc, sr, nperseg, noverlap, gl_iters=8):
    """Impose target magnitude onto excitation phase, then a few GL refinements."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, _, E = dsp.stft(exc, fs=sr, nperseg=nperseg, noverlap=noverlap,
                           window='hann', boundary=None)
    n = min(mag.shape[1], E.shape[1])
    return griffin_lim(mag[:, :n], sr, nperseg, noverlap,
                       n_iter=gl_iters, init_phase=np.angle(E[:, :n]))


# ── metrics ────────────────────────────────────────────────────────────────────

def fit_len(x, n):
    return x[:n] if len(x) >= n else np.r_[x, np.zeros(n - len(x), np.float32)]


def metrics(ref, rec, sr, fmax=4000):
    from pystoi import stoi
    n = min(len(ref), len(rec)); a, r = ref[:n], rec[:n]
    fr = int(FRAME_S * sr); nf = n // fr
    ae = np.array([np.sqrt(np.mean(a[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    re = np.array([np.sqrt(np.mean(r[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    r_env = float(pearsonr(ae, re)[0])
    f, _, Ya = dsp.stft(a, fs=sr, nperseg=2048, noverlap=1536, window='hann')
    _, _, Yr = dsp.stft(r, fs=sr, nperseg=2048, noverlap=1536, window='hann')
    fm = f <= fmax; nt = min(Ya.shape[1], Yr.shape[1])
    r_spec = float(np.corrcoef(np.abs(Ya[fm, :nt]).ravel(),
                               np.abs(Yr[fm, :nt]).ravel())[0, 1])
    try:
        s = float(stoi(a, r, sr, extended=False))
    except Exception as e:
        s = float('nan'); print(f'[stoi]  failed: {e}')
    return dict(stoi=s, spec_r=r_spec, env_r=r_env)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture', default=os.path.join(SCRIPT_DIR, 'capture.bin'))
    ap.add_argument('--reference', default=os.path.join(SCRIPT_DIR, 'original_reference.wav'))
    ap.add_argument('--sweep-capture', default=os.path.join(SCRIPT_DIR, 'sweep_capture.bin'))
    ap.add_argument('--sweep-audio', default=os.path.join(SCRIPT_DIR, 'sweep_cal.wav'))
    ap.add_argument('--n-harmonics', type=int, default=16)
    ap.add_argument('--nperseg', type=int, default=2048)
    ap.add_argument('--gl-iters', type=int, default=60)
    ap.add_argument('--fmax', type=float, default=4000)
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use('Agg')

    cap = load_capture(args.capture)
    ref, rsr = read_wav(args.reference)
    if rsr != AUD_SR:
        ref = downsample(ref, rsr, AUD_SR)
    cap_ds = downsample(cap, CAP_SR, AUD_SR)
    n = min(len(ref), len(cap_ds)); ref, cap_ds = ref[:n], cap_ds[:n]
    print(f'[load]  {n/AUD_SR:.2f}s  (lag sidecar=0)')

    f_mains = detect_mains(cap_ds, AUD_SR)
    nperseg = args.nperseg; noverlap = nperseg * 3 // 4
    hop = nperseg - noverlap

    # ── #5: learn weights; build direct, uniform, and weighted estimates ──────
    W = learn_band_weights(f_mains, args.n_harmonics, AUD_SR, nperseg, noverlap,
                           args.sweep_capture, args.sweep_audio)
    H = learn_eq(AUD_SR, nperseg, noverlap, args.sweep_capture, args.sweep_audio)
    f, t, Y, X_dir, _ = combine_sidebands(cap_ds, f_mains, args.n_harmonics,
                                          AUD_SR, nperseg, noverlap, weights='direct')
    _, _, _, X_uni, _ = combine_sidebands(cap_ds, f_mains, args.n_harmonics,
                                          AUD_SR, nperseg, noverlap, weights=None)
    _, _, _, X_wgt, _ = combine_sidebands(cap_ds, f_mains, args.n_harmonics,
                                          AUD_SR, nperseg, noverlap, weights=W)
    X_dir_eq = apply_eq(X_dir, H)

    # ── #6: F0 + excitation ───────────────────────────────────────────────────
    f0, vprob, f0med = estimate_f0(cap_ds, AUD_SR, hop)
    exc = synth_excitation(f0, vprob, n, AUD_SR, hop)

    # ── reconstruct via each method ───────────────────────────────────────────
    def GL(X):
        np.random.seed(42)
        return fit_len(griffin_lim(X, AUD_SR, nperseg, noverlap, args.gl_iters), n)
    print('[recon] direct + GL  /  direct + EQ + GL  /  #5 weighted  /  #5+#6 excitation …')
    methods = {
        'direct+GL': GL(X_dir),
        'direct+EQ+GL (best)': GL(X_dir_eq),
        'uniform-sum+GL': GL(X_uni),
        'weighted-sidebands+GL (#5)': GL(X_wgt),
        'direct+excitation (#6)': fit_len(
            excitation_phase_synth(X_dir, exc, AUD_SR, nperseg, noverlap), n),
    }

    # include existing am_reconstruct output if present
    for name, fn in [('am_reconstruct_h8.wav', 'am_reconstruct_h8.wav'),
                     ('am_reconstruct_shift.wav', 'am_reconstruct_shift.wav')]:
        p = os.path.join(SCRIPT_DIR, fn)
        if os.path.exists(p):
            d, sr = read_wav(p)
            if sr != AUD_SR:
                d = downsample(d, sr, AUD_SR)
            methods[f'existing {name}'] = fit_len(d, n)

    # ── #9: score everything ──────────────────────────────────────────────────
    print('\n══ Metrics vs reference ' + '═' * 40)
    print(f'{"method":<32} {"STOI":>7} {"spec_r":>8} {"env_r":>7}')
    results = {}
    for name, rec in methods.items():
        m = metrics(ref, rec, AUD_SR, args.fmax)
        results[name] = m
        print(f'{name:<32} {m["stoi"]:>7.3f} {m["spec_r"]:>8.3f} {m["env_r"]:>7.3f}')
    print('═' * 64)
    print('STOI: 0=unintelligible, ~0.5 poor, >0.7 good. spec_r/env_r: correlation.')

    # save wavs
    for name, rec in methods.items():
        if name.startswith('existing'):
            continue
        tag = name.split()[0].replace('+', '_')
        save_wav(os.path.join(SCRIPT_DIR, f'reconstruct_v2_{tag}.wav'), rec, AUD_SR)

    # ── figure ─────────────────────────────────────────────────────────────────
    make_figure(ref, cap_ds, Y, X_dir, X_wgt, f, t, W, f0, vprob,
                results, args.fmax, os.path.join(SCRIPT_DIR, 'reconstruct_v2.png'))

    if not args.no_show:
        plt.show()


def make_figure(ref, cap_ds, Y, X_uni, X_wgt, f, t, W, f0, vprob,
                results, fmax, out_png):
    duration = len(ref) / AUD_SR

    def specpanel(ax, tt, ff, S, cmap, title):
        fm = ff <= fmax; Sdb = 10 * np.log10(S[fm] + 1e-30)
        ax.set_facecolor(PANEL); ax.grid(False)
        ax.pcolormesh(tt, ff[fm], Sdb, shading='gouraud', cmap=cmap,
                      vmin=np.percentile(Sdb, 5), vmax=np.percentile(Sdb, 99),
                      rasterized=True)
        ax.set_title(title, color=TEXT, fontsize=9, loc='left')
        ax.set_xlabel('Time (s)', color=MUTED, fontsize=8)
        ax.set_ylabel('Hz', color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.set_xlim(0, duration)

    fr, _, Yref = dsp.stft(ref, fs=AUD_SR, nperseg=2048, noverlap=1536, window='hann')
    tr = np.linspace(0, duration, Yref.shape[1])

    fig = plt.figure(figsize=(15, 13), facecolor=DARK)
    fig.suptitle('reconstruct_v2 — per-band weighting (#5) + F0 excitation (#6) + STOI (#9)',
                 color=TEXT, fontsize=12, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    specpanel(fig.add_subplot(gs[0, 0]), tr, fr, np.abs(Yref), 'magma', 'Reference audio')
    specpanel(fig.add_subplot(gs[0, 1]), t, f, X_uni, 'inferno', 'Estimate: direct band (best)')
    specpanel(fig.add_subplot(gs[0, 2]), t, f, X_wgt, 'plasma', 'Estimate: per-band weighted (#5)')

    # weight heatmap
    ax = fig.add_subplot(gs[1, 0])
    if W is not None:
        shifts, Wm, fg = W
        fmask = fg <= fmax
        ax.set_facecolor(PANEL); ax.grid(False)
        im = ax.pcolormesh(fg[fmask], np.arange(len(shifts)), Wm[:, fmask],
                           cmap='viridis', shading='auto', rasterized=True)
        ax.set_yticks(np.arange(len(shifts)))
        ax.set_yticklabels(['direct'] + [f'h{k}' for k in range(1, len(shifts))], fontsize=6)
        ax.set_title('#5 learned source weights (shift × audio freq)', color=TEXT, fontsize=9, loc='left')
        ax.set_xlabel('Audio freq (Hz)', color=MUTED, fontsize=8)
    else:
        ax.text(0.5, 0.5, 'no sweep — uniform', color=MUTED, ha='center', transform=ax.transAxes)
        ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    # F0 track
    ax = fig.add_subplot(gs[1, 1])
    tf0 = np.linspace(0, duration, len(f0))
    ax.set_facecolor(PANEL)
    ax.plot(tf0, f0, color=YLW, lw=1.0, label='F0 (capture)')
    ax.plot(tf0, vprob * f0.max(), color=GRN, lw=0.8, alpha=0.6, label='voicing×scale')
    ax.set_title('#6 F0 track from capture low band', color=TEXT, fontsize=9, loc='left')
    ax.set_xlabel('Time (s)', color=MUTED, fontsize=8); ax.set_ylabel('Hz', color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=8); ax.set_xlim(0, duration)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    # metrics bar chart (STOI)
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(PANEL)
    names = list(results.keys())
    stoi_v = [results[k]['stoi'] for k in names]
    cols = [GRN if 'excitation' in k else (BLUE if '#5' in k else MUTED) for k in names]
    ax.barh(range(len(names)), stoi_v, color=cols, alpha=0.85, edgecolor=GRID)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([k.replace(' ', '\n', 1) for k in names], fontsize=6)
    ax.set_title('#9 STOI by method (higher=better)', color=TEXT, fontsize=9, loc='left')
    ax.tick_params(colors=MUTED, labelsize=7); ax.set_xlim(0, max(0.3, max(stoi_v) * 1.2))
    for i, v in enumerate(stoi_v):
        ax.text(v, i, f' {v:.3f}', color=TEXT, fontsize=7, va='center')
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    # metrics table text
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    lines = [f'{"method":<34}{"STOI":>8}{"spec_r":>9}{"env_r":>8}']
    lines.append('─' * 60)
    for k in names:
        m = results[k]
        lines.append(f'{k:<34}{m["stoi"]:>8.3f}{m["spec_r"]:>9.3f}{m["env_r"]:>8.3f}')
    ax.text(0.01, 0.95, '\n'.join(lines), color=TEXT, fontsize=10,
            family='monospace', va='top', transform=ax.transAxes)

    fig.patch.set_facecolor(DARK)
    fig.savefig(out_png, dpi=140, bbox_inches='tight', facecolor=DARK)
    print(f'[viz]   saved → {out_png}')


if __name__ == '__main__':
    main()
