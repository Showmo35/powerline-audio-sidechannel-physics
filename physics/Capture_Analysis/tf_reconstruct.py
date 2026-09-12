#!/usr/bin/env python3
"""
tf_reconstruct.py — Reconstruct the audio spectrogram from a powerline capture
                    using the measured transfer function H(f).

Pipeline:
  1. Load H(f) from transfer_function.npz (produced by tf_estimate.py)
  2. Load a new capture.bin (powerline response to arbitrary audio)
  3. Downsample capture 200 kHz → 22050 Hz and apply lag correction
  4. STFT of downsampled capture → Y(f, t)
  5. Per-bin Wiener deconvolution:
       X̂(f, t) = Y(f, t) · H*(f) / (|H(f)|² + λ)
     where λ = reg · max(|H|²) regularises division where coupling is weak
  6. Magnitude spectrogram:  |X̂(f, t)|  — the estimated audio STFT
  7. Griffin-Lim phase reconstruction → time-domain audio
  8. Save reconstructed_tf.wav and comparison figure

Outputs:
  reconstructed_tf.wav   reconstructed audio
  tf_reconstruct.png     4-panel comparison figure

Usage:
    python3 tf_reconstruct.py
    python3 tf_reconstruct.py --audio hello5.wav --capture capture.bin
    python3 tf_reconstruct.py --reg 0.05 --gl-iters 80 --fmax 8000 --no-show
"""

import argparse
import glob
import os
import subprocess
import sys
import wave
from math import gcd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as dsp
from scipy.stats import pearsonr

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH = os.path.join(SCRIPT_DIR, 'capture.bin')
TF_PATH      = os.path.join(SCRIPT_DIR, 'transfer_function.npz')
AUDIO_EXTS   = ('*.wav', '*.mp3', '*.flac', '*.ogg', '*.aac', '*.m4a')
CAP_SR       = 200_000
AUD_SR       = 22_050
FRAME_S      = 0.05

DARK  = '#0d0d14'
PANEL = '#1e1e2e'
GRID  = '#313244'
TEXT  = '#cdd6f4'
MUTED = '#a6adc8'
BLUE  = '#89b4fa'
RED   = '#f38ba8'
GRN   = '#a6e3a1'
YLW   = '#f9e2af'
ORNG  = '#fab387'
PURP  = '#cba6f7'


# ── I/O helpers ───────────────────────────────────────────────────────────────

def find_audio(directory):
    for pat in AUDIO_EXTS:
        hits = sorted(glob.glob(os.path.join(directory, pat)))
        if hits:
            return hits[0]
    return None


def load_capture(path):
    data = np.fromfile(path, dtype=np.float32)
    print(f'[cap]   {len(data):,} samples  ({len(data)/CAP_SR:.4f}s @ {CAP_SR} Hz)')
    return data


def decode_audio(path, rate=AUD_SR):
    cmd = ['ffmpeg', '-v', 'quiet', '-i', path,
           '-f', 'f32le', '-ac', '1', '-ar', str(rate), 'pipe:1']
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())
    data = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    print(f'[audio] {len(data):,} samples  ({len(data)/rate:.4f}s @ {rate} Hz)')
    return data


def downsample(x, src_sr, dst_sr):
    g = gcd(int(src_sr), int(dst_sr))
    return dsp.resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def save_wav(path, data, rate):
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak * 0.9
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(rate); wf.writeframes(pcm.tobytes())
    print(f'[wav]   saved → {path}  ({os.path.getsize(path)//1024} KB)')


# ── lag ───────────────────────────────────────────────────────────────────────

def detect_lag(aud, cap_ds):
    f = int(FRAME_S * AUD_SR)
    n = min(len(aud) // f, len(cap_ds) // f)
    if n < 4:
        return 0.0
    a_rms = np.array([np.sqrt(np.mean(aud[i*f:(i+1)*f]**2))    for i in range(n)])
    c_rms = np.array([np.sqrt(np.mean(cap_ds[i*f:(i+1)*f]**2)) for i in range(n)])
    a = (a_rms - a_rms.mean()) / (a_rms.std() + 1e-12)
    c = (c_rms - c_rms.mean()) / (c_rms.std() + 1e-12)
    xc = np.correlate(a, c, mode='full') / n
    lags = np.arange(-(n - 1), n)
    mf = int(round(0.700 / FRAME_S))
    valid = (-mf <= lags) & (lags <= 0)
    return -float(lags[valid][np.argmax(xc[valid])]) * FRAME_S * 1000


def apply_lag(aud, cap_ds, lag_ms):
    lag_samp = int(round(abs(lag_ms) / 1000 * AUD_SR))
    if lag_ms > 0 and lag_samp > 0:
        cap_ds = cap_ds[lag_samp:]
    elif lag_ms < 0 and lag_samp > 0:
        cap_ds = np.r_[np.zeros(lag_samp, dtype=np.float32), cap_ds]
    n = min(len(aud), len(cap_ds))
    return aud[:n].copy(), cap_ds[:n].copy()


# ── transfer function loading + interpolation ─────────────────────────────────

def load_tf(path):
    data = np.load(path, allow_pickle=False)
    f_tf   = data['f'].astype(np.float64)
    H      = data['H_complex']
    coh    = data['coherence'].astype(np.float64)
    sr_tf  = float(data['samp_rate'])
    print(f'[TF]    loaded {os.path.basename(path)}'
          f'  ({len(f_tf)} bins  SR={sr_tf:.0f} Hz  '
          f'coherence_mean={coh.mean():.3f})')
    return f_tf, H, coh, sr_tf


def interp_tf(f_tf, H, coh, f_stft, coh_floor=0.1):
    """
    Interpolate H(f) from the TF frequency grid onto the STFT frequency grid.
    Bins where coherence < coh_floor are considered unreliable; their gain is
    clamped to the mean(|H|) so they don't amplify noise.
    """
    H_mag  = np.abs(H)
    H_ph   = np.angle(H)
    mean_H = H_mag.mean()

    # Linear interpolation of magnitude and phase separately (avoid phase wrapping issues
    # for small phase values; unwrap first)
    H_ph_uw = np.unwrap(H_ph)

    mag_interp = np.interp(f_stft, f_tf, H_mag)
    ph_interp  = np.interp(f_stft, f_tf, H_ph_uw)
    coh_interp = np.interp(f_stft, f_tf, coh)

    # Clamp unreliable bins to mean magnitude (zero phase)
    bad = coh_interp < coh_floor
    if bad.any():
        mag_interp[bad] = mean_H
        ph_interp[bad]  = 0.0

    return (mag_interp * np.exp(1j * ph_interp)).astype(np.complex64)


# ── Wiener deconvolution ──────────────────────────────────────────────────────

def wiener_deconvolve(Y, H_interp, reg):
    """
    Wiener deconvolution applied to every STFT frame simultaneously.

    Y        : complex STFT  [n_freqs, n_frames]
    H_interp : complex H(f)  [n_freqs]
    reg      : Tikhonov regularisation  λ = reg · max(|H|²)

    Returns X_est [n_freqs, n_frames] (complex).
    """
    H_mag2 = (np.abs(H_interp) ** 2).astype(np.float64)
    H_conj = np.conj(H_interp)
    lam    = reg * float(H_mag2.max())

    # Broadcast H over time frames
    X_est = Y * H_conj[:, None] / (H_mag2[:, None] + lam)
    return X_est.astype(np.complex64)


# ── Griffin-Lim ───────────────────────────────────────────────────────────────

def griffin_lim(mag, sr, nperseg, noverlap, n_iter=50):
    """Reconstruct a signal from a magnitude STFT via Griffin-Lim."""
    import warnings
    phase = np.exp(1j * np.random.uniform(-np.pi, np.pi, mag.shape))
    kw = dict(fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann')
    for _ in range(n_iter):
        S = mag * phase
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, x = dsp.istft(S, **kw)
        _, _, S_new = dsp.stft(x, **kw)
        n = min(S.shape[1], S_new.shape[1])
        phase = np.exp(1j * np.angle(S_new[:, :n]))
        mag   = mag[:, :n]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, out = dsp.istft(mag * phase, **kw)
    return out.astype(np.float32)


# ── plotting ──────────────────────────────────────────────────────────────────

def _ax(ax, title, xlabel, ylabel, xlim=None):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, loc='left', pad=4)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    if xlim: ax.set_xlim(*xlim)


def spec_panel(ax, x, sr, fmax, cmap, title):
    nperseg = 2048
    f_s, t_s, S = dsp.spectrogram(x, fs=sr, window='hann',
                                    nperseg=min(nperseg, len(x)),
                                    noverlap=nperseg * 3 // 4)
    fm = f_s <= fmax
    Sdb = 10 * np.log10(S[fm] + 1e-30)
    ax.set_facecolor(PANEL)
    ax.grid(False)
    ax.pcolormesh(t_s, f_s[fm], Sdb, shading='gouraud', cmap=cmap,
                  vmin=np.percentile(Sdb, 5), vmax=np.percentile(Sdb, 99),
                  rasterized=True)
    duration = len(x) / sr
    _ax(ax, title, 'Time (s)', 'Freq (Hz)', xlim=(0, duration))
    return t_s, f_s[fm], Sdb


def spec_from_stft(ax, t_s, f_s, S_mag, fmax, cmap, title, duration):
    """Display a spectrogram from a pre-computed STFT magnitude."""
    fm = f_s <= fmax
    Sdb = 10 * np.log10(S_mag[fm] + 1e-30)
    ax.set_facecolor(PANEL)
    ax.grid(False)
    ax.pcolormesh(t_s, f_s[fm], Sdb, shading='gouraud', cmap=cmap,
                  vmin=np.percentile(Sdb, 5), vmax=np.percentile(Sdb, 99),
                  rasterized=True)
    _ax(ax, title, 'Time (s)', 'Freq (Hz)', xlim=(0, duration))


def plot_results(aud, cap_ds, recon,
                 Y_mag, X_est_mag, f_stft,
                 f_tf, H, coh_interp,
                 t_stft, lag_ms, fmax, r_env, audio_name, out_png):
    duration = len(aud) / AUD_SR
    xlim = (0, duration)
    fr = int(FRAME_S * AUD_SR)
    nf = min(len(aud), len(recon)) // fr
    aud_rms  = np.array([np.sqrt(np.mean(aud[i*fr:(i+1)*fr]**2))   for i in range(nf)])
    rec_rms  = np.array([np.sqrt(np.mean(recon[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    t_fr = np.arange(nf) * FRAME_S + FRAME_S / 2

    fig = plt.figure(figsize=(15, 14), facecolor=DARK)
    fig.suptitle(
        f'TF-based Reconstruction  —  "{audio_name}"\n'
        f'lag = {lag_ms:+.0f} ms  |  envelope r = {r_env:.3f}',
        color=TEXT, fontsize=11, fontweight='bold', y=0.99,
    )
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.55, wspace=0.32,
                           left=0.07, right=0.97, top=0.93, bottom=0.05)

    # ── Row 0: three spectrograms ─────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    spec_panel(ax00, aud, AUD_SR, fmax, 'magma', 'Reference audio')

    ax01 = fig.add_subplot(gs[0, 1])
    fm = f_stft <= fmax
    spec_from_stft(ax01, t_stft, f_stft, Y_mag, fmax, 'inferno',
                   'Powerline  |Y(f,t)|  (raw)', duration)

    ax02 = fig.add_subplot(gs[0, 2])
    spec_from_stft(ax02, t_stft, f_stft, X_est_mag, fmax, 'plasma',
                   'Estimated audio  |X̂(f,t)|  after TF⁻¹', duration)

    # ── Row 1: TF magnitude + coherence ──────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, :2])
    fmask = f_tf <= fmax
    H_db = 20 * np.log10(np.abs(H[fmask]) + 1e-30)
    H_coh = coh_interp[fmask] if coh_interp is not None else None
    ax10.set_facecolor(PANEL)
    ax10.fill_between(f_tf[fmask], H_db.min() - 5, H_db, color=BLUE, alpha=0.15)
    ax10.plot(f_tf[fmask], H_db, color=BLUE, lw=1.5, label='|H(f)| dB')
    ax10.set_xscale('log')
    _ax(ax10, 'Transfer function applied (|H(f)|)',
        'Frequency (Hz)', 'Magnitude (dB)', xlim=(max(f_tf[fmask][1], 20), fmax))
    ax10.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    ax11 = fig.add_subplot(gs[1, 2])
    ax11.set_facecolor(PANEL)
    ax11.scatter(aud_rms / (aud_rms.max() + 1e-9),
                 rec_rms / (rec_rms.max() + 1e-9),
                 c=t_fr, cmap='plasma', s=20, alpha=0.8)
    ax11.plot([0, 1], [0, 1], color=GRID, lw=1, ls='--')
    ax11.set_xlabel('Audio RMS (norm.)', color=MUTED, fontsize=8)
    ax11.set_ylabel('Reconstructed RMS (norm.)', color=MUTED, fontsize=8)
    ax11.set_title(f'Envelope scatter  (r = {r_env:.3f})', color=TEXT, fontsize=9, loc='left')
    ax11.grid(True, color=GRID, lw=0.5, alpha=0.6)
    for sp in ax11.spines.values(): sp.set_edgecolor(GRID)
    ax11.tick_params(colors=MUTED, labelsize=8)

    # ── Row 2: envelope comparison ────────────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, :])
    aud_rms_n = aud_rms / (aud_rms.max() + 1e-9)
    rec_rms_n = rec_rms / (rec_rms.max() + 1e-9)
    ax20.plot(t_fr, aud_rms_n, color=BLUE, lw=2.0, label='Reference audio RMS')
    ax20.plot(t_fr, rec_rms_n, color=PURP, lw=2.0, alpha=0.9, label='Reconstructed RMS')
    ax20.text(0.98, 0.92, f'r = {r_env:.3f}',
              transform=ax20.transAxes, color=YLW, fontsize=10, ha='right', va='top',
              bbox=dict(boxstyle='round,pad=0.3', fc=PANEL, ec=GRID))
    _ax(ax20, 'Envelope comparison (50 ms RMS)', 'Time (s)', 'Norm. RMS', xlim=xlim)
    ax20.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Row 3: waveform overlay + cross-correlation ───────────────────────────
    ax30 = fig.add_subplot(gs[3, :2])
    step = max(1, len(aud) // 60_000)
    t_a = np.linspace(0, duration, len(aud))
    t_r = np.linspace(0, duration, len(recon))
    aud_n   = aud   / (np.abs(aud).max()   + 1e-9)
    recon_n = recon / (np.abs(recon).max() + 1e-9)
    ax30.plot(t_a[::step],   aud_n[::step],   color=BLUE, lw=0.6, alpha=0.7, label='Audio')
    ax30.plot(t_r[::step], recon_n[::step], color=PURP, lw=0.8, alpha=0.85,
              label='Reconstructed')
    _ax(ax30, 'Waveform overlay', 'Time (s)', 'Norm. amplitude', xlim=xlim)
    ax30.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    ax31 = fig.add_subplot(gs[3, 2])
    n_cmp = min(len(aud), len(recon))
    a_n = aud[:n_cmp]   / (np.abs(aud[:n_cmp]).max()   + 1e-9)
    r_n = recon[:n_cmp] / (np.abs(recon[:n_cmp]).max() + 1e-9)
    xc  = np.correlate(a_n, r_n, mode='full') / n_cmp
    lags_ms = (np.arange(len(xc)) - n_cmp + 1) / AUD_SR * 1000
    zoom = np.abs(lags_ms) <= 500
    ax31.plot(lags_ms[zoom], xc[zoom], color=PURP, lw=0.8)
    pk     = xc[zoom].max()
    pk_lag = lags_ms[zoom][np.argmax(xc[zoom])]
    ax31.axvline(pk_lag, color=YLW, lw=1.5, ls='--',
                 label=f'peak={pk:.4f} @ {pk_lag:.1f} ms')
    ax31.axhline(0, color=GRID, lw=0.6)
    _ax(ax31, 'Cross-correlation', 'Lag (ms)', 'Xcorr', xlim=(-500, 500))
    ax31.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.patch.set_facecolor(DARK)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=DARK)
    print(f'[viz]   saved → {out_png}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture',  default=CAPTURE_PATH)
    ap.add_argument('--audio',    default=None,
                    help='Reference audio for comparison (optional)')
    ap.add_argument('--tf',       default=TF_PATH,
                    help='Transfer function file (default: transfer_function.npz)')
    ap.add_argument('--fmax',     type=float, default=8000,
                    help='Upper frequency for analysis (Hz, default 8000)')
    ap.add_argument('--nperseg',  type=int,   default=2048,
                    help='STFT window length (default 2048)')
    ap.add_argument('--reg',      type=float, default=0.02,
                    help='Wiener regularisation λ/max(|H|²), default 0.02')
    ap.add_argument('--coh-floor',type=float, default=0.1,
                    help='Coherence floor below which H bins are clamped, default 0.1')
    ap.add_argument('--gl-iters', type=int,   default=60,
                    help='Griffin-Lim iterations (default 60)')
    ap.add_argument('--no-show',  action='store_true')
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use('Agg')

    # ── load TF ───────────────────────────────────────────────────────────────
    if not os.path.exists(args.tf):
        sys.exit(f'[error] transfer function not found: {args.tf}\n'
                 f'        Run tf_estimate.py first with a sweep capture.')
    f_tf, H_tf, coh_tf, sr_tf = load_tf(args.tf)

    # ── load signals ──────────────────────────────────────────────────────────
    if not os.path.exists(args.capture):
        sys.exit(f'[error] capture file not found: {args.capture}')
    cap = load_capture(args.capture)

    audio_path = args.audio or find_audio(SCRIPT_DIR)
    has_ref    = audio_path is not None
    if has_ref:
        print(f'[audio] {os.path.basename(audio_path)}')
        aud = decode_audio(audio_path)
    else:
        print('[audio] No reference audio — proceeding without comparison')

    # ── downsample ────────────────────────────────────────────────────────────
    print('[ds]    downsampling capture 200 kHz → 22050 Hz …')
    cap_ds = downsample(cap, CAP_SR, AUD_SR)

    # ── lag ───────────────────────────────────────────────────────────────────
    lag_path = os.path.splitext(args.capture)[0] + '.lag'
    if has_ref:
        if os.path.exists(lag_path):
            lag_ms = float(open(lag_path).read().strip())
            print(f'[lag]   read from sidecar: {lag_ms:+.0f} ms')
        else:
            n = min(len(aud), len(cap_ds))
            lag_ms = detect_lag(aud[:n], cap_ds[:n])
            print(f'[lag]   auto-detected: {lag_ms:+.0f} ms')
        aud, cap_ds = apply_lag(aud, cap_ds, lag_ms)
    else:
        lag_ms = 0.0
        n = len(cap_ds)

    duration = len(cap_ds) / AUD_SR
    print(f'[align] {len(cap_ds):,} samples  ({duration:.4f}s)')

    # ── STFT of powerline capture ─────────────────────────────────────────────
    nperseg  = min(args.nperseg, len(cap_ds) // 4)
    noverlap = nperseg * 3 // 4
    f_stft, t_stft, Y = dsp.stft(cap_ds, fs=AUD_SR, window='hann',
                                   nperseg=nperseg, noverlap=noverlap, boundary=None)
    Y_mag = np.abs(Y)
    print(f'[STFT]  Y shape={Y.shape}  '
          f'freq_res={AUD_SR/nperseg:.1f} Hz  '
          f'time_res={nperseg*(1 - noverlap/nperseg)/AUD_SR*1000:.1f} ms')

    # ── interpolate H onto STFT frequency grid ────────────────────────────────
    H_interp = interp_tf(f_tf, H_tf, coh_tf, f_stft, args.coh_floor)
    print(f'[TF]    interpolated to {len(f_stft)} STFT bins')

    # ── Wiener deconvolution ──────────────────────────────────────────────────
    print(f'[decon] Wiener deconvolution  reg={args.reg}  …')
    X_est = wiener_deconvolve(Y, H_interp, args.reg)
    X_est_mag = np.abs(X_est)

    # Normalise estimated spectrogram for display
    # (Wiener output amplitude is relative; scale to match reference or raw cap)
    Y_energy   = float(Y_mag.mean())
    Xe_energy  = float(X_est_mag.mean())
    if Xe_energy > 0:
        X_est_mag_disp = X_est_mag * Y_energy / Xe_energy
    else:
        X_est_mag_disp = X_est_mag

    print(f'[decon] done  |X̂| range: [{X_est_mag.min():.3e}, {X_est_mag.max():.3e}]')

    # ── Griffin-Lim audio reconstruction ─────────────────────────────────────
    print(f'[GL]    Griffin-Lim  {args.gl_iters} iterations …')
    np.random.seed(42)
    recon = griffin_lim(X_est_mag, AUD_SR, nperseg, noverlap, args.gl_iters)
    print(f'[GL]    output: {len(recon)/AUD_SR:.3f}s  peak={np.abs(recon).max():.4f}')

    # ── envelope correlation (if reference available) ─────────────────────────
    r_env = 0.0
    if has_ref:
        fr = int(FRAME_S * AUD_SR)
        n_ref = len(aud)
        n_rec = len(recon)
        nf = min(n_ref, n_rec) // fr
        aud_rms = np.array([np.sqrt(np.mean(aud[i*fr:(i+1)*fr]**2))   for i in range(nf)])
        rec_rms = np.array([np.sqrt(np.mean(recon[i*fr:(i+1)*fr]**2)) for i in range(nf)])
        r_env, _ = pearsonr(aud_rms, rec_rms)
        print(f'[stat]  envelope r = {r_env:.4f}')

    # ── save WAV ──────────────────────────────────────────────────────────────
    out_wav = os.path.join(SCRIPT_DIR, 'reconstructed_tf.wav')
    save_wav(out_wav, recon, AUD_SR)

    # ── plot ──────────────────────────────────────────────────────────────────
    ref_aud = aud if has_ref else np.zeros_like(cap_ds)
    audio_name = os.path.basename(audio_path) if has_ref else '(no reference)'
    out_png = os.path.join(SCRIPT_DIR, 'tf_reconstruct.png')

    plot_results(
        ref_aud, cap_ds, recon,
        Y_mag, X_est_mag_disp, f_stft,
        f_tf, H_tf, coh_tf,
        t_stft, lag_ms, args.fmax, r_env,
        audio_name, out_png,
    )

    print()
    print('══ Results ════════════════════════════════════════')
    print(f'  Lag             : {lag_ms:+.0f} ms')
    print(f'  Regularisation  : {args.reg}')
    print(f'  Coherence floor : {args.coh_floor}')
    print(f'  Griffin-Lim     : {args.gl_iters} iters')
    if has_ref:
        print(f'  Envelope r      : {r_env:.4f}')
    print(f'  Output WAV      : reconstructed_tf.wav')
    print()
    print('  Listen:')
    print('    ffplay -nodisp reconstructed_tf.wav')
    if has_ref:
        print('    ffplay -nodisp original_reference.wav')
    print('═══════════════════════════════════════════════════')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
