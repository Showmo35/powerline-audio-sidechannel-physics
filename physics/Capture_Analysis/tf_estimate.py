#!/usr/bin/env python3
"""
tf_estimate.py — Estimate the powerline transfer function H(f) from a sweep capture.

Run this AFTER capturing the powerline response to a known sweep signal
(sweep_cal.wav or stepped_tones_sweep.mp3) via simple_rx.py.

Algorithm — H1 estimator (optimal for additive output noise):
    H(f) = Syx(f) / Sxx(f)
where
    x(t) = audio input (the known sweep)
    y(t) = powerline capture (resampled to AUD_SR)
    Syx  = cross-power spectral density  E[Y(f) · conj(X(f))]
    Sxx  = power spectral density of audio  E[|X(f)|²]

Quality metric — magnitude-squared coherence:
    γ²(f) = |Syx(f)|² / (Sxx(f) · Syy(f))   ∈ [0, 1]
    γ² ≈ 1 → H(f) reliable;  γ² ≈ 0 → noise floor (no useful coupling)

Outputs:
  transfer_function.npz   f, H_complex, H_mag_db, H_phase_deg, coherence
  tf_estimate.png         4-panel analysis figure

Usage:
    python3 tf_estimate.py
    python3 tf_estimate.py --audio sweep_cal.wav --capture capture.bin
    python3 tf_estimate.py --nperseg 8192 --fmax 10000 --no-show
"""

import argparse
import glob
import os
import subprocess
import sys
from math import gcd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as dsp
from scipy.signal import csd, welch, coherence

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH = os.path.join(SCRIPT_DIR, 'capture.bin')
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


# ── I/O ───────────────────────────────────────────────────────────────────────

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


# ── lag detection ─────────────────────────────────────────────────────────────

def detect_lag(aud, cap_ds, frame_s=FRAME_S):
    a_f = int(frame_s * AUD_SR)
    c_f = int(frame_s * AUD_SR)   # cap already resampled to AUD_SR
    n = min(len(aud) // a_f, len(cap_ds) // c_f)
    if n < 4:
        return 0.0

    a_rms = np.array([np.sqrt(np.mean(aud[i*a_f:(i+1)*a_f]**2)) for i in range(n)])
    c_rms = np.array([np.sqrt(np.mean(cap_ds[i*c_f:(i+1)*c_f]**2)) for i in range(n)])

    a = (a_rms - a_rms.mean()) / (a_rms.std() + 1e-12)
    c = (c_rms - c_rms.mean()) / (c_rms.std() + 1e-12)
    xcorr = np.correlate(a, c, mode='full') / n
    lags  = np.arange(-(n - 1), n)

    max_lag_f = int(round(0.700 / frame_s))
    valid = (-max_lag_f <= lags) & (lags <= 0)
    numpy_lag = int(lags[valid][np.argmax(xcorr[valid])])
    lag_ms = -numpy_lag * frame_s * 1000
    return lag_ms


def apply_lag(aud, cap_ds, lag_ms, frame_s=FRAME_S):
    """Trim lag_ms from front of cap_ds so it aligns with aud."""
    lag_samp = int(round(abs(lag_ms) / 1000 * AUD_SR))
    if lag_ms > 0 and lag_samp > 0:
        cap_ds = cap_ds[lag_samp:]
    elif lag_ms < 0 and lag_samp > 0:
        cap_ds = np.r_[np.zeros(lag_samp, dtype=np.float32), cap_ds]
    n = min(len(aud), len(cap_ds))
    return aud[:n], cap_ds[:n]


# ── stepped-tone analysis ─────────────────────────────────────────────────────

def analyse_stepped_tones(aud, cap_ds, sr, min_rms_frac=0.1):
    """
    Detect stepped-tone segments from the audio RMS envelope.

    For each tone segment runs a per-segment H1 estimator
    (Syx / Sxx over that segment's windows only) — this gives correct
    coherence and magnitude even for stepped-tone sweeps where the global
    H1 estimator fails (different bins excited in different windows).

    Returns (freqs, H_complex_per_tone, coherence_per_tone) each shape [n_tones],
    or (None, None, None) if signal doesn't look like stepped tones.
    """
    frame = int(0.1 * sr)   # 100 ms
    n = len(aud) // frame
    rms = np.array([np.sqrt(np.mean(aud[i*frame:(i+1)*frame]**2)) for i in range(n)])
    thr = min_rms_frac * rms.max()
    active = rms > thr

    # Find contiguous active runs
    padded = np.r_[False, active, False]
    starts = np.where(np.diff(padded.astype(int)) == 1)[0]
    ends   = np.where(np.diff(padded.astype(int)) == -1)[0]

    if len(starts) < 2:
        return None, None, None   # not stepped tones

    freqs, H_vals, coh_vals = [], [], []
    # nperseg chosen to give ≥ 8 Welch windows per tone segment so coherence
    # estimates are meaningful.  A 0.8 s tone (middle 60%) ≈ 13 000 samples;
    # nperseg=1024 → ~25 windows, freq-res=21.5 Hz (enough to resolve tone peaks).
    nperseg_tone = 1024

    for s, e in zip(starts, ends):
        dur = (e - s) * 0.1
        if dur < 0.3:
            continue
        # Middle 60% of each run to avoid click transients
        trim = int((e - s) * 0.2)
        s2, e2 = s + trim, e - trim
        if s2 >= e2:
            s2, e2 = s, e

        seg_aud = aud   [s2*frame : e2*frame]
        seg_cap = cap_ds[s2*frame : e2*frame]
        nw = min(nperseg_tone, len(seg_aud) // 4)
        if nw < 64:
            continue

        # Identify centre frequency from audio PSD peak
        f_a, P_a = welch(seg_aud, fs=sr, nperseg=nw, window='hann')
        peak_idx = int(np.argmax(P_a))
        fc = float(f_a[peak_idx])
        if fc < 10 or fc > sr / 2 - 10:
            continue

        # Amplitude-ratio |H| at the tone peak bin (robust to low per-segment SNR)
        f_c, P_c = welch(seg_cap, fs=sr, nperseg=nw, window='hann')
        H_mag = float(np.sqrt(P_c[peak_idx] / (P_a[peak_idx] + 1e-60)))

        # Phase from cross-spectrum at the tone peak bin
        kw = dict(fs=sr, nperseg=nw, noverlap=nw // 2, window='hann')
        _, Syx     = csd(seg_cap, seg_aud, **kw)
        _, coh_seg = coherence(seg_aud, seg_cap, **kw)

        H_phase = float(np.angle(Syx[peak_idx]))
        coh_fc  = float(coh_seg[peak_idx])

        H_fc = H_mag * np.exp(1j * H_phase)
        freqs.append(fc)
        H_vals.append(H_fc)
        coh_vals.append(coh_fc)

    if len(freqs) < 2:
        return None, None, None

    order = np.argsort(freqs)
    return (np.array(freqs)[order],
            np.array(H_vals,  dtype=np.complex128)[order],
            np.array(coh_vals, dtype=np.float64)[order])


# ── H1 estimator ──────────────────────────────────────────────────────────────

def h1_estimate(aud, cap_ds, sr, nperseg):
    noverlap = nperseg * 3 // 4
    kw = dict(fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann')

    f, Pyx = csd(cap_ds, aud, **kw)          # E[Y · conj(X)]
    _, Pxx = welch(aud,    **kw)              # E[|X|²]
    _, coh = coherence(aud, cap_ds, **kw)    # |Pyx|² / (Pxx · Pyy)

    H = Pyx / (Pxx + 1e-30)
    return f.astype(np.float64), H, coh.astype(np.float64)


# ── plot ──────────────────────────────────────────────────────────────────────

def _ax(ax, title, xlabel, ylabel, xlim=None, ylim=None, xlog=False):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, loc='left', pad=4)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    if xlog: ax.set_xscale('log')


def plot_tf(f, H, coh, aud, cap_ds, sr, fmax, audio_name,
            tone_freqs, tone_mags, lag_ms, out_png):
    fmask = f <= fmax
    f_plot = f[fmask]
    H_mag  = np.abs(H[fmask])
    H_db   = 20 * np.log10(H_mag + 1e-30)
    H_ph   = np.degrees(np.angle(H[fmask]))
    coh_p  = coh[fmask]

    duration = len(aud) / sr
    xlim = (max(f_plot[1], 20), fmax)

    fig = plt.figure(figsize=(14, 12), facecolor=DARK)
    fig.suptitle(
        f'Powerline Transfer Function H(f)  —  "{audio_name}"\n'
        f'lag = {lag_ms:+.0f} ms  |  '
        f'mean coherence = {coh_p.mean():.3f}  |  '
        f'mean |H| = {H_db.mean():.1f} dB',
        color=TEXT, fontsize=11, fontweight='bold', y=0.99,
    )

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.5, wspace=0.35,
                           left=0.08, right=0.97, top=0.93, bottom=0.06)

    # ── Panel 1: TF magnitude ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _ax(ax1, '|H(f)| — Transfer function magnitude',
        'Frequency (Hz)', 'Magnitude (dB)', xlim=xlim, xlog=True)

    ax1.fill_between(f_plot, H_db.min() - 5, H_db, color=BLUE, alpha=0.15)
    ax1.plot(f_plot, H_db, color=BLUE, lw=1.5, label='H1 estimate')

    if tone_freqs is not None:
        tone_db = 20 * np.log10(tone_mags + 1e-30)
        ax1.scatter(tone_freqs, tone_db, color=YLW, s=80, zorder=5,
                    label='Per-tone ratio', edgecolors=DARK, lw=0.8)
        ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # Annotate peak
    peak_idx = np.argmax(H_db)
    ax1.annotate(
        f'peak: {H_db[peak_idx]:.1f} dB\n@ {f_plot[peak_idx]:.0f} Hz',
        xy=(f_plot[peak_idx], H_db[peak_idx]),
        xytext=(f_plot[peak_idx] * 1.5, H_db[peak_idx] + 2),
        color=YLW, fontsize=8,
        arrowprops=dict(arrowstyle='->', color=YLW, lw=1.0),
    )

    # ── Panel 2: TF phase ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _ax(ax2, '∠H(f) — Phase response',
        'Frequency (Hz)', 'Phase (°)', xlim=xlim, ylim=(-200, 200), xlog=True)

    # Mask low-coherence bins to avoid noisy phase
    coh_thr = 0.5
    phase_plot = H_ph.copy()
    phase_plot[coh_p < coh_thr] = np.nan
    ax2.plot(f_plot, phase_plot, color=ORNG, lw=1.2, label=f'Phase (γ²≥{coh_thr})')
    ax2.axhline(0, color=GRID, lw=0.8, ls='--')
    ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 3: Coherence ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _ax(ax3, 'γ²(f) — Coherence (measurement quality)',
        'Frequency (Hz)', 'Coherence γ²', xlim=xlim, ylim=(0, 1.05), xlog=True)

    ax3.fill_between(f_plot, 0, coh_p,
                     where=coh_p >= 0.5, color=GRN, alpha=0.5, label='Good (γ²≥0.5)')
    ax3.fill_between(f_plot, 0, coh_p,
                     where=coh_p < 0.5, color=RED, alpha=0.4, label='Poor (γ²<0.5)')
    ax3.plot(f_plot, coh_p, color=GRN, lw=1.0)
    ax3.axhline(0.5, color=YLW, lw=0.8, ls='--', label='γ²=0.5 threshold')
    ax3.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    good_frac = (coh_p >= 0.5).mean() * 100
    ax3.text(0.02, 0.92,
             f'{good_frac:.0f}% of bins with γ²≥0.5',
             transform=ax3.transAxes, color=YLW, fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', fc=PANEL, ec=GRID))

    # ── Panel 4: spectrogram comparison ──────────────────────────────────────
    nperseg_sp = 2048
    noverlap_sp = nperseg_sp * 3 // 4

    def spec(x, sr, fmax):
        f_s, t_s, S = dsp.spectrogram(x, fs=sr, window='hann',
                                        nperseg=min(nperseg_sp, len(x)),
                                        noverlap=noverlap_sp)
        fm = f_s <= fmax
        return t_s, f_s[fm], 10 * np.log10(S[fm] + 1e-30)

    ax4 = fig.add_subplot(gs[2, 0])
    t_s, f_s, S_aud = spec(aud, sr, fmax)
    ax4.set_facecolor(PANEL)
    ax4.grid(False)
    ax4.pcolormesh(t_s, f_s, S_aud, shading='gouraud', cmap='magma',
                   vmin=np.percentile(S_aud, 5), vmax=np.percentile(S_aud, 99),
                   rasterized=True)
    _ax(ax4, 'Audio spectrogram', 'Time (s)', 'Freq (Hz)',
        xlim=(0, duration))

    ax5 = fig.add_subplot(gs[2, 1])
    t_s, f_s, S_cap = spec(cap_ds, sr, fmax)
    ax5.set_facecolor(PANEL)
    ax5.grid(False)
    ax5.pcolormesh(t_s, f_s, S_cap, shading='gouraud', cmap='inferno',
                   vmin=np.percentile(S_cap, 5), vmax=np.percentile(S_cap, 99),
                   rasterized=True)
    _ax(ax5, 'Powerline spectrogram (downsampled to AUD_SR)',
        'Time (s)', 'Freq (Hz)', xlim=(0, duration))

    fig.patch.set_facecolor(DARK)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=DARK)
    print(f'[viz]   saved → {out_png}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture',  default=CAPTURE_PATH)
    ap.add_argument('--audio',    default=None)
    ap.add_argument('--fmax',     type=float, default=10000,
                    help='Upper frequency for analysis (Hz, default 10000)')
    ap.add_argument('--nperseg',  type=int,   default=8192,
                    help='FFT window length (default 8192; must be ≤ signal length)')
    ap.add_argument('--no-show',  action='store_true')
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use('Agg')

    # ── load ─────────────────────────────────────────────────────────────────
    if not os.path.exists(args.capture):
        sys.exit(f'[error] capture file not found: {args.capture}')
    cap = load_capture(args.capture)

    audio_path = args.audio or find_audio(SCRIPT_DIR)
    if not audio_path:
        sys.exit('[error] no audio file found; specify with --audio')
    print(f'[audio] {os.path.basename(audio_path)}')
    aud = decode_audio(audio_path)

    # ── downsample capture ────────────────────────────────────────────────────
    print('[ds]    downsampling capture 200 kHz → 22050 Hz …')
    cap_ds = downsample(cap, CAP_SR, AUD_SR)

    # ── lag ───────────────────────────────────────────────────────────────────
    lag_path = os.path.splitext(args.capture)[0] + '.lag'
    if os.path.exists(lag_path):
        lag_ms = float(open(lag_path).read().strip())
        print(f'[lag]   read from sidecar: {lag_ms:+.0f} ms')
    else:
        lag_ms = detect_lag(aud, cap_ds)
        print(f'[lag]   auto-detected: {lag_ms:+.0f} ms')

    aud, cap_ds = apply_lag(aud, cap_ds, lag_ms)
    print(f'[align] {len(aud):,} samples aligned  ({len(aud)/AUD_SR:.4f}s)')

    # ── H1 estimator ──────────────────────────────────────────────────────────
    nperseg = min(args.nperseg, len(aud) // 4)
    if nperseg < 64:
        sys.exit(f'[error] signal too short for nperseg={args.nperseg}; '
                 f'need at least {args.nperseg * 4} samples ({args.nperseg * 4 / AUD_SR:.1f}s)')
    print(f'[H1]    nperseg={nperseg}  '
          f'(freq resolution={AUD_SR/nperseg:.2f} Hz  '
          f'n_windows≈{len(aud)//nperseg * 2 - 1})')

    f, H, coh = h1_estimate(aud, cap_ds, AUD_SR, nperseg)
    fmask = f <= args.fmax
    print(f'[H1]    H(f) computed  {fmask.sum()} bins up to {args.fmax:.0f} Hz')

    H_db_mean  = 20 * np.log10(np.abs(H[fmask]).mean() + 1e-30)
    coh_mean   = float(coh[fmask].mean())
    good_frac  = float((coh[fmask] >= 0.5).mean()) * 100
    print(f'[stat]  mean |H| = {H_db_mean:.1f} dB  '
          f'mean coherence = {coh_mean:.3f}  '
          f'good bins (γ²≥0.5) = {good_frac:.0f}%')

    # ── per-tone analysis + choose best H(f) ─────────────────────────────────
    tone_freqs, tone_H, tone_coh = analyse_stepped_tones(aud, cap_ds, AUD_SR)
    tone_mags = np.abs(tone_H) if tone_H is not None else None

    if tone_freqs is not None and len(tone_freqs) >= 4:
        print(f'[tones] Detected {len(tone_freqs)} tones → rebuilding H(f) '
              f'from per-segment H1 measurements')
        for fc, hm, hc in zip(tone_freqs, tone_mags, tone_coh):
            print(f'        {fc:7.1f} Hz  |H|={20*np.log10(hm+1e-30):+6.1f} dB  '
                  f'γ²={hc:.3f}')

        # Interpolate onto the full frequency grid (log-log for magnitude)
        log_f_tone = np.log(tone_freqs)
        log_H_mag  = np.log(tone_mags + 1e-30)
        H_ph       = np.unwrap(np.angle(tone_H))
        safe_f     = np.maximum(f, tone_freqs[0] * 0.5)

        H_mag_interp = np.exp(np.interp(np.log(safe_f), log_f_tone, log_H_mag,
                                         left=log_H_mag[0], right=log_H_mag[-1]))
        H_ph_interp  = np.interp(f, tone_freqs, H_ph,
                                  left=H_ph[0], right=H_ph[-1])
        H   = (H_mag_interp * np.exp(1j * H_ph_interp)).astype(np.complex128)
        coh = np.clip(np.interp(f, tone_freqs, tone_coh, left=0.0, right=0.0),
                      0.0, 1.0)
        coh[f < tone_freqs[0]] = 0.0
        coh[f > tone_freqs[-1]] = 0.0

        H_db_mean = 20 * np.log10(np.abs(H[fmask]).mean() + 1e-30)
        coh_mean  = float(coh[fmask].mean())
        good_frac = float((coh[fmask] >= 0.5).mean()) * 100
        print(f'[tones] rebuilt  mean |H|={H_db_mean:.1f} dB  '
              f'mean γ²={coh_mean:.3f}  good={good_frac:.0f}%')
    else:
        print('[tones] No stepped tones detected — using global H1 estimate')
        tone_freqs = tone_mags = None

    # ── save TF ───────────────────────────────────────────────────────────────
    out_npz = os.path.join(SCRIPT_DIR, 'transfer_function.npz')
    np.savez(out_npz,
             f          = f,
             H_complex  = H,
             H_mag_db   = 20 * np.log10(np.abs(H) + 1e-30),
             H_phase_deg= np.degrees(np.angle(H)),
             coherence  = coh,
             samp_rate  = AUD_SR,
             fmax       = args.fmax,
             audio_file = os.path.basename(audio_path),
             tone_freqs = tone_freqs if tone_freqs is not None else np.array([]),
             tone_mags  = tone_mags  if tone_mags  is not None else np.array([]),
    )
    print(f'[save]  transfer_function.npz  ({os.path.getsize(out_npz)//1024} KB)')

    # ── plot ──────────────────────────────────────────────────────────────────
    out_png = os.path.join(SCRIPT_DIR, 'tf_estimate.png')
    plot_tf(f, H, coh, aud, cap_ds, AUD_SR, args.fmax,
            os.path.basename(audio_path),
            tone_freqs, tone_mags, lag_ms, out_png)

    print()
    print('══ Summary ════════════════════════════════════════')
    print(f'  Audio file     : {os.path.basename(audio_path)}')
    print(f'  Lag detected   : {lag_ms:+.0f} ms')
    print(f'  Mean |H(f)|    : {H_db_mean:.1f} dB')
    print(f'  Mean coherence : {coh_mean:.3f}')
    print(f'  Good bins      : {good_frac:.0f}%  (γ²≥0.5)')
    print(f'  Saved          : transfer_function.npz')
    print()
    print('  Next: run tf_reconstruct.py to apply H(f)⁻¹ to a new capture.')
    print('═══════════════════════════════════════════════════')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
