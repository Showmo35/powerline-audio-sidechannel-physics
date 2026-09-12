#!/usr/bin/env python3
"""
am_reconstruct.py — Reconstruct audio spectrogram via mains AM demodulation.

Physical model
--------------
The audio device's power supply draws current proportional to the audio signal.
This AM-modulates the mains carrier (50/60 Hz), producing sidebands:

    audio at f_audio  →  powerline spectrum shows energy at  n·f_mains ± f_audio

Unlike tf_reconstruct.py which looked for audio content AT audio frequencies
(where coherence was near zero), this script looks at the SIDEBAND positions
around each mains harmonic — where the audio information actually lives.

Pipeline
--------
1. Detect exact mains frequency from the long-term power spectrum of the capture
2. Optionally estimate H_AM(f) from a sweep capture:
       H_AM(f_k) = |Y_capture(f_mains + f_k)| / |Y_audio(f_k)|
3. Compute STFT of the speech capture
4. Build estimated audio STFT magnitude:
       S_audio_est(f, t) = Σ_n  |Y(n·f_mains + f, t)| + |Y(|n·f_mains − f|, t)|
5. Divide by H_AM(f) (or flat if no sweep TF available)
6. Griffin-Lim → reconstructed audio waveform
7. Side-by-side spectrogram comparison

Outputs
-------
  am_reconstruct.wav   reconstructed audio
  am_reconstruct.png   5-panel comparison figure

Usage
-----
    python3 am_reconstruct.py
    python3 am_reconstruct.py --sweep-capture sweep_capture.bin \\
                               --audio hello5.wav --n-harmonics 8
    python3 am_reconstruct.py --mains 60 --fmax 4000 --no-show
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
SWEEP_CAP    = os.path.join(SCRIPT_DIR, 'sweep_capture.bin')
SWEEP_WAV    = os.path.join(SCRIPT_DIR, 'sweep_cal.wav')
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

def detect_and_apply_lag(aud, cap_ds, lag_path=None):
    if lag_path and os.path.exists(lag_path):
        lag_ms = float(open(lag_path).read().strip())
        print(f'[lag]   read from sidecar: {lag_ms:+.0f} ms')
    else:
        f = int(FRAME_S * AUD_SR)
        n = min(len(aud) // f, len(cap_ds) // f)
        a = np.array([np.sqrt(np.mean(aud[i*f:(i+1)*f]**2)) for i in range(n)])
        c = np.array([np.sqrt(np.mean(cap_ds[i*f:(i+1)*f]**2)) for i in range(n)])
        an = (a - a.mean()) / (a.std() + 1e-12)
        cn = (c - c.mean()) / (c.std() + 1e-12)
        xc = np.correlate(an, cn, mode='full') / n
        lags = np.arange(-(n-1), n)
        mf = int(round(0.700 / FRAME_S))
        valid = (-mf <= lags) & (lags <= 0)
        lag_ms = -float(lags[valid][np.argmax(xc[valid])]) * FRAME_S * 1000
        print(f'[lag]   auto-detected: {lag_ms:+.0f} ms')

    lag_samp = int(round(abs(lag_ms) / 1000 * AUD_SR))
    if lag_ms > 0 and lag_samp > 0:
        cap_ds = cap_ds[lag_samp:]
    elif lag_ms < 0 and lag_samp > 0:
        cap_ds = np.r_[np.zeros(lag_samp, dtype=np.float32), cap_ds]
    n = min(len(aud), len(cap_ds))
    return aud[:n].copy(), cap_ds[:n].copy(), lag_ms


# ── mains detection ───────────────────────────────────────────────────────────

def detect_mains(cap_ds, sr, mains_guess=50.0, search_hz=5.0):
    """
    Detect the exact mains frequency from the long-term average spectrum.
    Searches within mains_guess ± search_hz Hz for the dominant peak.
    """
    nperseg = min(65536, len(cap_ds))
    f, Pxx = dsp.welch(cap_ds, fs=sr, nperseg=nperseg, window='blackman')
    mask = np.abs(f - mains_guess) < search_hz
    if not mask.any():
        print(f'[mains] no peak near {mains_guess:.0f} Hz — using guess')
        return mains_guess
    peak_f = float(f[mask][np.argmax(Pxx[mask])])
    peak_db = 10 * np.log10(Pxx[mask].max() / (Pxx.mean() + 1e-30))
    print(f'[mains] detected f_mains = {peak_f:.3f} Hz  '
          f'(peak is {peak_db:.1f} dB above mean)')
    return peak_f


# ── AM transfer function from sweep ──────────────────────────────────────────

def estimate_h_am(sweep_cap_ds, sweep_aud, f_mains, sr, nperseg_stft,
                  min_rms_frac=0.05):
    """
    Estimate H_AM(f_audio) from a stepped-tone or chirp sweep.

    For each detected tone at f_k in the sweep audio, measure:
        |Y_capture(f_mains + f_k, t)|   (upper sideband)
        |Y_audio(f_k, t)|               (direct audio power)
    and compute H_AM(f_k) = sideband_amp / audio_amp.

    Returns (f_tones, H_am_mag) arrays at measured frequencies,
    or (None, None) if no tones detected or sweep too short.
    """
    noverlap = nperseg_stft * 3 // 4
    kw = dict(fs=sr, nperseg=nperseg_stft, noverlap=noverlap,
              window='hann', boundary=None)

    f_stft, t_stft, Y_cap = dsp.stft(sweep_cap_ds, **kw)
    _,      _,      Y_aud = dsp.stft(sweep_aud,     **kw)
    df = float(f_stft[1] - f_stft[0])

    # Detect active tone segments from audio RMS envelope
    frame = int(0.1 * sr)
    n_fr = len(sweep_aud) // frame
    rms = np.array([np.sqrt(np.mean(sweep_aud[i*frame:(i+1)*frame]**2))
                    for i in range(n_fr)])
    thr = min_rms_frac * rms.max()
    active = rms > thr
    padded = np.r_[False, active, False]
    starts = np.where(np.diff(padded.astype(int)) == 1)[0]
    ends   = np.where(np.diff(padded.astype(int)) == -1)[0]

    if len(starts) < 1:
        print('[H_AM]  no tone segments detected in sweep audio')
        return None, None

    f_tones, H_mags = [], []

    for s, e in zip(starts, ends):
        dur = (e - s) * 0.1
        if dur < 0.3:
            continue
        # Middle 60% of segment to avoid transients
        trim = max(int((e - s) * 0.2), 1)
        s2, e2 = s + trim, e - trim
        if s2 >= e2:
            s2, e2 = s, e

        seg_aud = sweep_aud[s2*frame : e2*frame]
        if len(seg_aud) < nperseg_stft:
            continue

        # Identify tone frequency from audio PSD peak
        nw_tone = min(4096, len(seg_aud) // 4)
        if nw_tone < 64:
            continue
        f_a, P_a = dsp.welch(seg_aud, fs=sr, nperseg=nw_tone, window='hann')
        peak_idx = int(np.argmax(P_a))
        fc = float(f_a[peak_idx])
        if fc < 10 or fc > sr / 2 - f_mains - 10:
            continue

        # Map STFT time frames to this segment
        t_mask = (t_stft >= s2 * 0.1) & (t_stft <= e2 * 0.1)
        if t_mask.sum() < 2:
            continue

        # Audio power at fc
        fc_bin = int(round(fc / df))
        fc_bin = np.clip(fc_bin, 0, len(f_stft) - 1)
        aud_amp = float(np.abs(Y_aud[fc_bin, t_mask]).mean())

        # Sideband amplitude at f_mains + fc (upper sideband)
        fsb = f_mains + fc
        fsb_bin = int(round(fsb / df))
        fsb_bin = np.clip(fsb_bin, 0, len(f_stft) - 1)

        # Reject if sideband position coincides with a mains harmonic
        # (the harmonic power would dominate the measurement)
        nearest_harmonic = round(fsb / f_mains) * f_mains
        if abs(fsb - nearest_harmonic) < 1.5 * df:
            print(f'  f={fc:7.1f} Hz  sideband@{fsb:.1f}Hz  '
                  f'SKIPPED (coincides with mains harmonic @{nearest_harmonic:.1f} Hz)')
            continue

        sb_amp = float(np.abs(Y_cap[fsb_bin, t_mask]).mean())

        if aud_amp < 1e-10:
            continue

        H_mag = sb_amp / aud_amp
        f_tones.append(fc)
        H_mags.append(H_mag)
        print(f'  f={fc:7.1f} Hz  sideband@{fsb:.1f}Hz  '
              f'|H_AM|={20*np.log10(H_mag+1e-30):+6.1f} dB')

    if len(f_tones) < 2:
        return None, None

    order = np.argsort(f_tones)
    return (np.array(f_tones)[order],
            np.array(H_mags)[order])


# ── AM sideband reconstruction ────────────────────────────────────────────────

def am_reconstruct_stft(cap_ds, f_mains, n_harmonics, sr, nperseg,
                        upper_only=False):
    """
    Build estimated audio STFT magnitude from AM sidebands.

    If upper_only=True (recommended for spectrogram fidelity):
        |X_est(f_audio, t)| = |Y_cap(f_mains + f_audio, t)|

    This is a pure frequency-shift: the audio spectrogram is the powerline
    spectrogram shifted down by f_mains Hz.  It gives a 1-to-1 frequency
    mapping with no cross-contamination from lower sidebands or higher harmonics.

    If upper_only=False (better envelope SNR):
        For each audio frequency bin f_audio and each mains harmonic n:
            |X_est(f_audio, t)| += |Y_cap(n·f_mains + f_audio, t)|   (upper)
                                 + |Y_cap(|n·f_mains − f_audio|, t)| (lower)

    Returns (f_stft, t_stft, Y_abs, X_est_mag).
    """
    import warnings
    noverlap = nperseg * 3 // 4
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f_stft, t_stft, Y = dsp.stft(cap_ds, fs=sr, nperseg=nperseg,
                                       noverlap=noverlap, window='hann')
    Y_abs = np.abs(Y)
    df = float(f_stft[1] - f_stft[0])
    n_f = len(f_stft)

    X_est_mag = np.zeros_like(Y_abs)

    if upper_only:
        # Pure frequency shift: audio at f ← capture at f_mains + f
        f_upper = f_mains + f_stft                       # [n_f]
        sb_bins = np.round(f_upper / df).astype(int)
        valid   = (sb_bins >= 0) & (sb_bins < n_f)
        X_est_mag[valid, :] = Y_abs[sb_bins[valid], :]
        print(f'[AM]    upper-only shift: audio(f) ← capture({f_mains:.1f}+f Hz)')
        print(f'        valid bins: {valid.sum()}/{n_f}  '
              f'({100*valid.sum()/n_f:.0f}%  up to '
              f'{f_stft[valid][-1]:.0f} Hz audio)')
    else:
        for n in range(1, n_harmonics + 1):
            # Upper sideband: audio at f → capture at n·f_mains + f
            f_upper = n * f_mains + f_stft           # [n_f]
            # Lower sideband: audio at f → capture at |n·f_mains − f|
            f_lower = np.abs(n * f_mains - f_stft)  # [n_f]

            for f_sb in [f_upper, f_lower]:
                sb_bins = np.round(f_sb / df).astype(int)
                valid   = (sb_bins >= 0) & (sb_bins < n_f)
                X_est_mag[valid, :] += Y_abs[sb_bins[valid], :]

        contributions = n_harmonics * 2
        X_est_mag /= contributions
        print(f'[AM]    sideband accumulation: {n_harmonics} harmonics × 2 '
              f'(upper+lower) = {contributions} maps per audio bin')

    print(f'[AM]    |X_est| range: [{X_est_mag.min():.3e}, {X_est_mag.max():.3e}]')
    return f_stft, t_stft, Y_abs, X_est_mag


def apply_h_am(X_est_mag, f_stft, f_tones, H_am_mag, reg=0.01):
    """
    Divide estimated audio magnitude by H_AM(f) (interpolated from tone measurements).
    Regularises where H_AM is very small to avoid noise amplification.
    """
    if f_tones is None or H_am_mag is None:
        print('[H_AM]  no calibration — using flat H_AM = 1')
        return X_est_mag.copy()

    # Log-log interpolation for smooth magnitude response
    log_f = np.log(f_tones)
    log_H = np.log(H_am_mag + 1e-30)
    safe_f = np.maximum(f_stft, f_tones[0] * 0.5)
    H_interp = np.exp(np.interp(np.log(safe_f), log_f, log_H,
                                  left=log_H[0], right=log_H[-1]))
    H_interp[f_stft < f_tones[0]] = H_am_mag[0]
    H_interp[f_stft > f_tones[-1]] = H_am_mag[-1]

    lam = reg * H_interp.max()
    X_cal = X_est_mag / (H_interp[:, None] + lam)
    print(f'[H_AM]  calibrated  |H_AM| range: [{20*np.log10(H_interp.min()+1e-30):.1f}, '
          f'{20*np.log10(H_interp.max()+1e-30):.1f}] dB')
    return X_cal


# ── Griffin-Lim ───────────────────────────────────────────────────────────────

def griffin_lim(mag, sr, nperseg, noverlap, n_iter=60):
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


def stft_panel(ax, t, f, S_mag, fmax, cmap, title, duration):
    fm = f <= fmax
    Sdb = 10 * np.log10(S_mag[fm] + 1e-30)
    ax.set_facecolor(PANEL); ax.grid(False)
    ax.pcolormesh(t, f[fm], Sdb, shading='gouraud', cmap=cmap,
                  vmin=np.percentile(Sdb, 5), vmax=np.percentile(Sdb, 99),
                  rasterized=True)
    _ax(ax, title, 'Time (s)', 'Freq (Hz)', xlim=(0, duration))


def plot_results(aud, cap_ds, recon,
                 f_stft, t_stft, Y_abs, X_est_mag,
                 f_mains, n_harmonics, lag_ms, r_env, fmax,
                 f_tones, H_am_mag,
                 audio_name, out_png):
    duration = len(aud) / AUD_SR
    xlim = (0, duration)

    # Reference audio STFT
    nperseg_ref = 2048
    f_r, t_r, Yr = dsp.stft(aud, fs=AUD_SR, nperseg=nperseg_ref,
                              noverlap=nperseg_ref*3//4, window='hann')

    # Envelope comparison
    fr = int(FRAME_S * AUD_SR)
    nf = min(len(aud), len(recon)) // fr
    aud_rms  = np.array([np.sqrt(np.mean(aud[i*fr:(i+1)*fr]**2))   for i in range(nf)])
    rec_rms  = np.array([np.sqrt(np.mean(recon[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    t_fr = np.arange(nf) * FRAME_S + FRAME_S / 2

    fig = plt.figure(figsize=(15, 14), facecolor=DARK)
    fig.suptitle(
        f'AM Sideband Reconstruction  —  "{audio_name}"\n'
        f'f_mains={f_mains:.2f} Hz  |  {n_harmonics} harmonics  |  '
        f'lag={lag_ms:+.0f} ms  |  envelope r={r_env:.3f}',
        color=TEXT, fontsize=11, fontweight='bold', y=0.99,
    )
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.55, wspace=0.32,
                           left=0.07, right=0.97, top=0.93, bottom=0.05)

    # Row 0: three spectrograms
    ax00 = fig.add_subplot(gs[0, 0])
    stft_panel(ax00, t_r, f_r, np.abs(Yr), fmax, 'magma',
               'Reference audio', duration)

    ax01 = fig.add_subplot(gs[0, 1])
    stft_panel(ax01, t_stft, f_stft, Y_abs, fmax, 'inferno',
               'Powerline  |Y(f,t)|  (raw, shifted-down view)', duration)

    ax02 = fig.add_subplot(gs[0, 2])
    stft_panel(ax02, t_stft, f_stft, X_est_mag, fmax, 'plasma',
               f'AM sideband estimate  |X̂(f,t)|', duration)

    # Row 1: mains spectrum + H_AM
    ax10 = fig.add_subplot(gs[1, :2])
    f_spec, Pxx = dsp.welch(cap_ds, fs=AUD_SR, nperseg=min(32768, len(cap_ds)),
                              window='blackman')
    fm = f_spec <= fmax * 1.1
    ax10.set_facecolor(PANEL)
    ax10.fill_between(f_spec[fm], 10*np.log10(Pxx[fm]+1e-30).min(),
                      10*np.log10(Pxx[fm]+1e-30), color=BLUE, alpha=0.2)
    ax10.plot(f_spec[fm], 10*np.log10(Pxx[fm]+1e-30), color=BLUE, lw=0.8,
              label='Powerline PSD')
    # Annotate mains harmonics
    for n in range(1, min(n_harmonics+1, int(fmax/f_mains)+2)):
        fh = n * f_mains
        if fh <= fmax * 1.1:
            ax10.axvline(fh, color=YLW, lw=0.7, ls='--', alpha=0.6)
    ax10.plot([], [], color=YLW, lw=1, ls='--', label=f'Mains harmonics ({f_mains:.1f} Hz)')
    _ax(ax10, f'Powerline PSD with mains harmonics (f_mains={f_mains:.2f} Hz)',
        'Frequency (Hz)', 'PSD (dB/Hz)', xlim=(0, fmax * 1.1))
    ax10.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    ax11 = fig.add_subplot(gs[1, 2])
    if f_tones is not None and H_am_mag is not None:
        ax11.set_facecolor(PANEL)
        H_db = 20 * np.log10(H_am_mag + 1e-30)
        ax11.plot(f_tones, H_db, color=ORNG, lw=1.5, label='|H_AM(f)| dB')
        ax11.scatter(f_tones, H_db, color=YLW, s=50, zorder=5, edgecolors=DARK, lw=0.8)
        _ax(ax11, 'AM coupling |H_AM(f)|', 'Audio freq (Hz)', 'dB',
            xlim=(0, fmax), xlog=True)
        ax11.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    else:
        ax11.set_facecolor(PANEL)
        ax11.text(0.5, 0.5, 'No sweep TF\n(flat H_AM=1)', transform=ax11.transAxes,
                  color=MUTED, ha='center', va='center', fontsize=10)
        for sp in ax11.spines.values(): sp.set_edgecolor(GRID)

    # Row 2: envelope comparison
    ax20 = fig.add_subplot(gs[2, :])
    aud_n = aud_rms / (aud_rms.max() + 1e-9)
    rec_n = rec_rms / (rec_rms.max() + 1e-9)
    ax20.plot(t_fr, aud_n, color=BLUE, lw=2.0, label='Reference audio RMS')
    ax20.plot(t_fr, rec_n, color=PURP, lw=2.0, alpha=0.9, label='AM reconstructed RMS')
    ax20.text(0.98, 0.92, f'r = {r_env:.3f}',
              transform=ax20.transAxes, color=YLW, fontsize=10, ha='right', va='top',
              bbox=dict(boxstyle='round,pad=0.3', fc=PANEL, ec=GRID))
    _ax(ax20, 'Envelope comparison (50 ms RMS)', 'Time (s)', 'Norm. RMS', xlim=xlim)
    ax20.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # Row 3: waveform + cross-correlation
    ax30 = fig.add_subplot(gs[3, :2])
    step = max(1, len(aud) // 60_000)
    t_a  = np.linspace(0, duration, len(aud))
    t_rc = np.linspace(0, duration, len(recon))
    aud_wn   = aud   / (np.abs(aud).max()   + 1e-9)
    recon_wn = recon / (np.abs(recon).max() + 1e-9)
    ax30.plot(t_a[::step],   aud_wn[::step],   color=BLUE, lw=0.6, alpha=0.7,
              label='Reference audio')
    ax30.plot(t_rc[::step], recon_wn[::step], color=PURP, lw=0.8, alpha=0.85,
              label='AM reconstruction')
    _ax(ax30, 'Waveform overlay', 'Time (s)', 'Norm. amplitude', xlim=xlim)
    ax30.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    ax31 = fig.add_subplot(gs[3, 2])
    n_cmp = min(len(aud), len(recon))
    an = aud[:n_cmp]   / (np.abs(aud[:n_cmp]).max()   + 1e-9)
    rn = recon[:n_cmp] / (np.abs(recon[:n_cmp]).max() + 1e-9)
    xc = np.correlate(an, rn, mode='full') / n_cmp
    lags_ms = (np.arange(len(xc)) - n_cmp + 1) / AUD_SR * 1000
    zoom = np.abs(lags_ms) <= 500
    pk = xc[zoom].max()
    pk_lag = lags_ms[zoom][np.argmax(xc[zoom])]
    ax31.set_facecolor(PANEL)
    ax31.plot(lags_ms[zoom], xc[zoom], color=PURP, lw=0.8)
    ax31.axvline(pk_lag, color=YLW, lw=1.5, ls='--',
                 label=f'peak={pk:.4f} @ {pk_lag:.1f} ms')
    ax31.axhline(0, color=GRID, lw=0.6)
    for sp in ax31.spines.values(): sp.set_edgecolor(GRID)
    ax31.tick_params(colors=MUTED, labelsize=8)
    ax31.set_title('Cross-correlation', color=TEXT, fontsize=9, loc='left')
    ax31.set_xlabel('Lag (ms)', color=MUTED, fontsize=8)
    ax31.set_ylabel('Xcorr', color=MUTED, fontsize=8)
    ax31.set_xlim(-500, 500)
    ax31.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax31.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.patch.set_facecolor(DARK)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=DARK)
    print(f'[viz]   saved → {out_png}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture',       default=CAPTURE_PATH,
                    help='Speech capture to reconstruct (default: capture.bin)')
    ap.add_argument('--audio',         default=None,
                    help='Reference audio for comparison')
    ap.add_argument('--sweep-capture', default=None,
                    help='Sweep capture for H_AM estimation (default: sweep_capture.bin if present)')
    ap.add_argument('--sweep-audio',   default=None,
                    help='Sweep audio file (default: sweep_cal.wav if present)')
    ap.add_argument('--mains',         type=float, default=50.0,
                    help='Mains frequency guess Hz (default 50; tries 60 too)')
    ap.add_argument('--n-harmonics',   type=int,   default=8,
                    help='Number of mains harmonics to use (default 8)')
    ap.add_argument('--nperseg',       type=int,   default=4096,
                    help='STFT window length (default 4096 → 5.4 Hz @ 22050 Hz)')
    ap.add_argument('--fmax',          type=float, default=4000,
                    help='Upper audio frequency for display (Hz, default 4000)')
    ap.add_argument('--reg',           type=float, default=0.02,
                    help='H_AM regularisation (default 0.02)')
    ap.add_argument('--gl-iters',      type=int,   default=60,
                    help='Griffin-Lim iterations (default 60)')
    ap.add_argument('--upper-only',    action='store_true',
                    help='Use only n=1 upper sideband (pure freq shift: '
                         'audio(f) ← capture(f_mains+f)). Cleaner spectrogram, '
                         'lower envelope SNR than the multi-harmonic accumulation.')
    ap.add_argument('--no-show',       action='store_true')
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use('Agg')

    # ── load speech capture ───────────────────────────────────────────────────
    if not os.path.exists(args.capture):
        sys.exit(f'[error] capture not found: {args.capture}')
    cap = load_capture(args.capture)

    audio_path = args.audio or find_audio(SCRIPT_DIR)
    has_ref = audio_path is not None
    if has_ref:
        print(f'[audio] {os.path.basename(audio_path)}')
        aud = decode_audio(audio_path)
    else:
        print('[audio] no reference audio')
        aud = np.zeros(int(len(cap) / CAP_SR * AUD_SR), dtype=np.float32)

    # ── downsample + lag ──────────────────────────────────────────────────────
    print('[ds]    downsampling capture 200 kHz → 22050 Hz …')
    cap_ds = downsample(cap, CAP_SR, AUD_SR)
    lag_path = os.path.splitext(args.capture)[0] + '.lag'
    aud, cap_ds, lag_ms = detect_and_apply_lag(aud, cap_ds,
                                                lag_path if has_ref else None)
    print(f'[align] {len(cap_ds):,} samples  ({len(cap_ds)/AUD_SR:.4f}s)')

    # ── detect mains frequency ────────────────────────────────────────────────
    f_mains = detect_mains(cap_ds, AUD_SR, args.mains)
    # Also check if 60 Hz might be stronger
    if abs(args.mains - 50) < 1:
        f60 = detect_mains(cap_ds, AUD_SR, 60.0, search_hz=5.0)
        f50_pwr = float(dsp.welch(cap_ds, fs=AUD_SR,
                                   nperseg=min(32768, len(cap_ds)))[1][
                            int(f_mains / (AUD_SR / min(32768, len(cap_ds))))])
        f60_pwr = float(dsp.welch(cap_ds, fs=AUD_SR,
                                   nperseg=min(32768, len(cap_ds)))[1][
                            int(f60 / (AUD_SR / min(32768, len(cap_ds))))])
        if f60_pwr > f50_pwr * 2:
            print(f'[mains] 60 Hz harmonic is stronger — using f_mains = {f60:.3f} Hz')
            f_mains = f60

    # ── optional H_AM calibration from sweep ──────────────────────────────────
    sweep_cap_path = args.sweep_capture or (SWEEP_CAP if os.path.exists(SWEEP_CAP) else None)
    sweep_aud_path = args.sweep_audio   or (SWEEP_WAV  if os.path.exists(SWEEP_WAV) else None)

    f_tones = H_am_mag = None
    if sweep_cap_path and sweep_aud_path and os.path.exists(sweep_cap_path):
        print(f'[H_AM]  estimating from {os.path.basename(sweep_cap_path)} '
              f'+ {os.path.basename(sweep_aud_path)} …')
        sc = load_capture(sweep_cap_path)
        sa = decode_audio(sweep_aud_path)
        sc_ds = downsample(sc, CAP_SR, AUD_SR)
        # Align sweep signals (lag should be 0 since sweep_capture was already aligned)
        n = min(len(sa), len(sc_ds))
        sa, sc_ds = sa[:n], sc_ds[:n]

        nperseg_h = min(args.nperseg, len(sa) // 8)
        f_tones, H_am_mag = estimate_h_am(sc_ds, sa, f_mains, AUD_SR, nperseg_h)
        if f_tones is not None:
            print(f'[H_AM]  {len(f_tones)} tone measurements  '
                  f'|H_AM| range: [{20*np.log10(H_am_mag.min()+1e-30):.1f}, '
                  f'{20*np.log10(H_am_mag.max()+1e-30):.1f}] dB')
        else:
            print('[H_AM]  estimation failed — proceeding without calibration')
    else:
        print('[H_AM]  no sweep capture available — using flat H_AM = 1')

    # ── AM sideband reconstruction ────────────────────────────────────────────
    nperseg = min(args.nperseg, len(cap_ds) // 8)
    print(f'[STFT]  nperseg={nperseg}  '
          f'freq_res={AUD_SR/nperseg:.1f} Hz  '
          f'harmonics={args.n_harmonics}')

    f_stft, t_stft, Y_abs, X_est_mag = am_reconstruct_stft(
        cap_ds, f_mains, args.n_harmonics, AUD_SR, nperseg,
        upper_only=args.upper_only,
    )

    # Apply H_AM calibration if available
    X_est_cal = apply_h_am(X_est_mag, f_stft, f_tones, H_am_mag, args.reg)

    # ── Griffin-Lim ───────────────────────────────────────────────────────────
    print(f'[GL]    Griffin-Lim  {args.gl_iters} iters …')
    np.random.seed(42)
    noverlap = nperseg * 3 // 4
    recon = griffin_lim(X_est_cal, AUD_SR, nperseg, noverlap, args.gl_iters)
    # Trim/pad to audio duration
    n_out = len(aud)
    if len(recon) > n_out:
        recon = recon[:n_out]
    else:
        recon = np.r_[recon, np.zeros(n_out - len(recon), dtype=np.float32)]
    print(f'[GL]    output: {len(recon)/AUD_SR:.3f}s  peak={np.abs(recon).max():.4f}')

    # ── metrics ───────────────────────────────────────────────────────────────
    fr = int(FRAME_S * AUD_SR)
    nf = min(len(aud), len(recon)) // fr
    aud_rms  = np.array([np.sqrt(np.mean(aud[i*fr:(i+1)*fr]**2))   for i in range(nf)])
    rec_rms  = np.array([np.sqrt(np.mean(recon[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    r_env = float(pearsonr(aud_rms, rec_rms)[0]) if has_ref else 0.0
    print(f'[stat]  envelope r = {r_env:.4f}')

    # Spectrogram correlation (direct comparison)
    nperseg_cmp = 2048
    f_c, t_c, Yc_aud  = dsp.stft(aud,   fs=AUD_SR, nperseg=nperseg_cmp,
                                   noverlap=nperseg_cmp*3//4, window='hann')
    f_c, t_c, Yc_rec  = dsp.stft(recon, fs=AUD_SR, nperseg=nperseg_cmp,
                                   noverlap=nperseg_cmp*3//4, window='hann')
    fm = f_c <= args.fmax
    spec_r = float(np.corrcoef(np.abs(Yc_aud[fm]).ravel(),
                                np.abs(Yc_rec[fm]).ravel())[0, 1]) if has_ref else 0.0
    print(f'[stat]  spectrogram r = {spec_r:.4f}')

    # ── save ─────────────────────────────────────────────────────────────────
    mode_tag = 'shift' if args.upper_only else f'h{args.n_harmonics}'
    out_wav = os.path.join(SCRIPT_DIR, f'am_reconstruct_{mode_tag}.wav')
    save_wav(out_wav, recon, AUD_SR)

    # ── plot ──────────────────────────────────────────────────────────────────
    out_png = os.path.join(SCRIPT_DIR, f'am_reconstruct_{mode_tag}.png')
    plot_results(
        aud, cap_ds, recon,
        f_stft, t_stft, Y_abs, X_est_cal,
        f_mains, args.n_harmonics, lag_ms, r_env, args.fmax,
        f_tones, H_am_mag,
        os.path.basename(audio_path) if has_ref else 'no reference',
        out_png,
    )

    mode_str = 'upper-sideband shift (n=1 only)' if args.upper_only \
               else f'multi-harmonic accumulation ({args.n_harmonics} harmonics)'
    print()
    print('══ Results ════════════════════════════════════════')
    print(f'  Mode            : {mode_str}')
    print(f'  Mains frequency : {f_mains:.3f} Hz')
    print(f'  Lag             : {lag_ms:+.0f} ms')
    print(f'  Envelope r      : {r_env:.4f}')
    print(f'  Spectrogram r   : {spec_r:.4f}')
    print(f'  Output WAV      : {os.path.basename(out_wav)}')
    print()
    print('  Listen:')
    print(f'    ffplay -nodisp {os.path.basename(out_wav)}')
    if has_ref:
        print('    ffplay -nodisp original_reference.wav')
    print('═══════════════════════════════════════════════════')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
