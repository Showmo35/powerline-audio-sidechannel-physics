#!/usr/bin/env python3
"""
reconstruct.py — Reconstruct audio from powerline leakage.

Two methods, compared side-by-side:

  Method 1  Power envelope
            Short-time RMS of the powerline tracks the audio amplitude envelope.
            Applies Wiener gain (speech/silence ratio) per frequency band,
            then upsamples to audio rate for listening.

  Method 2  Spectral subtraction + Griffin-Lim
            Uses the silence period to build a noise baseline per frequency bin.
            Subtracts baseline from the powerline STFT magnitude, floors at zero,
            then iteratively reconstructs phase (Griffin-Lim) and resamples to
            audio rate.

Output files:
  reconstructed_envelope.wav    Method 1 result
  reconstructed_spectral.wav    Method 2 result
  reconstruction.png            4-panel comparison figure

Usage:
    python3 reconstruct.py
    python3 reconstruct.py --fmax 4000 --gl-iters 60 --no-show
"""

import argparse
import glob
import os
import subprocess
import sys
import wave

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as dsp

# ── constants ─────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH = os.path.join(SCRIPT_DIR, 'capture.bin')
AUDIO_EXTS   = ('*.wav', '*.mp3', '*.flac', '*.ogg', '*.aac', '*.m4a')
CAP_SR       = 200_000
AUD_SR       = 22_050
FRAME_S      = 0.05          # 50 ms analysis frame for speech/silence labelling

DARK  = '#0d0d14'
PANEL = '#1e1e2e'
GRID  = '#313244'
TEXT  = '#cdd6f4'
MUTED = '#a6adc8'
BLUE  = '#89b4fa'
RED   = '#f38ba8'
GRN   = '#a6e3a1'
ORNG  = '#fab387'
YLW   = '#f9e2af'

# ── I/O helpers ───────────────────────────────────────────────────────────────

def find_audio(directory):
    for pat in AUDIO_EXTS:
        hits = sorted(glob.glob(os.path.join(directory, pat)))
        if hits:
            return hits[0]
    return None


def load_capture(path):
    data = np.fromfile(path, dtype=np.float32)
    return data


def decode_audio(path, rate=AUD_SR):
    cmd = ['ffmpeg', '-v', 'quiet', '-i', path,
           '-f', 'f32le', '-ac', '1', '-ar', str(rate), 'pipe:1']
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def save_wav(path, data, rate):
    """Save float32 array as 16-bit WAV."""
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak * 0.9           # normalise to -1 dB FS
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    print(f'  saved → {path}  ({os.path.getsize(path)//1024} KB)')


def resample_to_audio(x, src_sr, dst_sr):
    """Polyphase resample from src_sr → dst_sr."""
    from math import gcd
    g = gcd(int(src_sr), int(dst_sr))
    up, down = int(dst_sr) // g, int(src_sr) // g
    return dsp.resample_poly(x, up, down).astype(np.float32)

# ── speech / silence labelling ────────────────────────────────────────────────

def label_frames(aud, cap):
    """Return (speech_mask, aud_rms, cap_rms) at FRAME_S resolution."""
    a_f = int(FRAME_S * AUD_SR)
    c_f = int(FRAME_S * CAP_SR)
    n   = min(len(aud) // a_f, len(cap) // c_f)
    a_rms = np.array([np.sqrt(np.mean(aud[i*a_f:(i+1)*a_f]**2)) for i in range(n)])
    c_rms = np.array([np.sqrt(np.mean(cap[i*c_f:(i+1)*c_f]**2)) for i in range(n)])
    thr   = 0.05 * a_rms.max()
    return a_rms > thr, a_rms, c_rms


# ── Method 1: power envelope ──────────────────────────────────────────────────

def reconstruct_envelope(cap, aud, speech_mask, fmax):
    """
    Per-frequency-band Wiener gain applied to the powerline short-time RMS.

    For each 1/3-octave band:
      - Bandpass-filter powerline around centre frequency
      - Compute short-time RMS (=envelope)
      - Apply Wiener gain  G = max(0, 1 - noise_rms / signal_rms)
      - Synthesise a sinusoid at the band centre modulated by the envelope
    Sum all bands → reconstructed signal at CAP_SR → resample to AUD_SR.
    """
    print('[M1] Power-envelope reconstruction …')
    c_f = int(FRAME_S * CAP_SR)
    n   = len(speech_mask)

    # Band definitions (third-octave from 100 Hz to fmax)
    centres = []
    fc = 100.0
    while fc <= fmax:
        centres.append(fc)
        fc *= 2 ** (1 / 3)

    # Silence RMS baseline per band
    sil_idx = np.where(~speech_mask)[0]

    t_cap    = np.arange(len(cap)) / CAP_SR
    output   = np.zeros(n * c_f, dtype=np.float64)

    for fc in centres:
        bw   = fc * (2 ** (1 / 6) - 2 ** (-1 / 6))
        lo   = max(fc - bw / 2, 1.0)
        hi   = min(fc + bw / 2, CAP_SR / 2 - 1)
        if lo >= hi:
            continue
        sos  = dsp.butter(4, [lo, hi], btype='band', fs=CAP_SR, output='sos')
        filt = dsp.sosfilt(sos, cap)

        # Short-time RMS per frame
        env = np.array([np.sqrt(np.mean(filt[i*c_f:(i+1)*c_f]**2))
                        for i in range(n)])
        noise_rms   = env[sil_idx].mean() if len(sil_idx) else env.mean()
        signal_rms  = env.max()

        # Wiener gain per frame
        gain = np.maximum(0.0, 1.0 - noise_rms / (env + 1e-30))

        # Synthesise carrier at fc modulated by gain envelope
        carrier = np.sin(2 * np.pi * fc * t_cap[:n * c_f])
        gain_up = np.repeat(gain, c_f)       # frame-rate → sample-rate
        output += gain_up * carrier

    out_aud = resample_to_audio(output, CAP_SR, AUD_SR)
    print(f'  duration: {len(out_aud)/AUD_SR:.3f}s  peak: {np.abs(out_aud).max():.4f}')
    return out_aud.astype(np.float32)


# ── Griffin-Lim phase reconstruction ─────────────────────────────────────────

def griffin_lim(magnitude, fs, nperseg, noverlap, n_iter=50):
    """
    Reconstruct a signal from a magnitude spectrogram via Griffin-Lim.
    Returns the time-domain signal at `fs`.
    """
    # Random phase initialisation
    phase = np.exp(1j * np.random.uniform(-np.pi, np.pi, magnitude.shape))
    for _ in range(n_iter):
        S = magnitude * phase
        _, x = dsp.istft(S, fs=fs, nperseg=nperseg, noverlap=noverlap,
                          window='hann', boundary=None)
        _, _, S_new = dsp.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                window='hann', boundary=None)
        # Trim or pad so shapes match (boundary artefacts)
        min_col = min(S.shape[1], S_new.shape[1])
        phase   = np.exp(1j * np.angle(S_new[:, :min_col]))
        magnitude = magnitude[:, :min_col]
    _, out = dsp.istft(magnitude * phase, fs=fs, nperseg=nperseg, noverlap=noverlap,
                        window='hann', boundary=None)
    return out.astype(np.float32)


# ── Method 2: spectral subtraction + Griffin-Lim ─────────────────────────────

def reconstruct_spectral(cap, aud, speech_mask, fmax, gl_iters, beta=1.5):
    """
    1. STFT of powerline at audio-compatible resolution (WIN=4096, HOP=1024).
    2. Estimate noise floor from silence frames (per frequency bin).
    3. Spectral subtraction: mag_clean = max(0, mag - beta * noise_mag).
    4. Zero out bins above fmax.
    5. Griffin-Lim phase reconstruction.
    6. Resample to AUD_SR.

    beta: oversubtraction factor (>1 suppresses more residual noise, default 1.5)
    """
    print(f'[M2] Spectral subtraction + Griffin-Lim ({gl_iters} iters) …')
    WIN    = 4096          # 20.5 ms at 200 kHz  →  48.8 Hz/bin
    HOP    = WIN // 4      # 75 % overlap

    f, t_stft, S = dsp.stft(cap, fs=CAP_SR, window='hann',
                              nperseg=WIN, noverlap=WIN - HOP, boundary=None)

    # Map each STFT frame to an analysis frame index
    c_f = int(FRAME_S * CAP_SR)
    n   = len(speech_mask)
    frame_of_stft = (t_stft / FRAME_S).astype(int)
    frame_of_stft = np.clip(frame_of_stft, 0, n - 1)
    sil_cols = np.where(~speech_mask[frame_of_stft])[0]

    mag = np.abs(S)

    # Noise floor: mean magnitude of silence frames per bin
    if sil_cols.size:
        noise_mag = mag[:, sil_cols].mean(axis=1, keepdims=True)
    else:
        noise_mag = mag.mean(axis=1, keepdims=True)

    # Spectral subtraction with oversubtraction
    mag_clean = np.maximum(0.0, mag - beta * noise_mag)

    # Zero bins above fmax
    fmask = f > fmax
    mag_clean[fmask, :] = 0.0

    print(f'  STFT shape: {S.shape}  active bins (0–{fmax:.0f} Hz): {(~fmask).sum()}')
    ratio = mag_clean.mean() / (noise_mag.mean() + 1e-30)
    print(f'  Clean/noise ratio: {ratio:.4f}  ({10*np.log10(ratio+1e-9):.1f} dB)')

    # Griffin-Lim
    out_cap = griffin_lim(mag_clean, CAP_SR, WIN, WIN - HOP, n_iter=gl_iters)

    # Resample to audio rate
    out_aud = resample_to_audio(out_cap, CAP_SR, AUD_SR)
    print(f'  duration: {len(out_aud)/AUD_SR:.3f}s  peak: {np.abs(out_aud).max():.4f}')
    return out_aud.astype(np.float32)


# ── spectrogram helper ────────────────────────────────────────────────────────

def make_spec(x, fs, fmax, nperseg=2048):
    noverlap = nperseg * 3 // 4
    f, t, Sxx = dsp.spectrogram(x, fs=fs, window='hann',
                                  nperseg=min(nperseg, len(x)),
                                  noverlap=noverlap, scaling='density')
    fmask = f <= fmax
    return t, f[fmask], 10 * np.log10(Sxx[fmask] + 1e-30)


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


def plot_m1(aud, m1, speech_mask, lag_ms, fmax, audio_name, out_png):
    duration  = len(aud) / AUD_SR
    xlim      = (0, duration)

    # Shift m1 left by lag so it aligns with audio (physical lag: positive = delayed)
    lag_samp = int(round(abs(lag_ms) / 1000 * AUD_SR))
    if lag_ms > 0:
        m1_aligned = np.r_[m1[lag_samp:], np.zeros(lag_samp, dtype=np.float32)]
    elif lag_ms < 0:
        m1_aligned = np.r_[np.zeros(lag_samp, dtype=np.float32), m1[:-lag_samp]]
    else:
        m1_aligned = m1.copy()
    n_cmp = min(len(m1_aligned), len(aud))
    m1_aligned = m1_aligned[:n_cmp]
    aud_cmp    = aud[:n_cmp]

    # Frame-level RMS for envelope comparison
    frame_samp_a = int(FRAME_S * AUD_SR)
    n_fr = min(len(aud_cmp), len(m1_aligned)) // frame_samp_a
    aud_rms_fr = np.array([np.sqrt(np.mean(aud_cmp[i*frame_samp_a:(i+1)*frame_samp_a]**2))
                            for i in range(n_fr)])
    m1_rms_fr  = np.array([np.sqrt(np.mean(m1_aligned[i*frame_samp_a:(i+1)*frame_samp_a]**2))
                            for i in range(n_fr)])
    t_fr = np.arange(n_fr) * FRAME_S + FRAME_S / 2

    from scipy.stats import pearsonr
    r_env, p_env = pearsonr(aud_rms_fr, m1_rms_fr)

    # Sample-level cross-correlation of lag-aligned signals
    a_n = aud_cmp  / (np.abs(aud_cmp).max()  + 1e-9)
    m_n = m1_aligned / (np.abs(m1_aligned).max() + 1e-9)
    xcorr  = np.correlate(a_n, m_n, mode='full') / n_cmp
    lags_s = (np.arange(len(xcorr)) - n_cmp + 1) / AUD_SR * 1000
    peak_r   = xcorr.max()
    peak_lag = lags_s[np.argmax(xcorr)]

    # Speech boundaries (from original lag-corrected mask)
    t_frames = np.arange(len(speech_mask)) * FRAME_S
    edges    = np.diff(speech_mask.astype(int))
    onsets   = t_frames[:-1][edges ==  1]
    offsets  = t_frames[:-1][edges == -1]
    if speech_mask[0]:  onsets  = np.r_[0.0, onsets]
    if speech_mask[-1]: offsets = np.r_[offsets, duration]

    def shade(ax):
        for on, off in zip(onsets, offsets):
            ax.axvspan(on, off, color=BLUE, alpha=0.10, lw=0)

    fig = plt.figure(figsize=(15, 13), facecolor=DARK)
    fig.suptitle(
        f'Method 1: Power-Envelope Reconstruction  —  "{audio_name}"\n'
        f'lag={lag_ms:+.0f} ms  |  envelope r = {r_env:.3f}  |  '
        f'sample xcorr = {peak_r:.4f} @ {peak_lag:.1f} ms',
        color=TEXT, fontsize=11, fontweight='bold', y=0.99,
    )
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.3,
                           left=0.08, right=0.97, top=0.93, bottom=0.06)

    t_aud = np.linspace(0, duration, len(aud))
    t_m1a = np.linspace(0, duration, len(m1_aligned))
    step  = max(1, len(aud) // 50_000)

    # ── Row 0: spectrograms (audio vs m1 aligned) ────────────────────────────
    def spec_panel(ax, x, sr, title, cmap):
        t_s, f_s, S = make_spec(x, sr, fmax, nperseg=1024)
        ax.grid(False)
        ax.pcolormesh(t_s, f_s, S, shading='gouraud', cmap=cmap,
                      vmin=np.percentile(S, 5), vmax=np.percentile(S, 99),
                      rasterized=True)
        shade(ax)
        _ax(ax, title, 'Time (s)', 'Freq (Hz)', xlim=xlim)

    ax00 = fig.add_subplot(gs[0, 0])
    spec_panel(ax00, aud, AUD_SR, 'Original audio spectrogram', 'magma')

    ax01 = fig.add_subplot(gs[0, 1])
    spec_panel(ax01, m1_aligned, AUD_SR,
               'M1 reconstruction spectrogram', 'plasma')

    # ── Row 1: waveform overlay ───────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, :])
    aud_n = aud / (np.abs(aud).max() + 1e-9)
    m1a_n = m1_aligned / (np.abs(m1_aligned).max() + 1e-9)
    ax10.plot(t_aud[::step],  aud_n[::step],  color=BLUE, lw=0.6, alpha=0.7, label='Original audio')
    ax10.plot(t_m1a[::step], m1a_n[::step],  color=ORNG, lw=0.8, alpha=0.85,
              label=f'M1 reconstruction (lag-aligned {lag_ms:+.0f} ms)')
    shade(ax10)
    _ax(ax10, f'Waveform overlay  (M1 lag-aligned {lag_ms:+.0f} ms)', 'Time (s)', 'Norm. amplitude', xlim=xlim)
    ax10.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Row 2: envelope (RMS) comparison ─────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, :])
    aud_rms_n = aud_rms_fr / (aud_rms_fr.max() + 1e-9)
    m1_rms_n  = m1_rms_fr  / (m1_rms_fr.max()  + 1e-9)
    ax20.plot(t_fr, aud_rms_n, color=BLUE, lw=2.0, label='Audio envelope (RMS)')
    ax20.plot(t_fr, m1_rms_n,  color=ORNG, lw=2.0, alpha=0.9, label='M1 envelope (RMS)')
    shade(ax20)
    ax20.text(0.98, 0.92, f'Envelope r = {r_env:.3f}  (p={p_env:.1e})',
              transform=ax20.transAxes, color=YLW, fontsize=10,
              ha='right', va='top',
              bbox=dict(boxstyle='round,pad=0.3', fc=PANEL, ec=GRID))
    _ax(ax20, 'Envelope (short-time RMS) comparison  —  do the peaks line up?',
        'Time (s)', 'Norm. RMS', xlim=xlim)
    ax20.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Row 3: cross-correlation ──────────────────────────────────────────────
    ax30 = fig.add_subplot(gs[3, 0])
    zoom = np.abs(lags_s) <= 500
    ax30.plot(lags_s[zoom], xcorr[zoom], color=ORNG, lw=0.8)
    ax30.axvline(peak_lag, color=YLW, lw=1.5, ls='--',
                 label=f'peak={peak_r:.4f} @ {peak_lag:.1f} ms')
    ax30.axhline(0, color=GRID, lw=0.6)
    _ax(ax30, 'Cross-correlation (M1 vs audio)',
        'Lag (ms)', 'Xcorr', xlim=(-500, 500))
    ax30.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Row 3 right: scatter plot envelope vs audio ───────────────────────────
    ax31 = fig.add_subplot(gs[3, 1])
    ax31.set_facecolor(PANEL)
    for sp in ax31.spines.values(): sp.set_edgecolor(GRID)
    ax31.tick_params(colors=MUTED, labelsize=8)
    ax31.scatter(aud_rms_n, m1_rms_n, c=t_fr, cmap='plasma', s=20, alpha=0.8)
    ax31.set_xlabel('Audio RMS (norm.)', color=MUTED, fontsize=8)
    ax31.set_ylabel('M1 RMS (norm.)', color=MUTED, fontsize=8)
    ax31.set_title(f'Envelope scatter  (r = {r_env:.3f})',
                   color=TEXT, fontsize=9, loc='left', pad=4)
    ax31.grid(True, color=GRID, lw=0.5, alpha=0.6)
    # Diagonal line = perfect match
    ax31.plot([0, 1], [0, 1], color=GRID, lw=1, ls='--')

    fig.patch.set_facecolor(DARK)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=DARK)
    print(f'[viz]  saved → {out_png}')
    return peak_r, peak_lag, r_env


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture',  default=CAPTURE_PATH)
    ap.add_argument('--audio',    default=None)
    ap.add_argument('--fmax',     type=float, default=4000,
                    help='Upper frequency for all processing (Hz, default 4000)')
    ap.add_argument('--no-show',  action='store_true')
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use('Agg')

    # ── load ─────────────────────────────────────────────────────────────────
    if not os.path.exists(args.capture):
        sys.exit(f'[error] {args.capture} not found')
    cap = load_capture(args.capture)
    print(f'[cap]   {len(cap):,} samples  ({len(cap)/CAP_SR:.4f}s @ {CAP_SR} sps)')

    audio_path = args.audio or find_audio(SCRIPT_DIR)
    if not audio_path:
        sys.exit('[error] no audio file found')
    aud = decode_audio(audio_path, AUD_SR)
    print(f'[audio] {len(aud):,} samples  ({len(aud)/AUD_SR:.4f}s @ {AUD_SR} sps)')

    duration = min(len(cap) / CAP_SR, len(aud) / AUD_SR)
    cap = cap[:int(duration * CAP_SR)]
    aud = aud[:int(duration * AUD_SR)]

    # ── lag: read sidecar or detect via xcorr ────────────────────────────────
    lag_path = os.path.splitext(args.capture)[0] + '.lag'
    if os.path.exists(lag_path):
        lag_ms = float(open(lag_path).read().strip())
        print(f'[lag]   read from sidecar: {lag_ms:+.0f} ms')
    else:
        _, _aud_rms, _cap_rms = label_frames(aud, cap)
        _n = min(len(_aud_rms), len(_cap_rms))
        _a = (_aud_rms[:_n] - _aud_rms[:_n].mean()) / (_aud_rms[:_n].std() + 1e-12)
        _c = (_cap_rms[:_n] - _cap_rms[:_n].mean()) / (_cap_rms[:_n].std() + 1e-12)
        _xc = np.correlate(_a, _c, mode='full') / _n
        _lags = np.arange(-(_n - 1), _n)
        _mf = int(round(0.700 / FRAME_S))
        _valid = (-_mf <= _lags) & (_lags <= 0)   # physical delay → negative numpy lag
        lag_ms = -float(_lags[_valid][np.argmax(_xc[_valid])]) * FRAME_S * 1000
        print(f'[lag]   auto-detected: {lag_ms:+.0f} ms')
    lag_delay = int(round(lag_ms / (FRAME_S * 1000)))

    # ── label frames (lag-corrected speech mask for powerline) ───────────────
    speech_mask, aud_rms, cap_rms = label_frames(aud, cap)
    n = len(speech_mask)
    if lag_delay > 0:
        speech_mask_lag = np.r_[np.zeros(lag_delay, dtype=bool), speech_mask[:-lag_delay]][:n]
    elif lag_delay < 0:
        _d = -lag_delay
        speech_mask_lag = np.r_[speech_mask[_d:], np.zeros(_d, dtype=bool)][:n]
    else:
        speech_mask_lag = speech_mask.copy()
    print(f'[lbl]   speech={speech_mask_lag.sum()}  silence={(~speech_mask_lag).sum()} frames (lag-corrected)')

    # ── reconstruct (Method 1 only) ───────────────────────────────────────────
    m1 = reconstruct_envelope(cap, aud, speech_mask_lag, args.fmax)

    # ── lag-align m1 for saving ───────────────────────────────────────────────
    lag_samp = int(round(abs(lag_ms) / 1000 * AUD_SR))
    if lag_ms > 0:
        m1_save = np.r_[m1[lag_samp:], np.zeros(lag_samp, dtype=np.float32)]
    elif lag_ms < 0:
        m1_save = np.r_[np.zeros(lag_samp, dtype=np.float32), m1[:-lag_samp]]
    else:
        m1_save = m1.copy()

    # ── save WAV ──────────────────────────────────────────────────────────────
    print('[wav]  saving …')
    save_wav(os.path.join(SCRIPT_DIR, 'reconstructed_envelope.wav'), m1_save, AUD_SR)
    save_wav(os.path.join(SCRIPT_DIR, 'original_reference.wav'),     aud,     AUD_SR)

    # ── plot ──────────────────────────────────────────────────────────────────
    out_png = os.path.join(SCRIPT_DIR, 'reconstruction.png')
    peak_r, peak_lag, r_env = plot_m1(
        aud, m1, speech_mask_lag, lag_ms, args.fmax,
        os.path.basename(audio_path), out_png,
    )

    print()
    print('══ Results ══════════════════════════════════════')
    print(f'  Lag detected       = {lag_ms:+.0f} ms')
    print(f'  Envelope r         = {r_env:.4f}')
    print(f'  Sample xcorr peak  = {peak_r:.4f}  @ {peak_lag:.1f} ms')
    print()
    print('  Listen:')
    print('    ffplay -nodisp reconstructed_envelope.wav')
    print('    ffplay -nodisp original_reference.wav')
    print('═════════════════════════════════════════════════')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
