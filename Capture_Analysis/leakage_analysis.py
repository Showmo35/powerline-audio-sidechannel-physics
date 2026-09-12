#!/usr/bin/env python3
"""
leakage_analysis.py — Quantify and visualise powerline audio leakage.

Uses the audio file as ground-truth to label every 50 ms frame as SPEECH or
SILENCE, then compares the powerline signal across those two conditions:

  Panel 1  Short-time RMS of audio and powerline on the same time axis
           (Pearson r printed) — shows they co-vary.
  Panel 2  PSD: speech period average vs silence period average.
  Panel 3  Leakage spectrum = speech_PSD / silence_PSD in dB.
           Positive dB at a frequency means the powerline carries more power
           at that frequency when audio is playing → direct leakage evidence.
  Panel 4  Powerline spectrogram with speech-onset / offset markers.
  Panel 5  Bar chart: mean leakage (dB) per frequency band.

Usage:
    python3 leakage_analysis.py
    python3 leakage_analysis.py --fmax 4000 --frame-ms 50 --no-show
    python3 leakage_analysis.py --capture capture.bin --audio hello5.wav
"""

import argparse
import glob
import os
import random
import subprocess
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as dsp
from scipy.stats import pearsonr

# ── defaults ──────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH  = os.path.join(SCRIPT_DIR, 'capture.bin')
CAP_SR        = 200_000
AUD_SR        = 22_050
AUDIO_EXTS    = ('*.wav', '*.mp3', '*.flac', '*.ogg', '*.aac', '*.m4a')
OUTPUT_PNG    = os.path.join(SCRIPT_DIR, 'leakage_analysis.png')

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

# ── I/O ───────────────────────────────────────────────────────────────────────

def find_audio(directory):
    for pat in AUDIO_EXTS:
        hits = sorted(glob.glob(os.path.join(directory, pat)))
        if hits:
            return hits[0]
    return None


def load_capture(path, start_s=0.0, dur_s=None):
    """Memmap-slice the capture so a window out of a huge .bin doesn't need a
    full-file read."""
    mm = np.memmap(path, dtype=np.float32, mode='r')
    i0 = int(round(start_s * CAP_SR))
    i1 = i0 + int(round(dur_s * CAP_SR)) if dur_s else len(mm)
    data = np.asarray(mm[i0:i1]).copy()
    del mm
    t = np.arange(len(data), dtype=np.float64) / CAP_SR
    return t, data


def capture_duration_s(path):
    return os.path.getsize(path) / 4 / CAP_SR


def decode_audio(path, rate=AUD_SR, start_s=0.0, dur_s=None):
    cmd = ['ffmpeg', '-v', 'quiet']
    if start_s:
        cmd += ['-ss', str(start_s)]
    cmd += ['-i', path]
    if dur_s:
        cmd += ['-t', str(dur_s)]
    cmd += ['-f', 'f32le', '-ac', '1', '-ar', str(rate), 'pipe:1']
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())
    data = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    t    = np.arange(len(data), dtype=np.float64) / rate
    return t, data


def audio_duration_s(path):
    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
           '-of', 'csv=p=0', path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return float(proc.stdout.strip())

# ── analysis ──────────────────────────────────────────────────────────────────

def frame_rms(x, frame_samples):
    """Non-overlapping short-time RMS, one value per frame."""
    n = len(x) // frame_samples
    blocks = x[:n * frame_samples].reshape(n, frame_samples)
    return np.sqrt(np.mean(blocks ** 2, axis=1))


def detect_speech(aud, aud_sr, frame_s, threshold_frac=0.05):
    """Return bool array (one entry per frame): True = speech."""
    frames = frame_rms(aud, int(frame_s * aud_sr))
    thr = threshold_frac * frames.max()
    return frames > thr, frames


def welch_psd(x, sr, nperseg=32768):
    f, Pxx = dsp.welch(x, fs=sr, window='blackman',
                        nperseg=min(nperseg, len(x)), scaling='density')
    return f, Pxx


def band_leakage(f, leakage_db, bands_hz):
    """Mean leakage dB per frequency band."""
    means = []
    labels = []
    for lo, hi in bands_hz:
        mask = (f >= lo) & (f < hi)
        means.append(leakage_db[mask].mean() if mask.any() else 0.0)
        labels.append(f'{lo//1000 if lo>=1000 else lo}–'
                      f'{"{}k".format(hi//1000) if hi>=1000 else hi} Hz')
    return labels, np.array(means)

# ── plot helpers ──────────────────────────────────────────────────────────────

def _ax(ax, title, xlabel, ylabel, xlim=None, ylim=None):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, loc='left', pad=4)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture', default=CAPTURE_PATH)
    ap.add_argument('--audio',   default=None)
    ap.add_argument('--fmax',    type=float, default=4000,
                    help='Upper frequency for spectral plots (Hz, default 4000)')
    ap.add_argument('--frame-ms', type=float, default=50,
                    help='Analysis frame length in ms (default 50)')
    ap.add_argument('--window',  type=float, default=None,
                    help='Analyze only this many seconds instead of the whole '
                         'file (default: full file). Cheap: memmap-slices the '
                         '.bin and ffmpeg -ss/-t trims the audio.')
    ap.add_argument('--start',   type=float, default=None,
                    help='Start offset (s) for --window. Default: a random '
                         'offset within the file.')
    ap.add_argument('--seed',    type=int, default=None,
                    help='Seed for the random --start pick (for reproducibility)')
    ap.add_argument('--out',     default=OUTPUT_PNG, help='Output PNG path')
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use('Agg')

    frame_s = args.frame_ms / 1000.0
    fmax    = args.fmax

    # ── load ─────────────────────────────────────────────────────────────────
    if not os.path.exists(args.capture):
        sys.exit(f'[error] {args.capture} not found')

    audio_path = args.audio or find_audio(SCRIPT_DIR)
    if not audio_path:
        sys.exit('[error] no audio file found')

    start_s, win_s = args.start, args.window
    if win_s:
        cap_total = capture_duration_s(args.capture)
        aud_total = audio_duration_s(audio_path)
        max_start = max(0.0, min(cap_total, aud_total) - win_s)
        if start_s is None:
            rng = random.Random(args.seed)
            start_s = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
        start_s = min(max(0.0, start_s), max_start)
        print(f'[window] {win_s:.0f}s starting at {start_s:.1f}s '
              f'(file spans cap={cap_total:.0f}s aud={aud_total:.0f}s)')
    else:
        start_s = start_s or 0.0

    _, cap = load_capture(args.capture, start_s=start_s, dur_s=win_s)
    print(f'[cap]   {len(cap):,} samples  ({len(cap)/CAP_SR:.4f}s)')

    _, aud = decode_audio(audio_path, AUD_SR, start_s=start_s, dur_s=win_s)
    print(f'[audio] {len(aud):,} samples  ({len(aud)/AUD_SR:.4f}s)')

    duration = min(len(cap) / CAP_SR, len(aud) / AUD_SR)
    cap = cap[:int(duration * CAP_SR)]
    aud = aud[:int(duration * AUD_SR)]

    # ── lag detection (physical: positive = powerline delayed behind audio) ────
    # np.correlate(a,c) convention: xcorr peaks at NEGATIVE numpy-lag when c
    # is delayed behind a.  physical_lag = -numpy_lag_at_peak.
    lag_path = os.path.splitext(args.capture)[0] + '.lag'
    if os.path.exists(lag_path):
        lag_ms = float(open(lag_path).read().strip())
        print(f'[lag]   read from sidecar: {lag_ms:+.0f} ms (positive = powerline delayed)')
    else:
        _a_tmp = frame_rms(aud, int(frame_s * AUD_SR))
        _c_tmp = frame_rms(cap, int(frame_s * CAP_SR))
        _n = min(len(_a_tmp), len(_c_tmp))
        _a = (_a_tmp[:_n] - _a_tmp[:_n].mean()) / (_a_tmp[:_n].std() + 1e-12)
        _c = (_c_tmp[:_n] - _c_tmp[:_n].mean()) / (_c_tmp[:_n].std() + 1e-12)
        _xc = np.correlate(_a, _c, mode='full') / _n
        _lags = np.arange(-(_n - 1), _n)
        _max_f = int(round(0.700 / frame_s))
        _valid = (-_max_f <= _lags) & (_lags <= 0)   # physical delay → negative numpy lag
        _numpy_lag = int(_lags[_valid][np.argmax(_xc[_valid])])
        lag_ms = -_numpy_lag * frame_s * 1000         # physical lag (positive = delayed)
        print(f'[lag]   auto-detected: {lag_ms:+.0f} ms (positive = powerline delayed)')
    lag_delay = int(round(lag_ms / (frame_s * 1000)))  # physical frames (positive = delayed)

    # ── frame-level analysis ──────────────────────────────────────────────────
    c_frame = int(frame_s * CAP_SR)
    a_frame = int(frame_s * AUD_SR)

    speech_mask, aud_rms = detect_speech(aud, AUD_SR, frame_s)
    cap_rms = frame_rms(cap, c_frame)
    n_frames = min(len(speech_mask), len(cap_rms))
    speech_mask = speech_mask[:n_frames]
    aud_rms     = aud_rms[:n_frames]
    cap_rms     = cap_rms[:n_frames]
    t_frames    = np.arange(n_frames) * frame_s + frame_s / 2

    # Align: powerline delayed by lag_delay frames → shift cap LEFT by lag_delay
    # so cap_aligned[t] = cap[t+lag_delay] = audio[t]
    # Speech mask: powerline frame t labels as speech[t-lag_delay] (prepend zeros)
    if lag_delay > 0:
        cap_rms_aligned = np.r_[cap_rms[lag_delay:], np.full(lag_delay, cap_rms[-1])]
        speech_mask_psd = np.r_[np.zeros(lag_delay, dtype=bool), speech_mask[:-lag_delay]]
    elif lag_delay < 0:
        _d = -lag_delay
        cap_rms_aligned = np.r_[np.full(_d, cap_rms[0]), cap_rms[:-_d]]
        speech_mask_psd = np.r_[speech_mask[_d:], np.zeros(_d, dtype=bool)]
    else:
        cap_rms_aligned = cap_rms.copy()
        speech_mask_psd = speech_mask.copy()
    cap_rms_aligned = cap_rms_aligned[:n_frames]
    speech_mask_psd = speech_mask_psd[:n_frames]

    r,         pval         = pearsonr(aud_rms, cap_rms)
    r_aligned, pval_aligned = pearsonr(aud_rms, cap_rms_aligned)
    print(f'[stat]  r(raw)={r:.4f}  r(lag-corrected)={r_aligned:.4f}  lag={lag_ms:+.0f} ms')

    # ── collect speech / silence cap samples ──────────────────────────────────
    speech_chunks  = []
    silence_chunks = []
    for i, is_speech in enumerate(speech_mask_psd):
        chunk = cap[i * c_frame:(i + 1) * c_frame]
        if is_speech:
            speech_chunks.append(chunk)
        else:
            silence_chunks.append(chunk)

    speech_cat  = np.concatenate(speech_chunks)  if speech_chunks  else np.array([0.0])
    silence_cat = np.concatenate(silence_chunks) if silence_chunks else np.array([0.0])
    print(f'[stat]  speech frames={len(speech_chunks)}  silence frames={len(silence_chunks)}')

    # ── PSD ───────────────────────────────────────────────────────────────────
    nperseg_cap = min(32768, len(silence_cat), len(speech_cat))
    f_s, Pxx_speech  = welch_psd(speech_cat,  CAP_SR, nperseg_cap)
    f_n, Pxx_silence = welch_psd(silence_cat, CAP_SR, nperseg_cap)
    fmask = f_s <= fmax

    # Leakage = ratio of speech PSD to silence PSD
    leakage_db  = 10 * np.log10(Pxx_speech[fmask] / (Pxx_silence[fmask] + 1e-30))
    f_leak      = f_s[fmask]
    total_leak  = leakage_db.mean()
    peak_leak   = leakage_db.max()
    peak_freq   = f_leak[np.argmax(leakage_db)]
    print(f'[stat]  mean leakage={total_leak:.2f} dB  peak={peak_leak:.2f} dB @ {peak_freq:.1f} Hz')

    # ── speech boundary times ─────────────────────────────────────────────────
    edges = np.diff(speech_mask.astype(int))
    onsets  = t_frames[:-1][edges ==  1]
    offsets = t_frames[:-1][edges == -1]
    if speech_mask[0]:  onsets  = np.r_[0.0, onsets]
    if speech_mask[-1]: offsets = np.r_[offsets, duration]

    # ── frequency bands (auto-scaled to fmax) ────────────────────────────────
    if fmax <= 5000:
        bands_hz = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, int(fmax))]
    elif fmax <= 20000:
        bands_hz = [(0, 1000), (1000, 4000), (4000, 8000), (8000, 16000), (16000, int(fmax))]
    else:
        bands_hz = [(0, 1000), (1000, 10000), (10000, 30000), (30000, 60000), (60000, int(fmax))]
    band_labels, band_db = band_leakage(f_leak, leakage_db, bands_hz)

    # ── spectrogram ───────────────────────────────────────────────────────────
    nperseg_sp = min(32768, len(cap))
    noverlap_sp = nperseg_sp * 3 // 4
    f_sp, t_sp, Sxx = dsp.spectrogram(cap, fs=CAP_SR, window='blackman',
                                        nperseg=nperseg_sp, noverlap=noverlap_sp,
                                        scaling='density')
    fsp_mask = f_sp <= fmax
    Sxx_db   = 10 * np.log10(Sxx[fsp_mask] + 1e-30)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 13), facecolor=DARK)
    fig.suptitle(
        f'Powerline Leakage Analysis  —  "{os.path.basename(audio_path)}"\n'
        f'mean leakage = {total_leak:+.2f} dB  |  peak = {peak_leak:+.2f} dB @ {peak_freq:.0f} Hz  |  '
        f'r={r:.3f} (raw)  r={r_aligned:.3f} (lag={lag_ms:+.0f} ms)',
        color=TEXT, fontsize=11, fontweight='bold', y=0.99,
    )

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.08, right=0.97, top=0.93, bottom=0.06)

    # ── Panel 1 (top, full width): RMS time series ────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _ax(ax1, 'Short-time RMS — audio (blue) vs powerline (red)',
        'Time (s)', 'RMS', xlim=(0, duration))

    # Shade speech regions
    for on, off in zip(onsets, offsets):
        ax1.axvspan(on, off, color=BLUE, alpha=0.12, lw=0)

    ax1_r = ax1.twinx()
    ax1_r.set_facecolor('none')
    ax1_r.tick_params(colors=MUTED, labelsize=8)
    ax1_r.set_ylabel('Powerline RMS', color=RED, fontsize=8)

    ax1.plot(t_frames, aud_rms, color=BLUE, lw=1.5, label='Audio RMS')
    ax1_r.plot(t_frames, cap_rms, color=RED, lw=1.0, alpha=0.4, ls='--', label='Powerline RMS (raw)')
    ax1_r.plot(t_frames, cap_rms_aligned, color=RED, lw=1.5, alpha=0.9,
               label=f'Powerline RMS (lag={lag_ms:+.0f} ms)')
    ax1.set_ylabel('Audio RMS', color=BLUE, fontsize=8)

    ax1.text(0.98, 0.92,
             f'r={r:.3f} (raw)\nr={r_aligned:.3f} (Δ={lag_ms:+.0f} ms)',
             transform=ax1.transAxes, color=YLW, fontsize=9,
             ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', fc=PANEL, ec=GRID))

    speech_patch = matplotlib.patches.Patch(color=BLUE, alpha=0.3, label='Speech')
    ax1.legend(handles=[
        matplotlib.lines.Line2D([],[],color=BLUE,lw=2,label='Audio RMS'),
        matplotlib.lines.Line2D([],[],color=RED, lw=1,ls='--',alpha=0.5,label='Powerline RMS (raw)'),
        matplotlib.lines.Line2D([],[],color=RED, lw=2,label=f'Powerline RMS (lag={lag_ms:+.0f} ms)'),
        speech_patch,
    ], loc='upper left', fontsize=8,
       facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 2 (mid-left): PSD speech vs silence ────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _ax(ax2, 'Power Spectral Density: speech vs silence',
        'Frequency (Hz)', 'dB/Hz', xlim=(0, fmax))

    Pdb_speech  = 10 * np.log10(Pxx_speech[fmask]  + 1e-30)
    Pdb_silence = 10 * np.log10(Pxx_silence[fmask] + 1e-30)

    ax2.fill_between(f_leak, Pdb_silence, Pdb_speech,
                     where=Pdb_speech > Pdb_silence,
                     color=GRN, alpha=0.25, label='Leakage region')
    ax2.plot(f_leak, Pdb_silence, color=BLUE,  lw=1.2, label='Silence PSD')
    ax2.plot(f_leak, Pdb_speech,  color=ORNG,  lw=1.2, label='Speech PSD')
    ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 3 (mid-right): leakage spectrum ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _ax(ax3, 'Leakage spectrum  (speech PSD − silence PSD)',
        'Frequency (Hz)', 'Leakage (dB)', xlim=(0, fmax))

    ax3.axhline(0, color=GRID, lw=1.0, ls='--', label='0 dB = no leakage')
    ax3.fill_between(f_leak, 0, leakage_db,
                     where=leakage_db > 0, color=RED,  alpha=0.35, label='Positive leakage')
    ax3.fill_between(f_leak, 0, leakage_db,
                     where=leakage_db < 0, color=BLUE, alpha=0.25, label='Negative (noise floor)')
    ax3.plot(f_leak, leakage_db, color=ORNG, lw=1.0, alpha=0.9)
    ax3.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # Peak annotation
    ax3.annotate(f'{peak_leak:+.1f} dB\n@ {peak_freq:.0f} Hz',
                 xy=(peak_freq, peak_leak),
                 xytext=(peak_freq + fmax * 0.08, peak_leak * 0.8),
                 color=YLW, fontsize=8, arrowprops=dict(arrowstyle='->', color=YLW))

    # ── Panel 4 (bottom-left): spectrogram + boundaries ──────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    _ax(ax4, 'Powerline spectrogram  (with speech boundaries)',
        'Time (s)', 'Frequency (Hz)', xlim=(0, duration))

    vmin = np.percentile(Sxx_db, 5)
    vmax = np.percentile(Sxx_db, 99)
    ax4.grid(False)
    ax4.pcolormesh(t_sp, f_sp[fsp_mask], Sxx_db,
                   shading='gouraud', cmap='inferno',
                   vmin=vmin, vmax=vmax, rasterized=True)

    for on  in onsets:  ax4.axvline(on,  color=GRN, lw=1.2, ls='--', alpha=0.8)
    for off in offsets: ax4.axvline(off, color=RED, lw=1.2, ls='--', alpha=0.8)

    ax4.plot([], [], color=GRN, lw=1.5, ls='--', label='Speech onset')
    ax4.plot([], [], color=RED, lw=1.5, ls='--', label='Speech offset')
    ax4.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, loc='upper right')

    # ── Panel 5 (bottom-right): per-band leakage bar chart ───────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    _ax(ax5, 'Mean leakage by frequency band',
        'Frequency band', 'Mean leakage (dB)')

    colors = [RED if v > 0 else BLUE for v in band_db]
    bars = ax5.bar(band_labels, band_db, color=colors, alpha=0.8, edgecolor=GRID, width=0.6)
    ax5.axhline(0, color=GRID, lw=1.0, ls='--')
    ax5.tick_params(axis='x', labelsize=7, rotation=20)

    for bar, val in zip(bars, band_db):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 val + (0.05 if val >= 0 else -0.15),
                 f'{val:+.2f} dB',
                 ha='center', va='bottom' if val >= 0 else 'top',
                 color=TEXT, fontsize=8, fontweight='bold')

    fig.patch.set_facecolor(DARK)
    fig.savefig(args.out, dpi=150, bbox_inches='tight', facecolor=DARK)
    print(f'[viz]   saved → {args.out}')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
