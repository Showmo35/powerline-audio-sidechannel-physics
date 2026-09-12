#!/usr/bin/env python3
"""
visualize.py — Powerline leakage aligned viewer.

Loads capture.bin (float32 @ 200 kSps) and the first audio file in the same
directory, then plots four time-locked panels:

  1. Audio waveform
  2. Powerline waveform
  3. Audio spectrogram  }  same time & frequency axes
  4. Powerline spectrogram  —  leakage appears as matching features

Usage:
    python3 visualize.py
    python3 visualize.py --capture capture.bin --audio audio_10s.mp3
    python3 visualize.py --fmax 5000          # zoom spectrograms to 0-5 kHz
    python3 visualize.py --no-show            # save PNG only, don't open window
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as dsp

# ── defaults ──────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH  = os.path.join(SCRIPT_DIR, 'capture.bin')
CAP_SAMP_RATE = 200_000          # Hz — must match simple_rx.py
AUD_SAMP_RATE = 22_050           # Hz — re-decode audio at this rate
AUDIO_EXTS    = ('*.wav', '*.mp3', '*.flac', '*.ogg', '*.aac', '*.m4a')
OUTPUT_PNG    = os.path.join(SCRIPT_DIR, 'leakage_aligned.png')

# ── I/O helpers ───────────────────────────────────────────────────────────────

def find_audio(directory):
    for pat in AUDIO_EXTS:
        hits = sorted(glob.glob(os.path.join(directory, pat)))
        if hits:
            return hits[0]
    return None


def load_capture(path):
    """Return (times, samples) for capture.bin (float32, CAP_SAMP_RATE)."""
    data = np.fromfile(path, dtype=np.float32)
    t    = np.arange(len(data), dtype=np.float64) / CAP_SAMP_RATE
    return t, data


def decode_audio(path, rate=AUD_SAMP_RATE):
    """Decode any audio file to mono float32 PCM via ffmpeg."""
    cmd = [
        'ffmpeg', '-v', 'quiet',
        '-i', path,
        '-f', 'f32le', '-ac', '1', '-ar', str(rate),
        'pipe:1',
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed:\n{proc.stderr.decode()}")
    data = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    t    = np.arange(len(data), dtype=np.float64) / rate
    return t, data


def decimate_for_display(t, x, max_pts=50_000):
    """Thin a signal to at most max_pts points for fast waveform rendering."""
    step = max(1, len(x) // max_pts)
    return t[::step], x[::step]


# ── spectrogram helpers ───────────────────────────────────────────────────────

def make_spectrogram(x, fs, nperseg, noverlap, fmax):
    f, t, Sxx = dsp.spectrogram(x, fs=fs, window='blackman',
                                 nperseg=nperseg, noverlap=noverlap,
                                 scaling='density')
    mask = f <= fmax
    return t, f[mask], 10 * np.log10(Sxx[mask] + 1e-30)   # dB/Hz


# ── plot ─────────────────────────────────────────────────────────────────────

def build_figure(t_aud, aud, t_cap, cap, audio_name, duration, fmax):
    fig = plt.figure(figsize=(15, 11), facecolor='#0d0d14')
    fig.suptitle(
        f'Powerline Leakage — aligned with  "{audio_name}"',
        color='#cdd6f4', fontsize=13, fontweight='bold', y=0.98,
    )

    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        hspace=0.52,
        height_ratios=[1, 1, 2, 2],
        left=0.08, right=0.97, top=0.93, bottom=0.07,
    )

    DARK   = '#0d0d14'
    PANEL  = '#1e1e2e'
    GRID   = '#313244'
    BLUE   = '#89b4fa'
    RED    = '#f38ba8'
    TEXT   = '#cdd6f4'
    LABEL  = '#a6adc8'

    def _style(ax, title, ylabel, xlim):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.tick_params(colors=LABEL, labelsize=8)
        ax.set_title(title, color=TEXT, fontsize=9, loc='left', pad=4)
        ax.set_ylabel(ylabel, color=LABEL, fontsize=8)
        ax.set_xlim(*xlim)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
        ax.xaxis.label.set_color(LABEL)

    xlim = (0.0, duration)

    # ── 1: audio waveform ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    td, xd = decimate_for_display(t_aud, aud)
    ax1.plot(td, xd, color=BLUE, lw=0.6, alpha=0.9)
    ax1.axhline(0, color=GRID, lw=0.5)
    _style(ax1, 'Audio waveform', 'Amplitude', xlim)
    ax1.tick_params(labelbottom=False)

    # ── 2: powerline waveform ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    td2, xd2 = decimate_for_display(t_cap, cap)
    ax2.plot(td2, xd2, color=RED, lw=0.6, alpha=0.9)
    ax2.axhline(0, color=GRID, lw=0.5)
    _style(ax2, 'Powerline capture  (200 kSps, real component)', 'Amplitude', xlim)
    ax2.tick_params(labelbottom=False)

    # ── 3: audio spectrogram ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    # nperseg chosen so frequency resolution ≈ 10 Hz at AUD_SAMP_RATE
    nperseg_a = min(4096, len(aud))
    noverlap_a = nperseg_a * 3 // 4
    t_s, f_s, S_a = make_spectrogram(aud, AUD_SAMP_RATE, nperseg_a, noverlap_a, fmax)
    vmin_a = np.percentile(S_a, 5)
    vmax_a = np.percentile(S_a, 99)
    im3 = ax3.pcolormesh(t_s, f_s / 1000, S_a,
                         shading='gouraud', cmap='magma',
                         vmin=vmin_a, vmax=vmax_a, rasterized=True)
    _style(ax3, 'Audio spectrogram  (dB/Hz)', 'Freq (kHz)', xlim)
    ax3.tick_params(labelbottom=False)
    cb3 = fig.colorbar(im3, ax=ax3, pad=0.01, fraction=0.015)
    cb3.ax.tick_params(colors=LABEL, labelsize=7)
    cb3.set_label('dB/Hz', color=LABEL, fontsize=7)

    # ── 4: powerline spectrogram ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    # Same frequency resolution as ax3 but at 200 kSps
    nperseg_c = 32_768          # → Δf ≈ 6.1 Hz  (matches GNURadio FFT)
    noverlap_c = nperseg_c * 3 // 4
    t_c, f_c, S_c = make_spectrogram(cap, CAP_SAMP_RATE, nperseg_c, noverlap_c, fmax)
    vmin_c = np.percentile(S_c, 5)
    vmax_c = np.percentile(S_c, 99)
    im4 = ax4.pcolormesh(t_c, f_c / 1000, S_c,
                         shading='gouraud', cmap='inferno',
                         vmin=vmin_c, vmax=vmax_c, rasterized=True)
    _style(ax4, 'Powerline spectrogram  (dB/Hz) — leakage mirrors audio', 'Freq (kHz)', xlim)
    ax4.set_xlabel('Time (s)  — t=0 is audio start', color=LABEL, fontsize=9)
    cb4 = fig.colorbar(im4, ax=ax4, pad=0.01, fraction=0.015)
    cb4.ax.tick_params(colors=LABEL, labelsize=7)
    cb4.set_label('dB/Hz', color=LABEL, fontsize=7)

    # Shared x-axis: hide intermediate x labels already done above via sharex
    fig.patch.set_facecolor(DARK)

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture', default=CAPTURE_PATH,
                    help=f'Path to capture.bin  (default: {CAPTURE_PATH})')
    ap.add_argument('--audio', default=None,
                    help='Path to audio file  (default: auto-detect in script dir)')
    ap.add_argument('--fmax', type=float, default=5000,
                    help='Upper frequency limit for spectrograms in Hz  (default: 5000)')
    ap.add_argument('--no-show', action='store_true',
                    help='Save PNG only, do not open interactive window')
    args = ap.parse_args()

    # ── load capture ──────────────────────────────────────────────────────────
    if not os.path.exists(args.capture):
        print(f'[error] capture file not found: {args.capture}')
        sys.exit(1)
    t_cap, cap = load_capture(args.capture)
    print(f'[capture] {len(cap):,} samples  ({t_cap[-1]:.4f}s @ {CAP_SAMP_RATE} sps)')

    # ── load audio ────────────────────────────────────────────────────────────
    audio_path = args.audio or find_audio(SCRIPT_DIR)
    if audio_path is None:
        print('[error] No audio file found — pass --audio <path>')
        sys.exit(1)
    t_aud, aud = decode_audio(audio_path, AUD_SAMP_RATE)
    print(f'[audio]   {len(aud):,} samples  ({t_aud[-1]:.4f}s @ {AUD_SAMP_RATE} sps)')

    # Both signals are already aligned: sample 0 == audio t=0
    duration = min(t_cap[-1], t_aud[-1])
    t_cap = t_cap[t_cap <= duration]
    cap   = cap[:len(t_cap)]
    t_aud = t_aud[t_aud <= duration]
    aud   = aud[:len(t_aud)]

    fmax = min(args.fmax, CAP_SAMP_RATE / 2, AUD_SAMP_RATE / 2)
    print(f'[viz]     duration={duration:.4f}s  fmax={fmax:.0f} Hz')

    # ── plot ──────────────────────────────────────────────────────────────────
    if args.no_show:
        matplotlib.use('Agg')

    fig = build_figure(t_aud, aud, t_cap, cap,
                       os.path.basename(audio_path), duration, fmax)

    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'[viz]     saved → {OUTPUT_PNG}')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
