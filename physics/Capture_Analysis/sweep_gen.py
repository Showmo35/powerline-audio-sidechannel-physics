#!/usr/bin/env python3
"""
sweep_gen.py — Generate a calibration frequency sweep for powerline TF measurement.

Produces a log-chirp from f_start to f_stop at constant amplitude.  A log-chirp
has equal energy per octave, so every frequency band is excited equally — giving
the flattest coherence across the audio range for the H1 transfer function
estimator in tf_estimate.py.

Optionally also generates an interleaved stepped-tone version at a chosen number
of logarithmically-spaced frequencies for a quick visual sanity check.

Output:
  sweep_cal.wav    log-chirp  (default: 50 Hz → 8 kHz, 30 s)

Usage:
    python3 sweep_gen.py
    python3 sweep_gen.py --f-start 50 --f-stop 8000 --duration 30
    python3 sweep_gen.py --stepped --n-tones 30 --tone-dur 1.0
    python3 sweep_gen.py --no-show
"""

import argparse
import os
import wave
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal as dsp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUD_SR = 22_050

DARK  = '#0d0d14'
PANEL = '#1e1e2e'
GRID  = '#313244'
TEXT  = '#cdd6f4'
MUTED = '#a6adc8'
BLUE  = '#89b4fa'
ORNG  = '#fab387'
GRN   = '#a6e3a1'
YLW   = '#f9e2af'


def make_log_chirp(f_start, f_stop, duration, sr, amplitude=0.9):
    """Log-chirp: exponentially sweeping from f_start to f_stop."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    chirp = dsp.chirp(t, f0=f_start, f1=f_stop, t1=duration,
                      method='logarithmic', phi=-90)
    return (amplitude * chirp).astype(np.float32)


def make_stepped_tones(f_start, f_stop, n_tones, tone_dur, silence_dur, sr, amplitude=0.9):
    """Stepped tones at n_tones log-spaced frequencies with silence gaps."""
    freqs = np.logspace(np.log10(f_start), np.log10(f_stop), n_tones)
    t_tone    = np.linspace(0, tone_dur,    int(tone_dur    * sr), endpoint=False)
    t_silence = np.zeros(int(silence_dur * sr), dtype=np.float32)
    chunks = []
    for fc in freqs:
        tone = amplitude * np.sin(2 * np.pi * fc * t_tone).astype(np.float32)
        # Cosine fade-in / fade-out (5 ms) to suppress clicks
        fade = int(0.005 * sr)
        if fade > 0 and 2 * fade < len(tone):
            window = np.ones(len(tone), dtype=np.float32)
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade)).astype(np.float32)
            window[:fade]  = ramp
            window[-fade:] = ramp[::-1]
            tone = tone * window
        chunks.append(tone)
        chunks.append(t_silence.copy())
    return np.concatenate(chunks), freqs


def save_wav(path, data, sr):
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak * 0.9
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    print(f'[saved] {path}  ({os.path.getsize(path)//1024} KB  |  {len(data)/sr:.2f}s)')


def plot_sweep(sig, sr, freqs, title, out_png):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), facecolor=DARK)
    fig.suptitle(title, color=TEXT, fontsize=11, fontweight='bold')

    duration = len(sig) / sr
    t = np.linspace(0, duration, len(sig))

    # Waveform (decimated for display)
    ax0 = axes[0]
    ax0.set_facecolor(PANEL)
    step = max(1, len(sig) // 80_000)
    ax0.plot(t[::step], sig[::step], color=BLUE, lw=0.5, alpha=0.8)
    ax0.set_xlim(0, duration)
    ax0.set_title('Waveform', color=TEXT, fontsize=9, loc='left')
    ax0.set_xlabel('Time (s)', color=MUTED, fontsize=8)
    ax0.set_ylabel('Amplitude', color=MUTED, fontsize=8)
    ax0.tick_params(colors=MUTED, labelsize=8)
    for sp in ax0.spines.values(): sp.set_edgecolor(GRID)
    ax0.grid(True, color=GRID, lw=0.5, alpha=0.6)

    # Spectrogram
    ax1 = axes[1]
    ax1.set_facecolor(PANEL)
    nperseg = 2048
    f_sp, t_sp, Sxx = dsp.spectrogram(sig, fs=sr, nperseg=nperseg,
                                        noverlap=nperseg * 3 // 4, window='hann')
    fmask = f_sp <= (max(freqs) * 1.2 if freqs is not None else sr / 2)
    Sdb = 10 * np.log10(Sxx[fmask] + 1e-30)
    ax1.grid(False)
    ax1.pcolormesh(t_sp, f_sp[fmask], Sdb, shading='gouraud', cmap='inferno',
                   vmin=np.percentile(Sdb, 5), vmax=np.percentile(Sdb, 99), rasterized=True)
    ax1.set_xlim(0, duration)
    ax1.set_title('Spectrogram', color=TEXT, fontsize=9, loc='left')
    ax1.set_xlabel('Time (s)', color=MUTED, fontsize=8)
    ax1.set_ylabel('Frequency (Hz)', color=MUTED, fontsize=8)
    ax1.tick_params(colors=MUTED, labelsize=8)
    for sp in ax1.spines.values(): sp.set_edgecolor(GRID)

    # Annotate stepped-tone frequencies if provided
    if freqs is not None and len(freqs) <= 40:
        for fc in freqs:
            ax1.axhline(fc, color=YLW, lw=0.7, ls='--', alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.patch.set_facecolor(DARK)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=DARK)
    print(f'[saved] {out_png}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--f-start',    type=float, default=50,
                    help='Start frequency Hz (default 50)')
    ap.add_argument('--f-stop',     type=float, default=8000,
                    help='Stop frequency Hz (default 8000)')
    ap.add_argument('--duration',   type=float, default=30,
                    help='Chirp duration seconds (default 30)')
    ap.add_argument('--amplitude',  type=float, default=0.9,
                    help='Peak amplitude 0-1 (default 0.9)')
    ap.add_argument('--stepped',    action='store_true',
                    help='Generate stepped tones instead of a chirp')
    ap.add_argument('--n-tones',    type=int,   default=20,
                    help='Number of stepped tones (default 20, stepped mode only)')
    ap.add_argument('--tone-dur',   type=float, default=1.0,
                    help='Duration of each tone in seconds (default 1.0)')
    ap.add_argument('--silence-dur',type=float, default=0.5,
                    help='Silence gap between tones in seconds (default 0.5)')
    ap.add_argument('--out',        default=None,
                    help='Output WAV path (default sweep_cal.wav)')
    ap.add_argument('--no-show',    action='store_true')
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use('Agg')

    out_path = args.out or os.path.join(SCRIPT_DIR, 'sweep_cal.wav')
    out_png  = os.path.splitext(out_path)[0] + '.png'

    if args.stepped:
        print(f'[gen]  Stepped tones  {args.f_start:.0f}–{args.f_stop:.0f} Hz'
              f'  n={args.n_tones}  tone={args.tone_dur}s  silence={args.silence_dur}s')
        sig, freqs = make_stepped_tones(
            args.f_start, args.f_stop, args.n_tones,
            args.tone_dur, args.silence_dur, AUD_SR, args.amplitude
        )
        title = (f'Stepped tones  {args.f_start:.0f}–{args.f_stop:.0f} Hz  '
                 f'({args.n_tones} tones × {args.tone_dur}s)')
        print(f'       frequencies: {", ".join(f"{f:.0f}" for f in freqs)} Hz')
    else:
        print(f'[gen]  Log-chirp  {args.f_start:.0f}–{args.f_stop:.0f} Hz  '
              f'duration={args.duration}s')
        sig    = make_log_chirp(args.f_start, args.f_stop, args.duration, AUD_SR, args.amplitude)
        freqs  = None
        title  = (f'Log-chirp  {args.f_start:.0f}–{args.f_stop:.0f} Hz  '
                  f'{args.duration:.0f}s')

    print(f'[gen]  {len(sig):,} samples  ({len(sig)/AUD_SR:.2f}s)  '
          f'peak={np.abs(sig).max():.3f}  rms={np.sqrt(np.mean(sig**2)):.3f}')

    save_wav(out_path, sig, AUD_SR)
    plot_sweep(sig, AUD_SR, freqs, title, out_png)

    print()
    print('  Next step: put sweep_cal.wav in this directory and run simple_rx.py')
    print('  to capture the powerline response, then run tf_estimate.py.')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
