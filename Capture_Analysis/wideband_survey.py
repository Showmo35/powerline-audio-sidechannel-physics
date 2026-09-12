#!/usr/bin/env python3
"""
wideband_survey.py — Decide whether the soundbar's AC-cord current carries audio
                     on a high-frequency switching carrier (Class-D / SMPS).

This is the decisive measurement: if a carrier with audio-modulated sidebands
exists above the 100 kHz Nyquist of the old capture, AM-demodulating it can
recover real audio bandwidth (→ intelligibility / ASR is on the table).  If not,
the AC-cord channel is envelope-limited and STOI is capped near ~0.56 regardless
of how much data you collect.

What it does
------------
  1. Wideband PSD of the real capture (0 → Fs/2).
  2. Find candidate carriers: narrowband peaks above --min-carrier Hz.
  3. For each candidate, AM-demodulate (bandpass → Hilbert envelope) and score:
       * If a reference audio is given → envelope correlation with the audio
         (lag-aligned).  High r  = the carrier carries the audio.
       * Always → "modulation ratio": envelope energy in 100 Hz–5 kHz
         (speech-like) vs <50 Hz (just power-line ripple).  >1 means the carrier
         is modulated by something speech-band, not just mains.
  4. Figure + a printed VERDICT with the best carrier, its sideband bandwidth,
     and a recommendation.

Usage
-----
    python3 wideband_survey.py --capture wideband.bin --samp-rate 4e6 --audio hello5.wav
    python3 wideband_survey.py --capture wideband.bin --samp-rate 4e6   # no reference
    python3 wideband_survey.py --capture capture.bin   --samp-rate 200e3 # sanity (old)
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUD_SR = 22_050

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


def resample_to(x, src, dst):
    if src == dst:
        return x.astype(np.float32)
    g = gcd(int(src), int(dst))
    return dsp.resample_poly(x, dst // g, src // g).astype(np.float32)


# ── carrier search ────────────────────────────────────────────────────────────

def wideband_psd(cap, sr, nperseg=1 << 16):
    f, P = dsp.welch(cap - cap.mean(), fs=sr, nperseg=min(nperseg, len(cap)),
                     window='blackman')
    return f, P


def find_carriers(f, P, min_carrier, n_top=5, prominence_db=8.0):
    """Find narrowband peaks above min_carrier Hz, ranked by prominence."""
    Pdb = 10 * np.log10(P + 1e-30)
    # local baseline = heavily-smoothed PSD
    win = max(11, (len(Pdb) // 200) | 1)
    base = dsp.medfilt(Pdb, win)
    excess = Pdb - base
    band = f >= min_carrier
    if band.sum() < 10:
        return []
    idx = np.where(band)[0]
    peaks, props = dsp.find_peaks(excess[idx], prominence=prominence_db,
                                  distance=max(3, len(idx) // 100))
    if len(peaks) == 0:
        return []
    order = np.argsort(props['prominences'])[::-1][:n_top]
    out = []
    for p in peaks[order]:
        gi = idx[p]
        out.append(dict(freq=float(f[gi]), psd_db=float(Pdb[gi]),
                        prominence_db=float(excess[gi])))
    return out


# ── AM demodulation + scoring ──────────────────────────────────────────────────

def am_demod(cap, sr, fc, bw):
    """Bandpass around fc±bw, Hilbert envelope → AM-demodulated baseband."""
    lo, hi = max(fc - bw, 1.0), min(fc + bw, sr / 2 - 1.0)
    sos = dsp.butter(4, [lo, hi], 'bp', fs=sr, output='sos')
    bp = dsp.sosfiltfilt(sos, cap - cap.mean())
    env = np.abs(dsp.hilbert(bp))
    return env - env.mean()


def modulation_ratio(env, sr):
    """Energy of the envelope in 100 Hz-5 kHz (speech-like) vs <50 Hz (ripple)."""
    f, Pe = dsp.welch(env, fs=sr, nperseg=min(1 << 14, len(env)), window='hann')
    speech = ((f >= 100) & (f <= 5000))
    ripple = (f < 50)
    s = Pe[speech].mean() if speech.any() else 0.0
    r = Pe[ripple].mean() if ripple.any() else 1e-30
    return float(s / (r + 1e-30))


def envelope_corr(env, sr, ref, ref_sr):
    """Lag-aligned correlation of demod envelope with reference audio envelope."""
    from scipy.stats import pearsonr
    env_a = resample_to(env, int(sr), AUD_SR)
    ref_a = resample_to(ref, int(ref_sr), AUD_SR)
    fr = int(0.02 * AUD_SR)
    n = min(len(env_a), len(ref_a)) // fr
    if n < 8:
        return 0.0, 0.0
    e = np.array([np.sqrt(np.mean(env_a[i*fr:(i+1)*fr]**2)) for i in range(n)])
    r = np.array([np.sqrt(np.mean(ref_a[i*fr:(i+1)*fr]**2)) for i in range(n)])
    en = (e - e.mean()) / (e.std() + 1e-12)
    rn = (r - r.mean()) / (r.std() + 1e-12)
    xc = np.correlate(en, rn, mode='full') / n
    lags = np.arange(-(n-1), n)
    win = np.abs(lags) <= int(0.7 / 0.02)
    best = xc[win].max()
    best_lag_ms = lags[win][np.argmax(xc[win])] * 20
    return float(best), float(best_lag_ms)


# ── plotting ──────────────────────────────────────────────────────────────────

def _ax(ax, title, xlabel, ylabel, xlim=None, xlog=False):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, loc='left', pad=4)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    if xlim: ax.set_xlim(*xlim)
    if xlog: ax.set_xscale('log')


def make_figure(f, P, sr, carriers, best, ref, ref_sr, bw, out_png):
    Pdb = 10 * np.log10(P + 1e-30)
    fig = plt.figure(figsize=(15, 11), facecolor=DARK)
    verdict = (f'carrier @ {best["freq"]/1e3:.1f} kHz  '
               f'(prom {best["prominence_db"]:.0f} dB)' if best else 'NO carrier found')
    fig.suptitle(f'Wideband switching-carrier survey  —  Fs={sr/1e6:.2f} MSps  —  {verdict}',
                 color=TEXT, fontsize=12, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.28,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)

    # full-band PSD
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(PANEL)
    ax.plot(f / 1e3, Pdb, color=BLUE, lw=0.6)
    for c in carriers:
        ax.axvline(c['freq'] / 1e3, color=YLW, lw=0.8, ls='--', alpha=0.7)
        ax.text(c['freq'] / 1e3, Pdb.max(), f'{c["freq"]/1e3:.0f}k',
                color=YLW, fontsize=7, rotation=90, va='top')
    _ax(ax, 'Wideband PSD (real capture)', 'Frequency (kHz)', 'PSD (dB)',
        xlim=(0, f.max() / 1e3))

    if best:
        # zoom around carrier — show sidebands
        ax = fig.add_subplot(gs[1, 0])
        m = np.abs(f - best['freq']) <= 4 * bw
        ax.set_facecolor(PANEL)
        ax.plot((f[m] - best['freq']) / 1e3, Pdb[m], color=ORNG, lw=0.9)
        ax.axvline(0, color=YLW, lw=0.8, ls='--')
        _ax(ax, f'Zoom @ {best["freq"]/1e3:.1f} kHz (sidebands = audio?)',
            'Offset from carrier (kHz)', 'PSD (dB)')

        # demod envelope spectrogram
        env = am_demod(load_figure_cap.cap, sr, best['freq'], bw)
        env_a = resample_to(env, int(sr), AUD_SR)
        ax = fig.add_subplot(gs[1, 1])
        fsp, tsp, S = dsp.spectrogram(env_a, fs=AUD_SR, nperseg=1024,
                                      noverlap=768, window='hann')
        fm = fsp <= 5000
        Sdb = 10 * np.log10(S[fm] + 1e-30)
        ax.set_facecolor(PANEL); ax.grid(False)
        ax.pcolormesh(tsp, fsp[fm], Sdb, shading='gouraud', cmap='inferno',
                      vmin=np.percentile(Sdb, 5), vmax=np.percentile(Sdb, 99),
                      rasterized=True)
        _ax(ax, 'Demodulated envelope spectrogram', 'Time (s)', 'Audio freq (Hz)')

        # envelope vs reference (or modulation spectrum)
        ax = fig.add_subplot(gs[2, :])
        ax.set_facecolor(PANEL)
        if ref is not None:
            env_n = resample_to(env, int(sr), AUD_SR)
            n = min(len(env_n), len(resample_to(ref, int(ref_sr), AUD_SR)))
            ref_n = resample_to(ref, int(ref_sr), AUD_SR)[:n]
            env_n = env_n[:n]
            fr = int(0.02 * AUD_SR); nf = n // fr
            ee = np.array([np.sqrt(np.mean(env_n[i*fr:(i+1)*fr]**2)) for i in range(nf)])
            rr = np.array([np.sqrt(np.mean(ref_n[i*fr:(i+1)*fr]**2)) for i in range(nf)])
            t = np.arange(nf) * 0.02
            ax.plot(t, rr / (rr.max()+1e-9), color=BLUE, lw=1.5, label='reference audio env')
            ax.plot(t, ee / (ee.max()+1e-9), color=PURP, lw=1.5, alpha=0.85, label='demod carrier env')
            _ax(ax, 'Demod envelope vs reference audio', 'Time (s)', 'norm. env')
            ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
        else:
            fe, Pe = dsp.welch(env, fs=sr, nperseg=min(1<<14, len(env)), window='hann')
            mm = fe <= 6000
            ax.plot(fe[mm], 10*np.log10(Pe[mm]+1e-30), color=PURP, lw=1.0)
            _ax(ax, 'Envelope modulation spectrum (speech band = 100 Hz-5 kHz)',
                'Modulation freq (Hz)', 'dB')

    fig.patch.set_facecolor(DARK)
    fig.savefig(out_png, dpi=140, bbox_inches='tight', facecolor=DARK)
    print(f'[viz]   saved → {out_png}')


class load_figure_cap:
    cap = None  # set in main so make_figure can demod without re-passing


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture', default=os.path.join(SCRIPT_DIR, 'wideband.bin'))
    ap.add_argument('--samp-rate', type=float, required=True,
                    help='Sample rate the capture was recorded at (Sps)')
    ap.add_argument('--audio', default=None, help='Reference audio that was played (optional)')
    ap.add_argument('--min-carrier', type=float, default=50e3,
                    help='Ignore peaks below this Hz (default 50 kHz — skips mains/PSU)')
    ap.add_argument('--bw', type=float, default=10e3,
                    help='Demod bandwidth around carrier, Hz (default 10 kHz)')
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use('Agg')

    if not os.path.exists(args.capture):
        sys.exit(f'[error] capture not found: {args.capture}')
    cap = load_capture(args.capture)
    sr = args.samp_rate
    load_figure_cap.cap = cap
    print(f'[cap]   {len(cap):,} samples  {len(cap)/sr:.2f}s @ {sr/1e6:.3f} MSps  '
          f'(Nyquist {sr/2e6:.3f} MHz)')
    print(f'[cap]   DC={cap.mean():.4f}  AC rms={cap.std():.4f}')

    ref = ref_sr = None
    if args.audio and os.path.exists(args.audio):
        ref, ref_sr = read_wav(args.audio)
        print(f'[ref]   {os.path.basename(args.audio)}  {len(ref)/ref_sr:.2f}s')

    f, P = wideband_psd(cap, sr)
    carriers = find_carriers(f, P, args.min_carrier)

    if not carriers:
        print(f'\n[scan]  NO narrowband carrier found above {args.min_carrier/1e3:.0f} kHz.')
    else:
        print(f'\n[scan]  {len(carriers)} candidate carrier(s) above {args.min_carrier/1e3:.0f} kHz:')
        print(f'        {"freq(kHz)":>10} {"prom(dB)":>9} {"modRatio":>9} {"audio_r":>8} {"lag(ms)":>8}')
        for c in carriers:
            env = am_demod(cap, sr, c['freq'], args.bw)
            c['mod_ratio'] = modulation_ratio(env, sr)
            if ref is not None:
                c['audio_r'], c['lag_ms'] = envelope_corr(env, sr, ref, ref_sr)
            else:
                c['audio_r'], c['lag_ms'] = float('nan'), float('nan')
            print(f'        {c["freq"]/1e3:>10.1f} {c["prominence_db"]:>9.1f} '
                  f'{c["mod_ratio"]:>9.2f} {c["audio_r"]:>8.3f} {c["lag_ms"]:>8.0f}')

    # pick best: by audio_r if reference, else by modulation ratio
    best = None
    if carriers:
        if ref is not None and any(np.isfinite(c['audio_r']) for c in carriers):
            best = max(carriers, key=lambda c: (c['audio_r'] if np.isfinite(c['audio_r']) else -1))
        else:
            best = max(carriers, key=lambda c: c['mod_ratio'])

    out_png = os.path.join(SCRIPT_DIR, 'wideband_survey.png')
    make_figure(f, P, sr, carriers, best, ref, ref_sr, args.bw, out_png)

    # ── verdict ────────────────────────────────────────────────────────────────
    print('\n══ VERDICT ' + '═' * 52)
    if best is None:
        print('  No switching carrier with audio sidebands was detected.')
        print('  → The AC-cord current channel is envelope-limited.  Collecting more')
        print('    data will help VAD / limited-vocabulary recognition, but not open')
        print('    intelligibility.  Consider: faster Fs (probe/ADC may roll off a')
        print('    higher carrier), a wider-bandwidth current probe, or probing a DC')
        print('    rail / speaker output closer to the amplifier.')
    else:
        have_ref = np.isfinite(best.get('audio_r', np.nan))
        # Decision: with a reference, ONLY the audio correlation is trustworthy.
        # mod_ratio is informational — broadband noise also yields mod_ratio≫1,
        # so it must NOT, on its own, be read as "carries audio".
        confirmed = have_ref and best['audio_r'] > 0.4
        print(f'  Best carrier : {best["freq"]/1e3:.1f} kHz  '
              f'(prominence {best["prominence_db"]:.0f} dB)')
        print(f'  Mod ratio    : {best["mod_ratio"]:.2f}  '
              f'(info only — noise also gives ≫1, do not trust alone)')
        if have_ref:
            print(f'  Audio corr   : r={best["audio_r"]:.3f} @ {best["lag_ms"]:.0f} ms  '
                  f'(decisive metric)')
        if confirmed:
            print('  → CONFIRMED: this carrier tracks the played audio.  Demodulate it')
            print('    (bandpass→Hilbert, or quadrature demod around this freq) to recover')
            print('    wideband audio.  Build the dataset on THIS carrier — intelligibility')
            print('    / ASR is now plausible.  Re-point the reconstruction here.')
        elif have_ref:
            print('  → NOT CONFIRMED: peaks exist but none track the audio (r≤0.4) — these')
            print('    are spurs / clock lines, not the audio channel.  The AC-cord current')
            print('    is envelope-limited at this Fs.  Try: higher --samp-rate (carrier may')
            print('    sit higher), wider current-probe bandwidth, or probe a DC rail /')
            print('    speaker output closer to the amplifier.  More data alone won’t help.')
        else:
            print('  → INCONCLUSIVE: no reference audio given, so audio tracking could not')
            print('    be tested (mod_ratio alone is unreliable).  Re-run with --audio set')
            print('    to the file you played, so the carrier can be confirmed or rejected.')
    print('═' * 63)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
