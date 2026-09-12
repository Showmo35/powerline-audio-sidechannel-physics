#!/usr/bin/env python3
"""
comb.py — line-locked per-harmonic mains-comb front-end.

WHY THIS EXISTS
---------------
Every prior comb front-end in this repo (M26 analyze_pair.demod_baseband,
M26 highfreq_probe.build_sideband_matrix, M11 am_sideband_stack16,
Capture_Analysis/am_reconstruct.py, ...) has two defects:

 1. The mains grid was WRONG. `analyze_pair.detect_mains()` snaps a Welch argmax to a
    3.05 Hz bin and returns 61.035 Hz. Measured here on chunk_001 (60 s, 16 kHz
    decimated, 0.0167 Hz resolution): the true fundamental is 59.9988 Hz and the comb
    lands on k*f_true to within 0.3 Hz out to k=100. The 61.035 grid is off by
    -20.7 Hz at k=20, -62 Hz at k=60 (a FULL harmonic spacing) and -103 Hz at k=100.
    M26 selected its "harmonics" at 3-4 kHz == k~50-67, precisely where its grid was a
    whole spacing misregistered.
 2. Demodulation was MAGNITUDE-ONLY (Hilbert |.|). No coherent / phase-locked demod
    and no line tracking exists anywhere in the repo.

This module fixes both: a mHz-accurate fundamental (fit over harmonic peaks) and a
COHERENT complex demodulation of each individual harmonic at +-BW Hz (BW < half the
60 Hz spacing, so harmonics do NOT mix) yielding AM and PHASE per harmonic.

MODEL / HYPOTHESIS
------------------
The PSU rectifier mixes the speaker's current draw onto every mains harmonic. To first
order each harmonic sees the same envelope scaled by a constant, a_k(t) = c_k * env(t)
=> the harmonic-gram is RANK-1 and the K harmonics are redundant copies (no new info).
If instead the transfer H_k(f_audio) differs per harmonic (load-dependent conduction
angle), the K demodulated band-envelopes form a crude FILTER BANK over the audio
spectrum -- i.e. phonetics. Phase (conduction angle) is a channel the envelope ignores.
Which is true is empirical: see probe_pair.py Gate A (rank test).
"""
import os
import numpy as np
from scipy.signal import resample_poly

CAP_SR = 200_000


# ── precise line frequency ────────────────────────────────────────────────────
def _parabolic(logmag, k):
    a, b, c = logmag[k - 1], logmag[k], logmag[k + 1]
    d = 0.5 * (a - c) / (a - 2 * b + c + 1e-30)
    return float(np.clip(d, -0.5, 0.5))


def estimate_mains(x, sr=CAP_SR, guess=60.0, dec_sr=16_000, kfit=(5, 100)):
    """mHz-accurate mains fundamental.

    Decimate to `dec_sr` (fine FFT bins), locate the comb peaks near k*guess for a
    range of k, then FIT f0 by least squares through (k, peak_freq). Fitting over high
    harmonics gives far better precision than the fundamental alone (and is immune to
    the 3 Hz bin-snapping that produced the bogus 61.035 Hz).
    """
    up, dn = _ratio(dec_sr, sr)
    xd = resample_poly(np.asarray(x, np.float64), up, dn)
    N = len(xd)
    X = np.abs(np.fft.rfft(xd * np.hanning(N)))
    f = np.fft.rfftfreq(N, 1.0 / dec_sr)
    df = f[1] - f[0]
    lm = np.log(X + 1e-30)

    def peak_near(target, half=2.0):
        m = np.abs(f - target) < half
        idx = np.where(m)[0]
        if len(idx) < 3:
            return None
        k = idx[np.argmax(X[idx])]
        if k <= 0 or k >= len(X) - 1:
            return None
        return f[k] + _parabolic(lm, k) * df

    ks, fs_ = [], []
    for k in range(kfit[0], kfit[1] + 1):
        t = k * guess
        if t > 0.45 * dec_sr:
            break
        p = peak_near(t, half=min(2.0, 0.4 * guess))
        if p is not None:
            ks.append(k); fs_.append(p)
    if len(ks) < 5:                                   # fallback: fundamental only
        p = peak_near(guess, half=5.0)
        return float(p if p else guess)
    ks = np.asarray(ks, float); fs_ = np.asarray(fs_)
    f0 = float(np.sum(ks * fs_) / np.sum(ks * ks))    # LS through origin: f_k = k*f0
    return f0


def _ratio(dst, src):
    from math import gcd
    g = gcd(int(dst), int(src))
    return int(dst) // g, int(src) // g


# ── coherent per-harmonic demodulation ───────────────────────────────────────
def harmonic_gram(x, f0, sr=CAP_SR, K=120, bw=25.0, T=32, kmin=1, mid_sr=32_000,
                  crop=None):
    """COHERENT complex demod of each individual harmonic -> (K, T) complex.

    TIME-DOMAIN demodulation (deliberately NOT an FFT band-slice): for harmonic k we
    mix down by exp(-j2*pi*k*f0*t) and low-pass the complex baseband at `bw`.

    NOTE (important, learned the hard way): an FFT-slice implementation must taper the
    window first to control leakage from the huge k=1 carrier -- but that taper is a
    COMMON multiplicative envelope across every harmonic, which corrupts the AM and
    MANUFACTURES a rank-1 gram. That would fake the very result Gate A tests for. The
    time-domain path applies no taper: the low-pass isolates each harmonic instead.

    bw MUST be < f0/2 (=30 Hz) so adjacent harmonics never mix. Speech band-envelopes
    live at 0-30 Hz, so +-25 Hz captures them while keeping harmonics separated.

    `crop=(t0_s, t1_s)`: demodulate the FULL padded window (so the narrow low-pass has
    room to settle) then keep only [t0_s, t1_s]. A 25 Hz filter on a bare 0.5 s word
    rings over most of the word, so callers should pad and crop.

    Returns H (K, T) complex and the carrier freqs.
    """
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, np.float64)
    # one decimation to mid_sr covers all harmonics up to mid_sr/2 (K*f0 must be below)
    up, dn = _ratio(mid_sr, sr)
    xm = resample_poly(x, up, dn)
    n = len(xm)
    t = np.arange(n) / float(mid_sr)
    ks = np.arange(kmin, kmin + K)
    fcs = ks * f0
    dec = max(1, int(round(mid_sr / 1000.0)))         # mid_sr -> ~1 kHz
    lp_sr = mid_sr / dec
    sos = butter(4, bw, btype='low', fs=lp_sr, output='sos')
    H = np.zeros((K, T), np.complex128)
    for i, fc in enumerate(fcs):
        if fc >= 0.45 * mid_sr:
            break
        y = xm * np.exp(-2j * np.pi * fc * t)         # shift harmonic to DC (no taper)
        y = resample_poly(y, 1, dec)                  # -> ~1 kHz complex baseband
        if len(y) <= 24:
            continue
        y = sosfiltfilt(sos, y)                       # isolate +-bw (full padded window)
        if crop is not None:                          # drop the filter's settling region
            a = max(0, int(round(crop[0] * lp_sr)))
            b = min(len(y), int(round(crop[1] * lp_sr)))
            if b - a >= 4:
                y = y[a:b]
        idx = np.linspace(0, len(y) - 1, T)
        H[i] = (np.interp(idx, np.arange(len(y)), y.real)
                + 1j * np.interp(idx, np.arange(len(y)), y.imag))
    return H, fcs


def gram_features(H, use_phase=True, eps=1e-12):
    """(K,T) complex -> feature blocks.

    AM  : log|H| per harmonic, per-harmonic mean-removed (so absolute gain c_k, which
          is a constant scale per harmonic, cannot by itself carry word identity).
    PM  : phase, unwrapped along time and linearly detrended (removes any residual
          carrier offset), leaving conduction-angle wobble.
    """
    A = np.log(np.abs(H) + eps)
    A = A - A.mean(axis=1, keepdims=True)
    if not use_phase:
        return A.astype(np.float32)
    P = np.unwrap(np.angle(H), axis=1)
    t = np.arange(H.shape[1])
    # remove per-harmonic linear trend in phase
    tm = t - t.mean()
    slope = (P * tm).sum(1, keepdims=True) / (tm ** 2).sum()
    P = P - slope * tm - P.mean(axis=1, keepdims=True)
    return np.concatenate([A, P], axis=0).astype(np.float32)


def harmonic_snr(x, f0, sr=CAP_SR, K=200, bw=25.0):
    """Per-harmonic SNR: carrier-band power vs. the local off-comb floor (midway
    between harmonics). Unlike M26's select_harmonics (raw Welch power, which conflates
    carrier with sidebands) this measures how far each harmonic stands above its own
    neighbourhood -> the honest 'which k are usable' curve."""
    x = np.asarray(x, np.float64)
    N = len(x)
    P = np.abs(np.fft.rfft(x * np.hanning(N))) ** 2
    df = sr / N
    out = np.zeros(K)
    for i, k in enumerate(range(1, K + 1)):
        c = int(round(k * f0 / df)); nb = max(2, int(round(bw / df)))
        m = int(round((k + 0.5) * f0 / df))           # midpoint to next harmonic
        if c + nb >= len(P) or m + nb >= len(P):
            break
        sig = P[c - nb:c + nb + 1].sum()
        flo = P[m - nb:m + nb + 1].sum() + 1e-30
        out[i] = 10 * np.log10(sig / flo)
    return out
