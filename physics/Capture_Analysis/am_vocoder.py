#!/usr/bin/env python3
"""
am_vocoder.py — AM-sideband front-end for the WaveRNN neural vocoder.

Why this exists
---------------
neural_vocoder.py builds the WaveRNN conditioning mel by *downsampling* the
200 kHz powerline capture to 22050 Hz and mel-ing it directly.  That collapses
everything to the speech ENVELOPE: the resulting "processed powerline mel" is
flat across frequency (every bin lights up together), so WaveRNN only knows
*when* speech happens, not *what* was said (envelope r high, sample xcorr ~0).

But the audio spectrum is not gone — per the AM-coupling model it lives in the
SIDEBANDS around each mains harmonic:  audio at f  →  capture at n·f_mains ± f.
am_reconstruct.py already showed those sidebands reconstruct the audio
spectrogram at r ≈ 0.74–0.80.  This script bridges the two: it builds the
WaveRNN conditioning mel from the AM-sideband estimate instead of the raw
downsampled capture, so the mel carries real formant structure.

The ONLY change vs neural_vocoder.py is the mel front-end (extract_mel_am).
Everything downstream — process_mel normalisation, WaveRNN, metrics, plotting —
is reused unchanged so the improvement is isolated to the conditioning input.

Note on calibration: process_mel re-aligns each mel bin's mean (in log space)
to the reference audio mel.  Any per-frequency multiplicative constant — i.e.
the H_AM(f) sweep calibration in am_reconstruct.py — is a constant log offset
per bin and is cancelled exactly by that alignment.  So H_AM is irrelevant on
the mel path and is omitted here.

Usage
-----
    python3 am_vocoder.py --no-show
    python3 am_vocoder.py --n-harmonics 8
    python3 am_vocoder.py --upper-only            # pure n=1 frequency shift
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchaudio.transforms as T
from scipy.stats import pearsonr

# Reuse the tested machinery from the sibling scripts.
import neural_vocoder as nv
from am_reconstruct import detect_mains

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── AM-sideband mel front-end (drop-in replacement for nv.extract_mel) ─────────

def extract_mel_am(cap_ds, f_mains, n_harmonics=8, upper_only=False,
                   sr=nv.AUD_SR):
    """Magnitude mel-spectrogram [n_mels, T] built from AM sidebands.

    Produces output in the *same domain* as nv.extract_mel (a magnitude mel on
    the WaveRNN time grid, hop=MEL_HOP), so it can be passed straight into
    nv.process_mel.

    Steps
    -----
    1. Complex STFT of the capture on the exact WaveRNN analysis grid
       (n_fft=MEL_N_FFT, win=MEL_WIN, hop=MEL_HOP) → |Y(f, t)|.
    2. Remap sidebands to baseband audio frequency:
         upper_only : |X(f, t)| = |Y(f_mains + f, t)|        (pure freq shift)
         else       : |X(f, t)| = Σ_n |Y(n·f_mains+f, t)| + |Y(|n·f_mains−f|, t)|
    3. Mel filterbank (same params as nv.extract_mel) → magnitude mel.
    """
    x_t = torch.from_numpy(cap_ds).float()
    window = torch.hann_window(nv.MEL_WIN)
    # center=True / normalized=False matches torchaudio.transforms.Spectrogram,
    # which is what nv.extract_mel uses internally → identical time framing.
    Y = torch.stft(x_t, n_fft=nv.MEL_N_FFT, hop_length=nv.MEL_HOP,
                   win_length=nv.MEL_WIN, window=window,
                   center=True, return_complex=True)        # [F, T]
    Y_abs = Y.abs().numpy()                                  # [F, T]
    n_f, n_t = Y_abs.shape
    df = sr / nv.MEL_N_FFT
    f_stft = np.arange(n_f) * df

    X_lin = np.zeros_like(Y_abs)
    if upper_only:
        sb_bins = np.round((f_mains + f_stft) / df).astype(int)
        valid = (sb_bins >= 0) & (sb_bins < n_f)
        X_lin[valid, :] = Y_abs[sb_bins[valid], :]
        print(f'[AM-mel] upper-only shift: audio(f) ← capture({f_mains:.1f}+f Hz)  '
              f'valid {valid.sum()}/{n_f} bins')
    else:
        for n in range(1, n_harmonics + 1):
            for f_sb in (n * f_mains + f_stft, np.abs(n * f_mains - f_stft)):
                sb_bins = np.round(f_sb / df).astype(int)
                valid = (sb_bins >= 0) & (sb_bins < n_f)
                X_lin[valid, :] += Y_abs[sb_bins[valid], :]
        X_lin /= (2 * n_harmonics)
        print(f'[AM-mel] {n_harmonics} harmonics × 2 sidebands accumulated')

    # Linear-magnitude spectrogram → mel.  MelScale defaults (norm=None,
    # mel_scale='htk') match the MelSpectrogram used in nv.extract_mel.
    mel_scale = T.MelScale(n_mels=nv.MEL_N_MELS, sample_rate=sr,
                           f_min=nv.MEL_FMIN, f_max=nv.MEL_FMAX, n_stft=n_f)
    mel = mel_scale(torch.from_numpy(X_lin).float()).numpy()  # [n_mels, T]
    print(f'[AM-mel] mel shape={mel.shape}  '
          f'range=[{mel.min():.3e}, {mel.max():.3e}]')
    return mel


# ── main (mirrors neural_vocoder.main, swapping only the front-end) ────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture',     default=nv.CAPTURE_PATH)
    ap.add_argument('--audio',       default=None)
    ap.add_argument('--mains',       type=float, default=60.0,
                    help='Mains frequency guess Hz (default 60)')
    ap.add_argument('--n-harmonics', type=int, default=8)
    ap.add_argument('--upper-only',  action='store_true',
                    help='Pure n=1 upper-sideband shift instead of multi-harmonic sum')
    ap.add_argument('--no-show',     action='store_true')
    ap.add_argument('--device',      default='cpu')
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use('Agg')

    # ── load ──────────────────────────────────────────────────────────────────
    cap = nv.load_capture(args.capture)
    print(f'[cap]   {len(cap):,} samples @ {nv.CAP_SR} Hz  ({len(cap)/nv.CAP_SR:.3f}s)')

    audio_path = args.audio or nv.find_audio(SCRIPT_DIR)
    if not audio_path:
        sys.exit('[error] no audio file found')
    aud = nv.decode_audio(audio_path, nv.AUD_SR)
    print(f'[audio] {os.path.basename(audio_path)}  {len(aud):,} samples')

    duration = min(len(cap) / nv.CAP_SR, len(aud) / nv.AUD_SR)
    cap = cap[:int(duration * nv.CAP_SR)]
    aud = aud[:int(duration * nv.AUD_SR)]

    # ── downsample capture to audio rate ──────────────────────────────────────
    print('[ds]    downsampling capture 200 kHz → 22050 Hz …')
    cap_ds = nv.downsample(cap, nv.CAP_SR, nv.AUD_SR)
    cap_ds = cap_ds[:len(aud)]

    # ── lag: read sidecar or detect via xcorr (same as neural_vocoder) ────────
    lag_path = os.path.splitext(args.capture)[0] + '.lag'
    if os.path.exists(lag_path):
        lag_ms = float(open(lag_path).read().strip())
        print(f'[lag]   read from sidecar: {lag_ms:+.0f} ms')
    else:
        _a_rms = nv.frame_rms(aud, nv.AUD_SR)
        _c_rms = nv.frame_rms(cap_ds, nv.AUD_SR)
        _n = min(len(_a_rms), len(_c_rms))
        _a = (_a_rms[:_n] - _a_rms[:_n].mean()) / (_a_rms[:_n].std() + 1e-12)
        _c = (_c_rms[:_n] - _c_rms[:_n].mean()) / (_c_rms[:_n].std() + 1e-12)
        _xc = np.correlate(_a, _c, mode='full') / _n
        _lags = np.arange(-(_n - 1), _n)
        _valid = (-int(round(0.700 / nv.FRAME_S)) <= _lags) & (_lags <= 0)
        lag_ms = -float(_lags[_valid][np.argmax(_xc[_valid])]) * nv.FRAME_S * 1000
        print(f'[lag]   auto-detected: {lag_ms:+.0f} ms')
    lag_delay = int(round(lag_ms / (nv.FRAME_S * 1000)))

    # ── speech/silence mask (lag-corrected for powerline) ─────────────────────
    sp_mask_audio = nv.speech_mask(aud, nv.AUD_SR)
    n_fr = len(sp_mask_audio)
    if lag_delay > 0:
        sp_mask_cap = np.r_[np.zeros(lag_delay, dtype=bool),
                            sp_mask_audio[:-lag_delay]][:n_fr]
    elif lag_delay < 0:
        _d = -lag_delay
        sp_mask_cap = np.r_[sp_mask_audio[_d:], np.zeros(_d, dtype=bool)][:n_fr]
    else:
        sp_mask_cap = sp_mask_audio.copy()
    print(f'[mask]  speech={sp_mask_cap.sum()}  silence={(~sp_mask_cap).sum()} frames')

    # ── detect mains ──────────────────────────────────────────────────────────
    f_mains = detect_mains(cap_ds, nv.AUD_SR, args.mains)

    # ── AM-SIDEBAND mel front-end (the only change vs neural_vocoder) ─────────
    print('[mel]   building AM-sideband mel from powerline capture …')
    mel_raw     = extract_mel_am(cap_ds, f_mains, args.n_harmonics, args.upper_only)
    mel_aud_ref = nv.extract_mel(aud)

    # ── normalise to audio mel stats (reused unchanged) ───────────────────────
    mel_proc, mel_is_speech = nv.process_mel(mel_raw, sp_mask_cap,
                                             mel_aud_ref, lag_delay)

    # ── WaveRNN inference ─────────────────────────────────────────────────────
    recon = nv.run_wavernn(mel_proc.astype(np.float32), device=args.device)

    # ── metrics: envelope r, sample xcorr, AND mel/spectrogram r ──────────────
    fr = int(nv.FRAME_S * nv.AUD_SR)
    nf = min(len(aud), len(recon)) // fr
    aud_rms = np.array([np.sqrt(np.mean(aud[i*fr:(i+1)*fr]**2))   for i in range(nf)])
    rec_rms = np.array([np.sqrt(np.mean(recon[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    r_env, _ = pearsonr(aud_rms, rec_rms)

    n_cmp = min(len(aud), len(recon))
    a_n = aud[:n_cmp]   / (np.abs(aud[:n_cmp]).max()   + 1e-9)
    r_n = recon[:n_cmp] / (np.abs(recon[:n_cmp]).max() + 1e-9)
    peak_xcorr = (np.correlate(a_n, r_n, mode='full') / n_cmp).max()

    # Mel-domain correlation: how well does the conditioning mel match the
    # reference audio mel *in its time-varying structure* (formants), measured
    # only over speech frames so silence doesn't inflate it.
    mel_ref_log = np.log(np.maximum(mel_aud_ref, 1e-5))
    Tm = min(mel_proc.shape[1], mel_ref_log.shape[1])
    sp_cols = np.zeros(Tm, dtype=bool)
    fpm = nv.FRAME_S * nv.AUD_SR / nv.MEL_HOP
    for i, is_sp in enumerate(sp_mask_cap):
        sp_cols[int(round(i*fpm)):min(int(round((i+1)*fpm)), Tm)] = is_sp
    if sp_cols.any():
        mel_r = float(np.corrcoef(mel_proc[:, :Tm][:, sp_cols].ravel(),
                                  mel_ref_log[:, :Tm][:, sp_cols].ravel())[0, 1])
    else:
        mel_r = float('nan')

    print(f'[metric] envelope r        = {r_env:.4f}')
    print(f'[metric] sample xcorr peak = {peak_xcorr:.4f}')
    print(f'[metric] mel r (speech)    = {mel_r:.4f}')

    # ── save ──────────────────────────────────────────────────────────────────
    print('[wav]   saving …')
    mode = 'shift' if args.upper_only else f'h{args.n_harmonics}'
    out_wav = os.path.join(SCRIPT_DIR, f'am_vocoder_{mode}.wav')
    nv.save_wav(out_wav, recon, nv.AUD_SR)
    nv.save_wav(os.path.join(SCRIPT_DIR, 'original_reference.wav'), aud, nv.AUD_SR)

    # ── plot (reuse neural_vocoder layout; "powerline mel" panel now = AM mel) ─
    out_png = os.path.join(SCRIPT_DIR, f'am_vocoder_{mode}.png')
    nv.plot_results(aud, cap_ds, recon, mel_proc, mel_is_speech,
                    r_env, peak_xcorr, os.path.basename(audio_path), out_png)

    print()
    print('══ Results ══════════════════════════════════════')
    print(f'  Front-end        = AM sidebands '
          f'({"upper-only shift" if args.upper_only else str(args.n_harmonics)+" harmonics"})')
    print(f'  Mains frequency  = {f_mains:.3f} Hz')
    print(f'  Envelope r       = {r_env:.4f}')
    print(f'  Sample xcorr     = {peak_xcorr:.4f}')
    print(f'  Mel r (speech)   = {mel_r:.4f}')
    print(f'  Output WAV       = {os.path.basename(out_wav)}')
    print()
    print('  Listen:')
    print(f'    ffplay -nodisp {os.path.basename(out_wav)}')
    print('    ffplay -nodisp original_reference.wav')
    print('═════════════════════════════════════════════════')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
