#!/usr/bin/env python3
"""
speech_presence.py — Is speech present in a soundbar powerline capture?

Runs the AM-sideband neural vocoder (am_vocoder front-end + WaveRNN) on a window
of a long soundbar capture chunk and asks one question: does the powerline
current carry recoverable speech, and *when*?

Unlike am_vocoder.py (hardcoded to the short capture.bin + sibling wav), this
driver:
  * reads only the requested window of a multi-GB chunk (np.fromfile count=…),
  * pairs it with the matching reference audio chunk and its .lag sidecar,
  * caps WaveRNN synthesis to that window (full 30-min chunk would be hours),
  * adds a speech-activity (VAD) summary so "presence" is quantified, not just
    eyeballed.

The reference audio is used ONLY to (a) measure how good the reconstruction is
(envelope r, mel r, sample xcorr) and (b) provide the ground-truth speech mask
for the presence summary — it is the original signal played through the soundbar.

Usage
-----
    python3 speech_presence.py                       # chunk_002, first 60 s
    python3 speech_presence.py --start 0 --dur 60
    python3 speech_presence.py --device cuda --no-show
"""

import argparse
import os
import sys
import wave

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import neural_vocoder as nv
from am_vocoder import extract_mel_am
from am_reconstruct import detect_mains

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPROOT = '<REPO_ROOT>/Powerline_Data_Captures'
DEF_CAP = os.path.join(CAPROOT, 'soundbar_bin_captures', 'chunk_002.bin')
DEF_AUD = os.path.join(CAPROOT, 'audio_chunks', 'chunk_002.wav')


def decode_wav(path, dst_sr=nv.AUD_SR):
    """Read a PCM WAV directly (no ffmpeg) → mono float32 at dst_sr."""
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    x /= float(np.iinfo(dt).max)
    if sr != dst_sr:
        x = nv.downsample(x, sr, dst_sr)
    return x.astype(np.float32)


def load_capture_window(path, start_s, dur_s, sr=nv.CAP_SR):
    """Read only [start_s, start_s+dur_s) of a real-float32 capture."""
    offset = int(round(start_s * sr))
    count = int(round(dur_s * sr))
    x = np.fromfile(path, dtype=np.float32, count=count,
                    offset=offset * 4)          # 4 bytes / float32
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture', default=DEF_CAP)
    ap.add_argument('--audio', default=DEF_AUD)
    ap.add_argument('--start', type=float, default=0.0, help='Window start (s)')
    ap.add_argument('--dur', type=float, default=60.0, help='Window length (s)')
    ap.add_argument('--mains', type=float, default=60.0)
    ap.add_argument('--n-harmonics', type=int, default=8)
    ap.add_argument('--upper-only', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-show', action='store_true')
    ap.add_argument('--tag', default=None, help='Output filename tag')
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use('Agg')
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[warn] cuda requested but unavailable — falling back to cpu')
        args.device = 'cpu'
    print(f'[dev]   torch device = {args.device}')

    tag = args.tag or f's{int(args.start)}_d{int(args.dur)}'

    # ── load capture window ───────────────────────────────────────────────────
    cap = load_capture_window(args.capture, args.start, args.dur)
    print(f'[cap]   {os.path.basename(args.capture)}  '
          f'[{args.start:.0f}..{args.start+args.dur:.0f}]s  '
          f'{len(cap):,} samples @ {nv.CAP_SR} Hz')

    # ── load matching reference-audio window ──────────────────────────────────
    aud_full = decode_wav(args.audio, nv.AUD_SR)
    a0 = int(round(args.start * nv.AUD_SR))
    a1 = int(round((args.start + args.dur) * nv.AUD_SR))
    aud = aud_full[a0:a1].copy()
    print(f'[audio] {os.path.basename(args.audio)}  {len(aud):,} samples @ {nv.AUD_SR} Hz')

    duration = min(len(cap) / nv.CAP_SR, len(aud) / nv.AUD_SR)
    cap = cap[:int(duration * nv.CAP_SR)]
    aud = aud[:int(duration * nv.AUD_SR)]

    # ── downsample capture 200 kHz → 22050 Hz ─────────────────────────────────
    print('[ds]    downsampling capture 200 kHz → 22050 Hz …')
    cap_ds = nv.downsample(cap, nv.CAP_SR, nv.AUD_SR)
    cap_ds = cap_ds[:len(aud)]

    # ── lag (sidecar; powerline lags audio) ───────────────────────────────────
    lag_path = os.path.splitext(args.capture)[0] + '.lag'
    if os.path.exists(lag_path):
        lag_ms = float(open(lag_path).read().strip())
        print(f'[lag]   read from sidecar: {lag_ms:+.0f} ms')
    else:
        lag_ms = 0.0
        print('[lag]   no sidecar — assuming 0 ms')
    lag_delay = int(round(lag_ms / (nv.FRAME_S * 1000)))

    # ── ground-truth speech mask from reference (lag-corrected to capture) ────
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

    # ── mains + AM-sideband mel front-end ─────────────────────────────────────
    f_mains = detect_mains(cap_ds, nv.AUD_SR, args.mains)
    print('[mel]   building AM-sideband mel from powerline capture …')
    mel_raw = extract_mel_am(cap_ds, f_mains, args.n_harmonics, args.upper_only)
    mel_aud_ref = nv.extract_mel(aud)
    mel_proc, mel_is_speech = nv.process_mel(mel_raw, sp_mask_cap,
                                             mel_aud_ref, lag_delay)

    # ── WaveRNN synthesis ─────────────────────────────────────────────────────
    recon = nv.run_wavernn(mel_proc.astype(np.float32), device=args.device)

    # ── reconstruction metrics ────────────────────────────────────────────────
    fr = int(nv.FRAME_S * nv.AUD_SR)
    nf = min(len(aud), len(recon)) // fr
    aud_rms = np.array([np.sqrt(np.mean(aud[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    rec_rms = np.array([np.sqrt(np.mean(recon[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    r_env, _ = pearsonr(aud_rms, rec_rms)

    n_cmp = min(len(aud), len(recon))
    a_n = aud[:n_cmp] / (np.abs(aud[:n_cmp]).max() + 1e-9)
    r_n = recon[:n_cmp] / (np.abs(recon[:n_cmp]).max() + 1e-9)
    # FFT-based correlation: np.correlate(mode='full') is O(N^2) and stalls for
    # minutes at 60 s (1.3M samples). scipy method='fft' is O(N log N).
    from scipy.signal import correlate as _xcorr
    peak_xcorr = (_xcorr(a_n, r_n, mode='full', method='fft') / n_cmp).max()

    mel_ref_log = np.log(np.maximum(mel_aud_ref, 1e-5))
    Tm = min(mel_proc.shape[1], mel_ref_log.shape[1])
    sp_cols = np.zeros(Tm, dtype=bool)
    fpm = nv.FRAME_S * nv.AUD_SR / nv.MEL_HOP
    for i, is_sp in enumerate(sp_mask_cap):
        sp_cols[int(round(i*fpm)):min(int(round((i+1)*fpm)), Tm)] = is_sp
    mel_r = (float(np.corrcoef(mel_proc[:, :Tm][:, sp_cols].ravel(),
                               mel_ref_log[:, :Tm][:, sp_cols].ravel())[0, 1])
             if sp_cols.any() else float('nan'))

    # ── speech-presence (VAD) summary on the recon vs reference ───────────────
    # Detected-active frames = recon RMS above 5% of its own peak (same rule as
    # speech_mask). Compare to ground-truth speech frames from the reference.
    rec_active = rec_rms > 0.05 * rec_rms.max()
    gt = sp_mask_cap[:nf]
    da = rec_active[:nf]
    tp = int((gt & da).sum()); fp = int((~gt & da).sum())
    fn = int((gt & ~da).sum()); tn = int((~gt & ~da).sum())
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    speech_frac_gt = gt.mean()

    # active segments (in capture/recon timeline) from the reference mask
    seg = []
    in_seg = False
    for i, v in enumerate(gt):
        t = i * nv.FRAME_S
        if v and not in_seg:
            in_seg = True; s0 = t
        elif not v and in_seg:
            in_seg = False; seg.append((s0, t))
    if in_seg:
        seg.append((s0, len(gt) * nv.FRAME_S))

    print()
    print('══ Speech presence ══════════════════════════════')
    print(f'  Window           = [{args.start:.0f}, {args.start+duration:.0f}] s')
    print(f'  Mains frequency  = {f_mains:.3f} Hz')
    print(f'  Speech (ref)     = {speech_frac_gt*100:.1f}% of window '
          f'({gt.sum()} / {nf} frames)')
    print(f'  Active segments  = {len(seg)}')
    for s0, s1 in seg[:40]:
        print(f'      {s0+args.start:6.2f} – {s1+args.start:6.2f} s  ({s1-s0:.2f}s)')
    if len(seg) > 40:
        print(f'      … (+{len(seg)-40} more)')
    print(f'  VAD precision    = {precision:.3f}   recall = {recall:.3f}'
          f'   (recon-active vs ref-speech, 50 ms frames)')
    print('  ── reconstruction fidelity ──')
    print(f'  Envelope r       = {r_env:.4f}')
    print(f'  Mel r (speech)   = {mel_r:.4f}')
    print(f'  Sample xcorr     = {peak_xcorr:.4f}')
    print('══════════════════════════════════════════════════')

    # ── save wavs + figure ────────────────────────────────────────────────────
    out_wav = os.path.join(SCRIPT_DIR, f'speech_presence_{tag}.wav')
    nv.save_wav(out_wav, recon, nv.AUD_SR)
    nv.save_wav(os.path.join(SCRIPT_DIR, f'reference_{tag}.wav'), aud, nv.AUD_SR)

    out_png = os.path.join(SCRIPT_DIR, f'speech_presence_{tag}.png')
    nv.plot_results(aud, cap_ds, recon, mel_proc, mel_is_speech,
                    r_env, peak_xcorr, os.path.basename(args.audio), out_png)

    print(f'\n  Listen:  ffplay -nodisp {os.path.basename(out_wav)}')
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
