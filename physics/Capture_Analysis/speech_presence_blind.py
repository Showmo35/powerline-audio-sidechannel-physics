#!/usr/bin/env python3
"""
speech_presence_blind.py — the HONEST version of speech_presence.py.

speech_presence.py looked good because process_mel() injected the clean REFERENCE
into the powerline mel: it overwrote every mel bin's mean (step 5) and every
frame's energy (step 6) with the reference's, then scored the result against...
the reference.  That's leakage — unreproducible at deployment, where there is no
reference.

Here the powerline → mel → WaveRNN path uses NO reference at all.  Normalisation
to WaveRNN's expected range uses only the powerline's OWN statistics plus fixed
constants.  The reference is loaded ONLY at the end, to score (mel r, envelope r,
and the metric that actually matters: WER via Whisper).

Runs both arms for a direct A/B through the SAME WaveRNN:
    BLIND : powerline-only normalisation (honest)
    LEAKY : original process_mel (reference-injected) — to reproduce the illusion

Usage:
    python3 speech_presence_blind.py --no-show
    python3 speech_presence_blind.py --dur 60 --arms blind,leaky
"""

import argparse, os, sys, wave
import numpy as np

import neural_vocoder as nv
from am_vocoder import extract_mel_am
from am_reconstruct import detect_mains

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPROOT = '<REPO_ROOT>/data'
DEF_CAP = os.path.join(CAPROOT, 'soundbar_bin_captures', 'chunk_002.bin')
DEF_WAV = os.path.join(CAPROOT, 'audio_chunks', 'chunk_002.wav')

T_LO, T_HI = -11.0, 2.0          # Tacotron2/LJSpeech log-mel range (fixed)


def load_wav(path, start_s, dur_s, dst_sr):
    with wave.open(path, 'rb') as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        w.setpos(int(start_s * sr))
        raw = w.readframes(int(dur_s * sr))
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(1)
    return nv.downsample(x, sr, dst_sr) if sr != dst_sr else x


def process_mel_blind(mel_raw, scale=2.2, offset=-5.0):
    """Powerline-ONLY normalisation to WaveRNN range. No reference anywhere.

    Preserves the powerline's relative spectral/temporal structure; only rescales
    globally (single scalars) into the Tacotron2 range with FIXED constants.
    """
    m = np.log(np.maximum(mel_raw, 1e-5))
    m = np.clip(m, T_LO, T_HI)
    m = (m - m.mean()) / (m.std() + 1e-5)        # global zero-mean/unit-std (own stats)
    m = m * scale + offset                        # fixed affine → ~[-11, 2]
    return np.clip(m, T_LO, T_HI).astype(np.float32)


def score(recon, ref, asr, ref_text):
    """Reference used ONLY here. Returns (env_r, mel_r_speech, wer)."""
    from scipy.stats import pearsonr
    fr = int(nv.FRAME_S * nv.AUD_SR)
    nf = min(len(ref), len(recon)) // fr
    a = np.array([np.sqrt(np.mean(ref[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    r = np.array([np.sqrt(np.mean(recon[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    env_r = pearsonr(a, r)[0] if nf > 2 else float('nan')

    mr = np.log(np.maximum(nv.extract_mel(ref), 1e-5))
    mc = np.log(np.maximum(nv.extract_mel(recon), 1e-5))
    Tm = min(mr.shape[1], mc.shape[1])
    sp = mr[:, :Tm].mean(0) > np.median(mr[:, :Tm].mean(0))
    mel_r = float(np.corrcoef(mc[:, :Tm][:, sp].ravel(),
                              mr[:, :Tm][:, sp].ravel())[0, 1])

    rec16 = nv.downsample(recon, nv.AUD_SR, 16000)
    hyp = asr.transcribe(rec16.astype(np.float32), language='en',
                         fp16=True, verbose=False)['text']
    w = wer(ref_text, hyp)
    return env_r, mel_r, w, hyp


def wer(ref, hyp):
    import re
    nrm = lambda s: ' '.join(re.sub(r"[^a-z0-9'\s]", ' ', s.lower()).split())
    r, h = nrm(ref).split(), nrm(hyp).split()
    n, m = len(r), len(h)
    d = np.zeros((n+1, m+1), int); d[:, 0] = np.arange(n+1); d[0, :] = np.arange(m+1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1]+(r[i-1] != h[j-1]))
    return d[n, m] / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', default=DEF_CAP)
    ap.add_argument('--audio', default=DEF_WAV)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--dur', type=float, default=60.0)
    ap.add_argument('--mains', type=float, default=60.0)
    ap.add_argument('--n-harmonics', type=int, default=8)
    ap.add_argument('--arms', default='blind,leaky')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()
    arms = args.arms.split(',')

    # ── load powerline window + reference window ──────────────────────────────
    off = int(args.start * nv.CAP_SR)
    cap = np.fromfile(args.capture, np.float32, count=int(args.dur*nv.CAP_SR), offset=off*4)
    cap_ds = nv.downsample(cap, nv.CAP_SR, nv.AUD_SR)
    ref = load_wav(args.audio, args.start, args.dur, nv.AUD_SR)
    n = min(len(cap_ds), len(ref)); cap_ds, ref = cap_ds[:n], ref[:n]
    print(f'[cap] {args.dur:.0f}s window  cap_ds={len(cap_ds)}  ref={len(ref)}')

    f_mains = detect_mains(cap_ds, nv.AUD_SR, args.mains)
    mel_raw = extract_mel_am(cap_ds, f_mains, args.n_harmonics)

    # ── Whisper for scoring + reference pseudo-truth (reference used ONLY here) ─
    import whisper
    asr = whisper.load_model('small', device=args.device)
    ref16 = nv.downsample(ref, nv.AUD_SR, 16000)
    ref_text = asr.transcribe(ref16.astype(np.float32), language='en',
                              fp16=True, verbose=False)['text']
    print(f'[ref ] truth: "{ref_text[:90]}…"')

    results = {}
    # BLIND arm ----------------------------------------------------------------
    if 'blind' in arms:
        mel_b = process_mel_blind(mel_raw)
        rec_b = nv.run_wavernn(mel_b, device=args.device)
        nv.save_wav(os.path.join(SCRIPT_DIR, 'blind_recon.wav'), rec_b, nv.AUD_SR)
        results['BLIND (no reference)'] = score(rec_b, ref, asr, ref_text)

    # LEAKY arm (reproduces the original illusion) ------------------------------
    if 'leaky' in arms:
        lag_path = os.path.splitext(args.capture)[0] + '.lag'
        lag_ms = float(open(lag_path).read().strip()) if os.path.exists(lag_path) else 0.0
        lag_delay = int(round(lag_ms / (nv.FRAME_S * 1000)))
        sp_mask = nv.speech_mask(ref, nv.AUD_SR)
        if lag_delay > 0:
            sp_mask = np.r_[np.zeros(lag_delay, bool), sp_mask[:-lag_delay]][:len(sp_mask)]
        mel_ref = nv.extract_mel(ref)
        mel_l, _ = nv.process_mel(mel_raw, sp_mask, mel_ref, lag_delay)
        rec_l = nv.run_wavernn(mel_l, device=args.device)
        nv.save_wav(os.path.join(SCRIPT_DIR, 'leaky_recon.wav'), rec_l, nv.AUD_SR)
        results['LEAKY (reference-injected)'] = score(rec_l, ref, asr, ref_text)

    # ── report ────────────────────────────────────────────────────────────────
    print('\n══ speech_presence: blind vs leaky (scored on reference) ══════════')
    print(f'  {"arm":28s} {"env r":>7} {"mel r":>7} {"WER":>8}')
    for name, (er, mr, w, hyp) in results.items():
        print(f'  {name:28s} {er:7.3f} {mr:7.3f} {w*100:7.1f}%')
    for name, (_, _, _, hyp) in results.items():
        print(f'\n  {name} transcript:\n    "{hyp[:160]}"')
    print('\n  (env r / mel r resemble the reference; WER is intelligibility.)')
    print('══════════════════════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
