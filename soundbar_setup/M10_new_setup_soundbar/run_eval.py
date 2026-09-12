#!/usr/bin/env python3
"""
run_eval.py — STEP 1 of the ground-up pipeline: the intelligibility go/no-go.

For a capture window:
  1. read powerline .bin window  → downsample → AM-sideband mel  (frontend)
  2. Griffin-Lim mel → audio → 16 kHz                            (reconstruct)
  3. Whisper-transcribe the reconstruction AND the clean reference
  4. WER(reconstruction vs reference-as-truth)                   (asr)
  5. dump transcripts + metrics JSON to outputs/

This deliberately uses NO neural vocoder and NO reference signal inside the
front-end, so the exact same path scales to the full 20 h corpus.  The reference
audio is used only to produce the pseudo ground-truth transcript and to report
how good the channel is.

Usage:
    python run_eval.py                               # chunk_002, 0–60 s, small
    python run_eval.py --chunk chunk_005 --start 120 --dur 60 --model medium.en
"""

import argparse
import json
import os
import time

import numpy as np

from config import CFG, OUT_DIR
import data_io as io
import frontend as fe
import reconstruct as rc
import asr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', default=CFG.default_chunk)
    ap.add_argument('--start', type=float, default=CFG.default_start_s)
    ap.add_argument('--dur',   type=float, default=CFG.default_dur_s)
    ap.add_argument('--model', default='small', help='Whisper model name')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[warn] cuda unavailable → cpu'); args.device = 'cpu'

    cfg = CFG
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f'{args.chunk}_s{int(args.start)}_d{int(args.dur)}'
    t0 = time.time()

    # ── 1. front-end: bin window → AM-sideband mel ────────────────────────────
    cap = io.read_bin_window(cfg.bin_path(args.chunk), args.start, args.dur, cfg.cap_sr)
    print(f'[cap] {args.chunk} [{args.start:.0f}..{args.start+args.dur:.0f}]s '
          f'{len(cap):,} @ {cfg.cap_sr} Hz')
    cap_ds = fe.downsample(cap, cfg.cap_sr, cfg.aud_sr)
    f_mains = fe.detect_mains(cap_ds, cfg.aud_sr, cfg.mains_guess_hz, cfg.mains_search_hz)
    mel = fe.am_sideband_mel(cap_ds, f_mains, cfg)

    # ── 2. reconstruct (Griffin-Lim, no vocoder) → 16 kHz ─────────────────────
    recon = rc.mel_to_audio(mel, cfg, device=args.device)               # @ aud_sr
    recon16 = io._resample(recon, cfg.aud_sr, cfg.asr_sr)
    recon_wav = os.path.join(OUT_DIR, f'recon_{tag}.wav')
    rc.save_wav(recon_wav, recon16, cfg.asr_sr)

    # ── 3. reference (clean, native 16 kHz) ───────────────────────────────────
    ref16 = io.read_wav_window(cfg.wav_path(args.chunk), args.start, args.dur, cfg.asr_sr)
    ref_wav = os.path.join(OUT_DIR, f'reference_{tag}.wav')
    rc.save_wav(ref_wav, ref16, cfg.asr_sr)

    # ── 4. Whisper on both + WER ──────────────────────────────────────────────
    model = asr.load_whisper(args.model, device=args.device)
    print('[asr] transcribing reference …')
    ref_txt = asr.transcribe(model, ref16)['text'].strip()
    print('[asr] transcribing reconstruction …')
    rec_txt = asr.transcribe(model, recon16)['text'].strip()
    score = asr.wer(ref_txt, rec_txt)

    # ── 5. report ─────────────────────────────────────────────────────────────
    dt = time.time() - t0
    result = {
        'chunk': args.chunk, 'start_s': args.start, 'dur_s': args.dur,
        'whisper_model': args.model, 'f_mains_hz': round(f_mains, 3),
        'reference_transcript': ref_txt,
        'recon_transcript': rec_txt,
        **score, 'wall_s': round(dt, 1),
    }
    out_json = os.path.join(OUT_DIR, f'eval_{tag}.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    print()
    print('══ STEP 1 — intelligibility (WER) ════════════════')
    print(f'  window         = {args.chunk} [{args.start:.0f},{args.start+args.dur:.0f}]s')
    print(f'  whisper model  = {args.model}')
    print(f'  WER            = {score["wer"]*100:.1f}%   '
          f'(S={score["sub"]} D={score["del"]} I={score["ins"]} / {score["ref_words"]} ref words)')
    print(f'  wall time      = {dt:.1f}s')
    print('  ── REFERENCE (pseudo-truth) ──')
    print('   ', ref_txt[:600])
    print('  ── RECONSTRUCTION (powerline) ──')
    print('   ', rec_txt[:600])
    print(f'\n  json  → {out_json}')
    print(f'  wavs  → {recon_wav}\n         {ref_wav}')
    print('══════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
