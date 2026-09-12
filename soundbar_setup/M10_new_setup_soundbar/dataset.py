#!/usr/bin/env python3
"""
dataset.py — STEP 2: build the powerline→transcript training set, one chunk at a
time (SLURM-array friendly).

For each LibriSpeech utterance fully inside a reference chunk, cut the matching
lag-corrected window out of the powerline .bin, run the blind AM-sideband mel
front-end, and store the mel + transcript.  No reference signal and no neural
vocoder are used, so this is exactly the feature the fine-tuned model will see.

Outputs (per chunk, under outputs/dataset/):
    <chunk>.npz            float16 mels keyed by utt_id   (the features)
    <chunk>.manifest.jsonl one line/utterance: id, text, timing, shape, f_mains

Alignment: powerline lags audio by lag_ms (sidecar), so for a reference-utterance
at audio-time [t0, t1] we read powerline samples at [t0+lag, t1+lag].
"""

import argparse
import json
import os
import time

import numpy as np

from config import CFG, OUT_DIR
import data_io as io
import frontend as fe
import labels as lb

DATASET_DIR = os.path.join(OUT_DIR, 'dataset')


def build_chunk(chunk_n: int, cfg=CFG, save_audio: bool = False,
                max_utts: int | None = None) -> dict:
    chunk = f'chunk_{chunk_n:03d}'
    bin_path = cfg.bin_path(chunk)
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f'no powerline capture: {bin_path}')

    os.makedirs(DATASET_DIR, exist_ok=True)
    index = lb.build_index()
    trans = lb.load_transcripts()
    utts = lb.utterances_in_chunk(chunk_n, index, trans, full_only=True)
    if max_utts:
        utts = utts[:max_utts]

    lag_s = io.read_lag_ms(cfg.lag_path(chunk)) / 1000.0
    bin_dur = os.path.getsize(bin_path) / 4 / cfg.cap_sr   # real float32

    # detect mains once per chunk from a stable 60 s window
    probe = io.read_bin_window(bin_path, 0, min(60.0, bin_dur), cfg.cap_sr)
    probe_ds = fe.downsample(probe, cfg.cap_sr, cfg.aud_sr)
    f_mains = fe.detect_mains(probe_ds, cfg.aud_sr, cfg.mains_guess_hz, cfg.mains_search_hz)
    del probe, probe_ds

    feats, manifest = {}, []
    audio_dir = os.path.join(DATASET_DIR, f'{chunk}_audio') if save_audio else None
    if save_audio:
        os.makedirs(audio_dir, exist_ok=True)
        import reconstruct as rc

    t0 = time.time()
    kept = skipped = 0
    for k, u in enumerate(utts):
        b0 = u['start_s'] + lag_s
        b1 = u['end_s'] + lag_s
        if b1 > bin_dur:                       # window runs past capture end
            skipped += 1
            continue
        win = io.read_bin_window(bin_path, b0, b1 - b0, cfg.cap_sr)
        if len(win) < int((b1 - b0) * cfg.cap_sr) * 0.95:
            skipped += 1
            continue
        win_ds = fe.downsample(win, cfg.cap_sr, cfg.aud_sr)
        mel = fe.am_sideband_mel(win_ds, f_mains, cfg)          # [n_mels, T]
        feats[u['utt_id']] = mel.astype(np.float16)

        rec = {
            'utt_id': u['utt_id'], 'chunk': chunk, 'text': u['text'],
            'start_s': round(u['start_s'], 3), 'end_s': round(u['end_s'], 3),
            'dur_s': round(u['dur_s'], 3), 'lag_ms': round(lag_s * 1000, 1),
            'f_mains_hz': round(f_mains, 3),
            'n_mels': int(mel.shape[0]), 'n_frames': int(mel.shape[1]),
            'shard': f'{chunk}.npz',
        }
        if save_audio:
            wav = rc.mel_to_audio(mel.astype(np.float32), cfg)
            wav16 = io._resample(wav, cfg.aud_sr, cfg.asr_sr)
            ap = os.path.join(audio_dir, f'{u["utt_id"]}.wav')
            rc.save_wav(ap, wav16, cfg.asr_sr)
            rec['audio'] = os.path.relpath(ap, DATASET_DIR)
        manifest.append(rec)
        kept += 1
        if (k + 1) % 25 == 0:
            print(f'  [{chunk}] {k+1}/{len(utts)} utts  ({time.time()-t0:.0f}s)')

    np.savez_compressed(os.path.join(DATASET_DIR, f'{chunk}.npz'), **feats)
    mpath = os.path.join(DATASET_DIR, f'{chunk}.manifest.jsonl')
    with open(mpath, 'w') as f:
        for rec in manifest:
            f.write(json.dumps(rec) + '\n')

    dt = time.time() - t0
    print(f'[{chunk}] kept={kept} skipped={skipped}  f_mains={f_mains:.3f} Hz  '
          f'{dt:.0f}s → {chunk}.npz + manifest')
    return {'chunk': chunk, 'kept': kept, 'skipped': skipped, 'wall_s': round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--chunk-index', type=int, help='single chunk number, e.g. 2')
    g.add_argument('--chunks', help='inclusive range "1-46"')
    ap.add_argument('--save-audio', action='store_true',
                    help='also write Griffin-Lim 16 kHz wavs (for audio-based fine-tune)')
    ap.add_argument('--max-utts', type=int, default=None, help='cap utts (debug)')
    args = ap.parse_args()

    if args.chunk_index is not None:
        nums = [args.chunk_index]
    else:
        lo, hi = (int(x) for x in args.chunks.split('-'))
        nums = list(range(lo, hi + 1))

    results = []
    for n in nums:
        try:
            results.append(build_chunk(n, save_audio=args.save_audio,
                                       max_utts=args.max_utts))
        except FileNotFoundError as e:
            print(f'[skip] {e}')
    print('\n══ dataset build summary ══')
    for r in results:
        print(f'  {r["chunk"]}: kept={r["kept"]} skipped={r["skipped"]} '
              f'({r["wall_s"]}s)')


if __name__ == '__main__':
    main()
