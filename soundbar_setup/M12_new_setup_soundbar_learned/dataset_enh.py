#!/usr/bin/env python3
"""
dataset_enh.py — build (powerline stack → clean mel) pairs, one chunk per task.

For each utterance in the shared manifest, cut the lag-corrected powerline window
and the matching clean window, compute the per-harmonic mel STACK (log) and the
clean log-mel TARGET, and store both (float16) keyed by utt_id.

Outputs under outputs/dataset_enh/:
    chunk_XXX.x.npz   utt_id → log stack  [2*n_harmonics, n_mels, T]
    chunk_XXX.y.npz   utt_id → clean log-mel [n_mels, T]
"""

import argparse
import json
import os
import time

import numpy as np

from config import CFG, FEAT_DIR, SHARED_MANIFEST
import data_io as io
import frontend_stack as fs

LOG_FLOOR = 1e-5


def _utts_for_chunk(chunk):
    rows = [json.loads(l) for l in open(SHARED_MANIFEST) if l.strip()]
    return [r for r in rows if r['chunk'] == chunk]


def build_chunk(chunk_n, cfg=CFG, max_utts=None):
    chunk = f'chunk_{chunk_n:03d}'
    bin_path = cfg.bin_path(chunk)
    if not os.path.exists(bin_path):
        raise FileNotFoundError(bin_path)
    os.makedirs(FEAT_DIR, exist_ok=True)

    utts = _utts_for_chunk(chunk)
    if max_utts:
        utts = utts[:max_utts]
    lag_s = io.read_lag_ms(cfg.lag_path(chunk)) / 1000.0
    bin_dur = os.path.getsize(bin_path) / 4 / cfg.cap_sr

    probe = io.read_bin_window(bin_path, 0, min(60.0, bin_dur), cfg.cap_sr)
    f_mains = fs.detect_mains(fs.downsample(probe, cfg.cap_sr, cfg.aud_sr),
                              cfg.aud_sr, cfg.mains_guess_hz, cfg.mains_search_hz)
    del probe

    X, Yt = {}, {}
    t0 = time.time(); kept = skipped = 0
    for k, u in enumerate(utts):
        dur = u['end_s'] - u['start_s']
        if u['end_s'] + lag_s > bin_dur:
            skipped += 1; continue
        cap = io.read_bin_window(bin_path, u['start_s'] + lag_s, dur, cfg.cap_sr)
        if len(cap) < int(dur * cfg.cap_sr) * 0.95:
            skipped += 1; continue
        stack = fs.am_sideband_stack(fs.downsample(cap, cfg.cap_sr, cfg.aud_sr),
                                     f_mains, cfg)                 # [C, M, T]
        a22 = io.read_wav_window(cfg.wav_path(chunk), u['start_s'], dur, cfg.aud_sr)
        y = fs.clean_logmel(a22, cfg)                             # [M, T]
        T = min(stack.shape[2], y.shape[1])
        X[u['utt_id']] = np.log(np.maximum(stack[:, :, :T], LOG_FLOOR)).astype(np.float16)
        Yt[u['utt_id']] = y[:, :T].astype(np.float16)
        kept += 1
        if (k + 1) % 25 == 0:
            print(f'  [{chunk}] {k+1}/{len(utts)} ({time.time()-t0:.0f}s)')

    np.savez_compressed(os.path.join(FEAT_DIR, f'{chunk}.x.npz'), **X)
    np.savez_compressed(os.path.join(FEAT_DIR, f'{chunk}.y.npz'), **Yt)
    print(f'[{chunk}] kept={kept} skipped={skipped} f_mains={f_mains:.3f} '
          f'{time.time()-t0:.0f}s')
    return {'chunk': chunk, 'kept': kept, 'skipped': skipped}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--chunk-index', type=int)
    g.add_argument('--chunks', help='range "1-46"')
    ap.add_argument('--max-utts', type=int, default=None)
    args = ap.parse_args()
    nums = ([args.chunk_index] if args.chunk_index is not None
            else list(range(*(lambda a, b: (a, b + 1))(*map(int, args.chunks.split('-'))))))
    for n in nums:
        try:
            build_chunk(n, max_utts=args.max_utts)
        except FileNotFoundError as e:
            print(f'[skip] {e}')


if __name__ == '__main__':
    main()
