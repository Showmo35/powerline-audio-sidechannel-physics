#!/usr/bin/env python3
"""
dataset_gan.py — build powerline 16 kHz mel STACKS, one chunk per array task.
Clean waveforms are read on the fly in training (from the 16 kHz reference wavs),
so only the (expensive) powerline stacks are precomputed here.

Output: outputs/dataset_gan/chunk_XXX.x.npz   utt_id → log stack [2*nh, n_mels, T]
"""

import argparse, json, os, time
import numpy as np
from config import CFG, FEAT_DIR, SHARED_MANIFEST
import data_io as io
import frontend_stack16 as fs


def build_chunk(chunk_n, cfg=CFG, max_utts=None):
    chunk = f'chunk_{chunk_n:03d}'
    bp = cfg.bin_path(chunk)
    if not os.path.exists(bp):
        raise FileNotFoundError(bp)
    os.makedirs(FEAT_DIR, exist_ok=True)
    rows = [json.loads(l) for l in open(SHARED_MANIFEST) if l.strip()]
    utts = [r for r in rows if r['chunk'] == chunk]
    if max_utts:
        utts = utts[:max_utts]
    lag_s = io.read_lag_ms(cfg.lag_path(chunk)) / 1000.0
    bin_dur = os.path.getsize(bp) / 4 / cfg.cap_sr
    probe = io.read_bin_window(bp, 0, min(60.0, bin_dur), cfg.cap_sr)
    fm = fs.detect_mains(fs.downsample(probe, cfg.cap_sr, cfg.sr), cfg.sr,
                         cfg.mains_guess_hz, cfg.mains_search_hz)
    del probe
    X = {}; t0 = time.time(); kept = skip = 0
    for k, u in enumerate(utts):
        dur = u['end_s'] - u['start_s']
        if u['end_s'] + lag_s > bin_dur:
            skip += 1; continue
        cap = io.read_bin_window(bp, u['start_s'] + lag_s, dur, cfg.cap_sr)
        if len(cap) < int(dur * cfg.cap_sr) * 0.95:
            skip += 1; continue
        st = fs.am_sideband_stack16(fs.downsample(cap, cfg.cap_sr, cfg.sr), fm, cfg)
        X[u['utt_id']] = st.astype(np.float16)
        kept += 1
        if (k + 1) % 25 == 0:
            print(f'  [{chunk}] {k+1}/{len(utts)} ({time.time()-t0:.0f}s)')
    np.savez_compressed(os.path.join(FEAT_DIR, f'{chunk}.x.npz'), **X)
    print(f'[{chunk}] kept={kept} skip={skip} f_mains={fm:.3f} {time.time()-t0:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--chunk-index', type=int)
    g.add_argument('--chunks')
    ap.add_argument('--max-utts', type=int, default=None)
    a = ap.parse_args()
    nums = ([a.chunk_index] if a.chunk_index is not None
            else list(range(*(lambda x, y: (x, y + 1))(*map(int, a.chunks.split('-'))))))
    for n in nums:
        try:
            build_chunk(n, max_utts=a.max_utts)
        except FileNotFoundError as e:
            print(f'[skip] {e}')


if __name__ == '__main__':
    main()
