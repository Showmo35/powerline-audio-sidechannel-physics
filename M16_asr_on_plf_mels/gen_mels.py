#!/usr/bin/env python3
"""
gen_mels.py — sample an M14 PowerLine-Flow mel for every manifest utterance.

Runs the frozen M14 checkpoint (EMA weights) over ALL 202 chunks: each utterance
window is tiled into M14's native 4 s / 400-frame windows (each tile read + lag-
aligned + RMS-normalised exactly like M14's dataset), sampled with the same CFG
ODE used for the step-73k eval, un-standardised back to log-mel and concatenated,
then cropped to the utterance's true frame count and saved as gen_mels/{utt}.npy
(float16 [80, T]).  Resumable: existing .npy files are skipped.

M14's modules are imported by putting its directory FIRST on sys.path — this
script must therefore never import M16's own config/dataset (names collide).
"""

import argparse, json, os, sys, time
import numpy as np
import torch

M14_DIR = '<REPO_ROOT>/M14_new_setup_soundbar_melgen'
M15_DIR = '<REPO_ROOT>/M15_soundbar_melgen_word'
sys.path.insert(0, M14_DIR)

from config import CFG as P          # M14 config
import data_io as io                 # M14 readers
import models as M                   # M14 PLF

sys.path.insert(0, M15_DIR)
import text as T                     # char vocab (utterance filter must match M16)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST   = os.path.join(M15_DIR, 'full_manifest.json')
MIN_DUR, MAX_DUR = 2.0, 16.0        # keep in sync with M16 config.py


def load_rows():
    rows = json.load(open(MANIFEST))
    return [r for r in rows
            if MIN_DUR <= r['dur_s'] <= MAX_DUR and len(T.encode(r['text'])) > 0]


def read_tile(row, k, lag_s):
    """k-th 4 s tile of the utterance's capture window, M14-style raw input."""
    start = row['start_s'] + lag_s + k * P.win_s
    x = io.read_bin_window(P.bin_path(row['chunk']), start, P.win_s, P.cap_sr, P.in_sr)
    x = np.asarray(x, dtype=np.float32)
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-8)
    out = np.zeros(P.in_len, dtype=np.float32)
    out[:min(len(x), P.in_len)] = x[:P.in_len]
    return torch.from_numpy(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(MODULE_DIR, 'm14_final_last.pt'))
    ap.add_argument('--out-dir', default=os.path.join(MODULE_DIR, 'gen_mels'))
    ap.add_argument('--steps', type=int, default=P.sample_steps)
    ap.add_argument('--cfg-scale', type=float, default=P.cfg_scale)
    ap.add_argument('--group', type=int, default=16, help='utterances per GPU batch')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=device)
    mean, std = ck['mel_mean'], ck['mel_std']
    model = M.build(P).to(device)
    model.load_state_dict(ck.get('ema', ck['model']))
    model.eval()
    print(f'[gen] ckpt step={ck.get("step")}  steps={args.steps} cfg={args.cfg_scale}')

    rows = [r for r in load_rows()
            if not os.path.exists(os.path.join(args.out_dir, r['utt_id'] + '.npy'))]
    print(f'[gen] {len(rows)} utterances to generate')
    json.dump({'ckpt': args.ckpt, 'ckpt_step': ck.get('step'),
               'sample_steps': args.steps, 'cfg_scale': args.cfg_scale,
               'seed': args.seed},
              open(os.path.join(args.out_dir, 'meta.json'), 'w'), indent=1)

    lag_cache = {}
    t0, done = time.time(), 0
    for g0 in range(0, len(rows), args.group):
        grp = rows[g0:g0 + args.group]
        tiles, owner = [], []                       # owner[i] = (row index in grp)
        for gi, r in enumerate(grp):
            ch = r['chunk']
            if ch not in lag_cache:
                lag_cache[ch] = io.read_lag_ms(P.lag_path(ch)) / 1000.0
            n_tiles = int(np.ceil(min(r['dur_s'], MAX_DUR) / P.win_s))
            for k in range(n_tiles):
                tiles.append(read_tile(r, k, lag_cache[ch]))
                owner.append(gi)
        raw = torch.stack(tiles).to(device)
        with torch.no_grad():
            gen = model.sample(raw, steps=args.steps, cfg_scale=args.cfg_scale)
        gen = (gen.float().cpu() * std + mean)      # un-standardised log-mel
        owner = np.asarray(owner)
        for gi, r in enumerate(grp):
            idx = np.nonzero(owner == gi)[0]
            m = torch.cat([gen[int(j)] for j in idx], dim=-1)      # [80, 400*n_tiles]
            vframes = min(int(round(min(r['dur_s'], MAX_DUR) * P.fps)), m.shape[-1])
            np.save(os.path.join(args.out_dir, r['utt_id'] + '.npy'),
                    m[:, :vframes].numpy().astype(np.float16))
        done += len(grp)
        if (g0 // args.group) % 20 == 0:
            rate = done / max(time.time() - t0, 1e-9)
            print(f'  {done}/{len(rows)} utts  ({rate:.1f} utt/s, '
                  f'eta {(len(rows) - done) / max(rate, 1e-9) / 60:.0f} min)', flush=True)
    print(f'[gen] done: {done} utterances in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
