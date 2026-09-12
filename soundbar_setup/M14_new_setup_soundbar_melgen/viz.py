#!/usr/bin/env python3
"""
viz.py — visualise PowerLine-Flow on random held-out windows.

For each sampled test window it shows four panels:
  1. input  : log-spectrogram of the raw 0-16 kHz powerline window (what the model sees)
  2. target : the true reference log-mel
  3. PLF gen: the mel sampled by the flow model (few-step ODE, CFG)
  4. envelope overlay: per-frame energy (mean over mel bins) target vs generated
Writes a PNG grid.

Usage:
    python viz.py --ckpt outputs/best.pt --n 6 --out outputs/plf_samples.png
"""

import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import CFG
import dataset as D
import models as M


def _spec(raw, n_fft=512, hop=160):
    x = torch.from_numpy(np.asarray(raw, dtype=np.float32))
    S = torch.stft(x, n_fft=n_fft, hop_length=hop, return_complex=True,
                   window=torch.hann_window(n_fft))
    return torch.log(S.abs() + 1e-5).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='outputs/best.pt')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--out', default='outputs/plf_samples.png')
    ap.add_argument('--steps', type=int, default=CFG.sample_steps)
    ap.add_argument('--cfg-scale', type=float, default=CFG.cfg_scale)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location=device)
    mean, std = ck['mel_mean'], ck['mel_std']
    model = M.build(CFG).to(device)
    model.load_state_dict(ck.get('ema', ck['model']))
    model.eval()
    print(f'[viz] loaded {args.ckpt}  step={ck.get("step")}  '
          f'metrics={ck.get("metrics")}')

    _, te_chunks = D.split_chunks(CFG)
    te_idx = D.build_index(te_chunks, CFG, seed=args.seed)
    ds = D.PLFWindows(te_idx, CFG)
    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(ds), size=args.n, replace=False)

    fig, axes = plt.subplots(args.n, 4, figsize=(20, 3.0 * args.n))
    if args.n == 1:
        axes = axes[None, :]
    for r, pi in enumerate(picks):
        item = ds[int(pi)]
        raw = item['raw']; tgt = item['mel'].numpy()
        with torch.no_grad():
            gen = model.sample(raw[None].to(device), steps=args.steps,
                               cfg_scale=args.cfg_scale)
        gen = (gen[0].float().cpu() * std + mean).numpy()

        sp = _spec(raw.numpy())
        vmin = min(tgt.min(), gen.min()); vmax = max(tgt.max(), gen.max())
        axes[r, 0].imshow(sp, origin='lower', aspect='auto', cmap='magma')
        axes[r, 0].set_ylabel(f'{item["chunk"]}\n@{item["start_s"]:.0f}s', fontsize=8)
        axes[r, 0].set_title('input: powerline 0-16kHz log-STFT' if r == 0 else '', fontsize=10)
        axes[r, 1].imshow(tgt, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
        axes[r, 1].set_title('target mel' if r == 0 else '', fontsize=10)
        axes[r, 2].imshow(gen, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
        axes[r, 2].set_title('PLF generated mel' if r == 0 else '', fontsize=10)
        te = tgt.mean(0); ge = gen.mean(0)
        ar = np.corrcoef(tgt.flatten(), gen.flatten())[0, 1]
        er = np.corrcoef(te, ge)[0, 1]
        axes[r, 3].plot(te, label='target env', lw=1)
        axes[r, 3].plot(ge, label='gen env', lw=1)
        axes[r, 3].set_title(f'envelope  mel_r={ar:.2f} env_r={er:.2f}' if r == 0
                             else f'mel_r={ar:.2f} env_r={er:.2f}', fontsize=9)
        axes[r, 3].legend(fontsize=7)
        for ci in range(3):
            axes[r, ci].set_xticks([]); axes[r, ci].set_yticks([])

    fig.suptitle('M14 PowerLine-Flow — powerline capture → generated audio mel-spectrogram',
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f'[viz] wrote {args.out}')


if __name__ == '__main__':
    main()
