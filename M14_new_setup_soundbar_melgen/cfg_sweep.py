#!/usr/bin/env python3
"""
cfg_sweep.py — sweep classifier-free-guidance scale to hear/see the
faithful↔realistic tradeoff.

For each cfg value:
  * generates + vocodes ONE fixed window (audio you can A/B),
  * renders its generated mel into a panel next to the target,
  * reports mel_r / env_r AVERAGED over several test windows (robust metric).

Usage:
  python cfg_sweep.py --ckpt outputs/cluster_a100/late_snap.pt --idx 2 \
      --cfgs 1.0 1.5 2.0 3.0 5.0 --n-metric 24 --out-dir outputs/cluster_a100/cfg_sweep
"""
import argparse, os
import numpy as np
import torch, torchaudio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from config import CFG
import dataset as D
import models as M
from mel_to_wav import mel_to_wav


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='outputs/cluster_a100/late_snap.pt')
    ap.add_argument('--idx', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cfgs', type=float, nargs='+', default=[1.0, 1.5, 2.0, 3.0, 5.0])
    ap.add_argument('--n-metric', type=int, default=24)
    ap.add_argument('--steps', type=int, default=32)
    ap.add_argument('--out-dir', default='outputs/cluster_a100/cfg_sweep')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    ck = torch.load(args.ckpt, map_location=device)
    mean, std = ck['mel_mean'], ck['mel_std']
    model = M.build(CFG).to(device); model.load_state_dict(ck.get('ema', ck['model'])); model.eval()

    _, te = D.split_chunks(CFG)
    ds = D.PLFWindows(D.build_index(te, CFG, seed=args.seed), CFG)

    # fixed window for audio + mel panel
    item = ds[args.idx % len(ds)]
    raw1, tgt1 = item['raw'], item['mel']
    tag = f'{item["chunk"]}_{item["start_s"]:.0f}s'

    # metric windows
    midx = np.random.RandomState(args.seed).choice(len(ds), size=args.n_metric, replace=False)
    mraw = torch.stack([ds[int(i)]['raw'] for i in midx])
    mtgt = torch.stack([ds[int(i)]['mel'] for i in midx])

    # target audio + mel once
    torchaudio.save(f'{args.out_dir}/{tag}_target.wav', mel_to_wav(tgt1, CFG).unsqueeze(0), CFG.ref_sr)

    ncol = len(args.cfgs) + 1
    fig, ax = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.4))
    vmin, vmax = float(tgt1.min()), float(tgt1.max())
    ax[0].imshow(tgt1.numpy(), origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    ax[0].set_title('target'); ax[0].set_xticks([]); ax[0].set_yticks([])

    print(f'ckpt step={ck.get("step")}  window={tag}  metric over {args.n_metric} test windows')
    print(f'{"cfg":>5} {"mel_r":>7} {"env_r":>7}   (avg over test windows)')
    rows = []
    for j, w in enumerate(args.cfgs):
        # audio + mel for the fixed window
        with torch.no_grad():
            g1 = model.sample(raw1[None].to(device), steps=args.steps, cfg_scale=w)
        g1 = (g1[0].float().cpu() * std + mean)
        torchaudio.save(f'{args.out_dir}/{tag}_cfg{w:.1f}.wav', mel_to_wav(g1, CFG).unsqueeze(0), CFG.ref_sr)
        ax[j + 1].imshow(g1.numpy(), origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
        r1 = _pearson(g1.flatten(), tgt1.flatten())
        ax[j + 1].set_title(f'cfg {w:.1f}\n(this win r={r1:.2f})'); ax[j + 1].set_xticks([]); ax[j + 1].set_yticks([])

        # averaged metric over metric windows (batched)
        mrs, ers = [], []
        for k in range(0, len(midx), 8):
            rb = mraw[k:k + 8].to(device)
            with torch.no_grad():
                gb = model.sample(rb, steps=args.steps, cfg_scale=w)
            gb = gb.float().cpu() * std + mean
            tb = mtgt[k:k + 8]
            for n in range(gb.shape[0]):
                mrs.append(_pearson(gb[n].flatten(), tb[n].flatten()))
                ers.append(_pearson(gb[n].mean(0), tb[n].mean(0)))
        mr, er = float(np.mean(mrs)), float(np.mean(ers))
        rows.append((w, mr, er))
        print(f'{w:>5.1f} {mr:>7.3f} {er:>7.3f}')

    fig.suptitle(f'CFG sweep — {tag}  (step {ck.get("step")})')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{args.out_dir}/cfg_sweep_mels.png', dpi=120)
    print(f'\n[out] mels  → {args.out_dir}/cfg_sweep_mels.png')
    print(f'[out] audio → {args.out_dir}/{tag}_target.wav + {tag}_cfg*.wav')


if __name__ == '__main__':
    main()
