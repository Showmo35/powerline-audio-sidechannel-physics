#!/usr/bin/env python3
"""
mel_to_wav.py — vocode PLF mels back to listenable audio (Griffin-Lim).

Samples a generated mel from the model for one held-out window and also takes the
matching target mel, then inverts BOTH to waveforms with a mel-pseudoinverse +
Griffin-Lim (no pretrained vocoder). Writes target.wav / generated.wav so they can
be compared by ear. Rough by design (no neural vocoder), but reflects the mel content.

Usage:
    python mel_to_wav.py --ckpt outputs/cluster_a100/best.pt --idx 0 --out-dir outputs/cluster_a100/audio
"""
import argparse, os
import numpy as np
import torch, torchaudio

from config import CFG
import dataset as D
import models as M


def mel_to_wav(logmel, cfg, n_iter=64):
    """log-mel [n_mels, T] → waveform (Griffin-Lim via mel pseudo-inverse)."""
    powmel = (logmel.exp() - cfg.log_eps).clamp(min=0)                 # power mel
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=cfg.n_fft // 2 + 1, f_min=cfg.fmin, f_max=cfg.fmax,
        n_mels=cfg.n_mels, sample_rate=cfg.ref_sr, norm=None, mel_scale='htk')  # [F, M]
    spec = (torch.linalg.pinv(fb.T) @ powmel).clamp(min=0)             # [F, T] power spec
    gl = torchaudio.transforms.GriffinLim(
        n_fft=cfg.n_fft, n_iter=n_iter, win_length=cfg.win_length,
        hop_length=cfg.hop, power=2.0)
    wav = gl(spec)
    return wav / (wav.abs().max() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='outputs/cluster_a100/best.pt')
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--n', type=int, default=1,
                    help='if >1, vocode the same N windows viz.py picks (RandomState(seed).choice)')
    ap.add_argument('--steps', type=int, default=CFG.sample_steps)
    ap.add_argument('--cfg-scale', type=float, default=CFG.cfg_scale)
    ap.add_argument('--out-dir', default='outputs/cluster_a100/audio')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    ck = torch.load(args.ckpt, map_location=device)
    mean, std = ck['mel_mean'], ck['mel_std']
    model = M.build(CFG).to(device); model.load_state_dict(ck.get('ema', ck['model'])); model.eval()

    _, te_chunks = D.split_chunks(CFG)
    ds = D.PLFWindows(D.build_index(te_chunks, CFG, seed=args.seed), CFG)
    # match viz.py's exact window selection when --n>1 (same seed + RandomState.choice)
    if args.n > 1:
        picks = np.random.RandomState(args.seed).choice(len(ds), size=args.n, replace=False)
    else:
        picks = [args.idx % len(ds)]

    for pi in picks:
        item = ds[int(pi)]
        raw, tgt = item['raw'], item['mel']
        with torch.no_grad():
            gen = model.sample(raw[None].to(device), steps=args.steps, cfg_scale=args.cfg_scale)
        gen = (gen[0].float().cpu() * std + mean)
        tw = mel_to_wav(tgt, CFG); gw = mel_to_wav(gen, CFG)
        tag = f'{item["chunk"]}_{item["start_s"]:.0f}s'
        tp = os.path.join(args.out_dir, f'{tag}_target.wav')
        gp = os.path.join(args.out_dir, f'{tag}_generated.wav')
        torchaudio.save(tp, tw.unsqueeze(0), CFG.ref_sr)
        torchaudio.save(gp, gw.unsqueeze(0), CFG.ref_sr)
        print(f'[wav] step={ck.get("step")}  window={tag}  → target+generated')


if __name__ == '__main__':
    main()
