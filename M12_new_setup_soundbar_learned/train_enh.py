#!/usr/bin/env python3
"""
train_enh.py — train the learned front-end: powerline stack → clean log-mel.

This is BOTH the model and the decisive probe.  We compare, on held-out chunks,
how well the LEARNED mel and the FIXED 8-harmonic sum match the clean reference
mel — overall, on speech frames, and in the formant band.  If the learned mel
clears the fixed baseline (especially formant-band r on speech frames), phonetic
content is recoverable and ASR is worth retrying; if it can't, the information
is not in the capture.

Usage:
    python train_enh.py --test-chunks 41-46 --epochs 40 --batch 32 --lr 3e-4
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import CFG, OUT_DIR, FEAT_DIR, SHARED_MANIFEST
from model_unet import MelUNet

FORMANT = slice(8, 50)        # ~250 Hz–3 kHz mel bins


# ── data ────────────────────────────────────────────────────────────────────
def load_rows():
    return [json.loads(l) for l in open(SHARED_MANIFEST) if l.strip()]


class PairDataset(Dataset):
    def __init__(self, rows, crop, train=True):
        self.rows, self.crop, self.train = rows, crop, train
        self._x, self._y = {}, {}

    def __len__(self):
        return len(self.rows)

    def _shard(self, cache, chunk, kind):
        if chunk not in cache:
            cache[chunk] = np.load(os.path.join(FEAT_DIR, f'{chunk}.{kind}.npz'))
        return cache[chunk]

    def __getitem__(self, i):
        r = self.rows[i]; uid, chunk = r['utt_id'], r['chunk']
        x = self._shard(self._x, chunk, 'x')[uid].astype(np.float32)   # [C,M,T]
        y = self._shard(self._y, chunk, 'y')[uid].astype(np.float32)   # [M,T]
        T = x.shape[2]; c = self.crop
        if T >= c:
            s = random.randint(0, T - c) if self.train else (T - c) // 2
            x, y = x[:, :, s:s + c], y[:, s:s + c]
        else:
            xf = x.min(); yf = y.min()
            x = np.pad(x, ((0, 0), (0, 0), (0, c - T)), constant_values=xf)
            y = np.pad(y, ((0, 0), (0, c - T)), constant_values=yf)
        x = (x - x.mean()) / (x.std() + 1e-6)                          # per-utt z-norm
        return torch.from_numpy(x), torch.from_numpy(y)


# ── probe metrics ─────────────────────────────────────────────────────────────
def _r(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def probe(pred, clean, x_logstack):
    """pred,clean: [M,T] log-mel. x_logstack: [C,M,T] (for fixed baseline)."""
    fixed = np.log(np.maximum(np.exp(x_logstack).mean(0), 1e-5))       # old front-end
    eng = clean.mean(0)
    sp = eng > np.median(eng)                                          # speech frames
    out = {}
    for name, P in (('learned', pred), ('fixed', fixed)):
        out[name] = {
            'r_all': _r(P, clean),
            'r_speech': _r(P[:, sp], clean[:, sp]) if sp.any() else 0.0,
            'r_formant_sp': _r(P[FORMANT][:, sp], clean[FORMANT][:, sp]) if sp.any() else 0.0,
            'l1': float(np.abs(P - clean).mean()),
        }
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    agg = {}
    n = 0
    for x, y in loader:
        xd = x.to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):
            p = model(xd).float().cpu().numpy()
        xn = x.numpy()
        for b in range(x.shape[0]):
            m = probe(p[b], y[b].numpy(), xn[b])
            for k in ('learned', 'fixed'):
                for kk, vv in m[k].items():
                    agg[(k, kk)] = agg.get((k, kk), 0.0) + vv
            n += 1
    return {f'{k}_{kk}': v / max(n, 1) for (k, kk), v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-chunks', default='41-46')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--eval-every', type=int, default=2, help='epochs')
    ap.add_argument('--out-dir', default=os.path.join(OUT_DIR, 'enh_unet'))
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    lo, hi = (int(v) for v in args.test_chunks.split('-'))
    test_chunks = {f'chunk_{n:03d}' for n in range(lo, hi + 1)}
    rows = load_rows()
    tr = [r for r in rows if r['chunk'] not in test_chunks]
    te = [r for r in rows if r['chunk'] in test_chunks]
    print(f'[data] train={len(tr)} test={len(te)} crop={CFG.crop_frames}')

    tr_dl = DataLoader(PairDataset(tr, CFG.crop_frames, True), batch_size=args.batch,
                       shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    te_dl = DataLoader(PairDataset(te, CFG.crop_frames, False), batch_size=args.batch,
                       shuffle=False, num_workers=args.workers)

    model = MelUNet(in_ch=2 * CFG.n_harmonics, base=CFG.base_ch).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'[model] MelUNet  {nparam/1e6:.2f}M params  in_ch={2*CFG.n_harmonics}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(tr_dl))
    scaler = torch.cuda.amp.GradScaler(enabled=device == 'cuda')
    lossf = torch.nn.L1Loss()

    log = {'args': vars(args), 'evals': []}
    best = -1.0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); run = 0.0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            run += loss.item()
        run /= len(tr_dl)

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = evaluate(model, te_dl, device)
            print(f'[ep{ep+1}] loss={run:.3f}  '
                  f'learned r_formant_sp={m["learned_r_formant_sp"]:.3f} '
                  f'(fixed {m["fixed_r_formant_sp"]:.3f})  '
                  f'learned r_speech={m["learned_r_speech"]:.3f} '
                  f'(fixed {m["fixed_r_speech"]:.3f})  L1={m["learned_l1"]:.3f} '
                  f'({time.time()-t0:.0f}s)')
            log['evals'].append({'epoch': ep + 1, 'train_loss': run, **m})
            if m['learned_r_formant_sp'] > best:
                best = m['learned_r_formant_sp']
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'best.pt'))
        else:
            print(f'[ep{ep+1}] loss={run:.3f} ({time.time()-t0:.0f}s)')

    log['best_learned_r_formant_sp'] = best
    log['wall_s'] = round(time.time() - t0, 1)
    with open(os.path.join(args.out_dir, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    final = log['evals'][-1]
    print('\n══ STEP 5 — learned front-end (probe) ════════════')
    print(f'  params           = {nparam/1e6:.2f}M    train/test = {len(tr)}/{len(te)}')
    print('  formant-band r on speech frames (the key metric):')
    print(f'     LEARNED  = {final["learned_r_formant_sp"]:.3f}   (best {best:.3f})')
    print(f'     FIXED    = {final["fixed_r_formant_sp"]:.3f}   (8-harmonic sum)')
    print(f'  speech-frame r : learned {final["learned_r_speech"]:.3f}  '
          f'fixed {final["fixed_r_speech"]:.3f}')
    print(f'  overall r      : learned {final["learned_r_all"]:.3f}  '
          f'fixed {final["fixed_r_all"]:.3f}')
    print('  → learned >> fixed in formant band ⇒ phonetic info recoverable; retry ASR.')
    print('══════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
