#!/usr/bin/env python3
"""
train.py — flow-matching trainer for PowerLine-Flow (PLF).

  raw 0-16 kHz powerline window  ─►  PLF (cond. rectified-flow DiT)  ─►  log-mel
  loss = || v_theta(x_t,t,c) - (x1-x0) ||^2        (rectified flow)

Eval samples mels with the few-step ODE (CFG) on held-out chunks and reports
  mel L1 (log-mel) · mel Pearson r · envelope Pearson r (per-frame energy).
An EMA copy of the weights is kept for evaluation / sampling (standard for
generative models).  Best checkpoint = lowest eval mel L1.

Usage:
    python train.py --epochs 60 --batch 32 --lr 3e-4
"""

import argparse, copy, json, math, os, time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG, OUT_DIR
import dataset as D
import models as M


# ── metrics ──────────────────────────────────────────────────────────────────
def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def eval_metrics(pred, tgt):
    """pred/tgt: (B, n_mels, T) log-mel (de-standardised)."""
    l1 = float((pred - tgt).abs().mean())
    rs, es = [], []
    for i in range(pred.shape[0]):
        rs.append(_pearson(pred[i].flatten(), tgt[i].flatten()))
        pe = pred[i].mean(0); te = tgt[i].mean(0)          # per-frame energy envelope
        es.append(_pearson(pe, te))
    return l1, float(np.mean(rs)), float(np.mean(es))


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


@torch.no_grad()
def evaluate(model, loader, device, mean, std, steps, cfg_scale, max_batches=3):
    model.eval()
    L1, R, E, n = 0.0, 0.0, 0.0, 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        raw = batch['raw'].to(device)
        tgt = batch['mel'].to(device)                       # raw log-mel
        with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            gen = model.sample(raw, steps=steps, cfg_scale=cfg_scale)
        gen = gen.float() * std + mean                       # de-standardise
        l1, r, e = eval_metrics(gen, tgt)
        L1 += l1; R += r; E += e; n += 1
    return {'mel_l1': L1 / n, 'mel_r': R / n, 'env_r': E / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--sample-steps', type=int, default=CFG.sample_steps)
    ap.add_argument('--cfg-scale', type=float, default=CFG.cfg_scale)
    ap.add_argument('--windows-per-chunk', type=int, default=CFG.windows_per_chunk)
    ap.add_argument('--resume', action='store_true', help='resume from outputs/last.pt')
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    CFG.windows_per_chunk = args.windows_per_chunk
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    # ── data ──
    tr_chunks, te_chunks = D.split_chunks(CFG)
    te_ds = D.PLFWindows(D.build_index(te_chunks, CFG, seed=2), CFG)
    stat_ds = D.PLFWindows(D.build_index(tr_chunks, CFG, seed=1), CFG)
    print(f'[data] train chunks={len(tr_chunks)} windows/epoch={len(stat_ds)} | '
          f'test chunks={len(te_chunks)} windows={len(te_ds)}')
    print(f'[data] in_len={CFG.in_len} ({CFG.in_sr}Hz, 0-{CFG.in_sr//2}Hz)  '
          f'mel=[{CFG.n_mels},{CFG.n_frames}] @ {CFG.fps:.0f}fps  '
          f'(windows RE-SAMPLED each epoch → full coverage)')

    print('[stats] estimating log-mel mean/std …')
    mean, std = D.estimate_mel_stats(stat_ds, n=512, seed=0)
    print(f'[stats] mel mean={mean:.3f} std={std:.3f}')

    def make_tr_loader(epoch):
        # fresh random windows every epoch so training eventually sees all data
        idx = D.build_index(tr_chunks, CFG, seed=1000 + epoch)
        ds = D.PLFWindows(idx, CFG)
        return DataLoader(ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=D.collate,
                          drop_last=True, pin_memory=(device == 'cuda'),
                          persistent_workers=False)

    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                       num_workers=4, collate_fn=D.collate)

    # ── model ──
    model = M.build(CFG).to(device)
    ema = EMA(model, args.ema)
    print(f'[model] PLF  params={model.count_params():,}  '
          f'(enc {sum(p.numel() for p in model.encoder.parameters()):,})')

    steps_per_epoch = max(1, len(stat_ds) // args.batch)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup)) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - args.warmup) /
                                           max(1, total_steps - args.warmup))))
         * 0.9 + 0.1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    # ── resume from last.pt if present ──
    last_path = os.path.join(args.out_dir, 'last.pt')
    log = {'args': vars(args), 'mel_mean': mean, 'mel_std': std,
           'params': model.count_params(), 'evals': []}
    best = math.inf; step = 0; start_ep = 0
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck['model']); ema.shadow.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        step = ck['step']; start_ep = ck.get('epoch', 0) + 1
        best = ck.get('best', math.inf); mean = ck['mel_mean']; std = ck['mel_std']
        print(f'[resume] from {last_path}: step={step} epoch={start_ep} best={best:.3f}')

    def save_ckpt(path, epoch, extra=None):
        d = {'model': model.state_dict(), 'ema': ema.shadow.state_dict(),
             'opt': opt.state_dict(), 'sched': sched.state_dict(),
             'scaler': scaler.state_dict(), 'mel_mean': mean, 'mel_std': std,
             'step': step, 'epoch': epoch, 'best': best}
        if extra:
            d.update(extra)
        torch.save(d, path)

    t0 = time.time(); done = False
    for ep in range(start_ep, args.epochs):
        tr_dl = make_tr_loader(ep)
        model.train()
        for batch in tr_dl:
            raw = batch['raw'].to(device, non_blocking=True)
            mel = (batch['mel'].to(device, non_blocking=True) - mean) / std
            with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
                loss = model.flow_loss(raw, mel)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            ema.update(model)
            step += 1

            if step % 50 == 0:
                print(f'  ep{ep} step{step}/{total_steps} loss={loss.item():.4f} '
                      f'lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)')
            if step % args.eval_every == 0 or step == total_steps:
                m = evaluate(ema.shadow, te_dl, device, mean, std,
                             args.sample_steps, args.cfg_scale)
                print(f'[eval] step{step}  mel_L1={m["mel_l1"]:.3f}  '
                      f'mel_r={m["mel_r"]:.3f}  env_r={m["env_r"]:.3f}')
                log['evals'].append({'step': step, **m})
                if m['mel_l1'] < best:
                    best = m['mel_l1']
                    save_ckpt(os.path.join(args.out_dir, 'best.pt'), ep, {'metrics': m})
                    print(f'    ↑ new best mel_L1={best:.3f} → saved')
                save_ckpt(last_path, ep)                 # resume point
                with open(os.path.join(args.out_dir, 'train_log.json'), 'w') as f:
                    json.dump(log, f, indent=2)
                model.train()
            if step >= total_steps:
                done = True; break
        save_ckpt(last_path, ep)                          # checkpoint every epoch
        if done:
            break

    save_ckpt(last_path, args.epochs - 1)
    log['best_mel_l1'] = best
    log['wall_s'] = round(time.time() - t0, 1)
    with open(os.path.join(args.out_dir, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print('\n══ M14 PowerLine-Flow result ═════════════════════')
    print(f'  params        = {model.count_params():,}')
    print(f'  train / test  = {len(stat_ds)}/epoch / {len(te_ds)} windows')
    print(f'  BEST mel L1   = {best:.3f} (log-mel)  on held-out chunks')
    print(f'  wall          = {log["wall_s"]}s   → {args.out_dir}/best.pt')
    print('══════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
