#!/usr/bin/env python3
"""
train.py — joint flow-matching + CTC word-loss trainer for PLFW-17 (M17).

  total = flow_matching + lambda_ctc · CTC(word)
Eval: samples 513-bin log-STFTs (few-step ODE) → spec_r / env_r on held-out
chunks, and CTC-greedy-decodes the word head → WER vs true LibriSpeech text.

Startup runs dataset.audit_split — training refuses to start if any utt_id or
text crosses the train/test boundary.
"""
import argparse, copy, json, math, os, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG, OUT_DIR
import dataset as D
import models as M
import text as T


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    r = float((a * b).sum() / (a.norm() * b.norm() + 1e-8))
    return 0.0 if (r != r) else r      # guard NaN


class EMA:
    def __init__(self, model, decay):
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
    L1 = R = E = 0.0; nb = 0
    refs, hyps = [], []
    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        raw = b['raw'].to(device); tgt = b['spec'].to(device); vf = b['vframes']
        gen = model.sample(raw, steps=steps, cfg_scale=cfg_scale)   # fp32 (stable)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits = model.ctc_logits(raw)
        gen = gen.float() * std + mean
        for i in range(gen.shape[0]):
            v = int(vf[i])
            R += _pearson(gen[i, :, :v].flatten(), tgt[i, :, :v].flatten())
            E += _pearson(gen[i, :, :v].mean(0), tgt[i, :, :v].mean(0))
            L1 += float((gen[i, :, :v] - tgt[i, :, :v]).abs().mean())
            nb += 1
        hyps.extend(T.ctc_greedy_decode(logits.float().cpu().transpose(0, 1)))
        refs.extend(b['texts'])
    wer = T.corpus_wer(refs, hyps)
    return {'spec_l1': L1 / nb, 'spec_r': R / nb, 'env_r': E / nb,
            'wer': wer['wer'], 'n_words': wer['n_words']}, list(zip(refs[:3], hyps[:3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--lambda-ctc', type=float, default=CFG.lambda_ctc)
    ap.add_argument('--sample-steps', type=int, default=CFG.sample_steps)
    ap.add_argument('--cfg-scale', type=float, default=CFG.cfg_scale)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    rows = D.load_manifest()
    audit = D.audit_split(rows, CFG)          # raises on any train/test leakage
    print(f'[split-audit] {audit}')
    tr_rows, te_rows, _ = D.split_rows(rows, CFG)
    tr_ds = D.PLFW17Utterances(tr_rows, CFG); te_ds = D.PLFW17Utterances(te_rows, CFG)
    print(f'[data] train utts={len(tr_ds)}  test utts={len(te_ds)}  '
          f'a_len={CFG.a_len} ({CFG.cap_sr}Hz lossless patchify)  '
          f'spec=[{CFG.n_bins},{CFG.n_frames}]')
    print('[stats] estimating spec mean/std …')
    mean, std = D.estimate_spec_stats(tr_ds, n=384)
    print(f'[stats] mean={mean:.3f} std={std:.3f}')

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       collate_fn=D.collate, drop_last=True, pin_memory=(device == 'cuda'),
                       persistent_workers=(args.workers > 0))
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=D.collate)

    model = M.build(CFG).to(device)
    ema = EMA(model, args.ema)
    print(f'[model] PLFW-17 params={model.count_params():,}  lambda_ctc={args.lambda_ctc}')

    steps_per_epoch = len(tr_dl)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup)) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - args.warmup) /
                                           max(1, total_steps - args.warmup)))) * 0.9 + 0.1))
    scaler = torch.amp.GradScaler('cuda', enabled=False)  # bf16 path: no loss scaling needed

    last_path = os.path.join(args.out_dir, 'last.pt')
    log = {'args': vars(args), 'spec_mean': mean, 'spec_std': std, 'audit': audit,
           'params': model.count_params(), 'evals': []}
    best = math.inf; step = 0; start_ep = 0
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck['model']); ema.shadow.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler'])
        step = ck['step']; start_ep = ck.get('epoch', 0) + 1; best = ck.get('best', math.inf)
        mean, std = ck['spec_mean'], ck['spec_std']
        print(f'[resume] step={step} epoch={start_ep} best={best:.3f}')

    def save(path, ep, extra=None):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        d = {'model': model.state_dict(), 'ema': ema.shadow.state_dict(), 'opt': opt.state_dict(),
             'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
             'spec_mean': mean, 'spec_std': std, 'step': step, 'epoch': ep, 'best': best}
        if extra: d.update(extra)
        torch.save(d, path)

    t0 = time.time(); done = False
    for ep in range(start_ep, args.epochs):
        model.train()
        for b in tr_dl:
            raw = b['raw'].to(device, non_blocking=True)
            spec = (b['spec'].to(device, non_blocking=True) - mean) / std
            vf = b['vframes'].to(device); lab = b['labels'].to(device); ll = b['label_lengths'].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                flow, ctc = model.losses(raw, spec, vf, lab, ll)
                loss = flow + args.lambda_ctc * ctc
            if not torch.isfinite(loss):          # bf16 rarely NaNs; skip if it ever does
                opt.zero_grad(set_to_none=True); continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); sched.step(); ema.update(model)
            step += 1
            if step % 50 == 0:
                print(f'  ep{ep} step{step}/{total_steps} loss={loss.item():.3f} '
                      f'flow={flow.item():.3f} ctc={ctc.item():.3f} '
                      f'lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)', flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                m, samples = evaluate(ema.shadow, te_dl, device, mean, std, args.sample_steps, args.cfg_scale)
                print(f'[eval] step{step}  spec_r={m["spec_r"]:.3f} env_r={m["env_r"]:.3f} '
                      f'WER={m["wer"]*100:.1f}% (n={m["n_words"]})', flush=True)
                for r, h in samples:
                    print(f'    REF: {r[:70]}\n    HYP: {h[:70]}')
                log['evals'].append({'step': step, **m})
                if m['wer'] < best:
                    best = m['wer']; save(os.path.join(args.out_dir, 'best.pt'), ep, {'metrics': m})
                    print(f'    ↑ new best WER={best*100:.1f}% → saved')
                save(last_path, ep)
                json.dump(log, open(os.path.join(args.out_dir, 'train_log.json'), 'w'), indent=2)
                model.train()
            if step >= total_steps:
                done = True; break
        save(last_path, ep)
        if done:
            break
    save(last_path, args.epochs - 1)
    print(f'\n══ M17 PLFW-17 ══  params={model.count_params():,}  best WER={best*100:.1f}%  '
          f'wall={time.time()-t0:.0f}s → {args.out_dir}')


if __name__ == '__main__':
    main()
