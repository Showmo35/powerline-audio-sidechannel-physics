#!/usr/bin/env python3
"""
train_retrieval.py — M22: train a powerline→mel generator FRESH from scratch over
the FULL open vocabulary, so its generated word-mels are RETRIEVAL-OPTIMAL.

Two-phase objective (fresh init → generations are noise at first):
  phase 1 (step < retr_start):   flow matching only  — learn to generate mels.
  phase 2 (ramped in after):     + retrieval loss, all in the kNN space
                                   (time-norm → flatten → mean-center → L2-norm,
                                    cosine == Pearson r):
       L = flow_matching(raw, real 4s mel)
         + w(step) * [ lambda_con * InfoNCE(gen → real, labels)
                     + lambda_corr * (1 - corr(gen_i, its own real)) ]
  where w ramps 0→1 over retr_ramp steps, and the generated word mel comes from a
  few-step DIFFERENTIABLE ODE rollout (gradients flow through sampling).

Open vocabulary: labels span all ~25k word types; InfoNCE is effectively
instance-level (a generated word-mel must retrieve its OWN real mel against the
batch), with same-word occurrences acting as extra positives when they collide.

Eval = the real metric: kNN top-1/top-5 over a real-mel gallery covering every word
(capped per word), with generated test-mel queries (full 32-step sampler).

No M14 checkpoint: weights are random-initialized and the log-mel mean/std are
estimated from the data.  Env tf_gpu; run via SLURM A100.

Usage:  python train_retrieval.py --steps 30000 --batch 16
"""

import argparse, copy, json, math, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG, RC, OUT_DIR
import models as M
import dataset_words as D


# ══════════════════════════════════════════════════════════════════════════════
# retrieval embedding + losses  (all in the kNN space)
# ══════════════════════════════════════════════════════════════════════════════
def embed(mel_bt):
    """(B, n_mels, T) -> (B, n_mels*T) mean-centered, L2-normalized.
    Cosine between two such embeddings == Pearson r of the mels."""
    z = mel_bt.reshape(mel_bt.shape[0], -1)
    z = z - z.mean(dim=1, keepdim=True)
    return F.normalize(z, dim=1)


def gen_word_mel(gen_full, frs):
    """gen_full (B, n_mels, n_frames) -> (B, n_mels, T) word mels (slice [:, :frs] then
    time-normalize; the word sits at the window start)."""
    outs = []
    for i in range(gen_full.shape[0]):
        w = gen_full[i:i + 1, :, :int(frs[i])]
        w = F.interpolate(w[None], size=(CFG.n_mels, RC.T),
                          mode='bilinear', align_corners=False)[0]
        outs.append(w)
    return torch.cat(outs, dim=0)


def supcon_loss(e_gen, e_real, labels, tau):
    """Supervised InfoNCE: gen anchors, real keys. Each anchor's own occurrence real
    is a positive; same-word reals in the batch are extra positives."""
    sim = (e_gen @ e_real.t()) / tau
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    log_prob = sim - torch.log(torch.exp(sim).sum(dim=1, keepdim=True) + 1e-9)
    pos = (labels[:, None] == labels[None, :]).float()
    return -((pos * log_prob).sum(1) / pos.sum(1).clamp(min=1)).mean()


def diff_sample(model, raw, steps):
    """Few-step Euler rollout of dx/dt=v(x,t,c) WITH gradients (no CFG). Standardized."""
    c = model.encode(raw)
    x = torch.randn(raw.shape[0], CFG.n_mels, CFG.n_frames, device=raw.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((raw.shape[0],), i * dt, device=raw.device)
        x = x + dt * model.velocity(x, t, c)
    return x


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


# ══════════════════════════════════════════════════════════════════════════════
# open-vocabulary kNN retrieval eval
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def build_gallery(loader, device):
    """Gallery = embeddings of REAL train word mels (model-independent, capped/word)."""
    embs, ys = [], []
    for b in loader:
        embs.append(embed(b['mel_word'].to(device)).half())
        ys.append(b['label'].to(device))
    return torch.cat(embs), torch.cat(ys)                       # (G,D) fp16, (G,)


@torch.no_grad()
def knn_eval(model, gallery_emb, gallery_y, test_loader, device, mean, std, use_amp):
    model.eval()
    k = min(5, gallery_emb.shape[0])
    top1 = top5 = n = 0
    for b in test_loader:
        raw = b['raw'].to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            gen = model.sample(raw, steps=RC.eval_steps, cfg_scale=RC.cfg_scale_eval)
        gen = gen.float() * std + mean
        q = embed(gen_word_mel(gen, b['frs'])).half()           # (b, D) fp16
        sims = q @ gallery_emb.t()                              # (b, G)
        top = sims.topk(k, dim=1).indices
        lab = gallery_y[top]
        y = b['label'].to(device)
        top1 += (lab[:, 0] == y).sum().item()
        top5 += (lab == y[:, None]).any(1).sum().item()
        n += len(y)
    return top1 / max(1, n), top5 / max(1, n), n


def retr_weight(step):
    if step < RC.retr_start:
        return 0.0
    return min(1.0, (step - RC.retr_start) / max(1, RC.retr_ramp))


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)          # fresh training
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--train-steps', type=int, default=RC.train_steps)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    use_amp = (device == 'cuda')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── data (all word occurrences, open vocab) ──
    tr_items, words = D.build_items('train')
    te_items, _     = D.build_items('test')
    rng = np.random.RandomState(0)
    if len(te_items) > RC.eval_queries:                        # bound eval cost
        te_items = [te_items[i] for i in rng.choice(len(te_items), RC.eval_queries, replace=False)]
    gal_items = D.cap_per_label(tr_items, RC.gallery_per_word)
    print(f'[data] OPEN vocab={len(words)} words | train_occ={len(tr_items)} '
          f'gallery={len(gal_items)} (<= {RC.gallery_per_word}/word) '
          f'eval_queries={len(te_items)}', flush=True)

    tr_ds  = D.WordSet(tr_items)
    te_ds  = D.WordSet(te_items)
    gal_ds = D.WordSet(gal_items, real_only=True)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                       num_workers=args.workers, collate_fn=D.collate,
                       pin_memory=use_amp, persistent_workers=(args.workers > 0))
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False, num_workers=4,
                       collate_fn=D.collate)
    gal_dl = DataLoader(gal_ds, batch_size=256, shuffle=False, num_workers=args.workers,
                        collate_fn=lambda bb: {'mel_word': torch.stack([x['mel_word'] for x in bb]),
                                               'label': torch.tensor([x['label'] for x in bb])})

    # ── model: FRESH random init (no M14) ──
    model = M.build(CFG).to(device)
    print(f'[model] PLF (fresh)  params={model.count_params():,}', flush=True)

    # ── log-mel mean/std estimated from the data (no checkpoint) ──
    print('[stats] estimating log-mel mean/std from data …', flush=True)
    mean, std = D.estimate_mel_stats(tr_ds, n=512, seed=0)
    print(f'[stats] mel mean={mean:.3f} std={std:.3f}', flush=True)

    ema = EMA(model, args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup)) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - args.warmup) /
                                           max(1, args.steps - args.warmup)))) * 0.9 + 0.1))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log = {'args': vars(args), 'rcfg': vars(RC), 'n_words': len(words),
           'mel_mean': mean, 'mel_std': std, 'evals': []}
    step, best = 0, -1.0
    last_path = os.path.join(args.out_dir, 'last.pt')
    if args.resume and os.path.exists(last_path):
        r = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(r['model']); ema.shadow.load_state_dict(r['ema'])
        opt.load_state_dict(r['opt']); sched.load_state_dict(r['sched'])
        scaler.load_state_dict(r['scaler']); step = r['step']; best = r.get('best', -1.0)
        mean, std = r['mel_mean'], r['mel_std']
        print(f'[resume] step={step} best_top1={best:.4f}', flush=True)

    def save(path, extra=None):
        d = {'model': model.state_dict(), 'ema': ema.shadow.state_dict(),
             'opt': opt.state_dict(), 'sched': sched.state_dict(),
             'scaler': scaler.state_dict(), 'mel_mean': mean, 'mel_std': std,
             'step': step, 'best': best, 'n_words': len(words)}
        if extra:
            d.update(extra)
        torch.save(d, path)

    print('[eval] building real-mel gallery (once) …', flush=True)
    gallery_emb, gallery_y = build_gallery(gal_dl, device)
    print(f'[eval] gallery size={gallery_emb.shape[0]}', flush=True)

    t0 = time.time()
    model.train()
    data_iter = iter(tr_dl)
    while step < args.steps:
        try:
            b = next(data_iter)
        except StopIteration:
            data_iter = iter(tr_dl); b = next(data_iter)

        raw = b['raw'].to(device, non_blocking=True)
        mel_full = b['mel_full'].to(device, non_blocking=True)
        w_r = retr_weight(step)

        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            l_fm = model.flow_loss(raw, (mel_full - mean) / std)   # base: learn to gen
            l_con = l_corr = torch.zeros((), device=device)
            if w_r > 0:                                            # retrieval phase
                mel_word = b['mel_word'].to(device, non_blocking=True)
                labels = b['label'].to(device, non_blocking=True)
                gen = diff_sample(model, raw, args.train_steps).float() * std + mean
                e_gen = embed(gen_word_mel(gen, b['frs']))
                e_real = embed(mel_word)
                l_con = supcon_loss(e_gen, e_real, labels, RC.tau)
                l_corr = (1 - (e_gen * e_real).sum(dim=1)).mean()
            loss = l_fm + w_r * (RC.lambda_con * l_con + RC.lambda_corr * l_corr)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        ema.update(model)
        step += 1

        if step % 50 == 0:
            print(f'  step{step}/{args.steps} loss={loss.item():.4f} '
                  f'(fm={l_fm.item():.3f} con={float(l_con):.3f} corr={float(l_corr):.3f} '
                  f'w_retr={w_r:.2f}) lr={sched.get_last_lr()[0]:.2e} '
                  f'({time.time()-t0:.0f}s)', flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            t1, t5, n = knn_eval(ema.shadow, gallery_emb, gallery_y, te_dl,
                                 device, mean, std, use_amp)
            print(f'[eval] step{step}  OPEN-kNN top1={t1:.4f}  top5={t5:.4f}  '
                  f'(gallery={gallery_emb.shape[0]}, queries={n})', flush=True)
            log['evals'].append({'step': step, 'top1': t1, 'top5': t5,
                                 'gallery': int(gallery_emb.shape[0]), 'queries': n})
            if t1 > best:
                best = t1
                save(os.path.join(args.out_dir, 'best.pt'), {'metrics': {'top1': t1, 'top5': t5}})
                print(f'    ↑ new best top1={best:.4f} → best.pt', flush=True)
            save(last_path)
            with open(os.path.join(args.out_dir, 'train_log.json'), 'w') as f:
                json.dump(log, f, indent=2)
            model.train()

    save(last_path)
    with open(os.path.join(args.out_dir, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print('\n══ M22 open-vocab retrieval-generator (fresh) ═══════')
    print(f'  vocab            = {len(words)} words (open)')
    print(f'  BEST OPEN top1   = {best:.4f}   (gallery={gallery_emb.shape[0]})')
    print(f'  wall             = {time.time()-t0:.0f}s   → {args.out_dir}/best.pt')
    print('══════════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
