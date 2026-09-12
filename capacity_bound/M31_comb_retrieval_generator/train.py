#!/usr/bin/env python3
"""
train.py — M31: full-scale open-vocab A/B, comb front-end vs raw front-end, from scratch.

  L = flow_matching(front(raw), real 4s mel)                       # learn to generate
    + w(step) * [ lambda_con  * SupCon(proj(gen), proj(real))      # M30 learned metric
                + lambda_proxy* ProxyAnchor(proj(gen), proxies)    # M30 prototypes
                + lambda_corr * (1 - cos(proj(gen_i), proj(real_i))) ]

vs M22 which used embed_raw (raw-pixel Pearson). Retrieval — train loss AND eval kNN —
runs in the LEARNED projected space (M30 showed that is worth ~+6/+4 points).

Front-ends (the ONLY difference between arms; identical data, objective, schedule):
  --frontend raw   : decimate the 200 kHz window to 32 kHz on GPU -> M22 PowerlineEncoder
  --frontend comb  : demodulate ALL K harmonics on GPU            -> CombEncoder

Usage:  python train.py --frontend comb --steps 30000
"""
import argparse, copy, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio

from config import CFG, RC, OUT_DIR
import models_comb as M
import dataset_words as D
import metrics as MET
from comb_gpu import harmonic_gram_gpu, gram_features_gpu


def gen_word_mel(gen_full, frs):
    outs = []
    for i in range(gen_full.shape[0]):
        w = gen_full[i:i + 1, :, :int(frs[i])]
        w = F.interpolate(w[None], size=(CFG.n_mels, RC.T), mode='bilinear',
                          align_corners=False)[0]
        outs.append(w)
    return torch.cat(outs, 0)


def diff_sample(model, cond, steps):
    c = model.encode(cond)
    x = torch.randn(cond.shape[0], CFG.n_mels, CFG.n_frames, device=cond.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((cond.shape[0],), i * dt, device=cond.device)
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


class FrontEnd:
    """Turns the raw 200 kHz padded window into the arm's condition tensor."""
    def __init__(self, kind, dev):
        self.kind = kind
        self.dev = dev
        if kind == 'raw':
            self.rs = torchaudio.transforms.Resample(CFG.cap_sr, CFG.in_sr).to(dev)

    def __call__(self, raw, f0):
        if self.kind == 'comb':
            H = harmonic_gram_gpu(raw, f0, CFG.cap_sr, K=RC.K, bw=RC.bw, T=RC.Tg,
                                  pad_frac=CFG.pad_frac)
            return gram_features_gpu(H)                       # (B,2,K,Tg)
        y = self.rs(raw)                                      # 200k -> 32k
        a = int(round(CFG.pad_s * CFG.in_sr))
        y = y[:, a:a + CFG.in_len]                            # crop pad -> exactly win_s
        if y.shape[1] < CFG.in_len:
            y = F.pad(y, (0, CFG.in_len - y.shape[1]))
        return y


@torch.no_grad()
def build_gallery(loader, proj, dev):
    embs, ys = [], []
    for b in loader:
        embs.append(proj(b['mel_word'].to(dev)).half())
        ys.append(b['label'].to(dev))
    return torch.cat(embs), torch.cat(ys)


@torch.no_grad()
def knn_eval(model, proj, front, gal_e, gal_y, dl, dev, mean, std, amp):
    model.eval(); proj.eval()
    k = min(5, gal_e.shape[0]); t1 = t5 = n = 0
    for b in dl:
        cond = front(b['raw'].to(dev), b['f0'].to(dev))
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            gen = model.sample(cond, steps=RC.eval_steps, cfg_scale=RC.cfg_scale_eval)
        gen = gen.float() * std + mean
        q = proj(gen_word_mel(gen, b['frs'])).half()
        top = (q @ gal_e.t()).topk(k, 1).indices
        lab = gal_y[top]; y = b['label'].to(dev)
        t1 += (lab[:, 0] == y).sum().item()
        t5 += (lab == y[:, None]).any(1).sum().item(); n += len(y)
    model.train(); proj.train()
    return t1 / max(1, n), t5 / max(1, n), n


def retr_w(step):
    if step < RC.retr_start:
        return 0.0
    return min(1.0, (step - RC.retr_start) / max(1, RC.retr_ramp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frontend', choices=['raw', 'comb'], required=True)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--train-steps', type=int, default=RC.train_steps)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = (dev == 'cuda')
    out = args.out_dir or os.path.join(OUT_DIR, args.frontend)
    os.makedirs(out, exist_ok=True)

    tr_items, words = D.build_items('train')
    te_items, _ = D.build_items('test')
    rng = np.random.RandomState(0)
    if len(te_items) > RC.eval_queries:
        te_items = [te_items[i] for i in rng.choice(len(te_items), RC.eval_queries, False)]
    gal_items = D.cap_per_label(tr_items, RC.gallery_per_word)
    K = len(words)
    print(f'[m31/{args.frontend}] OPEN vocab={K} train={len(tr_items)} '
          f'gallery={len(gal_items)} queries={len(te_items)}', flush=True)
    if args.frontend == 'comb':
        print(f'[m31] comb: K={RC.K} harmonics (to {RC.K*60/1000:.0f} kHz) bw={RC.bw} '
              f'Tg={RC.Tg}', flush=True)

    tr_ds = D.WordSet(tr_items)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                       num_workers=args.workers, collate_fn=D.collate, pin_memory=amp,
                       persistent_workers=(args.workers > 0), prefetch_factor=4)
    te_dl = DataLoader(D.WordSet(te_items), batch_size=16, shuffle=False, num_workers=4,
                       collate_fn=D.collate)
    gal_dl = DataLoader(D.WordSet(gal_items, real_only=True), batch_size=256, shuffle=False,
                        num_workers=args.workers,
                        collate_fn=lambda b: {'mel_word': torch.stack([x['mel_word'] for x in b]),
                                              'label': torch.tensor([x['label'] for x in b])})

    model = M.build(CFG, args.frontend).to(dev)
    front = FrontEnd(args.frontend, dev)
    proj = MET.Projector(CFG.n_mels * RC.T, RC.proj_hidden, RC.proj_dim).to(dev)
    proxies = nn.Parameter(torch.randn(K, RC.proj_dim, device=dev) * 0.01)
    print(f'[m31/{args.frontend}] params: gen={model.count_params():,} '
          f'proj={sum(p.numel() for p in proj.parameters()):,} proxies={K*RC.proj_dim:,}',
          flush=True)

    print('[stats] estimating mel mean/std from data …', flush=True)
    mean, std = D.estimate_mel_stats(tr_ds, n=256)
    print(f'[stats] mean={mean:.3f} std={std:.3f}', flush=True)

    ema = EMA(model)
    opt = torch.optim.AdamW([{'params': model.parameters()},
                             {'params': proj.parameters()},
                             {'params': [proxies], 'lr': args.lr * 10}],
                            lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup)) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - args.warmup) /
                                           max(1, args.steps - args.warmup)))) * 0.9 + 0.1))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    log = {'frontend': args.frontend, 'args': vars(args), 'n_words': K,
           'mel_mean': mean, 'mel_std': std, 'evals': []}
    step, best = 0, -1.0
    last = os.path.join(out, 'last.pt')
    if args.resume and os.path.exists(last):
        r = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(r['model']); ema.shadow.load_state_dict(r['ema'])
        proj.load_state_dict(r['proj']); proxies.data = r['proxies'].to(dev)
        opt.load_state_dict(r['opt']); sched.load_state_dict(r['sched'])
        scaler.load_state_dict(r['scaler']); step = r['step']; best = r.get('best', -1.0)
        mean, std = r['mel_mean'], r['mel_std']
        print(f'[resume] step={step} best={best:.4f}', flush=True)

    def save(p, extra=None):
        d = {'model': model.state_dict(), 'ema': ema.shadow.state_dict(),
             'proj': proj.state_dict(), 'proxies': proxies.detach().cpu(),
             'opt': opt.state_dict(), 'sched': sched.state_dict(),
             'scaler': scaler.state_dict(), 'mel_mean': mean, 'mel_std': std,
             'step': step, 'best': best, 'frontend': args.frontend}
        if extra: d.update(extra)
        torch.save(d, p)

    print('[eval] building real-mel gallery …', flush=True)
    it = iter(tr_dl); t0 = time.time(); model.train()
    while step < args.steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(tr_dl); b = next(it)
        raw = b['raw'].to(dev, non_blocking=True)
        f0 = b['f0'].to(dev, non_blocking=True)
        mel_full = b['mel_full'].to(dev, non_blocking=True)
        w = retr_w(step)

        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            cond = front(raw, f0)
            l_fm = model.flow_loss(cond, (mel_full - mean) / std)
            l_con = l_prox = l_corr = torch.zeros((), device=dev)
            if w > 0:
                mw = b['mel_word'].to(dev, non_blocking=True)
                lab = b['label'].to(dev, non_blocking=True)
                gen = diff_sample(model, cond, args.train_steps).float() * std + mean
                zg = proj(gen_word_mel(gen, b['frs']))
                zr = proj(mw)
                l_con = MET.supcon(zg, zr, lab, lab, RC.tau)
                l_prox = MET.proxy_anchor(zg, lab, proxies, K, RC.proxy_alpha, RC.proxy_delta)
                l_corr = (1 - (zg * zr).sum(1)).mean()
            loss = l_fm + w * (RC.lambda_con * l_con + RC.lambda_proxy * l_prox
                               + RC.lambda_corr * l_corr)

        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(proj.parameters()), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        ema.update(model); step += 1

        if step % 50 == 0:
            print(f'  step{step}/{args.steps} loss={loss.item():.4f} (fm={l_fm.item():.3f} '
                  f'con={float(l_con):.3f} prox={float(l_prox):.3f} corr={float(l_corr):.3f} '
                  f'w={w:.2f}) {time.time()-t0:.0f}s', flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            ge, gy = build_gallery(gal_dl, proj, dev)
            t1, t5, n = knn_eval(ema.shadow, proj, front, ge, gy, te_dl, dev, mean, std, amp)
            print(f'[eval] step{step} OPEN-kNN(M30 learned metric) top1={t1:.4f} '
                  f'top5={t5:.4f} (gallery={ge.shape[0]}, q={n})', flush=True)
            log['evals'].append({'step': step, 'top1': t1, 'top5': t5})
            if t1 > best:
                best = t1; save(os.path.join(out, 'best.pt'), {'metrics': {'top1': t1, 'top5': t5}})
                print(f'    ↑ best top1={best:.4f}', flush=True)
            save(last)
            json.dump(log, open(os.path.join(out, 'train_log.json'), 'w'), indent=2)

    save(last); json.dump(log, open(os.path.join(out, 'train_log.json'), 'w'), indent=2)
    print(f'\n══ M31 [{args.frontend}] open-vocab retrieval (M30 learned metric) ══')
    print(f'  BEST top1 = {best:.4f}   (vocab={K}, refs: M22 basic-metric top1=0.039/top5=0.107)')
    print('════════════════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
