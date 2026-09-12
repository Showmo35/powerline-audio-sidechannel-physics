#!/usr/bin/env python3
"""
train.py — unified trainer for the M13 raw-signal ASR sweep.

One entrypoint, five architectures (--arch m3|m4|m5|m6|m7).  Each is
RawFrontEnd → arch → CTC-greedy word WER on held-out chunks (default 41-46),
directly comparable to the M10 Whisper baseline (~103%) and the M11/M12 ~100%.

  m3  CTC BiGRU            : CTC loss
  m4  U-Net enhance + CTC  : CTC + MSTFT(enh, clean_mel)
  m5  Conformer CTC        : CTC loss
  m6  Hybrid CTC/Attn      : 0.7*CTC + 0.3*attention CE
  m7  FullSubNet + CTC     : CTC + MSTFT(enh, clean_mel)

Usage:
    python train.py --arch m5 --epochs 30 --batch 8 --lr 3e-4
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG, OUT_DIR
import dataset as D
import models as M
import text as T


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    refs, hyps = [], []
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        raw = batch['raw'].to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            out = model(raw)
        hyps.extend(T.ctc_greedy_decode(out['ctc'].float().cpu()))
        refs.extend(batch['texts'])
    return T.corpus_wer(refs, hyps), list(zip(refs[:4], hyps[:4]))


def attn_targets(labels, lengths, device):
    """Build (dec_in, dec_out) for the M6 attention decoder."""
    B, S = labels.shape
    dec_in = torch.full((B, S + 1), T.BLANK_IDX, dtype=torch.long, device=device)
    dec_out = torch.full((B, S + 1), -100, dtype=torch.long, device=device)
    dec_in[:, 0] = T.SOS_IDX
    for i in range(B):
        L = int(lengths[i])
        dec_in[i, 1:1 + L] = labels[i, :L]
        dec_out[i, :L] = labels[i, :L]
        dec_out[i, L] = T.EOS_IDX
    return dec_in, dec_out


def compute_loss(model, batch, device, enh_w=10.0):
    raw = batch['raw'].to(device)
    labels = batch['labels'].to(device)
    lab_lens = batch['label_lengths'].to(device)

    tgt_tokens = None
    if model.arch == 'm6':
        dec_in, dec_out = attn_targets(labels, lab_lens, device)
        tgt_tokens = dec_in

    out = model(raw, tgt_tokens) if model.arch == 'm6' else model(raw)
    ctc = out['ctc']                              # (Tt, B, V)
    Tt, B, _ = ctc.shape
    in_lens = torch.full((B,), Tt, dtype=torch.long, device=device)
    logp = F.log_softmax(ctc.float(), dim=-1)
    ctc_loss = F.ctc_loss(logp, labels, in_lens, lab_lens,
                          blank=T.BLANK_IDX, zero_infinity=True)

    log = {'ctc': ctc_loss.item()}
    loss = ctc_loss

    if model.arch == 'm6':
        attn = out['attn']                        # (B, S+1, Vdec)
        ce = F.cross_entropy(attn.reshape(-1, attn.size(-1)),
                             dec_out.reshape(-1), ignore_index=-100,
                             label_smoothing=0.1)
        loss = 0.7 * ctc_loss + 0.3 * ce
        log['attn'] = ce.item()

    if model.is_enh:
        clean = batch['clean_mel'].to(device).unsqueeze(1)   # (B,1,F,T)
        mstft = model.mstft(out['enh'], clean)
        loss = ctc_loss + enh_w * mstft
        log['mstft'] = mstft.item()

    return loss, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True, choices=list(M.ARCHS))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--enh-weight', type=float, default=10.0)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    out_dir = args.out_dir or os.path.join(OUT_DIR, args.arch)
    os.makedirs(out_dir, exist_ok=True)
    print(f'[dev] {device}  arch={args.arch}  fps={CFG.fps:.1f}  '
          f'L={int(CFG.max_dur_s * CFG.cap_sr)}')

    # ── data ──
    rows = D.load_manifest()
    test_chunks = D.chunk_set(CFG.test_chunks_spec)
    tr_rows, te_rows = D.split_by_chunk(rows, test_chunks)
    want_clean = args.arch in M._ENH
    tr_ds = D.RawUttDataset(tr_rows, CFG, want_clean=want_clean)
    te_ds = D.RawUttDataset(te_rows, CFG, want_clean=want_clean)
    print(f'[data] train={len(tr_rows)}  test={len(te_rows)}  '
          f'T_feat={tr_ds.T_feat}  clean_target={want_clean}')
    collate = D.make_collate(want_clean)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, collate_fn=collate, drop_last=True,
                       pin_memory=(device == 'cuda'))
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, collate_fn=collate)

    # ── model ──
    model = M.build(args.arch, CFG).to(device)
    print(f'[model] {args.arch}  params={model.count_params():,}')

    steps_per_epoch = len(tr_dl)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup)) *
        max(0.0, (total_steps - s) / max(1, total_steps - args.warmup)))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    log = {'args': vars(args), 'fps': CFG.fps, 'params': model.count_params(),
           'evals': []}
    best = math.inf
    step = 0
    t0 = time.time()
    done = False
    for ep in range(args.epochs):
        model.train()
        for batch in tr_dl:
            with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
                loss, ldict = compute_loss(model, batch, device, args.enh_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(); sched.step()
            step += 1

            if step % 50 == 0:
                extra = ' '.join(f'{k}={v:.3f}' for k, v in ldict.items())
                print(f'  ep{ep} step{step}/{total_steps} loss={loss.item():.3f} '
                      f'{extra} lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)')
            if step % args.eval_every == 0 or step == total_steps:
                sc, samples = evaluate(model, te_dl, device)
                print(f'[eval] step{step}  test WER = {sc["wer"]*100:.2f}%  '
                      f'(n={sc["n_utts"]}, words={sc["n_words"]})')
                for r, h in samples[:3]:
                    print(f'    REF: {r[:80]}\n    HYP: {h[:80]}')
                log['evals'].append({'step': step, **sc})
                if sc['wer'] < best:
                    best = sc['wer']
                    torch.save({'model_state': model.state_dict(),
                                'arch': args.arch, 'wer': best, 'step': step},
                               os.path.join(out_dir, 'best.pt'))
                    print(f'    ↑ new best {best*100:.2f}% → saved')
                model.train()
            if step >= total_steps:
                done = True; break
        if done:
            break

    log['best_wer'] = best
    log['wall_s'] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print('\n══ M13 raw-ASR result ════════════════════════════')
    print(f'  arch            = {args.arch}')
    print(f'  params          = {model.count_params():,}')
    print(f'  train / test    = {len(tr_rows)} / {len(te_rows)} utts')
    print(f'  BEST test WER   = {best*100:.2f}%')
    print(f'  refs: M10 Whisper-FT ~103% · M11 GAN ~100% · M12 U-Net ~100%')
    print(f'  wall            = {log["wall_s"]}s   → {out_dir}/best.pt')
    print('══════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
