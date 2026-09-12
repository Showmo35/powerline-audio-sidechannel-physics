#!/usr/bin/env python3
"""
train.py — train the ViT-CTC reader on one input source (gen | real | null).

Identical architecture / hyperparameters / split for all three; the only thing
that differs is the mel source, so the held-out WER gap between them measures how
much lexical content survives (real = ceiling, null = envelope-only floor,
gen = the M14 reconstruction under test). Greedy WER/CER each epoch; the LM and
LLM back-ends run afterwards on dumped emissions (decode.py).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from config import CFG, OUT_DIR
import dataset as D
import models as M
import text as T


def spec_augment(mel, mel_lens, cfg=CFG):
    B, n_mels, _ = mel.shape
    for b in range(B):
        L = int(mel_lens[b])
        for _ in range(cfg.n_freq_masks):
            f = np.random.randint(0, cfg.freq_mask + 1)
            f0 = np.random.randint(0, max(1, n_mels - f))
            mel[b, f0:f0 + f, :] = 0.0
        for _ in range(cfg.n_time_masks):
            t = np.random.randint(0, max(1, int(L * cfg.time_mask_frac)) + 1)
            t0 = np.random.randint(0, max(1, L - t))
            mel[b, :, t0:t0 + t] = 0.0
    return mel


@torch.no_grad()
def evaluate(model, dl, device, mean, std, n_show=3):
    model.eval()
    refs, hyps = [], []
    for batch in dl:
        mel = ((batch['mel'] - mean) / std).to(device)
        logits, _ = model(mel, batch['mel_lens'])
        hyps += T.ctc_greedy_decode(logits.float().cpu())
        refs += batch['texts']
    w = T.corpus_wer(refs, hyps)
    cer_err = sum(T._edit_distance(list(r), list(h)) for r, h in zip(refs, hyps))
    cer = cer_err / max(1, sum(len(r) for r in refs))
    for r, h in list(zip(refs, hyps))[:n_show]:
        print(f'    ref: {r[:88]}')
        print(f'    hyp: {h[:88]}')
    model.train()
    return {'wer': w['wer'], 'cer': cer, 'n_utts': w['n_utts']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', choices=('gen', 'real', 'null', 'envamp'), required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=1500)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--no-specaug', action='store_true')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(OUT_DIR, f'vit_{args.input}')
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rows = D.load_rows()
    tr_rows, te_rows = D.split_rows(rows)
    tr_ds = D.MelText(tr_rows, args.input, train=True)
    te_ds = D.MelText(te_rows, args.input, train=False)
    print(f'[data] input={args.input}  train={len(tr_ds)}  test={len(te_ds)} utts', flush=True)

    stats_path = os.path.join(out_dir, 'mel_stats.json')
    if os.path.exists(stats_path):
        st = json.load(open(stats_path)); mean, std = st['mean'], st['std']
    else:
        mean, std = D.estimate_mel_stats(tr_ds)
        json.dump({'mean': mean, 'std': std}, open(stats_path, 'w'))
    print(f'[stats] mean={mean:.3f} std={std:.3f}', flush=True)

    tr_dl = torch.utils.data.DataLoader(
        tr_ds, batch_size=args.batch, shuffle=True, collate_fn=D.collate,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0))
    te_dl = torch.utils.data.DataLoader(
        te_ds, batch_size=args.batch, shuffle=False, collate_fn=D.collate,
        num_workers=4, pin_memory=True)

    model = M.build(CFG).to(device)
    print(f'[model] ViT-CTC params={model.count_params():,}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    total_steps = args.epochs * len(tr_dl)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / max(1, args.warmup)
        p = (s - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))

    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    ctc = torch.nn.CTCLoss(blank=T.BLANK_IDX, zero_infinity=True)

    step, start_ep, best = 0, 0, float('inf')
    log = {'args': vars(args), 'evals': []}
    last_path, best_path = os.path.join(out_dir, 'last.pt'), os.path.join(out_dir, 'best.pt')
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        step, start_ep, best = ck['step'], ck['epoch'] + 1, ck['best_wer']; log = ck['log']
        print(f'[resume] epoch={start_ep} step={step} best={best:.3f}', flush=True)

    import time; t0 = time.time()
    for ep in range(start_ep, args.epochs):
        for batch in tr_dl:
            step += 1
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            mel = (batch['mel'] - mean) / std
            if not args.no_specaug:
                mel = spec_augment(mel, batch['mel_lens'])
            mel = mel.to(device, non_blocking=True)
            with torch.autocast('cuda', enabled=(device == 'cuda')):
                logits, out_lens = model(mel, batch['mel_lens'])
                loss = ctc(F.log_softmax(logits.float(), dim=-1), batch['labels'].to(device),
                           out_lens.to(device), batch['label_lengths'].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if step % 100 == 0:
                print(f'  ep{ep} step{step}/{total_steps} ctc={loss.item():.3f} '
                      f'lr={lr_at(step):.2e} ({time.time()-t0:.0f}s)', flush=True)

        ev = evaluate(model, te_dl, device, mean, std)
        ev.update(epoch=ep, step=step)
        log['evals'].append(ev)
        print(f'[eval] ep{ep}  WER={ev["wer"]*100:.1f}%  CER={ev["cer"]*100:.1f}% '
              f'(n={ev["n_utts"]})', flush=True)
        state = {'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step,
                 'epoch': ep, 'best_wer': min(best, ev['wer']), 'mean': mean, 'std': std, 'log': log}
        torch.save(state, last_path)
        if ev['wer'] < best:
            best = ev['wer']; torch.save(state, best_path)
            print(f'    ↑ new best WER={best*100:.1f}% → saved', flush=True)
        json.dump(log, open(os.path.join(out_dir, 'train_log.json'), 'w'), indent=1)

    print(f'[done] input={args.input}  best greedy WER={best*100:.1f}%', flush=True)


if __name__ == '__main__':
    main()
