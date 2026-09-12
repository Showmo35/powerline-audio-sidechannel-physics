#!/usr/bin/env python3
"""
train_whisper.py — STEP 3: fine-tune Whisper to transcribe powerline mels.

Manual PyTorch loop (no Seq2SeqTrainer) for robustness against transformers API
churn: training uses model(input_features, labels).loss; eval uses
model.generate() → corpus WER on held-out chunks.

The model is fed AM-sideband mels DIRECTLY (no audio reconstruction).  Compare the
held-out WER against the Step-1 off-the-shelf baseline (94.6%).

Usage:
    python train_whisper.py --model openai/whisper-small \
        --test-chunks 41-46 --epochs 6 --batch 16 --lr 1e-5
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import OUT_DIR
import data as D
import asr


def chunk_set(spec):
    lo, hi = (int(x) for x in spec.split('-'))
    return {f'chunk_{n:03d}' for n in range(lo, hi + 1)}


@torch.no_grad()
def evaluate(model, loader, processor, device, max_batches=None):
    model.eval()
    refs, hyps = [], []
    gen_kw = dict(max_new_tokens=200, num_beams=1, no_repeat_ngram_size=4)
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        feats = batch['input_features'].to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            ids = model.generate(input_features=feats, **gen_kw)
        hyps.extend(processor.batch_decode(ids, skip_special_tokens=True))
        refs.extend(batch['texts'])
    score = asr.corpus_wer([asr.normalize_text(r) for r in refs],
                           [asr.normalize_text(h) for h in hyps])
    return score, list(zip(refs[:6], hyps[:6]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='openai/whisper-small')
    ap.add_argument('--test-chunks', default='41-46')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--eval-every', type=int, default=400, help='steps between evals')
    ap.add_argument('--max-steps', type=int, default=0, help='0 = use epochs')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out-dir', default=os.path.join(OUT_DIR, 'whisper_ft'))
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[dev] {device}  model={args.model}')

    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained(args.model)
    try:
        processor.tokenizer.set_prefix_tokens(language='en', task='transcribe')
    except Exception as e:
        print(f'[warn] set_prefix_tokens: {e}')
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    # All generation control must live on generation_config in this transformers
    # version (setting model.config.* raises).
    for attr, val in (('language', 'en'), ('task', 'transcribe'),
                      ('forced_decoder_ids', None), ('suppress_tokens', [])):
        try:
            setattr(model.generation_config, attr, val)
        except Exception as e:
            print(f'[warn] generation_config.{attr}: {e}')

    # ── data ──────────────────────────────────────────────────────────────────
    rows = D.load_manifest()
    test_chunks = chunk_set(args.test_chunks)
    tr_rows, te_rows = D.split_by_chunk(rows, test_chunks)
    print(f'[data] train={len(tr_rows)} utts  test={len(te_rows)} utts  '
          f'(test={sorted(test_chunks)[0]}…{sorted(test_chunks)[-1]})')

    collate = D.make_collate()
    tr_ds = D.PowerlineMelDataset(tr_rows, processor.tokenizer)
    te_ds = D.PowerlineMelDataset(te_rows, processor.tokenizer)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, collate_fn=collate, drop_last=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, collate_fn=collate)

    steps_per_epoch = len(tr_dl)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup)) *
        max(0.0, (total_steps - s) / max(1, total_steps - args.warmup)))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    # ── baseline eval (untrained on our features) ─────────────────────────────
    print('[eval] pre-training (model on raw powerline mels) …')
    base, _ = evaluate(model, te_dl, processor, device, max_batches=10)
    print(f'[eval] pre-train WER (10 batches) = {base["wer"]*100:.1f}%')

    # ── train ─────────────────────────────────────────────────────────────────
    log = {'args': vars(args), 'pretrain_wer': base['wer'], 'evals': []}
    best = math.inf
    step = 0
    t0 = time.time()
    model.train()
    done = False
    for ep in range(args.epochs):
        for batch in tr_dl:
            feats = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
                loss = model(input_features=feats, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(); sched.step()
            step += 1

            if step % 50 == 0:
                print(f'  ep{ep} step{step}/{total_steps} loss={loss.item():.3f} '
                      f'lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)')
            if step % args.eval_every == 0 or step == total_steps:
                sc, samples = evaluate(model, te_dl, processor, device)
                print(f'[eval] step{step}  test WER = {sc["wer"]*100:.2f}%  '
                      f'(n={sc["n_utts"]})')
                for r, h in samples[:3]:
                    print(f'    REF: {r[:80]}\n    HYP: {h[:80]}')
                log['evals'].append({'step': step, **sc})
                if sc['wer'] < best:
                    best = sc['wer']
                    model.save_pretrained(os.path.join(args.out_dir, 'best'))
                    processor.save_pretrained(os.path.join(args.out_dir, 'best'))
                    print(f'    ↑ new best {best*100:.2f}% → saved')
                model.train()
            if step >= total_steps:
                done = True; break
        if done:
            break

    log['best_wer'] = best
    log['wall_s'] = round(time.time() - t0, 1)
    with open(os.path.join(args.out_dir, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print('\n══ STEP 3 — fine-tune result ═════════════════════')
    print(f'  model           = {args.model}')
    print(f'  train / test    = {len(tr_rows)} / {len(te_rows)} utts')
    print(f'  pre-train WER   = {base["wer"]*100:.1f}%  (on raw powerline mels)')
    print(f'  BEST test WER   = {best*100:.2f}%')
    print(f'  Step-1 baseline = 94.6%  (off-the-shelf GL audio)')
    print(f'  wall            = {log["wall_s"]}s   → {args.out_dir}/best')
    print('══════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
