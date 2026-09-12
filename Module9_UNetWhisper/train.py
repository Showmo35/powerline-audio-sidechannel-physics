"""
train.py
--------
Module 9: Fine-tune Whisper on UNet-enhanced mel spectrograms.

Key difference from Module 8:
  - Input is UNet-enhanced mel (already in Whisper's format), NOT raw audio
  - WhisperProcessor used only for tokenisation, not feature extraction
  - Encoder is unfrozen by default (must adapt to enhanced mel domain)

Usage:
    python train.py --data data/unet_whisper_data.npz --out checkpoints
"""

import os, argparse, time, json, math
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dataset import UNetMelDataset, UNetWhisperCollator


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _edit(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def compute_cer(preds, refs):
    d, l = 0, 0
    for p, r in zip(preds, refs):
        d += _edit(p, r); l += max(len(r), 1)
    return d / max(l, 1)

def compute_wer(preds, refs):
    d, l = 0, 0
    for p, r in zip(preds, refs):
        d += _edit(p.split(), r.split()); l += max(len(r.split()), 1)
    return d / max(l, 1)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, device, scaler):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        feats  = batch['input_features'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast('cuda'):
                loss = model(input_features=feats, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss = model(input_features=feats, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, processor, device, n_decode=64):
    model.eval()
    total, n = 0.0, 0
    preds, refs = [], []
    for batch in loader:
        feats  = batch['input_features'].to(device)
        labels = batch['labels'].to(device)
        with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
            loss = model(input_features=feats, labels=labels).loss
        total += loss.item(); n += 1

        if len(preds) < n_decode:
            ids = model.generate(feats, max_new_tokens=225)
            dec = processor.batch_decode(ids, skip_special_tokens=True)
            ref_ids = labels.clone()
            ref_ids[ref_ids == -100] = processor.tokenizer.pad_token_id
            ref = processor.batch_decode(ref_ids, skip_special_tokens=True)
            preds.extend([p.lower().strip() for p in dec])
            refs.extend( [r.lower().strip() for r in ref])

    cer = compute_cer(preds, refs) if preds else 1.0
    wer = compute_wer(preds, refs) if preds else 1.0
    return total / max(n, 1), cer, wer, preds[:3], refs[:3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(args.out, exist_ok=True)
    device  = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f'Device: {device} | AMP: {use_amp}')

    # Load Whisper
    print(f'Loading {args.base_model} ...')
    processor = WhisperProcessor.from_pretrained(args.base_model)
    model     = WhisperForConditionalGeneration.from_pretrained(
                    args.base_model).to(device)
    model.generation_config.forced_decoder_ids = None

    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Encoder FROZEN. Trainable params: {n_train:,}')
    else:
        n_total = sum(p.numel() for p in model.parameters())
        print(f'Full model. Trainable params: {n_total:,}')

    # Load data
    print(f'Loading data: {args.data}')
    d = np.load(args.data, allow_pickle=True)
    collator   = UNetWhisperCollator()
    train_ds   = UNetMelDataset(d['mel_train'], d['text_train'], processor)
    val_ds     = UNetMelDataset(d['mel_val'],   d['text_val'],   processor)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  '
          f'Mel: {d["mel_train"].shape[1:]} → padded to (80, 3000)')

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collator)

    # Optimizer + warmup-cosine schedule
    trainable    = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-7 / args.lr, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda') if use_amp else None

    best_loss, best_cer, patience_count = float('inf'), float('inf'), 0
    history = []

    print(f'\nTraining {args.epochs} epochs | lr={args.lr} | batch={args.batch}')
    print(f'Warmup: {warmup_steps}/{total_steps} steps | Patience: {args.patience}\n')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        vl_loss, vl_cer, vl_wer, s_preds, s_refs = validate(
            model, val_loader, processor, device, args.n_decode_val)
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f'Epoch {epoch:3d} | train={tr_loss:.4f} val={vl_loss:.4f} '
              f'CER={vl_cer:.4f} WER={vl_wer:.4f} lr={lr_now:.2e} ({elapsed:.0f}s)')

        if s_preds and epoch % 5 == 0:
            for ref, pred in zip(s_refs[:2], s_preds[:2]):
                print(f'  REF : {ref[:100]}')
                print(f'  PRED: {pred[:100]}')

        history.append({'epoch': epoch, 'train_loss': tr_loss,
                        'val_loss': vl_loss, 'val_cer': vl_cer,
                        'val_wer': vl_wer, 'lr': lr_now})

        state = {'epoch': epoch, 'model_state': model.state_dict(),
                 'val_loss': vl_loss, 'val_cer': vl_cer, 'args': vars(args)}

        if vl_loss < best_loss:
            best_loss = vl_loss
            torch.save(state, os.path.join(args.out, 'best_loss_model.pt'))
            print(f'  ** New best loss ({vl_loss:.4f})')

        if vl_cer < best_cer:
            best_cer = vl_cer; patience_count = 0
            torch.save(state, os.path.join(args.out, 'best_cer_model.pt'))
            print(f'  ** New best CER  ({vl_cer:.4f})')
        else:
            patience_count += 1

        if args.save_every and epoch % args.save_every == 0:
            torch.save(state, os.path.join(args.out, f'checkpoint_ep{epoch}.pt'))

        if patience_count >= args.patience:
            print(f'\nEarly stopping at epoch {epoch} (best CER={best_cer:.4f})')
            break

    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    processor.save_pretrained(os.path.join(args.out, 'processor'))
    print(f'\nBest val_loss: {best_loss:.4f}')
    print(f'Best val_cer : {best_cer:.4f}')
    print(f'Checkpoints  : {args.out}')


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',           required=True)
    parser.add_argument('--out',            default=os.path.join(_HERE, 'checkpoints'))
    parser.add_argument('--base_model',     default='openai/whisper-small.en')
    parser.add_argument('--freeze_encoder', action='store_true', default=False)
    parser.add_argument('--epochs',         type=int,   default=30)
    parser.add_argument('--lr',             type=float, default=1e-5)
    parser.add_argument('--batch',          type=int,   default=16)
    parser.add_argument('--patience',       type=int,   default=10)
    parser.add_argument('--save_every',     type=int,   default=5)
    parser.add_argument('--n_decode_val',   type=int,   default=64)
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--device',         default='cuda')
    args = parser.parse_args()
    main(args)
