"""
train.py
--------
Module 8: Fine-tune Whisper-small.en on powerline audio.

Strategy to prevent overfitting on ~5k samples:
  - Freeze the Whisper encoder (pre-trained acoustic features are already good)
  - Only train the decoder (~88M params instead of 244M total)
  - Low learning rate (1e-5) with cosine decay
  - Early stopping on val loss

Usage:
    python train.py --data data/powerline_whisper_data.npz --out checkpoints
"""

import os, sys, argparse, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dataset import PowerlineWhisperDataset, WhisperDataCollator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def edit_distance(a, b):
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
        d += edit_distance(p, r)
        l += max(len(r), 1)
    return d / max(l, 1)


def compute_wer(preds, refs):
    d, l = 0, 0
    for p, r in zip(preds, refs):
        d += edit_distance(p.split(), r.split())
        l += max(len(r.split()), 1)
    return d / max(l, 1)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        input_features = batch['input_features'].to(device)
        labels         = batch['labels'].to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                out  = model(input_features=input_features, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(input_features=input_features, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, processor, device, n_decode=50):
    """Compute val loss + CER on up to n_decode samples."""
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_refs = [], []

    for batch in loader:
        input_features = batch['input_features'].to(device)
        labels         = batch['labels'].to(device)

        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                out = model(input_features=input_features, labels=labels)
        else:
            out = model(input_features=input_features, labels=labels)

        total_loss += out.loss.item()
        n += 1

        # Greedy decode a subset for CER
        if len(all_preds) < n_decode:
            pred_ids = model.generate(
                input_features,
                max_new_tokens=225,
            )
            decoded = processor.batch_decode(pred_ids, skip_special_tokens=True)
            # Decode references (replace -100 pad with pad_token_id)
            ref_ids = labels.clone()
            ref_ids[ref_ids == -100] = processor.tokenizer.pad_token_id
            refs    = processor.batch_decode(ref_ids, skip_special_tokens=True)
            all_preds.extend([p.lower().strip() for p in decoded])
            all_refs.extend( [r.lower().strip() for r in refs])

    val_loss = total_loss / max(n, 1)
    cer = compute_cer(all_preds, all_refs) if all_preds else 1.0
    wer = compute_wer(all_preds, all_refs) if all_preds else 1.0
    return val_loss, cer, wer, all_preds[:3], all_refs[:3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(args.out, exist_ok=True)
    device  = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f'Device: {device} | AMP: {use_amp}')

    # Load processor and model
    print(f'Loading {args.base_model} ...')
    processor = WhisperProcessor.from_pretrained(args.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model).to(device)

    # Disable forced decoder ids (let model predict freely)
    # Configure generation via generation_config (required in transformers 5.x)
    # whisper-small.en is English-only: no language/task tokens needed
    model.generation_config.forced_decoder_ids = None

    # Freeze encoder — only fine-tune decoder
    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        enc_params = sum(p.numel() for p in model.model.encoder.parameters())
        dec_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Encoder frozen ({enc_params:,} params). Training decoder: {dec_params:,} params')
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f'Full model training: {total:,} params')

    # Load data
    print(f'Loading data from {args.data} ...')
    d = np.load(args.data, allow_pickle=True)
    audio_train = d['audio_train']   # (N, win_samples)
    audio_val   = d['audio_val']
    text_train  = d['text_train']
    text_val    = d['text_val']
    print(f'Train: {len(audio_train)}  Val: {len(audio_val)}')
    print(f'Clip length: {audio_train.shape[1]/16000:.1f}s')

    collator   = WhisperDataCollator(processor=processor)
    train_ds   = PowerlineWhisperDataset(audio_train, text_train, processor)
    val_ds     = PowerlineWhisperDataset(audio_val,   text_val,   processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collator)

    # Optimizer + cosine LR schedule with warmup
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        return max(1e-7 / args.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss  = float('inf')
    best_val_cer   = float('inf')
    patience_count = 0
    history        = []

    print(f'\nTraining {args.epochs} epochs | lr={args.lr} | batch={args.batch}')
    print(f'Warmup: {warmup_steps}/{total_steps} steps')
    print(f'Patience: {args.patience} epochs on val loss\n')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_loss, val_cer, val_wer, sample_preds, sample_refs = validate(
            model, val_loader, processor, device, n_decode=args.n_decode_val)
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f'Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} '
              f'CER={val_cer:.4f} WER={val_wer:.4f} lr={lr_now:.2e} ({elapsed:.0f}s)')

        if sample_preds and epoch % 5 == 0:
            for ref, pred in zip(sample_refs[:2], sample_preds[:2]):
                print(f'  REF : {ref[:100]}')
                print(f'  PRED: {pred[:100]}')

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_cer': val_cer, 'val_wer': val_wer,
            'lr': lr_now,
        })

        save_state = {
            'epoch': epoch, 'model_state': model.state_dict(),
            'val_loss': val_loss, 'val_cer': val_cer,
            'args': vars(args),
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(save_state, os.path.join(args.out, 'best_loss_model.pt'))
            print(f'  ** New best loss ({val_loss:.4f})')

        if val_cer < best_val_cer:
            best_val_cer   = val_cer
            patience_count = 0
            torch.save(save_state, os.path.join(args.out, 'best_cer_model.pt'))
            print(f'  ** New best CER  ({val_cer:.4f})')
        else:
            patience_count += 1

        if args.save_every and epoch % args.save_every == 0:
            torch.save(save_state, os.path.join(
                args.out, f'checkpoint_ep{epoch}.pt'))

        if patience_count >= args.patience:
            print(f'\nEarly stopping at epoch {epoch} (best CER={best_val_cer:.4f})')
            break

    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Also save processor/tokenizer alongside checkpoints for inference
    processor.save_pretrained(os.path.join(args.out, 'processor'))

    print(f'\nBest val_loss: {best_val_loss:.4f}')
    print(f'Best val_cer : {best_val_cer:.4f}')
    print(f'Checkpoints  : {args.out}')


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',           required=True)
    parser.add_argument('--out',            default=os.path.join(_HERE, 'checkpoints'))
    parser.add_argument('--base_model',     default='openai/whisper-small.en')
    parser.add_argument('--freeze_encoder', action='store_true', default=True)
    parser.add_argument('--no_freeze_encoder', dest='freeze_encoder',
                        action='store_false')
    parser.add_argument('--epochs',         type=int,   default=30)
    parser.add_argument('--lr',             type=float, default=1e-5)
    parser.add_argument('--batch',          type=int,   default=8)
    parser.add_argument('--patience',       type=int,   default=10)
    parser.add_argument('--save_every',     type=int,   default=5)
    parser.add_argument('--n_decode_val',   type=int,   default=64,
                        help='Max val samples to greedy-decode for CER')
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--device',         default='cuda')
    args = parser.parse_args()
    main(args)
