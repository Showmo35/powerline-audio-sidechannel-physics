"""
train.py
--------
Train hybrid CTC/Attention model (Conformer encoder + Transformer decoder).

Joint loss: CTC_weight * CTC_loss + (1 - CTC_weight) * Attention_loss
Default: 0.7 CTC + 0.3 Attention (ESPnet recipe).

Features:
  - Dual checkpoints (best_loss + best_cer)
  - Full val CER on ALL samples
  - Early stopping on CER
  - Warmup + cosine LR
  - Label smoothing 0.1 on attention decoder

Usage:
    python train.py --data data/transcribe_data.npz --out checkpoints
"""

import os, sys, argparse, time, json, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import (HybridCTCAttentionModel, greedy_decode,
                   BLANK_IDX, DEC_VOCAB_SIZE, IDX_TO_CHAR)
from dataset import HybridMelDataset


def compute_cer(pred_texts, ref_texts):
    total_dist, total_len = 0, 0
    for pred, ref in zip(pred_texts, ref_texts):
        total_dist += _edit_distance(pred, ref)
        total_len  += max(len(ref), 1)
    return total_dist / max(total_len, 1)


def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def decode_labels(texts, text_lens):
    result = []
    for t, tl in zip(texts, text_lens):
        chars = [IDX_TO_CHAR.get(int(c), '') for c in t[:tl]]
        result.append(''.join(chars))
    return result


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(self.warmup_steps, 1)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def train_epoch(model, loader, optimizer, scheduler, device,
                ctc_weight, scaler=None):
    model.train()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    attn_loss_fn = nn.CrossEntropyLoss(
        ignore_index=BLANK_IDX, label_smoothing=0.1)

    total_loss, total_ctc, total_attn, n_batches = 0, 0, 0, 0

    for mels, ctc_texts, dec_inputs, dec_targets, text_lens in loader:
        mels = mels.to(device)
        ctc_texts = ctc_texts.to(device)
        dec_inputs = dec_inputs.to(device)
        dec_targets = dec_targets.to(device)
        text_lens = torch.tensor(text_lens, dtype=torch.long)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                ctc_logits, attn_logits = model(mels, dec_inputs)

                # CTC loss
                log_probs = ctc_logits.log_softmax(dim=-1)
                T, B, _ = log_probs.shape
                input_lens = torch.full((B,), T, dtype=torch.long)
                ctc_loss = ctc_loss_fn(log_probs, ctc_texts, input_lens, text_lens)

                # Attention loss
                attn_loss = attn_loss_fn(
                    attn_logits.reshape(-1, attn_logits.size(-1)),
                    dec_targets.reshape(-1))

                loss = ctc_weight * ctc_loss + (1 - ctc_weight) * attn_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            ctc_logits, attn_logits = model(mels, dec_inputs)

            log_probs = ctc_logits.log_softmax(dim=-1)
            T, B, _ = log_probs.shape
            input_lens = torch.full((B,), T, dtype=torch.long)
            ctc_loss = ctc_loss_fn(log_probs, ctc_texts, input_lens, text_lens)

            attn_loss = attn_loss_fn(
                attn_logits.reshape(-1, attn_logits.size(-1)),
                dec_targets.reshape(-1))

            loss = ctc_weight * ctc_loss + (1 - ctc_weight) * attn_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        total_ctc  += ctc_loss.item()
        total_attn += attn_loss.item()
        n_batches  += 1

    n = max(n_batches, 1)
    return total_loss / n, total_ctc / n, total_attn / n


@torch.no_grad()
def validate_full(model, loader, device):
    """Full val: CTC greedy CER on ALL samples."""
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    total_loss, n_batches = 0, 0
    all_preds, all_refs = [], []

    for mels, ctc_texts, dec_inputs, dec_targets, text_lens in loader:
        mels = mels.to(device)
        ctc_texts = ctc_texts.to(device)
        text_lens_t = torch.tensor(text_lens, dtype=torch.long)

        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                ctc_logits = model(mels)
        else:
            ctc_logits = model(mels)

        log_probs = ctc_logits.log_softmax(dim=-1)
        T, B, _ = log_probs.shape
        input_lens = torch.full((B,), T, dtype=torch.long)

        loss = ctc_loss_fn(log_probs.float(), ctc_texts, input_lens, text_lens_t)
        total_loss += loss.item()
        n_batches  += 1

        preds = greedy_decode(ctc_logits.cpu())
        refs  = decode_labels(ctc_texts.cpu().numpy(), text_lens)
        all_preds.extend(preds)
        all_refs.extend(refs)

    avg_loss = total_loss / max(n_batches, 1)
    cer = compute_cer(all_preds, all_refs) if all_preds else 1.0
    return avg_loss, cer, all_preds[:5], all_refs[:5]


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device} | AMP: {use_amp}")

    # Load data
    d = np.load(args.data)
    train_ds = HybridMelDataset(
        d['predicted_train'], d['text_train'], d['text_len_train'])
    val_ds = HybridMelDataset(
        d['predicted_val'], d['text_val'], d['text_len_val'])

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    model = HybridCTCAttentionModel(
        n_mels=80, d_model=args.d_model,
        enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        num_heads=args.num_heads, conv_kernel=args.conv_kernel,
        dropout=args.dropout,
    ).to(device)
    print(f"HybridCTCAttn: {model.count_params():,} params "
          f"(d={args.d_model}, enc={args.enc_layers}, dec={args.dec_layers})")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_pct * total_steps)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    best_val_cer  = float('inf')
    patience_counter = 0
    history = []
    start_epoch = 1

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Loaded model from epoch {ckpt['epoch']}")

        hist_path = os.path.join(args.out, 'history.json')
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
            history = [h for h in history if h['epoch'] <= ckpt['epoch']]
            if history:
                best_val_loss = min(h['val_loss'] for h in history)
                best_val_cer  = min(h['val_cer']  for h in history)
                patience_counter = 0
                for h in reversed(history):
                    if h['val_cer'] > best_val_cer:
                        patience_counter += 1
                    else:
                        break
            print(f"  Restored: best_cer={best_val_cer:.4f} "
                  f"best_loss={best_val_loss:.4f} patience={patience_counter}")

        elapsed_steps = (start_epoch - 1) * len(train_loader)
        for _ in range(elapsed_steps):
            scheduler.step()
        print(f"  Scheduler fast-forwarded {elapsed_steps} steps "
              f"(lr={scheduler.get_lr():.2e})")

    print(f"\nTraining {args.epochs} epochs | CTC weight={args.ctc_weight}")
    print(f"Warmup: {warmup_steps}/{total_steps} steps")
    print(f"Starting from epoch {start_epoch}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, ctc_l, attn_l = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            args.ctc_weight, scaler)
        val_loss, val_cer, sample_preds, sample_refs = validate_full(
            model, val_loader, device)
        elapsed = time.time() - t0
        lr = scheduler.get_lr()

        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} "
              f"(ctc={ctc_l:.4f} attn={attn_l:.4f}) "
              f"val_loss={val_loss:.4f} CER={val_cer:.4f} "
              f"lr={lr:.1e} ({elapsed:.1f}s)")

        if sample_preds and epoch % 10 == 0:
            print(f"  REF : {sample_refs[0][:80]}")
            print(f"  PRED: {sample_preds[0][:80]}")

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'ctc_loss': ctc_l, 'attn_loss': attn_l,
            'val_loss': val_loss, 'val_cer': val_cer, 'lr': lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'val_loss': val_loss, 'val_cer': val_cer,
                'args': vars(args),
            }, os.path.join(args.out, 'best_loss_model.pt'))

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'val_loss': val_loss, 'val_cer': val_cer,
                'args': vars(args),
            }, os.path.join(args.out, 'best_cer_model.pt'))
            print(f"  ** New best CER ({val_cer:.4f})")
        else:
            patience_counter += 1

        if args.save_every and epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'val_loss': val_loss, 'val_cer': val_cer,
            }, os.path.join(args.out, f'checkpoint_ep{epoch}.pt'))

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (best CER={best_val_cer:.4f})")
            break

    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Best val_cer:  {best_val_cer:.4f}")
    print(f"Checkpoints: {args.out}")


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default=os.path.join(_HERE, 'checkpoints'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--conv_kernel', type=int, default=31)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--ctc_weight', type=float, default=0.7)
    parser.add_argument('--warmup_pct', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(args)
