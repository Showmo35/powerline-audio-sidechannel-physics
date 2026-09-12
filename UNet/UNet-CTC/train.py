"""
train.py
--------
Train CTC transcription model on UNet-predicted spectrograms.

Input: .npz from prepare_data.py containing predicted_train/val mels + text labels.
The model sees only UNet-predicted spectrograms (not clean or noisy).
"""

import os, sys, argparse, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import CTCEncoder, greedy_decode, BLANK_IDX, IDX_TO_CHAR


class PredictedMelDataset(Dataset):
    def __init__(self, mels, texts, text_lens):
        self.mels      = mels       # (N, 1, 80, T)
        self.texts     = texts      # (N, max_text_len)
        self.text_lens = text_lens  # (N,)

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.mels[idx]).float(),
                torch.from_numpy(self.texts[idx]).long(),
                int(self.text_lens[idx]))


def compute_cer(pred_texts, ref_texts):
    """Character Error Rate via edit distance."""
    total_dist, total_len = 0, 0
    for pred, ref in zip(pred_texts, ref_texts):
        d = _edit_distance(pred, ref)
        total_dist += d
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
    """Decode integer labels back to strings."""
    result = []
    for t, tl in zip(texts, text_lens):
        chars = [IDX_TO_CHAR.get(int(c), '') for c in t[:tl]]
        result.append(''.join(chars))
    return result


def train_epoch(model, loader, optimizer, device):
    model.train()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    total_loss, n_batches = 0.0, 0

    for mels, texts, text_lens in loader:
        mels = mels.to(device)
        texts = texts.to(device)
        text_lens = torch.tensor(text_lens, dtype=torch.long)

        logits = model(mels)                    # (T, B, V)
        log_probs = logits.log_softmax(dim=-1)
        T, B, _ = log_probs.shape
        input_lens = torch.full((B,), T, dtype=torch.long)

        loss = ctc_loss_fn(log_probs, texts, input_lens, text_lens)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, decode_batches=4):
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    total_loss, n_batches = 0.0, 0
    all_preds, all_refs = [], []

    for i, (mels, texts, text_lens) in enumerate(loader):
        mels = mels.to(device)
        texts = texts.to(device)
        text_lens_t = torch.tensor(text_lens, dtype=torch.long)

        logits = model(mels)
        log_probs = logits.log_softmax(dim=-1)
        T, B, _ = log_probs.shape
        input_lens = torch.full((B,), T, dtype=torch.long)

        loss = ctc_loss_fn(log_probs, texts, input_lens, text_lens_t)
        total_loss += loss.item()
        n_batches  += 1

        if i < decode_batches:
            preds = greedy_decode(logits.cpu())
            refs  = decode_labels(texts.cpu().numpy(), text_lens)
            all_preds.extend(preds)
            all_refs.extend(refs)

    avg_loss = total_loss / max(n_batches, 1)
    cer = compute_cer(all_preds, all_refs) if all_preds else 1.0
    return avg_loss, cer, all_preds[:5], all_refs[:5]


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {args.data} ...")
    d = np.load(args.data)
    train_mels = d['predicted_train']
    val_mels   = d['predicted_val']
    train_text = d['text_train']
    val_text   = d['text_val']
    train_tlen = d['text_len_train']
    val_tlen   = d['text_len_val']

    print(f"Train: {len(train_mels)} | Val: {len(val_mels)}")
    print(f"Mel shape: {train_mels.shape[1:]}")

    train_ds = PredictedMelDataset(train_mels, train_text, train_tlen)
    val_ds   = PredictedMelDataset(val_mels, val_text, val_tlen)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    model = CTCEncoder(
        n_mels=80, hidden=args.gru_hidden, n_layers=args.gru_layers,
        dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CTCEncoder: {n_params:,} params "
          f"(hidden={args.gru_hidden}, layers={args.gru_layers})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_cer, sample_preds, sample_refs = validate(
            model, val_loader, device, args.decode_batches)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} CER={val_cer:.4f} "
              f"lr={lr:.1e} ({elapsed:.1f}s)")

        if sample_preds and epoch % 10 == 0:
            print(f"  REF : {sample_refs[0][:80]}")
            print(f"  PRED: {sample_preds[0][:80]}")

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_cer': val_cer, 'lr': lr,
        })

        # Checkpointing
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_cer': val_cer,
                'args': vars(args),
            }, os.path.join(args.out, 'best_model.pt'))
            print(f"  ** New best (val_loss={val_loss:.4f}, CER={val_cer:.4f})")
        else:
            patience_counter += 1

        if args.save_every and epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_loss': val_loss,
                'val_cer': val_cer,
            }, os.path.join(args.out, f'checkpoint_ep{epoch}.pt'))

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    # Save history
    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to .npz from prepare_data.py')
    parser.add_argument('--out', required=True, help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gru_hidden', type=int, default=256)
    parser.add_argument('--gru_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--decode_batches', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args)
