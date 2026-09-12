"""
train_lm.py
-----------
Pre-train the CharLM on Alice in Wonderland plain text.

Downloads text from Project Gutenberg (or reads a local file)
and trains a character-level LSTM language model.

Usage:
    python train_lm.py --text_path alice.txt --out_path lm.pt --epochs 50
"""

import argparse, os, sys, urllib.request
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import CharLM, VOCAB, VOCAB_SIZE, BLANK_IDX

ALICE_URL = (
    "https://www.gutenberg.org/files/11/11-0.txt"
)

# ── dataset ────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len=128):
        self.tokens  = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, i):
        x = torch.tensor(self.tokens[i:i + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[i+1:i + self.seq_len + 1], dtype=torch.long)
        return x, y


def encode_text(text):
    tokens = []
    for c in text.lower():
        if c in VOCAB:
            tokens.append(VOCAB[c])
    return tokens


def load_text(text_path):
    if text_path and os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    # Download from Gutenberg
    print(f"Downloading Alice in Wonderland from Project Gutenberg ...")
    try:
        with urllib.request.urlopen(ALICE_URL, timeout=30) as r:
            text = r.read().decode('utf-8', errors='ignore')
        if text_path:
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
        return text
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please provide --text_path to a local .txt file.")
        sys.exit(1)


# ── training ───────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    text   = load_text(args.text_path)
    tokens = encode_text(text)
    print(f"Text chars: {len(text)} | Vocabulary tokens: {len(tokens)}")

    split  = int(0.9 * len(tokens))
    train_ds = TextDataset(tokens[:split], args.seq_len)
    val_ds   = TextDataset(tokens[split:], args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model = CharLM(
        vocab_size=VOCAB_SIZE,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"CharLM params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=BLANK_IDX)

    best_val  = float('inf')
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            emb = model.embed(x)
            out, _ = model.lstm(emb)
            logits = model.linear(out)         # (B, T, V)
            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── val ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                emb = model.embed(x)
                out, _ = model.lstm(emb)
                logits = model.linear(out)
                val_loss += criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    y.reshape(-1)).item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"ppl={2**val_loss:.1f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), args.out_path)
            print(f"  --> saved {args.out_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping.")
                break

    print(f"\nBest val loss: {best_val:.4f}  (ppl={2**best_val:.1f})")
    print(f"LM saved -> {args.out_path}")


if __name__ == '__main__':
    _here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', default=os.path.join(_here, 'data', 'alice.txt'))
    parser.add_argument('--out_path',  default=os.path.join(_here, 'lm.pt'))
    parser.add_argument('--epochs',    type=int,   default=80)
    parser.add_argument('--batch_size',type=int,   default=64)
    parser.add_argument('--seq_len',   type=int,   default=128)
    parser.add_argument('--embed_dim', type=int,   default=64)
    parser.add_argument('--hidden',    type=int,   default=256)
    parser.add_argument('--n_layers',  type=int,   default=2)
    parser.add_argument('--dropout',   type=float, default=0.3)
    parser.add_argument('--lr',        type=float, default=3e-3)
    parser.add_argument('--patience',  type=int,   default=10)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    train(args)
