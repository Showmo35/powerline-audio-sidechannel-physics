#!/usr/bin/env python3
"""Train a small CNN-CTC ASR model from scratch using the USB->MP3 aligned examples.

This script reuses the USB alignment utilities from the existing
`usb_whisper_finetune.py` to produce (audio, text) examples, converts audio to
log-mel spectrograms, trains a lightweight Conv-CTC model, and saves the
checkpoint + tokenizer mapping for later inference.
"""
import argparse
import json
import os
import sys
from typing import List

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure we can import the existing utilities in the same folder
HERE = os.path.dirname(__file__)
sys.path.append(HERE)
try:
    from usb_whisper_finetune import build_usb_text_examples, find_usb_files
except Exception:
    raise RuntimeError("Could not import build_usb_text_examples from usb_whisper_finetune.py")


def build_tokenizer(texts: List[str]):
    # Normalize to lowercase and collapse whitespace so tokenizer is compact
    norm_texts = [" ".join(t.lower().split()) for t in texts]
    chars = sorted(list({c for t in norm_texts for c in t}))
    # Reserve 0 for CTC blank
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i + 1: c for i, c in enumerate(chars)}
    stoi["<blank>"] = 0
    itos[0] = ""
    return stoi, itos


def text_to_ints(text: str, stoi: dict):
    if text is None:
        return []
    t = " ".join(text.lower().split())
    return [stoi.get(c, stoi.get(" ", 0)) for c in t]


def compute_log_mel(y: np.ndarray, sr: int, n_mels=80, hop_length=160, n_fft=512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalize to mean 0, var 1 per utterance
    S_norm = (S_db - S_db.mean()) / (S_db.std() + 1e-9)
    return S_norm.astype(np.float32)


class USBASRDataset(Dataset):
    def __init__(self, examples, stoi, sample_rate=16000, n_mels=80, hop_length=160):
        self.examples = examples
        self.stoi = stoi
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.spec_augment = False

    def enable_spec_augment(self):
        self.spec_augment = True

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        audio = ex["audio"]["array"].astype(np.float32)
        sr = ex["audio"]["sampling_rate"]
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        mel = compute_log_mel(audio, sr=self.sample_rate, n_mels=self.n_mels, hop_length=self.hop_length)
        # mel: (n_mels, T)
        label = text_to_ints(ex.get("text", ""), self.stoi)

        # Apply simple SpecAugment (time and frequency masking) during training
        if self.spec_augment:
            # freq mask
            n_mels = mel.shape[0]
            f = np.random.randint(0, max(1, n_mels // 8))
            f0 = np.random.randint(0, max(1, n_mels - f + 1))
            mel[f0 : f0 + f, :] = 0.0
            # time mask
            n_t = mel.shape[1]
            t = np.random.randint(0, max(1, n_t // 8))
            t0 = np.random.randint(0, max(1, n_t - t + 1))
            mel[:, t0 : t0 + t] = 0.0
        return mel, np.array(label, dtype=np.int32)


def collate_fn(batch):
    mels, labels = zip(*batch)
    max_T = max(m.shape[1] for m in mels)
    max_L = max(len(l) for l in labels)

    batch_size = len(mels)
    n_mels = mels[0].shape[0]

    x = np.zeros((batch_size, 1, n_mels, max_T), dtype=np.float32)
    x_lens = np.zeros(batch_size, dtype=np.int32)
    y = np.zeros((batch_size, max_L), dtype=np.int32)
    y_lens = np.zeros(batch_size, dtype=np.int32)

    for i, (m, lab) in enumerate(zip(mels, labels)):
        x[i, 0, : m.shape[0], : m.shape[1]] = m
        x_lens[i] = m.shape[1]
        y[i, : len(lab)] = lab
        y_lens[i] = len(lab)

    return (
        torch.from_numpy(x),
        torch.from_numpy(x_lens).long(),
        torch.from_numpy(y).long(),
        torch.from_numpy(y_lens).long(),
    )


class ConvCTCModel(nn.Module):
    def __init__(self, n_mels: int, num_classes: int, channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels // 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(channels // 2),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(channels // 2, channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(channels),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(channels),
        )

        # Frequency dimension after pooling: n_mels // 4
        self.fc = nn.Linear((n_mels // 4) * channels, channels)
        # Add a small bidirectional LSTM for temporal modeling before classification
        self.rnn = nn.LSTM(input_size=channels, hidden_size=channels // 2, num_layers=1, bidirectional=True)
        # classifier maps bi-LSTM output (hidden_size*2 == channels) to classes
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = self.conv(x)
        # x: (B, C, n_mels//4, T)
        b, c, f, t = x.shape
        x = x.permute(3, 0, 1, 2)  # (T, B, C, F)
        x = x.reshape(t, b, c * f)
        x = self.fc(x)
        x = torch.relu(x)
        # x: (T, B, channels) -> rnn expects (T, B, features)
        rnn_out, _ = self.rnn(x)
        logits = self.classifier(rnn_out)  # (T, B, num_classes)
        log_probs = torch.log_softmax(logits, dim=2)
        return log_probs


def train(args):
    # Build aligned examples using existing utilities
    usb_paths = find_usb_files(args.data_roots)
    if not usb_paths:
        raise RuntimeError("No USB *_img.bin files found under --data-roots")

    examples = build_usb_text_examples(
        usb_paths=usb_paths,
        mp3_root=args.mp3_root,
        teacher_model_name=args.teacher_model,
        language=args.language,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        bandwidth=args.bandwidth,
        baseband_lpf_cutoff=args.baseband_lpf_cutoff,
        asr_sample_rate=args.asr_sample_rate,
        max_seconds_per_usb=args.max_seconds_per_usb,
        chapter_start_offset_sec=args.chapter_start_offset_sec,
    )

    if len(examples) < 8:
        raise RuntimeError(f"Only {len(examples)} examples; need more aligned data for stable training.")

    texts = [e["text"] for e in examples]
    stoi, itos = build_tokenizer(texts)
    num_classes = max(itos.keys()) + 1

    # Split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(examples))
    rng.shuffle(idx)
    split = int((1 - args.eval_ratio) * len(idx))
    train_idx, eval_idx = idx[:split], idx[split:]

    train_examples = [examples[i] for i in train_idx]
    eval_examples = [examples[i] for i in eval_idx]

    train_ds = USBASRDataset(train_examples, stoi, sample_rate=args.asr_sample_rate)
    eval_ds = USBASRDataset(eval_examples, stoi, sample_rate=args.asr_sample_rate)

    # Enable SpecAugment on training dataset if requested
    if args.spec_augment:
        train_ds.enable_spec_augment()

    # Smoke mode: reduce dataset size and force fewer epochs for quick testing
    if args.smoke:
        max_ex = int(args.smoke_max_examples)
        train_ds.examples = train_ds.examples[:max_ex]
        eval_ds.examples = eval_ds.examples[:min(max(1, max_ex // 10), len(eval_ds.examples))]
        print(f"SMOKE MODE: using {len(train_ds.examples)} train examples and {len(eval_ds.examples)} eval examples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    model = ConvCTCModel(n_mels=args.n_mels, num_classes=num_classes, channels=args.channels).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, x_lens, y, y_lens in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            x = x.to(device)
            y = y.to(device)
            # Forward
            log_probs = model(x)  # (T, B, C)
            T, B, C = log_probs.shape
            input_lengths = (x_lens.to(device) // 1)  # approximate in frames
            # Convert input_lengths to output frame counts after conv/pooling
            # Our pooling reduces freq dimension only, time stays the same, so T_out ~= input hop frames
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            # Prepare targets as 1D
            targets = []
            for i in range(B):
                targets.append(y[i, : y_lens[i]].cpu().numpy())
            targets_concat = torch.from_numpy(np.concatenate(targets)).to(device)
            target_lengths = y_lens.to(device)

            loss = criterion(log_probs, targets_concat, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} average train loss: {avg_loss:.4f}")

        # quick eval loop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for x, x_lens, y, y_lens in eval_loader:
                x = x.to(device)
                y = y.to(device)
                log_probs = model(x)
                T, B, C = log_probs.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                targets = []
                for i in range(B):
                    targets.append(y[i, : y_lens[i]].cpu().numpy())
                targets_concat = torch.from_numpy(np.concatenate(targets)).to(device)
                target_lengths = y_lens.to(device)
                l = criterion(log_probs, targets_concat, input_lengths, target_lengths)
                val_loss += float(l.item())

            val_loss /= max(1, len(eval_loader))
            print(f"Epoch {epoch} validation loss: {val_loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "cnn_ctc_model.pth"))
    with open(os.path.join(args.output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos}, f, ensure_ascii=False, indent=2)

    print(f"Saved model and tokenizer to: {args.output_dir}")
    # -----------------------------
    # Evaluate on the held-out eval set (greedy CTC decode)
    # -----------------------------
    def greedy_decode_batch(log_probs, itos):
        # log_probs: (T, B, C) torch tensor
        preds = log_probs.argmax(dim=2).cpu().numpy()  # (T, B)
        B = preds.shape[1]
        texts = []
        for b in range(B):
            seq = preds[:, b].tolist()
            out_chars = []
            prev = None
            for s in seq:
                if s == prev:
                    prev = s
                    continue
                if s != 0:
                    out_chars.append(itos.get(int(s), ""))
                prev = s
            texts.append(''.join(out_chars))
        return texts

    def edit_distance(a: List[str], b: List[str]) -> int:
        # simple Levenshtein
        n, m = len(a), len(b)
        if n == 0:
            return m
        if m == 0:
            return n
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[n][m]

    def cer(ref: str, hyp: str) -> float:
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        if len(ref_chars) == 0:
            return float(len(hyp_chars))
        return edit_distance(ref_chars, hyp_chars) / float(len(ref_chars))

    def wer(ref: str, hyp: str) -> float:
        ref_words = ref.split()
        hyp_words = hyp.split()
        if len(ref_words) == 0:
            return float(len(hyp_words))
        return edit_distance(ref_words, hyp_words) / float(len(ref_words))

    # Build mappings for decoding
    # itos currently maps ints->char
    # Evaluate
    model.eval()
    all_preds = []
    all_refs = []
    with torch.no_grad():
        for x, x_lens, y, y_lens in tqdm(eval_loader, desc="Evaluating"):
            x = x.to(device)
            log_probs = model(x)  # (T, B, C)
            batch_texts = greedy_decode_batch(log_probs, itos)

            B = y.shape[0]
            for i in range(B):
                lab_len = int(y_lens[i].item())
                lab_seq = [int(int(v)) for v in y[i, :lab_len].cpu().numpy().tolist()]
                ref_chars = [itos.get(int(v), "") for v in lab_seq]
                ref_text = ''.join(ref_chars)
                all_refs.append(ref_text)
            all_preds.extend(batch_texts)

    # Compute metrics
    n = len(all_refs)
    total_cer = 0.0
    total_wer = 0.0
    counted = 0
    for ref, pred in zip(all_refs, all_preds):
        total_cer += cer(ref, pred)
        total_wer += wer(ref, pred)
        counted += 1

    avg_cer = total_cer / max(1, counted)
    avg_wer = total_wer / max(1, counted)

    print(f"Evaluation examples: {counted}")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")

    # Save predictions
    pred_lines = []
    for ref, pred in zip(all_refs, all_preds):
        pred_lines.append(json.dumps({"reference": ref, "prediction": pred}, ensure_ascii=False))
    pred_file = os.path.join(args.output_dir, "eval_predictions_cnn.jsonl")
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write("\n".join(pred_lines))
    print(f"Saved CNN eval predictions to: {pred_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-roots", nargs="+", default=[
        "<REPO_ROOT>/May29_Alice",
        "<REPO_ROOT>/July10_Podcasts",
    ])
    p.add_argument("--mp3-root", default="<REPO_ROOT>/Alice_In_Wonderland_mp3")
    p.add_argument("--sample-rate", type=int, default=200000)
    p.add_argument("--asr-sample-rate", type=int, default=16000)
    p.add_argument("--center-freq", type=float, default=20000.0)
    p.add_argument("--bandwidth", type=float, default=2500.0)
    p.add_argument("--baseband-lpf-cutoff", type=float, default=2500.0)
    p.add_argument("--chapter-start-offset-sec", type=float, default=0.0)
    p.add_argument("--max-seconds-per-usb", type=float, default=None)
    p.add_argument("--teacher-model", default="base")
    p.add_argument("--language", default="en")
    p.add_argument("--output-dir", default="<REPO_ROOT>/output/cnn_asr")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--use-cuda", action="store_true")
    p.add_argument("--spec-augment", action="store_true", help="Enable simple SpecAugment (freq/time masking) on training examples")
    p.add_argument("--smoke", action="store_true", help="Run a quick smoke training run with limited examples/epochs")
    p.add_argument("--smoke-max-examples", type=int, default=64, help="Maximum training examples to use in smoke mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
