"""
inference.py
------------
Evaluate hybrid CTC/Attention model on validation set.
Compares CTC greedy decode vs attention decoder greedy decode.

Usage:
    python inference.py --data data/transcribe_data.npz \
                        --ckpt checkpoints/best_cer_model.pt
"""

import os, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import (HybridCTCAttentionModel, greedy_decode,
                   IDX_TO_CHAR, DEC_VOCAB_SIZE, SOS_IDX, EOS_IDX)


def decode_labels(texts, text_lens):
    result = []
    for t, tl in zip(texts, text_lens):
        chars = [IDX_TO_CHAR.get(int(c), '') for c in t[:tl]]
        result.append(''.join(chars))
    return result


def edit_distance(a, b):
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


def compute_cer(preds, refs):
    total_dist, total_len = 0, 0
    for p, r in zip(preds, refs):
        total_dist += edit_distance(p, r)
        total_len  += max(len(r), 1)
    return total_dist / max(total_len, 1)


def compute_wer(preds, refs):
    total_dist, total_len = 0, 0
    for p, r in zip(preds, refs):
        pw, rw = p.split(), r.split()
        total_dist += edit_distance(pw, rw)
        total_len  += max(len(rw), 1)
    return total_dist / max(total_len, 1)


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load data
    d = np.load(args.data)
    val_predicted = d['predicted_val']
    val_text      = d['text_val']
    val_tlen      = d['text_len_val']
    print(f"Val samples: {len(val_predicted)}")

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})

    model = HybridCTCAttentionModel(
        n_mels=80,
        d_model=ckpt_args.get('d_model', 256),
        enc_layers=ckpt_args.get('enc_layers', 6),
        dec_layers=ckpt_args.get('dec_layers', 3),
        num_heads=ckpt_args.get('num_heads', 4),
        conv_kernel=ckpt_args.get('conv_kernel', 31),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, "
          f"CER={ckpt.get('val_cer', 0):.4f})")

    all_refs = decode_labels(val_text, val_tlen)

    # ── CTC greedy decode ────────────────────────────────────────────────────
    print("\n>>> CTC greedy decode ...")
    ctc_preds = []
    with torch.no_grad():
        for i in range(0, len(val_predicted), args.batch):
            mels = torch.from_numpy(val_predicted[i:i+args.batch]).float().to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    ctc_logits = model(mels)
            else:
                ctc_logits = model(mels)
            preds = greedy_decode(ctc_logits.cpu())
            ctc_preds.extend(preds)

    ctc_cer = compute_cer(ctc_preds, all_refs)
    ctc_wer = compute_wer(ctc_preds, all_refs)
    print(f"    CER: {ctc_cer:.4f}  WER: {ctc_wer:.4f}")

    # ── Attention decoder greedy decode ──────────────────────────────────────
    print("\n>>> Attention decoder greedy decode ...")
    attn_preds = []
    with torch.no_grad():
        for i in range(0, len(val_predicted), args.batch):
            mels = torch.from_numpy(val_predicted[i:i+args.batch]).float().to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    enc_out = model.encode(mels)
            else:
                enc_out = model.encode(mels)
            preds = model.greedy_decode_attn(enc_out, max_len=200)
            attn_preds.extend(preds)

    attn_cer = compute_cer(attn_preds, all_refs)
    attn_wer = compute_wer(attn_preds, all_refs)
    print(f"    CER: {attn_cer:.4f}  WER: {attn_wer:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {'Method':<25} {'CER':>8} {'WER':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    print(f"  {'CTC Greedy':<25} {ctc_cer:>8.4f} {ctc_wer:>8.4f}")
    print(f"  {'Attn Greedy':<25} {attn_cer:>8.4f} {attn_wer:>8.4f}")
    print(f"{'='*60}")

    # Save results
    summary = {
        'n_val': len(all_refs),
        'ctc_greedy': {'cer': ctc_cer, 'wer': ctc_wer},
        'attn_greedy': {'cer': attn_cer, 'wer': attn_wer},
        'checkpoint': args.ckpt,
    }
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save transcriptions
    with open(os.path.join(args.out, 'transcriptions.txt'), 'w') as f:
        f.write(f"CTC  CER: {ctc_cer:.4f} | WER: {ctc_wer:.4f}\n")
        f.write(f"Attn CER: {attn_cer:.4f} | WER: {attn_wer:.4f}\n")
        f.write('=' * 70 + '\n\n')
        for i, ref in enumerate(all_refs):
            c_cer = edit_distance(ctc_preds[i], ref) / max(len(ref), 1)
            a_cer = edit_distance(attn_preds[i], ref) / max(len(ref), 1)
            f.write(f"--- Sample {i} (CTC={c_cer:.3f}, Attn={a_cer:.3f}) ---\n")
            f.write(f"  REF : {ref}\n")
            f.write(f"  CTC : {ctc_preds[i]}\n")
            f.write(f"  ATTN: {attn_preds[i]}\n\n")

    print(f"Saved to: {args.out}")


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default=os.path.join(_HERE, 'inference_output'))
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args)
