"""
inference.py
------------
Full validation evaluation for Conformer CTC model.
Evaluates ALL samples, computes CER/WER, saves detailed results.

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

from model import ConformerCTCModel, greedy_decode, IDX_TO_CHAR


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

    model = ConformerCTCModel(
        n_mels=80,
        d_model=ckpt_args.get('d_model', 256),
        n_layers=ckpt_args.get('n_layers', 6),
        num_heads=ckpt_args.get('num_heads', 4),
        conv_kernel=ckpt_args.get('conv_kernel', 31),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, "
          f"CER={ckpt.get('val_cer', 0):.4f})")

    # Inference
    all_preds, all_refs = [], []
    with torch.no_grad():
        for i in range(0, len(val_predicted), args.batch):
            mels = torch.from_numpy(val_predicted[i:i+args.batch]).float().to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(mels)
            else:
                logits = model(mels)
            preds = greedy_decode(logits.cpu())
            refs  = decode_labels(val_text[i:i+args.batch], val_tlen[i:i+args.batch])
            all_preds.extend(preds)
            all_refs.extend(refs)

    cer = compute_cer(all_preds, all_refs)
    wer = compute_wer(all_preds, all_refs)

    print(f"\n{'='*60}")
    print(f"  CER: {cer:.4f}")
    print(f"  WER: {wer:.4f}")
    print(f"  Val samples: {len(all_preds)}")
    print(f"{'='*60}")

    # Per-sample analysis
    sample_results = []
    for i, (p, r) in enumerate(zip(all_preds, all_refs)):
        s_cer = edit_distance(p, r) / max(len(r), 1)
        sample_results.append({'id': i, 'cer': s_cer, 'pred': p, 'ref': r})

    sample_results.sort(key=lambda x: x['cer'])
    cers = [r['cer'] for r in sample_results]

    # Save transcriptions
    with open(os.path.join(args.out, 'transcriptions.txt'), 'w') as f:
        f.write(f"CER: {cer:.4f} | WER: {wer:.4f} | Samples: {len(all_preds)}\n")
        f.write(f"Mean CER: {np.mean(cers):.4f} | Median: {np.median(cers):.4f} | "
                f"Std: {np.std(cers):.4f}\n")
        f.write('=' * 70 + '\n\n')
        f.write("--- WORST 10 ---\n\n")
        for r in sample_results[-10:]:
            f.write(f"Sample {r['id']} (CER={r['cer']:.3f})\n")
            f.write(f"  REF : {r['ref']}\n  PRED: {r['pred']}\n\n")
        f.write("--- BEST 10 ---\n\n")
        for r in sample_results[:10]:
            f.write(f"Sample {r['id']} (CER={r['cer']:.3f})\n")
            f.write(f"  REF : {r['ref']}\n  PRED: {r['pred']}\n\n")

    # Save JSON
    summary = {
        'cer': cer, 'wer': wer, 'n_val': len(all_preds),
        'mean_cer': float(np.mean(cers)), 'median_cer': float(np.median(cers)),
        'std_cer': float(np.std(cers)),
        'checkpoint': args.ckpt,
    }
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # CER histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(cers, bins=30, alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(cers), color='red', linestyle='--',
               label=f'Mean={np.mean(cers):.3f}')
    ax.axvline(np.median(cers), color='orange', linestyle='--',
               label=f'Median={np.median(cers):.3f}')
    ax.set_xlabel('CER per sample')
    ax.set_ylabel('Count')
    ax.set_title('Conformer CTC - CER Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'cer_distribution.png'), dpi=150)
    plt.close()

    print(f"Saved to: {args.out}")


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default=os.path.join(_HERE, 'inference_output'))
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args)
