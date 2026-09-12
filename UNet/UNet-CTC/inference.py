"""
inference.py
------------
Inference and evaluation for CTC model trained on UNet-predicted spectrograms.

Computes CER/WER, saves sample transcriptions and mel plots.
"""

import os, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import CTCEncoder, greedy_decode, IDX_TO_CHAR


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


def plot_mel_comparison(predicted, clean, noisy, pred_text, ref_text, idx, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))

    for ax, mel, title in zip(axes,
        [noisy[0], predicted[0], clean[0]],
        ['Noisy (bandpass)', 'UNet Predicted', 'Clean (MP3)']):
        ax.imshow(mel, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Mel bin')

    axes[-1].set_xlabel('Time frame')
    fig.suptitle(f"Sample {idx}\nREF: {ref_text[:80]}\nPRED: {pred_text[:80]}",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'sample_{idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load data
    print(f"Loading data from {args.data} ...")
    d = np.load(args.data)
    val_predicted = d['predicted_val']
    val_noisy     = d['noisy_val']
    val_clean     = d['clean_val']
    val_text      = d['text_val']
    val_tlen      = d['text_len_val']

    print(f"Val samples: {len(val_predicted)}")

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})

    model = CTCEncoder(
        n_mels=80,
        hidden=ckpt_args.get('gru_hidden', 256),
        n_layers=ckpt_args.get('gru_layers', 2),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.4f}, CER={ckpt.get('val_cer', 0):.4f})")

    # Run inference on all val
    all_preds, all_refs = [], []
    batch_size = args.batch

    with torch.no_grad():
        for i in range(0, len(val_predicted), batch_size):
            mels = torch.from_numpy(val_predicted[i:i+batch_size]).float().to(device)
            logits = model(mels)
            preds = greedy_decode(logits.cpu())
            refs  = decode_labels(
                val_text[i:i+batch_size], val_tlen[i:i+batch_size])
            all_preds.extend(preds)
            all_refs.extend(refs)

    cer = compute_cer(all_preds, all_refs)
    wer = compute_wer(all_preds, all_refs)

    print(f"\n{'='*60}")
    print(f"  CER: {cer:.4f}")
    print(f"  WER: {wer:.4f}")
    print(f"  Val samples: {len(all_preds)}")
    print(f"{'='*60}\n")

    # Save transcription comparison
    with open(os.path.join(args.out, 'transcriptions.txt'), 'w') as f:
        f.write(f"CER: {cer:.4f} | WER: {wer:.4f} | Samples: {len(all_preds)}\n")
        f.write('=' * 70 + '\n\n')
        for i, (p, r) in enumerate(zip(all_preds, all_refs)):
            sample_cer = edit_distance(p, r) / max(len(r), 1)
            f.write(f"--- Sample {i} (CER={sample_cer:.3f}) ---\n")
            f.write(f"REF : {r}\n")
            f.write(f"PRED: {p}\n\n")

    # Plot samples
    n_plot = min(args.n_samples, len(val_predicted))
    for i in range(n_plot):
        plot_mel_comparison(
            val_predicted[i], val_clean[i], val_noisy[i],
            all_preds[i], all_refs[i], i, args.out)

    # Save summary
    summary = {
        'cer': cer, 'wer': wer,
        'n_val': len(all_preds),
        'checkpoint': args.ckpt,
        'checkpoint_epoch': ckpt.get('epoch'),
        'checkpoint_val_loss': ckpt.get('val_loss'),
    }
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved to: {args.out}")
    print(f"  transcriptions.txt, results.json, {n_plot} mel plots")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default='inference_output')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args)
