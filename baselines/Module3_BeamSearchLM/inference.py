"""
inference.py
------------
Evaluation with greedy, beam search, and beam search + LM decoding.
Evaluates ALL validation samples. Computes CER/WER for each method.

Usage:
    python inference.py --data data/transcribe_data.npz \
                        --ckpt checkpoints/best_cer_model.pt \
                        --lm data/lm.pkl --out inference_output
"""

import os, argparse, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import CTCEncoder, greedy_decode, IDX_TO_CHAR, BLANK_IDX
from beam_search import ctc_beam_search_decode_batch
from train_lm import CharNgramLM


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
    print(f"Loading data from {args.data} ...")
    d = np.load(args.data)
    val_predicted = d['predicted_val']
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
          f"val_cer={ckpt.get('val_cer', 0):.4f})")

    # Load LM if provided
    lm = None
    if args.lm and os.path.exists(args.lm):
        print(f"Loading LM from {args.lm} ...")
        lm = CharNgramLM.load(args.lm)
        print(f"  LM loaded (order={lm.order})")

    # Decode references
    all_refs = decode_labels(val_text, val_tlen)

    # ── Method 1: Greedy decode ──────────────────────────────────────────────
    print("\n>>> Greedy decoding ...")
    t0 = time.time()
    greedy_preds = []
    with torch.no_grad():
        for i in range(0, len(val_predicted), args.batch):
            mels = torch.from_numpy(val_predicted[i:i+args.batch]).float().to(device)
            logits = model(mels)
            preds = greedy_decode(logits.cpu())
            greedy_preds.extend(preds)
    greedy_cer = compute_cer(greedy_preds, all_refs)
    greedy_wer = compute_wer(greedy_preds, all_refs)
    greedy_time = time.time() - t0
    print(f"    CER: {greedy_cer:.4f}  WER: {greedy_wer:.4f}  ({greedy_time:.1f}s)")

    # ── Method 2: Beam search (no LM) ───────────────────────────────────────
    print(f"\n>>> Beam search (width={args.beam_width}, no LM) ...")
    t0 = time.time()
    beam_preds = []
    with torch.no_grad():
        for i in range(0, len(val_predicted), args.batch):
            mels = torch.from_numpy(val_predicted[i:i+args.batch]).float().to(device)
            logits = model(mels)
            log_probs = logits.log_softmax(dim=-1)
            preds = ctc_beam_search_decode_batch(
                log_probs.cpu(), beam_width=args.beam_width)
            beam_preds.extend(preds)
    beam_cer = compute_cer(beam_preds, all_refs)
    beam_wer = compute_wer(beam_preds, all_refs)
    beam_time = time.time() - t0
    print(f"    CER: {beam_cer:.4f}  WER: {beam_wer:.4f}  ({beam_time:.1f}s)")

    # ── Method 3: Beam search + LM ──────────────────────────────────────────
    beam_lm_cer, beam_lm_wer, beam_lm_time = None, None, None
    beam_lm_preds = []
    if lm is not None:
        print(f"\n>>> Beam search (width={args.beam_width}, "
              f"LM weight={args.lm_weight}) ...")
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(val_predicted), args.batch):
                mels = torch.from_numpy(val_predicted[i:i+args.batch]).float().to(device)
                logits = model(mels)
                log_probs = logits.log_softmax(dim=-1)
                preds = ctc_beam_search_decode_batch(
                    log_probs.cpu(), beam_width=args.beam_width,
                    lm=lm, lm_weight=args.lm_weight)
                beam_lm_preds.extend(preds)
        beam_lm_cer = compute_cer(beam_lm_preds, all_refs)
        beam_lm_wer = compute_wer(beam_lm_preds, all_refs)
        beam_lm_time = time.time() - t0
        print(f"    CER: {beam_lm_cer:.4f}  WER: {beam_lm_wer:.4f}  "
              f"({beam_lm_time:.1f}s)")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {'Method':<25} {'CER':>8} {'WER':>8} {'Time':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Greedy':<25} {greedy_cer:>8.4f} {greedy_wer:>8.4f} "
          f"{greedy_time:>7.1f}s")
    print(f"  {f'Beam (w={args.beam_width})':<25} {beam_cer:>8.4f} "
          f"{beam_wer:>8.4f} {beam_time:>7.1f}s")
    if beam_lm_cer is not None:
        print(f"  {f'Beam+LM (w={args.beam_width})':<25} {beam_lm_cer:>8.4f} "
              f"{beam_lm_wer:>8.4f} {beam_lm_time:>7.1f}s")
    print(f"{'='*60}")

    # ── Save results ─────────────────────────────────────────────────────────
    # Per-sample CER analysis
    sample_results = []
    for i, ref in enumerate(all_refs):
        entry = {
            'id': i,
            'ref': ref,
            'greedy_pred': greedy_preds[i],
            'greedy_cer': edit_distance(greedy_preds[i], ref) / max(len(ref), 1),
            'beam_pred': beam_preds[i],
            'beam_cer': edit_distance(beam_preds[i], ref) / max(len(ref), 1),
        }
        if beam_lm_preds:
            entry['beam_lm_pred'] = beam_lm_preds[i]
            entry['beam_lm_cer'] = edit_distance(beam_lm_preds[i], ref) / max(len(ref), 1)
        sample_results.append(entry)

    # Sort by beam CER for error analysis
    sample_results.sort(key=lambda x: x['beam_cer'])

    # Save transcriptions
    with open(os.path.join(args.out, 'transcriptions.txt'), 'w') as f:
        f.write(f"Greedy CER: {greedy_cer:.4f} | WER: {greedy_wer:.4f}\n")
        f.write(f"Beam   CER: {beam_cer:.4f} | WER: {beam_wer:.4f}\n")
        if beam_lm_cer is not None:
            f.write(f"Beam+LM CER: {beam_lm_cer:.4f} | WER: {beam_lm_wer:.4f}\n")
        f.write('=' * 70 + '\n\n')
        f.write("--- WORST 10 (by beam CER) ---\n\n")
        for r in sample_results[-10:]:
            f.write(f"Sample {r['id']} (CER greedy={r['greedy_cer']:.3f} "
                    f"beam={r['beam_cer']:.3f})\n")
            f.write(f"  REF:    {r['ref']}\n")
            f.write(f"  GREEDY: {r['greedy_pred']}\n")
            f.write(f"  BEAM:   {r['beam_pred']}\n\n")
        f.write("\n--- BEST 10 (by beam CER) ---\n\n")
        for r in sample_results[:10]:
            f.write(f"Sample {r['id']} (CER greedy={r['greedy_cer']:.3f} "
                    f"beam={r['beam_cer']:.3f})\n")
            f.write(f"  REF:    {r['ref']}\n")
            f.write(f"  GREEDY: {r['greedy_pred']}\n")
            f.write(f"  BEAM:   {r['beam_pred']}\n\n")

    # Save summary JSON
    summary = {
        'n_val': len(all_refs),
        'greedy': {'cer': greedy_cer, 'wer': greedy_wer, 'time_s': greedy_time},
        'beam': {'cer': beam_cer, 'wer': beam_wer, 'time_s': beam_time,
                 'beam_width': args.beam_width},
    }
    if beam_lm_cer is not None:
        summary['beam_lm'] = {
            'cer': beam_lm_cer, 'wer': beam_lm_wer, 'time_s': beam_lm_time,
            'beam_width': args.beam_width, 'lm_weight': args.lm_weight,
        }

    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # CER distribution histogram
    greedy_cers = [r['greedy_cer'] for r in sample_results]
    beam_cers   = [r['beam_cer'] for r in sample_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(greedy_cers, bins=30, alpha=0.6, label=f'Greedy (mean={np.mean(greedy_cers):.3f})')
    ax.hist(beam_cers, bins=30, alpha=0.6, label=f'Beam (mean={np.mean(beam_cers):.3f})')
    ax.set_xlabel('CER per sample')
    ax.set_ylabel('Count')
    ax.set_title('CER Distribution: Greedy vs Beam Search')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'cer_distribution.png'), dpi=150)
    plt.close()

    print(f"\nSaved to: {args.out}")


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default=os.path.join(_HERE, 'inference_output'))
    parser.add_argument('--lm', default=None, help='Path to LM pickle file')
    parser.add_argument('--beam_width', type=int, default=50)
    parser.add_argument('--lm_weight', type=float, default=0.3)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args)
