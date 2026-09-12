"""
inference.py
------------
Module 8: Evaluate fine-tuned Whisper on the powerline val set.

Computes CER and WER, prints best/worst samples, saves results.json
and transcriptions.txt.

Usage:
    python inference.py \
        --data  data/powerline_whisper_data.npz \
        --ckpt  checkpoints/best_cer_model.pt \
        --out   inference_output
"""

import os, argparse, json, time
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# ---------------------------------------------------------------------------
# Metrics
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
# Main
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    print(f'Loading data from {args.data} ...')
    d          = np.load(args.data, allow_pickle=True)
    audio_val  = d['audio_val']
    text_val   = d['text_val']
    N          = len(audio_val)
    print(f'Val samples: {N}')

    # Load processor
    proc_dir = os.path.join(os.path.dirname(args.ckpt), 'processor')
    if os.path.isdir(proc_dir):
        processor = WhisperProcessor.from_pretrained(proc_dir)
    else:
        processor = WhisperProcessor.from_pretrained('openai/whisper-small.en')
    print('Processor loaded.')

    # Load model
    print(f'Loading checkpoint: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})
    base_model = ckpt_args.get('base_model', 'openai/whisper-small.en')

    model = WhisperForConditionalGeneration.from_pretrained(base_model).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    model.generation_config.forced_decoder_ids = None

    print(f'Model loaded (epoch {ckpt.get("epoch","?")}, '
          f'val_cer={ckpt.get("val_cer", 0):.4f})')

    # Run greedy inference on all val samples
    refs  = [str(t).lower().strip() for t in text_val]
    preds = []
    t0    = time.time()

    print(f'\nRunning inference on {N} samples (batch={args.batch}) ...')
    with torch.no_grad():
        for i in range(0, N, args.batch):
            batch_audio = audio_val[i: i + args.batch]
            inputs = processor(
                list(batch_audio), sampling_rate=16000,
                return_tensors='pt', padding=True,
            )
            input_features = inputs.input_features.to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    pred_ids = model.generate(
                        input_features,
                        max_new_tokens=225,
                        language='en',
                        task='transcribe',
                    )
            else:
                pred_ids = model.generate(
                    input_features,
                    max_new_tokens=225,
                )
            decoded = processor.batch_decode(pred_ids, skip_special_tokens=True)
            preds.extend([p.lower().strip() for p in decoded])

            if (i // args.batch) % 20 == 0:
                done = min(i + args.batch, N)
                print(f'  {done}/{N} ...', flush=True)

    elapsed = time.time() - t0
    cer = compute_cer(preds, refs)
    wer = compute_wer(preds, refs)

    print(f'\n{"="*60}')
    print(f'  Whisper fine-tuned on powerline (_real.bin 50-4kHz)')
    print(f'  Val samples : {N}')
    print(f'  CER         : {cer:.4f}')
    print(f'  WER         : {wer:.4f}')
    print(f'  Time        : {elapsed:.1f}s')
    print(f'{"="*60}')

    # Per-sample analysis
    sample_results = []
    for i in range(N):
        sc = edit_distance(preds[i], refs[i]) / max(len(refs[i]), 1)
        sample_results.append({'id': i, 'ref': refs[i],
                               'pred': preds[i], 'cer': sc})
    sample_results.sort(key=lambda x: x['cer'])

    # Save transcriptions.txt
    txt_path = os.path.join(args.out, 'transcriptions.txt')
    with open(txt_path, 'w') as f:
        f.write(f'CER: {cer:.4f} | WER: {wer:.4f} | N={N}\n')
        f.write('=' * 70 + '\n\n')
        f.write('--- WORST 10 ---\n\n')
        for r in sample_results[-10:]:
            f.write(f"Sample {r['id']} (CER={r['cer']:.3f})\n")
            f.write(f"  REF : {r['ref']}\n")
            f.write(f"  PRED: {r['pred']}\n\n")
        f.write('--- BEST 10 ---\n\n')
        for r in sample_results[:10]:
            f.write(f"Sample {r['id']} (CER={r['cer']:.3f})\n")
            f.write(f"  REF : {r['ref']}\n")
            f.write(f"  PRED: {r['pred']}\n\n")

    # Save results.json
    results = {
        'n_val':    N,
        'cer':      cer,
        'wer':      wer,
        'time_s':   elapsed,
        'base_model': base_model,
        'checkpoint': args.ckpt,
        'checkpoint_epoch': ckpt.get('epoch'),
        'checkpoint_val_cer': ckpt.get('val_cer'),
    }
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nSaved to: {args.out}')


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   required=True)
    parser.add_argument('--ckpt',   required=True)
    parser.add_argument('--out',    default=os.path.join(_HERE, 'inference_output'))
    parser.add_argument('--batch',  type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args)
