"""
inference.py
------------
Module 9: Evaluate fine-tuned Whisper on UNet-enhanced mel val set.

Usage:
    python inference.py \
        --data  data/unet_whisper_data.npz \
        --ckpt  checkpoints/best_cer_model.pt \
        --out   inference_output
"""

import os, argparse, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dataset import UNetMelDataset, UNetWhisperCollator


def _edit(a, b):
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
        d += _edit(p, r); l += max(len(r), 1)
    return d / max(l, 1)

def compute_wer(preds, refs):
    d, l = 0, 0
    for p, r in zip(preds, refs):
        d += _edit(p.split(), r.split()); l += max(len(r.split()), 1)
    return d / max(l, 1)


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    print(f'Loading data: {args.data}')
    d       = np.load(args.data, allow_pickle=True)
    mel_val = d['mel_val']
    txt_val = d['text_val']
    N       = len(mel_val)
    print(f'Val samples: {N}')

    # Load processor
    proc_dir = os.path.join(os.path.dirname(args.ckpt), 'processor')
    processor = WhisperProcessor.from_pretrained(
        proc_dir if os.path.isdir(proc_dir) else 'openai/whisper-small.en')

    # Load model
    print(f'Loading checkpoint: {args.ckpt}')
    ckpt      = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})
    model     = WhisperForConditionalGeneration.from_pretrained(
                    ckpt_args.get('base_model', 'openai/whisper-small.en')).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    model.generation_config.forced_decoder_ids = None
    print(f'  epoch {ckpt.get("epoch","?")}  val_cer={ckpt.get("val_cer",0):.4f}')

    # Inference
    collator = UNetWhisperCollator()
    val_ds   = UNetMelDataset(mel_val, txt_val, processor)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=collator)

    refs, preds = [str(t).lower().strip() for t in txt_val], []
    t0 = time.time()
    print(f'\nRunning inference on {N} samples ...')

    with torch.no_grad():
        for i, batch in enumerate(val_dl):
            feats = batch['input_features'].to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    ids = model.generate(feats, max_new_tokens=225)
            else:
                ids = model.generate(feats, max_new_tokens=225)
            dec = processor.batch_decode(ids, skip_special_tokens=True)
            preds.extend([p.lower().strip() for p in dec])
            if i % 10 == 0:
                print(f'  {min((i+1)*args.batch, N)}/{N} ...', flush=True)

    elapsed = time.time() - t0
    cer = compute_cer(preds, refs)
    wer = compute_wer(preds, refs)

    print(f'\n{"="*62}')
    print(f'  Module 9: Perceptual-Loss UNet + Whisper fine-tuned')
    print(f'  Val samples : {N}')
    print(f'  CER         : {cer:.4f}')
    print(f'  WER         : {wer:.4f}')
    print(f'  Time        : {elapsed:.1f}s')
    print(f'{"="*62}')

    # Per-sample analysis
    samples = sorted([
        {'id': i, 'ref': refs[i], 'pred': preds[i],
         'cer': _edit(preds[i], refs[i]) / max(len(refs[i]), 1)}
        for i in range(N)
    ], key=lambda x: x['cer'])

    with open(os.path.join(args.out, 'transcriptions.txt'), 'w') as f:
        f.write(f'CER: {cer:.4f} | WER: {wer:.4f} | N={N}\n')
        f.write('=' * 70 + '\n\n')
        f.write('--- WORST 10 ---\n\n')
        for r in samples[-10:]:
            f.write(f"Sample {r['id']} (CER={r['cer']:.3f})\n")
            f.write(f"  REF : {r['ref']}\n  PRED: {r['pred']}\n\n")
        f.write('--- BEST 10 ---\n\n')
        for r in samples[:10]:
            f.write(f"Sample {r['id']} (CER={r['cer']:.3f})\n")
            f.write(f"  REF : {r['ref']}\n  PRED: {r['pred']}\n\n")

    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump({'n_val': N, 'cer': cer, 'wer': wer, 'time_s': elapsed,
                   'checkpoint': args.ckpt,
                   'checkpoint_epoch': ckpt.get('epoch'),
                   'checkpoint_val_cer': ckpt.get('val_cer')}, f, indent=2)

    print(f'Saved to: {args.out}')


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   required=True)
    parser.add_argument('--ckpt',   required=True)
    parser.add_argument('--out',    default=os.path.join(_HERE, 'inference_output'))
    parser.add_argument('--batch',  type=int, default=16)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args)
