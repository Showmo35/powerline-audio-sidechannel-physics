#!/usr/bin/env python3
"""
overfit_test.py — can a fresh ViT MEMORIZE a tiny set of utterances per source?

Trains hard on N utts (no SpecAugment, no dropout effect via eval mode off) for
many epochs and reports train WER on those same utts. If real/null memorize
(→ ~0 %) but gen cannot, gen mels lack separable content (true result). If gen
also memorizes, the full-run failure was an optimization/regularization issue.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
import dataset as D, models as M, text as T


def run(src, rows, n, epochs, device):
    ds = D.MelText(rows[:n], src, train=False)          # deterministic (no aug)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, collate_fn=D.collate)
    mean, std = D.estimate_mel_stats(ds, n=n)
    model = M.build(CFG).to(device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    ctc = torch.nn.CTCLoss(blank=T.BLANK_IDX, zero_infinity=True)
    last = 0.0
    for ep in range(epochs):
        for b in dl:
            mel = ((b['mel'] - mean) / std).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                logits, out_lens = model(mel, b['mel_lens'])
                loss = ctc(F.log_softmax(logits.float(), -1), b['labels'].to(device),
                           out_lens.to(device), b['label_lengths'].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        last = loss.item()
    # train WER on the memorized set
    model.eval(); refs, hyps = [], []
    with torch.no_grad():
        for b in dl:
            mel = ((b['mel'] - mean) / std).to(device)
            logits, _ = model(mel, b['mel_lens'])
            hyps += T.ctc_greedy_decode(logits.float().cpu()); refs += b['texts']
    w = T.corpus_wer(refs, hyps)['wer'] * 100
    print(f'[overfit {src:5}] n={n} ep={epochs}  final_ctc={last:.3f}  TRAIN WER={w:.1f}%')
    for r, h in list(zip(refs, hyps))[:2]:
        print(f'    ref: {r[:70]}\n    hyp: {h[:70]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40); ap.add_argument('--epochs', type=int, default=200)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = D.load_rows(); tr, _ = D.split_rows(rows)
    torch.manual_seed(0); np.random.seed(0)
    for src in ('real', 'null', 'gen'):
        run(src, tr, args.n, args.epochs, device)


if __name__ == '__main__':
    main()
