#!/usr/bin/env python3
"""
train_lm.py — train the from-scratch CharLM on TRAIN-SPLIT transcripts only.

No test text, no outside corpus: the LM learns English character statistics from
exactly the utterances the acoustic model trains on. Saves outputs/charlm.pt.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from config import CFG, OUT_DIR
import dataset as D
import lm as LM


def batches(seqs, bs, max_len, rng):
    order = rng.permutation(len(seqs))
    for i in range(0, len(seqs) - bs + 1, bs):
        chunk = [seqs[j] for j in order[i:i + bs]]
        L = min(max_len, max(len(s) for s in chunk))
        x = np.zeros((bs, L + 1), dtype=np.int64)             # BOS + chars
        for k, s in enumerate(chunk):
            s = s[:L]
            x[k, 0] = LM.LM_BOS
            x[k, 1:1 + len(s)] = s
        yield torch.from_numpy(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--max-len', type=int, default=300)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'charlm.pt'))
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rows = D.load_rows()
    tr_rows, te_rows = D.split_rows(rows)
    seqs = [LM.encode_text_lm(r['text']) for r in tr_rows]
    seqs = [s for s in seqs if len(s) > 1]
    print(f'[lm] train transcripts={len(seqs)}  (test held out, not seen)')

    model = LM.CharLM(d=args.d, layers=args.layers, max_len=args.max_len + 1).to(device)
    print(f'[lm] params={sum(p.numel() for p in model.parameters()):,}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.RandomState(0)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = n = 0.0
        for x in batches(seqs, args.batch, args.max_len, rng):
            x = x.to(device)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, LM.LM_VSIZE), x[:, 1:].reshape(-1),
                                   ignore_index=LM.LM_BOS)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); n += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f'  ep{ep} nats/char={tot/n:.3f} ppl={np.exp(tot/n):.2f} ({time.time()-t0:.0f}s)',
                  flush=True)
    torch.save({'model': model.state_dict(),
                'cfg': {'d': args.d, 'layers': args.layers, 'max_len': args.max_len + 1}}, args.out)
    print(f'[lm] saved → {args.out}')


if __name__ == '__main__':
    main()
