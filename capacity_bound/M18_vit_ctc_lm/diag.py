#!/usr/bin/env python3
"""diag.py — train-vs-test greedy WER for one condition (overfitting check)."""
import argparse, os
import torch
from config import CFG, OUT_DIR
import dataset as D, models as M, text as T


@torch.no_grad()
def wer_on(model, rows, src, mean, std, device, n):
    ds = D.MelText(rows[:n], src, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=D.collate)
    refs, hyps = [], []
    for b in dl:
        mel = ((b['mel'] - mean) / std).to(device)
        logits, _ = model(mel, b['mel_lens'])
        hyps += T.ctc_greedy_decode(logits.float().cpu()); refs += b['texts']
    cer = sum(T._edit_distance(list(r), list(h)) for r, h in zip(refs, hyps)) / max(1, sum(len(r) for r in refs))
    return T.corpus_wer(refs, hyps)['wer'] * 100, cer * 100, list(zip(refs, hyps))[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True); ap.add_argument('--n', type=int, default=160)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(os.path.join(OUT_DIR, f'vit_{args.input}', 'best.pt'),
                    map_location=device, weights_only=False)
    model = M.build(CFG).to(device); model.load_state_dict(ck['model']); model.eval()
    rows = D.load_rows(); tr, te = D.split_rows(rows)
    for name, rr in (('TRAIN', tr), ('TEST', te)):
        w, c, samp = wer_on(model, rr, args.input, ck['mean'], ck['std'], device, args.n)
        print(f'[{args.input}:{name}] n={args.n}  WER={w:.1f}%  CER={c:.1f}%')
        for r, h in samp:
            print(f'    ref: {r[:80]}\n    hyp: {h[:80]}')


if __name__ == '__main__':
    main()
