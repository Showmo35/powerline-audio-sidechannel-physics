#!/usr/bin/env python3
"""
decode.py — decode one trained condition's test set three ways and report WER.

  greedy          : CTC argmax collapse                          (acoustic only)
  beam            : CTC prefix beam search, no LM
  beam+LM         : + from-scratch CharLM shallow fusion         (rigorous headline)

Also dumps per-utterance hypotheses (nbest.json) so llm_agent.py can run the
pretrained-LLM arm on the same emissions. Use --limit to cap test utts for the
(slow) beam passes.
"""
import argparse, json, os
import torch

from config import CFG, OUT_DIR
import dataset as D
import models as M
import lm as LM
import text as T


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', choices=('gen', 'real', 'null'), required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--charlm', default=os.path.join(OUT_DIR, 'charlm.pt'))
    ap.add_argument('--limit', type=int, default=600)
    ap.add_argument('--beam', type=int, default=24)
    ap.add_argument('--alpha', type=float, default=0.4)
    ap.add_argument('--beta', type=float, default=1.0)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.join(OUT_DIR, f'vit_{args.input}')
    ckpt = args.ckpt or os.path.join(out_dir, 'best.pt')

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    mean, std = ck['mean'], ck['std']
    model = M.build(CFG).to(device); model.load_state_dict(ck['model']); model.eval()

    charlm = None
    if os.path.exists(args.charlm):
        lc = torch.load(args.charlm, map_location=device, weights_only=False)
        charlm = LM.CharLM(**lc['cfg']).to(device); charlm.load_state_dict(lc['model']); charlm.eval()

    rows = D.load_rows(); _, te_rows = D.split_rows(rows)
    te_rows = te_rows[:args.limit]
    te_ds = D.MelText(te_rows, args.input, train=False)
    dl = torch.utils.data.DataLoader(te_ds, batch_size=1, shuffle=False, collate_fn=D.collate)

    refs, g_hyps, b_hyps, l_hyps, dump = [], [], [], [], []
    for batch in dl:
        mel = ((batch['mel'] - mean) / std).to(device)
        logits, out_lens = model(mel, batch['mel_lens'])
        logp = torch.log_softmax(logits.float(), -1)[:int(out_lens[0]), 0].cpu()
        ref = batch['texts'][0]
        g = LM.greedy(logp)
        b = LM.beam_search(logp, lm=None, beam=args.beam)
        l = LM.beam_search(logp, lm=charlm, device=device, beam=args.beam,
                           alpha=args.alpha, beta=args.beta) if charlm else b
        refs.append(ref); g_hyps.append(g); b_hyps.append(b); l_hyps.append(l)
        dump.append({'utt_id': batch['utt_ids'][0], 'ref': ref, 'greedy': g, 'beam': b, 'beam_lm': l})

    def wer(h):
        return T.corpus_wer(refs, h)['wer'] * 100
    res = {'input': args.input, 'n': len(refs), 'limit': args.limit,
           'wer_greedy': wer(g_hyps), 'wer_beam': wer(b_hyps), 'wer_beam_lm': wer(l_hyps)}
    print(f'[decode:{args.input}] n={res["n"]}  greedy={res["wer_greedy"]:.1f}%  '
          f'beam={res["wer_beam"]:.1f}%  beam+LM={res["wer_beam_lm"]:.1f}%', flush=True)
    json.dump(res, open(os.path.join(out_dir, 'decode.json'), 'w'), indent=1)
    json.dump(dump, open(os.path.join(out_dir, 'nbest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
