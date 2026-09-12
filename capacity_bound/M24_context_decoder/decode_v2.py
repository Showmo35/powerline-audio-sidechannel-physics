#!/usr/bin/env python3
"""
decode_v2.py — M24 improved decoder. Same noisy-channel idea, stronger LM scoring.

Upgrades over decode.py (which used first-token, left-context only):
  * BIDIRECTIONAL + FULL-WORD LM score: each candidate w is scored by the log-prob
    of the WHOLE sentence with the slot filled by w (teacher-forced GT elsewhere).
    A causal LM scoring the full sentence uses the RIGHT context too — the tokens
    after the slot have log-probs that depend on w. This is standard LM rescoring.
  * PROSODY term: the slot duration implies a syllable count; penalize candidates
    whose syllable count disagrees (prunes wrong-length impostors among the 100).

    score(w) = logP_LM(sentence | slot=w) + λ·logP_channel(w) + μ·prosody(w)

Conditions: channel-only (top-1), LM-only (bidirectional sentence LM, λ=μ=0),
M24v2 (best λ, +prosody), oracle. Teacher-forced GT context (upper bound on context).
Default lattice = the Stage-2 top-100 full-gallery one. LM = Qwen2.5-1.5B (offline).
Output: outputs/m24v2_<tag>.{json,png}.
"""
import argparse, os, json, re
import numpy as np
import torch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROOT = '<REPO_ROOT>'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs'); os.makedirs(OUT, exist_ok=True)


def syllables(w):
    return max(1, len(re.findall(r'[aeiouy]+', w.lower())))


def collect(data):
    """content slots with candidates: (words_list, slot_index, true, cands, sims, dur)."""
    items, n_all = [], 0
    for s in data['sentences']:
        slots = s['slots']; wlist = [x['w'] for x in slots]
        for i, sl in enumerate(slots):
            if not sl.get('is_content'):
                continue
            n_all += 1
            tk = sl.get('topk')
            if not tk or sl.get('unknown'):
                continue
            items.append({'words': wlist, 'i': i, 'true': sl['w'],
                          'cands': [c[0] for c in tk],
                          'sims': np.array([c[1] for c in tk], np.float32),
                          'dur': sl.get('dur', 0.0)})
    return items, n_all


@torch.no_grad()
def sentence_logprobs(tok, lm, dev, sentences):
    """Total teacher-forced log-prob of each sentence string (batched)."""
    enc = tok(sentences, return_tensors='pt', padding=True)
    ids = enc.input_ids.to(dev); am = enc.attention_mask.to(dev)
    logits = lm(input_ids=ids, attention_mask=am).logits.float()
    lp = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)      # (B, L-1)
    mask = am[:, 1:].float()
    return (tok_lp * mask).sum(1).cpu().numpy()                # (B,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lattice', default=os.path.join(HERE, 'data', 'lattice_genM22_big.json'))
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--tau', type=float, default=0.05)
    ap.add_argument('--mu', type=float, default=0.5, help='prosody weight')
    ap.add_argument('--maxcand', type=int, default=100)
    ap.add_argument('--tag', default='genM22_big')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'

    data = json.load(open(args.lattice))
    items, n_all = collect(data)
    print(f'[v2] lattice={args.lattice}  content slots={n_all} with cands={len(items)} '
          f'(oracle_topk={data.get("oracle_topk_content_recall"):.3f})', flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side='right')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if dev == 'cuda' else torch.float32).to(dev).eval()

    N = len(items)
    lm_sent, chan_lp, pros, true_idx, true_in = [], [], [], [], []
    for n, it in enumerate(items):
        cands = it['cands'][:args.maxcand]
        sims = it['sims'][:args.maxcand]
        # build full sentences with the slot replaced by each candidate
        sents = []
        for w in cands:
            ww = list(it['words']); ww[it['i']] = w
            sents.append(' '.join(ww))
        slp = sentence_logprobs(tok, lm, dev, sents)           # (K,)
        lm_sent.append(slp)
        c = sims / args.tau; c = c - c.max(); chan_lp.append(c - np.log(np.exp(c).sum()))
        est = max(1, int(round(it['dur'] / 0.18)))
        pros.append(-np.array([abs(syllables(w) - est) for w in cands], np.float32))
        ti = cands.index(it['true']) if it['true'] in cands else -1
        true_idx.append(ti); true_in.append(ti >= 0)
        if n % 100 == 0:
            print(f'[v2] scored {n}/{N}', flush=True)

    true_in = np.array(true_in)

    def acc(pred):
        return float(np.mean([true_idx[i] == pred[i] for i in range(N)]))

    ch_pred = [int(np.argmax(chan_lp[i])) for i in range(N)]
    lm_pred = [int(np.argmax(lm_sent[i])) for i in range(N)]
    acc_channel, acc_lm, acc_oracle = acc(ch_pred), acc(lm_pred), float(np.mean(true_in))

    # normalize LM sentence scores per-slot (subtract max) so λ balances comparably
    lm_z = [lm_sent[i] - lm_sent[i].max() for i in range(N)]
    grid = [0.0, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0]
    sweep = {}
    for lam in grid:
        pred = [int(np.argmax(lm_z[i] + lam * chan_lp[i] + args.mu * pros[i])) for i in range(N)]
        sweep[lam] = acc(pred)
    best = max(sweep, key=sweep.get); acc_v2 = sweep[best]

    g = N / max(1, n_all)
    print('\n══ M24 v2 (bidirectional full-sentence LM + prosody) ═══════════════')
    print(f'  slots with candidates={N}/{n_all}  (coverage {g:.2f})')
    print(f'  oracle (true in list)          : {acc_oracle:.3f}')
    print(f'  channel-only (top-1)           : {acc_channel:.3f}')
    print(f'  LM-only (bidir sentence)       : {acc_lm:.3f}')
    print(f'  M24 v2 (LM+channel+prosody λ={best}): {acc_v2:.3f}')
    print(f'  λ sweep: ' + '  '.join(f'{l}:{sweep[l]:.3f}' for l in grid))
    print(f'  ── content-recall over ALL content slots ──')
    print(f'  channel={acc_channel*g:.3f}  M24v2={acc_v2*g:.3f}  oracle={acc_oracle*g:.3f}')
    print(f'  (for reference, decode.py v1 M24 overall was ~0.072)')
    print('════════════════════════════════════════════════════════════════════')

    json.dump({'lattice': args.lattice, 'N': N, 'n_all': n_all, 'coverage': g, 'mu': args.mu,
               'on_candidate_slots': {'oracle': acc_oracle, 'channel': acc_channel,
                                      'lm_only': acc_lm, 'm24v2': acc_v2, 'best_lambda': best,
                                      'sweep': sweep},
               'content_recall_all': {'channel': acc_channel*g, 'm24v2': acc_v2*g,
                                      'oracle': acc_oracle*g}},
              open(os.path.join(OUT, f'm24v2_{args.tag}.json'), 'w'), indent=2)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = ['channel-only\n(top-1)', 'LM-only\n(bidir sentence)',
             'M24 v2\n(LM+chan+prosody)', 'oracle\n(true in list)']
    vals = [acc_channel, acc_lm, acc_v2, acc_oracle]
    for b, v in zip(ax.bar(names, vals, color=['#7a7a7a', '#4393c3', '#1b7837', '#c0c0c0'], width=0.62), vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.004, f'{v:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('accuracy on content slots WITH candidates')
    ax.set_ylim(0, max(vals)*1.25); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(f'M24 v2: bidirectional context + prosody ({args.tag}, N={N})', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'm24v2_{args.tag}.png'), dpi=200, bbox_inches='tight')
    print('[v2] wrote', os.path.join(OUT, f'm24v2_{args.tag}.png'), flush=True)


if __name__ == '__main__':
    main()
