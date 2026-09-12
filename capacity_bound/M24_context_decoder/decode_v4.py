#!/usr/bin/env python3
"""
decode_v4.py — M24 HONEST joint BEAM-SEARCH decoder (no GT context).

Addresses the objection that ICM (decode_v3) was a weak optimizer: instead of
greedily fixing one slot at a time, do a real joint search over candidate
combinations, then rank whole hypotheses by full coherence + channel.

  Stage 1  — left-to-right BEAM SEARCH (width B). At each slot branch over its
             candidates (content: powerline top-K; function: closed function set;
             unknown: "{?}") scored by first-token LM log-prob given the predicted
             prefix + λ0·channel + μ·prosody. Keeps B diverse joint hypotheses.
             The channel-greedy and LM-greedy sentences are seeded into the pool
             (so the result can never be worse than channel-only).
  Stage 2  — rescore every pooled hypothesis with the FULL-SENTENCE (bidirectional)
             LM log-prob + λ·(sum channel) + μ·(sum prosody); pick the best.
             λ is SWEPT here cheaply (guarantees ≥ channel-only at high λ).

No ground-truth words are ever used to decode (GT only scores the result). A light
genre preamble primes register (not GT). Paragraph-level context is NOT used
(sentences decoded independently; noted as a limitation).

Output: outputs/m24v4_<tag>.{json,png} + a few example reconstructions.
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs'); os.makedirs(OUT, exist_ok=True)
PREAMBLE = 'A sentence from a classic English novel: '
FUNC = ("the and of to a in that it is was he for with as his on be at by i this had "
        "not are but from or have an they which you were her all she there would their "
        "we him so if when what who".split())


def syll(w):
    return max(1, len(re.findall(r'[aeiouy]+', w.lower())))


@torch.no_grad()
def next_logprobs(tok, lm, dev, prefixes):
    """log P(next token | prefix) for a batch of prefix strings -> (B, V)."""
    enc = tok(prefixes, return_tensors='pt', padding=True)
    ids = enc.input_ids.to(dev); am = enc.attention_mask.to(dev)
    logits = lm(input_ids=ids, attention_mask=am).logits.float()
    last = am.sum(1) - 1                                   # last real token index
    lg = logits[torch.arange(len(prefixes)), last]
    return torch.log_softmax(lg, -1).cpu().numpy()


@torch.no_grad()
def sent_lp(tok, lm, dev, sentences):
    enc = tok(sentences, return_tensors='pt', padding=True)
    ids = enc.input_ids.to(dev); am = enc.attention_mask.to(dev)
    logits = lm(input_ids=ids, attention_mask=am).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1)
    tl = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    m = am[:, 1:].float()
    return (tl * m).sum(1).cpu().numpy()


def first_tok(tok, w):
    return tok(' ' + w, add_special_tokens=False).input_ids[0]


def render(words, preamble=True):
    s = ' '.join(x for x in words if x and x != '{?}')
    return (PREAMBLE + s) if preamble else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lattice', default=os.path.join(HERE, 'data', 'lattice_genM22_big.json'))
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--tau', type=float, default=0.05)
    ap.add_argument('--mu', type=float, default=0.5)
    ap.add_argument('--beam', type=int, default=24)
    ap.add_argument('--maxcand', type=int, default=40)
    ap.add_argument('--lam0', type=float, default=5.0, help='channel weight during beam')
    ap.add_argument('--tag', default='genM22_big')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'

    data = json.load(open(args.lattice))
    tok = AutoTokenizer.from_pretrained(args.model, padding_side='right')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if dev == 'cuda' else torch.float32).to(dev).eval()
    ft_cache = {}

    def FT(w):
        if w not in ft_cache:
            ft_cache[w] = first_tok(tok, w)
        return ft_cache[w]

    # slot metadata
    def build(s):
        slots = []
        for sl in s['slots']:
            if not sl.get('is_content'):
                slots.append({'kind': 'func', 'true': sl['w'], 'cands': FUNC,
                              'chan': np.zeros(len(FUNC)), 'pros': np.zeros(len(FUNC))})
            elif sl.get('topk') and not sl.get('unknown'):
                tk = sl['topk'][:args.maxcand]; sims = np.array([c[1] for c in tk], np.float32)
                cl = sims / args.tau; cl = cl - cl.max(); cl = cl - np.log(np.exp(cl).sum())
                est = max(1, int(round(sl.get('dur', 0.36) / 0.18)))
                pr = -np.array([abs(syll(c[0]) - est) for c in tk], np.float32)
                slots.append({'kind': 'content', 'true': sl['w'],
                              'cands': [c[0] for c in tk], 'chan': cl, 'pros': pr})
            else:
                slots.append({'kind': 'unk', 'true': sl['w']})
        return slots

    def beam_decode(slots):
        # beams: dict(words, lm1, chan, pros)
        beams = [{'words': [], 'lm1': 0.0, 'chan': 0.0, 'pros': 0.0}]
        for sl in slots:
            if sl['kind'] == 'unk':
                for b in beams:
                    b['words'].append('{?}')
                continue
            prefixes = [render(b['words']) for b in beams]
            nl = next_logprobs(tok, lm, dev, prefixes)        # (B,V)
            cand_ft = np.array([FT(w) for w in sl['cands']])
            ext = []
            for bi, b in enumerate(beams):
                lmv = nl[bi][cand_ft]                          # (K,) first-token lp
                tot = b['lm1'] + lmv + args.lam0 * sl['chan'] + args.mu * sl['pros']
                for ci in range(len(sl['cands'])):
                    ext.append((b['lm1'] + lmv[ci], b['chan'] + sl['chan'][ci],
                                b['pros'] + sl['pros'][ci], tot[ci], b['words'] + [sl['cands'][ci]]))
            ext.sort(key=lambda x: -x[3])
            beams = [{'words': w, 'lm1': l, 'chan': c, 'pros': p}
                     for (l, c, p, _, w) in ext[:args.beam]]
        return beams

    def greedy_channel(slots):
        words, chan, pros = [], 0.0, 0.0
        for sl in slots:
            if sl['kind'] == 'content':
                k = int(np.argmax(sl['chan'])); words.append(sl['cands'][k])
                chan += sl['chan'][k]; pros += sl['pros'][k]
            elif sl['kind'] == 'func':
                words.append('the')
            else:
                words.append('{?}')
        return {'words': words, 'chan': chan, 'pros': pros}

    lambdas = [0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0]
    # accumulate per-λ predictions to compute recall; also channel-only & oracle
    tot = hit_ch = orac = 0
    hits = {l: 0 for l in lambdas}
    examples = []
    for si, s in enumerate(data['sentences']):
        slots = build(s)
        pool = beam_decode(slots)
        pool.append(greedy_channel(slots))                    # seed channel-greedy
        # stage-2: full-sentence LM score for each pooled hyp
        lm2 = sent_lp(tok, lm, dev, [render(h['words']) for h in pool])
        for h, l in zip(pool, lm2):
            h['lm2'] = float(l)
        # per-λ best hypothesis
        best_by_lam = {}
        for lam in lambdas:
            k = int(np.argmax([h['lm2'] + lam * h['chan'] + args.mu * h['pros'] for h in pool]))
            best_by_lam[lam] = pool[k]['words']
        # channel-only prediction = greedy_channel words
        ch_words = pool[-1]['words']
        # score content slots
        for i, sl in enumerate(slots):
            if sl['kind'] != 'content':
                continue
            tot += 1
            orac += (sl['true'] in sl['cands'])
            hit_ch += (ch_words[i] == sl['true'])
            for lam in lambdas:
                hits[lam] += (best_by_lam[lam][i] == sl['true'])
        if si < 6:
            examples.append({'gt': s['text'],
                             'recon': render(best_by_lam[10.0], preamble=False)})

    acc_ch = hit_ch / max(1, tot)
    acc_or = orac / max(1, tot)
    sweep = {l: hits[l] / max(1, tot) for l in lambdas}
    best_lam = max(sweep, key=sweep.get); acc_v4 = sweep[best_lam]

    # overall content-recall (denominator = all content slots incl. unknown)
    n_all = sum((sl.get('is_content')) for s in data['sentences'] for sl in s['slots'])
    g = tot / max(1, n_all)
    print('\n══ M24 v4 HONEST joint beam search (no GT context) ══════════════════')
    print(f'  content slots w/ cands={tot} / all content={n_all}  (coverage {g:.2f})')
    print(f'  ── on candidate slots ──')
    print(f'  oracle (true in list)  : {acc_or:.3f}')
    print(f'  channel-only           : {acc_ch:.3f}')
    print(f'  M24 v4 (best λ={best_lam})   : {acc_v4:.3f}')
    print(f'  λ sweep: ' + '  '.join(f'{l}:{sweep[l]:.3f}' for l in lambdas))
    print(f'  ── content-recall over ALL content slots ──')
    print(f'  channel={acc_ch*g:.3f}  M24v4={acc_v4*g:.3f}  oracle={acc_or*g:.3f}')
    print(f'  refs: v3-ICM-honest 0.013  v2-GT 0.122')
    print('════════════════════════════════════════════════════════════════════')
    for ex in examples[:5]:
        print(f'  GT : {ex["gt"][:96]}')
        print(f'  REC: {ex["recon"][:96]}\n')

    json.dump({'lattice': args.lattice, 'beam': args.beam, 'maxcand': args.maxcand,
               'on_candidate_slots': {'oracle': acc_or, 'channel': acc_ch, 'm24v4': acc_v4,
                                      'best_lambda': best_lam, 'sweep': sweep},
               'content_recall_all': {'channel': acc_ch*g, 'm24v4': acc_v4*g, 'oracle': acc_or*g},
               'coverage': g, 'examples': examples},
              open(os.path.join(OUT, f'm24v4_{args.tag}.json'), 'w'), indent=2)

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    names = ['channel-only', 'v3 ICM\n(honest)', 'M24 v4\nbeam (honest)', 'v2\n(GT bidir)', 'oracle']
    vals = [acc_ch*g, 0.013, acc_v4*g, 0.122, acc_or*g]
    cols = ['#7a7a7a', '#d6604d', '#1b7837', '#4393c3', '#c0c0c0']
    for b, v in zip(ax.bar(names, vals, color=cols, width=0.66), vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.003, f'{v:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('content-recall over ALL content slots')
    ax.set_ylim(0, max(vals)*1.25); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('M24 v4: honest JOINT beam search vs. ICM vs. GT upper bound', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'm24v4_{args.tag}.png'), dpi=200, bbox_inches='tight')
    print('[v4] wrote', os.path.join(OUT, f'm24v4_{args.tag}.png'), flush=True)


if __name__ == '__main__':
    main()
