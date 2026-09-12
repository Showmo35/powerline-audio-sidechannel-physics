#!/usr/bin/env python3
"""
decode_v3.py — M24 HONEST attack decoder. No ground-truth context anywhere.

Everything the decoder conditions on is either the powerline evidence or its OWN
predictions — never the true surrounding words. This is the attack-grade number.

Method: joint decoding by iterative refinement (coordinate ascent / ICM).
  init : content slots (with candidates) = powerline top-1 ; function slots = "the" ;
         content slots with no evidence = "{?}".
  refine (T passes): visit each slot in order and re-pick the value that maximizes
         logP_LM(current sentence with this slot = w) + λ·logP_channel + μ·prosody,
         where the "current sentence" uses the OTHER slots' CURRENT PREDICTIONS
         (bidirectional, but from predicted — not GT — neighbours).
         content slot  -> choose among its powerline candidates (channel + LM + prosody)
         function slot -> choose among a small closed function-word set (LM only;
                          the attacker guesses grammatical glue, no GT)
         unknown slot  -> left as "{?}" (no evidence, counts as a miss)

Eval: content-recall = fraction of GT content words recovered (GT used ONLY to score,
never to decode). Compared against channel-only, the v2 teacher-forced upper bound,
and the oracle. Default lattice = Stage-2 top-100 full-gallery. LM = Qwen2.5-1.5B.
Output: outputs/m24v3_<tag>.{json,png}.
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

FUNC = ("the and of to a in that it is was he for with as his on be at by i this had "
        "not are but from or have an they which you were her all she there would their "
        "we him so if when what who".split())


def syllables(w):
    return max(1, len(re.findall(r'[aeiouy]+', w.lower())))


@torch.no_grad()
def sent_lp(tok, lm, dev, sentences):
    enc = tok(sentences, return_tensors='pt', padding=True)
    ids = enc.input_ids.to(dev); am = enc.attention_mask.to(dev)
    logits = lm(input_ids=ids, attention_mask=am).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1)
    tl = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    m = am[:, 1:].float()
    return (tl * m).sum(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lattice', default=os.path.join(HERE, 'data', 'lattice_genM22_big.json'))
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--tau', type=float, default=0.05)
    ap.add_argument('--lam', type=float, default=1.0)     # channel weight (v2 best)
    ap.add_argument('--mu', type=float, default=0.5)      # prosody weight
    ap.add_argument('--maxcand', type=int, default=50)
    ap.add_argument('--passes', type=int, default=4)
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

    # ── per-sentence slot metadata (NO GT words used for decoding) ────────────
    sents = []
    for s in data['sentences']:
        slots = []
        for sl in s['slots']:
            if not sl.get('is_content'):
                slots.append({'kind': 'func', 'true': sl['w']})
            elif sl.get('topk') and not sl.get('unknown'):
                tk = sl['topk'][:args.maxcand]
                sims = np.array([c[1] for c in tk], np.float32)
                cl = sims / args.tau; cl = cl - cl.max(); cl = cl - np.log(np.exp(cl).sum())
                est = max(1, int(round(sl.get('dur', 0.36) / 0.18)))
                pr = -np.array([abs(syllables(c[0]) - est) for c in tk], np.float32)
                slots.append({'kind': 'content', 'true': sl['w'],
                              'cands': [c[0] for c in tk], 'chan': cl, 'pros': pr})
            else:
                slots.append({'kind': 'unk', 'true': sl['w']})
        sents.append(slots)

    n_content = sum(sl['kind'] == 'content' for s in sents for sl in s)

    # ── init predictions ──
    for s in sents:
        for sl in s:
            sl['pred'] = (sl['cands'][int(np.argmax(sl['chan']))] if sl['kind'] == 'content'
                          else 'the' if sl['kind'] == 'func' else '{?}')

    def render(slots, i, w):
        out = []
        for j, sl in enumerate(slots):
            x = w if j == i else sl['pred']
            if x and x != '{?}':
                out.append(x)
        return ' '.join(out)

    def content_recall():
        hit = tot = 0
        for s in sents:
            for sl in s:
                if sl['kind'] == 'content':
                    tot += 1; hit += (sl['pred'] == sl['true'])
        return hit / max(1, tot)

    # ── iterative refinement (predicted context only) ─────────────────────────
    hist = []
    for p in range(args.passes):
        for s in sents:
            for i, sl in enumerate(s):
                if sl['kind'] == 'unk':
                    continue
                cands = sl['cands'] if sl['kind'] == 'content' else FUNC
                sc = sent_lp(tok, lm, dev, [render(s, i, w) for w in cands])
                if sl['kind'] == 'content':
                    sc = sc + args.lam * sl['chan'] + args.mu * sl['pros']
                sl['pred'] = cands[int(np.argmax(sc))]
        r = content_recall(); hist.append(r)
        print(f'[v3] pass {p+1}/{args.passes}  content-recall(cand slots)={r:.3f}', flush=True)

    # accuracy restricted to content slots that HAVE candidates (comparable to v2)
    hit = tot = 0; orac = 0
    for s in sents:
        for sl in s:
            if sl['kind'] == 'content':
                tot += 1; hit += (sl['pred'] == sl['true']); orac += (sl['true'] in sl['cands'])
    acc_cand = hit / max(1, tot); acc_oracle = orac / max(1, tot)
    g = tot / max(1, n_content)   # here tot==n_content (all content w/ cands are 'content')
    # overall content-recall over ALL content slots (unk slots are automatic misses)
    n_all_content = sum(sl['kind'] in ('content', 'unk') for s in sents for sl in s)
    overall = hit / max(1, n_all_content)
    orac_overall = orac / max(1, n_all_content)

    print('\n══ M24 v3 HONEST attack (no GT context, joint ICM decode) ═══════════')
    print(f'  content slots w/ candidates = {tot}  (of {n_all_content} content slots)')
    print(f'  ── on candidate slots ──')
    print(f'  oracle (true in list)   : {acc_oracle:.3f}')
    print(f'  M24 v3 HONEST           : {acc_cand:.3f}   (v2 teacher-forced was 0.199)')
    print(f'  ── content-recall over ALL content slots ──')
    print(f'  M24 v3 HONEST={overall:.3f}   oracle={orac_overall:.3f}')
    print(f'  refs: channel-only~0.020  v1(GT)~0.072  v2(GT bidir)~0.122')
    print(f'  pass history: ' + ' '.join(f'{h:.3f}' for h in hist))
    print('════════════════════════════════════════════════════════════════════')

    json.dump({'lattice': args.lattice, 'passes': args.passes, 'lam': args.lam, 'mu': args.mu,
               'n_content_all': n_all_content, 'n_cand_slots': tot,
               'on_candidate_slots': {'oracle': acc_oracle, 'm24_v3_honest': acc_cand},
               'content_recall_all': {'m24_v3_honest': overall, 'oracle': orac_overall},
               'pass_history': hist,
               'refs': {'channel_only': 0.020, 'v1_gt': 0.072, 'v2_gt_bidir': 0.122}},
              open(os.path.join(OUT, f'm24v3_{args.tag}.json'), 'w'), indent=2)

    # ── figure: the honest number in the full arc (overall content-recall) ────
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    names = ['channel-only', 'M24 v1\n(GT ctx)', 'M24 v2\n(GT bidir)',
             'M24 v3 HONEST\n(no GT)', 'oracle\n(ceiling)']
    vals = [0.020, 0.072, 0.122, overall, orac_overall]
    cols = ['#7a7a7a', '#9ecae1', '#4393c3', '#1b7837', '#c0c0c0']
    for b, v in zip(ax.bar(names, vals, color=cols, width=0.66), vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.004, f'{v:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('content-recall over ALL content slots (open vocab)')
    ax.set_ylim(0, max(vals)*1.25); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('M24: honest attack-grade content recovery (no ground-truth context)\n'
                 f'top-100 full-gallery lattice · {n_all_content} content slots · Qwen-1.5B',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'm24v3_{args.tag}.png'), dpi=200, bbox_inches='tight')
    print('[v3] wrote', os.path.join(OUT, f'm24v3_{args.tag}.png'), flush=True)


if __name__ == '__main__':
    main()
