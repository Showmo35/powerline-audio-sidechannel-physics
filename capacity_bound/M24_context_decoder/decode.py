#!/usr/bin/env python3
"""
decode.py — M24: does sentence CONTEXT let us pick the right word among the
powerline retrieval candidates?  (Noisy-channel reranking test.)

The M23 attack's LLM only reached content-recall 0.027 of a ~0.13 oracle ceiling —
it barely used the lattice. Here we test the hypothesis directly and generously:

For every held-out CONTENT slot that has a candidate list (top-K powerline words +
cosine sims), we rerank the candidates with a language model conditioned on the
sentence's GT left-context (TEACHER-FORCED — the most generous case for context):

    score(w) = logP_LM(w | left-context) + λ · logP_channel(w)
    logP_channel = log_softmax(sims / τ)     (channel evidence as a distribution)

and compare:
    channel-only  (λ → ∞, = powerline top-1)
    LM-only       (λ = 0,  context prior alone)
    M24           (best λ,  context + channel)
    oracle        (is the true word in the candidate list at all — the ceiling)

If M24 ≫ channel-only and approaches oracle, context resolves the rhyme confusions
(hypothesis TRUE). If M24 ≈ channel-only, context cannot help — the failures are
words that aren't in the list at all, which no LM can select.

Input: an M23 attack lattice (default: the M22-generator lattice). LM: Qwen2.5-1.5B
(offline). Output: outputs/m24_result.json + outputs/m24_context.png.
"""
import argparse, os, json
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


def collect_slots(data):
    """Yield (context_str, true_word, [cand_words], np.array sims) for every content
    slot that has a candidate list; also count all content slots for the global rate."""
    items, n_content_all = [], 0
    for s in data['sentences']:
        slots = s['slots']
        for i, sl in enumerate(slots):
            if not sl.get('is_content'):
                continue
            n_content_all += 1
            tk = sl.get('topk')
            if not tk or sl.get('unknown'):
                continue
            ctx = ' '.join(x['w'] for x in slots[:i])
            words = [c[0] for c in tk]
            sims = np.array([c[1] for c in tk], dtype=np.float32)
            items.append((ctx, sl['w'], words, sims))
    return items, n_content_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lattice', default=os.path.join(
        PROOT, 'M23_crossmodal_word_retrieval', 'data_attack_genM22', 'sentences.json'))
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--tau', type=float, default=0.05, help='channel softmax temperature')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='genM22')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'

    data = json.load(open(args.lattice))
    items, n_content_all = collect_slots(data)
    print(f'[m24] lattice={args.lattice}', flush=True)
    print(f'[m24] content slots total={n_content_all}  with candidates={len(items)}  '
          f'(lattice oracle_topk={data.get("oracle_topk_content_recall"):.3f})', flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if dev == 'cuda' else torch.float32).to(dev).eval()

    # LM log-prob of each candidate's FIRST token given the left context (one forward
    # per slot). First-token approximation — fine for reranking ~10 candidates.
    lm_lp, chan_lp, true_in, true_idx = [], [], [], []
    with torch.no_grad():
        for n, (ctx, true, words, sims) in enumerate(items):
            ids = tok(ctx if ctx else tok.bos_token or '\n',
                      return_tensors='pt').input_ids.to(dev)
            logits = lm(ids).logits[0, -1].float()
            logsm = torch.log_softmax(logits, -1)
            ft = [tok(' ' + w, add_special_tokens=False).input_ids[0] for w in words]
            lp = logsm[torch.tensor(ft, device=dev)].cpu().numpy()
            lm_lp.append(lp)
            c = sims / args.tau; c = c - c.max()
            chan_lp.append(c - np.log(np.exp(c).sum()))
            true_in.append(true in words)
            true_idx.append(words.index(true) if true in words else -1)
            if n % 200 == 0:
                print(f'[m24] scored {n}/{len(items)}', flush=True)

    true_in = np.array(true_in)
    N = len(items)

    def acc(pred_idx):
        return float(np.mean([true_idx[i] == pred_idx[i] for i in range(N)]))

    # channel-only = argmax sim ; LM-only = argmax lm_lp
    ch_pred = [int(np.argmax(chan_lp[i])) for i in range(N)]
    lm_pred = [int(np.argmax(lm_lp[i])) for i in range(N)]
    acc_channel, acc_lm = acc(ch_pred), acc(lm_pred)
    acc_oracle = float(np.mean(true_in))

    lambdas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 40.0]
    sweep = {}
    for lam in lambdas:
        pred = [int(np.argmax(lm_lp[i] + lam * chan_lp[i])) for i in range(N)]
        sweep[lam] = acc(pred)
    best_lam = max(sweep, key=sweep.get)
    acc_m24 = sweep[best_lam]

    # global rates (denominator = ALL content slots; no-candidate slots are misses)
    g = len(items) / max(1, n_content_all)
    print('\n══ M24 context-reranking result (teacher-forced GT context) ══════════')
    print(f'  content slots with candidates : {N}  (of {n_content_all} content slots)')
    print(f'  ── accuracy on slots WITH candidates ──')
    print(f'  oracle (true in candidate list): {acc_oracle:.3f}   <- ceiling')
    print(f'  channel-only (powerline top-1) : {acc_channel:.3f}')
    print(f'  LM-only (context prior)        : {acc_lm:.3f}')
    print(f'  M24 (context + channel, λ={best_lam}) : {acc_m24:.3f}')
    print(f'  λ sweep: ' + '  '.join(f'{l}:{sweep[l]:.3f}' for l in lambdas))
    print(f'  ── content-recall over ALL content slots (×coverage {g:.2f}) ──')
    print(f'  channel-only={acc_channel*g:.3f}  M24={acc_m24*g:.3f}  oracle={acc_oracle*g:.3f}')
    print('══════════════════════════════════════════════════════════════════════')

    res = {'lattice': args.lattice, 'model': args.model, 'N_with_cands': N,
           'n_content_all': n_content_all, 'coverage': g,
           'acc_on_candidate_slots': {'oracle': acc_oracle, 'channel_only': acc_channel,
                                      'lm_only': acc_lm, 'm24': acc_m24, 'best_lambda': best_lam,
                                      'lambda_sweep': sweep},
           'content_recall_all': {'channel_only': acc_channel*g, 'm24': acc_m24*g,
                                  'oracle': acc_oracle*g}}
    json.dump(res, open(os.path.join(OUT, f'm24_result_{args.tag}.json'), 'w'), indent=2)

    # ── figure: accuracy on candidate slots ──
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    names = ['channel-only\n(powerline top-1)', 'LM-only\n(context)',
             f'M24\n(context+channel)', 'oracle\n(true in list)']
    vals = [acc_channel, acc_lm, acc_m24, acc_oracle]
    cols = ['#7a7a7a', '#4393c3', '#1b7837', '#c0c0c0']
    bars = ax.bar(names, vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('accuracy on content slots WITH candidates')
    ax.set_ylim(0, max(vals)*1.25)
    ax.set_title(f'M24: does context pick the right word among powerline candidates?\n'
                 f'({args.tag} lattice · N={N} slots · teacher-forced GT context)', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'm24_context_{args.tag}.png'), dpi=200, bbox_inches='tight')
    print('[m24] wrote', os.path.join(OUT, f'm24_context_{args.tag}.png'), flush=True)


if __name__ == '__main__':
    main()
