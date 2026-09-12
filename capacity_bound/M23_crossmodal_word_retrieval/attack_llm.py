#!/usr/bin/env python3
"""
attack_llm.py — M23 attack stage 2 (GPU). THE HEADLINE EXPERIMENT.

Reconstruct each held-out sentence from its content-word lattice with a LOCAL
instruct LLM, and measure the LIFT over a blind baseline that gets the same
sentence structure (slot count + which slots are content vs function) but NO
powerline-recovered identities. Lift = information that actually leaked through
the powerline, above the LLM's language prior.

  POWERLINE arm : content slots show top-N candidate words + confidence.
  BLIND arm     : content slots shown as "{?}" (identity withheld).
Metrics: content-word recall, word-level F1, WER (all vs ground truth), reported
for both arms; lift = POWERLINE - BLIND. Oracle top-K recall = channel ceiling.

Model: Qwen2.5-1.5B-Instruct (cached, offline). Output -> results_attack/.
"""
import os, sys, json, re, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SUF  = '' if os.environ.get('M23_GEN', 'm14') == 'm14' else f"_gen{os.environ['M23_GEN'].upper()}"
DATA = os.path.join(HERE, 'data_attack' + SUF, 'sentences.json')
OUT  = os.path.join(HERE, 'results_attack' + SUF); os.makedirs(OUT, exist_ok=True)
MODEL = os.environ.get('M23_LLM', 'Qwen/Qwen2.5-1.5B-Instruct')
PROMPT_TOPN = 5
BATCH = 8

SYS = ("You reconstruct the exact English sentence a person spoke, recovered from a "
       "noisy acoustic side-channel. The words are given in order. A slot like "
       "{HOUSE:0.42|HOME:0.31} lists recovered content-word guesses with confidence "
       "(0-1); choose the one that best fits grammar and meaning, or a close variant. "
       "A slot '_' is an unknown short function word (the/a/of/and/to/was/...): fill it "
       "in. A slot '{?}' is an unknown content word: guess the most plausible one. "
       "Reply with ONLY the reconstructed sentence in uppercase, no punctuation, no notes.")


def slot_str(slot, arm):
    if not slot['is_content']:
        return '_'
    if arm == 'blind':
        return '{?}'
    cands = slot.get('topk', [])[:PROMPT_TOPN]
    return '{' + '|'.join(f'{w}:{s:.2f}' for w, s in cands) + '}' if cands else '{?}'


def build_prompt(sent, arm):
    seq = ' '.join(slot_str(s, arm) for s in sent['slots'])
    return f"Word slots in order ({len(sent['slots'])} words):\n{seq}\n\nReconstructed sentence:"


def norm(t):
    return re.sub(r'[^A-Z ]', ' ', t.upper()).split()


def is_content(w, STOP):
    return len(w) >= 4 and w not in STOP


def wer(ref, hyp):
    d = np.zeros((len(ref)+1, len(hyp)+1), int)
    d[:, 0] = np.arange(len(ref)+1); d[0, :] = np.arange(len(hyp)+1)
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1] + (ref[i-1] != hyp[j-1]))
    return d[len(ref), len(hyp)] / max(1, len(ref))


def score(gt_text, rec_text, STOP):
    gt, rec = norm(gt_text), norm(rec_text)
    gtc = [w for w in gt if is_content(w, STOP)]
    recset = set(rec)
    crec = np.mean([w in recset for w in gtc]) if gtc else 0.0
    inter = len(set(gt) & set(rec))
    prec = inter / max(1, len(set(rec))); reca = inter / max(1, len(set(gt)))
    f1 = 2*prec*reca/(prec+reca) if prec+reca else 0.0
    return {'content_recall': float(crec), 'word_f1': float(f1), 'wer': float(wer(gt, rec))}


@torch.no_grad()
def generate(model, tok, prompts):
    out = []
    for b in range(0, len(prompts), BATCH):
        chunk = prompts[b:b+BATCH]
        msgs = [[{'role': 'system', 'content': SYS}, {'role': 'user', 'content': p}] for p in chunk]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(texts, return_tensors='pt', padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for i in range(len(chunk)):
            new = gen[i, enc['input_ids'].shape[1]:]
            out.append(tok.decode(new, skip_special_tokens=True).strip())
        print(f'[llm] {min(b+BATCH,len(prompts))}/{len(prompts)}', flush=True)
    return out


def main():
    import build_mels as B
    STOP = B.STOP
    data = json.load(open(DATA))
    sents = data['sentences']
    print(f'[attack] sentences={len(sents)}  oracle_topk={data["oracle_topk_content_recall"]:.3f}  '
          f'model={MODEL}', flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                 device_map='cuda').eval()

    p_prompts = [build_prompt(s, 'powerline') for s in sents]
    b_prompts = [build_prompt(s, 'blind') for s in sents]
    t0 = time.time()
    p_rec = generate(model, tok, p_prompts)
    b_rec = generate(model, tok, b_prompts)
    print(f'[attack] generation done {time.time()-t0:.0f}s', flush=True)

    rows = []
    for s, pr, br in zip(sents, p_rec, b_rec):
        sp = score(s['text'], pr, STOP); sb = score(s['text'], br, STOP)
        rows.append({'utt_id': s['utt_id'], 'text': s['text'],
                     'powerline_rec': pr, 'blind_rec': br,
                     'powerline': sp, 'blind': sb})

    def agg(key):
        return {m: float(np.mean([r[key][m] for r in rows]))
                for m in ('content_recall', 'word_f1', 'wer')}
    P, Bl = agg('powerline'), agg('blind')
    lift = {m: P[m] - Bl[m] for m in P}

    summary = {'model': MODEL, 'n_sentences': len(sents),
               'oracle_topk_content_recall': data['oracle_topk_content_recall'],
               'content_slots_in_vocab': data['content_slots_in_vocab'],
               'powerline': P, 'blind': Bl, 'lift': lift}
    json.dump({'summary': summary, 'rows': rows}, open(f'{OUT}/attack_results.json', 'w'), indent=1)

    print('\n================ M23 ATTACK VERDICT ================', flush=True)
    print(f'oracle top-K content recall (channel ceiling) = {data["oracle_topk_content_recall"]:.3f}', flush=True)
    print(f'{"metric":16s} {"POWERLINE":>10s} {"BLIND":>10s} {"LIFT":>10s}', flush=True)
    for m in ('content_recall', 'word_f1', 'wer'):
        print(f'{m:16s} {P[m]:10.3f} {Bl[m]:10.3f} {lift[m]:+10.3f}', flush=True)
    print('\nexamples:', flush=True)
    for r in rows[:4]:
        print(f'  GT : {r["text"]}', flush=True)
        print(f'  PL : {r["powerline_rec"]}', flush=True)
        print(f'  BL : {r["blind_rec"]}', flush=True)
    verdict = 'LEAK CONFIRMED' if lift['content_recall'] > 0.05 else 'NO MEASURABLE LEAK'
    print(f'\n=> {verdict}: powerline lifts content recall by {lift["content_recall"]:+.3f} '
          f'over the blind LLM prior.', flush=True)


if __name__ == '__main__':
    main()
