#!/usr/bin/env python3
"""
llm_agent.py — the exploratory "intelligent agent" arm: a pretrained LLM reads the
acoustic model's noisy hypothesis and reconstructs a clean sentence.

Run on all three conditions (gen / real / null) from their nbest.json. The NULL
result is the control: LibriSpeech texts are public-domain books the LLM has
likely memorised, so if the LLM reconstructs the true sentence from NULL's
envelope-only garbage as well as from GEN, it is reciting/hallucinating, not
recovering. Report WER per condition; the meaningful quantity is WER(gen) vs
WER(null), not WER(gen) alone.
"""
import argparse, json, os
import torch

from config import OUT_DIR
import text as T

PROMPT = (
    "You are correcting the noisy output of a speech recognizer. "
    "The following text is a garbled transcription of one English sentence. "
    "Reconstruct the most likely original sentence. Reply with ONLY the sentence.\n\n"
    "Garbled: {hyp}\nSentence:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', choices=('gen', 'real', 'null'), required=True)
    ap.add_argument('--field', default='beam_lm', choices=('greedy', 'beam', 'beam_lm'))
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--limit', type=int, default=600)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.join(OUT_DIR, f'vit_{args.input}')

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()

    dump = json.load(open(os.path.join(out_dir, 'nbest.json')))[:args.limit]
    refs, hyps = [], []
    for i, d in enumerate(dump):
        msg = [{'role': 'user', 'content': PROMPT.format(hyp=d[args.field] or '(silence)')}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors='pt').to(device)
        with torch.no_grad():
            out = llm.generate(ids, max_new_tokens=80, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        refs.append(d['ref']); hyps.append(T.normalize(text))
        if i < 3:
            print(f'    ref : {d["ref"][:80]}')
            print(f'    in  : {d[args.field][:80]}')
            print(f'    llm : {hyps[-1][:80]}')

    w = T.corpus_wer(refs, hyps)['wer'] * 100
    res = {'input': args.input, 'field': args.field, 'model': args.model, 'n': len(refs), 'wer_llm': w}
    print(f'[llm:{args.input}] n={len(refs)} field={args.field} → WER={w:.1f}%', flush=True)
    json.dump(res, open(os.path.join(out_dir, 'llm.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
