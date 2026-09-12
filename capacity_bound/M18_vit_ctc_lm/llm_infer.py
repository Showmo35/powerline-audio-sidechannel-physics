#!/usr/bin/env python3
"""
llm_infer.py — can an LLM infer the true sentence from a model's noisy output?

Focus: the NULL model's output is phonetically close to the truth. Does an LLM
recover the real sentence from it? Controls for LLM memorisation of LibriSpeech
(public-domain books) with a letter-SCRAMBLE control: same words, letters shuffled
within each word → phonetic content destroyed, length/word-count kept. If the LLM
recovers the truth from the real garble but NOT from the scramble, it is denoising
phonetics (real); if it recovers either way, it is reciting from memory.

Reports, per condition: raw model WER, LLM-reconstructed WER, and (null) the
scramble-control WER. Also runs real (legit reference) and gen (powerline-derived,
expected unusable) for contrast.
"""
import argparse, os, random
import torch

from config import CFG, OUT_DIR
import dataset as D, models as M, text as T, lm as LM

PROMPT = ("You are correcting the noisy output of a speech recognizer. The text below "
          "is a garbled phonetic transcription of one English sentence. Reconstruct the "
          "most likely original sentence. Reply with ONLY the sentence.\n\nGarbled: {hyp}\nSentence:")


def scramble(s, rng):
    out = []
    for w in s.split():
        cs = list(w); rng.shuffle(cs); out.append(''.join(cs))
    return ' '.join(out)


@torch.no_grad()
def decode(src, rows, device, n):
    ck = torch.load(os.path.join(OUT_DIR, f'vit_{src}', 'best.pt'),
                    map_location=device, weights_only=False)
    model = M.build(CFG).to(device); model.load_state_dict(ck['model']); model.eval()
    ds = D.MelText(rows[:n], src, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=D.collate)
    refs, hyps = [], []
    for b in dl:
        mel = ((b['mel'] - ck['mean']) / ck['std']).to(device)
        logits, _ = model(mel, b['mel_lens'])
        hyps += T.ctc_greedy_decode(logits.float().cpu()); refs += b['texts']
    return refs, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = random.Random(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()

    def llm_fix(hyp):
        msg = [{'role': 'user', 'content': PROMPT.format(hyp=hyp or '(silence)')}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors='pt').to(device)
        out = llm.generate(ids, max_new_tokens=70, do_sample=False, pad_token_id=tok.eos_token_id)
        return T.normalize(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))

    rows = D.load_rows(); _, te = D.split_rows(rows)
    wer = lambda refs, hs: T.corpus_wer(refs, hs)['wer'] * 100

    for src in ('null', 'real', 'gen'):
        refs, hyps = decode(src, te, device, args.n)
        raw = wer(refs, hyps)
        fixed = [llm_fix(h) for h in hyps]
        wf = wer(refs, fixed)
        line = f'[{src:5}] raw={raw:.1f}%  →LLM={wf:.1f}%'
        if src == 'null':
            scr = [llm_fix(scramble(h, rng)) for h in hyps]
            line += f'   [scramble-control →LLM={wer(refs, scr):.1f}%]'
        print(line, flush=True)
        for i in range(3):
            print(f'    ref : {refs[i][:78]}')
            print(f'    hyp : {hyps[i][:78]}')
            print(f'    llm : {fixed[i][:78]}')


if __name__ == '__main__':
    main()
