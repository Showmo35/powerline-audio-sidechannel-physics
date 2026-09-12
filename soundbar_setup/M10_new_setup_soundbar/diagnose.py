#!/usr/bin/env python3
"""
diagnose.py — STEP 4: why did fine-tuning fail?  Eval-only decision matrix.

No training.  For a sample of utterances we run Whisper.generate under several
feature paths and report corpus WER, isolating the failure:

  A clean-native     clean ref audio → Whisper's OWN 16 kHz feature extractor
                     → off-the-shelf Whisper.  Harness sanity (expect ~5-15%).
  B clean-adapter    clean ref audio → OUR 22 kHz htk mel → whisper_features
                     adapter → off-the-shelf Whisper.  Tests whether the
                     adapter/convention destroys intelligibility on GOOD input.
  C powerline-OTS    stored powerline mel → adapter → off-the-shelf Whisper.
  D powerline-FT     stored powerline mel → adapter → FINE-TUNED model (test).
  E powerline-FT-tr  same, but on TRAIN utterances → can the model fit acoustics
                     at all, or only the LM prior?

Reading:
  * B low  → adapter fine; powerline signal is the problem (fix the front-end).
  * B high → our convention/adapter is lossy (rebuild in Whisper-native 16 kHz).
  * E low but D high → overf/generalization. E high too → no word info learned.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torchaudio.transforms as TT

from config import CFG, OUT_DIR
import data as D
import data_io as io
import asr
from whisper_features import mel_to_input_features

NATIVE_FPS = CFG.aud_sr / CFG.mel_hop
DATASET_DIR = os.path.join(OUT_DIR, 'dataset')
_MELSPEC = TT.MelSpectrogram(sample_rate=CFG.aud_sr, n_fft=CFG.mel_n_fft,
                             win_length=CFG.mel_win, hop_length=CFG.mel_hop,
                             n_mels=CFG.mel_n_mels, f_min=CFG.mel_fmin,
                             f_max=CFG.mel_fmax, power=1.0)


def clean_mel(audio22):
    return _MELSPEC(torch.from_numpy(audio22).float()).numpy()      # [80, T]


def load_powerline_mel(row, _cache={}):
    shard = row['shard']
    if shard not in _cache:
        _cache[shard] = np.load(os.path.join(DATASET_DIR, shard))
    return _cache[shard][row['utt_id']]


@torch.no_grad()
def gen_wer(model, processor, feats, refs, device, batch=12, tag=''):
    hyps = []
    for i in range(0, len(feats), batch):
        fb = torch.from_numpy(np.stack(feats[i:i + batch])).float().to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            ids = model.generate(input_features=fb, max_new_tokens=200,
                                 num_beams=1, no_repeat_ngram_size=4)
        hyps.extend(processor.batch_decode(ids, skip_special_tokens=True))
    sc = asr.corpus_wer([asr.normalize_text(r) for r in refs],
                        [asr.normalize_text(h) for h in hyps])
    print(f'  [{tag}] WER = {sc["wer"]*100:6.1f}%   (n={sc["n_utts"]})')
    return sc, hyps


def set_gen(model):
    for a, v in (('language', 'en'), ('task', 'transcribe'),
                 ('forced_decoder_ids', None), ('suppress_tokens', [])):
        try:
            setattr(model.generation_config, a, v)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='openai/whisper-small')
    ap.add_argument('--ft-ckpt', default=os.path.join(OUT_DIR, 'whisper_ft_small', 'best'))
    ap.add_argument('--test-chunks', default='41-46')
    ap.add_argument('--n', type=int, default=200, help='utterances sampled per split')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    random.seed(0)

    rows = D.load_manifest()
    lo, hi = (int(x) for x in args.test_chunks.split('-'))
    test_chunks = {f'chunk_{n:03d}' for n in range(lo, hi + 1)}
    tr_rows = [r for r in rows if r['chunk'] not in test_chunks]
    te_rows = [r for r in rows if r['chunk'] in test_chunks]
    te_s = random.sample(te_rows, min(args.n, len(te_rows)))
    tr_s = random.sample(tr_rows, min(args.n, len(tr_rows)))
    print(f'[data] sampled test={len(te_s)} train={len(tr_s)}')

    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(args.model)
    try:
        proc.tokenizer.set_prefix_tokens(language='en', task='transcribe')
    except Exception:
        pass
    ots = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    set_gen(ots)

    fe = proc.feature_extractor

    # build feature lists for the test sample
    A_feats, B_feats, C_feats, refs = [], [], [], []
    for r in te_s:
        dur = r['end_s'] - r['start_s']
        a16 = io.read_wav_window(CFG.wav_path(r['chunk']), r['start_s'], dur, CFG.asr_sr)
        a22 = io.read_wav_window(CFG.wav_path(r['chunk']), r['start_s'], dur, CFG.aud_sr)
        A_feats.append(fe(a16, sampling_rate=CFG.asr_sr, return_tensors='np').input_features[0])
        B_feats.append(mel_to_input_features(clean_mel(a22), NATIVE_FPS))
        C_feats.append(mel_to_input_features(load_powerline_mel(r), NATIVE_FPS))
        refs.append(r['text'])

    print('\n══ off-the-shelf Whisper-small ══')
    gen_wer(ots, proc, A_feats, refs, device, tag='A clean-native ')
    gen_wer(ots, proc, B_feats, refs, device, tag='B clean-adapter')
    gen_wer(ots, proc, C_feats, refs, device, tag='C powerline-OTS')

    # fine-tuned model
    results = {}
    if os.path.isdir(args.ft_ckpt):
        ft = WhisperForConditionalGeneration.from_pretrained(args.ft_ckpt).to(device).eval()
        set_gen(ft)
        ftp = WhisperProcessor.from_pretrained(args.ft_ckpt)
        print('\n══ fine-tuned model ══')
        scD, _ = gen_wer(ft, ftp, C_feats, refs, device, tag='D powerline-FT  (test)')
        tr_feats = [mel_to_input_features(load_powerline_mel(r), NATIVE_FPS) for r in tr_s]
        tr_refs = [r['text'] for r in tr_s]
        scE, hypsE = gen_wer(ft, ftp, tr_feats, tr_refs, device, tag='E powerline-FT  (train)')
        results['D_powerline_ft_test'] = scD['wer']
        results['E_powerline_ft_train'] = scE['wer']
        print('\n  sample TRAIN ref/hyp (does it fit its own training data?):')
        for r, h in list(zip(tr_refs, hypsE))[:4]:
            print(f'    REF: {asr.normalize_text(r)[:75]}')
            print(f'    HYP: {asr.normalize_text(h)[:75]}')
    else:
        print(f'[warn] no fine-tuned ckpt at {args.ft_ckpt}')

    with open(os.path.join(OUT_DIR, 'diagnose.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\n[done] → outputs/diagnose.json')


if __name__ == '__main__':
    main()
