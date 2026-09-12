#!/usr/bin/env python3
"""
asr_eval.py — Step 5c: does the learned enhancement actually lower WER?

Controlled comparison — push three mels through the SAME Whisper path so any WER
difference is attributable to the front-end:
    clean target / fixed 8-harm sum / learned U-Net mel  → adapter → Whisper.
Plus the deployable number: learned mel → Griffin-Lim → 16 kHz → Whisper-native.

Baselines to beat: fixed-adapter (~100%), and the base module's GL path (94.6%).
Ceiling: clean-adapter (~56%, adapter cost) and clean-native (~4%).
"""

import argparse
import json
import os
import random
import re

import numpy as np
import torch
import torchaudio.functional as AF
import torchaudio.transforms as TT

from config import CFG, OUT_DIR, FEAT_DIR, SHARED_MANIFEST
from model_unet import MelUNet
import data_io as io

NATIVE_FPS = CFG.aud_sr / CFG.mel_hop
_PUNCT = re.compile(r"[^a-z0-9'\s]")


def norm(s):
    return ' '.join(_PUNCT.sub(' ', s.lower()).split())


def wer_corpus(refs, hyps):
    S = D = I = N = 0
    for r, h in zip(refs, hyps):
        r, h = norm(r).split(), norm(h).split()
        n, m = len(r), len(h)
        d = np.zeros((n + 1, m + 1), int)
        d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1,
                              d[i-1, j-1] + (r[i-1] != h[j-1]))
        # edits split via final cell (approx S/D/I not needed for corpus WER)
        S += d[n, m]; N += n
    return S / max(N, 1)


def logmel_to_input_features(logmel, n_frames=3000):
    L = np.asarray(logmel, np.float32)
    L = np.maximum(L, L.max() - 8.0)
    L = (L - L.mean()) / (L.std() + 1e-6)
    M, T = L.shape
    new_T = max(1, int(round(T * 100.0 / NATIVE_FPS)))
    xi = np.linspace(0, T - 1, new_T)
    Lr = np.stack([np.interp(xi, np.arange(T), L[b]) for b in range(M)]).astype(np.float32)
    out = np.full((M, n_frames), float(Lr.min()), np.float32)
    out[:, :min(new_T, n_frames)] = Lr[:, :min(new_T, n_frames)]
    return out


_FB = AF.melscale_fbanks(CFG.mel_n_fft // 2 + 1, CFG.mel_fmin, CFG.mel_fmax,
                         CFG.mel_n_mels, CFG.aud_sr, norm=None, mel_scale='htk')
_FB_PINV = torch.linalg.pinv(_FB)
_GL = TT.GriffinLim(CFG.mel_n_fft, win_length=CFG.mel_win, hop_length=CFG.mel_hop,
                    power=1.0, n_iter=48)


def logmel_to_audio16(logmel):
    lin = torch.from_numpy(np.exp(np.asarray(logmel, np.float32)))
    spec = (_FB_PINV.T @ lin).clamp(min=0)
    wav = _GL(spec).numpy()
    return io.resample(wav, CFG.aud_sr, CFG.asr_sr)


@torch.no_grad()
def gen(model, processor, feats, device, batch=16):
    hyps = []
    for i in range(0, len(feats), batch):
        fb = torch.from_numpy(np.stack(feats[i:i+batch])).float().to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):
            ids = model.generate(input_features=fb, max_new_tokens=200,
                                 num_beams=1, no_repeat_ngram_size=4)
        hyps.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(OUT_DIR, 'enh_unet', 'best.pt'))
    ap.add_argument('--model', default='openai/whisper-small')
    ap.add_argument('--test-chunks', default='41-46')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    random.seed(0)

    lo, hi = (int(v) for v in args.test_chunks.split('-'))
    tc = {f'chunk_{n:03d}' for n in range(lo, hi + 1)}
    rows = [json.loads(l) for l in open(SHARED_MANIFEST) if l.strip()]
    te = [r for r in rows if r['chunk'] in tc]
    sample = random.sample(te, min(args.n, len(te)))
    print(f'[data] {len(sample)} test utterances')

    net = MelUNet(in_ch=2 * CFG.n_harmonics, base=CFG.base_ch).to(device).eval()
    net.load_state_dict(torch.load(args.ckpt, map_location=device))

    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(args.model)
    try:
        proc.tokenizer.set_prefix_tokens(language='en', task='transcribe')
    except Exception:
        pass
    wm = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    for a, v in (('language', 'en'), ('task', 'transcribe'),
                 ('forced_decoder_ids', None), ('suppress_tokens', [])):
        try:
            setattr(wm.generation_config, a, v)
        except Exception:
            pass
    fe = proc.feature_extractor

    xcache, ycache = {}, {}
    cl, fx, en, en_gl, refs = [], [], [], [], []
    for r in sample:
        ch, uid = r['chunk'], r['utt_id']
        if ch not in xcache:
            xcache[ch] = np.load(os.path.join(FEAT_DIR, f'{ch}.x.npz'))
            ycache[ch] = np.load(os.path.join(FEAT_DIR, f'{ch}.y.npz'))
        x = xcache[ch][uid].astype(np.float32)             # log stack [C,M,T]
        y = ycache[ch][uid].astype(np.float32)             # clean logmel [M,T]
        xz = (x - x.mean()) / (x.std() + 1e-6)
        with torch.no_grad():
            pred = net(torch.from_numpy(xz)[None].to(device)).float().cpu().numpy()[0]
        fixed = np.log(np.maximum(np.exp(x).mean(0), 1e-5))
        cl.append(logmel_to_input_features(y))
        fx.append(logmel_to_input_features(fixed))
        en.append(logmel_to_input_features(pred))
        a16 = logmel_to_audio16(pred)
        en_gl.append(fe(a16, sampling_rate=CFG.asr_sr, return_tensors='np').input_features[0])
        refs.append(r['text'])

    print('\n══ Step 5c — WER (Whisper-small) ════════════════')
    res = {}
    for tag, feats in (('clean-target  (adapter, ceiling)', cl),
                       ('fixed-sum     (adapter, baseline)', fx),
                       ('LEARNED U-Net (adapter)', en),
                       ('LEARNED U-Net → GL → native', en_gl)):
        w = wer_corpus(refs, gen(wm, proc, feats, device))
        res[tag] = w
        print(f'  {tag:36s} WER = {w*100:5.1f}%')
    print('  reference points: clean-native ~4%, base-module GL 94.6%')
    print('══════════════════════════════════════════════════')
    with open(os.path.join(OUT_DIR, 'asr_eval.json'), 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
