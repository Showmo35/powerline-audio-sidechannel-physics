#!/usr/bin/env python3
"""
Stage 1 — forced-align the audio to index every word occurrence.

Writes word_index.json: {word: [[chunk, start_s, end_s], ...]} over the aligned
chunks. Chunk split (every 12th test) is applied later at train time. Frequent
words accumulate the most occurrences (frequent-first priority).
"""
import argparse, json, os, collections
import numpy as np, torch, wave
import torchaudio
from scipy.signal import resample_poly

ROOT = '<REPO_ROOT>'
WAVD = ROOT + '/Powerline_Data_Captures/audio_chunks'
MANIFEST = ROOT + '/M15_soundbar_melgen_word/full_manifest.json'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def read_wav(path, sr_out=16000):
    with wave.open(path, 'rb') as w:
        sr, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    return (resample_poly(x, sr_out, sr).astype(np.float32) if sr != sr_out else x), sr_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nchunks', type=int, default=120)
    ap.add_argument('--minlen', type=float, default=0.14)
    ap.add_argument('--maxlen', type=float, default=1.0)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'word_index.json'))
    args = ap.parse_args()

    man = json.load(open(MANIFEST))
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    w2v = bundle.get_model().to(dev).eval(); labels = bundle.get_labels()
    lab2id = {c: i for i, c in enumerate(labels)}

    def align(wave16, text):
        tnorm = ''.join(c for c in text.upper() if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ' ")
        toks = [lab2id[c] for c in tnorm.replace(' ', '|') if c in lab2id]
        if len(toks) < 2:
            return []
        with torch.inference_mode():
            emission, _ = w2v(torch.from_numpy(wave16)[None].to(dev)); logp = torch.log_softmax(emission, -1)
        if len(toks) >= emission.shape[1]:
            return []
        try:
            aligned, scores = torchaudio.functional.forced_align(logp, torch.tensor([toks], device=dev), blank=0)
        except RuntimeError:
            return []
        spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())
        ratio = wave16.shape[0] / emission.shape[1] / 16000.0
        words, cur, cs, prev_end = [], [], None, 0
        for sp in spans:
            ch = labels[sp.token]
            if ch == '|':
                if cur:
                    words.append((''.join(cur).lower(), cs * ratio, prev_end * ratio)); cur = []
                continue
            if not cur:
                cs = sp.start
            cur.append(ch); prev_end = sp.end
        if cur:
            words.append((''.join(cur).lower(), cs * ratio, prev_end * ratio))
        return words

    chunks = sorted({r['chunk'] for r in man})[:args.nchunks]
    occ = collections.defaultdict(list)
    for ci, ch in enumerate(chunks):
        if not os.path.exists(f'{WAVD}/{ch}.wav'):
            continue
        wav, _ = read_wav(f'{WAVD}/{ch}.wav')
        for r in [r for r in man if r['chunk'] == ch]:
            seg = wav[int(r['start_s'] * 16000):int(r['end_s'] * 16000)]
            if len(seg) < 3200:
                continue
            for w, ws, we in align(seg, r['text']):
                if args.minlen <= (we - ws) <= args.maxlen and w.isalpha():
                    occ[w].append([ch, round(r['start_s'] + ws, 4), round(r['start_s'] + we, 4)])
        if (ci + 1) % 5 == 0 or ci == len(chunks) - 1:
            print(f'  aligned {ci+1}/{len(chunks)} chunks  ({len(occ)} word types so far)', flush=True)
            json.dump(occ, open(args.out, 'w'))          # checkpoint incrementally

    json.dump(occ, open(args.out, 'w'))
    top = sorted(occ, key=lambda k: -len(occ[k]))
    print(f'[done] {sum(len(v) for v in occ.values())} occurrences, {len(occ)} word types')
    print('  top 30:', [(w, len(occ[w])) for w in top[:30]])


if __name__ == '__main__':
    main()
