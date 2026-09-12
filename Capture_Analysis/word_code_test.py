#!/usr/bin/env python3
"""
Test: same word -> same powerline code? On the clean soundbar dataset.

Forced-align audio (wav2vec2) -> word timings. For frequent words, extract each
occurrence's POWERLINE segment -> M14 encoder code, and the AUDIO segment ->
wav2vec2 code (positive control). Compare same-word vs different-word cosine.

  wav2vec2 (audio): same-word >> diff-word   (proves the test detects word identity)
  M14 (powerline):  same-word >> diff-word ?  <- the hypothesis
"""
import argparse, json, sys, collections, importlib
import numpy as np, torch, wave
import torchaudio
from scipy.signal import resample_poly

ROOT = '<REPO_ROOT>'
BIN = ROOT + '/Powerline_Data_Captures/soundbar_bin_captures'
WAVD = ROOT + '/Powerline_Data_Captures/audio_chunks'
CAP_SR = 200_000
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def read_wav(path, sr_out=16000):
    with wave.open(path, 'rb') as w:
        sr, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    if sr != sr_out:
        x = resample_poly(x, sr_out, sr).astype(np.float32)
    return x, sr_out


def read_lag_ms(chunk):
    import os
    p = f'{BIN}/{chunk}.lag'
    return float(open(p).read().strip()) if os.path.exists(p) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nchunks', type=int, default=8)
    ap.add_argument('--per-word', type=int, default=40)
    ap.add_argument('--nwords', type=int, default=6)
    ap.add_argument('--minlen', type=float, default=0.18)
    ap.add_argument('--maxlen', type=float, default=0.9)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    man = json.load(open(ROOT + '/M15_soundbar_melgen_word/full_manifest.json'))
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    w2v = bundle.get_model().to(dev).eval()
    labels = bundle.get_labels()
    lab2id = {c: i for i, c in enumerate(labels)}          # blank='-' at 0, sep='|'

    def align(wave16, text):
        """return list of (word, start_s, end_s) via CTC forced alignment."""
        tnorm = ''.join(c for c in text.upper() if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ' ")
        toks = [lab2id[c] for c in tnorm.replace(' ', '|') if c in lab2id]
        if len(toks) < 2:
            return []
        with torch.inference_mode():
            emission, _ = w2v(torch.from_numpy(wave16)[None].to(dev))
            logp = torch.log_softmax(emission, -1)
        # CTC needs at least len(targets)+#repeats frames; skip mismatched/short rows
        if len(toks) >= emission.shape[1]:
            return []
        targets = torch.tensor([toks], device=dev)
        try:
            aligned, scores = torchaudio.functional.forced_align(logp, targets, blank=0)
        except RuntimeError:
            return []
        spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())
        ratio = wave16.shape[0] / emission.shape[1] / 16000.0     # sec per frame
        # walk spans, split into words at '|'
        words, cur, cs = [], [], None
        for sp in spans:
            ch = labels[sp.token]
            if ch == '|':
                if cur:
                    words.append((''.join(cur), cs * ratio, prev_end * ratio)); cur = []
                continue
            if not cur:
                cs = sp.start
            cur.append(ch); prev_end = sp.end
        if cur:
            words.append((''.join(cur), cs * ratio, prev_end * ratio))
        return words

    # ---- align a subset of chunks ----
    chunks = sorted({r['chunk'] for r in man})[:args.nchunks]
    occ = collections.defaultdict(list)      # word -> [(chunk, abs_start, abs_end)]
    for ci, ch in enumerate(chunks):
        wav, _ = read_wav(f'{WAVD}/{ch}.wav')
        rows = [r for r in man if r['chunk'] == ch]
        for r in rows:
            s0, s1 = r['start_s'], r['end_s']
            seg = wav[int(s0 * 16000):int(s1 * 16000)]
            if len(seg) < 3200:
                continue
            for w, ws, we in align(seg, r['text']):
                if args.minlen <= (we - ws) <= args.maxlen:
                    occ[w.lower()].append((ch, s0 + ws, s0 + we))
        print(f'  aligned {ch} ({ci+1}/{len(chunks)})', flush=True)
        if args.smoke and ci == 0:
            common = sorted(occ, key=lambda k: -len(occ[k]))[:5]
            for w in common:
                print(f'    [smoke] "{w}" x{len(occ[w])} e.g. {occ[w][0]}')
            return

    targets = [w for w, _ in collections.Counter({k: len(v) for k, v in occ.items()}).most_common()
               if len(w) >= 4 and len(occ[w]) >= args.per_word][:args.nwords]
    print(f'[words] testing {targets} (>= {args.per_word} occ each)', flush=True)

    # ---- M14 encoder ----
    sys.path.insert(0, ROOT + '/M14_new_setup_soundbar_melgen')
    cfg = importlib.import_module('config').CFG
    m14 = importlib.import_module('models').build(cfg).to(dev)
    ck = torch.load(ROOT + '/M14_new_setup_soundbar_melgen/outputs/cluster_a100/last.pt',
                    map_location=dev, weights_only=False)
    m14.load_state_dict(ck.get('ema', ck['model'])); m14.eval()

    pl_cache = {}
    def pl_load(ch):
        if ch not in pl_cache:
            pl_cache[ch] = np.fromfile(f'{BIN}/{ch}.bin', dtype=np.float32)
        return pl_cache[ch]

    @torch.no_grad()
    def codes(ch, s, e):
        lag = read_lag_ms(ch) / 1000.0
        pl = pl_load(ch)
        a, b = int((s + lag) * CAP_SR), int((e + lag) * CAP_SR)
        x = pl[a:b]
        if len(x) < 800:
            return None
        x = x / (x.std() + 1e-8)
        xr = resample_poly(x, cfg.in_sr, CAP_SR).astype(np.float32)
        m14code = m14.encode(torch.from_numpy(xr)[None].to(dev))[0].mean(0).float().cpu()
        # audio code (wav2vec2 features)
        wav, _ = read_wav(f'{WAVD}/{ch}.wav')
        aw = wav[int(s * 16000):int(e * 16000)]
        if len(aw) < 400:
            return None
        feats, _ = w2v.extract_features(torch.from_numpy(aw)[None].to(dev))
        a2 = feats[-1][0].mean(0).float().cpu()
        return m14code, a2

    M14, W2V, WORD = [], [], []
    for w in targets:
        for (ch, s, e) in occ[w][:args.per_word]:
            c = codes(ch, s, e)
            if c is None:
                continue
            M14.append(c[0]); W2V.append(c[1]); WORD.append(w)
    M14 = torch.stack(M14); W2V = torch.stack(W2V); WORD = np.array(WORD)
    print(f'[data] {len(WORD)} word segments across {len(set(WORD))} words', flush=True)

    def same_vs_diff(X, name):
        X = X / (X.norm(dim=1, keepdim=True) + 1e-9); S = (X @ X.T).numpy()
        same, diff = [], []
        for i in range(len(WORD)):
            for j in range(i + 1, len(WORD)):
                (same if WORD[i] == WORD[j] else diff).append(S[i, j])
        print(f'[{name}] same-word cos={np.mean(same):.3f}  diff-word cos={np.mean(diff):.3f}  '
              f'DELTA={np.mean(same)-np.mean(diff):+.3f}', flush=True)

    same_vs_diff(W2V, 'wav2vec2 audio (control)')
    same_vs_diff(M14, 'M14 powerline (hypothesis)')


if __name__ == '__main__':
    main()
