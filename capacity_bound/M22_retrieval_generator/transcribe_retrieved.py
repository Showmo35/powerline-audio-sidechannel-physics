#!/usr/bin/env python3
"""
transcribe_retrieved.py — M22 as a RETRIEVAL recognizer.

For each word of a test sentence: M22 generates the word-mel, we REPLACE it with its
nearest REAL word-mel from the gallery (open-vocab kNN top-1), then stitch the
retrieved REAL words' audio into a sentence and transcribe with Whisper.

So the transcription = the sequence of words M22 retrieves, spoken as real audio.
Compares GT vs retrieved-word sequence vs Whisper-of-stitched-real-audio (which should
match the retrieved words, since they are real recordings). Test-only, no M22 leak.
"""
import os, json, numpy as np, torch, wave
import torchaudio
from torch.utils.data import DataLoader

from config import CFG, RC
import models as M
import dataset_words as D
import data_io as io
from train_retrieval import embed, gen_word_mel, build_gallery

ROOT = '<REPO_ROOT>'
MAN = os.path.join(ROOT, 'M15_soundbar_melgen_word', 'full_manifest.json')
WIDX = os.path.join(ROOT, 'M20_powerline_word_classifier', 'word_index.json')
OUT = 'outputs/asr_retrieved'; os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CTX = RC.ctx_s

ck = torch.load('outputs/best.pt', map_location=dev, weights_only=False)
model = M.build(CFG).to(dev); model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
mm, ms = ck['mel_mean'], ck['mel_std']
print(f'[m22] loaded step={ck.get("step")}', flush=True)

# ── gallery: real train word-mels (embeddings) + parallel timings ──
tr_items, words = D.build_items('train')
gal_items = D.cap_per_label(tr_items, 30)                       # (chunk,s,e,label), order preserved
gal_ds = D.WordSet(gal_items, real_only=True)
gal_dl = DataLoader(gal_ds, batch_size=256, shuffle=False, num_workers=8,
                    collate_fn=lambda bb: {'mel_word': torch.stack([x['mel_word'] for x in bb]),
                                           'label': torch.tensor([x['label'] for x in bb])})
print(f'[gallery] building {len(gal_items)} real word-mels over {len(words)} words …', flush=True)
G, gy = build_gallery(gal_dl, dev)                              # (Ng,D) fp16, (Ng,)
print(f'[gallery] {G.shape[0]} embeddings', flush=True)

# ── per-chunk word timeline (to segment a sentence into words) ──
occ = json.load(open(WIDX))
by_chunk = {}
for w, rows in occ.items():
    for c, s, e in rows:
        by_chunk.setdefault(c, []).append((float(s), float(e), w))
for c in by_chunk:
    by_chunk[c].sort()

melfn = torchaudio.transforms.MelSpectrogram(CFG.ref_sr, CFG.n_fft, hop_length=CFG.hop,
                                             win_length=CFG.win_length, n_mels=CFG.n_mels,
                                             f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)
_fb = torchaudio.functional.melscale_fbanks(CFG.n_fft // 2 + 1, CFG.fmin, CFG.fmax, CFG.n_mels,
                                            CFG.ref_sr, norm=None, mel_scale='htk')
_pinv = torch.linalg.pinv(_fb.T)
_gl = torchaudio.transforms.GriffinLim(CFG.n_fft, 48, win_length=CFG.win_length, hop_length=CFG.hop, power=2.0)
import whisper; wm = whisper.load_model('small', device=dev)


def savewav(p, x):
    with wave.open(p, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(CFG.ref_sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())

def whisp(x):
    r = wm.transcribe(np.ascontiguousarray(x.astype(np.float32)), language='en', fp16=(dev == 'cuda'), verbose=False)
    return ' '.join(''.join(c for c in r['text'].lower() if c.isalpha() or c == ' ').split())

def norm(t): return ' '.join(''.join(c for c in t.lower() if c.isalpha() or c == ' ').split())

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r)+1, len(h)+1), int); d[:, 0] = np.arange(len(r)+1); d[0, :] = np.arange(len(h)+1)
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1]+(r[i-1] != h[j-1]))
    return d[len(r), len(h)] / max(1, len(r))


@torch.no_grad()
def retrieve_word(ch, s, e):
    """M22 gen word-mel -> nearest gallery item index."""
    lag = io.read_lag_ms(CFG.lag_path(ch)) / 1000.0
    raw = io.read_bin_window(CFG.bin_path(ch), (s - CTX) + lag, CFG.win_s, CFG.cap_sr, CFG.in_sr).astype(np.float32)
    raw = raw / (np.sqrt(np.mean(raw ** 2)) + 1e-8)
    x = np.zeros(CFG.in_len, np.float32); x[:min(len(raw), CFG.in_len)] = raw[:CFG.in_len]
    gen = model.sample(torch.from_numpy(x)[None].to(dev)).float() * ms + mm
    wdur = (e - s) + 2 * CTX
    frs = torch.tensor([max(6, int(round(wdur * CFG.fps)))])
    q = embed(gen_word_mel(gen, frs)).half()                   # (1,D)
    gi = int((q @ G.t()).argmax().item())
    return gi


def real_word_audio(ch, s, e):
    return io.read_wav_window(CFG.wav_path(ch), s - CTX, (e - s) + 2 * CTX, CFG.ref_sr).astype(np.float32)


# ── pick 5 test sentences, segment into words, retrieve, stitch, transcribe ──
man = json.load(open(MAN))
test = [r for r in man if CFG.is_test(r['chunk']) and 2.0 <= r['dur_s'] <= 3.8 and len(r['text'].split()) >= 4]
rng = np.random.RandomState(1); rng.shuffle(test)

sil = np.zeros(int(0.06 * CFG.ref_sr), np.float32)
results = []
picked = 0
for r in test:
    if picked >= 5:
        break
    ch, us, ue = r['chunk'], r['start_s'], r['end_s']
    seg = [(s, e, w) for (s, e, w) in by_chunk.get(ch, []) if s >= us - 0.05 and e <= ue + 0.05]
    if len(seg) < 4:
        continue
    retr_words, retr_audio, true_audio = [], [], []
    for (s, e, w) in seg:
        gi = retrieve_word(ch, s, e)
        rch, rs, re, rlab = gal_items[gi]
        retr_words.append(words[rlab])
        retr_audio.append(real_word_audio(rch, rs, re)); retr_audio.append(sil)
        true_audio.append(real_word_audio(ch, s, e)); true_audio.append(sil)
    retr_wav = np.concatenate(retr_audio); true_wav = np.concatenate(true_audio)
    savewav(f'{OUT}/utt{picked}_retrieved.wav', retr_wav)
    gt = norm(r['text']); retr_seq = ' '.join(retr_words)
    hyp_asr = whisp(retr_wav); hyp_true = whisp(true_wav)
    row = {'chunk': ch, 'gt': gt, 'retrieved_words': retr_seq, 'asr_of_retrieved': hyp_asr,
           'asr_of_true_words': hyp_true, 'n_words': len(seg),
           'wer_retrieved': wer(gt, retr_seq), 'wer_asr': wer(gt, hyp_asr)}
    results.append(row)
    print(f'\n=== utt{picked} [{ch}]  ({len(seg)} words) ===', flush=True)
    print(f'  GT (words)      : {gt}', flush=True)
    print(f'  M22 RETRIEVED   : {retr_seq}   (WER {row["wer_retrieved"]*100:.0f}%)', flush=True)
    print(f'  ASR of retrieved: {hyp_asr}   (WER {row["wer_asr"]*100:.0f}%)', flush=True)
    print(f'  ASR of true wrds: {hyp_true}   (stitch sanity)', flush=True)
    picked += 1

json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1)
print(f'\n==== SUMMARY ({picked} sentences) ====', flush=True)
print(f'  mean WER  M22-retrieved words   = {np.mean([r["wer_retrieved"] for r in results])*100:.0f}%', flush=True)
print(f'  mean WER  ASR-of-retrieved      = {np.mean([r["wer_asr"] for r in results])*100:.0f}%', flush=True)
print(f'  wavs -> {OUT}/', flush=True)
