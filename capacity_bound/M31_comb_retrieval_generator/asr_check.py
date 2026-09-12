#!/usr/bin/env python3
"""
asr_check.py — is the VOCODER the bottleneck, or the MEL?  (decisive, minutes)

Runs ASR over three renderings of the SAME held-out utterances, all through the SAME
Griffin-Lim vocoder so the vocoder is held constant:

  A  orig        true audio                      -> the audio ceiling
  B  GL(real mel)   real mel -> GL -> audio      -> THE VOCODER CEILING
  C  GL(gen mel)    M31 raw-arm mel -> GL -> audio

Logic:
  * If B is intelligible and C is not, the mel->audio step is NOT the bottleneck --
    the generated MEL is wrong, and NO vocoder (HiFi-GAN included) can fix that,
    because a vocoder cannot add information the mel does not contain. A neural
    vocoder would only make C sound cleaner while still saying the wrong words.
  * If B were ALSO bad, then the vocoder WOULD be worth replacing.

Sentence-level utterances (full_manifest) so WER is meaningful. The generated mel
covers [utt_start, utt_start+win_s]; we crop it to the utterance duration so the audio
contains exactly the sentence.

Output: outputs/asr_check.json (+ a few wavs in outputs/audio_asr/)
"""
import os, sys, json, re, argparse
import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import CFG, RC, OUT_DIR
import models_comb as M
import data_io_raw as io
import dataset_words as D
from train import FrontEnd

PROOT = '<REPO_ROOT>'
MANIFEST = os.path.join(PROOT, 'Powerline_Data_Captures', 'full_manifest.json')
AUD = os.path.join(OUT_DIR, 'audio_asr'); os.makedirs(AUD, exist_ok=True)

_melfn = torchaudio.transforms.MelSpectrogram(
    CFG.ref_sr, CFG.n_fft, hop_length=CFG.hop, win_length=CFG.win_length,
    n_mels=CFG.n_mels, f_min=CFG.fmin, f_max=CFG.fmax, power=2.0)
_fb = torchaudio.functional.melscale_fbanks(CFG.n_fft // 2 + 1, CFG.fmin, CFG.fmax,
                                            CFG.n_mels, CFG.ref_sr, norm=None, mel_scale='htk')
_pinv = torch.linalg.pinv(_fb.T)
_gl = torchaudio.transforms.GriffinLim(CFG.n_fft, n_iter=64, win_length=CFG.win_length,
                                       hop_length=CFG.hop, power=2.0)


def mel_to_wav(logmel):
    m = torch.as_tensor(logmel).float()
    spec = (_pinv @ (m.exp() - CFG.log_eps).clamp(min=0)).clamp(min=0)
    w = _gl(spec)
    return (w / (w.abs().max() + 1e-8)).numpy()


def norm(t):
    return re.sub(r'[^A-Z ]', ' ', t.upper()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r:
        return 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), np.int32)
    d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1,
                          d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[len(r), len(h)] / len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--model', default='small')
    ap.add_argument('--arm', default='raw')
    ap.add_argument('--save-wavs', type=int, default=3)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = (dev == 'cuda')

    man = json.load(open(MANIFEST))
    utts = [u for u in man if CFG.is_test(u['chunk']) and 1.5 <= u['dur_s'] <= 3.8]
    rng = np.random.RandomState(0); rng.shuffle(utts); utts = utts[:args.n]
    print(f'[asr] {len(utts)} held-out utterances, whisper={args.model}, arm={args.arm}',
          flush=True)

    ck = torch.load(os.path.join(OUT_DIR, args.arm, 'best.pt'), map_location=dev,
                    weights_only=False)
    model = M.build(CFG, frontend=args.arm).to(dev)
    model.load_state_dict(ck.get('ema', ck['model'])); model.eval()
    mean, std = ck['mel_mean'], ck['mel_std']
    front = FrontEnd(args.arm, dev)

    import whisper
    asr = whisper.load_model(args.model, device=dev)

    def tr(x):
        r = asr.transcribe(np.ascontiguousarray(x.astype(np.float32)), language='en',
                           fp16=(dev == 'cuda'), verbose=False)
        return r['text']

    rows = []
    for i, u in enumerate(utts):
        ch, s, dur, txt = u['chunk'], u['start_s'], u['dur_s'], u['text']
        lag = io.read_lag_ms(CFG.lag_path(ch)) / 1000.0
        raw = io.read_bin_window(CFG.bin_path(ch), s - CFG.pad_s + lag, CFG.tot_s, CFG.cap_sr)
        need = int(round(CFG.tot_s * CFG.cap_sr))
        if len(raw) < need * 0.9:
            continue
        raw = np.pad(raw, (0, max(0, need - len(raw))))[:need]
        raw = raw / (np.std(raw) + 1e-8)
        a4 = io.read_wav_window(CFG.wav_path(ch), s, CFG.win_s, CFG.ref_sr)
        nf = min(CFG.n_frames, int(round(dur * CFG.fps)))

        with torch.no_grad():
            cond = front(torch.from_numpy(raw[None]).float().to(dev),
                         torch.tensor([D.chunk_f0(ch)], device=dev))
            torch.manual_seed(0)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                g = model.sample(cond, steps=RC.eval_steps, cfg_scale=RC.cfg_scale_eval)
            gm = (g.float() * std + mean)[0].cpu()[:, :nf]

        rm = torch.log(_melfn(torch.from_numpy(np.asarray(a4, np.float32))) + CFG.log_eps)[:, :nf]
        wA = np.asarray(a4[:int(dur * CFG.ref_sr)], np.float32)
        wB = mel_to_wav(rm)
        wC = mel_to_wav(gm)
        hA, hB, hC = tr(wA), tr(wB), tr(wC)
        rows.append({'utt': u['utt_id'], 'ref': txt,
                     'A_orig': {'hyp': hA, 'wer': wer(txt, hA)},
                     'B_realGL': {'hyp': hB, 'wer': wer(txt, hB)},
                     'C_genGL': {'hyp': hC, 'wer': wer(txt, hC)}})
        if i < args.save_wavs:
            for tag, x in [('A_orig', wA), ('B_realGL', wB), ('C_genGL', wC)]:
                torchaudio.save(os.path.join(AUD, f'{i}_{tag}.wav'),
                                torch.from_numpy(np.asarray(x, np.float32))[None], CFG.ref_sr)
        if i % 10 == 0:
            print(f'[asr] {i}/{len(utts)}', flush=True)

    A = float(np.mean([r['A_orig']['wer'] for r in rows]))
    B = float(np.mean([r['B_realGL']['wer'] for r in rows]))
    C = float(np.mean([r['C_genGL']['wer'] for r in rows]))
    print('\n══ IS THE VOCODER THE BOTTLENECK? (n=%d, whisper-%s) ══' % (len(rows), args.model))
    print(f'  A  orig audio          WER = {A:.3f}   <- audio ceiling')
    print(f'  B  GL(real mel)        WER = {B:.3f}   <- THE VOCODER CEILING')
    print(f'  C  GL(M31 {args.arm} mel)    WER = {C:.3f}   <- what the powerline gives')
    print('  ----------------------------------------------------------')
    if B < 0.5 and C > 0.85:
        print('  => B intelligible, C is not: the MEL is the bottleneck, NOT the vocoder.')
        print('     A neural vocoder cannot fix C — it would only make wrong words sound clean.')
    elif B > 0.85:
        print('  => B is ALSO bad: the vocoder IS lossy here and worth replacing.')
    else:
        print('  => intermediate; read the per-utterance rows.')
    print('══════════════════════════════════════════════════════════')
    for r in rows[:4]:
        print(f'\n  REF : {r["ref"][:80]}')
        print(f'  orig: {r["A_orig"]["hyp"].strip()[:80]}')
        print(f'  B-GL: {r["B_realGL"]["hyp"].strip()[:80]}')
        print(f'  C-GEN:{r["C_genGL"]["hyp"].strip()[:80]}')
    json.dump({'n': len(rows), 'whisper': args.model, 'arm': args.arm,
               'WER': {'A_orig': A, 'B_realGL_vocoder_ceiling': B, 'C_genGL': C},
               'rows': rows}, open(os.path.join(OUT_DIR, 'asr_check.json'), 'w'), indent=2)
    print('\n[asr] wrote outputs/asr_check.json + outputs/audio_asr/', flush=True)


if __name__ == '__main__':
    main()
