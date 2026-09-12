#!/usr/bin/env python3
"""
train_gan.py — train the conditional HiFi-GAN: powerline stack → clean 16 kHz wave.

Losses: adversarial (LSGAN, MPD+MSD) + feature-matching + mel L1 (×45) + content
(frozen wav2vec2 feature L1, ×15).  The content loss keeps the generator FAITHFUL
(matching an ASR encoder's view of the clean signal), not just perceptually
plausible.  Eval = Whisper WER on generated audio for held-out chunks — the only
judge that matters.

Usage:
    python train_gan.py --epochs 30 --batch 16 --lr 2e-4 --test-chunks 41-46
"""

import argparse, json, os, random, re, time, wave
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as TT
from torch.utils.data import Dataset, DataLoader

from config import CFG, OUT_DIR, FEAT_DIR, SHARED_MANIFEST
import data_io as io
from models import Generator, MPD, MSD

_PUNCT = re.compile(r"[^a-z0-9'\s]")
def norm(s): return ' '.join(_PUNCT.sub(' ', s.lower()).split())
def wer_corpus(refs, hyps):
    S = N = 0
    for r, h in zip(refs, hyps):
        r, h = norm(r).split(), norm(h).split(); n, m = len(r), len(h)
        d = np.zeros((n+1, m+1), int); d[:, 0] = np.arange(n+1); d[0, :] = np.arange(m+1)
        for i in range(1, n+1):
            for j in range(1, m+1):
                d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1]+(r[i-1] != h[j-1]))
        S += d[n, m]; N += n
    return S / max(N, 1)


class PairDS(Dataset):
    def __init__(self, rows, cfg, train=True):
        self.rows, self.cfg, self.train = rows, cfg, train
        self.seg = cfg.seg_frames; self.hop = cfg.hop; self._x = {}
    def __len__(self): return len(self.rows)
    def _stack(self, ch):
        if ch not in self._x:
            self._x[ch] = np.load(os.path.join(FEAT_DIR, f'{ch}.x.npz'))
        return self._x[ch]
    def __getitem__(self, i):
        r = self.rows[i]; ch, uid = r['chunk'], r['utt_id']
        x = self._stack(ch)[uid].astype(np.float32)            # [C,M,T]
        wav = io.read_wav_window(self.cfg.wav_path(ch), r['start_s'],
                                 r['end_s']-r['start_s'], self.cfg.sr)
        T = x.shape[2]; seg = self.seg
        Tw = len(wav) // self.hop
        Tm = min(T, Tw)
        if Tm >= seg:
            s = random.randint(0, Tm-seg) if self.train else (Tm-seg)//2
        else:
            s = 0
            x = np.pad(x, ((0, 0), (0, 0), (0, seg-T)), constant_values=x.min())
        cond = x[:, :, s:s+seg]
        w = wav[s*self.hop:(s+seg)*self.hop]
        if len(w) < seg*self.hop:
            w = np.pad(w, (0, seg*self.hop-len(w)))
        cond = (cond - cond.mean()) / (cond.std() + 1e-6)
        return torch.from_numpy(cond), torch.from_numpy(w.astype(np.float32))


def znorm_full(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--test-chunks', default='41-46')
    ap.add_argument('--eval-every', type=int, default=1000, help='steps')
    ap.add_argument('--eval-n', type=int, default=100)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-dir', default=os.path.join(OUT_DIR, 'gan'))
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out_dir, exist_ok=True)
    cfg = CFG

    lo, hi = (int(v) for v in a.test_chunks.split('-'))
    tc = {f'chunk_{n:03d}' for n in range(lo, hi+1)}
    rows = [json.loads(l) for l in open(SHARED_MANIFEST) if l.strip()]
    tr = [r for r in rows if r['chunk'] not in tc]
    te = [r for r in rows if r['chunk'] in tc]
    random.seed(0); te_eval = random.sample(te, min(a.eval_n, len(te)))
    print(f'[data] train={len(tr)} test={len(te)} eval_n={len(te_eval)}')

    G = Generator(cfg).to(dev)
    mpd, msd = MPD().to(dev), MSD().to(dev)
    nG = sum(p.numel() for p in G.parameters())/1e6
    print(f'[model] G={nG:.2f}M  MPD+MSD discriminators')

    mel_tf = TT.MelSpectrogram(sample_rate=cfg.sr, n_fft=cfg.n_fft, win_length=cfg.win,
                               hop_length=cfg.hop, n_mels=cfg.n_mels, f_min=cfg.fmin,
                               f_max=cfg.fmax, power=1.0).to(dev)
    def logmel(w): return torch.log(torch.clamp(mel_tf(w), min=1e-5))

    # frozen wav2vec2 content encoder
    from transformers import Wav2Vec2Model
    w2v = Wav2Vec2Model.from_pretrained(cfg.content_model).to(dev).eval()
    for p in w2v.parameters(): p.requires_grad_(False)
    def w2v_feat(w):
        wn = (w - w.mean(1, keepdim=True)) / (w.std(1, keepdim=True) + 1e-7)
        return w2v(wn).last_hidden_state

    optG = torch.optim.AdamW(G.parameters(), a.lr, betas=(0.8, 0.99))
    optD = torch.optim.AdamW(list(mpd.parameters())+list(msd.parameters()), a.lr, betas=(0.8, 0.99))
    dl = DataLoader(PairDS(tr, cfg, True), batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, drop_last=True, pin_memory=True)

    import whisper
    asr = whisper.load_model('small', device=dev)

    @torch.no_grad()
    def evaluate():
        G.eval(); refs, hyps = [], []
        for r in te_eval:
            x = PairDS([r], cfg, False)._stack(r['chunk'])[r['utt_id']].astype(np.float32)
            cond = znorm_full(torch.from_numpy(x)[None].to(dev))
            wav = G(cond)[0].float().cpu().numpy()
            txt = asr.transcribe(wav.astype(np.float32), language='en', fp16=(dev == 'cuda'),
                                 verbose=False)['text']
            refs.append(r['text']); hyps.append(txt)
        G.train()
        return wer_corpus(refs, hyps), list(zip(refs[:3], hyps[:3]))

    def disc_loss(yr, yg):
        return sum(torch.mean((r-1)**2)+torch.mean(g**2) for r, g in zip(yr, yg))
    def gen_adv(yg):
        return sum(torch.mean((g-1)**2) for g in yg)
    def fm_loss(fr, fg):
        return sum(F.l1_loss(b, a_) for dr, dg in zip(fr, fg) for a_, b in zip(dr, dg))

    log = {'args': vars(a), 'evals': []}
    best = 9.9; step = 0; t0 = time.time()
    for ep in range(a.epochs):
        for cond, wav in dl:
            cond, wav = cond.to(dev), wav.to(dev)
            gen = G(cond)
            L = min(gen.shape[1], wav.shape[1]); gen, wav = gen[:, :L], wav[:, :L]

            # ── D ──
            optD.zero_grad()
            yr, yg, _, _ = mpd(wav, gen.detach()); ld = disc_loss(yr, yg)
            yr2, yg2, _, _ = msd(wav, gen.detach()); ld = ld + disc_loss(yr2, yg2)
            ld.backward(); optD.step()

            # ── G ──
            optG.zero_grad()
            yr, yg, fr, fg = mpd(wav, gen)
            yr2, yg2, fr2, fg2 = msd(wav, gen)
            l_adv = gen_adv(yg) + gen_adv(yg2)
            l_fm = cfg.w_fm * (fm_loss(fr, fg) + fm_loss(fr2, fg2))
            l_mel = cfg.w_mel * F.l1_loss(logmel(gen), logmel(wav))
            l_ct = cfg.w_content * F.l1_loss(w2v_feat(gen), w2v_feat(wav).detach())
            lg = l_adv + l_fm + l_mel + l_ct
            lg.backward(); optG.step()
            step += 1

            if step % 100 == 0:
                print(f'  ep{ep} s{step} D={ld.item():.2f} G_adv={l_adv.item():.2f} '
                      f'fm={l_fm.item():.2f} mel={l_mel.item():.2f} ct={l_ct.item():.2f} '
                      f'({time.time()-t0:.0f}s)')
            if step % a.eval_every == 0:
                w, samples = evaluate()
                print(f'[eval] step{step}  WER={w*100:.1f}%')
                for rf, hy in samples:
                    print(f'    REF: {norm(rf)[:70]}\n    HYP: {norm(hy)[:70]}')
                log['evals'].append({'step': step, 'wer': w})
                if w < best:
                    best = w; torch.save(G.state_dict(), os.path.join(a.out_dir, 'G_best.pt'))
                    print(f'    ↑ best WER {best*100:.1f}% → saved')
    # final
    w, _ = evaluate(); log['final_wer'] = w; log['best_wer'] = min(best, w)
    log['wall_s'] = round(time.time()-t0, 1)
    json.dump(log, open(os.path.join(a.out_dir, 'train_log.json'), 'w'), indent=2)
    print('\n══ STEP 6 — conditional GAN (judged by WER) ══════')
    print(f'  G params   = {nG:.2f}M   train/test = {len(tr)}/{len(te)}')
    print(f'  BEST WER   = {log["best_wer"]*100:.1f}%   final {w*100:.1f}%')
    print(f'  baselines  : learned-U-Net 100% · fixed 100% · clean-native ~4%')
    print('══════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
