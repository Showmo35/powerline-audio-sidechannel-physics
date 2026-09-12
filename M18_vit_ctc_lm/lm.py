#!/usr/bin/env python3
"""
lm.py — a small from-scratch CHARACTER language model + CTC prefix beam search.

The LM is trained ONLY on the training-split transcripts (no outside knowledge),
so any WER improvement it produces on held-out chunks over the greedy/null result
is genuine acoustic recovery — not memorised text. Used for shallow-fusion beam
search over the ViT-CTC emissions.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import text as T

# LM vocab = CTC label vocab minus blank, plus BOS. Chars are text.IDX_TO_CHAR[1..28].
CHARS = [T.IDX_TO_CHAR[i] for i in range(1, T.VOCAB_SIZE)]   # ' ', a..z, '
LM_STOI = {c: i + 1 for i, c in enumerate(CHARS)}           # 0 = BOS/pad
LM_BOS = 0
LM_VSIZE = len(CHARS) + 1
# map a CTC label id (1..28) → LM token id
CTC_TO_LM = {i: LM_STOI[T.IDX_TO_CHAR[i]] for i in range(1, T.VOCAB_SIZE)}


class CharLM(nn.Module):
    def __init__(self, d=256, layers=4, heads=4, max_len=320):
        super().__init__()
        self.emb = nn.Embedding(LM_VSIZE, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, LM_VSIZE)
        self.max_len = max_len

    def forward(self, x):                                   # x: (B, L) LM tokens
        L = x.shape[1]
        h = self.emb(x) + self.pos[:, :L]
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        h = self.tf(h, mask=mask)
        return self.head(self.norm(h))                      # (B, L, V)

    @torch.no_grad()
    def logprob_next(self, prefix_lm_tokens, device):
        """log P(next char | prefix) over LM vocab. prefix_lm_tokens: list[int]."""
        x = torch.tensor([[LM_BOS] + prefix_lm_tokens[-(self.max_len - 1):]], device=device)
        return F.log_softmax(self.forward(x)[0, -1], dim=-1)


def encode_text_lm(s):
    return [LM_STOI[c] for c in T.normalize(s) if c in LM_STOI]


def _logadd(a, b):
    if a <= -1e29:
        return b
    if b <= -1e29:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


# ── CTC prefix beam search with optional char-LM shallow fusion ────────────────
def beam_search(logp, lm=None, device='cpu', beam=24, alpha=0.4, beta=1.0, lm_cache=None):
    """
    logp: (Tp, V) CTC log-probs (V=29, blank=0). Returns best string.
    A prefix is a tuple of CTC label ids; its LM tokens derive directly from it,
    so no separate bookkeeping is needed. alpha = LM weight, beta = insertion bonus.
    lm_cache: optional dict caching lm.logprob_next per prefix across the utterance.
    """
    Tp, V = logp.shape
    NEG = -1e30
    if lm_cache is None:
        lm_cache = {}

    def lm_next(prefix):
        if prefix not in lm_cache:
            toks = [CTC_TO_LM[c] for c in prefix]
            lm_cache[prefix] = lm.logprob_next(toks, device)
        return lm_cache[prefix]

    beams = {(): [0.0, NEG]}                    # prefix → [p_blank, p_nonblank]
    for t in range(Tp):
        lt = logp[t]
        nxt = {}

        def get(prefix):
            if prefix not in nxt:
                nxt[prefix] = [NEG, NEG]
            return nxt[prefix]

        for prefix, (pb, pnb) in beams.items():
            ptot = _logadd(pb, pnb)
            e = get(prefix)
            e[0] = _logadd(e[0], ptot + lt[0].item())               # blank → same prefix
            if prefix:                                              # repeat last → non-blank
                e[1] = _logadd(e[1], pnb + lt[prefix[-1]].item())
            lm_lp = lm_next(prefix) if (lm is not None and alpha > 0) else None
            for c in range(1, V):                                   # extend by char c
                src = pb if (prefix and c == prefix[-1]) else ptot
                np_ = src + lt[c].item()
                if np_ <= NEG:
                    continue
                ne = get(prefix + (c,))
                score = np_ + (alpha * lm_lp[CTC_TO_LM[c]].item() + beta if lm_lp is not None else 0.0)
                ne[1] = _logadd(ne[1], score)
        beams = dict(sorted(nxt.items(),
                            key=lambda kv: _logadd(kv[1][0], kv[1][1]), reverse=True)[:beam])
    best = max(beams.items(), key=lambda kv: _logadd(kv[1][0], kv[1][1]))[0]
    return ''.join(T.IDX_TO_CHAR[c] for c in best)


def greedy(logp):
    ids = logp.argmax(-1).tolist()
    out, prev = [], 0
    for i in ids:
        if i != prev and i != 0:
            out.append(T.IDX_TO_CHAR[i])
        prev = i
    return ''.join(out)
