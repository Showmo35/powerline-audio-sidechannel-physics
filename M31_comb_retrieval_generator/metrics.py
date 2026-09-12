#!/usr/bin/env python3
"""
metrics.py — the retrieval embedding + losses.

BASIC (M22 original, the "too basic" objective we compare against):
  embed_raw(mel) = flatten -> mean-center -> L2-norm  ⇒ cosine == Pearson r.
  loss = supcon(raw) + (1 - cos(gen_i, real_i))           (raw-pixel similarity)

LEARNED (the upgrade): a trained projection head g(flatten(mel)) -> unit embedding,
with SUPERVISED CONTRASTIVE (multi-positive over the closed vocab) + PROXYANCHOR
(learnable per-class prototypes) losses — deep metric learning in place of raw-pixel
Pearson correlation. Retrieval (train loss AND eval kNN) runs in this learned space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── BASIC raw-Pearson embedding (M22) ──
def embed_raw(mel_bt):
    z = mel_bt.reshape(mel_bt.shape[0], -1)
    z = z - z.mean(dim=1, keepdim=True)
    return F.normalize(z, dim=1)


# ── LEARNED projection head ──
class Projector(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim))

    def forward(self, mel_bt):
        z = mel_bt.reshape(mel_bt.shape[0], -1)
        return F.normalize(self.net(z), dim=1)


# ── supervised contrastive (multi-positive) ──
def supcon(anchors, keys, a_lab, k_lab, tau=0.1):
    sim = (anchors @ keys.t()) / tau
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    logp = sim - torch.log(torch.exp(sim).sum(dim=1, keepdim=True) + 1e-9)
    pos = (a_lab[:, None] == k_lab[None, :]).float()
    return -((pos * logp).sum(1) / pos.sum(1).clamp(min=1)).mean()


# ── ProxyAnchor (Kim et al. CVPR'20): learnable per-class prototypes ──
def proxy_anchor(x, labels, proxies, K, alpha=32.0, delta=0.1):
    P = F.normalize(proxies, dim=1)
    sim = x @ P.t()                                   # (N, K) cosine
    onehot = F.one_hot(labels, K).float()             # (N, K)
    neg = 1.0 - onehot
    with_pos = (onehot.sum(0) > 0).float()            # (K,) proxies present in batch
    n_pos = with_pos.sum().clamp(min=1)
    pos_term = torch.log1p((torch.exp(-alpha * (sim - delta)) * onehot).sum(0))  # (K,)
    neg_term = torch.log1p((torch.exp( alpha * (sim + delta)) * neg).sum(0))     # (K,)
    return (pos_term * with_pos).sum() / n_pos + neg_term.sum() / P.shape[0]
