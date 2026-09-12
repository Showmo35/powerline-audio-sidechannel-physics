"""
model.py
--------
Module 6: Hybrid CTC/Attention model (ESPnet-style).

Conformer encoder + CTC head + Transformer attention decoder.
Joint training with CTC loss (0.7) + attention cross-entropy (0.3).

The attention decoder cross-attends to encoder states, providing
acoustically grounded decoding with label smoothing to prevent
hallucination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

VOCAB_SIZE     = 29   # CTC vocab (blank + 26 letters + space + apostrophe)
BLANK_IDX      = 0
SOS_IDX        = 29   # Start of sequence for attention decoder
EOS_IDX        = 30   # End of sequence
DEC_VOCAB_SIZE = 31   # 29 + SOS + EOS

IDX_TO_CHAR = {0: '', 1: ' ', 28: "'"}
for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
    IDX_TO_CHAR[i + 2] = c


# =============================================================================
# Conformer Components (same as Module 5)
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm)
        return self.dropout(out)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, expansion=4, dropout=dropout)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, expansion=4, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.self_attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class Conv2dSubsampling(nn.Module):
    def __init__(self, n_mels, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.linear = nn.Linear(32 * (n_mels // 4), d_model)

    def forward(self, x):
        x = self.conv(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        return self.linear(x)


# =============================================================================
# Transformer Decoder
# =============================================================================

class TransformerDecoderLayer(nn.Module):
    """Single decoder layer: self-attn -> cross-attn -> feed-forward."""

    def __init__(self, d_model=256, num_heads=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        # Self-attention (causal)
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        # Cross-attention to encoder
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.cross_attn(tgt2, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        # Feed-forward
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.dropout(self.ff(tgt2))
        return tgt


class TransformerDecoder(nn.Module):
    """
    Autoregressive Transformer decoder for attention-based ASR.

    Input:  target tokens (B, S) + encoder memory (B, T, D)
    Output: logits (B, S, DEC_VOCAB_SIZE)
    """

    def __init__(self, vocab_size=DEC_VOCAB_SIZE, d_model=256, num_heads=4,
                 n_layers=3, max_len=210, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_tokens, memory):
        B, S = tgt_tokens.shape
        positions = torch.arange(S, device=tgt_tokens.device).unsqueeze(0)
        x = self.embed(tgt_tokens) + self.pos_embed(positions)

        # Causal mask: prevent attending to future tokens
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            x = layer(x, memory, tgt_mask=causal_mask)

        x = self.norm(x)
        return self.output_proj(x)


# =============================================================================
# Hybrid CTC/Attention Model
# =============================================================================

class HybridCTCAttentionModel(nn.Module):
    """
    Hybrid CTC/Attention model (ESPnet-style).

    Conformer encoder with dual output:
      - CTC head for CTC loss
      - Transformer decoder for attention loss

    Input : (B, 1, 80, T) - mel spectrogram
    Output:
      Training:  ctc_logits (T', B, V_ctc), attn_logits (B, S, V_dec)
      Inference: ctc_logits (T', B, V_ctc)
    """

    def __init__(self, n_mels=80, ctc_vocab=VOCAB_SIZE,
                 dec_vocab=DEC_VOCAB_SIZE, d_model=256,
                 enc_layers=6, dec_layers=3, num_heads=4,
                 conv_kernel=31, dropout=0.1):
        super().__init__()
        self.subsample = Conv2dSubsampling(n_mels, d_model)
        self.encoder = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel, dropout)
            for _ in range(enc_layers)
        ])
        self.ctc_head = nn.Linear(d_model, ctc_vocab)
        self.decoder = TransformerDecoder(
            dec_vocab, d_model, num_heads, dec_layers,
            max_len=210, dropout=dropout)

    def encode(self, x):
        """Encode mel spectrogram. x: (B, 1, 80, T) -> (B, T//4, d_model)"""
        x = self.subsample(x)
        for block in self.encoder:
            x = block(x)
        return x

    def forward(self, x, tgt_tokens=None):
        """
        Args:
            x: (B, 1, 80, T) mel spectrogram
            tgt_tokens: (B, S) decoder input tokens (SOS-prepended).
                        If None, only returns CTC logits.
        """
        enc_out = self.encode(x)
        ctc_logits = self.ctc_head(enc_out).permute(1, 0, 2)  # (T, B, V)

        if tgt_tokens is not None:
            attn_logits = self.decoder(tgt_tokens, enc_out)    # (B, S, V_dec)
            return ctc_logits, attn_logits
        else:
            return ctc_logits

    def greedy_decode_attn(self, enc_out, max_len=200):
        """Autoregressive greedy decode with attention decoder."""
        B = enc_out.shape[0]
        device = enc_out.device
        ys = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = self.decoder(ys, enc_out)     # (B, S, V)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == EOS_IDX).all():
                break

        # Convert to text (skip SOS, stop at EOS)
        texts = []
        for seq in ys:
            chars = []
            for idx in seq[1:].tolist():  # skip SOS
                if idx == EOS_IDX:
                    break
                char = IDX_TO_CHAR.get(idx, '')
                chars.append(char)
            texts.append(''.join(chars))
        return texts

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def greedy_decode(logits):
    """CTC greedy decode. logits: (T, B, V) -> list of strings."""
    preds = logits.argmax(dim=-1).T
    texts = []
    for seq in preds:
        chars = []
        prev = BLANK_IDX
        for idx in seq.tolist():
            if idx != prev and idx != BLANK_IDX:
                chars.append(IDX_TO_CHAR.get(idx, ''))
            prev = idx
        texts.append(''.join(chars))
    return texts


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    model = HybridCTCAttentionModel(
        n_mels=80, d_model=256, enc_layers=6, dec_layers=3,
        num_heads=4, conv_kernel=31, dropout=0.1)
    print(f"HybridCTCAttentionModel params: {model.count_params():,}")

    x = torch.randn(2, 1, 80, 498)
    tgt = torch.randint(0, DEC_VOCAB_SIZE, (2, 50))

    # Training mode
    ctc_logits, attn_logits = model(x, tgt)
    print(f"Input:       {x.shape}")
    print(f"CTC logits:  {ctc_logits.shape}")
    print(f"Attn logits: {attn_logits.shape}")

    # Inference mode
    ctc_only = model(x)
    print(f"CTC only:    {ctc_only.shape}")

    texts = greedy_decode(ctc_only)
    print(f"CTC decoded: {texts}")
    print("Test passed.")
