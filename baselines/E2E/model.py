"""
model.py
--------
End-to-end powerline -> text model components.

Path 1:
  U-Net envelope recon + CTC head (direct envelope -> text).

Path 2:
  U-Net envelope recon + envelope-conditioned autoregressive decoder.

Both heads can be trained jointly in one end-to-end graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared vocabulary:
#   0: blank/pad
#   1: space
#   2..27: a-z
#   28: apostrophe
VOCAB = {" ": 1, "'": 28}
for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    VOCAB[c] = i + 2
VOCAB_SIZE = 29
BLANK_IDX = 0
IDX_TO_CHAR = {v: k for k, v in VOCAB.items()}
IDX_TO_CHAR[BLANK_IDX] = ""

# ─── UNet building blocks ─────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f = self.conv(x)
        return self.pool(f), f


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.1):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])
        return self.conv(torch.cat([x, skip], dim=1))


# ─── U-Net ────────────────────────────────────────────────────────────────────

class PowerlineUNet(nn.Module):
    """Bandpass PL mel → predicted clean mel spectrogram."""

    def __init__(self, in_ch=1, base_ch=64, dropout=0.1):
        super().__init__()
        b = base_ch
        self.e1 = EncoderBlock(in_ch, b,     dropout)
        self.e2 = EncoderBlock(b,     b*2,   dropout)
        self.e3 = EncoderBlock(b*2,   b*4,   dropout)
        self.e4 = EncoderBlock(b*4,   b*8,   dropout)

        self.bottleneck = nn.Sequential(
            ResBlock(b*8, dropout), ResBlock(b*8, dropout))

        self.d4 = DecoderBlock(b*8,  b*8, b*4, dropout)
        self.d3 = DecoderBlock(b*4,  b*4, b*2, dropout)
        self.d2 = DecoderBlock(b*2,  b*2, b,   dropout)
        self.d1 = DecoderBlock(b,    b,   b,   dropout)

        self.out_conv = nn.Conv2d(b, in_ch, 1)

    def forward(self, x):
        x1, s1 = self.e1(x)
        x2, s2 = self.e2(x1)
        x3, s3 = self.e3(x2)
        x4, s4 = self.e4(x3)

        x4 = self.bottleneck(x4)

        d = self.d4(x4, s4)
        d = self.d3(d,  s3)
        d = self.d2(d,  s2)
        d = self.d1(d,  s1)

        return self.out_conv(d) + x  # global residual


# ─── CTC Encoder ──────────────────────────────────────────────────────────────

class CTCEncoder(nn.Module):
    """
    Predicted mel → character logits.

    Input : (B, 1, M, T)   M=80 mel bins, T=~498 frames
    Output: (T', B, V)     T'=T//4, V=VOCAB_SIZE  (CTC format)
    """

    def __init__(self, n_mels=80, hidden=256, n_layers=2,
                 vocab_size=VOCAB_SIZE, dropout=0.2):
        super().__init__()
        # Two stride-2 convs: T → T/4
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        # After two stride-2 convs on (1, 80, T): shape (32, 20, T//4)
        # Flatten freq → 32*20 = 640 input to GRU
        conv_out_freq = n_mels // 4
        gru_in = 32 * conv_out_freq

        self.gru = nn.GRU(
            gru_in, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout  = nn.Dropout(dropout)
        self.linear   = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (B, 1, 80, T)
        c = self.conv(x)                    # (B, 32, 20, T//4)
        B, C, M, T = c.shape
        c = c.permute(0, 3, 1, 2).reshape(B, T, C * M)  # (B, T, 640)
        out, _ = self.gru(c)                # (B, T, 2*hidden)
        out    = self.dropout(out)
        logits = self.linear(out)           # (B, T, V)
        return logits.permute(1, 0, 2)      # (T, B, V)  — CTC format


# ─── Character LM (text-only pretraining helper) ─────────────────────────────

class CharLM(nn.Module):
    """
    Character-level LSTM language model.
    Pre-trained on Alice in Wonderland text (see train_lm.py).
    Used for shallow fusion during beam search at inference.

    Input : (B, 1) integer tokens
    Output: (B, V) log-probabilities over next character
    State : (h, c) for caching across beam steps
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64,
                 hidden=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=BLANK_IDX)
        self.lstm    = nn.LSTM(embed_dim, hidden, num_layers=n_layers,
                               batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.linear  = nn.Linear(hidden, vocab_size)
        self.n_layers = n_layers
        self.hidden   = hidden

    def forward(self, x, state=None):
        """
        x    : (B, 1) or (B, T) integer tokens
        state: (h, c) or None
        Returns logprobs (B, V), new_state
        """
        emb  = self.embed(x)                # (B, T, embed_dim)
        out, state = self.lstm(emb, state)  # (B, T, hidden)
        logits = self.linear(out[:, -1, :]) # (B, V) — last step
        return F.log_softmax(logits, dim=-1), state

    def init_state(self, batch_size, device):
        h = torch.zeros(self.n_layers, batch_size, self.hidden, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden, device=device)
        return (h, c)


# ─── Envelope-conditioned autoregressive decoder ──────────────────────────────

class EnvelopeConditionedDecoder(nn.Module):
    """
    Autoregressive decoder conditioned on predicted envelope spectrogram.

    Inputs
      mel      : (B, 1, 80, T)
      tokens_in: (B, L) teacher-forced prefix tokens

    Output
      logits   : (B, L, V)
    """

    def __init__(
        self,
        n_mels=80,
        vocab_size=VOCAB_SIZE,
        d_model=256,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        max_text_len=256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_text_len = max_text_len

        # Acoustic memory encoder over time.
        self.acoustic = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
        )
        self.mem_norm = nn.LayerNorm(d_model)

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=BLANK_IDX)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_text_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(length, device):
        # True means "blocked" for nn.TransformerDecoder.
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, mel, tokens_in):
        # mel: (B,1,80,T) -> (B,80,T)
        x = mel[:, 0]
        mem = self.acoustic(x).transpose(1, 2)  # (B,Tm,D)
        mem = self.mem_norm(mem)

        B, L = tokens_in.shape
        if L > self.max_text_len:
            raise ValueError(f"tokens_in length {L} exceeds max_text_len={self.max_text_len}")

        tok = self.token_embed(tokens_in)
        tok = tok + self.pos_embed[:, :L, :]
        mask = self._causal_mask(L, tokens_in.device)

        out = self.decoder(tgt=tok, memory=mem, tgt_mask=mask)
        out = self.out_norm(out)
        return self.out_proj(out)  # (B,L,V)


# ─── Combined E2E model ───────────────────────────────────────────────────────

class E2EModel(nn.Module):
    def __init__(
        self,
        base_ch=64,
        gru_hidden=256,
        gru_layers=2,
        dropout=0.1,
        dec_d_model=256,
        dec_heads=4,
        dec_layers=3,
        dec_max_text_len=256,
    ):
        super().__init__()
        self.unet = PowerlineUNet(base_ch=base_ch, dropout=dropout)
        self.encoder = CTCEncoder(hidden=gru_hidden, n_layers=gru_layers,
                                  dropout=dropout)
        self.cond_decoder = EnvelopeConditionedDecoder(
            d_model=dec_d_model,
            n_heads=dec_heads,
            n_layers=dec_layers,
            dropout=dropout,
            max_text_len=dec_max_text_len,
        )

    def forward(self, noisy, tokens_in=None):
        predicted = self.unet(noisy)          # (B, 1, 80, T)
        ctc_logits = self.encoder(predicted)  # (T', B, V)
        lm_logits = None
        if tokens_in is not None:
            lm_logits = self.cond_decoder(predicted, tokens_in)  # (B, L, V)
        return predicted, ctc_logits, lm_logits


# ─── Loss functions ───────────────────────────────────────────────────────────

class MSTFTLoss(nn.Module):
    """Multi-scale STFT loss (spectral convergence + L1 magnitude)."""

    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

    def forward(self, pred, target):
        loss = 0.0
        for s in self.scales:
            # Pool along time to simulate different scales
            p = F.avg_pool2d(pred,   (1, s), (1, s)) if s > 1 else pred
            t = F.avg_pool2d(target, (1, s), (1, s)) if s > 1 else target
            # Spectral convergence
            sc = torch.norm(t - p, p='fro') / (torch.norm(t, p='fro') + 1e-8)
            # L1 magnitude
            l1 = F.l1_loss(p, t)
            loss = loss + sc + l1
        return loss / len(self.scales)


class E2ELoss(nn.Module):
    """
    Combined reconstruction + CTC loss.

    alpha * (L1 + MSTFT) + beta * CTC
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, blank_idx=BLANK_IDX):
        super().__init__()
        self.alpha = alpha   # recon weight
        self.beta = beta     # ctc weight
        self.gamma = gamma   # conditioned decoder CE weight
        self.mstft = MSTFTLoss()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean',
                                   zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=blank_idx)

    def forward(self, predicted, target_mel, ctc_logits, text, text_len, lm_logits=None):
        """
        predicted  : (B, 1, M, T)   reconstructed mel
        target_mel : (B, 1, M, T)   clean MP3 mel
        ctc_logits : (T', B, V)     CTC logits
        text       : (B, max_text)  integer targets (padded)
        text_len   : (B,)           true text lengths
        lm_logits  : (B, max_text, V) or None
        """
        # Reconstruction losses
        l1 = F.l1_loss(predicted, target_mel)
        mstft = self.mstft(predicted, target_mel)
        recon = l1 + mstft

        # CTC loss
        t_out = ctc_logits.shape[0]
        bsz = ctc_logits.shape[1]
        log_probs = F.log_softmax(ctc_logits, dim=-1)
        input_lens = torch.full((bsz,), t_out, dtype=torch.long, device=ctc_logits.device)

        # Flatten text targets (CTCLoss expects 1D concatenated)
        targets_flat = []
        for i in range(bsz):
            tl = int(text_len[i].item())
            if tl > 0:
                targets_flat.append(text[i, :tl])
        if targets_flat:
            targets_cat = torch.cat(targets_flat)
            ctc = self.ctc_loss(log_probs, targets_cat, input_lens, text_len)
        else:
            ctc = torch.zeros((), device=ctc_logits.device)

        seq = torch.zeros((), device=ctc_logits.device)
        if lm_logits is not None:
            bsz, max_t, vocab = lm_logits.shape
            seq = self.ce_loss(
                lm_logits.reshape(bsz * max_t, vocab),
                text.reshape(bsz * max_t),
            )

        total = (self.alpha * recon) + (self.beta * ctc) + (self.gamma * seq)
        return total, l1.item(), mstft.item(), ctc.item(), seq.item()
