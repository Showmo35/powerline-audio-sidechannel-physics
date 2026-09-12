"""
model.py
--------
Experiment 1: Pure CTC encoder -- no U-Net, raw noisy mel -> text directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary (same as E2E)
VOCAB = {" ": 1, "'": 28}
for _i, _c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    VOCAB[_c] = _i + 2
VOCAB_SIZE = 29
BLANK_IDX = 0
IDX_TO_CHAR = {v: k for k, v in VOCAB.items()}
IDX_TO_CHAR[BLANK_IDX] = ""


class PureCTCEncoder(nn.Module):
    """
    Raw noisy mel -> character logits.  No denoising front-end.

    Input : (B, 1, 80, T)
    Output: (T', B, V)  where T' = T//4
    """

    def __init__(self, n_mels=80, hidden=256, n_layers=2,
                 vocab_size=VOCAB_SIZE, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        conv_out_freq = n_mels // 4
        gru_in = 32 * conv_out_freq

        self.gru = nn.GRU(
            gru_in, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        c = self.conv(x)
        B, C, M, T = c.shape
        c = c.permute(0, 3, 1, 2).reshape(B, T, C * M)
        out, _ = self.gru(c)
        out = self.dropout(out)
        logits = self.linear(out)
        return logits.permute(1, 0, 2)  # (T, B, V)


class PureCTCLoss(nn.Module):
    def __init__(self, blank_idx=BLANK_IDX):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean',
                                   zero_infinity=True)

    def forward(self, ctc_logits, text, text_len):
        t_out = ctc_logits.shape[0]
        bsz   = ctc_logits.shape[1]
        log_probs  = F.log_softmax(ctc_logits, dim=-1)
        input_lens = torch.full((bsz,), t_out, dtype=torch.long,
                                device=ctc_logits.device)
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
        return ctc


if __name__ == "__main__":
    m = PureCTCEncoder(hidden=256, n_layers=2)
    n = sum(p.numel() for p in m.parameters())
    print(f"PureCTC params: {n:,}")
    x = torch.randn(2, 1, 80, 498)
    logits = m(x)
    print(f"Input: {x.shape} -> Logits: {logits.shape}")
