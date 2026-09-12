"""
model.py
--------
FullSubNet+-style complex mask estimator.

Input features per frame/bin:
- noisy real
- noisy imag
- noisy log-magnitude

Output:
- complex ratio mask (CRM): 2 channels [mask_real, mask_imag]
"""

import torch
import torch.nn as nn
import torch.nn.functional as tF


class FullSubNetPlusLite(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        fb_channels: int = 48,
        fb_hidden: int = 64,
        sb_hidden: int = 96,
        subband_size: int = 7,
    ):
        super().__init__()
        if subband_size % 2 == 0:
            raise ValueError('subband_size must be odd.')

        self.subband_size = subband_size

        self.fullband_conv = nn.Sequential(
            nn.Conv2d(in_ch, fb_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(fb_channels, fb_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.fullband_gru = nn.GRU(
            input_size=fb_channels,
            hidden_size=fb_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.fullband_proj = nn.Linear(fb_hidden * 2, fb_hidden)

        subband_input_dim = in_ch * subband_size + fb_hidden
        self.subband_gru = nn.GRU(
            input_size=subband_input_dim,
            hidden_size=sb_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.mask_head = nn.Linear(sb_hidden * 2, 2)

    @staticmethod
    def _freq_unfold(x: torch.Tensor, kernel: int):
        """
        x: (B, C, F, T)
        returns (B, F, T, C * kernel)
        """
        bsz, channels, n_freq, n_time = x.shape
        pad = kernel // 2

        x_bt = x.permute(0, 3, 1, 2).reshape(bsz * n_time, channels, n_freq)
        x_bt = tF.pad(x_bt, (pad, pad), mode='replicate')
        x_bt = x_bt.unfold(dimension=2, size=kernel, step=1)  # (B*T, C, F, K)
        x_bt = x_bt.permute(0, 2, 1, 3).contiguous()          # (B*T, F, C, K)
        x_bt = x_bt.view(bsz, n_time, n_freq, channels * kernel)
        return x_bt.permute(0, 2, 1, 3).contiguous()           # (B, F, T, C*K)

    def forward(self, feat: torch.Tensor):
        """
        feat: (B, 3, F, T)
        returns CRM: (B, 2, F, T)
        """
        bsz, _, n_freq, n_time = feat.shape

        fb = self.fullband_conv(feat)                 # (B, Cfb, F, T)
        fb_pool = fb.mean(dim=2).permute(0, 2, 1)    # (B, T, Cfb)

        fb_ctx, _ = self.fullband_gru(fb_pool)        # (B, T, 2*Hfb)
        fb_ctx = self.fullband_proj(fb_ctx)           # (B, T, Hfb)
        fb_ctx = fb_ctx.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, n_freq, -1)
        fb_ctx = fb_ctx.permute(0, 2, 3, 1).contiguous()  # (B, F, T, Hfb)

        sb_feat = self._freq_unfold(feat, self.subband_size)   # (B, F, T, C*K)
        sb_in = torch.cat([sb_feat, fb_ctx], dim=-1)           # (B, F, T, C*K+Hfb)

        sb_in = sb_in.view(bsz * n_freq, n_time, -1)
        sb_out, _ = self.subband_gru(sb_in)                    # (B*F, T, 2*Hsb)

        mask = self.mask_head(sb_out)                          # (B*F, T, 2)
        mask = mask.view(bsz, n_freq, n_time, 2).permute(0, 3, 1, 2).contiguous()

        # Keep masks bounded and stable.
        return 2.0 * torch.tanh(mask)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = FullSubNetPlusLite()
    x = torch.randn(2, 3, 257, 397)
    y = model(x)
    print('Input :', x.shape)
    print('Output:', y.shape)
    print('Params:', model.count_params())
