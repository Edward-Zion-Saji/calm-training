import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale STFT Discriminator (MS-STFTD)
# ─────────────────────────────────────────────────────────────────────────────
# This is the ONLY discriminator used by Mimi (Moshi paper, Section 3.3.1).
# The paper references the Audiocraft repo for exact params:
#   n_ffts:      [1024, 2048, 512, 256, 128]
#   hop_lengths: [ 256,  512, 128,  64,  32]
#   win_lengths: [1024, 2048, 512, 256, 128]
# Loss: HINGE (not LSGAN) — confirmed from Audiocraft default config.
# MPD and MSD are NOT used by Mimi/CALM.

class STFTSubDiscriminator(nn.Module):
    """Single STFT-resolution discriminator operating on complex spectrogram."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int,
                 filters: int = 32, n_layers: int = 4):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        in_ch  = 2   # real + imaginary stacked as channels
        layers = []
        for i in range(n_layers):
            out_ch = min(filters * (2 ** i), 1024)
            layers += [
                nn.utils.weight_norm(
                    nn.Conv2d(in_ch, out_ch,
                              kernel_size=(3, 9) if i == 0 else (3, 3),
                              stride=(1, 2) if i < n_layers - 1 else (1, 1),
                              padding=(1, 4) if i == 0 else (1, 1))
                ),
                nn.LeakyReLU(0.3, inplace=True),   # slope=0.3 per Audiocraft
            ]
            in_ch = out_ch
        layers.append(
            nn.utils.weight_norm(
                nn.Conv2d(in_ch, 1, kernel_size=(3, 3), padding=(1, 1))
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, audio: torch.Tensor):
        """audio: [B, 1, T] → (logits, feature_maps)"""
        x_w = audio.squeeze(1)
        stft = torch.stft(
            x_w, self.n_fft, self.hop_length, self.win_length,
            window=torch.hann_window(self.win_length, device=x_w.device),
            return_complex=True,
        )
        x = torch.stack([stft.real, stft.imag], dim=1)   # [B, 2, F, T_s]

        feature_maps = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feature_maps.append(x)
        return x, feature_maps


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    5-resolution MS-STFT discriminator — the only discriminator used in Mimi.
    Parameters taken directly from the Audiocraft default config referenced
    by the Moshi paper (Defossez et al. 2024, Section 3.3.1).
    """

    # (n_fft, hop_length, win_length) per Audiocraft defaults
    RESOLUTIONS = [
        (1024, 256, 1024),
        (2048, 512, 2048),
        ( 512, 128,  512),
        ( 256,  64,  256),
        ( 128,  32,  128),
    ]

    def __init__(self, filters: int = 32):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTSubDiscriminator(n, h, w, filters=filters)
            for n, h, w in self.RESOLUTIONS
        ])

    def forward(self, audio: torch.Tensor):
        all_logits, all_features = [], []
        for disc in self.discriminators:
            logits, feats = disc(audio)
            all_logits.append(logits)
            all_features.append(feats)
        return all_logits, all_features
