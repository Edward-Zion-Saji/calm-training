import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-Scale STFT Discriminator (MS-STFTD)
# ─────────────────────────────────────────────────────────────────────────────
# Primary discriminator for audio codecs (used in EnCodec, Mimi, Stable Audio).
# Operates on complex STFT spectrograms at 5 different FFT resolutions.
# Each sub-discriminator is a 2D convolutional network on the [freq × time] plane.

class STFTDiscriminator(nn.Module):
    """Single-resolution complex STFT discriminator."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int,
                 filters: int = 32, max_filters: int = 1024,
                 n_layers: int = 4):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # 2D conv stack on the STFT magnitude/phase planes
        in_ch = 2   # real + imaginary parts
        layers = []
        for i in range(n_layers):
            out_ch = min(filters * (2 ** i), max_filters)
            layers += [
                nn.utils.weight_norm(
                    nn.Conv2d(in_ch, out_ch,
                              kernel_size=(3, 9) if i == 0 else (3, 3),
                              stride=(1, 2) if i < n_layers - 1 else (1, 1),
                              padding=(1, 4) if i == 0 else (1, 1))
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        layers.append(
            nn.utils.weight_norm(
                nn.Conv2d(in_ch, 1, kernel_size=(3, 3), padding=(1, 1))
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, audio: torch.Tensor):
        """
        audio: [B, 1, T]
        Returns: (logits, feature_maps)
        """
        audio = audio.squeeze(1)   # [B, T]
        stft  = torch.stft(
            audio, self.n_fft, self.hop_length, self.win_length,
            window=torch.hann_window(self.win_length, device=audio.device),
            return_complex=True,
        )                          # [B, F, T_s]
        # Stack real and imaginary as channels
        x = torch.stack([stft.real, stft.imag], dim=1)   # [B, 2, F, T_s]

        feature_maps = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feature_maps.append(x)

        return x, feature_maps


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    5-resolution MS-STFT discriminator as in EnCodec/Mimi.
    n_ffts:       [2048, 1024,  512, 256, 128]
    hop_lengths:  [ 512,  256,  128,  64,  32]
    win_lengths:  [2048, 1024,  512, 256, 128]
    """

    RESOLUTIONS = [
        (2048, 512,  2048),
        (1024, 256,  1024),
        ( 512, 128,   512),
        ( 256,  64,   256),
        ( 128,  32,   128),
    ]

    def __init__(self, filters: int = 32):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(n, h, w, filters=filters)
            for n, h, w in self.RESOLUTIONS
        ])

    def forward(self, audio: torch.Tensor):
        all_logits, all_features = [], []
        for disc in self.discriminators:
            logits, feats = disc(audio)
            all_logits.append(logits)
            all_features.append(feats)
        return all_logits, all_features


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Period Discriminator (MPD)
# ─────────────────────────────────────────────────────────────────────────────
# From HiFi-GAN. Reshapes the waveform into a 2D grid with period p
# and applies 2D convolutions. Each period captures different harmonic structures.

class PeriodDiscriminator(nn.Module):
    """Single-period sub-discriminator."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3,
                 channels: list[int] | None = None):
        super().__init__()
        self.period = period
        if channels is None:
            channels = [1, 32, 128, 512, 1024, 1024]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv2d(
                        channels[i], channels[i + 1],
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        layers.append(
            nn.utils.weight_norm(
                nn.Conv2d(channels[-1], 1, kernel_size=(3, 1), padding=(1, 0))
            )
        )
        self.net = nn.ModuleList(layers)

    def forward(self, audio: torch.Tensor):
        """audio: [B, 1, T]"""
        x = audio.squeeze(1)   # [B, T]
        B, T = x.shape

        # Pad to multiple of period
        pad_len = (self.period - T % self.period) % self.period
        x = F.pad(x, (0, pad_len))
        x = x.view(B, 1, -1, self.period)   # [B, 1, T//p, p]

        feature_maps = []
        for layer in self.net:
            x = layer(x)
            if hasattr(layer, '__len__') and len(layer) > 1:
                feature_maps.append(x)
        return x, feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """
    MPD with periods [2, 3, 5, 7, 11] — prime numbers ensure
    each sub-discriminator attends to non-overlapping harmonic structures.
    """

    PERIODS = [2, 3, 5, 7, 11]

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in self.PERIODS
        ])

    def forward(self, audio: torch.Tensor):
        all_logits, all_features = [], []
        for disc in self.discriminators:
            logits, feats = disc(audio)
            all_logits.append(logits)
            all_features.append(feats)
        return all_logits, all_features


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-Scale Waveform Discriminator (MSD)
# ─────────────────────────────────────────────────────────────────────────────
# From MelGAN. Operates on raw waveform at 3 scales (raw, ×2, ×4 downsampled).

class ScaleDiscriminator(nn.Module):
    """Single waveform-scale discriminator."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.net = nn.ModuleList([
            norm(nn.Conv1d(1,   128, 15, stride=1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, stride=2, padding=20, groups=4)),
            norm(nn.Conv1d(128, 256, 41, stride=2, padding=20, groups=16)),
            norm(nn.Conv1d(256, 512, 41, stride=4, padding=20, groups=16)),
            norm(nn.Conv1d(512, 1024, 41, stride=4, padding=20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 41, stride=1, padding=20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 5, stride=1, padding=2)),
            norm(nn.Conv1d(1024, 1, 3, stride=1, padding=1)),
        ])
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, audio: torch.Tensor):
        feature_maps = []
        x = audio
        for i, layer in enumerate(self.net[:-1]):
            x = self.activation(layer(x))
            feature_maps.append(x)
        x = self.net[-1](x)
        return x, feature_maps


class MultiScaleDiscriminator(nn.Module):
    """MSD at 3 downsampling levels."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),   # raw
            ScaleDiscriminator(),                          # ×2
            ScaleDiscriminator(),                          # ×4
        ])
        self.pooling = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, audio: torch.Tensor):
        all_logits, all_features = [], []
        x = audio
        for pool, disc in zip(self.pooling, self.discriminators):
            x_d = pool(x)
            logits, feats = disc(x_d)
            all_logits.append(logits)
            all_features.append(feats)
        return all_logits, all_features


# ─────────────────────────────────────────────────────────────────────────────
# Combined discriminator wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CombinedDiscriminator(nn.Module):
    """
    Runs all three discriminators and returns combined logits + feature maps.
    Used for both adversarial loss and feature matching loss.
    """

    def __init__(self):
        super().__init__()
        self.msstftd = MultiScaleSTFTDiscriminator()
        self.mpd     = MultiPeriodDiscriminator()
        self.msd     = MultiScaleDiscriminator()

    def forward(self, audio: torch.Tensor):
        results = {}
        results["msstftd"] = self.msstftd(audio)
        results["mpd"]     = self.mpd(audio)
        results["msd"]     = self.msd(audio)
        return results   # dict of (logits_list, features_list) per discriminator
