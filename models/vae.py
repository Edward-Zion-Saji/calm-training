import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# Causal convolution helpers
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Conv1d that only looks at past samples (causal padding on the left)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, bias: bool = True):
        super().__init__()
        # For strided convolutions the receptive field in the input is
        # (kernel_size - 1) * dilation samples back. We pad exactly that many
        # zeros on the left so that the first output frame only sees t=0.
        self.pad = (kernel_size - 1) * dilation
        self.stride = stride
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      stride=stride, dilation=dilation,
                      padding=0, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """Transposed causal conv for the decoder upsampling blocks."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, bias: bool = True):
        super().__init__()
        self.stride = stride
        self.conv = nn.utils.weight_norm(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=0, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove the non-causal future samples introduced by transposed conv
        out = self.conv(x)
        # trim right: remove (kernel_size - stride) samples from the right
        trim = self.conv.kernel_size[0] - self.stride
        if trim > 0:
            out = out[:, :, :-trim]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Residual unit (dilated causal convolutions)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualUnit(nn.Module):
    """
    Two dilated causal convolutions with a residual skip.
    Dilation pattern: 3× blocks with dilations [1, 3, 9] per EncoderBlock.
    """

    def __init__(self, channels: int, dilation: int = 1, kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            CausalConv1d(channels, channels, kernel_size, dilation=dilation),
            nn.ELU(),
            CausalConv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Causal Transformer block (RoPE + sliding-window attention)
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) — used in Mimi."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, T, D]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class LayerScale(nn.Module):
    """Per-channel learnable scale — Moshi paper: init=0.01 for training stability."""
    def __init__(self, dim: int, init: float = 0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class CausalTransformerBlock(nn.Module):
    """
    Single causal Transformer layer with RoPE, sliding-window mask, and LayerScale.
    LayerScale (Moshi paper Section 3.3.1): init=0.01 for training stability.
    """

    def __init__(self, dim: int, heads: int, ffn_dim: int,
                 sliding_window: int = 125, dropout: float = 0.0,
                 layer_scale_init: float = 0.01):
        super().__init__()
        assert dim % heads == 0
        self.heads    = heads
        self.head_dim = dim // heads
        self.window   = sliding_window

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.rope = RotaryEmbedding(self.head_dim)
        # LayerScale after attention and FFN (Moshi paper init=0.01)
        self.ls1 = LayerScale(dim, init=layer_scale_init)
        self.ls2 = LayerScale(dim, init=layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape

        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, x.device)
        Q, K = apply_rope(Q, K, cos, sin)

        mask = torch.full((T, T), float("-inf"), device=x.device)
        for i in range(T):
            start = max(0, i - self.window + 1)
            mask[i, start: i + 1] = 0.0
        mask = mask.unsqueeze(0).unsqueeze(0)

        scale = math.sqrt(self.head_dim)
        attn  = (Q @ K.transpose(-2, -1)) / scale + mask
        attn  = attn.softmax(dim=-1)
        attn_out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.ls1(self.o_proj(attn_out))   # LayerScale on attn output

        x = x + self.ls2(self.ffn(self.norm2(x))) # LayerScale on FFN output
        return x


class CausalTransformer(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, ffn_dim: int,
                 sliding_window: int = 125):  # 10s at 12.5Hz per paper Table 13
        super().__init__()
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(dim, heads, ffn_dim, sliding_window)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] (channels-first from conv stack)
        x = x.transpose(1, 2)          # → [B, T, C]
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)       # → [B, C, T]


# ─────────────────────────────────────────────────────────────────────────────
# Encoder & Decoder blocks
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    Downsampling block:  3× ResidualUnit (dilations 1,3,9)  +  strided conv.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            ResidualUnit(in_ch, dilation=1),
            ResidualUnit(in_ch, dilation=3),
            ResidualUnit(in_ch, dilation=9),
            nn.ELU(),
            CausalConv1d(in_ch, out_ch, kernel_size=2 * stride, stride=stride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """
    Upsampling block: transposed conv  +  3× ResidualUnit (dilations 1,3,9).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            CausalConvTranspose1d(in_ch, out_ch,
                                  kernel_size=2 * stride, stride=stride),
            ResidualUnit(out_ch, dilation=1),
            ResidualUnit(out_ch, dilation=3),
            ResidualUnit(out_ch, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    SEANet encoder + causal Transformer.
    Input:  [B, 1, T_audio]   (24kHz waveform)
    Output: [B, transformer_hidden, T_latent]  (12.5Hz feature map)
    """

    def __init__(self,
                 base_channels: int = 64,
                 strides: list[int] | None = None,
                 transformer_dim: int = 512,
                 transformer_layers: int = 8,
                 transformer_heads: int = 8,
                 transformer_ffn: int = 2048,
                 sliding_window: int = 125):  # 10s × 12.5Hz = 125 frames — paper Table 13
        super().__init__()
        if strides is None:
            strides = [6, 5, 4, 4, 4]

        # Build channel progression: 64 → 128 → 256 → 512 → 512 → 512
        ch = [base_channels * min(2 ** i, 8) for i in range(len(strides) + 1)]
        ch = [min(c, 512) for c in ch]

        layers: list[nn.Module] = [
            CausalConv1d(1, ch[0], kernel_size=7)
        ]
        for i, stride in enumerate(strides):
            layers.append(EncoderBlock(ch[i], ch[i + 1], stride))

        self.conv_stack = nn.Sequential(*layers)
        self.transformer = CausalTransformer(
            dim=ch[-1],
            layers=transformer_layers,
            heads=transformer_heads,
            ffn_dim=transformer_ffn,
            sliding_window=sliding_window,
        )
        self.out_channels = ch[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_stack(x)
        return self.transformer(h)


# ─────────────────────────────────────────────────────────────────────────────
# Full Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    SEANet decoder + causal Transformer.
    Input:  [B, latent_dim, T_latent]
    Output: [B, 1, T_audio]
    """

    def __init__(self,
                 latent_dim: int = 32,
                 base_channels: int = 64,
                 strides: list[int] | None = None,
                 transformer_dim: int = 512,
                 transformer_layers: int = 8,
                 transformer_heads: int = 8,
                 transformer_ffn: int = 2048,
                 sliding_window: int = 125):  # 10s × 12.5Hz = 125 frames — paper Table 13
        super().__init__()
        if strides is None:
            strides = [6, 5, 4, 4, 4]

        ch = [base_channels * min(2 ** i, 8) for i in range(len(strides) + 1)]
        ch = [min(c, 512) for c in ch]
        ch_rev = list(reversed(ch))   # [512, 512, 512, 256, 128, 64]

        self.input_proj = nn.Linear(latent_dim, ch_rev[0])

        self.transformer = CausalTransformer(
            dim=ch_rev[0],
            layers=transformer_layers,
            heads=transformer_heads,
            ffn_dim=transformer_ffn,
            sliding_window=sliding_window,
        )

        dec_layers: list[nn.Module] = []
        for i, stride in enumerate(reversed(strides)):
            dec_layers.append(DecoderBlock(ch_rev[i], ch_rev[i + 1], stride))

        self.conv_stack = nn.Sequential(*dec_layers)
        self.output_proj = nn.Sequential(
            nn.ELU(),
            CausalConv1d(ch_rev[-1], 1, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim, T]
        h = self.input_proj(z.transpose(1, 2)).transpose(1, 2)  # [B, C, T]
        h = self.transformer(h)
        h = self.conv_stack(h)
        return self.output_proj(h)


# ─────────────────────────────────────────────────────────────────────────────
# VAE Bottleneck
# ─────────────────────────────────────────────────────────────────────────────

class VAEBottleneck(nn.Module):
    """
    Converts encoder hidden states [B, encoder_dim, T] into:
      - μ (mean):    [B, latent_dim, T]
      - logvar:      [B, latent_dim, T]
      - z (sample):  [B, latent_dim, T]  via reparameterisation

    At inference (CALM backbone uses the encoder):
      - Use z = μ  (deterministic, no noise)
      - Or z = μ + sqrt(τ) · σ · ε  for temperature-τ sampling
    """

    def __init__(self, encoder_dim: int = 512, latent_dim: int = 32):
        super().__init__()
        self.mu_proj     = nn.Linear(encoder_dim, latent_dim)
        self.logvar_proj = nn.Linear(encoder_dim, latent_dim)

    def forward(self, h: torch.Tensor, sample: bool = True,
                temperature: float = 1.0):
        """
        h:         [B, encoder_dim, T]   — output of Encoder
        sample:    True during training; False for deterministic encoding
        Returns:
          z:       [B, latent_dim, T]
          mu:      [B, latent_dim, T]
          logvar:  [B, latent_dim, T]
        """
        # h is channels-first; project along channel dim
        h_t = h.transpose(1, 2)                   # [B, T, encoder_dim]
        mu     = self.mu_proj(h_t).transpose(1, 2)      # [B, latent_dim, T]
        logvar = self.logvar_proj(h_t).transpose(1, 2)  # [B, latent_dim, T]

        # Clamp logvar for numerical stability (avoids exp overflow)
        logvar = logvar.clamp(-30.0, 20.0)

        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z   = mu + math.sqrt(temperature) * std * eps
        else:
            z = mu   # deterministic — used for CALM backbone at inference

        return z, mu, logvar


# ─────────────────────────────────────────────────────────────────────────────
# Full VAE (Encoder + Bottleneck + Decoder in one module)
# ─────────────────────────────────────────────────────────────────────────────

class SpeechVAE(nn.Module):
    """
    Full causal Speech VAE-GAN generator.
    Encoder → Bottleneck → Decoder.

    Usage:
        vae = SpeechVAE()
        z, mu, logvar, recon = vae(audio)    # training
        z, mu, logvar, _     = vae(audio, decode=False)  # encode only
    """

    def __init__(self,
                 base_channels: int = 64,
                 strides: list[int] | None = None,
                 latent_dim: int = 32,
                 transformer_dim: int = 512,
                 transformer_layers: int = 8,
                 transformer_heads: int = 8,
                 transformer_ffn: int = 2048,
                 sliding_window: int = 125):  # 10s × 12.5Hz = 125 frames — paper Table 13
        super().__init__()
        if strides is None:
            strides = [6, 5, 4, 4, 4]

        self.encoder = Encoder(
            base_channels=base_channels,
            strides=strides,
            transformer_dim=transformer_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ffn=transformer_ffn,
            sliding_window=sliding_window,
        )
        self.bottleneck = VAEBottleneck(
            encoder_dim=self.encoder.out_channels,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            strides=strides,
            transformer_dim=transformer_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ffn=transformer_ffn,
            sliding_window=sliding_window,
        )

    def encode(self, audio: torch.Tensor,
               sample: bool = True,
               temperature: float = 1.0):
        """
        audio: [B, 1, T_audio]
        Returns z, mu, logvar each [B, latent_dim, T_latent]
        """
        h                 = self.encoder(audio)
        z, mu, logvar     = self.bottleneck(h, sample=sample,
                                            temperature=temperature)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim, T_latent]  →  [B, 1, T_audio]"""
        return self.decoder(z)

    def forward(self, audio: torch.Tensor, decode: bool = True,
                sample: bool = True, temperature: float = 1.0):
        z, mu, logvar = self.encode(audio, sample=sample,
                                    temperature=temperature)
        recon = self.decode(z) if decode else None
        return z, mu, logvar, recon