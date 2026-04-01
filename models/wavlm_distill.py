import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WavLMModel


class WavLMDistillation(nn.Module):
    """
    Frozen WavLM-Large teacher + trainable linear projection.

    Follows the Moshi paper (Defossez et al. 2024, Section 3.3.2) exactly:
      - Input audio downsampled to 16kHz before WavLM
      - WavLM last hidden state used (1024-dim, ~50Hz)
      - Average pooled with kernel=8, stride=4 (NON-causal) to reach 12.5Hz
      - Linear projection from latent_dim → 1024
      - Cosine distance loss

    CALM extends this to the entire 32-dim latent (not just first codebook as Mimi does).
    """

    WAVLM_DIM   = 1024      # wavlm-large hidden size
    WAVLM_SR    = 16_000    # WavLM only accepts 16kHz
    WAVLM_RATE  = 50        # WavLM output ~50Hz at 16kHz input
    VAE_RATE    = 12.5      # VAE latent frame rate
    POOL_KERNEL = 8         # Moshi paper: kernel_size=8 (non-causal avg pool)
    POOL_STRIDE = 4         # Moshi paper: stride=4 → 50/4 = 12.5Hz

    def __init__(self, latent_dim: int = 32, vae_sr: int = 24_000,
                 wavlm_model_id: str = "microsoft/wavlm-large"):
        super().__init__()
        self.vae_sr = vae_sr

        # Load and freeze WavLM
        self.wavlm = WavLMModel.from_pretrained(wavlm_model_id)
        self.wavlm.eval()
        for p in self.wavlm.parameters():
            p.requires_grad = False

        # Resample 24kHz → 16kHz (nn.Module, moves with .to(device))
        self.resampler = torchaudio.transforms.Resample(vae_sr, self.WAVLM_SR)

        # Trainable projection: latent_dim → WavLM dim
        self.projection = nn.Linear(latent_dim, self.WAVLM_DIM, bias=False)

    @torch.no_grad()
    def extract_wavlm_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio:   [B, 1, T_audio] at vae_sr (24kHz)
        returns: [B, T_lat, 1024] at 12.5Hz
        """
        audio_16k = self.resampler(audio.squeeze(1))   # [B, T_16k]

        # Zero-mean unit-variance normalisation (WavLM pre-training convention)
        mean = audio_16k.mean(dim=-1, keepdim=True)
        std  = audio_16k.std(dim=-1, keepdim=True).clamp(min=1e-5)
        audio_16k = (audio_16k - mean) / std

        # WavLM last hidden state — Moshi paper uses full model output (not a
        # specific intermediate layer). Shape: [B, T_50hz, 1024]
        outputs = self.wavlm(audio_16k)
        feats   = outputs.last_hidden_state   # [B, T_50hz, 1024]

        # Average pool from 50Hz → 12.5Hz
        # Moshi paper: kernel=8, stride=4, NON-CAUSAL
        # Non-causal is fine — WavLM features are only used during training
        feats = feats.transpose(1, 2)                          # [B, 1024, T_50hz]
        feats = F.avg_pool1d(feats, kernel_size=self.POOL_KERNEL,
                             stride=self.POOL_STRIDE,
                             padding=self.POOL_KERNEL // 2)    # non-causal symmetric pad
        return feats.transpose(1, 2)                           # [B, T_12.5hz, 1024]

    def forward(self, audio: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        audio:  [B, 1, T_audio]           — original 24kHz waveform
        mu:     [B, latent_dim, T_latent]  — VAE encoder mean (deterministic)

        Returns scalar cosine-similarity distillation loss in [0, 2].
        """
        teacher = self.extract_wavlm_features(audio)   # [B, T_lat, 1024]

        mu_t    = mu.transpose(1, 2)                   # [B, T_lat, latent_dim]
        student = self.projection(mu_t)                 # [B, T_lat, 1024]

        T = min(teacher.shape[1], student.shape[1])
        teacher = teacher[:, :T, :]
        student = student[:, :T, :]

        return 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
