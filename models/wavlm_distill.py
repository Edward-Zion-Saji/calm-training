import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WavLMModel


class WavLMDistillation(nn.Module):
    """
    Frozen WavLM teacher + projection head for distilling semantic
    representations into all VAE latent dimensions.

    The projection is a simple linear layer: latent_dim → wavlm_dim.
    It is trained alongside the VAE and learns to align the latent space
    with WavLM's internal representations.
    """

    WAVLM_DIM    = 1024    # wavlm-large hidden size
    WAVLM_SR     = 16_000  # WavLM only accepts 16kHz
    WAVLM_RATE   = 50      # WavLM output frame rate (Hz) at 16kHz input
    VAE_RATE     = 12.5    # VAE latent frame rate (Hz)
    POOL_FACTOR  = int(WAVLM_RATE / VAE_RATE)   # = 4

    def __init__(self, latent_dim: int = 32, vae_sr: int = 24_000,
                 wavlm_model_id: str = "microsoft/wavlm-large",
                 wavlm_layer: int = 7):
        super().__init__()
        self.vae_sr      = vae_sr
        self.wavlm_layer = wavlm_layer

        # Load and freeze WavLM
        self.wavlm = WavLMModel.from_pretrained(wavlm_model_id)
        self.wavlm.eval()
        for p in self.wavlm.parameters():
            p.requires_grad = False

        # Resample from VAE sample rate to WavLM's 16kHz
        self.resampler = torchaudio.transforms.Resample(vae_sr, self.WAVLM_SR)

        # Trainable projection: latent_dim → WavLM hidden dim
        # This projection is the only trainable part of this module
        self.projection = nn.Linear(latent_dim, self.WAVLM_DIM, bias=False)

    @torch.no_grad()
    def extract_wavlm_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio:   [B, 1, T_audio]  at vae_sr (24kHz)
        returns: [B, T_wavlm_aligned, wavlm_dim]  at VAE frame rate (12.5Hz)
        """
        # Resample to 16kHz for WavLM
        audio_16k = self.resampler(audio.squeeze(1))   # [B, T_16k]

        # Normalise to zero mean, unit variance (WavLM expects this)
        mean = audio_16k.mean(dim=-1, keepdim=True)
        std  = audio_16k.std(dim=-1, keepdim=True).clamp(min=1e-5)
        audio_16k = (audio_16k - mean) / std

        # Run WavLM — extract specified hidden layer
        outputs = self.wavlm(audio_16k, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) × [B, T_50hz, 1024]
        feats = outputs.hidden_states[self.wavlm_layer]   # [B, T_50hz, 1024]

        # Downsample from 50Hz to 12.5Hz via average pooling (factor 4)
        # channels-first for avg_pool1d
        feats = feats.transpose(1, 2)                         # [B, 1024, T_50hz]
        feats = F.avg_pool1d(feats, kernel_size=self.POOL_FACTOR,
                             stride=self.POOL_FACTOR)          # [B, 1024, T_12.5hz]
        return feats.transpose(1, 2)                           # [B, T_12.5hz, 1024]

    def forward(self, audio: torch.Tensor,
                vae_latents: torch.Tensor) -> torch.Tensor:
        """
        Compute WavLM distillation loss.

        audio:        [B, 1, T_audio]      — original 24kHz audio (not reconstruction)
        vae_latents:  [B, latent_dim, T_latent]  — sampled z from VAE bottleneck

        Returns: scalar cosine-similarity distillation loss
        """
        # Extract teacher features (no grad)
        teacher = self.extract_wavlm_features(audio)   # [B, T_lat, 1024]

        # Project VAE latents to WavLM dim
        z_t = vae_latents.transpose(1, 2)              # [B, T_lat, latent_dim]
        student = self.projection(z_t)                  # [B, T_lat, 1024]

        # Align temporal dimension (may differ by 1 frame due to rounding)
        T = min(teacher.shape[1], student.shape[1])
        teacher = teacher[:, :T, :]
        student = student[:, :T, :]

        # Cosine similarity loss (1 - cosine_sim → minimise)
        # Mean over batch and time
        loss = 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
        return loss
