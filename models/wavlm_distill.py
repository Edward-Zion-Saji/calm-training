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
        self.wavlm_sr    = self.WAVLM_SR
        self.wavlm_layer = wavlm_layer

        # Load and freeze WavLM
        self.wavlm = WavLMModel.from_pretrained(wavlm_model_id)
        self.wavlm.eval()
        for p in self.wavlm.parameters():
            p.requires_grad = False

        # torchaudio.transforms.Resample is an nn.Module — registering it here
        # ensures it moves to the correct device with .to(device)
        self.resampler = torchaudio.transforms.Resample(vae_sr, self.WAVLM_SR)

        # Trainable projection: latent_dim → WavLM hidden dim
        # This is the only trainable part; WavLM and resampler are frozen
        self.projection = nn.Linear(latent_dim, self.WAVLM_DIM, bias=False)

    @torch.no_grad()
    def extract_wavlm_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio:   [B, 1, T_audio]  at vae_sr (24kHz)
        returns: [B, T_wavlm_aligned, wavlm_dim]  at VAE frame rate (12.5Hz)

        Note: @torch.no_grad() is correct here — WavLM is frozen and we never
        need gradients through the teacher features.
        """
        # Resample to 16kHz. resampler is an nn.Module and was moved to the
        # correct device when WavLMDistillation.to(device) was called.
        audio_16k = self.resampler(audio.squeeze(1))   # [B, T_16k]

        # WavLM expects zero-mean, unit-variance input (same normalisation as
        # during its pre-training on 16kHz speech).
        mean = audio_16k.mean(dim=-1, keepdim=True)
        std  = audio_16k.std(dim=-1, keepdim=True).clamp(min=1e-5)
        audio_16k = (audio_16k - mean) / std

        # Run frozen WavLM — extract specified hidden layer.
        # hidden_states is a tuple: index 0 = CNN feature extractor output,
        # indices 1..25 = transformer layer outputs for WavLM-Large (24 layers).
        # Paper says "cosine similarity loss" on the inner latent; Mimi uses
        # layer 6 (0-indexed from Transformer, = index 7 in hidden_states tuple).
        outputs = self.wavlm(audio_16k, output_hidden_states=True)
        feats = outputs.hidden_states[self.wavlm_layer]   # [B, T_50hz, 1024]

        # Downsample from ~50Hz to 12.5Hz (factor = 4) to align with VAE latents.
        feats = feats.transpose(1, 2)                         # [B, 1024, T_50hz]
        feats = F.avg_pool1d(feats, kernel_size=self.POOL_FACTOR,
                             stride=self.POOL_FACTOR)          # [B, 1024, T_12.5hz]
        return feats.transpose(1, 2)                           # [B, T_12.5hz, 1024]

    def forward(self, audio: torch.Tensor,
                mu: torch.Tensor) -> torch.Tensor:
        """
        Compute WavLM distillation loss (paper Section 5.1, cosine similarity).

        audio:  [B, 1, T_audio]          — original 24kHz waveform (NOT reconstruction)
        mu:     [B, latent_dim, T_latent] — VAE encoder mean (deterministic, no noise)
                                            Use mu, not z, so the distillation target
                                            is not corrupted by reparameterisation noise.

        Returns: scalar loss in [0, 2]; 0 = perfect alignment.
        """
        # Teacher features from frozen WavLM (no gradients)
        teacher = self.extract_wavlm_features(audio)   # [B, T_lat, 1024]

        # Project mu to WavLM's hidden dim (trainable linear layer)
        mu_t    = mu.transpose(1, 2)                   # [B, T_lat, latent_dim]
        student = self.projection(mu_t)                 # [B, T_lat, 1024]

        # Align temporal dimension — can differ by ±1 frame due to pooling rounding
        T = min(teacher.shape[1], student.shape[1])
        teacher = teacher[:, :T, :]
        student = student[:, :T, :]

        # Cosine similarity loss: 1 − cos_sim (paper: "cosine similarity loss")
        loss = 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
        return loss
