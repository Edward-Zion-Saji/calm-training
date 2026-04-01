# CALM Speech VAE-GAN — Complete Training Guide

> Based on **"Continuous Audio Language Models" (CALM)** — Rouard, Orsini, Roebel, Zeghidour, Défossez (Kyutai / IRCAM, arXiv: 2509.06926)

This guide covers **Stage 1 of 3** in the CALM TTS pipeline: training the causal Speech VAE-GAN.
The VAE is trained first, then frozen. The CALM backbone and consistency head in Stage 2 only
use the encoder to produce training targets and the decoder to synthesise audio at inference.

---

## Table of Contents

1. [Overview & Architecture Blueprint](#1-overview--architecture-blueprint)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Preparation](#3-dataset-preparation)
4. [VAE Encoder & Decoder Architecture](#4-vae-encoder--decoder-architecture)
5. [VAE Bottleneck & Reparameterisation](#5-vae-bottleneck--reparameterisation)
6. [Discriminators](#6-discriminators)
7. [WavLM Distillation](#7-wavlm-distillation)
8. [Loss Functions & Weights](#8-loss-functions--weights)
9. [Full Training Loop](#9-full-training-loop)
10. [Latent Normalisation Stats](#10-latent-normalisation-stats)
11. [Evaluation, Checkpointing & Metrics](#11-evaluation-checkpointing--metrics)
12. [A100 40GB Tips & Troubleshooting](#12-a100-40gb-tips--troubleshooting)

---

## 1. Overview & Architecture Blueprint

### Why a VAE and not Mimi directly?

Mimi (Défossez et al., 2024 — the codec from the Moshi paper) is a **VQ-VAE**: its bottleneck
quantises latents into discrete RVQ token indices. CALM requires a **continuous Gaussian latent
space** so that the downstream consistency head can sample pure Gaussian noise and denoise it
into a plausible latent frame. These are fundamentally different bottlenecks:

```
Mimi (VQ-VAE):     encoder → RVQ quantiser → discrete token indices  (not usable for CALM)
CALM Speech VAE:   encoder → Linear(μ) + Linear(logvar) → z = μ + σε  (continuous, Gaussian)
```

The encoder and decoder *architecture* (SEANet convolutions + Transformer layers) is inherited
from Mimi. Only the bottleneck is replaced.

---

### The three-stage CALM TTS pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1 — VAE-GAN  (this guide)                                 │
│                                                                  │
│   raw audio (24kHz) ──► Causal SEANet Encoder                    │
│                              │                                   │
│                         ┌────▼─────┐                             │
│                         │  μ, logσ │  ← VAE Bottleneck (32 dims) │
│                         └────┬─────┘                             │
│                              │  z = μ + σε                       │
│                         ┌────▼──────────┐                        │
│                         │ Causal SEANet │                        │
│                         │   Decoder     │                        │
│                         └────┬──────────┘                        │
│                              │                                   │
│                    reconstructed audio ──► GAN discriminators    │
│                                       ──► WavLM distillation     │
│                                       ──► KL loss                │
└──────────────────────────────────────────────────────────────────┘
          ▼  (freeze VAE after training)
┌──────────────────────────────────────────────────────────────────┐
│  Stage 2 — CALM Backbone + Consistency Head                      │
│   text tokens + latent frames ──► 302M Transformer               │
│                                        │                         │
│                                  z_long + z_short                │
│                                        │                         │
│                                  10M Consistency MLP             │
│                                        │                         │
│                              next latent frame x̂ˢ               │
└──────────────────────────────────────────────────────────────────┘
          ▼  (optional)
┌──────────────────────────────────────────────────────────────────┐
│  Stage 3 — Latent Distillation → Pocket TTS (100M, CPU real-time)│
└──────────────────────────────────────────────────────────────────┘
```

---

### Paper-specified hyperparameters for the Speech VAE

| Parameter                  | Value                        | Notes                                        |
|----------------------------|------------------------------|----------------------------------------------|
| Sample rate                | 24 kHz                       | Mono                                         |
| Frame rate                 | 12.5 Hz                      | 1 latent frame = 80 ms                       |
| Latent dimension           | **32**                       | Continuous, Gaussian                         |
| Conv downsampling ratios   | `[6, 5, 4, 4, 4]`            | Product = 1920 → 24000/1920 = 12.5 Hz        |
| Encoder Transformer layers | 8                            | Causal, sliding window = 250 frames (~20 s)  |
| Decoder Transformer layers | 8                            | Causal                                       |
| Transformer hidden dim     | 512                          | From Mimi config                             |
| Transformer heads          | 8                            | Head dim = 64                                |
| Transformer FFN dim        | 2048                         |                                              |
| Batch size                 | 64                           | Reduce to 4–8 on single A100                 |
| Audio sample length        | 12 seconds                   | Per training clip                            |
| KL loss weight λ_KL        | **0.01**                     |                                              |
| Reconstruction loss        | **None** (speech only)       | Adversarial + KL + WavLM only                |
| WavLM distillation weight  | **25**                       | Applied to **entire** 32-dim latent          |
| Learning rate              | 8 × 10⁻⁴                     | Cosine schedule                              |
| Optimizer                  | AdamW β₁=0.8, β₂=0.99       |                                              |
| Model size (VAE only)      | ~20M parameters              | Encoder + Decoder (no discriminator)         |

---

### Component map

```
CALM Speech VAE
├── Encoder
│   ├── CausalConv1d  (1 → 64, k=7)          input projection
│   ├── EncoderBlock  (64  → 128, stride=6)  3× ResidualUnit + strided conv
│   ├── EncoderBlock  (128 → 256, stride=5)
│   ├── EncoderBlock  (256 → 512, stride=4)
│   ├── EncoderBlock  (512 → 512, stride=4)
│   ├── EncoderBlock  (512 → 512, stride=4)
│   ├── CausalTransformer  (8 layers, d=512)  long-range context
│   └── Linear  (512 → 64)                    split into μ (32) + logvar (32)
│
├── VAEBottleneck
│   ├── Linear  512 → 32  (μ head)
│   ├── Linear  512 → 32  (logvar head)
│   └── reparameterise: z = μ + exp(0.5·logvar) · ε,  ε ~ N(0,I)
│
├── Decoder
│   ├── Linear  (32 → 512)                    latent projection
│   ├── CausalTransformer  (8 layers, d=512)
│   ├── DecoderBlock  (512 → 512, stride=4)   3× ResidualUnit + transposed conv
│   ├── DecoderBlock  (512 → 512, stride=4)
│   ├── DecoderBlock  (512 → 256, stride=4)
│   ├── DecoderBlock  (256 → 128, stride=5)
│   ├── DecoderBlock  (128 →  64, stride=6)
│   └── CausalConv1d  (64 → 1, k=7)           output projection + tanh
│
├── Discriminators  (trained jointly, NOT part of frozen VAE)
│   ├── MultiScaleSTFTDiscriminator            5 resolutions
│   ├── MultiPeriodDiscriminator               periods [2,3,5,7,11]
│   └── MultiScaleDiscriminator                3 downsampling levels
│
└── WavLM Teacher  (frozen, microsoft/wavlm-large, 316M)
    └── cosine similarity loss vs all 32 latent dims
```

---

---

## 2. Environment Setup

### Hardware requirements

| Component | Minimum           | Recommended (paper)    |
|-----------|-------------------|------------------------|
| GPU       | A100 40GB (×1)    | 8–48× H100             |
| RAM       | 64 GB             | 256 GB                 |
| Storage   | 500 GB SSD        | 2–4 TB NVMe            |
| CUDA      | 11.8+             | 12.1+                  |

A single A100 40GB is sufficient for the VAE training stage. Expect **5–10 days** to reach
convergence (~300K–500K steps). The CALM backbone (Stage 2, 302M params) also fits on one A100
with gradient checkpointing.

---

### Python environment

```bash
# Create a clean conda environment
conda create -n calm python=3.10 -y
conda activate calm

# Install PyTorch with CUDA 12.1 (adjust cuda version to match your driver)
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core audio / ML dependencies
pip install \
    transformers==4.44.2 \
    datasets==2.21.0 \
    accelerate==0.33.0 \
    torchaudio==2.3.1 \
    einops==0.8.0 \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    webdataset==0.2.100 \
    wandb \
    tqdm \
    numpy \
    scipy

# EnCodec (for MS-STFT discriminator reference)
pip install encodec

# audiocraft (for SEANet modules, loss balancer, MSSTFTD)
pip install -U audiocraft

# WavLM / HuggingFace
pip install transformers accelerate
```

---

### Project layout

```
calm-training/
├── VAE_TRAINING_GUIDE.md          ← this file
├── train_vae.py                   ← main training entry point
├── configs/
│   └── speech_vae.yaml            ← all hyperparameters
├── models/
│   ├── __init__.py
│   ├── vae.py                     ← encoder, bottleneck, decoder
│   ├── discriminators.py          ← MS-STFT, MPD, MSD
│   └── wavlm_distill.py           ← WavLM teacher + distillation loss
├── data/
│   ├── __init__.py
│   └── speech_dataset.py          ← HuggingFace dataset loader
├── losses/
│   ├── __init__.py
│   └── vae_losses.py              ← KL, adversarial, feature matching
└── utils/
    ├── __init__.py
    └── normalise.py               ← compute + save latent stats
```

---

### Verify GPU access

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # NVIDIA A100-SXM4-40GB
print(torch.cuda.get_device_properties(0).total_memory / 1e9)  # ~40.0 GB

# Enable TF32 — A100-native, ~2× faster than fp32 with no quality loss
# Do NOT use fp16 AMP for audio GANs — known to cause NaN
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

---

### configs/speech_vae.yaml

```yaml
# ── Audio ──────────────────────────────────────────────────────────
sample_rate: 24000
frame_rate: 12.5
channels: 1
clip_seconds: 2.0          # shorter clips → bigger batch → stable GAN training
                           # paper uses 12s but that requires 64-GPU setup

# ── VAE Architecture ───────────────────────────────────────────────
encoder_base_channels: 64
encoder_strides: [6, 5, 4, 4, 4]     # total downsample = 1920
latent_dim: 32
transformer_hidden: 512
transformer_layers: 8
transformer_heads: 8
transformer_ffn_dim: 2048
transformer_sliding_window: 250       # frames (~20s at 12.5Hz)

# ── Training ───────────────────────────────────────────────────────
batch_size: 8                  # safe for A100 40GB with 2s clips
grad_accumulation: 8           # effective batch = 64
learning_rate: 8.0e-4
disc_learning_rate: 3.0e-4
betas: [0.8, 0.99]
lr_schedule: cosine
max_steps: 500000
warmup_steps: 5000
grad_clip_norm: 1000.0

# ── Loss weights ───────────────────────────────────────────────────
lambda_kl: 0.01
lambda_adv: 1.0
lambda_feat: 5.0
lambda_distil: 25.0
lambda_recon: 0.0             # speech: NO reconstruction loss

# ── WavLM ──────────────────────────────────────────────────────────
wavlm_model: "microsoft/wavlm-large"
wavlm_layer: 7                # which hidden layer to distil from (0-indexed)
wavlm_input_sr: 16000         # WavLM only accepts 16kHz

# ── Checkpointing ──────────────────────────────────────────────────
save_every: 10000
eval_every: 5000
keep_last_n: 5
output_dir: "./checkpoints/speech_vae"

# ── Data ───────────────────────────────────────────────────────────
datasets:
  - name: libritts_r
    hf_id: "mythicinfinity/libritts_r"
    config: "all"
    split: "train.clean.360+train.clean.100+train.other.500"
    weight: 1.0
  - name: voxpopuli
    hf_id: "facebook/voxpopuli"
    config: "en"
    split: "train"
    weight: 0.5
```

---

---

## 3. Dataset Preparation

### Available speech datasets on HuggingFace

The CALM paper trains on a proprietary French + English corpus totalling 88k hours. Below are the
best freely available alternatives, ranked by quality and ease of access.

| Dataset | HF ID | Hours | SR | License | Speaker IDs | Notes |
|---------|-------|-------|----|---------|-------------|-------|
| **LibriTTS-R** | `mythicinfinity/libritts_r` | 585h | 24kHz | CC BY 4.0 | ✅ 2,456 | Best starting point — already 24kHz, clean |
| **Emilia (YODAS EN)** | `amphion/Emilia-Dataset` | ~140k h | ~24kHz MP3 | CC BY 4.0 | ✅ yes | Largest commercially-licensed in-the-wild speech |
| **VoxPopuli EN** | `facebook/voxpopuli` | 543h | 16kHz | CC0 | ✅ yes | Parliament speech — must upsample to 24kHz |
| **MLS English** | `parler-tts/mls_eng` | ~44k h | 16kHz | CC BY 4.0 | ✅ yes | Audiobook speech |
| **GigaSpeech XL** | `speechcolab/gigaspeech` | 10k h | 16kHz | Non-commercial | ⚠️ partial | Needs HF + Google Form approval |
| **AMI IHM** | `edinburghcstr/ami` | 100h | 16kHz | CC BY 4.0 | ✅ yes | Meeting speech |

> **Recommended starting set for A100 single-GPU training:**
> `LibriTTS-R` (585h, already 24kHz) + `parler-tts/mls_eng` (streaming, 44k h).
> This gives enough variety without hitting gating issues.

> **For large-scale training:** Add `amphion/Emilia-Dataset` English YODAS subset (~140k h CC BY 4.0).

---

### Dataset loading — `data/speech_dataset.py`

```python
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset, interleave_datasets
import random


TARGET_SR = 24_000


def load_hf_speech_dataset(hf_id: str, config: str | None, split: str,
                            streaming: bool = True):
    """Load a HuggingFace speech dataset, returning an iterable."""
    kwargs = dict(split=split, streaming=streaming, trust_remote_code=True)
    if config:
        return load_dataset(hf_id, config, **kwargs)
    return load_dataset(hf_id, **kwargs)


class SpeechClipDataset(Dataset):
    """
    Wraps a HuggingFace audio dataset.
    - Resamples to TARGET_SR (24kHz)
    - Returns fixed-length mono clips of `clip_samples` samples
    - Clips too short are zero-padded; clips too long are randomly cropped
    """

    def __init__(self, hf_dataset, clip_seconds: float = 2.0,
                 target_sr: int = TARGET_SR):
        self.data = list(hf_dataset)   # materialise for map-style Dataset
        self.clip_samples = int(clip_seconds * target_sr)
        self.target_sr = target_sr
        self._resamplers: dict[int, T.Resample] = {}

    def _get_resampler(self, orig_sr: int) -> T.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(orig_sr, self.target_sr)
        return self._resamplers[orig_sr]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_dict = item["audio"]
        waveform = torch.tensor(audio_dict["array"], dtype=torch.float32)
        orig_sr   = audio_dict["sampling_rate"]

        # Ensure 1D (mono)
        if waveform.ndim == 2:
            waveform = waveform.mean(0)

        # Resample if needed
        if orig_sr != self.target_sr:
            waveform = self._get_resampler(orig_sr)(waveform)

        # Normalise amplitude to [-1, 1]
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak.clamp(min=1.0)

        # Crop / pad to fixed length
        if waveform.shape[0] >= self.clip_samples:
            start = random.randint(0, waveform.shape[0] - self.clip_samples)
            waveform = waveform[start: start + self.clip_samples]
        else:
            pad = self.clip_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform.unsqueeze(0)   # [1, T]


class StreamingSpeechDataset(torch.utils.data.IterableDataset):
    """
    Streaming version for very large datasets (e.g. Emilia 140k h).
    Does NOT require materialising the entire dataset into memory.
    """

    def __init__(self, hf_dataset, clip_seconds: float = 2.0,
                 target_sr: int = TARGET_SR, buffer_size: int = 1000):
        self.dataset = hf_dataset.shuffle(seed=42, buffer_size=buffer_size)
        self.clip_samples = int(clip_seconds * target_sr)
        self.target_sr = target_sr

    def __iter__(self):
        for item in self.dataset:
            audio_dict = item["audio"]
            waveform = torch.tensor(audio_dict["array"], dtype=torch.float32)
            orig_sr   = audio_dict["sampling_rate"]

            if waveform.ndim == 2:
                waveform = waveform.mean(0)

            if orig_sr != self.target_sr:
                resampler = T.Resample(orig_sr, self.target_sr)
                waveform   = resampler(waveform)

            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak.clamp(min=1.0)

            # Yield multiple non-overlapping clips from one recording
            n_clips = waveform.shape[0] // self.clip_samples
            for i in range(max(n_clips, 1)):
                start = i * self.clip_samples
                clip  = waveform[start: start + self.clip_samples]
                if clip.shape[0] < self.clip_samples:
                    clip = torch.nn.functional.pad(
                        clip, (0, self.clip_samples - clip.shape[0]))
                yield clip.unsqueeze(0)   # [1, T]


def build_dataloader(cfg: dict, num_workers: int = 4) -> DataLoader:
    """
    Build a DataLoader from one or more HuggingFace datasets.
    Uses streaming for datasets > 1000h.
    """
    clip_seconds = cfg.get("clip_seconds", 2.0)

    # LibriTTS-R — map-style (fits in memory at ~200GB)
    libritts = load_hf_speech_dataset(
        "mythicinfinity/libritts_r", "all",
        "train.clean.360+train.clean.100+train.other.500",
        streaming=False
    )
    ds_libritts = SpeechClipDataset(libritts, clip_seconds=clip_seconds)

    # Emilia YODAS English — streaming (too large to materialise)
    emilia_raw = load_hf_speech_dataset(
        "amphion/Emilia-Dataset", None, "train", streaming=True
    )
    ds_emilia = StreamingSpeechDataset(emilia_raw, clip_seconds=clip_seconds)

    # Combine with ConcatDataset for map-style; for mixed map+iterable,
    # use a weighted interleaved approach via HuggingFace interleave_datasets.
    # Simple approach: just use LibriTTS-R for initial training
    return DataLoader(
        ds_libritts,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
```

---

### Sample rate note

WavLM **only accepts 16kHz input**. You must maintain two resamplers during training:

```python
# In training loop
resample_24_to_16 = torchaudio.transforms.Resample(24000, 16000).to(device)

# Usage
audio_16k = resample_24_to_16(audio_24k)   # for WavLM only
# encoder always receives original 24kHz audio
```

---

### Quick dataset sanity check

```python
from data.speech_dataset import SpeechClipDataset, load_hf_speech_dataset

ds_raw = load_hf_speech_dataset("mythicinfinity/libritts_r", "clean",
                                  "train.clean.100", streaming=False)
ds = SpeechClipDataset(ds_raw, clip_seconds=2.0)
clip = ds[0]
print(clip.shape)          # torch.Size([1, 48000])  — 2s at 24kHz
print(clip.min(), clip.max())  # values in [-1, 1]
```

---

---

## 4. VAE Encoder & Decoder Architecture

The backbone is a **SEANet** (Streaming Encoder-dEcoder Network) with causal convolutions —
the same backbone used in EnCodec and Mimi. Transformer layers are inserted after the
convolutional stack in both encoder and decoder to capture long-range dependencies.
All convolutions and attention are **fully causal** (no future context).

### models/vae.py

```python
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
        self.pad = (kernel_size - 1) * dilation
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


class CausalTransformerBlock(nn.Module):
    """Single causal Transformer layer with RoPE and sliding window mask."""

    def __init__(self, dim: int, heads: int, ffn_dim: int,
                 sliding_window: int = 250, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads      = heads
        self.head_dim   = dim // heads
        self.window     = sliding_window

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape

        # Self-attention with causal + sliding-window mask
        h = self.norm1(x)
        Q = self.q_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, x.device)
        Q, K = apply_rope(Q, K, cos, sin)

        # Build causal mask restricted to sliding window
        mask = torch.full((T, T), float("-inf"), device=x.device)
        for i in range(T):
            start = max(0, i - self.window + 1)
            mask[i, start: i + 1] = 0.0
        mask = mask.unsqueeze(0).unsqueeze(0)   # [1,1,T,T]

        scale = math.sqrt(self.head_dim)
        attn  = (Q @ K.transpose(-2, -1)) / scale + mask
        attn  = attn.softmax(dim=-1)
        out   = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        x     = x + self.o_proj(out)

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, ffn_dim: int,
                 sliding_window: int = 250):
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
                 sliding_window: int = 250):
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
                 sliding_window: int = 250):
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
```

---

### Shape walkthrough

For a 2-second clip at 24kHz:

```
Input:           [B, 1, 48000]
After conv_stack (strides [6,5,4,4,4]):
  → [B,  64, 48000]    after input conv
  → [B, 128,  8000]    after stride-6 block
  → [B, 256,  1600]    after stride-5 block
  → [B, 512,   400]    after stride-4 block
  → [B, 512,   100]    after stride-4 block
  → [B, 512,    25]    after stride-4 block  ← 25 frames for 2s = 12.5Hz ✓
After Transformer: [B, 512, 25]
After bottleneck:  [B, 32,  25]  (μ or z sampled)
```

---

---

## 5. VAE Bottleneck & Reparameterisation

This section covers the bottleneck that converts the encoder's hidden state into a Gaussian
distribution and samples from it — the core element that makes this a VAE rather than a plain
autoencoder or VQ-VAE.

### Why Gaussian?

The downstream CALM consistency head operates by:
1. Sampling pure noise: `ε ~ N(0, I)` in latent space
2. Denoising it in one step to produce a plausible next latent frame

For this to work, the VAE's aggregate posterior must be close to `N(0, I)`. The KL divergence
term in the loss enforces this. The smoother and more Gaussian the latent space, the easier the
consistency head's denoising task.

---

### Bottleneck code — append to `models/vae.py`

```python
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
                 sliding_window: int = 250):
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
```

---

### KL divergence loss

```python
def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from posterior N(μ, σ²) to prior N(0, I).
    Returns scalar mean over batch and time.

    KL = -0.5 * sum(1 + logvar - mu² - exp(logvar))
    """
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
```

With `λ_KL = 0.01`, this term is kept small so reconstruction quality is prioritised. If you
notice the latent space is not well-structured (consistency head struggles at inference), try
gradually annealing `λ_KL` from 0 to 0.01 over the first 50k steps (KL annealing).

---

### Inference encoding (CALM backbone)

At inference time the CALM backbone calls the encoder deterministically — no noise, just `μ`:

```python
# During CALM training and inference — always deterministic encoding
with torch.no_grad():
    z, mu, logvar = vae.encode(audio, sample=False)
    # z == mu here — used as ground-truth latent targets for the consistency head
```

Temperature sampling in CALM happens in the **consistency head**, not the VAE bottleneck.

---

### Parameter count sanity check

```python
from models.vae import SpeechVAE

vae = SpeechVAE()
total = sum(p.numel() for p in vae.parameters())
enc   = sum(p.numel() for p in vae.encoder.parameters())
dec   = sum(p.numel() for p in vae.decoder.parameters())
bn    = sum(p.numel() for p in vae.bottleneck.parameters())
print(f"Encoder:    {enc/1e6:.1f}M")
print(f"Bottleneck: {bn/1e6:.3f}M")
print(f"Decoder:    {dec/1e6:.1f}M")
print(f"Total VAE:  {total/1e6:.1f}M")
# Expected: ~8–10M each for enc/dec, ~20M total
```

---

---

## 6. Discriminators

The VAE generator is trained adversarially against a discriminator ensemble. The discriminators
are **not saved as part of the final VAE** — they are used only during training. Three
complementary discriminators are used, each measuring audio quality at a different granularity.

| Discriminator | What it captures | From |
|---|---|---|
| Multi-Scale STFT (MS-STFT) | Spectral artifacts across frequency resolutions | EnCodec / Mimi |
| Multi-Period (MPD) | Periodic waveform structure (pitch, harmonics) | HiFi-GAN |
| Multi-Scale Waveform (MSD) | Waveform shape at multiple downsampling levels | MelGAN |

---

### models/discriminators.py

```python
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
```

---

### Discriminator parameter count

```python
from models.discriminators import CombinedDiscriminator
disc = CombinedDiscriminator()
print(sum(p.numel() for p in disc.parameters()) / 1e6)
# ~120–180M total across all discriminators
# These are NOT saved as part of the final VAE
```

---

---

## 7. WavLM Distillation

This is **the most important loss** for the speech VAE and the biggest departure from a standard
audio VAE-GAN setup. WavLM (Chen et al., 2021) is a large self-supervised speech model trained
to predict masked speech frames. Its internal representations encode rich phonetic and semantic
information. By forcing the VAE latents to match WavLM's representations, we ensure:

1. The latent space is **semantically organised** — phonetically similar sounds cluster together
2. The CALM backbone has an **easier prediction target** — it predicts which phoneme cluster comes
   next, not arbitrary acoustic noise
3. The latents encode what is being **said** (semantic) not just how it sounds (acoustic)

**Key CALM paper difference from Mimi:** Mimi distils WavLM only into the *first* RVQ codebook.
CALM extends distillation to **all 32 latent dimensions**.

---

### Frame rate alignment

```
WavLM input:   16kHz audio
WavLM output:  ~50 Hz features  (1 feature per 320 samples at 16kHz)

VAE latents:   12.5 Hz  (1 frame per 1920 samples at 24kHz)

Ratio: 50 / 12.5 = 4
→ average-pool WavLM features by factor 4 to align with VAE latents
```

---

### models/wavlm_distill.py

```python
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
```

---

### Which WavLM layer to use?

The paper does not specify the exact WavLM layer. Mimi uses layer 6 (0-indexed). A common
empirical finding across phoneme-probing studies is that layers 6–9 of WavLM-Large carry the
richest phonetic information. The config sets `wavlm_layer: 7` as a good default. You can tune
this by checking phoneme discriminability (ABX score) after training.

```python
# To test multiple layers during debugging:
for layer_idx in [6, 7, 8, 9]:
    distiller = WavLMDistillation(wavlm_layer=layer_idx)
    loss = distiller(audio_batch, latent_batch)
    print(f"Layer {layer_idx}: distil_loss={loss.item():.4f}")
```

---

### Memory note

WavLM-Large has 316M parameters. Even frozen, it occupies ~1.2 GB of VRAM in fp32. On an A100
40GB this is fine, but load it once and keep it on GPU throughout training:

```python
# Load once at training start
distiller = WavLMDistillation().to(device)
# WavLM is frozen; its memory footprint is fixed, no gradients stored
```

---

---

## 8. Loss Functions & Weights

The full VAE loss from the paper (Eq. 2):

```
L_VAE = λ_t · L_t(x, x̂)            ← temporal reconstruction  [speech: 0]
      + λ_f · L_f(x, x̂)            ← frequency reconstruction  [speech: 0]
      + λ_adv · L_adv(x̂)           ← adversarial loss
      + λ_feat · L_feat(x, x̂)      ← feature matching
      + λ_KL · L_KL(μ, logvar)      ← KL divergence  [weight: 0.01]
      + λ_distil · L_distil         ← WavLM distillation  [weight: 25]
```

For **speech**, reconstruction losses are zero — only adversarial + KL + WavLM.
For **music**, reconstruction losses are included (not covered in this TTS-focused guide).

---

### losses/vae_losses.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# KL Divergence
# ─────────────────────────────────────────────────────────────────────────────

def kl_divergence_loss(mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
    """
    KL( N(μ,σ²) || N(0,1) ) = -0.5 * mean(1 + logvar - μ² - exp(logvar))
    Returns a scalar.
    """
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial losses (LSGAN — least squares)
# ─────────────────────────────────────────────────────────────────────────────
# LSGAN is more stable than vanilla GAN and the standard choice for audio codecs.
# Hinge loss is an alternative — both are included here.

def generator_adversarial_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """
    Generator adversarial loss: wants discriminator to output 1 for fake audio.
    L_gen = mean over all discriminators of  mean((D(x̂) - 1)²)
    fake_logits: list of discriminator output tensors for fake audio
    """
    loss = 0.0
    for logit in fake_logits:
        loss = loss + F.mse_loss(logit, torch.ones_like(logit))
    return loss / len(fake_logits)


def discriminator_adversarial_loss(real_logits: list[torch.Tensor],
                                   fake_logits: list[torch.Tensor]
                                   ) -> torch.Tensor:
    """
    Discriminator loss: real → 1, fake → 0
    L_disc = 0.5 * mean((D(x)-1)² + D(x̂)²)
    """
    loss = 0.0
    for real, fake in zip(real_logits, fake_logits):
        loss = loss + (
            F.mse_loss(real, torch.ones_like(real))
            + F.mse_loss(fake, torch.zeros_like(fake))
        )
    return 0.5 * loss / len(real_logits)


# ─────────────────────────────────────────────────────────────────────────────
# Feature matching loss
# ─────────────────────────────────────────────────────────────────────────────
# L1 distance between discriminator intermediate features for real vs. fake.
# This is a perceptual loss that stabilises GAN training.

def feature_matching_loss(real_features: list[list[torch.Tensor]],
                           fake_features: list[list[torch.Tensor]]) -> torch.Tensor:
    """
    real_features: list (per discriminator) of lists (per layer) of tensors
    fake_features: same structure for fake audio
    """
    loss = 0.0
    n    = 0
    for real_layers, fake_layers in zip(real_features, fake_features):
        for r, f in zip(real_layers, fake_layers):
            loss = loss + F.l1_loss(f, r.detach())
            n   += 1
    return loss / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction helper: flatten all logits and features from CombinedDiscriminator
# ─────────────────────────────────────────────────────────────────────────────

def flatten_disc_outputs(disc_outputs: dict):
    """
    disc_outputs: return value of CombinedDiscriminator.forward()
      {"msstftd": (logits_list, feats_list), "mpd": ..., "msd": ...}
    Returns:
      all_logits:   flat list of logit tensors
      all_features: list of feature-map lists (one per sub-discriminator)
    """
    all_logits   = []
    all_features = []
    for name, (logits, feats) in disc_outputs.items():
        all_logits.extend(logits)
        all_features.extend(feats if isinstance(feats[0], list) else [feats])
    return all_logits, all_features


# ─────────────────────────────────────────────────────────────────────────────
# Combined generator loss
# ─────────────────────────────────────────────────────────────────────────────

class VAEGeneratorLoss(nn.Module):
    """
    Combines all generator-side losses with the paper's weights.
    Usage:
        loss_fn   = VAEGeneratorLoss()
        g_loss, breakdown = loss_fn(mu, logvar, fake_logits, real_feats,
                                    fake_feats, wavlm_distil_loss)
    """

    def __init__(self,
                 lambda_kl:     float = 0.01,
                 lambda_adv:    float = 1.0,
                 lambda_feat:   float = 5.0,
                 lambda_distil: float = 25.0,
                 lambda_recon:  float = 0.0):    # 0 for speech
        super().__init__()
        self.lambda_kl     = lambda_kl
        self.lambda_adv    = lambda_adv
        self.lambda_feat   = lambda_feat
        self.lambda_distil = lambda_distil
        self.lambda_recon  = lambda_recon

    def forward(self,
                mu:           torch.Tensor,
                logvar:       torch.Tensor,
                fake_logits:  list[torch.Tensor],
                real_features: list,
                fake_features: list,
                distil_loss:  torch.Tensor,
                recon_loss:   torch.Tensor | None = None,
                ) -> Tuple[torch.Tensor, dict]:

        l_kl   = kl_divergence_loss(mu, logvar)
        l_adv  = generator_adversarial_loss(fake_logits)
        l_feat = feature_matching_loss(real_features, fake_features)
        l_dist = distil_loss

        total = (
            self.lambda_kl     * l_kl
            + self.lambda_adv  * l_adv
            + self.lambda_feat * l_feat
            + self.lambda_distil * l_dist
        )

        if recon_loss is not None and self.lambda_recon > 0:
            total = total + self.lambda_recon * recon_loss

        breakdown = {
            "loss/kl":       l_kl.item(),
            "loss/adv_gen":  l_adv.item(),
            "loss/feat":     l_feat.item(),
            "loss/distil":   l_dist.item(),
            "loss/total_gen": total.item(),
        }
        return total, breakdown
```

---

### Loss weight rationale

| Loss | Weight | Why |
|---|---|---|
| `λ_KL = 0.01` | Small | Prioritise reconstruction quality; gentle Gaussian push. Increase if latent space is chaotic. |
| `λ_adv = 1.0` | Standard | Drives perceptual realism. Lower if training is unstable early on. |
| `λ_feat = 5.0` | High | Feature matching is the primary stability anchor — keeps generator from mode collapse. |
| `λ_distil = 25.0` | Very high | WavLM semantics is the dominant signal for speech VAE. This is what makes the latent space useful for CALM. |
| `λ_recon = 0.0` | Zero (speech) | Explicit waveform/spectrogram matching is counterproductive when WavLM distillation is present — WavLM already captures what matters. |

---

### KL annealing (optional but recommended)

If the model collapses early (posterior = prior, all latents are pure noise) due to the KL term
dominating, use a linear warmup:

```python
def get_kl_weight(step: int, warmup_steps: int = 50_000,
                  target: float = 0.01) -> float:
    """Linearly anneal λ_KL from 0 to target over warmup_steps."""
    return target * min(step / warmup_steps, 1.0)

# In training loop:
lambda_kl = get_kl_weight(global_step)
```

---

*Next section: the complete training loop tying everything together.*
