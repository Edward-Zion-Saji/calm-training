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

*Next section: VAE encoder and decoder architecture in full.*
