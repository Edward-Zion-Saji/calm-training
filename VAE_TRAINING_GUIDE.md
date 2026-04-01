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

*Next section: environment setup and dependency installation.*
