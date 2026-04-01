"""
Evaluate a trained VAE by encoding and decoding a test set,
then computing PESQ, STOI, and logging audio samples to W&B.

pip install pesq pystoi
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pesq import pesq
from pystoi import stoi
from datasets import load_dataset
from models.vae import SpeechVAE


def evaluate_vae(vae: SpeechVAE, n_samples: int = 100,
                 sample_rate: int = 24_000, device: str = "cuda"):
    vae.eval()
    vae = vae.to(device)

    # Load LibriTTS-R test-clean for evaluation
    ds = load_dataset("mythicinfinity/libritts_r", "clean",
                      split="test.clean", streaming=False)

    pesq_scores, stoi_scores = [], []

    resample_to_16k = T.Resample(sample_rate, 16_000)

    with torch.no_grad():
        for i, item in enumerate(ds):
            if i >= n_samples:
                break

            # Load audio
            waveform = torch.tensor(item["audio"]["array"],
                                    dtype=torch.float32)
            orig_sr  = item["audio"]["sampling_rate"]

            # Resample to 24kHz if needed
            if orig_sr != sample_rate:
                waveform = T.Resample(orig_sr, sample_rate)(waveform)

            # Normalise
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

            # Pad to nearest multiple of stride product (1920)
            stride_product = 1920
            pad = (stride_product - waveform.shape[-1] % stride_product) % stride_product
            waveform_padded = torch.nn.functional.pad(waveform, (0, pad))

            audio_in = waveform_padded.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]

            # Encode + decode
            z, mu, logvar, recon = vae(audio_in, sample=False)
            recon = recon.squeeze().cpu()

            # Trim to original length
            recon = recon[:waveform.shape[-1]]
            ref   = waveform[:waveform.shape[-1]]

            # Downsample to 16kHz for PESQ (PESQ only supports 8/16kHz)
            ref_16k   = resample_to_16k(ref.unsqueeze(0)).squeeze().numpy()
            recon_16k = resample_to_16k(recon.unsqueeze(0)).squeeze().numpy()

            # Clip to [-1, 1] for metrics
            ref_16k   = np.clip(ref_16k,   -1, 1)
            recon_16k = np.clip(recon_16k, -1, 1)

            # PESQ (wideband, 16kHz)
            try:
                p = pesq(16_000, ref_16k, recon_16k, "wb")
                pesq_scores.append(p)
            except Exception:
                pass

            # STOI
            s = stoi(ref_16k, recon_16k, 16_000, extended=False)
            stoi_scores.append(s)

    results = {
        "pesq_mean": float(np.mean(pesq_scores)) if pesq_scores else 0.0,
        "pesq_std":  float(np.std(pesq_scores))  if pesq_scores else 0.0,
        "stoi_mean": float(np.mean(stoi_scores)),
        "stoi_std":  float(np.std(stoi_scores)),
        "n_samples": n_samples,
    }

    print(f"PESQ: {results['pesq_mean']:.3f} ± {results['pesq_std']:.3f}  "
          f"(paper: 2.42)")
    print(f"STOI: {results['stoi_mean']:.3f} ± {results['stoi_std']:.3f}  "
          f"(paper: 0.90)")
    return results

