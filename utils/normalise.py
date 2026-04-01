"""
utils/normalise.py

After VAE training, run:
    python -m utils.normalise \
        --vae_ckpt checkpoints/speech_vae/vae_final.pt \
        --config   configs/speech_vae.yaml \
        --out      checkpoints/speech_vae/latent_stats.pt \
        --n_batches 500
"""

import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

from models.vae          import SpeechVAE
from data.speech_dataset import build_dataloader


def compute_latent_stats(vae: SpeechVAE, loader, device: torch.device,
                         n_batches: int = 500):
    """
    Pass n_batches of audio clips through the VAE encoder (deterministic, μ only).
    Compute the per-dimension mean and std of the resulting latents.

    Returns:
        mean:  [latent_dim]  — per-dimension mean across all frames
        std:   [latent_dim]  — per-dimension std across all frames
    """
    vae.eval()
    all_latents = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Computing latent stats",
                                       total=n_batches)):
            if i >= n_batches:
                break
            audio = batch.to(device)                        # [B, 1, T]
            z, mu, _ = vae.encode(audio, sample=False)     # [B, latent_dim, T]
            # Flatten to [B*T, latent_dim] to compute global statistics
            mu = mu.permute(0, 2, 1).reshape(-1, mu.shape[1])  # [N, latent_dim]
            all_latents.append(mu.cpu())

    all_latents = torch.cat(all_latents, dim=0)   # [N_total, latent_dim]
    mean = all_latents.mean(dim=0)                # [latent_dim]
    std  = all_latents.std(dim=0).clamp(min=1e-5) # [latent_dim]

    return mean, std


def normalise_latents(z: torch.Tensor, mean: torch.Tensor,
                      std: torch.Tensor) -> torch.Tensor:
    """
    Apply normalisation to a batch of latents.
    z:    [B, latent_dim, T]
    mean: [latent_dim]
    std:  [latent_dim]
    Returns: normalised z, same shape
    """
    mean = mean.to(z.device).view(1, -1, 1)
    std  = std.to(z.device).view(1, -1, 1)
    return (z - mean) / std


def denormalise_latents(z_norm: torch.Tensor, mean: torch.Tensor,
                         std: torch.Tensor) -> torch.Tensor:
    """Inverse of normalise_latents — used before passing to VAE decoder."""
    mean = mean.to(z_norm.device).view(1, -1, 1)
    std  = std.to(z_norm.device).view(1, -1, 1)
    return z_norm * std + mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt",  required=True)
    parser.add_argument("--config",    required=True)
    parser.add_argument("--out",       default="checkpoints/speech_vae/latent_stats.pt")
    parser.add_argument("--n_batches", type=int, default=500)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained VAE
    vae = SpeechVAE(
        base_channels      = cfg.get("encoder_base_channels", 64),
        strides            = cfg.get("encoder_strides", [6, 5, 4, 4, 4]),
        latent_dim         = cfg.get("latent_dim", 32),
        transformer_dim    = cfg.get("transformer_hidden", 512),
        transformer_layers = cfg.get("transformer_layers", 8),
        transformer_heads  = cfg.get("transformer_heads", 8),
        transformer_ffn    = cfg.get("transformer_ffn_dim", 2048),
        sliding_window     = cfg.get("transformer_sliding_window", 125),  # 10s×12.5Hz per Table 13
    ).to(device)

    ckpt = torch.load(args.vae_ckpt, map_location=device)
    # Handle both raw state_dict and wrapped checkpoint
    state = ckpt.get("vae", ckpt)
    vae.load_state_dict(state)
    print(f"Loaded VAE from {args.vae_ckpt}")

    loader = build_dataloader(cfg, num_workers=2)
    mean, std = compute_latent_stats(vae, loader, device, args.n_batches)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std}, args.out)

    print(f"\nLatent stats saved to {args.out}")
    print(f"  mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  std  range: [{std.min():.4f},  {std.max():.4f}]")
    print(f"\nExpected: mean ≈ 0, std ≈ 1 (KL pushes toward N(0,I))")
    print(f"If std >> 1, consider increasing λ_KL or training longer.")


if __name__ == "__main__":
    main()
