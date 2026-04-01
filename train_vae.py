"""
train_vae.py — CALM Speech VAE-GAN training script
Single A100 40GB, TF32 precision, gradient checkpointing enabled.

Usage:
    python train_vae.py --config configs/speech_vae.yaml
"""

import os
import math
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
import wandb
from tqdm import tqdm

from models.vae             import SpeechVAE
from models.discriminators  import CombinedDiscriminator
from models.wavlm_distill   import WavLMDistillation
from losses.vae_losses      import (
    VAEGeneratorLoss, discriminator_adversarial_loss,
    flatten_disc_outputs
)
from data.speech_dataset    import build_dataloader


# ─────────────────────────────────────────────────────────────────────────────
# A100 performance flags
# ─────────────────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32  = True   # ~2× faster matmuls on A100
torch.backends.cudnn.allow_tf32        = True
torch.backends.cudnn.benchmark         = True
# Do NOT use fp16 AMP — known to cause NaN in audio GANs.
# TF32 is sufficient and stable.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/speech_vae.yaml")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(step: int, vae: SpeechVAE, disc: CombinedDiscriminator,
                    opt_g: AdamW, opt_d: AdamW,
                    sch_g, sch_d, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":      step,
        "vae":       vae.state_dict(),
        "disc":      disc.state_dict(),
        "opt_g":     opt_g.state_dict(),
        "opt_d":     opt_d.state_dict(),
        "sch_g":     sch_g.state_dict(),
        "sch_d":     sch_d.state_dict(),
    }, os.path.join(out_dir, f"ckpt_{step:07d}.pt"))
    # Symlink latest for convenience
    latest = os.path.join(out_dir, "latest.pt")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(f"ckpt_{step:07d}.pt", latest)


def load_checkpoint(path: str, vae, disc, opt_g, opt_d, sch_g, sch_d, device):
    ckpt = torch.load(path, map_location=device)
    vae.load_state_dict(ckpt["vae"])
    disc.load_state_dict(ckpt["disc"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    sch_g.load_state_dict(ckpt["sch_g"])
    sch_d.load_state_dict(ckpt["sch_d"])
    return ckpt["step"]


def get_kl_weight(step: int, warmup: int = 50_000,
                  target: float = 0.01) -> float:
    return target * min(step / max(warmup, 1), 1.0)


def train(cfg: dict, resume_path: str | None = None,
          use_wandb: bool = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Models ────────────────────────────────────────────────────────────────
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

    disc = CombinedDiscriminator().to(device)

    distiller = WavLMDistillation(
        latent_dim    = cfg.get("latent_dim", 32),
        vae_sr        = cfg.get("sample_rate", 24_000),
        wavlm_layer   = cfg.get("wavlm_layer", 7),
    ).to(device)

    # Enable gradient checkpointing on the VAE to save ~3× VRAM
    # (30% speed penalty — worth it on a single A100)
    vae.encoder.transformer.blocks = nn.ModuleList([
        torch.utils.checkpoint.checkpoint_wrapper(block)
        for block in vae.encoder.transformer.blocks
    ])

    loss_fn = VAEGeneratorLoss(
        lambda_kl     = cfg.get("lambda_kl", 0.01),
        lambda_adv    = cfg.get("lambda_adv", 1.0),
        lambda_feat   = cfg.get("lambda_feat", 5.0),
        lambda_distil = cfg.get("lambda_distil", 25.0),
        lambda_recon  = cfg.get("lambda_recon", 0.0),
    )

    # ── Optimisers ────────────────────────────────────────────────────────────
    # Generator: VAE + distillation projection (WavLM itself is frozen)
    # Moshi paper: AdamW with β₁=0.5, β₂=0.9, weight decay ONLY on Transformer params
    # "We apply weight decay only to the parameters of the Transformers, weight=5e-2"
    transformer_params, other_params = [], []
    for name, p in vae.named_parameters():
        if "transformer" in name:
            transformer_params.append(p)
        else:
            other_params.append(p)
    other_params += list(distiller.projection.parameters())

    gen_param_groups = [
        {"params": transformer_params, "weight_decay": cfg.get("transformer_weight_decay", 5e-2)},
        {"params": other_params,       "weight_decay": 0.0},
    ]
    betas = tuple(cfg.get("betas", [0.5, 0.9]))  # Moshi: β₁=0.5, β₂=0.9
    opt_g = AdamW(gen_param_groups, lr=cfg.get("learning_rate", 8e-4), betas=betas)

    # Discriminator uses same betas, no weight decay
    opt_d = AdamW(disc.parameters(),
                  lr=cfg.get("disc_learning_rate", 3e-4),
                  betas=betas,
                  weight_decay=0.0)

    # EMA on generator weights — Moshi paper: decay=0.99
    # torch.optim.swa_utils.AveragedModel handles EMA cleanly
    from torch.optim.swa_utils import AveragedModel
    ema_decay = cfg.get("ema_decay", 0.99)
    vae_ema   = AveragedModel(vae, multi_avg_fn=lambda averaged, current, num_averaged:
                              ema_decay * averaged + (1 - ema_decay) * current)

    max_steps = cfg.get("max_steps", 500_000)
    sch_g = CosineAnnealingLR(opt_g, T_max=max_steps, eta_min=1e-6)
    sch_d = CosineAnnealingLR(opt_d, T_max=max_steps, eta_min=1e-6)

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = build_dataloader(cfg, num_workers=4)

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step = 0
    if resume_path:
        global_step = load_checkpoint(
            resume_path, vae, disc, opt_g, opt_d, sch_g, sch_d, device)
        print(f"Resumed from step {global_step}")

    # ── WandB ─────────────────────────────────────────────────────────────────
    if use_wandb:
        wandb.init(project="calm-vae", config=cfg, resume="allow")

    # ── Training loop ─────────────────────────────────────────────────────────
    grad_accum    = cfg.get("grad_accumulation", 8)
    save_every    = cfg.get("save_every", 10_000)
    grad_clip     = cfg.get("grad_clip_norm", 1000.0)
    out_dir       = cfg.get("output_dir", "./checkpoints/speech_vae")

    vae.train()
    disc.train()

    opt_g.zero_grad()
    opt_d.zero_grad()
    accum_step = 0

    pbar = tqdm(total=max_steps, initial=global_step, desc="VAE training")

    while global_step < max_steps:
        for real_audio in loader:
            if global_step >= max_steps:
                break

            real_audio = real_audio.to(device)   # [B, 1, T]

            # ── DISCRIMINATOR UPDATE ────────────────────────────────────────
            # Encode+decode with no gradients for the D step — detach the
            # reconstruction so disc sees a stopped-gradient fake sample.
            vae.eval()  # batchnorm/dropout in eval for D step
            with torch.no_grad():
                _, _, _, fake_audio = vae(real_audio)
            vae.train()

            real_out = disc(real_audio.detach())
            fake_out = disc(fake_audio.detach())

            real_logits, _ = flatten_disc_outputs(real_out)
            fake_logits, _ = flatten_disc_outputs(fake_out)

            d_loss = discriminator_adversarial_loss(real_logits, fake_logits)
            (d_loss / grad_accum).backward()

            # ── GENERATOR UPDATE ───────────────────────────────────────────
            # Full forward pass again — gradients flow through VAE this time.
            z, mu, logvar, fake_audio = vae(real_audio)

            # WavLM distillation uses mu (deterministic mean), not the noisy
            # sample z — matches paper Section 5.1: "inner latent representation"
            distil_loss = distiller(real_audio, mu)

            # Discriminator forward on fake (need gradients this time)
            fake_out  = disc(fake_audio)
            real_out  = disc(real_audio.detach())

            fake_logits,  fake_feats = flatten_disc_outputs(fake_out)
            _,            real_feats = flatten_disc_outputs(real_out)

            # Dynamic KL annealing
            kl_w = get_kl_weight(global_step,
                                  warmup=cfg.get("kl_warmup_steps", 50_000),
                                  target=cfg.get("lambda_kl", 0.01))
            loss_fn.lambda_kl = kl_w

            g_loss, breakdown = loss_fn(
                mu, logvar,
                fake_logits, real_feats, fake_feats,
                distil_loss,
            )
            g_loss = g_loss / grad_accum
            g_loss.backward()

            accum_step += 1

            # ── Gradient accumulation step ─────────────────────────────────
            if accum_step == grad_accum:
                torch.nn.utils.clip_grad_norm_(gen_params,  grad_clip)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)

                opt_g.step(); sch_g.step(); opt_g.zero_grad()
                opt_d.step(); sch_d.step(); opt_d.zero_grad()
                accum_step = 0
                global_step += 1

                # ── Logging ───────────────────────────────────────────────
                breakdown["loss/disc"] = d_loss.item() * grad_accum
                breakdown["train/kl_weight"] = kl_w
                breakdown["train/lr_gen"] = sch_g.get_last_lr()[0]
                breakdown["train/step"] = global_step

                pbar.set_postfix({
                    "g": f"{breakdown['loss/total_gen']:.3f}",
                    "d": f"{breakdown['loss/disc']:.3f}",
                    "kl": f"{breakdown['loss/kl']:.4f}",
                    "dist": f"{breakdown['loss/distil']:.4f}",
                })
                pbar.update(1)

                if use_wandb:
                    wandb.log(breakdown, step=global_step)

                # ── Checkpoint ────────────────────────────────────────────
                # Update EMA after each optimiser step
                vae_ema.update_parameters(vae)

                if global_step % save_every == 0:
                    save_checkpoint(global_step, vae, disc,
                                    opt_g, opt_d, sch_g, sch_d, out_dir)
                    # Also save EMA weights — use these for inference/evaluation
                    torch.save(vae_ema.module.state_dict(),
                               os.path.join(out_dir, f"vae_ema_{global_step:07d}.pt"))
                    print(f"\nSaved checkpoint at step {global_step}")

    pbar.close()

    # Save final model
    save_checkpoint(global_step, vae, disc, opt_g, opt_d, sch_g, sch_d, out_dir)

    # Save VAE-only weights for downstream CALM training (no discriminator)
    torch.save(vae.state_dict(),
               os.path.join(out_dir, "vae_final.pt"))
    # EMA weights are what you should use for the CALM backbone
    torch.save(vae_ema.module.state_dict(),
               os.path.join(out_dir, "vae_ema_final.pt"))
    print(f"\nTraining complete. VAE saved to {out_dir}/vae_final.pt")
    print(f"EMA weights saved to {out_dir}/vae_ema_final.pt — use these for CALM training.")


if __name__ == "__main__":
    args   = parse_args()
    cfg    = load_config(args.config)
    train(cfg, resume_path=args.resume, use_wandb=not args.no_wandb)