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
    Generator hinge loss — Moshi paper uses hinge (not LSGAN).
    L_gen = mean(ReLU(1 - D(x̂)))  →  generator wants D(fake) ≥ 1
    """
    loss = 0.0
    for logit in fake_logits:
        loss = loss + F.relu(1.0 - logit).mean()
    return loss / len(fake_logits)


def discriminator_adversarial_loss(real_logits: list[torch.Tensor],
                                   fake_logits: list[torch.Tensor]
                                   ) -> torch.Tensor:
    """
    Discriminator hinge loss — Moshi paper (Audiocraft config: adv_loss=hinge).
    L_disc = mean(ReLU(1 - D(real))) + mean(ReLU(1 + D(fake)))
    """
    loss = 0.0
    for real, fake in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return loss / len(real_logits)


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

    Each discriminator returns (logits_list, feats_list) where:
      - logits_list is a list of tensors (one per sub-discriminator resolution)
      - feats_list  is a list of lists   (one inner list per sub-discriminator,
                                          each containing intermediate tensors)

    Returns:
      all_logits:   flat list[Tensor]       — one per sub-discriminator
      all_features: list[list[Tensor]]      — one inner list per sub-discriminator
    """
    all_logits   = []
    all_features = []
    for name, (logits, feats) in disc_outputs.items():
        all_logits.extend(logits)
        # feats is already list[list[Tensor]] from MSSTFTD/MPD/MSD
        # Guard against an empty feature list or flat list of tensors
        if feats and isinstance(feats[0], list):
            all_features.extend(feats)
        elif feats:
            all_features.append(feats)   # single-disc case: wrap in outer list
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