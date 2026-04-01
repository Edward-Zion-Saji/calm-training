"""
data/speech_dataset.py

Loads English speech from local paths on the VM:
  - LibriTTS train-clean-360  (24kHz, ready to use)
  - SPICOR English 2 speakers (44.1kHz, resampled on-the-fly to 24kHz)

No HuggingFace downloading — reads directly from /datadrive/data/.
"""

import os
import random
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset


TARGET_SR = 24_000


class LocalWavDataset(Dataset):
    """
    Loads all .wav files recursively from a directory.
    Resamples to TARGET_SR on-the-fly if needed.
    Returns fixed-length mono clips of clip_samples samples.
    """

    def __init__(self, root: str, clip_seconds: float = 2.0,
                 target_sr: int = TARGET_SR, orig_sr: int | None = None):
        self.clip_samples = int(clip_seconds * target_sr)
        self.target_sr    = target_sr
        self.orig_sr      = orig_sr   # if known ahead of time, skip torchaudio.info

        self.files = sorted(glob.glob(os.path.join(root, "**", "*.wav"),
                                      recursive=True))
        if not self.files:
            raise ValueError(f"No .wav files found under {root}")

        print(f"  {Path(root).name}: {len(self.files):,} files")

        # Build resampler once if orig_sr is fixed
        self._resampler = None
        if orig_sr is not None and orig_sr != target_sr:
            self._resampler = T.Resample(orig_sr, target_sr)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]

        waveform, sr = torchaudio.load(path)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        # Resample
        if sr != self.target_sr:
            if self._resampler is not None:
                waveform = self._resampler(waveform)
            else:
                waveform = T.Resample(sr, self.target_sr)(waveform)

        waveform = waveform.squeeze(0)   # [T]

        # Normalise amplitude
        peak = waveform.abs().max()
        if peak > 1e-6:
            waveform = waveform / peak

        # Crop or pad to fixed length
        T_audio = waveform.shape[0]
        if T_audio >= self.clip_samples:
            start = random.randint(0, T_audio - self.clip_samples)
            waveform = waveform[start: start + self.clip_samples]
        else:
            waveform = F.pad(waveform, (0, self.clip_samples - T_audio))

        return waveform.unsqueeze(0)   # [1, T]


def build_dataloader(cfg: dict, num_workers: int = 4) -> DataLoader:
    """
    Builds a DataLoader from local English speech datasets.
    Only uses what is confirmed present on /datadrive/data/.
    """
    clip_seconds = cfg.get("clip_seconds", 2.0)
    batch_size   = cfg.get("batch_size", 8)

    datasets = []

    # ── LibriTTS train-clean-360  (24kHz, no resampling needed) ──────────────
    libritts_root = cfg.get(
        "libritts_root",
        "/datadrive/data/libri_tts/LibriTTS/train-clean-360"
    )
    if os.path.exists(libritts_root):
        print("Loading LibriTTS train-clean-360...")
        datasets.append(LocalWavDataset(
            root        = libritts_root,
            clip_seconds = clip_seconds,
            target_sr   = TARGET_SR,
            orig_sr     = 24_000,   # already 24kHz — skip torchaudio.info call
        ))
    else:
        print(f"WARNING: LibriTTS not found at {libritts_root}")

    # ── SPICOR English (44.1kHz → resample to 24kHz) ─────────────────────────
    spicor_root = cfg.get(
        "spicor_root",
        "/datadrive/data/spicor/IISc_SPICOR_Data"
    )
    if os.path.exists(spicor_root):
        print("Loading SPICOR English...")
        datasets.append(LocalWavDataset(
            root        = spicor_root,
            clip_seconds = clip_seconds,
            target_sr   = TARGET_SR,
            orig_sr     = 44_100,   # all SPICOR files are 44.1kHz
        ))
    else:
        print(f"WARNING: SPICOR not found at {spicor_root}")

    if not datasets:
        raise RuntimeError("No datasets found. Check libritts_root and spicor_root in config.")

    combined = ConcatDataset(datasets)
    total_clips = len(combined)
    total_hours = total_clips * clip_seconds / 3600
    print(f"Total: {total_clips:,} clips  (~{total_hours:.0f}h at {clip_seconds}s each)")

    return DataLoader(
        combined,
        batch_size       = batch_size,
        shuffle          = True,
        num_workers      = num_workers,
        pin_memory       = True,
        drop_last        = True,
        persistent_workers = num_workers > 0,
    )
