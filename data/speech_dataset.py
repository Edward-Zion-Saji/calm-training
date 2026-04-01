"""
data/speech_dataset.py

Loads English speech from local paths on the VM.
Handles the short-clip problem by concatenating consecutive utterances
from the same speaker/chapter until a full window can be sampled.
This matches how Moshi/CALM trained on continuous long-form audio.

Datasets:
  - LibriTTS train-clean-360  (24kHz, avg 6.1s per utterance)
  - SPICOR English 2 speakers (44.1kHz -> 24kHz, avg 7.1s per utterance)
"""

import os
import random
import glob
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset


TARGET_SR = 24_000


def _group_by_speaker(files):
    """
    Group .wav files by speaker/chapter so consecutive clips can be
    concatenated to fill the target window length.

    LibriTTS structure: .../train-clean-360/{speaker}/{chapter}/*.wav
    SPICOR structure:   .../IISc_SPICOR_Data/{speaker}/wav/*.wav
    Returns: list of file-groups, each group is a sorted list of paths
    from the same speaker/chapter.
    """
    groups = defaultdict(list)
    for f in files:
        parts = Path(f).parts
        # Use parent dir as group key — files in same dir = same speaker/chapter
        key = str(Path(f).parent)
        groups[key].append(f)
    # Sort files within each group for deterministic ordering
    return [sorted(v) for v in groups.values() if v]


class ConcatWindowDataset(Dataset):
    """
    Builds fixed-length windows by concatenating consecutive utterances
    from the same speaker/chapter group.

    For each sample:
      1. Pick a random group (speaker/chapter)
      2. Pick a random starting utterance within that group
      3. Concatenate utterances forward until >= clip_samples
      4. Randomly crop a clip_samples window from the concatenated audio

    This avoids zero-padding short clips and produces natural-sounding
    windows that match how Moshi trained on continuous audio streams.
    """

    def __init__(self, root: str, clip_seconds: float = 12.0,
                 target_sr: int = TARGET_SR, orig_sr: int = None):
        self.clip_samples = int(clip_seconds * target_sr)
        self.target_sr    = target_sr
        self.orig_sr      = orig_sr

        all_files = sorted(glob.glob(
            os.path.join(root, "**", "*.wav"), recursive=True))
        if not all_files:
            raise ValueError(f"No .wav files found under {root}")

        self.groups = _group_by_speaker(all_files)
        self.n_files = len(all_files)
        print(f"  {Path(root).name}: {self.n_files:,} files in "
              f"{len(self.groups)} speaker/chapter groups")

        # Resample if needed
        self._resampler = None
        if orig_sr and orig_sr != target_sr:
            self._resampler = T.Resample(orig_sr, target_sr)

    def __len__(self):
        # Each group can yield many non-overlapping windows
        # Use n_files as a proxy — each file contributes ~1 window on average
        return self.n_files

    def _load_wav(self, path: str) -> torch.Tensor:
        """Load, mono-downmix, resample. Returns 1D tensor."""
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = waveform.squeeze(0)   # [T]
        if sr != self.target_sr:
            rs = self._resampler if (self._resampler and sr == self.orig_sr)                  else T.Resample(sr, self.target_sr)
            waveform = rs(waveform)
        return waveform

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Pick a random group and build a window by concatenation
        group = random.choice(self.groups)

        # Pick a random start position within the group
        start_idx = random.randint(0, len(group) - 1)

        # Concatenate utterances forward until we have enough audio
        chunks = []
        total = 0
        i = start_idx
        while total < self.clip_samples and i < len(group):
            wav = self._load_wav(group[i])
            chunks.append(wav)
            total += len(wav)
            i += 1

        # If still not enough (group was too short), wrap around or pad lightly
        if total < self.clip_samples:
            # Try from the beginning of the group
            i = 0
            while total < self.clip_samples and i < start_idx:
                wav = self._load_wav(group[i])
                chunks.append(wav)
                total += len(wav)
                i += 1

        audio = torch.cat(chunks)   # [total_samples]

        # Normalise amplitude
        peak = audio.abs().max()
        if peak > 1e-6:
            audio = audio / peak

        # Random crop if longer than needed
        if len(audio) >= self.clip_samples:
            start = random.randint(0, len(audio) - self.clip_samples)
            audio = audio[start: start + self.clip_samples]
        else:
            # Only pad as last resort (very short groups)
            audio = F.pad(audio, (0, self.clip_samples - len(audio)))

        return audio.unsqueeze(0)   # [1, T]


def build_dataloader(cfg: dict, num_workers: int = 4) -> DataLoader:
    """Builds a DataLoader from local English speech using concatenation windows."""
    clip_seconds = cfg.get("clip_seconds", 12.0)
    batch_size   = cfg.get("batch_size", 2)

    datasets = []

    libritts_root = cfg.get(
        "libritts_root",
        "/datadrive/data/libri_tts/LibriTTS/train-clean-360")
    if os.path.exists(libritts_root):
        print("Loading LibriTTS train-clean-360 (concat-window mode)...")
        datasets.append(ConcatWindowDataset(
            root         = libritts_root,
            clip_seconds = clip_seconds,
            target_sr    = TARGET_SR,
            orig_sr      = 24_000,
        ))

    spicor_root = cfg.get(
        "spicor_root",
        "/datadrive/data/spicor/IISc_SPICOR_Data")
    if os.path.exists(spicor_root):
        print("Loading SPICOR English (concat-window mode)...")
        datasets.append(ConcatWindowDataset(
            root         = spicor_root,
            clip_seconds = clip_seconds,
            target_sr    = TARGET_SR,
            orig_sr      = 44_100,
        ))

    if not datasets:
        raise RuntimeError("No datasets found.")

    combined = ConcatDataset(datasets)
    total_clips = len(combined)
    total_hours = total_clips * clip_seconds / 3600
    print(f"Total: {total_clips:,} windows  (~{total_hours:.0f}h at {clip_seconds}s each)")

    return DataLoader(
        combined,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = True,
        drop_last          = True,
        persistent_workers = num_workers > 0,
    )
