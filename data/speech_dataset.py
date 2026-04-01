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
