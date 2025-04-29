import logging
import math
import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio  # type: ignore
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def is_silence(audio: Tensor, thresh: int = -60):
    dBmax = 20 * torch.log10(torch.flatten(audio.abs()).max())
    return dBmax < thresh


class Mean(nn.Module):
    def __init__(self, dim: int = 0, keepdim: bool = True):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input):
        return torch.mean(input, dim=self.dim, keepdim=self.keepdim)


class WAVDataset(Dataset):
    def __init__(
        self,
        wav_files: Sequence[str],
        transforms: Optional[Callable] = None,
        target_sr: int = 16000,
        target_length: float = 10,
        mono: bool = False,
        random_slice: bool = False,
        audio_slice_length: Optional[float] = None,
        audio_slice_overlap: Optional[float] = None,
        check_silence: bool = False,
        window_size: Optional[int] = None,
        overlap: float = 0.5,
    ):
        self.data = []
        self.wavs = wav_files
        self.sample_rate = target_sr
        self.check_silence = check_silence

        self.target_length = target_length

        # Sequential crop settings
        self.audio_slice_length = audio_slice_length
        self.audio_slice_overlap = audio_slice_overlap

        # Get sample rate from the first file
        if self.wavs[0].endswith(".wav"):
            waveform, sample_rate = torchaudio.load(self.wavs[0])
        elif self.wavs[0].endswith(".npy"):
            waveform = torch.from_numpy(np.load(self.wavs[0]))
            sample_rate = 16000

        self.target_sample_rate = sample_rate if not sample_rate else self.sample_rate

        self.wav_shape = None
        self.perform_checks = False
        self.window_size = (
            window_size if not None else int(self.target_length * self.sample_rate)
        )
        self.overlap = overlap

        # index for sequential cropping of audio files
        if audio_slice_overlap is not None:
            # Get index for cropping the audio files sequentially
            length_audio = waveform.shape[-1]
            self.hop_length = int(self.window_size * (1 - self.overlap))

            for wav in self.wavs:
                for i in range(0, length_audio - self.hop_length, self.hop_length):
                    self.data.append((wav, i, i + self.window_size))
        else:
            self.data = [(wav, 0, -1) for wav in self.wavs]

        self.ratio_lists = int(len(self.data) / len(self.wavs))

        if random_slice:
            self.audio_fn = self.optimized_random_crop
        elif audio_slice_overlap is not None:
            self.audio_fn = self.optimized_sequential_crop
        else:
            self.audio_fn = self.get_audio

        if transforms is not None:
            self.transforms = transforms
        elif mono:
            self.transforms = nn.Sequential(
                Mean(dim=0, keepdim=True) if mono else nn.Identity(),
            )
        else:
            self.transforms = nn.Identity()

        self.resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=self.target_sample_rate
        ).to(waveform)

    # Instead of loading the whole file and chopping out our crop,
    # we only load what we need.
    def optimized_random_crop(self, idx: int) -> Tuple[Tensor, int]:
        # Get length/audio info
        info = torchaudio.info(self.wavs[idx])
        length = info.num_frames
        sample_rate = info.sample_rate

        # Calculate correct number of samples to read based on actual
        # and intended sample rate
        ratio = (
            1
            if (self.target_sample_rate is None)
            else sample_rate / self.target_sample_rate
        )
        crop_size = (
            length
            if (self.window_size is None)
            else math.ceil(self.window_size * ratio)
        )  # type: ignore
        frame_offset = random.randint(0, max(length - crop_size, 0))

        # Load the samples
        waveform, sample_rate = torchaudio.load(
            uri=self.wavs[idx], frame_offset=frame_offset, num_frames=crop_size
        )

        # Pad with zeroes if the sizes aren't quite right
        # (e.g., rates aren't exact multiples)
        if len(waveform[0]) < crop_size:
            waveform = torch.nn.functional.pad(
                waveform,
                pad=(0, crop_size - len(waveform[0])),
                mode="constant",
                value=0,
            )

        return waveform, sample_rate

    def optimized_sequential_crop(self, idx: int) -> Tuple[Tensor, int]:
        # Get length/audio info
        filename, start, end = self.data[idx]
        info = torchaudio.info(filename)
        length = info.num_frames
        sample_rate = info.sample_rate

        # Load the samples
        waveform, sample_rate = torchaudio.load(
            uri=filename, frame_offset=start, num_frames=end - start
        )

        if len(waveform[0]) != self.window_size:
            waveform = torch.nn.functional.pad(
                waveform,
                pad=(0, self.window_size - len(waveform[0])),
                mode="constant",
                value=0,
            )

        return waveform, sample_rate

    def get_audio(self, idx: int) -> Tuple[Tensor, int]:
        audio, sr = torchaudio.load(uri=self.wavs[idx])
        # Pad with zeroes if the length is too short
        if len(audio[0]) < self.target_length * sr:
            audio = torch.nn.functional.pad(
                audio,
                pad=(0, int(self.target_length * sr) - len(audio[0])),
                mode="constant",
                value=0,
            )

        # Slice the audio if it is too long
        if len(audio[0]) > self.target_length * sr:
            audio = audio[:, : int(self.target_length * sr)]

        return audio, sr

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tensor,
        Tuple[Tensor, int],
        Tuple[Tensor, Tensor],
        Tuple[Tensor, List[str], List[str]],
    ]:  # type: ignore
        waveform, sample_rate = self.audio_fn(int(idx))

        # Apply sample rate transform if necessary
        if self.target_sample_rate and sample_rate != self.target_sample_rate:
            waveform = self.resample(waveform)

            # Downsampling can result in slightly different sizes.
            if hasattr(self, "window_size"):
                waveform = waveform[:, : self.window_size]

        # Apply other transforms
        if self.transforms is not None:
            waveform = self.transforms(waveform)

        if self.perform_checks:
            # Check silence after transforms (useful for random crops)
            if self.check_silence and is_silence(waveform):
                invalid_audio = True
                logger.warning(f"Silent audio file: {self.data[idx][0]}")

            if self.wav_shape and waveform.shape != self.wav_shape:
                pass
            self.wav_shape = waveform.shape

        return waveform

    def __len__(self) -> int:
        return len(self.data)
