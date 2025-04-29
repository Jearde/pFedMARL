import logging

import torch
import torch.nn as nn
import torchaudio  # type: ignore

logger = logging.getLogger(__name__)


class Linear_LMBE(nn.Module):
    def __init__(
        self,
        n_frames: int,
        n_mels: int,
        frame_hop_length: int,
        n_fft: int,
        hop_length: int,
        power: int,
        fmax: int,
        fmin: int,
        win_length: int,
        sr: int = 16000,
        shuffle_vectors: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.training = False

        self.n_frames = n_frames
        self.n_mels = n_mels
        self.frame_hop_length = int(1)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.fmax = fmax
        self.fmin = fmin
        self.win_length = win_length
        self.sr = sr

        self.create_vectors = True
        self.shuffle_vectors = shuffle_vectors

        self.dims = self.n_mels * self.n_frames

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            power=self.power,
            normalized=False,
            mel_scale="htk",  # 'htk' or 'slaney'
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power" if self.power == 2 else "magnitude",
        )

        # Set all submodules to eval mode
        for module in self.children():
            module.eval()

    def to_vector(self, x):
        n_vectors = x.shape[-1] - self.n_frames + 1

        # skip too short clips
        if n_vectors < 1:
            return torch.empty((0, self.dims))

        # Above for loop but faster as it does not use for loop
        vectors = x.unfold(dimension=-1, size=n_vectors, step=self.frame_hop_length)
        vectors = vectors.permute(0, 2, 1, 3)
        vectors = vectors.contiguous().view(x.shape[0], n_vectors, -1)

        vectors = vectors.view(-1, self.dims)

        return vectors

    def labels_to_vector(self, y, x):
        n_vectors = int(x.shape[0] / y[0].shape[0])
        return [
            torch.repeat_interleave(yi, n_vectors, dim=0).to(yi.dtype).to(x.device)
            for yi in y
        ]

    def forward(self, x, y=None, shuffle=None):
        shuffle = shuffle if shuffle is not None else self.shuffle_vectors
        if x.dim() == 3:
            x = torch.mean(x, dim=-2, keepdim=False)

        mel_spec = self.mel_spectrogram(x)
        lmbes = self.amplitude_to_db(mel_spec)

        if self.create_vectors:
            vectors = self.to_vector(lmbes).to(x.device)
            y = self.labels_to_vector(y, vectors) if y is not None else None

            if shuffle:
                indices = torch.randperm(vectors.shape[0])
                vectors = vectors[indices]
                y = [yi[indices] for yi in y] if y is not None else None
        else:
            vectors = lmbes.reshape(x.shape[0], -1)

        return vectors, y
