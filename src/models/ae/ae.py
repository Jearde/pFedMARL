import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, fc_units: int, latent_dim: int):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.fc_units = fc_units
        self.latent_dim = latent_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.fc_units),
            torch.nn.BatchNorm1d(self.fc_units, momentum=0.01, eps=1e-03),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc_units, self.fc_units),
            torch.nn.BatchNorm1d(self.fc_units, momentum=0.01, eps=1e-03),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc_units, self.fc_units),
            torch.nn.BatchNorm1d(self.fc_units, momentum=0.01, eps=1e-03),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc_units, self.fc_units),
            torch.nn.BatchNorm1d(self.fc_units, momentum=0.01, eps=1e-03),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc_units, self.latent_dim),
            torch.nn.BatchNorm1d(self.latent_dim, momentum=0.01, eps=1e-03),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, fc_units: int, latent_dim: int):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.fc_units = fc_units
        self.latent_dim = latent_dim

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.fc_units),
            torch.nn.BatchNorm1d(self.fc_units, momentum=0.01, eps=1e-03),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc_units, self.fc_units),
            torch.nn.BatchNorm1d(self.fc_units, momentum=0.01, eps=1e-03),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc_units, self.fc_units),
            torch.nn.BatchNorm1d(self.fc_units, momentum=0.01, eps=1e-03),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc_units, self.fc_units),
            torch.nn.BatchNorm1d(self.fc_units, momentum=0.01, eps=1e-03),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc_units, self.input_dim),
        )

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(
        self, input_dim: list, block_size: int, fc_units: int = 128, latent_dim: int = 8
    ):
        super().__init__()
        self.input_dim = input_dim[-1]
        self.fc_units = fc_units
        self.latent_dim = latent_dim

        self.encoder = Encoder(self.input_dim, self.fc_units, self.latent_dim)
        self.decoder = Decoder(self.input_dim, self.fc_units, self.latent_dim)
        self.cov_source = torch.nn.Parameter(
            torch.zeros(block_size, block_size), requires_grad=False
        )
        self.cov_target = torch.nn.Parameter(
            torch.zeros(block_size, block_size), requires_grad=False
        )

    def forward(self, x):
        z = self.encoder(x.view(-1, self.input_dim))
        y = self.decoder(z)

        return y, z
