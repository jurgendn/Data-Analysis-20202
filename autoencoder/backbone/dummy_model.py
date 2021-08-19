from typing import Any

import torch

import torch.nn as nn
import torch.nn.functional as F


class DummyAE(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super(DummyAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=3),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=3),
            nn.BatchNorm2d(128)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=3),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=3),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, kernel_size=(5, 5)),
            nn.BatchNorm2d(3)
        )

    def forward(self, x: torch.Tensor, only_encoder: bool) -> torch.Tensor:
        x = self.encoder(x)
        codes = F.relu(x)
        if only_encoder:
            return codes
        x = self.decoder(codes)
        x = torch.sigmoid(x)
        return x

