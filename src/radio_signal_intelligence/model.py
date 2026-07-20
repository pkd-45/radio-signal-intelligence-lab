from __future__ import annotations

import torch
from torch import nn


class TinyMultimodalCNN(nn.Module):
    """Small ON/OFF waterfall CNN with a metadata side branch."""

    def __init__(self, n_metadata: int = 6, n_classes: int = 5):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(n_metadata),
            nn.Linear(n_metadata, 16),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(24 + 16, 32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, n_classes),
        )

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images).flatten(1)
        metadata_features = self.metadata_encoder(metadata)
        return self.classifier(torch.cat([image_features, metadata_features], dim=1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
