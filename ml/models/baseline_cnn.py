"""
VoiceShield Baseline CNN Model
==============================
A simple, classic 2D Spectrogram Convolutional Neural Network baseline.
Architecture:
  Input: Single-channel Log-Mel Spectrogram [B, 1, 64, T]
  Conv2D (1 -> 32, 3x3) + BN + ReLU + MaxPool(2, 2)
  Conv2D (32 -> 64, 3x3) + BN + ReLU + MaxPool(2, 2)
  Conv2D (64 -> 128, 3x3) + BN + ReLU + AdaptiveAvgPool2D(1, 1)
  Linear(128 -> 64) + ReLU + Dropout(0.3)
  Linear(64 -> 1) -> Single Logit output
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input has 3 channels (e.g. from AudioFeatureExtractor), select log-mel channel (channel 0)
        if x.dim() == 4 and x.size(1) == 3:
            x = x[:, :1, :, :]
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return logits
