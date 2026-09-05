"""
VoiceShield Neural Network Architecture
=======================================
Deep Convolutional Neural Network with Spectro-Temporal Residual blocks
and Attention Pooling for Audio Deepfake & Voice Clone Detection.

RESEARCH DESIGN NOTES:
----------------------
1. TEMPORAL & SPECTRAL PATTERN DISCRIMINATION:
   Neural vocoder artifacts manifest as micro-spectral discontinuities along
   the frequency axis (formant phase mismatch) and unnatural smoothness along
   the temporal axis. The multi-scale 2D convolutions simultaneously capture
   both spectral patterns (vertical filter extent) and temporal transitions (horizontal).

2. GENERALIZATION TO UNSEEN VOCODERS (Out-of-Distribution):
   We apply Spatial Dropout2D and Batch Normalization to prevent the network from
   overfitting to specific speaker pitch or specific vocoder spectral signatures,
   promoting generalized boundary learning.

3. TEMPORAL POOLING:
   Adaptive Average Pooling aggregates the 2D feature maps over variable temporal
   lengths into a fixed-dimension embedding vector, ensuring consistent classification
   across varying utterance lengths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class ConvBlock(nn.Module):
    """Convolutional block with Batch Normalization, LeakyReLU, and Spatial Dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        # Residual projection matching spatial downsampling exactly
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.residual = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


class VoiceShieldNet(nn.Module):
    """
    VoiceShield Spectro-Temporal Deepfake Classifier.
    Input: [Batch, 3, n_mels (64), time_frames (~47)]
    Output: Single raw logit (apply sigmoid for AI-generated probability)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, dropout: float = 0.3):
        super().__init__()

        # Acoustic representation encoders
        self.block1 = ConvBlock(in_channels, 32, dropout_rate=0.1)
        self.block2 = ConvBlock(32, 64, dropout_rate=0.15)
        self.block3 = ConvBlock(64, 128, dropout_rate=0.2)
        self.block4 = ConvBlock(128, 256, dropout_rate=0.25)

        # Global aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns: raw logits of shape [Batch, 1]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes calibrated probabilities: (ai_probability, human_probability)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            ai_probs = torch.sigmoid(logits).squeeze(-1)
            human_probs = 1.0 - ai_probs
        return ai_probs, human_probs


def get_model(device: torch.device = torch.device('cpu')) -> VoiceShieldNet:
    """Helper factory to create model on specified device."""
    model = VoiceShieldNet()
    return model.to(device)
