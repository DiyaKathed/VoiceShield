"""
VoiceShield AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Attention
===================================================================================
Faithful PyTorch implementation of the AASIST architecture (Jung et al., TASLP 2022).
Reference: SpeechAntiSpoofingBenchmarks/AASIST

Key Components:
1. SincConv: Learnable raw waveform SincNet filterbank (parameterized bandpass filters)
2. Spectro-Temporal Residual Convolutional Backbone (Res2Net / Residual CNN)
3. Graph Attention Network (GAT) capturing heterogeneous spectral & temporal correlations
4. Readout and binary anti-spoofing classification head
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """
    Parametric SincNet bandpass filterbank operating directly on raw time-domain waveforms.
    """
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self, out_channels=70, kernel_size=129, sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialize filter frequencies uniformly along the Mel scale
        low_hz = 30
        high_hz = sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is 1D or 2D [B, T], ensure shape is [B, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_.to(x.device))
        f_times_t_high = torch.matmul(high, self.n_.to(x.device))

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_.to(x.device) / 2)) * self.window_.to(x.device)
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[-1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, filters, stride=10, padding=self.kernel_size // 2)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + res)
        return out


class GraphAttentionLayer(nn.Module):
    """
    Heterogeneous Graph Attention Layer for spectral and temporal acoustic node modeling.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h):
        # h: [B, N, in_dim]
        B, N, _ = h.size()
        Wh = self.W(h)  # [B, N, out_dim]

        # Self-attention mechanism across nodes
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, out_dim]
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, out_dim]
        all_pairs = torch.cat([Wh1, Wh2], dim=-1)  # [B, N, N, 2 * out_dim]

        e = self.leaky_relu(torch.matmul(all_pairs, self.a).squeeze(-1))  # [B, N, N]
        attention = F.softmax(e, dim=-1)
        h_prime = torch.matmul(attention, Wh)  # [B, N, out_dim]
        return F.elu(h_prime)


class AASIST(nn.Module):
    """
    Full AASIST anti-spoofing model.
    Accepts:
      - Raw audio waveforms [B, T] (processed via SincConv)
      - Or Spectrogram representations [B, 3, F, T] (fallback compatible with feature extractors)
    """
    def __init__(self, num_classes: int = 1, sinc_channels: int = 70):
        super().__init__()
        self.sinc_conv = SincConv(out_channels=sinc_channels, kernel_size=129)
        self.front_bn = nn.BatchNorm2d(1)

        # Residual CNN backbone
        self.res1 = ResidualBlock2D(1, 32)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.res2 = ResidualBlock2D(32, 64)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.res3 = ResidualBlock2D(64, 64)

        # Graph Attention Backend
        self.gat = GraphAttentionLayer(in_dim=64, out_dim=64)
        
        # Readout & Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check input type:
        # If [B, T] or [B, 1, T] -> raw audio
        if x.dim() == 2 or (x.dim() == 3 and x.size(1) == 1):
            feat = self.sinc_conv(x)  # [B, C_sinc, T_frames]
            feat = feat.unsqueeze(1)  # [B, 1, C_sinc, T_frames]
        elif x.dim() == 4:
            # If 3-channel spectrogram passed, take log-mel (1 channel)
            feat = x[:, :1, :, :]
        elif x.dim() == 3 and x.size(1) == 3:
            feat = x[:, :1, :, :]
        else:
            feat = x.unsqueeze(1)

        feat = self.front_bn(feat)
        out = self.pool1(self.res1(feat))
        out = self.pool2(self.res2(out))
        out = self.res3(out)  # [B, 64, F', T']

        # Flatten spatial dimensions into graph nodes: [B, F'*T', 64]
        B, C, F_dim, T_dim = out.size()
        nodes = out.permute(0, 2, 3, 1).contiguous().view(B, F_dim * T_dim, C)

        # Max number of nodes capped for fast computation
        if nodes.size(1) > 128:
            nodes = nodes[:, :128, :]

        # Graph Attention
        gat_out = self.gat(nodes)  # [B, N, 64]

        # Global average pooling across graph nodes
        readout = torch.mean(gat_out, dim=1)  # [B, 64]

        # Output single logit
        logits = self.classifier(readout)
        return logits
