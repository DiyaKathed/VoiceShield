"""
VoiceShield Dataset Module
==========================
PyTorch Dataset loaders for VoiceShield deepfake voice detection.
Supports:
- Direct reading from split manifests (train.csv, val.csv, test.csv, dataset_manifest.csv)
- Audio standardization to 16kHz mono float32 with VAD
- Fast RAM caching via preload_audio()
- Configurable data augmentations (noise, SpecAugment, channel distortion)
- Consistent label mapping: 0.0 = HUMAN / REAL, 1.0 = AI / FAKE
"""

import os
import io
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from torch.utils.data import Dataset

from ml.features import AudioFeatureExtractor, TARGET_SR, CHUNK_SAMPLES


BASE_DIR = Path(__file__).resolve().parent.parent


import scipy.signal

class AudioDataAugmenter:
    """
    Audio & feature augmentation for deepfake detection robustness.
    Simulates:
    - Additive Gaussian noise (varying SNR)
    - Microphone frequency response & telephony lowpass filtering (3.4kHz - 7kHz)
    - Room acoustic reflections / reverberation
    - Dynamic range compression & codec quantization
    - 2D SpecAugment (frequency and time masking across mel channels)
    """

    def __init__(
        self,
        apply_noise: bool = True,
        apply_masking: bool = True,
        noise_level: float = 0.005,
        freq_mask_param: int = 8,
        time_mask_param: int = 6,
        telephony_simulation: bool = True,
        reverb_simulation: bool = True
    ):
        self.apply_noise = apply_noise
        self.apply_masking = apply_masking
        self.noise_level = noise_level
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.telephony_simulation = telephony_simulation
        self.reverb_simulation = reverb_simulation

    def augment_raw_audio(self, audio: np.ndarray) -> np.ndarray:
        """Applies raw audio augmentations including mic frequency roll-off and room acoustics."""
        out = audio.copy()
        
        # 1. Additive Gaussian noise (varying background room noise)
        if self.apply_noise and np.random.rand() > 0.5:
            noise = np.random.randn(*out.shape) * np.random.uniform(0.001, self.noise_level)
            out = out + noise.astype(np.float32)

        # 2. Microphone / Telephony / Codec bandpass filtering (removes clean studio bias)
        if self.telephony_simulation and np.random.rand() > 0.4 and len(out) > 256:
            try:
                cutoff = np.random.choice([3400.0, 4000.0, 5000.0, 6500.0])
                sos = scipy.signal.butter(4, cutoff, btype='low', fs=16000, output='sos')
                out = scipy.signal.sosfilt(sos, out).astype(np.float32)
            except Exception:
                pass

        # 3. Room acoustic reflection / mild reverberation simulation
        if self.reverb_simulation and np.random.rand() > 0.5 and len(out) > 800:
            delay = np.random.randint(160, 480)  # 10ms - 30ms reflection
            decay = np.random.uniform(0.08, 0.22)
            reflected = out.copy()
            reflected[delay:] += decay * out[:-delay]
            out = reflected

        # 4. Gain fluctuation & peak protection
        gain = np.random.uniform(0.85, 1.15)
        out = np.clip(out * gain, -1.0, 1.0)

        return out.astype(np.float32)

    def augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        Applies 2D SpecAugment (Frequency & Time masking) across [3, n_mels, frames].
        """
        if not self.apply_masking or np.random.rand() > 0.5:
            return features

        augmented = features.copy()
        c, n_mels, time_frames = augmented.shape

        # Frequency Masking
        f_len = np.random.randint(1, min(self.freq_mask_param, max(2, n_mels // 2)))
        f_start = np.random.randint(0, max(1, n_mels - f_len))
        augmented[:, f_start:f_start + f_len, :] = 0.0

        # Time Masking
        t_len = np.random.randint(1, min(self.time_mask_param, max(2, time_frames // 2)))
        t_start = np.random.randint(0, max(1, time_frames - t_len))
        augmented[:, :, t_start:t_start + t_len] = 0.0

        return augmented


class VoiceShieldDataset(Dataset):
    """
    PyTorch Dataset loading samples from VoiceShield manifests or splits.
    Reads audio filepaths from CSV (train.csv, val.csv, test.csv, or dataset_manifest.csv),
    standardizes audio to 16kHz mono, and extracts 3-channel spectro-temporal features.
    """

    def __init__(
        self,
        manifest: Union[str, pd.DataFrame, Path],
        feature_extractor: Optional[AudioFeatureExtractor] = None,
        augment: bool = False,
        max_samples: Optional[int] = None
    ):
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.is_absolute():
                manifest_path = BASE_DIR / manifest_path
            self.df = pd.read_csv(manifest_path)
        else:
            self.df = manifest.copy()

        # Optional sample cap for fast testing
        if max_samples and max_samples < len(self.df):
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.df = self.df.reset_index(drop=True)
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.augmenter = AudioDataAugmenter() if augment else None

        self._audio_cache = {}
        self._feature_cache = {}
        self.skipped_count = 0

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_filepath(self, row: pd.Series) -> Path:
        """Resolves audio filepath to an absolute path."""
        for col in ['filepath', 'rel_filepath', 'audio_path']:
            if col in row and pd.notna(row[col]):
                p = Path(str(row[col]))
                if p.is_absolute() and p.exists():
                    return p
                cand = BASE_DIR / p
                if cand.exists():
                    return cand

        # Fallback: check filename in common data dirs
        if 'filename' in row and pd.notna(row['filename']):
            fn = str(row['filename'])
            for sub in ['train/real', 'train/fake', 'val/real', 'val/fake', 'test/real', 'test/fake', 'sample_demo', '']:
                cand = BASE_DIR / 'data' / sub / fn
                if cand.exists():
                    return cand

        # Return whatever was specified
        val = row.get('filepath', row.get('rel_filepath', ''))
        return Path(str(val))

    def preload_audio(self):
        """
        Preloads all audio samples sequentially into RAM.
        Eliminates disk I/O latency during training epochs.
        """
        if len(self.df) == 0:
            return

        print(f"[+] Preloading {len(self.df)} audio samples into memory...", flush=True)
        for idx in range(len(self.df)):
            if idx in self._audio_cache:
                continue
            row = self.df.iloc[idx]
            fp = self._resolve_filepath(row)
            try:
                if fp.exists():
                    audio, _ = self.feature_extractor.load_audio(str(fp), apply_vad=True)
                else:
                    audio = np.zeros(self.feature_extractor.chunk_samples, dtype=np.float32)
            except Exception as e:
                self.skipped_count += 1
                audio = np.zeros(self.feature_extractor.chunk_samples, dtype=np.float32)

            self._audio_cache[idx] = audio

        print(f"[✓] Successfully cached {len(self._audio_cache)} samples in RAM.", flush=True)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx in self._feature_cache and self.augmenter is None:
            return self._feature_cache[idx]

        row = self.df.iloc[idx]

        # Determine label: 0.0 = Real / Human, 1.0 = AI / Fake
        if 'label' in row:
            label = float(row['label'])
        elif 'is_tts' in row:
            label = float(row['is_tts'])
        else:
            label = 0.0

        if idx in self._audio_cache:
            audio = self._audio_cache[idx]
        else:
            fp = self._resolve_filepath(row)
            try:
                if fp.exists():
                    audio, _ = self.feature_extractor.load_audio(str(fp), apply_vad=True)
                else:
                    audio = np.zeros(self.feature_extractor.chunk_samples, dtype=np.float32)
            except Exception:
                self.skipped_count += 1
                audio = np.zeros(self.feature_extractor.chunk_samples, dtype=np.float32)

            self._audio_cache[idx] = audio

        if self.augmenter is not None:
            audio = self.augmenter.augment_raw_audio(audio)

        # Extract [3, n_mels, time_frames] features with random crop during training
        is_training = (self.augmenter is not None)
        features = self.feature_extractor.extract_features(audio, random_crop=is_training)

        if self.augmenter is not None:
            features = self.augmenter.augment_features(features)

        features_tensor = torch.from_numpy(features).float()
        label_tensor = torch.tensor([label], dtype=torch.float32)

        if self.augmenter is None:
            self._feature_cache[idx] = (features_tensor, label_tensor)

        return features_tensor, label_tensor


# Backward compatibility aliases
VoiceDataset = VoiceShieldDataset
IndicTTSDataset = VoiceShieldDataset
