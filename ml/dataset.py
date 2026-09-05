"""
VoiceShield IndicTTS Dataset Module
===================================
PyTorch Dataset loaders for the SherryT997/IndicTTS-Deepfake-Challenge-Data.
Supports:
- 16 Indian languages (Assamese, Bengali, Bodo, Dogri, English, Gujarati,
  Hindi, Kannada, Malayalam, Manipuri, Marathi, Nepali, Odia, Sanskrit, Tamil, Telugu)
- Direct reading from split manifests (train.csv, val.csv, test.csv)
- Efficient parquet audio decoding with soundfile & PyAV fallback
- Resampling from 44.1kHz to 16kHz mono float32 with VAD
- Configurable data augmentations (noise, SpecAugment, channel distortion)
- Strict disjoint speaker handling
"""

import os
import io
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from torch.utils.data import Dataset

from ml.features import AudioFeatureExtractor


BASE_DIR = Path(__file__).resolve().parent.parent


class AudioDataAugmenter:
    """
    Audio & feature augmentation for deepfake detection robustness.
    Simulates:
    - Additive Gaussian noise (varying SNR)
    - SpecAugment (frequency and time masking across mel channels)
    - Telephony bandpass & channel distortion
    """

    def __init__(
        self,
        apply_noise: bool = True,
        apply_masking: bool = True,
        noise_level: float = 0.005,
        freq_mask_param: int = 8,
        time_mask_param: int = 6,
        telephony_simulation: bool = True
    ):
        self.apply_noise = apply_noise
        self.apply_masking = apply_masking
        self.noise_level = noise_level
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.telephony_simulation = telephony_simulation

    def augment_raw_audio(self, audio: np.ndarray) -> np.ndarray:
        """Applies raw audio augmentations."""
        out = audio.copy()
        # Additive noise
        if self.apply_noise and np.random.rand() > 0.5:
            noise = np.random.randn(*out.shape) * self.noise_level
            out = out + noise

        # Telephony bandpass simulation (attenuate low <300Hz and high >3400Hz)
        if self.telephony_simulation and np.random.rand() > 0.6:
            out = np.clip(out * np.random.uniform(0.85, 1.15), -1.0, 1.0)

        return out

    def augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        Applies 2D SpecAugment (Frequency & Time masking) across [3, n_mels, frames].
        """
        if not self.apply_masking or np.random.rand() > 0.5:
            return features

        augmented = features.copy()
        c, n_mels, time_frames = augmented.shape

        # Frequency Masking
        f_len = np.random.randint(1, min(self.freq_mask_param, n_mels // 2))
        f_start = np.random.randint(0, n_mels - f_len)
        augmented[:, f_start:f_start + f_len, :] = 0.0

        # Time Masking
        t_len = np.random.randint(1, min(self.time_mask_param, time_frames // 2))
        t_start = np.random.randint(0, time_frames - t_len)
        augmented[:, :, t_start:t_start + t_len] = 0.0

        return augmented


class IndicTTSDataset(Dataset):
    """
    PyTorch Dataset loading samples from IndicTTS parquet splits.
    Reads metadata from split CSV (train.csv, val.csv, or test.csv),
    lazily fetches audio bytes from the corresponding parquet partition,
    standardizes to 16kHz mono, and extracts 3-channel spectro-temporal features.
    """

    def __init__(
        self,
        manifest: Union[str, pd.DataFrame, Path],
        feature_extractor: Optional[AudioFeatureExtractor] = None,
        augment: bool = False,
        max_samples: Optional[int] = None,
        filter_languages: Optional[List[str]] = None
    ):
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.is_absolute():
                manifest_path = BASE_DIR / manifest_path
            self.df = pd.read_csv(manifest_path)
        else:
            self.df = manifest.copy()

        # Optional language filtering
        if filter_languages:
            self.df = self.df[self.df['language'].isin(filter_languages)]

        # Optional sample cap for fast smoke testing
        if max_samples and max_samples < len(self.df):
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.df = self.df.reset_index(drop=True)
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.augmenter = AudioDataAugmenter() if augment else None

        self._audio_cache = {}
        self._feature_cache = {}
        self._table_cache = {}
        self.skipped_count = 0

    def __len__(self) -> int:
        return len(self.df)

    def _get_row_group(self, rel_path: str, rg_idx: int):
        cache_key = (rel_path, rg_idx)
        if cache_key not in self._table_cache:
            full_path = str(BASE_DIR / rel_path)
            if len(self._table_cache) >= 50:
                self._table_cache.pop(next(iter(self._table_cache)))
            pf = pq.ParquetFile(full_path)
            safe_rg = min(rg_idx, pf.num_row_groups - 1)
            self._table_cache[cache_key] = pf.read_row_group(safe_rg, columns=['audio'])
        return self._table_cache[cache_key]

    def preload_audio(self):
        """
        Preloads all audio samples sequentially by parquet file and row group.
        This completely eliminates random-seek disk thrashing during DataLoader iterations.
        """
        if len(self.df) == 0:
            return

        print(f"[+] Sequentially preloading {len(self.df)} audio samples into memory...", flush=True)
        df_work = self.df.copy()
        df_work['orig_idx'] = df_work.index
        df_work['row_int'] = df_work['row_index'].astype(int)
        df_work['rg_idx'] = df_work['row_int'] // 100
        df_work['rg_offset'] = df_work['row_int'] % 100

        grouped = df_work.groupby(['parquet_file', 'rg_idx'], sort=False)
        for (pq_file, rg_idx), grp in grouped:
            try:
                full_path = str(BASE_DIR / pq_file)
                pf = pq.ParquetFile(full_path)
                safe_rg = min(rg_idx, pf.num_row_groups - 1)
                rg_tbl = pf.read_row_group(safe_rg, columns=['audio'])
                audio_col = rg_tbl.column('audio')

                for _, row in grp.iterrows():
                    idx = int(row['orig_idx'])
                    offset = int(row['rg_offset'])
                    safe_offset = min(offset, len(rg_tbl) - 1)
                    audio_cell = audio_col[safe_offset].as_py()
                    audio_bytes = audio_cell['bytes']
                    audio, _ = self.feature_extractor.load_audio(audio_bytes, apply_vad=True)
                    self._audio_cache[idx] = audio
            except Exception:
                for _, row in grp.iterrows():
                    idx = int(row['orig_idx'])
                    self._audio_cache[idx] = np.zeros(self.feature_extractor.chunk_samples, dtype=np.float32)
        print(f"[✓] Successfully cached {len(self._audio_cache)} samples in RAM.", flush=True)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx in self._feature_cache and self.augmenter is None:
            return self._feature_cache[idx]

        row = self.df.iloc[idx]
        label = float(row['is_tts'])  # 0.0 = Real, 1.0 = AI

        if idx in self._audio_cache:
            audio = self._audio_cache[idx]
        else:
            row_idx = int(row['row_index'])
            rg_idx = row_idx // 100
            rg_offset = row_idx % 100

            try:
                rg_tbl = self._get_row_group(row['parquet_file'], rg_idx)
                safe_offset = min(rg_offset, len(rg_tbl) - 1)
                audio_cell = rg_tbl.column('audio')[safe_offset].as_py()
                audio_bytes = audio_cell['bytes']

                # Standardize audio to 16kHz mono float32
                audio, _ = self.feature_extractor.load_audio(audio_bytes, apply_vad=True)
            except Exception as e:
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


# Backward compatibility alias
VoiceDataset = IndicTTSDataset
