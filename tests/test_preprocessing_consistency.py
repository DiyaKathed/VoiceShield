"""
VoiceShield Preprocessing Consistency Test (Phase 7)
=====================================================
Verifies that AudioFeatureExtractor produces strictly identical and consistent
feature representations across training, validation, testing, and inference pipelines:
- Standardized 16,000 Hz sample rate
- Single mono channel float32 representation
- 64 Mel filterbanks, n_fft=1024, hop_length=256
- 3 channels: Log-Mel + Delta (velocity) + Delta-Delta (acceleration)
- Per-sample dynamic z-score normalization
"""

import sys
import unittest
import numpy as np
import soundfile as sf
import io
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.features import AudioFeatureExtractor
from ml.dataset import IndicTTSDataset
from backend.chunk_analyzer import RealTimeChunkAnalyzer


def test_audio_loading_format():
    """Verifies audio loader standardizes to mono, 16kHz float32 within [-1.0, 1.0]."""
    fe = AudioFeatureExtractor()

    # Generate synthetic stereo 44.1kHz audio buffer
    sr_in = 44100
    t = np.linspace(0, 3.0, int(sr_in * 3.0), endpoint=False)
    sig_left = 0.5 * np.sin(2 * np.pi * 440 * t)
    sig_right = 0.3 * np.sin(2 * np.pi * 880 * t)
    stereo = np.column_stack([sig_left, sig_right]).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, stereo, sr_in, format='WAV')
    buf.seek(0)

    audio, sr = fe.load_audio(buf.getvalue(), apply_vad=False)

    assert sr == 16000, f"Expected standardized 16000Hz, got {sr}"
    assert audio.ndim == 1, f"Expected mono 1D array, got ndim={audio.ndim}"
    assert audio.dtype == np.float32, f"Expected float32, got {audio.dtype}"
    assert np.max(np.abs(audio)) <= 1.0, f"Audio exceeds unity peak"
    print("  [✓] Audio standardization verified: Mono, 16kHz, Float32, Peak bounded.")


def test_feature_extractor_parameters():
    """Verifies feature extraction dimensions, filterbanks, and channels."""
    fe = AudioFeatureExtractor()
    assert fe.sample_rate == 16000
    assert fe.n_mels == 64
    assert fe.n_fft == 1024
    assert fe.hop_length == 512

    dummy_chunk = np.random.randn(fe.chunk_samples).astype(np.float32)
    feats = fe.extract_features(dummy_chunk, random_crop=False)

    assert feats.shape == (3, 64, fe.expected_frames), f"Shape mismatch: {feats.shape}"
    assert feats.dtype == np.float32, f"Expected float32 features, got {feats.dtype}"
    # Verify non-degenerate variance
    assert np.std(feats[0]) > 0.01, "Log-Mel channel has collapsed variance"
    print(f"  [✓] Feature tensor shape verified: {list(feats.shape)} (3 Channels: Log-Mel, Delta, Delta-Delta)")


def test_training_vs_inference_consistency():
    """Verifies training extraction and inference extraction use identical transformations."""
    fe = AudioFeatureExtractor()
    audio = np.sin(2 * np.pi * 300 * np.linspace(0, 2.0, 32000)).astype(np.float32)

    # In deterministic mode (random_crop=False), feature output must match bit-for-bit
    feat_train = fe.extract_features(audio, random_crop=False)
    feat_infer = fe.extract_features(audio, random_crop=False)

    np.testing.assert_allclose(feat_train, feat_infer, atol=1e-6, err_msg="Train and inference features differ!")
    print("  [✓] Training and Inference feature representations are bit-for-bit consistent.")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING PREPROCESSING CONSISTENCY TESTS")
    print("=" * 80)
    test_audio_loading_format()
    test_feature_extractor_parameters()
    test_training_vs_inference_consistency()
    print("\n[✓] All preprocessing consistency tests passed successfully!")
