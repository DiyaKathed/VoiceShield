"""
VoiceShield Label Mapping Sanity Test (Phase 3)
================================================
Validates that label 0 is strictly HUMAN and label 1 is strictly AI_GENERATED.
Fails immediately if any component in the pipeline reverses the polarity.
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
SPLITS_DIR = BASE_DIR / "data" / "splits"


def test_20_random_samples_mapping():
    """Prints 20 random samples and verifies label semantics."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    sample_20 = full_df.sample(n=20, random_state=42).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("VERIFYING 20 RANDOM SAMPLES LABEL MAPPING")
    print("=" * 90)
    print(f"{'Sample Filename':<25} | {'Label Name':<18} | {'Raw Label':<10} | {'Interpreted Class':<18}")
    print("-" * 90)

    for _, row in sample_20.iterrows():
        raw_val = int(row['is_tts'])
        assert raw_val in (0, 1), f"Invalid raw label: {raw_val}"
        
        interpreted = "HUMAN" if raw_val == 0 else "AI_GENERATED"
        fname = str(row.get('filename', row.get('id', 'sample')))
        lbl_name = str(row.get('label_name', ''))
        print(f"{fname:<25} | {lbl_name:<18} | {raw_val:<10} | {interpreted:<18}")

        if raw_val == 0:
            assert interpreted == "HUMAN", "Label 0 MUST correspond to HUMAN"
        elif raw_val == 1:
            assert interpreted == "AI_GENERATED", "Label 1 MUST correspond to AI_GENERATED"


def test_label_not_reversed():
    """Explicitly tests that Human is 0 and AI is 1 in dataset loading."""
    from ml.dataset import IndicTTSDataset
    from ml.features import AudioFeatureExtractor

    df = pd.DataFrame([
        {"id": "human_sample", "language": "Hindi", "is_tts": 0, "speaker_id": "spk_1", "parquet_file": "dummy", "row_index": 0},
        {"id": "ai_sample", "language": "Hindi", "is_tts": 1, "speaker_id": "spk_2", "parquet_file": "dummy", "row_index": 1},
    ])
    
    fe = AudioFeatureExtractor()
    ds = IndicTTSDataset(df, feature_extractor=fe, augment=False)
    # Manually populate audio cache to bypass parquet loading
    dummy_audio = np.zeros(fe.chunk_samples, dtype=np.float32)
    ds._audio_cache[0] = dummy_audio
    ds._audio_cache[1] = dummy_audio

    feat_human, label_human = ds[0]
    feat_ai, label_ai = ds[1]

    assert label_human.item() == 0.0, f"Expected HUMAN label 0.0, got {label_human.item()}"
    assert label_ai.item() == 1.0, f"Expected AI label 1.0, got {label_ai.item()}"
    assert label_human.item() != label_ai.item(), "Human and AI labels cannot be identical"


if __name__ == "__main__":
    test_20_random_samples_mapping()
    test_label_not_reversed()
    print("\n[✓] Label mapping sanity test passed successfully!")
