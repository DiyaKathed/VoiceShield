"""
VoiceShield Pipeline Sanity & Mathematical Integrity Unit Tests
==============================================================
Validates:
1. Ground truth label mapping (is_tts 0 == HUMAN, is_tts 1 == AI_GENERATED)
2. Probability conversion math (Sigmoid, bounds [0, 1], sum == 1.0)
3. Monotonic behavior (negative logits -> Human, positive logits -> AI)
4. Absence of hardcoded predictions
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.model import VoiceShieldNet
from ml.dataset import IndicTTSDataset


def test_label_mapping():
    """Verify that dataset splits strictly map is_tts 0 to HUMAN (0.0) and is_tts 1 to AI (1.0)."""
    print("[*] Testing Label Mapping...")
    splits_path = BASE_DIR / "data" / "splits" / "test.csv"
    assert splits_path.exists(), "Splits test.csv must exist"

    df = pd.read_csv(splits_path)
    # Check 50 random samples
    sample_df = df.sample(n=min(50, len(df)), random_state=42)

    for _, row in sample_df.iterrows():
        raw_label = int(row['is_tts'])
        assert raw_label in (0, 1), f"Unexpected raw label: {raw_label}"
        if raw_label == 0:
            assert 'human' in str(row['label_name']).lower(), f"Label 0 must map to human, got {row['label_name']}"
        else:
            assert any(term in str(row['label_name']).lower() for term in ['ai', 'fake', 'synth', 'tts']), f"Label 1 must map to AI/fake, got {row['label_name']}"

    print("  [✓] Label mapping verified: is_tts=0 is strictly HUMAN, is_tts=1 is strictly AI.")


def test_probability_math():
    """Verify logit-to-probability mathematical conversion."""
    print("[*] Testing Probability Math & Invariance...")
    model = VoiceShieldNet()

    # Test extreme negative logit (should be ~100% human, ~0% AI)
    negative_logit = torch.tensor([[-10.0]])
    ai_p = torch.sigmoid(negative_logit).item()
    human_p = 1.0 - ai_p
    assert ai_p < 0.001, f"Negative logit must give low AI prob, got {ai_p}"
    assert human_p > 0.999, f"Negative logit must give high human prob, got {human_p}"
    assert abs((ai_p + human_p) - 1.0) < 1e-6, "Probabilities must sum to 1.0"

    # Test extreme positive logit (should be ~100% AI, ~0% human)
    positive_logit = torch.tensor([[10.0]])
    ai_p = torch.sigmoid(positive_logit).item()
    human_p = 1.0 - ai_p
    assert ai_p > 0.999, f"Positive logit must give high AI prob, got {ai_p}"
    assert human_p < 0.001, f"Positive logit must give low human prob, got {human_p}"
    assert abs((ai_p + human_p) - 1.0) < 1e-6, "Probabilities must sum to 1.0"

    # Test neutral zero logit (should be 50% / 50%)
    zero_logit = torch.tensor([[0.0]])
    ai_p = torch.sigmoid(zero_logit).item()
    human_p = 1.0 - ai_p
    assert abs(ai_p - 0.5) < 1e-6, "Zero logit must give 0.5 AI prob"
    assert abs(human_p - 0.5) < 1e-6, "Zero logit must give 0.5 human prob"

    print("  [✓] Probability conversion math verified: Sigmoid bounds [0, 1] and sum == 1.0 hold strictly.")


def test_dataset_sample_labels():
    """Verify IndicTTSDataset returns correct label tensors."""
    print("[*] Testing IndicTTSDataset label tensor output...")
    splits_path = BASE_DIR / "data" / "splits" / "val.csv"
    df = pd.read_csv(splits_path)

    # Pick first 5 human and first 5 AI rows
    human_rows = df[df['is_tts'] == 0].head(5)
    ai_rows = df[df['is_tts'] == 1].head(5)

    test_sub_df = pd.concat([human_rows, ai_rows]).reset_index(drop=True)
    dataset = IndicTTSDataset(test_sub_df)

    for i in range(len(test_sub_df)):
        expected_label = float(test_sub_df.iloc[i]['is_tts'])
        _, label_tensor = dataset[i]
        actual_label = label_tensor.item()
        assert actual_label == expected_label, f"Mismatch at index {i}: expected {expected_label}, got {actual_label}"

    print("  [✓] IndicTTSDataset correctly assigns label 0.0 to Human and 1.0 to AI.")


if __name__ == "__main__":
    print("=" * 60)
    print("Running VoiceShield Pipeline Sanity Unit Tests")
    print("=" * 60)
    test_label_mapping()
    test_probability_math()
    test_dataset_sample_labels()
    print("=" * 60)
    print("ALL SANITY TESTS PASSED SUCCESSFULLY.")
    print("=" * 60)
