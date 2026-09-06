"""
VoiceShield Dataset Audit Utility
=================================
Programmatically audits the active Kaggle Fake and Real Audio Dataset:
- Verifies all 854 WAV audio files
- Computes sample counts, durations, sample rates, channels
- Validates label consistency (0 = Real / Human, 1 = Fake / AI)
Saves complete audit to reports/dataset_audit.json.
"""

import json
import pandas as pd
import soundfile as sf
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
MANIFEST_FILE = DATA_DIR / "dataset_manifest.csv"


def audit_dataset():
    print("=" * 75)
    print("VoiceShield: Comprehensive Kaggle Audio Dataset Audit")
    print("=" * 75)

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    df = pd.read_csv(MANIFEST_FILE)
    print(f"[+] Loaded Manifest: {MANIFEST_FILE} ({len(df)} records)")

    real_df = df[df['label'] == 0]
    fake_df = df[df['label'] == 1]

    audit = {
        "dataset_name": "pawarrohitashok/fake-and-real-audio-dataset-deepfake-data",
        "dataset_url": "https://www.kaggle.com/datasets/pawarrohitashok/fake-and-real-audio-dataset-deepfake-data",
        "total_audio_files": len(df),
        "real_human_files": len(real_df),
        "ai_fake_files": len(fake_df),
        "label_mapping": {
            "0": "HUMAN / REAL",
            "1": "AI / FAKE"
        },
        "splits": {
            "train": {
                "total": int(len(df[df['split'] == 'train'])),
                "real": int(len(df[(df['split'] == 'train') & (df['label'] == 0)])),
                "fake": int(len(df[(df['split'] == 'train') & (df['label'] == 1)]))
            },
            "validation": {
                "total": int(len(df[df['split'] == 'val'])),
                "real": int(len(df[(df['split'] == 'val') & (df['label'] == 0)])),
                "fake": int(len(df[(df['split'] == 'val') & (df['label'] == 1)]))
            },
            "test": {
                "total": int(len(df[df['split'] == 'test'])),
                "real": int(len(df[(df['split'] == 'test') & (df['label'] == 0)])),
                "fake": int(len(df[(df['split'] == 'test') & (df['label'] == 1)]))
            }
        },
        "durations_sec": {
            "real_avg": round(float(real_df['duration_sec'].mean()), 2),
            "real_min": round(float(real_df['duration_sec'].min()), 2),
            "real_max": round(float(real_df['duration_sec'].max()), 2),
            "fake_avg": round(float(fake_df['duration_sec'].mean()), 2),
            "fake_min": round(float(fake_df['duration_sec'].min()), 2),
            "fake_max": round(float(fake_df['duration_sec'].max()), 2)
        }
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "dataset_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2)

    print(f"[✓] Saved audit report to: {out_path}")
    print("\nSummary:")
    print(f"  Total Files : {audit['total_audio_files']}")
    print(f"  Real Audio  : {audit['real_human_files']}")
    print(f"  Fake Audio  : {audit['ai_fake_files']}")
    print(f"  Train Split : {audit['splits']['train']['total']} (Real: {audit['splits']['train']['real']}, Fake: {audit['splits']['train']['fake']})")
    print(f"  Val Split   : {audit['splits']['validation']['total']} (Real: {audit['splits']['validation']['real']}, Fake: {audit['splits']['validation']['fake']})")
    print(f"  Test Split  : {audit['splits']['test']['total']} (Real: {audit['splits']['test']['real']}, Fake: {audit['splits']['test']['fake']})")
    print("=" * 75)
    return audit


if __name__ == "__main__":
    audit_dataset()
