"""
VoiceShield IndicTTS Dataset Audit Script
=========================================
Audits the downloaded SherryT997/IndicTTS-Deepfake-Challenge-Data dataset:
- Total sample counts and label distribution (Real vs AI)
- Language breakdown across all 16 Indic languages
- Sample rates and duration statistics
- Verifies integrity of audio bytes and parquet partitions
- Saves comprehensive report to reports/dataset_audit.json
"""

import os
import io
import glob
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataindictts" / "data"
REPORTS_DIR = BASE_DIR / "reports"


def run_audit(sample_audio_check_count: int = 500):
    print("=" * 65)
    print("VoiceShield: Auditing SherryT997/IndicTTS-Deepfake-Challenge-Data")
    print("=" * 65)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_files = sorted(glob.glob(str(DATA_DIR / "train-*.parquet")))
    test_files = sorted(glob.glob(str(DATA_DIR / "test-*.parquet")))

    print(f"[+] Discovered {len(train_files)} train parquet partitions")
    print(f"[+] Discovered {len(test_files)} test parquet partitions")

    if not train_files:
        raise FileNotFoundError(f"No train parquet files found in {DATA_DIR}")

    # Read all metadata columns without decompressing all audio blobs
    print("\n[+] Reading metadata from all 35 train partitions...")
    dfs = []
    for f in train_files:
        tbl = pq.read_table(f, columns=['id', 'language', 'is_tts', 'text'])
        dfs.append(tbl.to_pandas())

    train_meta_df = pd.concat(dfs, ignore_index=True)
    total_train = len(train_meta_df)

    test_meta_dfs = []
    for f in test_files:
        tbl = pq.read_table(f, columns=['id', 'language', 'is_tts', 'text'])
        test_meta_dfs.append(tbl.to_pandas())
    test_meta_df = pd.concat(test_meta_dfs, ignore_index=True)
    total_test = len(test_meta_df)

    # Label statistics
    label_counts = train_meta_df['is_tts'].value_counts().to_dict()
    real_count = int(label_counts.get(0, 0))
    ai_count = int(label_counts.get(1, 0))

    # Language statistics
    lang_counts = train_meta_df['language'].value_counts().to_dict()
    crosstab = pd.crosstab(train_meta_df['language'], train_meta_df['is_tts']).to_dict(orient='index')

    # Audio integrity and duration sampling
    print(f"\n[+] Inspecting audio integrity & durations across {sample_audio_check_count} representative samples...")
    durations = []
    sample_rates = set()
    corrupt_count = 0

    # Sample evenly across partitions
    step = max(1, len(train_files) // 10)
    sampled_files = train_files[::step]

    samples_per_file = max(5, sample_audio_check_count // len(sampled_files))
    for f in sampled_files:
        tbl = pq.read_table(f, columns=['audio', 'is_tts', 'language'])
        rows = tbl.slice(0, min(samples_per_file, tbl.num_rows)).to_pylist()
        for r in rows:
            try:
                audio_bytes = r['audio']['bytes']
                data, sr = sf.read(io.BytesIO(audio_bytes))
                sample_rates.add(sr)
                durations.append(len(data) / sr)
            except Exception as e:
                corrupt_count += 1

    dur_arr = np.array(durations) if durations else np.array([0.0])

    report = {
        "dataset_name": "SherryT997/IndicTTS-Deepfake-Challenge-Data",
        "dataset_location": str(DATA_DIR),
        "total_train_samples": total_train,
        "total_official_test_samples": total_test,
        "label_distribution": {
            "human_real_0": real_count,
            "ai_tts_1": ai_count,
            "real_percentage": round(real_count / total_train * 100, 2),
            "ai_percentage": round(ai_count / total_train * 100, 2)
        },
        "total_languages": len(lang_counts),
        "languages_list": sorted(list(lang_counts.keys())),
        "samples_per_language": lang_counts,
        "language_by_label": crosstab,
        "audio_properties": {
            "native_sample_rates": list(sample_rates),
            "target_model_sample_rate": 16000,
            "duration_stats_sec": {
                "min": round(float(np.min(dur_arr)), 2),
                "max": round(float(np.max(dur_arr)), 2),
                "mean": round(float(np.mean(dur_arr)), 2),
                "std": round(float(np.std(dur_arr)), 2)
            },
            "audited_samples_count": len(durations),
            "corrupt_samples_found": corrupt_count
        }
    }

    report_file = REPORTS_DIR / "dataset_audit.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "-" * 65)
    print("                    DATASET AUDIT REPORT")
    print("-" * 65)
    print(f"  Dataset               : {report['dataset_name']}")
    print(f"  Total Labeled Samples : {total_train:,}")
    print(f"  Human / Real (0)      : {real_count:,} ({report['label_distribution']['real_percentage']}%)")
    print(f"  AI / TTS (1)          : {ai_count:,} ({report['label_distribution']['ai_percentage']}%)")
    print(f"  Total Languages       : {len(lang_counts)}")
    print(f"  Audio Duration Range  : {report['audio_properties']['duration_stats_sec']['min']}s - {report['audio_properties']['duration_stats_sec']['max']}s (Mean: {report['audio_properties']['duration_stats_sec']['mean']}s)")
    print(f"  Native Sample Rates   : {report['audio_properties']['native_sample_rates']}")
    print(f"  Corrupt Samples       : {corrupt_count}")
    print("-" * 65)
    print("  Language Breakdown (Real vs AI):")
    for lang, counts in crosstab.items():
        print(f"    {lang:<15} -> Real (0): {counts.get(0, 0):^6} | AI (1): {counts.get(1, 0):^6} | Total: {counts.get(0, 0) + counts.get(1, 0)}")
    print("-" * 65)
    print(f"[✓] Saved complete audit report to: {report_file}")

    return report


if __name__ == "__main__":
    run_audit()
