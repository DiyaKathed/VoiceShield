"""
VoiceShield Kaggle Dataset Preparation and Splitting Script
===========================================================
Organizes the Kaggle Fake and Real Audio Dataset into VoiceShield structure:
- data/train/{real,fake}
- data/val/{real,fake}
- data/test/{real,fake}
- data/sample_demo/
- data/dataset_manifest.csv
- data/splits/{train.csv, val.csv, test.csv, split_summary.json}
Computes authentic audio statistics across all samples.
"""

import os
import sys
import json
import shutil
import random
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "kaggle_raw"
SPLITS_DIR = DATA_DIR / "splits"
SAMPLE_DEMO_DIR = DATA_DIR / "sample_demo"


def find_audio_dirs():
    """Locates Fake_Audios and Real_Audios directories inside RAW_DIR."""
    fake_dirs = list(RAW_DIR.rglob("Fake_Audios"))
    real_dirs = list(RAW_DIR.rglob("Real_Audios"))

    if not fake_dirs or not real_dirs:
        raise FileNotFoundError(f"Could not locate Fake_Audios and Real_Audios in {RAW_DIR}")

    return fake_dirs[0], real_dirs[0]


def collect_audio_stats(file_list, label_name):
    """Gathers audio statistics using soundfile."""
    stats = {
        "count": len(file_list),
        "formats": set(),
        "sample_rates": set(),
        "channels": set(),
        "durations": []
    }

    for f in file_list:
        try:
            info = sf.info(f)
            stats["formats"].add(info.format)
            stats["sample_rates"].add(info.samplerate)
            stats["channels"].add(info.channels)
            stats["durations"].append(info.duration)
        except Exception as e:
            print(f"Warning reading {f}: {e}")

    durations = np.array(stats["durations"])
    return {
        "label": label_name,
        "count": stats["count"],
        "formats": sorted(list(stats["formats"])),
        "sample_rates": sorted(list(stats["sample_rates"])),
        "channels": sorted(list(stats["channels"])),
        "avg_duration_sec": float(np.mean(durations)) if len(durations) else 0.0,
        "min_duration_sec": float(np.min(durations)) if len(durations) else 0.0,
        "max_duration_sec": float(np.max(durations)) if len(durations) else 0.0,
        "total_duration_sec": float(np.sum(durations)) if len(durations) else 0.0
    }


def prepare_dataset(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    fake_dir, real_dir = find_audio_dirs()

    fake_files = sorted(list(fake_dir.glob("*.wav")))
    real_files = sorted(list(real_dir.glob("*.wav")))

    print("=" * 80)
    print("VOICESHIELD: KAGGLE DATASET AUDIT & STATISTICS")
    print("=" * 80)
    print(f"[+] Total Fake audio files found: {len(fake_files)}")
    print(f"[+] Total Real audio files found: {len(real_files)}")
    print(f"[+] Grand total audio files: {len(fake_files) + len(real_files)}")

    real_stats = collect_audio_stats(real_files, "HUMAN / REAL")
    fake_stats = collect_audio_stats(fake_files, "AI / FAKE")

    print("\n--- REAL AUDIO STATISTICS ---")
    print(f"  Count: {real_stats['count']}")
    print(f"  File Formats: {real_stats['formats']}")
    print(f"  Sample Rates: {real_stats['sample_rates']} Hz")
    print(f"  Channels: {real_stats['channels']} (1=mono, 2=stereo)")
    print(f"  Average Duration: {real_stats['avg_duration_sec']:.2f} s")
    print(f"  Min Duration: {real_stats['min_duration_sec']:.2f} s")
    print(f"  Max Duration: {real_stats['max_duration_sec']:.2f} s")
    print(f"  Total Duration: {real_stats['total_duration_sec']/60:.1f} mins")

    print("\n--- FAKE AUDIO STATISTICS ---")
    print(f"  Count: {fake_stats['count']}")
    print(f"  File Formats: {fake_stats['formats']}")
    print(f"  Sample Rates: {fake_stats['sample_rates']} Hz")
    print(f"  Channels: {fake_stats['channels']} (1=mono, 2=stereo)")
    print(f"  Average Duration: {fake_stats['avg_duration_sec']:.2f} s")
    print(f"  Min Duration: {fake_stats['min_duration_sec']:.2f} s")
    print(f"  Max Duration: {fake_stats['max_duration_sec']:.2f} s")
    print(f"  Total Duration: {fake_stats['total_duration_sec']/60:.1f} mins")

    # Clean existing train, val, test target directories
    for split in ["train", "val", "test"]:
        for cat in ["real", "fake"]:
            d = DATA_DIR / split / cat
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DEMO_DIR.mkdir(parents=True, exist_ok=True)

    # Stratified 70% / 15% / 15% Split
    # Shuffle with fixed seed
    shuffled_real = real_files.copy()
    random.shuffle(shuffled_real)

    shuffled_fake = fake_files.copy()
    random.shuffle(shuffled_fake)

    # 423 real: 296 train, 64 val, 63 test
    # 431 fake: 301 train, 65 val, 65 test
    n_real_tr = int(round(len(shuffled_real) * 0.70))
    n_real_val = int(round(len(shuffled_real) * 0.15))
    real_train = shuffled_real[:n_real_tr]
    real_val = shuffled_real[n_real_tr:n_real_tr + n_real_val]
    real_test = shuffled_real[n_real_tr + n_real_val:]

    n_fake_tr = int(round(len(shuffled_fake) * 0.70))
    n_fake_val = int(round(len(shuffled_fake) * 0.15))
    fake_train = shuffled_fake[:n_fake_tr]
    fake_val = shuffled_fake[n_fake_tr:n_fake_tr + n_fake_val]
    fake_test = shuffled_fake[n_fake_tr + n_fake_val:]

    print("\n--- SPLIT DISTRIBUTION ---")
    print(f"  Train: Real={len(real_train)}, Fake={len(fake_train)} -> Total={len(real_train)+len(fake_train)}")
    print(f"  Val:   Real={len(real_val)}, Fake={len(fake_val)} -> Total={len(real_val)+len(fake_val)}")
    print(f"  Test:  Real={len(real_test)}, Fake={len(fake_test)} -> Total={len(real_test)+len(fake_test)}")

    records = []

    def process_files(file_list, split_name, label, label_name, subcat):
        for src_path in file_list:
            dest_name = src_path.name
            dest_path = DATA_DIR / split_name / subcat / dest_name
            shutil.copyfile(src_path, dest_path)

            try:
                dur = round(sf.info(dest_path).duration, 2)
            except Exception:
                dur = 0.0

            speaker_id = f"{subcat}_{src_path.stem}"
            records.append({
                "filepath": str(dest_path),
                "rel_filepath": f"data/{split_name}/{subcat}/{dest_name}",
                "filename": dest_name,
                "label": label,
                "is_tts": label,
                "label_name": label_name,
                "speaker_id": speaker_id,
                "duration_sec": dur,
                "split": split_name
            })

    process_files(real_train, "train", 0, "genuine_human", "real")
    process_files(fake_train, "train", 1, "ai_fake", "fake")
    process_files(real_val, "val", 0, "genuine_human", "real")
    process_files(fake_val, "val", 1, "ai_fake", "fake")
    process_files(real_test, "test", 0, "genuine_human", "real")
    process_files(fake_test, "test", 1, "ai_fake", "fake")

    df_manifest = pd.DataFrame(records)
    manifest_csv = DATA_DIR / "dataset_manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"[✓] Saved manifest to: {manifest_csv} ({len(df_manifest)} entries)")

    # Save split CSVs in data/splits/
    for split_name in ["train", "val", "test"]:
        sub_df = df_manifest[df_manifest["split"] == split_name].reset_index(drop=True)
        sub_csv = SPLITS_DIR / f"{split_name}.csv"
        sub_df.to_csv(sub_csv, index=False)
        print(f"[✓] Saved split CSV to: {sub_csv} ({len(sub_df)} entries)")

    # Create sample demo files
    demo_samples = [
        (real_test[0], SAMPLE_DEMO_DIR / "sample_real_01.wav", "Real Voice Sample 1"),
        (fake_test[0], SAMPLE_DEMO_DIR / "sample_fake_01.wav", "AI Cloned Voice Sample 1"),
        (real_test[1], SAMPLE_DEMO_DIR / "sample_real_02.wav", "Real Voice Sample 2"),
        (fake_test[1], SAMPLE_DEMO_DIR / "sample_fake_02.wav", "AI Cloned Voice Sample 2"),
    ]
    for src, dst, desc in demo_samples:
        shutil.copyfile(src, dst)
        print(f"[✓] Created demo sample: {dst.name} ({desc})")

    # Save split summary
    summary_data = {
        "dataset_name": "pawarrohitashok/fake-and-real-audio-dataset-deepfake-data",
        "dataset_url": "https://www.kaggle.com/datasets/pawarrohitashok/fake-and-real-audio-dataset-deepfake-data",
        "total_samples": len(df_manifest),
        "real_count": len(real_files),
        "fake_count": len(fake_files),
        "audio_statistics": {
            "real": real_stats,
            "fake": fake_stats
        },
        "splits": {
            "train": {
                "count": len(real_train) + len(fake_train),
                "real": len(real_train),
                "fake": len(fake_train)
            },
            "validation": {
                "count": len(real_val) + len(fake_val),
                "real": len(real_val),
                "fake": len(fake_val)
            },
            "test": {
                "count": len(real_test) + len(fake_test),
                "real": len(real_test),
                "fake": len(fake_test)
            }
        },
        "speaker_disjoint_possible": False,
        "speaker_disjoint_limitation": "Filenames in the Kaggle dataset are sequentially indexed numbers (fake_001.wav - fake_431.wav, real_001.wav - real_423.wav) without speaker IDs, actor metadata, or transcripts. Stratified random splitting with fixed seed=42 was applied."
    }

    summary_json = SPLITS_DIR / "split_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[✓] Saved split summary to: {summary_json}")

    return summary_data


if __name__ == "__main__":
    prepare_dataset()
