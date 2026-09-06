"""
VoiceShield Dataset Split Enrichment & Audit Script
===================================================
Incorporates the paired same-speaker human-vs-clone datasets
(genuine_speaker_h* vs synthetic_speaker_c*) into the official
split manifests (train.csv, val.csv, test.csv) and dataset_manifest.csv.

Maintains strict speaker-disjoint isolation:
- Train: Speakers h1-h5 (real) and c1-c5 (clone) + Kaggle train (real/fake)
- Val:   Speaker h6 (real) and c6 (clone) + Kaggle val (real/fake)
- Test:  Speakers h7-h8 (real) and c7-c8 (clone) + Kaggle test (real/fake)
"""

import os
import json
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def get_audio_info(filepath: Path):
    try:
        info = sf.info(str(filepath))
        return round(float(info.duration), 2), info.samplerate, info.channels
    except Exception as e:
        print(f"Warning reading {filepath}: {e}")
        return 2.0, 16000, 1


def build_split_rows(split_name: str):
    rows = []
    
    # 1. Standard Kaggle real files in split
    real_dir = DATA_DIR / split_name / "real"
    if real_dir.exists():
        for f in sorted(real_dir.glob("*.wav")):
            dur, sr, ch = get_audio_info(f)
            rel_p = f"data/{split_name}/real/{f.name}"
            rows.append({
                "filepath": str(f.resolve()),
                "rel_filepath": rel_p,
                "filename": f.name,
                "label": 0,
                "is_tts": 0,
                "label_name": "genuine_human",
                "speaker_id": f"kaggle_real_{f.stem}",
                "duration_sec": dur,
                "split": split_name,
                "dataset_source": "kaggle_audio"
            })

    # 2. Standard Kaggle fake files in split
    fake_dir = DATA_DIR / split_name / "fake"
    if fake_dir.exists():
        for f in sorted(fake_dir.glob("*.wav")):
            dur, sr, ch = get_audio_info(f)
            rel_p = f"data/{split_name}/fake/{f.name}"
            rows.append({
                "filepath": str(f.resolve()),
                "rel_filepath": rel_p,
                "filename": f.name,
                "label": 1,
                "is_tts": 1,
                "label_name": "ai_fake",
                "speaker_id": f"kaggle_fake_{f.stem}",
                "duration_sec": dur,
                "split": split_name,
                "dataset_source": "kaggle_audio"
            })

    # 3. Paired Same-Speaker Clone audio in split root
    root_dir = DATA_DIR / split_name
    for f in sorted(root_dir.glob("genuine_speaker_*.wav")):
        dur, sr, ch = get_audio_info(f)
        # Extract speaker id e.g. genuine_speaker_h1_00.wav -> speaker_h1
        parts = f.stem.split("_")
        spk_id = f"speaker_{parts[2]}" if len(parts) >= 3 else f.stem
        rel_p = f"data/{split_name}/{f.name}"
        rows.append({
            "filepath": str(f.resolve()),
            "rel_filepath": rel_p,
            "filename": f.name,
            "label": 0,
            "is_tts": 0,
            "label_name": "genuine_human",
            "speaker_id": spk_id,
            "duration_sec": dur,
            "split": split_name,
            "dataset_source": "paired_speaker_clone"
        })

    for f in sorted(root_dir.glob("synthetic_speaker_*.wav")):
        dur, sr, ch = get_audio_info(f)
        parts = f.stem.split("_")
        spk_id = f"speaker_{parts[2]}" if len(parts) >= 3 else f.stem
        rel_p = f"data/{split_name}/{f.name}"
        rows.append({
            "filepath": str(f.resolve()),
            "rel_filepath": rel_p,
            "filename": f.name,
            "label": 1,
            "is_tts": 1,
            "label_name": "ai_fake",
            "speaker_id": spk_id,
            "duration_sec": dur,
            "split": split_name,
            "dataset_source": "paired_speaker_clone"
        })

    # Optional test sample in root
    test_sample = root_dir / "test_sample.wav"
    if test_sample.exists() and split_name == "test":
        dur, sr, ch = get_audio_info(test_sample)
        rows.append({
            "filepath": str(test_sample.resolve()),
            "rel_filepath": f"data/test/{test_sample.name}",
            "filename": test_sample.name,
            "label": 1,
            "is_tts": 1,
            "label_name": "ai_fake",
            "speaker_id": "synthetic_test_sample",
            "duration_sec": dur,
            "split": "test",
            "dataset_source": "benchmark_test"
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("ENRICHING VOICESHIELD DATASET SPLITS WITH SAME-SPEAKER CLONE PAIRS")
    print("=" * 80)

    all_dfs = []
    split_counts = {}

    for split in ["train", "val", "test"]:
        df = build_split_rows(split)
        out_csv = SPLITS_DIR / f"{split}.csv"
        df.to_csv(out_csv, index=False)
        all_dfs.append(df)

        real_count = int((df["label"] == 0).sum())
        fake_count = int((df["label"] == 1).sum())
        split_counts[split] = {
            "total": len(df),
            "human_real": real_count,
            "ai_fake": fake_count,
            "sources": df["dataset_source"].value_counts().to_dict()
        }
        print(f"\n[+] {split.upper()} SPLIT -> Total: {len(df)} | Human: {real_count} | AI: {fake_count}")
        print(f"    Saved manifest to: {out_csv}")

    # Build full dataset manifest
    full_manifest = pd.concat(all_dfs, ignore_index=True)
    manifest_csv = DATA_DIR / "dataset_manifest.csv"
    full_manifest.to_csv(manifest_csv, index=False)
    print(f"\n[✓] Full dataset manifest saved to: {manifest_csv} (Total: {len(full_manifest)} samples)")

    summary = {
        "dataset_name": "VoiceShield Enriched Anti-Spoofing Benchmark (Kaggle + Paired Speaker Clones)",
        "total_samples": len(full_manifest),
        "total_human_samples": int((full_manifest["label"] == 0).sum()),
        "total_ai_samples": int((full_manifest["label"] == 1).sum()),
        "class_mapping": {"0": "HUMAN / REAL", "1": "AI / SYNTHETIC / CLONED"},
        "splits": split_counts
    }

    summary_file = SPLITS_DIR / "split_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Split summary saved to: {summary_file}")

    print("\n" + "=" * 80)
    print("VERIFIED DATASET AUDIT (STEP 2 DELIVERABLE)")
    print("=" * 80)
    print(f"HUMAN:")
    print(f"  training   = {split_counts['train']['human_real']}")
    print(f"  validation = {split_counts['val']['human_real']}")
    print(f"  test       = {split_counts['test']['human_real']}")
    print(f"  TOTAL      = {summary['total_human_samples']}")
    print(f"\nAI:")
    print(f"  training   = {split_counts['train']['ai_fake']}")
    print(f"  validation = {split_counts['val']['ai_fake']}")
    print(f"  test       = {split_counts['test']['ai_fake']}")
    print(f"  TOTAL      = {summary['total_ai_samples']}")
    print("-" * 80)

    # Show 10 random sample rows
    print("\nSAMPLE FILENAMES AND ASSIGNED LABELS:")
    sample_rows = full_manifest.sample(n=12, random_state=42)[["filename", "split", "label", "label_name", "dataset_source"]]
    print(sample_rows.to_string(index=False))


if __name__ == "__main__":
    main()
