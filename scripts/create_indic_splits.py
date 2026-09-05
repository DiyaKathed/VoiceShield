"""
VoiceShield IndicTTS Data Splitting & Demo Asset Extraction
===========================================================
1. Indexes all 35 train parquet partitions of SherryT997/IndicTTS-Deepfake-Challenge-Data.
2. Performs strict speaker-level disjoint splitting (zero speaker/data leakage):
   - Train: 70%
   - Validation: 15%
   - Test: 15%
3. Verifies disjointness:
   - Train ∩ Val = ∅
   - Train ∩ Test = ∅
   - Val ∩ Test = ∅
4. Saves manifests to:
   - data/splits/train.csv
   - data/splits/val.csv
   - data/splits/test.csv
5. Extracts representative authentic Indian-language audio samples into data/sample_demo/indic/
   for 1-click testing in the VoiceShield web dashboard.
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
SPLITS_DIR = BASE_DIR / "data" / "splits"
INDIC_DEMO_DIR = BASE_DIR / "data" / "sample_demo" / "indic"


def main():
    print("=" * 65)
    print("VoiceShield: Creating Disjoint IndicTTS Splits")
    print("=" * 65)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    INDIC_DEMO_DIR.mkdir(parents=True, exist_ok=True)

    train_files = sorted(glob.glob(str(DATA_DIR / "train-*.parquet")))
    if not train_files:
        raise FileNotFoundError(f"No train parquet files found in {DATA_DIR}")

    print(f"[+] Indexing {len(train_files)} parquet partitions...")
    index_records = []

    for file_idx, fpath in enumerate(train_files):
        tbl = pq.read_table(fpath, columns=['id', 'language', 'is_tts'])
        df = tbl.to_pandas()
        for row_idx, r in enumerate(df.itertuples(index=False)):
            spk_parts = r.id.split('_')
            # Extract speaker group: {LANG}_{GENDER}_{CATEGORY}
            speaker_id = '_'.join(spk_parts[:3]) if len(spk_parts) >= 3 else r.id
            index_records.append({
                "id": r.id,
                "language": r.language,
                "is_tts": int(r.is_tts),
                "label_name": "ai_tts" if r.is_tts == 1 else "genuine_human",
                "speaker_id": speaker_id,
                "parquet_file": os.path.relpath(fpath, BASE_DIR),
                "row_index": row_idx
            })

    full_df = pd.DataFrame(index_records)
    print(f"[+] Successfully indexed {len(full_df):,} samples across {full_df['language'].nunique()} languages.")

    # Stratified disjoint speaker splitting per language
    np.random.seed(42)
    train_indices = []
    val_indices = []
    test_indices = []

    for lang, lang_df in full_df.groupby('language'):
        speakers = lang_df['speaker_id'].unique()
        np.random.shuffle(speakers)

        n_spk = len(speakers)
        n_test_spk = max(1, int(round(n_spk * 0.15)))
        n_val_spk = max(1, int(round(n_spk * 0.15)))

        test_spks = set(speakers[:n_test_spk])
        val_spks = set(speakers[n_test_spk:n_test_spk + n_val_spk])
        train_spks = set(speakers[n_test_spk + n_val_spk:])

        # Fallback if group count is very small: split at utterance level within language
        if len(train_spks) == 0:
            shuffled_idx = lang_df.index.tolist()
            np.random.shuffle(shuffled_idx)
            n_t = max(1, int(len(shuffled_idx) * 0.15))
            n_v = max(1, int(len(shuffled_idx) * 0.15))
            test_indices.extend(shuffled_idx[:n_t])
            val_indices.extend(shuffled_idx[n_t:n_t + n_v])
            train_indices.extend(shuffled_idx[n_t + n_v:])
        else:
            test_indices.extend(lang_df[lang_df['speaker_id'].isin(test_spks)].index)
            val_indices.extend(lang_df[lang_df['speaker_id'].isin(val_spks)].index)
            train_indices.extend(lang_df[lang_df['speaker_id'].isin(train_spks)].index)

    train_df = full_df.loc[train_indices].reset_index(drop=True)
    val_df = full_df.loc[val_indices].reset_index(drop=True)
    test_df = full_df.loc[test_indices].reset_index(drop=True)

    # Verify disjointness
    train_speakers = set(train_df['speaker_id'].unique())
    val_speakers = set(val_df['speaker_id'].unique())
    test_speakers = set(test_df['speaker_id'].unique())

    overlap_tv = train_speakers.intersection(val_speakers)
    overlap_tt = train_speakers.intersection(test_speakers)
    overlap_vt = val_speakers.intersection(test_speakers)

    print("\n[+] Speaker Disjointness Verification:")
    print(f"  Train ∩ Val Overlap  : {len(overlap_tv)} speakers (Zero leakage: {len(overlap_tv) == 0})")
    print(f"  Train ∩ Test Overlap : {len(overlap_tt)} speakers (Zero leakage: {len(overlap_tt) == 0})")
    print(f"  Val ∩ Test Overlap   : {len(overlap_vt)} speakers (Zero leakage: {len(overlap_vt) == 0})")

    # Save split manifests
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    test_csv = SPLITS_DIR / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    split_stats = {
        "dataset": "SherryT997/IndicTTS-Deepfake-Challenge-Data",
        "total_samples": len(full_df),
        "splits": {
            "train": {
                "count": len(train_df),
                "percentage": round(len(train_df) / len(full_df) * 100, 2),
                "real_count": int((train_df['is_tts'] == 0).sum()),
                "ai_count": int((train_df['is_tts'] == 1).sum()),
                "speakers": len(train_speakers)
            },
            "validation": {
                "count": len(val_df),
                "percentage": round(len(val_df) / len(full_df) * 100, 2),
                "real_count": int((val_df['is_tts'] == 0).sum()),
                "ai_count": int((val_df['is_tts'] == 1).sum()),
                "speakers": len(val_speakers)
            },
            "test": {
                "count": len(test_df),
                "percentage": round(len(test_df) / len(full_df) * 100, 2),
                "real_count": int((test_df['is_tts'] == 0).sum()),
                "ai_count": int((test_df['is_tts'] == 1).sum()),
                "speakers": len(test_speakers)
            }
        },
        "languages": sorted(full_df['language'].unique().tolist())
    }

    with open(SPLITS_DIR / "split_summary.json", "w") as f:
        json.dump(split_stats, f, indent=2)

    print("\n" + "-" * 65)
    print("                    DATASET SPLIT SUMMARY")
    print("-" * 65)
    print(f"  Train Set      : {len(train_df):,} samples ({split_stats['splits']['train']['percentage']}%) | Real: {split_stats['splits']['train']['real_count']} | AI: {split_stats['splits']['train']['ai_count']}")
    print(f"  Validation Set : {len(val_df):,} samples ({split_stats['splits']['validation']['percentage']}%) | Real: {split_stats['splits']['validation']['real_count']} | AI: {split_stats['splits']['validation']['ai_count']}")
    print(f"  Test Set       : {len(test_df):,} samples ({split_stats['splits']['test']['percentage']}%) | Real: {split_stats['splits']['test']['real_count']} | AI: {split_stats['splits']['test']['ai_count']}")
    print("-" * 65)

    # Extract demo audio files for web UI (Hindi, Marathi, Tamil, etc.)
    print("\n[+] Extracting authentic Indic demo audio samples from test split...")
    demo_targets = [
        ("Hindi", 0, "indic_hindi_genuine.wav"),
        ("Hindi", 1, "indic_hindi_synthetic_clone.wav"),
        ("Marathi", 0, "indic_marathi_genuine.wav"),
        ("Marathi", 1, "indic_marathi_synthetic_clone.wav"),
        ("Tamil", 0, "indic_tamil_genuine.wav"),
        ("Tamil", 1, "indic_tamil_synthetic_clone.wav")
    ]

    for target_lang, target_label, filename in demo_targets:
        match = test_df[(test_df['language'] == target_lang) & (test_df['is_tts'] == target_label)]
        if not match.empty:
            row = match.iloc[0]
            pf = BASE_DIR / row['parquet_file']
            tbl = pq.read_table(str(pf), columns=['audio'])
            audio_bytes = tbl.slice(int(row['row_index']), 1).to_pylist()[0]['audio']['bytes']
            out_path = INDIC_DEMO_DIR / filename
            with open(out_path, 'wb') as f:
                f.write(audio_bytes)
            print(f"  [DEMO] Extracted {filename} ({target_lang}, Label: {target_label})")

    # Create spliced attack sample: Hindi genuine speech + Hindi synthetic clone directive
    h_real_path = INDIC_DEMO_DIR / "indic_hindi_genuine.wav"
    h_fake_path = INDIC_DEMO_DIR / "indic_hindi_synthetic_clone.wav"
    if h_real_path.exists() and h_fake_path.exists():
        real_data, sr = sf.read(str(h_real_path))
        fake_data, _ = sf.read(str(h_fake_path))
        spliced = np.concatenate([real_data[:int(sr * 2.5)], fake_data[:int(sr * 3.5)]])
        spliced_path = INDIC_DEMO_DIR / "indic_spliced_attack.wav"
        sf.write(str(spliced_path), spliced, sr)
        print(f"  [DEMO] Created {spliced_path.name} (Spliced Real+Synthetic Attack)")

    print(f"\n[✓] Splits created and saved to: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
