"""
VoiceShield Kaggle Dataset Ingestion Script
===========================================
Downloads and extracts pawarrohitashok/fake-and-real-audio-dataset-deepfake-data.
"""

import os
import sys
import time
import ssl
import certifi
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_ZIP = DATA_DIR / "kaggle_raw.zip"
EXTRACT_DIR = DATA_DIR / "kaggle_raw"

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/pawarrohitashok/fake-and-real-audio-dataset-deepfake-data"
EXPECTED_SIZE = 2122568484


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=certifi.where())

    existing_size = RAW_ZIP.stat().st_size if RAW_ZIP.exists() else 0
    if existing_size == EXPECTED_SIZE:
        print(f"[✓] Archive already fully downloaded: {RAW_ZIP} ({existing_size:,} bytes)")
        return

    headers = {"User-Agent": "Mozilla/5.0 VoiceShield/2.0"}
    mode = "wb"
    if existing_size > 0:
        print(f"[+] Resuming download from byte {existing_size:,}...")
        headers["Range"] = f"bytes={existing_size}-{EXPECTED_SIZE - 1}"
        mode = "ab"
    else:
        print(f"[+] Starting download of Kaggle dataset ({EXPECTED_SIZE / (1024**3):.2f} GB)...")

    req = urllib.request.Request(KAGGLE_URL, headers=headers)
    t0 = time.time()
    last_print = t0
    downloaded = existing_size

    with urllib.request.urlopen(req, context=ctx) as response, open(RAW_ZIP, mode) as out_file:
        chunk_size = 1024 * 1024  # 1 MB chunk
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_print >= 5.0 or downloaded == EXPECTED_SIZE:
                pct = (downloaded / EXPECTED_SIZE) * 100
                mb = downloaded / (1024 * 1024)
                speed = (downloaded - existing_size) / max(1e-5, (now - t0)) / (1024 * 1024)
                print(f"  [Download Progress] {mb:.1f} MB / {EXPECTED_SIZE / (1024**2):.1f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s", flush=True)
                last_print = now

    final_size = RAW_ZIP.stat().st_size
    print(f"[✓] Download complete: {RAW_ZIP} ({final_size:,} bytes) in {time.time() - t0:.1f}s")


def extract_dataset():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Extracting {RAW_ZIP} to {EXTRACT_DIR}...")
    t0 = time.time()
    with zipfile.ZipFile(RAW_ZIP, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"[✓] Extraction finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    download_dataset()
    extract_dataset()
