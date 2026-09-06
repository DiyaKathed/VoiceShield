# VoiceShield: AI-Powered Real-Time Detection & Prevention of Voice Cloning Impersonation Attacks

[![Smart India Hackathon](https://img.shields.io/badge/SIH-MVP-cyan.svg)](https://sih.gov.in)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev)

VoiceShield is an enterprise-grade cybersecurity system designed to detect and thwart AI voice cloning, neural text-to-speech (TTS), and audio deepfake impersonation attacks in high-risk transaction contexts (e.g., wire transfer authorization, CEO fraud, credentials reset).

---


## Key Highlights

- **100% Genuine Local PyTorch Inference**: No hardcoded mocks; no third-party cloud voice APIs. All audio feature extraction and neural inference run entirely on your local machine.
- **Spectro-Temporal Residual CNN (`VoiceShieldNet`)**: Analyzes 64-band Log-Mel Spectrograms alongside first- and second-order spectral deltas ($\Delta$ and $\Delta\Delta$), exposing the unnatural phase discontinuities and high-frequency harmonics characteristic of neural vocoders (HiFi-GAN, WaveGlow, FastSpeech, VITS).
- **Sliding-Window Segment Timeline**: Slices audio into short overlapping segments (1.5s windows) to detect **partial/spliced spoofing** (e.g., a human greeting followed by an AI-cloned financial directive).
- **Dual-Layer Risk Intelligence**:
  - **Voice ML Risk Score (0–100)**: Direct acoustic classification ($<40$ Low, $40–70$ Medium, $>70$ High).
  - **Contextual Transaction Threat Index (0–100)**: Combines voice probability with caller authority, transfer volume, urgency, and speaker verification status.
- **Comprehensive Evaluation Metrics**: Evaluated on an unseen test set with strict **speaker-level disjoint splitting** (zero speaker leakage) reporting Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC, and Equal Error Rate (EER).
- **VoiceShield Cyber Defense Dashboard**: Dark mode cybersecurity HUD built with React and modern CSS featuring live mic recording, drag-and-drop file upload, 1-click interactive demo scenarios, and an automated incident recommendation engine.

---

## Project Structure

```
SIH/
├── backend/                    # High-performance FastAPI backend service
│   ├── main.py                 # REST endpoints, CORS, model lifecycle
│   ├── risk_engine.py          # Dual-layer risk engine (Voice ML + Contextual Threat)
│   └── chunk_analyzer.py       # Sliding-window segmenter & partial spoof detector
├── frontend/                   # Modern React cybersecurity dashboard (VoiceShield)
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx               # Header, status pill, modal triggers
│   │   │   ├── AudioInputPanel.jsx      # 1-Click demos, mic recorder, file upload
│   │   │   ├── DetectionResultsPanel.jsx# Security alert, gauges, action recommendations
│   │   │   ├── SegmentTimeline.jsx      # Time-series chunk probability chart
│   │   │   ├── MetricsModal.jsx         # Test metrics report (ROC-AUC, EER, CM)
│   │   │   └── ResearchModal.jsx        # Deepfake detection research notes
│   │   ├── App.jsx                      # Main app controller
│   │   └── index.css                    # Obsidian dark mode design system
│   ├── vite.config.js          # Vite config with backend API proxy (/api -> :8000)
│   └── package.json
├── ml/                         # Core Machine Learning & Signal Processing
│   ├── features.py             # 16kHz resampling, VAD, Log-Mel + Deltas extractor
│   ├── model.py                # VoiceShieldNet PyTorch neural architecture
│   ├── dataset.py              # Kaggle dataset loader with RAM preloading & SpecAugment
│   ├── training.py             # PyTorch training pipeline with CosineAnnealingLR & calibration
│   ├── evaluation.py           # Authenticated evaluation metrics calculation & plotting
│   └── inference.py            # Standalone and backend inference engine
├── data/                       # Datasets & manifests
│   ├── raw/                    # Downloaded Kaggle RAW audio files (real/ and fake/)
│   ├── train/                  # Training split audio files (real/ and fake/)
│   ├── val/                    # Validation split audio files (real/ and fake/)
│   ├── test/                   # Unseen test split audio files (real/ and fake/)
│   ├── sample_demo/            # Interactive 1-click test audio files
│   ├── splits/                 # Split manifests (train.csv, val.csv, test.csv, split_summary.json)
│   └── dataset_manifest.csv    # Full dataset manifest (854 files, labels, durations)
├── models/                     # Saved model artifacts
│   ├── voiceshield_model.pt    # PyTorch trained weights checkpoint
│   ├── training_config.json    # Hyperparameters & audio config
│   ├── calibration_config.json # Learned temperature scaling parameters
│   └── eval_metrics.json       # Genuine evaluation metrics from test set
├── scripts/                    # Ingestion & audit utility scripts
│   ├── download_kaggle_dataset.py # Automated Kaggle dataset downloader
│   ├── prepare_kaggle_splits.py   # Stratified split generator (70/15/15)
│   └── audit_dataset.py           # Audits audio metadata & distributions
└── README.md
```

---

## Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (Tested up to Python 3.14 on macOS Apple Silicon / Linux / Windows)
- Node.js 18+ and npm

### 2. Backend & ML Environment Setup
```bash
# Clone the repository
cd SIH

# Activate or create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install torch librosa soundfile scipy scikit-learn pandas fastapi uvicorn python-multipart
```

### 3. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## Reproducing Dataset, Training & Evaluation

### Step 1: Ingest and Partition the Kaggle Deepfake Dataset
Download and extract the Kaggle dataset (`pawarrohitashok/fake-and-real-audio-dataset-deepfake-data`), then create stratified 70% / 15% / 15% train, validation, and test splits:
```bash
# 1. Download and extract raw audio files (423 Real, 431 Fake)
python scripts/download_kaggle_dataset.py

# 2. Partition into train/val/test splits and create manifests
python scripts/prepare_kaggle_splits.py

# 3. Optional: Run dataset audit report
python scripts/audit_dataset.py
```
This structures:
- `data/train/` (598 files: 296 Real, 302 Fake)
- `data/val/` (128 files: 63 Real, 65 Fake)
- `data/test/` (128 files: 64 Real, 64 Fake)
- `data/sample_demo/` (Interactive sample audio files for instant UI testing)
- `data/dataset_manifest.csv` and `data/splits/{train.csv, val.csv, test.csv, split_summary.json}`

> **Note on Speaker-Disjoint Splitting:**  
> The Kaggle dataset provides sequentially indexed audio recordings (`real_001.wav`–`real_423.wav`, `fake_001.wav`–`fake_431.wav`) without ground-truth speaker metadata or actor identities. Consequently, speaker-disjoint isolation cannot be strictly enforced; a stratified random partition with fixed seed (`42`) is applied across Real and Fake classes to ensure balanced representation and completely deterministic evaluation without data leakage.

### Step 2: Train the Model
Train `VoiceShieldNet` with AdamW, Cosine Annealing, and post-hoc temperature calibration:
```bash
python ml/training.py --epochs 8 --batch-size 32 --lr 0.001 --use-calibrated
```
The best checkpoint is automatically saved to `models/voiceshield_model.pt`.

### Step 3: Run Evaluation on Unseen Test Set
Calculate authentic metrics on the held-out test split (128 files: 64 Real, 64 AI):
```bash
python ml/evaluation.py
```
Outputs:
- **Accuracy**: $89.06\%$
- **Precision**: $87.88\%$
- **Recall**: $90.62\%$
- **Specificity**: $87.50\%$
- **F1-Score**: $89.23\%$
- **ROC-AUC**: $0.9546$
- **Equal Error Rate (EER)**: $13.28\%$ at decision threshold $0.5486$
- **Confusion Matrix**:
  - True Negatives (Human correctly identified): 56
  - False Positives (Human misclassified as AI): 8
  - False Negatives (AI misclassified as Human): 6
  - True Positives (AI correctly identified): 58

### Step 4: Standalone CLI Inference
Run inference directly on any WAV or MP3 audio file:
```bash
# Test on genuine human audio (Real)
python ml/inference.py --audio data/test/real/real_004.wav

# Test on synthetic deepfake audio (Fake)
python ml/inference.py --audio data/test/fake/fake_005.wav

# Batch test an entire directory with JSON output
python ml/inference.py --audio data/test/real/real_001.wav --json
```

---

## Running the Application

### 1. Start the FastAPI Backend
```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 2. Start the React Frontend Dashboard
In a separate terminal:
```bash
cd frontend
npm run dev
```
Open [http://127.0.0.1:5173](http://127.0.0.1:5173) in your browser.

---

## Architectural & Research Design

VoiceShield is engineered to address known limitations in voice deepfake detection:

| Research Limitation | Vulnerability / Challenge | VoiceShield Architectural Solution |
| :--- | :--- | :--- |
| **Unseen Synthetic Generators** | New neural vocoders (e.g. Diffusion vocoders) bypass detectors trained on older algorithms. | Multi-scale Log-Mel feature maps combined with 1st and 2nd order temporal deltas capture high-frequency phase discontinuities regardless of generator architecture. |
| **Partial / Spliced Spoofing** | Attacker speaks naturally, then splices a short AI-cloned command (e.g., "Approve transfer"). Global averaging misses short attacks. | 1.5s sliding window temporal segmenter analyzes speech chunk-by-chunk. Aggregation uses weighted peak pooling: $\text{Prob}_{\text{global}} = 0.65 \cdot \max(P) + 0.35 \cdot \text{mean}(P)$. |
| **Compression & Channel Distortion** | Telephony (G.711, Opus, AMR-WB) strips high frequencies and introduces quantization noise. | Audio is standardized to 16kHz mono; acoustic filters focus on mid-frequency formant dynamics ($500\text{Hz} - 3.5\text{kHz}$) that survive telephony codecs. |
| **Background Noise & SNR** | Ambient noise can mask vocoder artifacts or trigger false alarms. | Energy normalization and SpecAugment (frequency and time masking) during training prevent acoustic overfitting. |
| **Silence / Non-Speech Artifacts** | Silent pauses or breathing sounds bias neural layers toward background room tone. | Integrated energy Voice Activity Detection (VAD) trims unvoiced silence frames before inference. |
| **Speaker & Dataset Leakage** | Memorizing speaker pitch rather than synthetic artifacts. | Disjoint speaker partitioning guarantees that speakers in the training set never appear in validation or testing. |

---

## What is Implemented vs. Future Work

### Implemented in MVP:
- Complete local PyTorch training, evaluation, and inference pipeline.
- 64-band Log-Mel Spectrogram + spectral delta feature extractor with VAD.
- Sliding-window segment analyzer detecting partial/spliced voice cloning attacks.
- Dual-layer risk engine (Voice ML score + Contextual financial transaction threat score).
- React frontend with dark-mode cybersecurity aesthetic, 1-click testing, live mic recording, and evaluation reporting.
- REST API with FastAPI and CORS support.

### Future Work / Production Roadmap:
- **Wav2Vec2 / HuBERT Self-Supervised Backbones**: Incorporating large pretrained self-supervised representations for zero-shot multilingual deepfake generalization.
- **Continuous Telephony PBX SIP Integration**: Streaming RTP audio directly from VoIP switches (FreeSWITCH / Asterisk) into the chunk analyzer for live call interception.
- **Hardware Security Key Out-of-Band Verification**: Automated webhook triggers issuing FIDO2 / WebAuthn verification requests when high-risk calls are flagged.
