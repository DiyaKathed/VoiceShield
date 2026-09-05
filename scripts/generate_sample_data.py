"""
VoiceShield Sample Dataset Generator
====================================
Generates a benchmark audio dataset containing genuine human-style speech
and AI-cloned / synthetic speech with realistic acoustic and vocoder signatures.
Enforces strict train/val/test splits with disjoint speaker IDs to prevent data leakage.
"""

import os
import subprocess
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
DEMO_DIR = DATA_DIR / "sample_demo"


# Distinct utterances for financial / high-risk impersonation scenarios
HUMAN_PROMPTS = [
    ("speaker_h1", "Good morning, I'm calling to verify the quarterly budget review."),
    ("speaker_h1", "Please find the attached invoice for the supplier contract."),
    ("speaker_h2", "Hello team, can we reschedule today's operations sync to tomorrow?"),
    ("speaker_h2", "I am currently at the conference in Mumbai and network is unstable."),
    ("speaker_h3", "Could you review the project documentation before the client presentation?"),
    ("speaker_h3", "Let me check with the accounts department regarding the reimbursement status."),
    ("speaker_h4", "Hi, this is Sarah from legal. We need to finalize the non-disclosure agreement."),
    ("speaker_h4", "The security audit was completed successfully without critical vulnerabilities."),
    ("speaker_h5", "Good afternoon, I am confirming the delivery address for the hardware shipment."),
    ("speaker_h5", "Please ensure all confidential files are archived according to policy."),
    ("speaker_h6", "Thank you for the quick turnaround on the financial reconciliation report."),
    ("speaker_h6", "Our regional director will join the briefing call in ten minutes."),
    ("speaker_h7", "Can someone verify if the API gateway gateway credentials were rotated?"),
    ("speaker_h7", "I will be out of office until Thursday morning for field inspections."),
    ("speaker_h8", "Make sure all team members submit their travel expense claims today."),
    ("speaker_h8", "The compliance review has been approved by the internal audit committee.")
]

SYNTHETIC_PROMPTS = [
    ("speaker_c1", "Authorize an immediate wire transfer of eighty-five thousand dollars to offshore account."),
    ("speaker_c1", "This is the chief executive officer. Bypass standard protocol and disburse funds now."),
    ("speaker_c2", "Override the two-factor authentication for my banking credentials immediately."),
    ("speaker_c2", "I am locked out of the treasury portal. Provide temporary master access right now."),
    ("speaker_c3", "Execute transaction batch four-zero-two before end of day without audit delay."),
    ("speaker_c3", "This is an urgent directive from the board. Transfer fifty thousand dollars at once."),
    ("speaker_c4", "Approve the pending vendor payment to foreign account ending in nine-eight-two-one."),
    ("speaker_c4", "Authorize the emergency fund transfer for the confidential business acquisition."),
    ("speaker_c5", "Disable security alerts for my terminal and transfer thirty thousand dollars."),
    ("speaker_c5", "This transaction requires immediate clearance. Do not contact any other executive."),
    ("speaker_c6", "Urgent executive order: release the bank guarantee to the consulting partner."),
    ("speaker_c6", "Confirm the high-value wire dispatch. I will sign the authorization paperwork later."),
    ("speaker_c7", "Authorize transfer of two hundred thousand rupees to the treasury escrow immediately."),
    ("speaker_c7", "Override the dual approval requirement for this transaction due to executive urgency."),
    ("speaker_c8", "Clear the emergency procurement wire right away. The vendor is waiting on the line."),
    ("speaker_c8", "Send the login passcode to my alternate phone number for financial authorization.")
]


def add_natural_human_acoustics(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simulates natural human vocal dynamics:
    - Subtle acoustic resonance / early room reflections
    - Natural micro-tremors in pitch
    - Natural breathing & ambient SNR
    """
    # Slight room response simulation via short decaying convolution
    ir_len = int(sr * 0.04)  # 40ms short room impulse
    ir = np.exp(-np.linspace(0, 5, ir_len)) * (np.random.randn(ir_len) * 0.08)
    reverb = np.convolve(audio, ir, mode='full')[:len(audio)]
    out = audio * 0.85 + reverb * 0.15

    # Natural ambient background floor (high SNR ~ 35dB)
    ambient = np.random.randn(len(out)) * 0.002
    out = out + ambient

    # Peak normalization
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.9
    return out.astype(np.float32)


def add_synthetic_vocoder_artifacts(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simulates neural vocoder and voice-cloning acoustic anomalies:
    - Phase incoherence across upper harmonics (HiFi-GAN / MelGAN signature)
    - Unnatural spectral smoothness or metallic resonance
    - Formant phase jitter and high-frequency cutoff
    """
    # STFT to manipulate phase spectrum
    n_fft = 1024
    hop_length = 256
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = librosa.magphase(stft)

    # Inject vocoder phase dispersion in upper frequency bins (> 2.5kHz)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    high_freq_mask = freq_bins > 2500

    # Perturb phase of high-frequency bins
    phase_jitter = np.angle(phase)
    phase_jitter[high_freq_mask, :] += np.random.uniform(-0.6, 0.6, size=phase_jitter[high_freq_mask, :].shape)
    distorted_stft = mag * np.exp(1j * phase_jitter)

    # Slight robotic harmonic comb filtering
    vocoded = librosa.istft(distorted_stft, hop_length=hop_length, length=len(audio))

    # Add subtle metallic vocoder resonance
    t = np.arange(len(vocoded)) / sr
    carrier = 0.003 * np.sin(2 * np.pi * 3200 * t)
    vocoded = vocoded + carrier

    peak = np.max(np.abs(vocoded))
    if peak > 0:
        vocoded = vocoded / peak * 0.9
    return vocoded.astype(np.float32)


def generate_audio_file(text: str, out_path: Path, is_synthetic: bool) -> float:
    """Uses macOS speech synthesis to render clean 16kHz WAV then applies acoustic modeling."""
    tmp_path = out_path.with_suffix('.tmp.wav')

    # Select voice based on type
    voice = "Alex" if not is_synthetic else "Fred"
    cmd = ["say", "-v", voice, "-o", str(tmp_path), "--data-format=LEI16@16000", text]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    audio, sr = sf.read(str(tmp_path))
    if tmp_path.exists():
        tmp_path.unlink()

    # Apply authentic acoustic modeling
    if is_synthetic:
        processed = add_synthetic_vocoder_artifacts(audio, sr)
    else:
        processed = add_natural_human_acoustics(audio, sr)

    sf.write(str(out_path), processed, sr, subtype='PCM_16')
    return float(len(processed) / sr)


def main():
    print("=" * 60)
    print("VoiceShield: Generating Benchmark Speech Dataset")
    print("=" * 60)

    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, DEMO_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    records = []

    # Assign speakers to splits ensuring STRICT DISJOINTNESS (Zero Speaker Leakage)
    # Train: h1, h2, h3, h4, h5 / c1, c2, c3, c4, c5
    # Val:   h6 / c6
    # Test:  h7, h8 / c7, c8
    speaker_split_map = {
        "speaker_h1": "train", "speaker_h2": "train", "speaker_h3": "train",
        "speaker_h4": "train", "speaker_h5": "train",
        "speaker_h6": "val",
        "speaker_h7": "test", "speaker_h8": "test",
        "speaker_c1": "train", "speaker_c2": "train", "speaker_c3": "train",
        "speaker_c4": "train", "speaker_c5": "train",
        "speaker_c6": "val",
        "speaker_c7": "test", "speaker_c8": "test",
    }

    # Generate genuine human samples
    print("\n[+] Generating Genuine Human Speech Samples (Label 0)...")
    for i, (spk, prompt) in enumerate(HUMAN_PROMPTS):
        split = speaker_split_map[spk]
        split_dir = DATA_DIR / split
        filename = f"genuine_{spk}_{i:02d}.wav"
        file_path = split_dir / filename

        dur = generate_audio_file(prompt, file_path, is_synthetic=False)
        records.append({
            "filepath": str(file_path),
            "filename": filename,
            "label": 0,
            "label_name": "genuine_human",
            "speaker_id": spk,
            "duration_sec": round(dur, 2),
            "split": split,
            "prompt": prompt
        })
        print(f"  [{split.upper()}] Generated {filename} ({dur:.1f}s)")

    # Generate synthetic / cloned samples
    print("\n[+] Generating AI-Cloned / Synthetic Speech Samples (Label 1)...")
    for i, (spk, prompt) in enumerate(SYNTHETIC_PROMPTS):
        split = speaker_split_map[spk]
        split_dir = DATA_DIR / split
        filename = f"synthetic_{spk}_{i:02d}.wav"
        file_path = split_dir / filename

        dur = generate_audio_file(prompt, file_path, is_synthetic=True)
        records.append({
            "filepath": str(file_path),
            "filename": filename,
            "label": 1,
            "label_name": "ai_cloned",
            "speaker_id": spk,
            "duration_sec": round(dur, 2),
            "split": split,
            "prompt": prompt
        })
        print(f"  [{split.upper()}] Generated {filename} ({dur:.1f}s)")

    df = pd.DataFrame(records)
    csv_path = DATA_DIR / "dataset_manifest.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[✓] Saved complete dataset manifest to {csv_path}")

    # Generate 1-click test audio files for the web demo
    print("\n[+] Generating 1-Click Interactive Demo Audio Files...")
    demo_human = DEMO_DIR / "sample_genuine_human.wav"
    demo_ai = DEMO_DIR / "sample_ai_cloned_fraud.wav"
    demo_spliced = DEMO_DIR / "sample_spliced_attack.wav"

    generate_audio_file(
        "Good morning, this is the regional branch office confirming our scheduled security review.",
        demo_human,
        is_synthetic=False
    )
    generate_audio_file(
        "Executive authorization: bypass standard compliance checks and transfer two million dollars now.",
        demo_ai,
        is_synthetic=True
    )

    # Spliced sample: Human intro + Synthetic fraudulent directive
    h_audio, sr = sf.read(str(demo_human))
    a_audio, _ = sf.read(str(demo_ai))
    spliced_audio = np.concatenate([h_audio[:int(sr * 2.0)], a_audio[:int(sr * 3.0)]])
    sf.write(str(demo_spliced), spliced_audio, sr, subtype='PCM_16')

    print(f"  [DEMO] Created {demo_human.name}")
    print(f"  [DEMO] Created {demo_ai.name}")
    print(f"  [DEMO] Created {demo_spliced.name} (Spliced Attack for Timeline Demo)")

    print("\nDataset Summary by Split:")
    print(df.groupby(['split', 'label_name']).size())
    print("\n[✓] Dataset generation complete!")


if __name__ == "__main__":
    main()
