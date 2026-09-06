"""
VoiceShield CLI Inference Module
================================
Performs genuine deepfake inference and segment-level chunk analysis on arbitrary audio files.
Uses the locally trained VoiceShield model.
Outputs AI/Human probabilities, voice risk rating, confidence, and temporal timeline.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.features import AudioFeatureExtractor, load_audio, is_sufficient_speech
from ml.model import VoiceShieldNet

MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL = MODELS_DIR / "voiceshield_model.pt"


class VoiceShieldDetector:
    """Production inference detector for VoiceShield."""

    def __init__(self, model_path: str = None):
        self.device = torch.device("cpu")  # CPU is optimal and sub-50ms for single-sample inference
        self.feature_extractor = AudioFeatureExtractor()

        if model_path is None:
            if DEFAULT_MODEL.exists():
                model_path = str(DEFAULT_MODEL)
            else:
                raise FileNotFoundError("No trained model weights found at models/voiceshield_model.pt. Run scripts/training.py first.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = VoiceShieldNet(in_channels=3, num_classes=1)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.model_path = model_path
        self.dataset_name = checkpoint.get("dataset_name", "Kaggle Fake and Real Audio Dataset")

    def analyze(self, audio_source, apply_vad: bool = True, filename: Optional[str] = None) -> dict:
        """
        Runs complete inference pipeline:
        1. Preprocessing & VAD (supports WAV and MP3 transparently)
        2. Speech energy / sufficiency verification
        3. Chunk-level sliding window feature extraction
        4. Neural model forward passes
        5. Temporal aggregation (trimmed mean & median)
        6. Voice risk scoring
        """
        # Load audio through central loader
        inferred_name = filename
        if isinstance(audio_source, str) and not inferred_name:
            inferred_name = Path(audio_source).name

        audio, sr = load_audio(audio_source, target_sr=self.feature_extractor.sample_rate, apply_vad=apply_vad, filename=inferred_name)
        total_duration = len(audio) / sr

        # Load calibration if available
        cal_temp = 1.0
        opt_thresh = 0.50
        cal_path = MODELS_DIR / "calibration_config.json"
        if cal_path.exists():
            try:
                import json
                with open(cal_path, "r") as f:
                    cal_data = json.load(f)
                    cal_temp = float(cal_data.get("temperature", 1.0))
                    opt_thresh = float(cal_data.get("optimal_threshold", 0.50))
            except Exception:
                pass

        # Check for silence or non-speech
        is_sufficient, active_speech_sec = is_sufficient_speech(audio, sr=sr)
        if not is_sufficient:
            return {
                "model": "VoiceShield Deepfake Detector",
                "model_path": self.model_path,
                "duration_sec": round(total_duration, 2),
                "total_chunks": 0,
                "ai_generated_probability": 0.0,
                "human_probability": 0.0,
                "classification": "INSUFFICIENT_SPEECH",
                "confidence": 0.0,
                "voice_risk_score": 0.0,
                "risk_status": "INSUFFICIENT_SPEECH",
                "partial_spoof_detected": False,
                "operating_threshold": opt_thresh,
                "speech_duration": round(active_speech_sec, 2),
                "chunk_timeline": []
            }

        # Slice into chunks for real-time style timeline analysis
        segments = self.feature_extractor.segment_audio(audio, overlap=0.5)

        if len(segments) == 0:
            return {
                "model": "VoiceShield Deepfake Detector",
                "model_path": self.model_path,
                "duration_sec": round(total_duration, 2),
                "total_chunks": 0,
                "ai_generated_probability": 0.0,
                "human_probability": 1.0,
                "classification": "GENUINE HUMAN SPEECH",
                "confidence": 1.0,
                "voice_risk_score": 0.0,
                "risk_status": "LOW RISK",
                "partial_spoof_detected": False,
                "operating_threshold": opt_thresh,
                "speech_duration": round(total_duration, 2),
                "chunk_timeline": []
            }

        chunk_results = []
        chunk_probs = []

        with torch.no_grad():
            for seg in segments:
                if isinstance(seg, tuple):
                    start_sec, end_sec, chunk_audio = seg
                    time_label = f"{start_sec:.1f}s - {end_sec:.1f}s"
                elif isinstance(seg, dict):
                    start_sec = seg["start_sec"]
                    end_sec = seg["end_sec"]
                    chunk_audio = seg["audio"]
                    time_label = seg.get("time_label", f"{start_sec:.1f}s - {end_sec:.1f}s")
                else:
                    continue

                feat = self.feature_extractor.extract_features(chunk_audio, random_crop=False)
                tensor = torch.from_numpy(feat).unsqueeze(0).to(self.device)

                raw_logit = self.model(tensor).item()
                scaled_logit = raw_logit / max(0.01, cal_temp)
                ai_prob = float(torch.sigmoid(torch.tensor(scaled_logit)).item())

                chunk_probs.append(ai_prob)

                # Segment risk category
                if ai_prob >= 0.70:
                    c_risk = "HIGH"
                elif ai_prob >= 0.40:
                    c_risk = "MEDIUM"
                else:
                    c_risk = "LOW"

                chunk_results.append({
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "time_label": time_label,
                    "ai_probability": round(ai_prob, 4),
                    "human_probability": round(1.0 - ai_prob, 4),
                    "risk_level": c_risk
                })

        # Temporal Aggregation: Trimmed Mean + Median
        probs_arr = np.array(chunk_probs)
        if len(probs_arr) >= 4:
            # 10% trimmed mean removes single outlier burst
            low_pct = np.percentile(probs_arr, 10)
            high_pct = np.percentile(probs_arr, 90)
            trimmed = probs_arr[(probs_arr >= low_pct) & (probs_arr <= high_pct)]
            aggregated_prob = float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.median(probs_arr))
        else:
            aggregated_prob = float(np.mean(probs_arr))

        # Check for partial spoofing
        high_risk_chunks = sum(1 for p in chunk_probs if p >= 0.75)
        partial_spoof = (high_risk_chunks >= 1 and high_risk_chunks < len(chunk_probs) * 0.7)

        # Global probability assignment
        global_ai_prob = max(aggregated_prob, 0.75) if partial_spoof else aggregated_prob
        global_human_prob = 1.0 - global_ai_prob

        # Calibrated voice risk score (0 - 100)
        voice_risk_score = int(round(global_ai_prob * 100))

        # Classification decision based on calibrated threshold
        if global_ai_prob >= opt_thresh or partial_spoof:
            classification = "AI-GENERATED / VOICE CLONE"
            risk_status = "CRITICAL RISK" if global_ai_prob >= 0.80 else "HIGH RISK"
        elif global_ai_prob >= (opt_thresh * 0.75):
            classification = "SUSPICIOUS / INCONCLUSIVE"
            risk_status = "MEDIUM RISK"
        else:
            classification = "GENUINE HUMAN SPEECH"
            risk_status = "LOW RISK"

        confidence = round(max(global_ai_prob, global_human_prob), 4)

        return {
            "model": "VoiceShield Deepfake Detector",
            "model_path": self.model_path,
            "duration_sec": round(total_duration, 2),
            "total_chunks": len(chunk_results),
            "ai_generated_probability": round(global_ai_prob, 4),
            "human_probability": round(global_human_prob, 4),
            "classification": classification,
            "confidence": confidence,
            "voice_risk_score": voice_risk_score,
            "risk_status": risk_status,
            "partial_spoof_detected": partial_spoof,
            "operating_threshold": opt_thresh,
            "chunk_timeline": chunk_results
        }


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Deepfake Inference")
    parser.add_argument("--file", "-f", "--audio", "-a", dest="file", type=str, required=True, help="Path to input audio file (WAV/MP3)")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--json", action="store_true", help="Print JSON output only")
    args = parser.parse_args()

    detector = VoiceShieldDetector(model_path=args.model)

    if not args.json:
        print("=" * 65)
        print("Initializing VoiceShield Inference Engine...")
        print(f"  Model Checkpoint : {detector.model_path}")
        print(f"  Sample Rate      : 16000 Hz (mono float32)")
        print(f"  Feature Config   : 64-band Log-Mel + Delta + Delta-Delta")
        print(f"  Class Mapping    : 0 = HUMAN, 1 = AI_GENERATED")
        print("=" * 65)

    results = detector.analyze(args.file)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("=" * 65)
    print("        VOICESHIELD DEEPFAKE DETECTION RESULT")
    print("=" * 65)
    print(f"  Target Audio File       : {args.file}")
    print(f"  Model Engine            : {results['model']}")
    print(f"  Duration                : {results['duration_sec']} seconds ({results['total_chunks']} chunks)")
    print(f"  AI-Generated Probability: {results['ai_generated_probability']*100:.1f}%")
    print(f"  Human Probability       : {results['human_probability']*100:.1f}%")
    print(f"  Classification          : {results['classification']}")
    print(f"  Model Confidence        : {results['confidence']*100:.1f}%")
    print(f"  Voice Risk Score        : {results['voice_risk_score']} / 100 ({results['risk_status']})")
    print("-" * 65)
    if results["total_chunks"] > 0:
        print("  Segment-Level Timeline:")
        for chunk in results["chunk_timeline"]:
            bar_len = int(chunk["ai_probability"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"   [{chunk['time_label']:^13}] AI: {chunk['ai_probability']*100:5.1f}% [{bar}] {chunk['risk_level']}")
    else:
        print(f"  Note: {results.get('classification', 'No active speech chunks')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
