"""
VoiceShield CLI Inference Module (IndicTTS)
===========================================
Performs genuine deepfake inference and segment-level chunk analysis on arbitrary audio files.
Uses the locally trained VoiceShield IndicTTS model.
Outputs AI/Human probabilities, voice risk rating, confidence, and temporal timeline.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.features import AudioFeatureExtractor
from ml.model import VoiceShieldNet

MODELS_DIR = BASE_DIR / "models"
BEST_INDIC_MODEL = MODELS_DIR / "voiceshield_indictts_best.pt"
DEFAULT_MODEL = MODELS_DIR / "voiceshield_model.pt"


class VoiceShieldDetector:
    """Production inference detector for VoiceShield IndicTTS."""

    def __init__(self, model_path: str = None):
        self.device = torch.device("cpu")  # CPU is optimal and sub-50ms for single-sample inference
        self.feature_extractor = AudioFeatureExtractor()

        if model_path is None:
            if BEST_INDIC_MODEL.exists():
                model_path = str(BEST_INDIC_MODEL)
            elif DEFAULT_MODEL.exists():
                model_path = str(DEFAULT_MODEL)
            else:
                raise FileNotFoundError("No trained model weights found in models/. Run ml/training.py first.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = VoiceShieldNet(in_channels=3, num_classes=1)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.model_path = model_path
        self.dataset_name = checkpoint.get("dataset_name", "IndicTTS-Deepfake-Challenge-Data")

    def analyze(self, audio_source, apply_vad: bool = True) -> dict:
        """
        Runs complete inference pipeline:
        1. Preprocessing & VAD
        2. Chunk-level sliding window feature extraction
        3. Neural model forward passes
        4. Temporal aggregation (mean & peak spoof score)
        5. Voice risk scoring
        """
        # Load audio
        audio, sr = self.feature_extractor.load_audio(audio_source, apply_vad=apply_vad)
        total_duration = len(audio) / sr

        # Slice into chunks for real-time style timeline analysis
        segments = self.feature_extractor.segment_audio(audio, overlap=0.5)

        chunk_results = []
        chunk_tensors = []

        for start_sec, end_sec, chunk_audio in segments:
            feat = self.feature_extractor.extract_features(chunk_audio)
            chunk_tensors.append(torch.from_numpy(feat).float())

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

        # Batch forward pass
        batch_feats = torch.stack(chunk_tensors).to(self.device)
        with torch.no_grad():
            logits = self.model(batch_feats)
            # Apply temperature scaling
            scaled_logits = logits / max(0.01, cal_temp)
            ai_probs = torch.sigmoid(scaled_logits).squeeze(-1).tolist()
            if isinstance(ai_probs, float):
                ai_probs = [ai_probs]

        for (start_sec, end_sec, _), ai_prob in zip(segments, ai_probs):
            chunk_risk = "LOW" if ai_prob < 0.4 else ("MEDIUM" if ai_prob <= 0.7 else "HIGH")
            chunk_results.append({
                "start_time": round(start_sec, 2),
                "end_time": round(end_sec, 2),
                "time_label": f"{start_sec:.1f}s - {end_sec:.1f}s",
                "ai_probability": round(ai_prob, 4),
                "human_probability": round(1.0 - ai_prob, 4),
                "risk_level": chunk_risk
            })

        # Robust statistical aggregation (trimmed mean & median)
        peak_prob = float(max(ai_probs))
        mean_prob = float(np.mean(ai_probs))
        median_prob = float(np.median(ai_probs))

        if len(ai_probs) >= 3:
            sorted_p = sorted(ai_probs)
            trimmed_mean = float(np.mean(sorted_p[1:-1]))
        else:
            trimmed_mean = mean_prob

        high_chunks = sum(p >= 0.65 for p in ai_probs)
        low_chunks = sum(p < 0.40 for p in ai_probs)
        partial_spoof = (high_chunks >= 2 and low_chunks >= 2)

        if partial_spoof:
            p80 = float(np.percentile(ai_probs, 80))
            global_ai_prob = float(0.60 * p80 + 0.40 * trimmed_mean)
        else:
            global_ai_prob = float(0.50 * trimmed_mean + 0.50 * median_prob)

        global_human_prob = 1.0 - global_ai_prob
        voice_risk_score = round(global_ai_prob * 100.0, 1)

        # Classification based on validation operating threshold
        if global_ai_prob >= opt_thresh:
            classification = "SYNTHETIC / AI VOICE CLONE"
            risk_status = "HIGH RISK" if global_ai_prob >= 0.70 else "MEDIUM RISK"
        elif global_ai_prob >= 0.40:
            classification = "SUSPICIOUS / ANOMALOUS SPEECH"
            risk_status = "MEDIUM RISK"
        else:
            classification = "GENUINE HUMAN SPEECH"
            risk_status = "LOW RISK"

        confidence = round(max(global_ai_prob, global_human_prob), 4)

        return {
            "model": "VoiceShield IndicTTS",
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
    parser = argparse.ArgumentParser(description="VoiceShield IndicTTS Deepfake Inference")
    parser.add_argument("--file", type=str, required=True, help="Path to input audio file (WAV/MP3)")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--json", action="store_true", help="Print JSON output only")
    args = parser.parse_args()

    detector = VoiceShieldDetector(model_path=args.model)
    results = detector.analyze(args.file)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("=" * 65)
    print("        VOICESHIELD INDICTTS DEEPFAKE DETECTION RESULT")
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
    print("  Segment-Level Timeline:")
    for chunk in results["chunk_timeline"]:
        bar_len = int(chunk["ai_probability"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"   [{chunk['time_label']:^13}] AI: {chunk['ai_probability']*100:5.1f}% [{bar}] {chunk['risk_level']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
