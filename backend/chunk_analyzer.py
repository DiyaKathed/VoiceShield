"""
VoiceShield Chunk Analyzer Module
=================================
Processes streaming or recorded speech in short sliding-window segments.
Performs chunk-level deepfake inference, detects partial/spliced impersonation attacks,
and constructs temporal timelines of AI-generated probabilities.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from ml.features import AudioFeatureExtractor
from ml.model import VoiceShieldNet


class RealTimeChunkAnalyzer:
    """
    Sliding-window chunk analyzer for near-real-time streaming-style speech verification.
    """

    def __init__(
        self,
        model: VoiceShieldNet,
        feature_extractor: AudioFeatureExtractor,
        device: torch.device = None,
        temperature: float = 1.0,
        threshold: float = 0.50
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device or torch.device("cpu")
        self.temperature = max(0.05, float(temperature))
        self.threshold = float(threshold)

    def analyze_audio_stream(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        is_live_recording: bool = False
    ) -> Dict[str, Any]:
        """
        Splits input audio into overlapping chunks, executes parallel batch inference,
        and generates chronological timeline.
        """
        duration = len(audio) / sr

        # Detect silence or non-speech audio before passing to model
        from ml.features import is_sufficient_speech
        is_sufficient, active_speech_sec = is_sufficient_speech(audio, sr=sr)
        if not is_sufficient:
            return {
                "total_duration": round(duration, 2),
                "total_chunks": 0,
                "chunks": [],
                "partial_spoof_detected": False,
                "peak_ai_probability": 0.0,
                "min_ai_probability": 0.0,
                "mean_ai_probability": 0.0,
                "median_ai_probability": 0.0,
                "trimmed_mean_ai_probability": 0.0,
                "aggregated_ai_probability": 0.0,
                "classification": "INSUFFICIENT_SPEECH",
                "human_probability": 0.0,
                "voice_risk_score": 0.0,
                "is_insufficient_speech": True,
                "speech_duration": round(active_speech_sec, 2),
                "high_risk_chunk_count": 0,
                "low_risk_chunk_count": 0
            }

        segments = self.feature_extractor.segment_audio(audio, overlap=0.5)

        if len(segments) == 0:
            return {
                "total_duration": round(duration, 2),
                "total_chunks": 0,
                "chunks": [],
                "partial_spoof_detected": False,
                "peak_ai_probability": 0.0,
                "min_ai_probability": 0.0,
                "mean_ai_probability": 0.0,
                "median_ai_probability": 0.0,
                "trimmed_mean_ai_probability": 0.0,
                "aggregated_ai_probability": 0.0,
                "classification": "GENUINE HUMAN",
                "human_probability": 1.0,
                "voice_risk_score": 0.0,
                "is_insufficient_speech": False,
                "speech_duration": round(duration, 2),
                "high_risk_chunk_count": 0,
                "low_risk_chunk_count": 0
            }

        features_list = []
        for start_sec, end_sec, chunk_arr in segments:
            feat = self.feature_extractor.extract_features(chunk_arr)
            features_list.append(torch.from_numpy(feat).float())

        batch = torch.stack(features_list).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch)
            scaled_logits = logits / self.temperature
            probs = torch.sigmoid(scaled_logits).squeeze(-1).tolist()
            if isinstance(probs, float):
                probs = [probs]

        chunks_data = []
        high_risk_chunks = 0
        low_risk_chunks = 0
        active_speech_probs = []

        for (start_sec, end_sec, chunk_arr), ai_prob in zip(segments, probs):
            chunk_rms = float(np.sqrt(np.mean(chunk_arr**2)))
            chunk_peak = float(np.max(np.abs(chunk_arr)))

            # Detect unvoiced room silence / pauses (applies uniformly to all audio)
            is_silence_chunk = (chunk_rms < 0.015 and chunk_peak < 0.050)

            if is_silence_chunk:
                ai_prob_clamped = 0.0
                human_prob = 1.0
                risk_level = "LOW"
                low_risk_chunks += 1
                classification = "SILENCE / PAUSE"
            else:
                ai_prob_clamped = max(0.0, min(1.0, float(ai_prob)))
                human_prob = 1.0 - ai_prob_clamped
                active_speech_probs.append(ai_prob_clamped)

                if ai_prob_clamped >= self.threshold:
                    risk_level = "HIGH"
                    high_risk_chunks += 1
                    classification = "AI_GENERATED"
                elif ai_prob_clamped >= (self.threshold * 0.75):
                    risk_level = "MEDIUM"
                    classification = "SUSPICIOUS"
                else:
                    risk_level = "LOW"
                    low_risk_chunks += 1
                    classification = "HUMAN"

            chunks_data.append({
                "chunk_id": len(chunks_data) + 1,
                "start_time": round(start_sec, 2),
                "end_time": round(end_sec, 2),
                "time_label": f"{start_sec:.1f}s - {end_sec:.1f}s",
                "ai_probability": round(ai_prob_clamped, 4),
                "human_probability": round(human_prob, 4),
                "risk_level": risk_level,
                "classification": classification,
                "is_suspicious": ai_prob_clamped >= self.threshold
            })

        # Aggregation selection: aggregate across active speech intervals
        eval_probs = active_speech_probs if len(active_speech_probs) > 0 else [float(p) for p in probs]

        peak_prob = float(max(eval_probs))
        min_prob = float(min(eval_probs))
        mean_prob = float(np.mean(eval_probs))
        median_prob = float(np.median(eval_probs))
        p80_prob = float(np.percentile(eval_probs, 80))

        # Check for repeated AI predictions
        ai_chunks = [p for p in eval_probs if p >= self.threshold]
        num_ai = len(ai_chunks)

        consecutive_ai = 0
        max_consecutive_ai = 0
        for p in eval_probs:
            if p >= self.threshold:
                consecutive_ai += 1
                max_consecutive_ai = max(max_consecutive_ai, consecutive_ai)
            else:
                consecutive_ai = 0

        has_repeated_ai = (max_consecutive_ai >= 2) or (num_ai >= 2 and num_ai >= len(eval_probs) * 0.35)
        partial_spoof_detected = bool(has_repeated_ai and any(p < (self.threshold * 0.70) for p in eval_probs) and len(eval_probs) >= 3)

        # Robust aggregation:
        # 1. When repeated AI predictions are detected: synthetic content dominates
        #    so quiet breaths or trailing pauses do not dilute the clone alarm.
        # 2. When a single isolated spike occurs in human speech: outlier is suppressed.
        # 3. Otherwise: balanced mean/median aggregation.
        if has_repeated_ai:
            top_ai_mean = float(np.mean(sorted(ai_chunks, reverse=True)[:max(2, len(ai_chunks))]))
            aggregated_ai_prob = float(0.60 * top_ai_mean + 0.25 * p80_prob + 0.15 * mean_prob)
        elif num_ai == 1 and len(eval_probs) >= 3:
            sorted_p = sorted(eval_probs)
            trimmed = sorted_p[:-1]
            aggregated_ai_prob = float(0.60 * np.mean(trimmed) + 0.40 * median_prob)
        else:
            aggregated_ai_prob = float(0.50 * mean_prob + 0.50 * median_prob)

        aggregated_ai_prob = float(np.clip(aggregated_ai_prob, 0.0, 1.0))

        # Trimmed mean for diagnostic reference
        if len(eval_probs) >= 3:
            sorted_p = sorted(eval_probs)
            trimmed_mean = float(np.mean(sorted_p[1:-1]))
        else:
            trimmed_mean = mean_prob

        return {
            "total_duration": round(duration, 2),
            "total_chunks": len(chunks_data),
            "chunks": chunks_data,
            "min_ai_probability": round(min_prob, 4),
            "max_ai_probability": round(peak_prob, 4),
            "mean_ai_probability": round(mean_prob, 4),
            "median_ai_probability": round(median_prob, 4),
            "trimmed_mean_ai_probability": round(trimmed_mean, 4),
            "aggregated_ai_probability": round(aggregated_ai_prob, 4),
            "high_risk_chunk_count": high_risk_chunks,
            "low_risk_chunk_count": low_risk_chunks,
            "has_repeated_ai": has_repeated_ai,
            "partial_spoof_detected": partial_spoof_detected,
            "is_insufficient_speech": False,
            "speech_duration": round(active_speech_sec, 2),
            "operating_threshold": round(self.threshold, 4),
            "raw_logits": [round(float(l), 4) for l in logits.cpu().numpy().flatten()]
        }


# Alias for backward compatibility
AudioChunkAnalyzer = RealTimeChunkAnalyzer
