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
            if is_live_recording:
                chunk_rms = float(np.sqrt(np.mean(chunk_arr**2)))
                chunk_peak = float(np.max(np.abs(chunk_arr)))
                # A chunk is unvoiced room silence if both RMS and peak are low
                is_silence_chunk = (chunk_rms < 0.012 and chunk_peak < 0.035)
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
                    if ai_prob_clamped < 0.40:
                        risk_level = "LOW"
                        low_risk_chunks += 1
                    elif ai_prob_clamped <= 0.70:
                        risk_level = "MEDIUM"
                    else:
                        risk_level = "HIGH"
                        high_risk_chunks += 1
                    classification = "AI_GENERATED" if ai_prob_clamped >= self.threshold else "HUMAN"
            else:
                ai_prob_clamped = max(0.0, min(1.0, float(ai_prob)))
                human_prob = 1.0 - ai_prob_clamped
                if ai_prob_clamped < 0.40:
                    risk_level = "LOW"
                    low_risk_chunks += 1
                elif ai_prob_clamped <= 0.70:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "HIGH"
                    high_risk_chunks += 1
                classification = "AI_GENERATED" if ai_prob_clamped >= self.threshold else "HUMAN"

            chunks_data.append({
                "chunk_id": len(chunks_data) + 1,
                "start_time": round(start_sec, 2),
                "end_time": round(end_sec, 2),
                "time_label": f"{start_sec:.1f}s - {end_sec:.1f}s",
                "ai_probability": round(ai_prob_clamped, 4),
                "human_probability": round(human_prob, 4),
                "risk_level": risk_level,
                "classification": classification,
                "is_suspicious": ai_prob_clamped >= 0.60
            })

        # Aggregation selection: for live microphone recordings, aggregate over active speech
        eval_probs = active_speech_probs if (is_live_recording and len(active_speech_probs) > 0) else probs

        peak_prob = float(max(eval_probs))
        min_prob = float(min(eval_probs))
        mean_prob = float(np.mean(eval_probs))
        median_prob = float(np.median(eval_probs))
        p80_prob = float(np.percentile(eval_probs, 80))

        # Trimmed mean: drop top & bottom outliers if 3+ chunks
        if len(eval_probs) >= 3:
            sorted_p = sorted(eval_probs)
            trimmed_mean = float(np.mean(sorted_p[1:-1]))
        else:
            trimmed_mean = mean_prob

        # Contiguous high-risk chunk count for partial spoofing detection
        consecutive_high = 0
        max_consecutive_high = 0
        for p in eval_probs:
            if p >= 0.65:
                consecutive_high += 1
                max_consecutive_high = max(max_consecutive_high, consecutive_high)
            else:
                consecutive_high = 0

        # Partial spoofing requires meaningful sustained synthetic content (>= 3 consecutive chunks or >= 35% of chunks)
        partial_spoof_detected = (
            (max_consecutive_high >= 3 and low_risk_chunks >= 2) or
            (high_risk_chunks >= max(3, int(len(eval_probs) * 0.35)) and low_risk_chunks >= 2)
        )

        # Robust recording-level aggregation:
        if partial_spoof_detected:
            aggregated_ai_prob = float(0.60 * p80_prob + 0.40 * trimmed_mean)
        else:
            aggregated_ai_prob = float(0.50 * trimmed_mean + 0.50 * median_prob)

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
            "partial_spoof_detected": partial_spoof_detected,
            "is_insufficient_speech": False,
            "speech_duration": round(active_speech_sec, 2)
        }


# Alias for backward compatibility
AudioChunkAnalyzer = RealTimeChunkAnalyzer
