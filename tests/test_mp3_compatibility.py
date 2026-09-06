"""
VoiceShield MP3 vs WAV Compatibility Unit Test (Section 7)
===========================================================
1. Loads a known WAV file.
2. Generates its MP3 counterpart in memory / on disk.
3. Loads both through the central load_audio() function.
4. Verifies:
   - Both load successfully
   - Same sample rate (16,000 Hz)
   - Mono waveform (1D array)
   - float32 dtype
   - Valid non-zero duration within 5% of original
   - Identical feature dimensions [3, 64, expected_frames]
5. Runs model inference on both and records predictions.
"""

import os
import io
import sys
import unittest
import numpy as np
import soundfile as sf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.features import load_audio, inspect_audio, AudioFeatureExtractor, TARGET_SR
from ml.inference import VoiceShieldDetector


class TestMP3Compatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = VoiceShieldDetector()
        cls.extractor = AudioFeatureExtractor()
        
        # Select known human reference audio file
        human_candidates = [
            BASE_DIR / "data" / "real_world_human" / "real_human_telugu_te_f_finance_01382.wav",
            BASE_DIR / "data" / "real_world_human" / "real_human_bengali_BEN_F_ANGER_00422.wav",
            BASE_DIR / "data" / "sample_demo" / "indic" / "indic_marathi_genuine.wav",
        ]
        cls.wav_file = None
        for cand in human_candidates:
            if cand.exists():
                cls.wav_file = cand
                break
        
        if cls.wav_file is None:
            cls.wav_file = BASE_DIR / "tests" / "test_ref.wav"
            t = np.linspace(0, 2.0, 32000, dtype=np.float32)
            sig = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
            sf.write(str(cls.wav_file), sig, 16000)

        # Create MP3 version
        data, orig_sr = sf.read(str(cls.wav_file))
        buf = io.BytesIO()
        sf.write(buf, data, orig_sr, format="MP3")
        cls.mp3_bytes = buf.getvalue()

    def test_01_both_load_successfully(self):
        """Verifies both WAV and MP3 are loaded without errors."""
        audio_wav, sr_wav = load_audio(str(self.wav_file), apply_vad=True)
        audio_mp3, sr_mp3 = load_audio(self.mp3_bytes, filename="sample.mp3", apply_vad=True)

        self.assertIsNotNone(audio_wav)
        self.assertIsNotNone(audio_mp3)
        self.assertGreater(len(audio_wav), 0)
        self.assertGreater(len(audio_mp3), 0)

    def test_02_sample_rate_and_mono(self):
        """Verifies standardized 16kHz mono output."""
        audio_wav, sr_wav = load_audio(str(self.wav_file), apply_vad=True)
        audio_mp3, sr_mp3 = load_audio(self.mp3_bytes, filename="sample.mp3", apply_vad=True)

        self.assertEqual(sr_wav, TARGET_SR, f"WAV SR must be {TARGET_SR}")
        self.assertEqual(sr_mp3, TARGET_SR, f"MP3 SR must be {TARGET_SR}")
        self.assertEqual(audio_wav.ndim, 1, "WAV audio must be mono (1D)")
        self.assertEqual(audio_mp3.ndim, 1, "MP3 audio must be mono (1D)")
        self.assertEqual(audio_wav.dtype, np.float32, "WAV audio dtype must be float32")
        self.assertEqual(audio_mp3.dtype, np.float32, "MP3 audio dtype must be float32")

    def test_03_compatible_duration(self):
        """Verifies durations are within reasonable compression variance (< 5%)."""
        audio_wav, sr_wav = load_audio(str(self.wav_file), apply_vad=False)
        audio_mp3, sr_mp3 = load_audio(self.mp3_bytes, filename="sample.mp3", apply_vad=False)

        dur_wav = len(audio_wav) / sr_wav
        dur_mp3 = len(audio_mp3) / sr_mp3
        rel_diff = abs(dur_wav - dur_mp3) / max(dur_wav, 1e-6)

        self.assertLess(rel_diff, 0.05, f"Duration mismatch exceeds 5%: WAV={dur_wav:.2f}s, MP3={dur_mp3:.2f}s")

    def test_04_feature_dimensions(self):
        """Verifies extracted feature tensors have identical shapes."""
        audio_wav, _ = load_audio(str(self.wav_file), apply_vad=True)
        audio_mp3, _ = load_audio(self.mp3_bytes, filename="sample.mp3", apply_vad=True)

        feat_wav = self.extractor.extract_features(audio_wav)
        feat_mp3 = self.extractor.extract_features(audio_mp3)

        self.assertEqual(feat_wav.shape, feat_mp3.shape, "Feature shapes must match exactly")
        self.assertEqual(feat_wav.shape[0], 3, "Feature channels must be 3 (Mel, Delta, Delta-Delta)")
        self.assertEqual(feat_wav.shape[1], 64, "Mel bands must be 64")

    def test_05_inference_execution(self):
        """Runs inference on both and logs results."""
        res_wav = self.detector.analyze(str(self.wav_file))
        res_mp3 = self.detector.analyze(self.mp3_bytes, filename="sample.mp3")

        print("\n" + "=" * 60)
        print("MP3 VS WAV COMPATIBILITY TEST RESULT")
        print("=" * 60)
        print(f"  Reference File       : {self.wav_file.name}")
        print(f"  WAV AI Probability   : {res_wav['ai_generated_probability']*100:.1f}%")
        print(f"  MP3 AI Probability   : {res_mp3['ai_generated_probability']*100:.1f}%")
        print(f"  WAV Classification   : {res_wav['classification']}")
        print(f"  MP3 Classification   : {res_mp3['classification']}")
        print(f"  WAV Duration         : {res_wav['duration_sec']}s ({res_wav['total_chunks']} chunks)")
        print(f"  MP3 Duration         : {res_mp3['duration_sec']}s ({res_mp3['total_chunks']} chunks)")
        print("=" * 60)

        self.assertTrue(0.0 <= res_wav['ai_generated_probability'] <= 1.0)
        self.assertTrue(0.0 <= res_mp3['ai_generated_probability'] <= 1.0)
        self.assertIn("voice_risk_score", res_wav)
        self.assertIn("voice_risk_score", res_mp3)


if __name__ == "__main__":
    unittest.main()
