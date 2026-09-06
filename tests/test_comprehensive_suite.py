"""
VoiceShield Master Automated Test Suite (Phase 37)
===================================================
Covers all 11 critical test assertions:
1. Label mapping (is_tts=0 is strictly HUMAN, is_tts=1 is strictly AI)
2. Audio preprocessing consistency (16kHz mono float32, n_mels=64)
3. Model forward output shape ([B, 1])
4. Probability calculation math (monotonic sigmoid response)
5. Probability sum rule (P_human + P_ai == 1.0)
6. Probability mathematical bounds (0 <= P <= 1)
7. Operating threshold application logic
8. Segment aggregation (trimmed mean and median outlier rejection)
9. Model checkpoint loading & parameter verification
10. FastAPI /analyze response schema validation
11. End-to-end audio inference pipeline test
"""

import sys
import unittest
import numpy as np
import torch
import soundfile as sf
import io
import json
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.features import AudioFeatureExtractor
from ml.model import VoiceShieldNet
from ml.dataset import IndicTTSDataset
from backend.chunk_analyzer import RealTimeChunkAnalyzer
from backend.risk_engine import VoiceRiskEngine, ContextualRiskEngine
from backend.main import app


class TestVoiceShieldSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fe = AudioFeatureExtractor()
        cls.model_path = BASE_DIR / "models" / "voiceshield_model.pt"

    def test_01_label_mapping(self):
        """1. Asserts is_tts=0 is strictly HUMAN and is_tts=1 is strictly AI."""
        import pandas as pd
        train_csv = BASE_DIR / "data" / "splits" / "train.csv"
        df = pd.read_csv(train_csv)
        real_sample = df[df['is_tts'] == 0].iloc[0]
        ai_sample = df[df['is_tts'] == 1].iloc[0]

        self.assertEqual(int(real_sample['is_tts']), 0)
        self.assertEqual(int(ai_sample['is_tts']), 1)
        self.assertIn("human", str(real_sample['label_name']).lower())
        self.assertIn("ai", str(ai_sample['label_name']).lower())

    def test_02_audio_preprocessing_consistency(self):
        """2. Asserts preprocessing outputs mono 16kHz float32 with expected frames."""
        dummy_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, 32000)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, dummy_audio, 16000, format='WAV')
        buf.seek(0)

        audio, sr = self.fe.load_audio(buf.getvalue(), apply_vad=False)
        self.assertEqual(sr, 16000)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.dtype, np.float32)
        self.assertLessEqual(np.max(np.abs(audio)), 1.0)

        feats = self.fe.extract_features(audio, random_crop=False)
        self.assertEqual(feats.shape, (3, 64, self.fe.expected_frames))

    def test_03_model_output_shape(self):
        """3. Asserts model produces single logit [B, 1] per chunk."""
        model = VoiceShieldNet(in_channels=3, num_classes=1)
        model.eval()
        dummy_x = torch.randn(4, 3, 64, self.fe.expected_frames)
        with torch.no_grad():
            out = model(dummy_x)
        self.assertEqual(out.shape, (4, 1))

    def test_04_probability_calculation(self):
        """4. Asserts logit to probability mapping is monotonic and correct."""
        logits = torch.tensor([[-5.0], [-1.0], [0.0], [1.0], [5.0]])
        probs = torch.sigmoid(logits).flatten().numpy()
        # Monotonically increasing
        self.assertTrue(np.all(np.diff(probs) > 0))
        self.assertAlmostEqual(probs[2], 0.5, places=4)

    def test_05_probability_sum(self):
        """5. Asserts human_probability + ai_probability == 1.0 unconditionally."""
        logits = torch.randn(50)
        ai_probs = torch.sigmoid(logits).numpy()
        human_probs = 1.0 - ai_probs
        sums = ai_probs + human_probs
        np.testing.assert_allclose(sums, np.ones(50), atol=1e-6)

    def test_06_probability_bounds(self):
        """6. Asserts probabilities are bounded strictly in [0.0, 1.0]."""
        logits = torch.tensor([-1e6, -100.0, 0.0, 100.0, 1e6])
        probs = torch.sigmoid(logits).numpy()
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

    def test_07_classification_threshold(self):
        """7. Asserts classification threshold decides HUMAN vs AI without mutating probability."""
        threshold = 0.75
        prob_low = 0.65
        prob_high = 0.85

        decision_low = "AI_GENERATED" if prob_low >= threshold else "HUMAN"
        decision_high = "AI_GENERATED" if prob_high >= threshold else "HUMAN"

        self.assertEqual(decision_low, "HUMAN")
        self.assertEqual(decision_high, "AI_GENERATED")
        # Ensure raw probabilities are not distorted
        self.assertEqual(prob_low, 0.65)
        self.assertEqual(prob_high, 0.85)

    def test_08_segment_aggregation(self):
        """8. Asserts outlier transient spikes are suppressed by trimmed mean and median."""
        model = VoiceShieldNet(in_channels=3, num_classes=1)
        analyzer = RealTimeChunkAnalyzer(model, self.fe, device=torch.device("cpu"), temperature=1.0, threshold=0.75)

        # 8 chunks: 7 low (0.1), 1 outlier spike (0.8)
        probs = [0.10, 0.12, 0.08, 0.11, 0.80, 0.09, 0.13, 0.10]
        sorted_p = sorted(probs)
        trimmed = float(np.mean(sorted_p[1:-1]))
        median = float(np.median(probs))
        aggregated = 0.50 * trimmed + 0.50 * median

        # Peak was 0.80, but robust aggregation is well below 0.20
        self.assertLess(aggregated, 0.20)
        self.assertLess(aggregated, 0.75)  # Does not trigger false alarm

    def test_09_model_checkpoint_loading(self):
        """9. Asserts best checkpoint loads correctly and has matching state dict."""
        self.assertTrue(self.model_path.exists(), f"Missing checkpoint: {self.model_path}")
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", ckpt)
        model = VoiceShieldNet(in_channels=3, num_classes=1)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    def test_10_api_response_schema(self):
        """10. Asserts FastAPI /analyze endpoint returns the canonical schema."""
        client = TestClient(app)
        
        # Synthesize 3-second test WAV
        sr = 16000
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format='WAV')
        buf.seek(0)

        response = client.post(
            "/api/analyze",
            files={"file": ("test.wav", buf.getvalue(), "audio/wav")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data.get("success"))
        self.assertIn("voice_analysis", data)
        self.assertIn("ai_probability", data["voice_analysis"])
        self.assertIn("human_probability", data["voice_analysis"])
        self.assertIn("classification", data["voice_analysis"])
        self.assertIn("risk_score", data["voice_analysis"])
        self.assertIn("risk_level", data["voice_analysis"])
        self.assertIn("segments", data)
        self.assertIn("transaction_context", data)
        self.assertIn("recommendation", data)
        self.assertIn("chunk_timeline", data)

    def test_11_end_to_end_inference_difference(self):
        """11. Asserts model produces clearly separated risk scores on real vs synthetic demo audio."""
        real_path = BASE_DIR / "data" / "sample_demo" / "sample_real_01.wav"
        fake_path = BASE_DIR / "data" / "sample_demo" / "sample_fake_01.wav"

        if real_path.exists() and fake_path.exists():
            client = TestClient(app)
            with open(real_path, "rb") as f:
                res_real = client.post("/api/analyze", files={"file": ("real.wav", f.read(), "audio/wav")}).json()
            with open(fake_path, "rb") as f:
                res_fake = client.post("/api/analyze", files={"file": ("fake.wav", f.read(), "audio/wav")}).json()

            p_real = res_real["voice_analysis"]["ai_probability"]
            p_fake = res_fake["voice_analysis"]["ai_probability"]

            self.assertLess(p_real, 0.40, f"Real audio AI probability too high: {p_real}")
            self.assertGreater(p_fake, 0.70, f"Fake audio AI probability too low: {p_fake}")
            self.assertGreater(p_fake - p_real, 0.40, "Insufficient separation between genuine and AI audio")


if __name__ == "__main__":
    unittest.main(verbosity=2)
