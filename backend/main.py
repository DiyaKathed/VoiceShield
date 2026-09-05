"""
VoiceShield FastAPI Backend
===========================
High-performance REST API service providing locally running PyTorch voice deepfake
detection, real-time chunk timeline analysis, dual-layer risk calculation, and demo assets.
"""

import os
import sys
import io
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import torch
import soundfile as sf
import numpy as np

# Project root path setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.features import AudioFeatureExtractor
from ml.model import VoiceShieldNet
from backend.risk_engine import VoiceRiskEngine, ContextualRiskEngine, TransactionContext
from backend.chunk_analyzer import RealTimeChunkAnalyzer


# Paths
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DEMO_DIR = DATA_DIR / "sample_demo"
INDIC_DEMO_DIR = DATA_DIR / "sample_demo" / "indic"
MODEL_PATH = MODELS_DIR / "voiceshield_indictts_best.pt" if (MODELS_DIR / "voiceshield_indictts_best.pt").exists() else (MODELS_DIR / "voiceshield_model.pt")
METRICS_PATH = MODELS_DIR / "eval_metrics.json"

app = FastAPI(
    title="VoiceShield AI IndicTTS Core API",
    description="Real-Time Detection & Prevention of Voice Cloning Impersonation Attacks across Indic Languages",
    version="2.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
state = {
    "model": None,
    "feature_extractor": None,
    "chunk_analyzer": None,
    "device": torch.device("cpu"),
    "model_loaded": False
}


@app.on_event("startup")
def load_ml_pipeline():
    """Initializes the local PyTorch model on startup."""
    print("=" * 60)
    print("Initializing VoiceShield IndicTTS Local ML Inference Engine...")

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    state["device"] = device
    feature_extractor = AudioFeatureExtractor()
    state["feature_extractor"] = feature_extractor

    active_weights = MODEL_PATH if MODEL_PATH.exists() else (MODELS_DIR / "voiceshield_model.pt")
    if active_weights.exists():
        checkpoint = torch.load(active_weights, map_location=device, weights_only=False)
        model = VoiceShieldNet(in_channels=3, num_classes=1)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        state["model"] = model

        # Load calibration parameters
        cal_file = MODELS_DIR / "calibration_config.json"
        cal_temp, cal_thresh = 1.0, 0.50
        if cal_file.exists():
            try:
                with open(cal_file, "r") as f:
                    c_data = json.load(f)
                    cal_temp = float(c_data.get("temperature", 1.0))
                    cal_thresh = float(c_data.get("optimal_threshold", 0.50))
            except Exception:
                pass

        state["chunk_analyzer"] = RealTimeChunkAnalyzer(
            model, feature_extractor, device=device, temperature=cal_temp, threshold=cal_thresh
        )
        state["model_loaded"] = True
        print(f"[✓] VoiceShieldNet loaded successfully on {device} (Weights: {active_weights})")
        print(f"[✓] Active Calibration: Temperature={cal_temp:.4f}, Operating Threshold={cal_thresh:.4f}")
    else:
        print(f"[!] Warning: Model weights not found at {active_weights}. Run ml/training.py first.")
        state["model_loaded"] = False
    print("=" * 60)


@app.get("/api/health")
def health_check():
    """Returns server and local ML engine status."""
    return {
        "status": "online",
        "service": "VoiceShield IndicTTS Deepfake Prevention Engine",
        "model_loaded": state["model_loaded"],
        "device": str(state["device"]),
        "model_weights": str(MODEL_PATH)
    }


@app.get("/model-info")
@app.get("/api/model-info")
def get_model_info():
    """Returns official model and dataset metadata for VoiceShield IndicTTS."""
    return {
        "model_name": "VoiceShield IndicTTS",
        "dataset": "IndicTTS Deepfake Challenge Dataset (SherryT997/IndicTTS-Deepfake-Challenge-Data)",
        "languages": [
            "Assamese", "Bengali", "Bodo", "Dogri", "English (Indian)", "Gujarati",
            "Hindi", "Kannada", "Malayalam", "Manipuri", "Marathi",
            "Nepali", "Odia", "Sanskrit", "Tamil", "Telugu"
        ],
        "total_languages": 16,
        "sample_rate": 16000,
        "classification": "Human vs AI-generated speech",
        "architecture": "VoiceShieldNet (Spectro-Temporal Residual CNN with Delta Features)",
        "evaluation_split": "Held-Out Disjoint Speakers (Zero Speaker Leakage)",
        "device": str(state["device"])
    }


@app.get("/api/evaluation-metrics")
def get_evaluation_metrics():
    """Returns authentic test evaluation metrics calculated from unseen test split."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return metrics
    return {"error": "Evaluation metrics not found. Run ml/evaluation.py."}


@app.get("/api/samples")
def list_demo_samples():
    """Returns available 1-click demo audio samples across Indic languages and scenarios."""
    samples = [
        {
            "id": "indic_hindi_synthetic_clone.wav",
            "title": "AI-Cloned Hindi Wire Fraud",
            "type": "AI-GENERATED / CLONE",
            "language": "Hindi",
            "expected_risk": "HIGH",
            "description": "Synthetic IndicTTS voice clone impersonating the Managing Director demanding emergency wire authorization.",
            "caller_name": "Vikram Singhania",
            "caller_role": "Managing Director",
            "amount": 5000000.0,
            "urgency": "Immediate",
            "speaker_verification": "Mismatch / Failed"
        },
        {
            "id": "indic_hindi_genuine.wav",
            "title": "Legitimate Hindi Speech",
            "type": "GENUINE HUMAN",
            "language": "Hindi",
            "expected_risk": "LOW",
            "description": "Natural human voice from Mumbai regional desk discussing scheduled operational audit.",
            "caller_name": "Rajesh Sharma",
            "caller_role": "Operations Manager",
            "amount": 150000.0,
            "urgency": "Normal",
            "speaker_verification": "Verified Enrolled"
        },
        {
            "id": "indic_marathi_synthetic_clone.wav",
            "title": "AI-Cloned Marathi Impersonation",
            "type": "AI-GENERATED / CLONE",
            "language": "Marathi",
            "expected_risk": "HIGH",
            "description": "Synthetic Marathi cloned speech attempting treasury fund redirection.",
            "caller_name": "Anand Deshmukh",
            "caller_role": "Treasury Head",
            "amount": 2500000.0,
            "urgency": "Immediate",
            "speaker_verification": "Mismatch / Failed"
        },
        {
            "id": "indic_marathi_genuine.wav",
            "title": "Legitimate Marathi Voice",
            "type": "GENUINE HUMAN",
            "language": "Marathi",
            "expected_risk": "LOW",
            "description": "Authentic human Marathi speech confirming commercial branch verification.",
            "caller_name": "Pooja Patil",
            "caller_role": "Branch Manager",
            "amount": 50000.0,
            "urgency": "Normal",
            "speaker_verification": "Verified Enrolled"
        },
        {
            "id": "indic_tamil_genuine.wav",
            "title": "Legitimate Tamil Voice",
            "type": "GENUINE HUMAN",
            "language": "Tamil",
            "expected_risk": "LOW",
            "description": "Authentic human Tamil speech during scheduled vendor clearance review.",
            "caller_name": "Karthik Subramanian",
            "caller_role": "Auditor",
            "amount": 80000.0,
            "urgency": "Normal",
            "speaker_verification": "Verified Enrolled"
        },
        {
            "id": "indic_spliced_attack.wav",
            "title": "Spliced Indic Partial-Spoof Attack",
            "type": "SPLICED / PARTIAL SPOOF",
            "language": "Hindi / Indic",
            "expected_risk": "HIGH",
            "description": "Real Hindi greeting spliced with an AI-cloned fraudulent wire transfer directive.",
            "caller_name": "Finance Desk Mumbai",
            "caller_role": "Senior Accountant",
            "amount": 1800000.0,
            "urgency": "Urgent",
            "speaker_verification": "Unregistered / Unknown"
        }
    ]
    return {"samples": samples}


@app.api_route("/api/sample-audio/{filename}", methods=["GET", "HEAD"])
def stream_sample_audio(filename: str):
    """Streams demo audio WAV files for in-browser playback."""
    # Check indic demo dir first, then fallback to general demo dir
    indic_path = INDIC_DEMO_DIR / filename
    if indic_path.exists():
        return FileResponse(indic_path, media_type="audio/wav")

    file_path = DEMO_DIR / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="audio/wav")

    raise HTTPException(status_code=404, detail=f"Audio sample '{filename}' not found")


@app.post("/analyze")
@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    caller_name: Optional[str] = Form("Executive Caller"),
    caller_role: Optional[str] = Form("Chief Executive Officer"),
    amount: Optional[float] = Form(75000.0),
    urgency: Optional[str] = Form("Immediate"),
    speaker_verification: Optional[str] = Form("Unregistered / Unknown")
):
    """
    Main Deepfake Detection Endpoint:
    Processes uploaded/recorded audio, performs sliding-window inference,
    computes voice risk and contextual threat scores.
    """
    if not state["model_loaded"]:
        # Attempt lazy reload if weights were generated
        load_ml_pipeline()
        if not state["model_loaded"]:
            raise HTTPException(status_code=503, detail="ML model is not loaded. Train the model first.")

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file received")

        # Load & standardize audio
        audio, sr = state["feature_extractor"].load_audio(content, apply_vad=True)

        # Segment-level chunk analysis
        chunk_analysis = state["chunk_analyzer"].analyze_audio_stream(audio, sr)

        # Global probability calculated via robust statistical aggregation (trimmed mean / median)
        global_ai_prob = float(chunk_analysis.get("aggregated_ai_probability", 0.5))
        global_human_prob = round(1.0 - global_ai_prob, 4)
        confidence = round(max(global_ai_prob, global_human_prob), 4)

        # 1. Voice ML Risk Engine
        voice_risk = VoiceRiskEngine.calculate_voice_risk(global_ai_prob)

        # 2. Contextual High-Risk Transaction Engine
        context = TransactionContext(
            caller_name=caller_name or "Executive Caller",
            caller_role=caller_role or "Chief Executive Officer",
            amount=amount if amount is not None else 75000.0,
            urgency=urgency or "Immediate",
            speaker_verification=speaker_verification or "Unregistered / Unknown"
        )
        context_risk = ContextualRiskEngine.evaluate(global_ai_prob, context)

        # Segments formatted for both timeline display and Phase 13 spec
        segments_list = [
            {
                "start": round(c["start_time"], 2),
                "end": round(c["end_time"], 2),
                "ai_probability": round(c["ai_probability"], 4),
                "human_probability": round(c["human_probability"], 4),
                "classification": c.get("classification", "AI_GENERATED" if c["ai_probability"] >= 0.5 else "HUMAN")
            }
            for c in chunk_analysis["chunks"]
        ]

        return {
            "success": True,
            "filename": file.filename,
            "duration_sec": chunk_analysis["total_duration"],
            "total_chunks": chunk_analysis["total_chunks"],
            "partial_spoof_detected": chunk_analysis["partial_spoof_detected"],
            # Phase 13 Spec Format
            "voice_analysis": {
                "ai_probability": round(global_ai_prob, 4),
                "human_probability": round(global_human_prob, 4),
                "classification": voice_risk["classification"],
                "confidence": confidence,
                "risk_score": voice_risk["voice_risk_score"],
                "risk_level": voice_risk["status"]
            },
            "segments": segments_list,
            "recommendation": context_risk["recommended_action"],
            # React Frontend Direct Format
            "ai_generated_probability": round(global_ai_prob, 4),
            "human_probability": round(global_human_prob, 4),
            "classification": voice_risk["classification"],
            "confidence": confidence,
            "voice_risk_score": voice_risk["voice_risk_score"],
            "voice_risk_status": voice_risk["status"],
            "voice_risk_color": voice_risk["color"],
            "transaction_context": context.dict(),
            "contextual_risk_score": context_risk["contextual_risk_score"],
            "threat_level": context_risk["threat_level"],
            "security_alert": context_risk["security_alert"],
            "recommended_action": context_risk["recommended_action"],
            "action_code": context_risk["action_code"],
            "chunk_timeline": chunk_analysis["chunks"],
            # Phase 19 Developer & Forensic Debug Payload
            "debug": {
                "audio_duration_sec": chunk_analysis["total_duration"],
                "sample_rate": sr,
                "total_segments": chunk_analysis["total_chunks"],
                "min_ai_probability": chunk_analysis.get("min_ai_probability", 0.0),
                "max_ai_probability": chunk_analysis.get("max_ai_probability", 0.0),
                "mean_ai_probability": chunk_analysis.get("mean_ai_probability", 0.0),
                "median_ai_probability": chunk_analysis.get("median_ai_probability", 0.0),
                "trimmed_mean_ai_probability": chunk_analysis.get("trimmed_mean_ai_probability", 0.0),
                "aggregated_ai_probability": round(global_ai_prob, 4),
                "partial_spoof_detected": chunk_analysis["partial_spoof_detected"]
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/api/reload-model")
def reload_model():
    """Dynamically reloads the latest model weights from disk without restarting."""
    load_ml_pipeline()
    return {
        "reloaded": True,
        "model_loaded": state["model_loaded"],
        "device": str(state["device"]),
        "model_path": str(MODEL_PATH)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
