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

from ml.features import AudioFeatureExtractor, load_audio, inspect_audio, is_sufficient_speech
from ml.model import VoiceShieldNet
from backend.risk_engine import VoiceRiskEngine, ContextualRiskEngine, TransactionContext
from backend.chunk_analyzer import RealTimeChunkAnalyzer


# Paths
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DEMO_DIR = DATA_DIR / "sample_demo"
MODEL_PATH = MODELS_DIR / "voiceshield_model.pt"
METRICS_PATH = MODELS_DIR / "eval_metrics.json"

app = FastAPI(
    title="VoiceShield AI Core API",
    description="Real-Time Detection & Prevention of Voice Cloning Impersonation Attacks",
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
    print("=" * 65)
    print("Initializing VoiceShield Local ML Inference Engine...")

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
        
        # Section 9: Verified model checkpoint startup report
        print(f"[✓] VoiceShieldNet loaded successfully on {device}")
        print(f"Model checkpoint       : {active_weights}")
        print(f"Sample rate            : 16000 Hz (mono float32)")
        print(f"Feature configuration  : 64-band Log-Mel + Delta (velocity) + Delta-Delta (accel)")
        print(f"Class mapping          : 0 = HUMAN, 1 = AI_GENERATED")
        print(f"Calibrated Temperature : T = {cal_temp:.4f}")
        print(f"Operating Threshold    : {cal_thresh:.4f}")
    else:
        print(f"[!] Warning: Model weights not found at {active_weights}. Run ml/training.py first.")
        state["model_loaded"] = False
    print("=" * 65)


@app.get("/api/health")
def health_check():
    """Returns server and local ML engine status."""
    return {
        "status": "online",
        "service": "VoiceShield Core Deepfake Prevention Engine",
        "model_loaded": state["model_loaded"],
        "device": str(state["device"]),
        "model_weights": str(MODEL_PATH)
    }


@app.get("/model-info")
@app.get("/api/model-info")
def get_model_info():
    """Returns official model and dataset metadata for VoiceShield."""
    return {
        "model_name": "VoiceShield Net",
        "dataset": "Kaggle Fake and Real Audio Dataset (pawarrohitashok/fake-and-real-audio-dataset-deepfake-data)",
        "sample_rate": 16000,
        "classification": "Human vs AI-generated speech",
        "architecture": "VoiceShieldNet (Spectro-Temporal Residual CNN with Delta Features)",
        "device": str(state["device"])
    }


@app.get("/api/evaluation-metrics")
@app.get("/api/metrics")
@app.get("/metrics")
def get_evaluation_metrics():
    """Returns authentic test evaluation metrics calculated from unseen test split."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        
        result = dict(metrics)
        # Flatten overall fields into root level for frontend components
        overall = metrics.get("overall", {})
        if isinstance(overall, dict):
            for k, v in overall.items():
                if k not in result:
                    result[k] = v
        
        # Ensure confusion matrix supports both naming styles
        cm = result.get("confusion_matrix", {})
        if isinstance(cm, dict):
            cm_normalized = {
                "true_negatives": cm.get("true_negatives_human", cm.get("true_negatives", 0)),
                "false_positives": cm.get("false_positives_ai_alarm", cm.get("false_positives", 0)),
                "false_negatives": cm.get("false_negatives_missed_ai", cm.get("false_negatives", 0)),
                "true_positives": cm.get("true_positives_ai_detected", cm.get("true_positives", 0)),
                **cm
            }
            result["confusion_matrix"] = cm_normalized
            if "overall" in result and isinstance(result["overall"], dict):
                result["overall"]["confusion_matrix"] = cm_normalized

        return result
    return {"error": "Evaluation metrics not found. Run scripts/evaluation.py."}


@app.get("/api/samples")
def list_demo_samples():
    """Returns available demo audio samples from the Kaggle dataset."""
    samples = [
        {
            "id": "sample_real_01.wav",
            "title": "Authentic Human Voice Sample 1",
            "type": "GENUINE HUMAN",
            "language": "Natural Speech",
            "expected_risk": "LOW",
            "description": "Authentic human speech recording with natural acoustic vocal tract resonances.",
            "caller_name": "Operations Lead",
            "caller_role": "Operations Manager",
            "amount": 45000.0,
            "urgency": "Normal",
            "speaker_verification": "Verified Enrolled"
        },
        {
            "id": "sample_fake_01.wav",
            "title": "AI Deepfake Voice Clone 1",
            "type": "AI-GENERATED / CLONE",
            "language": "Synthetic AI",
            "expected_risk": "HIGH",
            "description": "Synthetic neural voice generation with phase inconsistencies in formant frequencies.",
            "caller_name": "Finance Director",
            "caller_role": "Managing Director",
            "amount": 2500000.0,
            "urgency": "Immediate",
            "speaker_verification": "Mismatch / Failed"
        },
        {
            "id": "sample_real_02.wav",
            "title": "Authentic Human Voice Sample 2",
            "type": "GENUINE HUMAN",
            "language": "Natural Speech",
            "expected_risk": "LOW",
            "description": "Legitimate human vocal recording discussing scheduled operational review.",
            "caller_name": "Regional Supervisor",
            "caller_role": "Branch Manager",
            "amount": 80000.0,
            "urgency": "Normal",
            "speaker_verification": "Verified Enrolled"
        },
        {
            "id": "sample_fake_02.wav",
            "title": "AI Deepfake Voice Clone 2",
            "type": "AI-GENERATED / CLONE",
            "language": "Synthetic AI",
            "expected_risk": "HIGH",
            "description": "Synthetic voice generation simulating executive voice for emergency wire transfer redirection.",
            "caller_name": "Treasury Executive",
            "caller_role": "Treasury Head",
            "amount": 4200000.0,
            "urgency": "Immediate",
            "speaker_verification": "Mismatch / Failed"
        },
        {
            "id": "sample_spliced_attack.wav",
            "title": "Spliced Partial-Spoof Attack",
            "type": "SPLICED / PARTIAL SPOOF",
            "language": "Composite",
            "expected_risk": "HIGH",
            "description": "Real human greeting concatenated with an AI-cloned fraudulent wire transfer directive.",
            "caller_name": "Accounts Desk",
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
    file_path = DEMO_DIR / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="audio/wav")

    raise HTTPException(status_code=404, detail=f"Audio sample '{filename}' not found")


@app.get("/audio-info")
@app.get("/api/audio-info")
async def get_audio_info(sample: Optional[str] = None):
    """
    Diagnostic Audio Inspection Endpoint (GET):
    Inspects stream properties of an existing sample audio file.
    Reports original format, original rate, channels, decoded rate, RMS, and speech duration.
    """
    target_path = None
    if sample:
        for p in [DEMO_DIR / sample, DATA_DIR / sample]:
            if p.exists():
                target_path = p
                break
        if target_path is None:
            raise HTTPException(status_code=404, detail=f"Sample file '{sample}' not found.")
    else:
        # Default to first available demo sample
        first_sample = next(DEMO_DIR.glob("*.wav"), None)
        if first_sample:
            target_path = first_sample
        else:
            raise HTTPException(status_code=404, detail="No audio sample specified or found.")

    try:
        info = inspect_audio(str(target_path), filename=target_path.name)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect audio: {str(e)}")


@app.post("/audio-info")
@app.post("/api/audio-info")
async def post_audio_info(file: UploadFile = File(...)):
    """
    Diagnostic Audio Inspection Endpoint (POST):
    Inspects stream properties of an uploaded MP3 or WAV file.
    Reports original format, rate, channels, decoded 16kHz PCM properties, and speech sufficiency.
    """
    filename = file.filename or "uploaded_audio"
    ext = Path(filename).suffix.lower()
    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".webm", ".aac", ".m4a"}
    if ext and ext not in valid_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format '{ext}'. Please upload MP3 or WAV.")

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file received")
        info = inspect_audio(content, filename=filename)
        return info
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio inspection failed: {str(e)}")


@app.post("/analyze")
@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    caller_name: Optional[str] = Form("Executive Caller"),
    caller_role: Optional[str] = Form("Chief Executive Officer"),
    amount: Optional[float] = Form(75000.0),
    urgency: Optional[str] = Form("Immediate"),
    speaker_verification: Optional[str] = Form("Unregistered / Unknown"),
    is_live_recording: Optional[str] = Form(None)
):
    """
    Main Deepfake Detection Endpoint:
    Accepts .wav and .mp3 audio, decodes to mono 16kHz PCM waveform,
    executes sliding-window inference through VoiceShieldNet,
    and returns probability, risk scores, and temporal timeline.
    """
    if not state["model_loaded"]:
        # Attempt lazy reload if weights were generated
        load_ml_pipeline()
        if not state["model_loaded"]:
            raise HTTPException(status_code=503, detail="ML model is not loaded. Train the model first.")

    filename = file.filename or "uploaded_audio.wav"
    ext = Path(filename).suffix.lower()
    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".webm", ".aac", ".m4a"}
    if ext and ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Please upload MP3 or WAV."
        )

    # Determine if this stream is a live microphone recording
    is_mic = (
        (is_live_recording is not None and str(is_live_recording).lower() in ("true", "1", "yes"))
        or filename.startswith("live_recording")
    )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file received")

        # Load & standardize audio through central loader (transparent MP3/WAV support)
        try:
            audio, sr = load_audio(content, filename=filename, apply_vad=True)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            raise HTTPException(status_code=400, detail=str(re))

        # Check speech energy and perform sliding-window analysis
        chunk_analysis = state["chunk_analyzer"].analyze_audio_stream(
            audio, sr, is_live_recording=is_mic
        )

        audio_format = ext.replace(".", "") if ext else "wav"
        if not audio_format or audio_format == "blob":
            audio_format = "wav"

        # Handle Insufficient Speech / Silence
        if chunk_analysis.get("is_insufficient_speech", False):
            return {
                "success": True,
                "filename": filename,
                "status": "INSUFFICIENT_SPEECH",
                "duration_sec": chunk_analysis["total_duration"],
                "total_chunks": 0,
                "partial_spoof_detected": False,
                # Phase 18 Spec
                "voice_analysis": {
                    "ai_probability": 0.0,
                    "human_probability": 0.0,
                    "classification": "INSUFFICIENT_SPEECH",
                    "confidence": 0.0,
                    "risk_score": 0.0,
                    "risk_level": "INSUFFICIENT_SPEECH"
                },
                "segments": [],
                "audio": {
                    "format": audio_format,
                    "sample_rate": sr,
                    "duration": chunk_analysis["total_duration"]
                },
                "recommendation": "Insufficient speech detected for reliable analysis. Ensure recording contains audible spoken words.",
                # Frontend Direct Compatibility
                "ai_generated_probability": 0.0,
                "human_probability": 0.0,
                "classification": "INSUFFICIENT_SPEECH",
                "confidence": 0.0,
                "voice_risk_score": 0.0,
                "voice_risk_status": "INSUFFICIENT_SPEECH",
                "voice_risk_color": "#94A3B8",
                "transaction_context": {
                    "caller_role": caller_role or "CEO / Executive",
                    "caller_name": caller_name or "Executive Caller",
                    "amount": amount if amount is not None else 75000.0,
                    "urgency": urgency or "Immediate",
                    "speaker_verification": speaker_verification or "Unregistered / Unknown"
                },
                "contextual_risk_score": 0.0,
                "threat_level": "INSUFFICIENT_SPEECH",
                "security_alert": "Insufficient speech detected for reliable analysis. Please provide a clear recording containing human speech.",
                "recommended_action": "Insufficient speech detected for reliable analysis. Ensure microphone is active and audio contains clear spoken speech before re-analyzing.",
                "action_code": "INSUFFICIENT_SPEECH",
                "chunk_timeline": []
            }

        # Global probability calculated via robust statistical aggregation (trimmed mean / median)
        global_ai_prob = float(chunk_analysis.get("aggregated_ai_probability", 0.5))
        global_human_prob = round(1.0 - global_ai_prob, 4)
        confidence = round(max(global_ai_prob, global_human_prob), 4)
        opt_thresh = float(chunk_analysis.get("operating_threshold", 0.4876))

        # 1. Voice ML Risk Engine (strictly calibrated decision boundary)
        voice_risk = VoiceRiskEngine.calculate_voice_risk(global_ai_prob, threshold=opt_thresh)

        # 2. Contextual High-Risk Transaction Engine
        context = TransactionContext(
            caller_name=caller_name or "Executive Caller",
            caller_role=caller_role or "Chief Executive Officer",
            amount=amount if amount is not None else 75000.0,
            urgency=urgency or "Immediate",
            speaker_verification=speaker_verification or "Unregistered / Unknown"
        )
        context_risk = ContextualRiskEngine.evaluate(global_ai_prob, context, threshold=opt_thresh)

        # Terminal-level diagnostic logging
        raw_logits = chunk_analysis.get("raw_logits", [])
        min_logit = min(raw_logits) if raw_logits else 0.0
        max_logit = max(raw_logits) if raw_logits else 0.0
        c_probs = [c["ai_probability"] for c in chunk_analysis.get("chunks", [])]
        min_cp = min(c_probs) if c_probs else 0.0
        max_cp = max(c_probs) if c_probs else 0.0

        print("\n" + "=" * 70, flush=True)
        print(f"[VoiceShield Inference Diagnostics]", flush=True)
        print(f"Audio Source       : {filename} ({'Microphone Recording' if is_mic else 'File Upload'})", flush=True)
        print(f"Sample Rate        : {sr} Hz | Channels: 1 (Mono Float32)", flush=True)
        print(f"Duration           : {chunk_analysis['total_duration']:.2f}s (Speech: {chunk_analysis.get('speech_duration', chunk_analysis['total_duration']):.2f}s)", flush=True)
        print(f"Waveform Shape     : {audio.shape} | Peak: {float(np.max(np.abs(audio))):.4f} | RMS: {float(np.sqrt(np.mean(audio**2))):.4f}", flush=True)
        print(f"Total Chunks       : {chunk_analysis['total_chunks']}", flush=True)
        print(f"Raw Model Logits   : [{min_logit:.4f}, {max_logit:.4f}]", flush=True)
        print(f"Chunk AI Probs     : [{min_cp:.4f}, {max_cp:.4f}]", flush=True)
        print(f"Aggregated AI Prob : {global_ai_prob:.4f} ({global_ai_prob * 100:.1f}%)", flush=True)
        print(f"Aggregated Human   : {global_human_prob:.4f} ({global_human_prob * 100:.1f}%)", flush=True)
        print(f"Operating Threshold: {opt_thresh:.4f}", flush=True)
        print(f"Repeated AI Clones : {chunk_analysis.get('has_repeated_ai', False)}", flush=True)
        print(f"Partial Spoof Flag : {chunk_analysis.get('partial_spoof_detected', False)}", flush=True)
        print(f"Final Classification: {voice_risk['classification']} [{voice_risk['status']}]", flush=True)
        print("=" * 70 + "\n", flush=True)

        # Segments formatted for both timeline display and Section 18 spec
        segments_list = [
            {
                "start": round(c["start_time"], 2),
                "end": round(c["end_time"], 2),
                "ai_probability": round(c["ai_probability"], 4),
                "human_probability": round(c["human_probability"], 4),
                "classification": c.get("classification", "AI_GENERATED" if c["ai_probability"] >= opt_thresh else "HUMAN")
            }
            for c in chunk_analysis["chunks"]
        ]

        return {
            "success": True,
            "filename": filename,
            "duration_sec": chunk_analysis["total_duration"],
            "total_chunks": chunk_analysis["total_chunks"],
            "partial_spoof_detected": chunk_analysis["partial_spoof_detected"],
            "operating_threshold": opt_thresh,
            # Section 18 Spec Format
            "voice_analysis": {
                "ai_probability": round(global_ai_prob, 4),
                "human_probability": round(global_human_prob, 4),
                "classification": voice_risk["classification"],
                "confidence": confidence,
                "risk_score": voice_risk["voice_risk_score"],
                "risk_level": voice_risk["status"],
                "operating_threshold": opt_thresh
            },
            "segments": segments_list,
            "audio": {
                "format": audio_format,
                "sample_rate": sr,
                "duration": chunk_analysis["total_duration"]
            },
            "recommendation": context_risk["recommended_action"],
            # React Frontend Direct Format
            "ai_probability": round(global_ai_prob, 4),
            "ai_generated_probability": round(global_ai_prob, 4),
            "human_probability": round(global_human_prob, 4),
            "classification": voice_risk["classification"],
            "confidence": confidence,
            "voice_risk_score": voice_risk["voice_risk_score"],
            "voice_risk_status": voice_risk["status"],
            "voice_risk_color": voice_risk["color"],
            "transaction_context": context.model_dump(),
            "contextual_risk_score": context_risk["contextual_risk_score"],
            "threat_level": context_risk["threat_level"],
            "security_alert": context_risk["security_alert"],
            "recommended_action": context_risk["recommended_action"],
            "action_code": context_risk["action_code"],
            "chunk_timeline": chunk_analysis["chunks"],
            # Forensic Debug Payload
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
                "partial_spoof_detected": chunk_analysis["partial_spoof_detected"],
                "has_repeated_ai": chunk_analysis.get("has_repeated_ai", False),
                "operating_threshold": opt_thresh
            }
        }

    except HTTPException:
        raise
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
