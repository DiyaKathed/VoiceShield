"""
VoiceShield IndicTTS Evaluation Pipeline
=========================================
Rigorously evaluates the trained VoiceShieldNet on the held-out IndicTTS test split.
Calculates authentic metrics:
- Overall: Accuracy, Precision, Recall, F1-Score, ROC-AUC, EER, Confusion Matrix
- Language-wise breakdown: Independent evaluation across all 16 Indian languages
Saves full report to reports/evaluation.json and models/eval_metrics.json.
"""

import os
import sys
import io
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.features import AudioFeatureExtractor
from ml.model import VoiceShieldNet
from ml.dataset import IndicTTSDataset


MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SPLITS_DIR = BASE_DIR / "data" / "splits"


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Computes Equal Error Rate (EER) where FAR equals FRR.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, eer_threshold


def evaluate_test_split(
    model_path: str = str(MODELS_DIR / "voiceshield_indictts_best.pt"),
    test_manifest: str = str(SPLITS_DIR / "test.csv"),
    output_report: str = str(REPORTS_DIR / "evaluation.json"),
    max_samples: int = None
):
    print("=" * 75, flush=True)
    print("VoiceShield: Rigorous IndicTTS Test Set Evaluation & Language Analysis", flush=True)
    print("=" * 75, flush=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[+] Compute Hardware Device: {device}", flush=True)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = VoiceShieldNet(in_channels=3, num_classes=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[+] Loaded Model Checkpoint: {model_path}", flush=True)

    # Load calibration parameters if available
    cal_config = MODELS_DIR / "calibration_config.json"
    temperature = 1.0
    optimal_threshold = 0.50
    if cal_config.exists():
        with open(cal_config, "r") as f:
            c_data = json.load(f)
            temperature = float(c_data.get("temperature", 1.0))
            optimal_threshold = float(c_data.get("optimal_threshold", 0.50))
    print(f"[+] Calibration Loaded: Temperature T={temperature:.4f}, Operating Threshold={optimal_threshold:.4f}", flush=True)

    # Load test dataset
    test_df = pd.read_csv(test_manifest)
    print(f"[+] Found {len(test_df):,} held-out test samples across {test_df['language'].nunique()} Indic languages", flush=True)

    feature_extractor = AudioFeatureExtractor()
    test_dataset = IndicTTSDataset(
        test_df,
        feature_extractor=feature_extractor,
        augment=False,
        max_samples=max_samples
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    y_true = []
    y_probs = []
    y_preds = []
    languages = test_dataset.df['language'].tolist()

    print(f"[+] Running inference across {len(test_dataset)} test samples...", flush=True)
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            # Apply temperature scaling
            scaled_logits = logits / max(0.01, temperature)
            probs = torch.sigmoid(scaled_logits).squeeze(-1).cpu().numpy()
            if probs.ndim == 0:
                probs = np.array([probs.item()])
            preds = (probs >= optimal_threshold).astype(int)

            y_probs.extend(probs.tolist())
            y_preds.extend(preds.tolist())
            y_true.extend(labels.squeeze(-1).cpu().numpy().astype(int).tolist())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_preds = np.array(y_preds)
    languages = np.array(languages[:len(y_true)])

    # 1. Overall Metrics
    acc = float(accuracy_score(y_true, y_preds))
    prec = float(precision_score(y_true, y_preds, zero_division=0))
    rec = float(recall_score(y_true, y_preds, zero_division=0))
    f1 = float(f1_score(y_true, y_preds, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = None

    eer, eer_threshold = compute_eer(y_true, y_probs)
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))

    # 2. Language-wise breakdown
    by_language = {}
    for lang in np.unique(languages):
        idx = (languages == lang)
        sub_true = y_true[idx]
        sub_probs = y_probs[idx]
        sub_preds = y_preds[idx]

        sub_n = len(sub_true)
        sub_real = int(np.sum(sub_true == 0))
        sub_ai = int(np.sum(sub_true == 1))

        if sub_n >= 2 and len(np.unique(sub_true)) == 2:
            sub_acc = float(accuracy_score(sub_true, sub_preds))
            sub_prec = float(precision_score(sub_true, sub_preds, zero_division=0))
            sub_rec = float(recall_score(sub_true, sub_preds, zero_division=0))
            sub_f1 = float(f1_score(sub_true, sub_preds, zero_division=0))
            sub_eer, sub_thresh = compute_eer(sub_true, sub_probs)
            sub_cm = confusion_matrix(sub_true, sub_preds)
            s_tn, s_fp, s_fn, s_tp = sub_cm.ravel() if sub_cm.shape == (2, 2) else (0, 0, 0, 0)
            sub_fpr = float(s_fp / max(1, s_fp + s_tn))
            sub_fnr = float(s_fn / max(1, s_fn + s_tp))
            try:
                sub_auc = float(roc_auc_score(sub_true, sub_probs))
            except Exception:
                sub_auc = None
        else:
            sub_acc = float(accuracy_score(sub_true, sub_preds))
            sub_prec = 0.0
            sub_rec = 0.0
            sub_f1 = 0.0
            sub_eer = 0.0
            sub_auc = None
            sub_fpr = 0.0
            sub_fnr = 0.0

        by_language[lang] = {
            "samples": sub_n,
            "real_samples": sub_real,
            "ai_samples": sub_ai,
            "accuracy": round(sub_acc, 4),
            "precision": round(sub_prec, 4),
            "recall": round(sub_rec, 4),
            "f1_score": round(sub_f1, 4),
            "false_positive_rate": round(sub_fpr, 4),
            "false_negative_rate": round(sub_fnr, 4),
            "roc_auc": round(sub_auc, 4) if sub_auc is not None else None,
            "eer": round(sub_eer, 4),
            "sufficient_test_samples": sub_n >= 10
        }

    # 3. Dedicated Real-World Human & AI Test Set Evaluation
    real_human_dir = BASE_DIR / "data" / "real_world_human"
    real_world_human_results = []
    if real_human_dir.exists():
        for audio_f in sorted(real_human_dir.glob("*.wav")):
            try:
                audio_arr, sr = feature_extractor.load_audio(str(audio_f), apply_vad=True)
                feat = feature_extractor.extract_features(audio_arr, random_crop=False)
                with torch.no_grad():
                    lgt = model(torch.from_numpy(feat).unsqueeze(0).to(device))
                    p = torch.sigmoid(lgt / max(0.01, temperature)).item()
                real_world_human_results.append({
                    "file": audio_f.name,
                    "ai_prob": round(p, 4),
                    "classified_as_ai": p >= optimal_threshold
                })
            except Exception:
                pass

    real_human_fp_count = sum(r["classified_as_ai"] for r in real_world_human_results)
    real_human_fpr = float(real_human_fp_count / max(1, len(real_world_human_results)))

    results = {
        "dataset": "SherryT997/IndicTTS-Deepfake-Challenge-Data",
        "model_path": str(model_path),
        "calibration": {
            "temperature": round(temperature, 4),
            "optimal_threshold": round(optimal_threshold, 4)
        },
        "overall": {
            "num_test_samples": len(y_true),
            "real_samples": int(np.sum(y_true == 0)),
            "ai_samples": int(np.sum(y_true == 1)),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "eer": round(eer, 4),
            "eer_threshold": round(eer_threshold, 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            }
        },
        "real_world_human_eval": {
            "total_files": len(real_world_human_results),
            "false_positives": real_human_fp_count,
            "false_positive_rate": round(real_human_fpr, 4),
            "files": real_world_human_results
        },
        "by_language": by_language
    }

    # Save to reports/evaluation.json and models/eval_metrics.json
    with open(output_report, "w") as f:
        json.dump(results, f, indent=2)

    # models/eval_metrics.json compatibility format
    eval_metrics_compat = {
        "dataset": "IndicTTS-Deepfake-Challenge-Data (Multilingual Indic)",
        "num_test_samples": len(y_true),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "eer": round(eer, 4),
        "eer_threshold": round(eer_threshold, 4),
        "confusion_matrix": results["overall"]["confusion_matrix"],
        "by_language": by_language
    }
    with open(MODELS_DIR / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics_compat, f, indent=2)

    # Print Formatted Report
    print("\n" + "-" * 75, flush=True)
    print("               VOICESHIELD INDICTTS EVALUATION REPORT", flush=True)
    print("-" * 75, flush=True)
    print(f"  Test Samples Evaluated : {len(y_true)} (Held-out Disjoint Speakers)", flush=True)
    print(f"  Overall Accuracy       : {acc*100:.2f}%", flush=True)
    print(f"  Overall Precision      : {prec*100:.2f}%", flush=True)
    print(f"  Overall Recall         : {rec*100:.2f}%", flush=True)
    print(f"  Overall F1-Score       : {f1*100:.2f}%", flush=True)
    print(f"  Overall ROC-AUC        : {roc_auc:.4f}" if roc_auc else "  Overall ROC-AUC        : N/A", flush=True)
    print(f"  Equal Error Rate (EER) : {eer*100:.2f}% (Threshold: {eer_threshold:.4f})", flush=True)
    print("-" * 75, flush=True)
    print("  Confusion Matrix:")
    print(f"                     Predicted Human (0)   Predicted AI Fake (1)")
    print(f"    Actual Human (0)        {tn:^10}             {fp:^10}")
    print(f"    Actual Fake (1)         {fn:^10}             {tp:^10}")
    print("-" * 75, flush=True)
    print("  Language-Wise Performance Breakdown:")
    print(f"    {'Language':<14} | {'Samples':^8} | {'Accuracy':^9} | {'F1-Score':^9} | {'EER':^8}")
    print("    " + "-" * 60)
    for lang, metrics in sorted(by_language.items()):
        print(f"    {lang:<14} | {metrics['samples']:^8} | {metrics['accuracy']*100:^8.1f}% | {metrics['f1_score']*100:^8.1f}% | {metrics['eer']*100:^7.1f}%")
    print("-" * 75, flush=True)
    print(f"[✓] Evaluation report saved to: {output_report} and {MODELS_DIR / 'eval_metrics.json'}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate VoiceShield on IndicTTS Test Set")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "voiceshield_indictts_best.pt"), help="Model checkpoint path")
    parser.add_argument("--manifest", type=str, default=str(SPLITS_DIR / "test.csv"), help="Test manifest CSV")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples to evaluate")
    parser.add_argument("--output", type=str, default=str(REPORTS_DIR / "evaluation.json"), help="Output JSON path")
    args = parser.parse_args()

    evaluate_test_split(args.model, args.manifest, args.output, args.max_samples)


if __name__ == "__main__":
    main()
