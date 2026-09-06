"""
VoiceShield Evaluation Pipeline
===============================
Rigorously evaluates the trained VoiceShieldNet on the held-out Kaggle test split.
Calculates authentic metrics:
- Overall: Accuracy, Precision, Recall, F1-Score, ROC-AUC, EER, Confusion Matrix
Saves full report to reports/evaluation.json and models/eval_metrics.json.
Generates evaluation visual plots (ROC Curve, Confusion Matrix).
"""

import os
import sys
import io
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from ml.dataset import VoiceShieldDataset


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
    model_path: str = str(MODELS_DIR / "voiceshield_model.pt"),
    test_manifest: str = str(SPLITS_DIR / "test.csv"),
    output_report: str = str(REPORTS_DIR / "evaluation.json"),
    max_samples: int = None
):
    print("=" * 75, flush=True)
    print("VoiceShield: Rigorous Test Set Evaluation on Kaggle Deepfake Dataset", flush=True)
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
    print(f"[+] Found {len(test_df):,} held-out test samples", flush=True)

    feature_extractor = AudioFeatureExtractor()
    test_dataset = VoiceShieldDataset(
        test_df,
        feature_extractor=feature_extractor,
        augment=False,
        max_samples=max_samples
    )
    test_dataset.preload_audio()
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    y_true = []
    y_probs = []
    y_preds = []

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
    specificity = float(tn / max(1, tn + fp))

    results = {
        "dataset": "pawarrohitashok/fake-and-real-audio-dataset-deepfake-data",
        "dataset_url": "https://www.kaggle.com/datasets/pawarrohitashok/fake-and-real-audio-dataset-deepfake-data",
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
            "specificity": round(specificity, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "eer": round(eer, 4),
            "eer_threshold": round(eer_threshold, 4),
            "confusion_matrix": {
                "true_negatives_human": int(tn),
                "false_positives_ai_alarm": int(fp),
                "false_negatives_missed_ai": int(fn),
                "true_positives_ai_detected": int(tp)
            }
        }
    }

    # Save to reports/evaluation.json
    with open(output_report, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Saved comprehensive evaluation report to: {output_report}", flush=True)

    # Save to models/eval_metrics.json for backend / API consumption
    eval_metrics_path = MODELS_DIR / "eval_metrics.json"
    with open(eval_metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Saved backend metrics to: {eval_metrics_path}", flush=True)

    # Generate Confusion Matrix visualization
    try:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, f"{z:,}", ha="center", va="center", fontsize=14, fontweight="bold")
        fig.colorbar(cax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Genuine Human (0)", "AI Cloned (1)"], fontsize=10)
        ax.set_yticklabels(["Genuine Human (0)", "AI Cloned (1)"], fontsize=10)
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=10)
        ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
        ax.set_title(f"VoiceShield Confusion Matrix\n(Accuracy: {acc*100:.1f}%, F1: {f1*100:.1f}%)", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        cm_path = REPORTS_DIR / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"[✓] Saved confusion matrix figure to: {cm_path}")
    except Exception as err:
        print(f"[!] Warning generating confusion matrix plot: {err}")

    # Generate ROC Curve visualization
    try:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_probs)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.plot(fpr_curve, tpr_curve, color="#2563eb", lw=2.5, label=f"VoiceShieldNet (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
        ax.scatter([eer], [1 - eer], color="#dc2626", s=60, zorder=5, label=f"EER = {eer*100:.1f}%")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FAR)", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate (1 - FRR)", fontsize=11, fontweight="bold")
        ax.set_title("VoiceShield ROC Curve (Held-out Test Split)", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = REPORTS_DIR / "roc_curve.png"
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"[✓] Saved ROC curve figure to: {roc_path}")
    except Exception as err:
        print(f"[!] Warning generating ROC curve plot: {err}")

    # Print Summary
    print("\n" + "=" * 75)
    print("VOICESHIELD FINAL EVALUATION REPORT")
    print("=" * 75)
    print(f"  Test Samples Evaluated:  {len(y_true)} (Real: {int(np.sum(y_true==0))}, AI: {int(np.sum(y_true==1))})")
    print(f"  Accuracy:                {acc * 100:.2f}%")
    print(f"  Precision:               {prec * 100:.2f}%")
    print(f"  Recall (Sensitivity):    {rec * 100:.2f}%")
    print(f"  Specificity:             {specificity * 100:.2f}%")
    print(f"  F1 Score:                {f1 * 100:.2f}%")
    if roc_auc is not None:
        print(f"  ROC-AUC:                 {roc_auc:.4f}")
    print(f"  Equal Error Rate (EER):  {eer * 100:.2f}% (Threshold: {eer_threshold:.4f})")
    print(f"  False Positive Rate:     {fpr * 100:.2f}%")
    print(f"  Confusion Matrix:        TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=" * 75 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate VoiceShield on Kaggle Test Split")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "voiceshield_model.pt"), help="Model checkpoint path")
    parser.add_argument("--manifest", type=str, default=str(SPLITS_DIR / "test.csv"), help="Test manifest CSV")
    parser.add_argument("--output", type=str, default=str(REPORTS_DIR / "evaluation.json"), help="Output report JSON")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap test samples")
    args = parser.parse_args()

    evaluate_test_split(
        model_path=args.model,
        test_manifest=args.manifest,
        output_report=args.output,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
