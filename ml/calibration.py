"""
VoiceShield Probability Calibration & Operating Threshold Module
================================================================
Performs:
1. Temperature Scaling on validation logits to calibrate output probabilities
   without altering ROC-AUC or ranking.
2. Optimal Decision Threshold Selection via Youden's J statistic (TPR - FPR)
   and Equal Error Rate (EER) operating point derived strictly from validation data.
3. Serialization of calibration parameters (T, optimal_threshold) to models/calibration_config.json.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.metrics import roc_curve, brier_score_loss

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


class TemperatureScaler(nn.Module):
    """
    Learns a single temperature parameter T > 0 on validation logits
    to calibrate output probabilities: p = sigmoid(logit / T).
    Parameterizes using log_temperature to guarantee strictly positive T.
    """

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).clamp(min=0.05, max=10.0).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        t = torch.exp(self.log_temperature).clamp(min=0.05, max=10.0)
        return logits / t

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray, max_iter: int = 100) -> float:
        """
        Fits temperature T on validation logits using BCEWithLogitsLoss.
        """
        logits_tensor = torch.tensor(val_logits, dtype=torch.float32).flatten()
        labels_tensor = torch.tensor(val_labels, dtype=torch.float32).flatten()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.log_temperature], lr=0.05, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_tensor)
            loss = criterion(scaled_logits, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self.temperature


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes optimal decision threshold using Youden's J statistic:
    J = TPR - FPR = Recall - FPR.
    Returns: (optimal_threshold, eer_threshold, max_youden_j)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[best_idx])
    max_j = float(j_scores[best_idx])

    # EER threshold
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer_threshold = float(thresholds[eer_idx])

    # Keep threshold bounded within realistic limits [0.35, 0.75]
    clamped_threshold = float(np.clip(optimal_threshold, 0.35, 0.75))

    return clamped_threshold, eer_threshold, max_j


def calibrate_model_validation(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    save_path: str = str(MODELS_DIR / "calibration_config.json")
) -> Dict[str, Any]:
    """
    Full calibration pipeline executed on validation split:
    1. Learns optimal temperature T
    2. Calculates Brier score before and after calibration
    3. Finds Youden's J threshold
    4. Saves calibration configuration
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Raw probabilities
    raw_probs = 1.0 / (1.0 + np.exp(-val_logits))
    brier_before = float(brier_score_loss(val_labels, raw_probs))

    # Fit temperature
    scaler = TemperatureScaler()
    temperature = scaler.fit(val_logits, val_labels)

    # Calibrated probabilities
    calibrated_logits = val_logits / temperature
    calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
    brier_after = float(brier_score_loss(val_labels, calibrated_probs))

    # Select threshold on validation set
    optimal_threshold, eer_threshold, youden_j = find_optimal_threshold(val_labels, calibrated_probs)

    calibration_data = {
        "temperature": round(temperature, 4),
        "brier_score_raw": round(brier_before, 4),
        "brier_score_calibrated": round(brier_after, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "eer_threshold": round(eer_threshold, 4),
        "youden_j_score": round(youden_j, 4),
        "default_threshold": 0.50
    }

    with open(save_path, "w") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"[✓] Probability Calibration Complete:")
    print(f"    Temperature T          : {temperature:.4f}")
    print(f"    Brier Score (Raw)      : {brier_before:.4f} -> Calibrated: {brier_after:.4f}")
    print(f"    Optimal Threshold (J)  : {optimal_threshold:.4f} (EER Thresh: {eer_threshold:.4f})")
    print(f"    Saved configuration to : {save_path}")

    return calibration_data


def load_calibration_config() -> Dict[str, Any]:
    """Loads calibration config if available, or returns sensible defaults."""
    config_file = MODELS_DIR / "calibration_config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            return json.load(f)
    return {
        "temperature": 1.0,
        "optimal_threshold": 0.50,
        "eer_threshold": 0.50
    }
