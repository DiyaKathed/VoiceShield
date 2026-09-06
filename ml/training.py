"""
VoiceShield Training Pipeline
=============================
Trains the VoiceShieldNet binary deepfake classifier on the Kaggle Fake and Real Audio Dataset.
Tracks Loss, Accuracy, Precision, Recall, F1 score.
Saves the best model checkpoint to models/voiceshield_model.pt
and training configuration to models/training_config.json.
"""

import os
import sys
import json
import yaml
import argparse
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.features import AudioFeatureExtractor
from ml.model import VoiceShieldNet
from ml.dataset import VoiceShieldDataset


MODELS_DIR = BASE_DIR / "models"
SPLITS_DIR = BASE_DIR / "data" / "splits"
CONFIG_FILE = BASE_DIR / "config" / "training.yaml"


def train_epoch(model, loader, optimizer, criterion, device, epoch: int = 1, total_epochs: int = 8):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    total_batches = len(loader)

    for b_idx, (features, labels) in enumerate(loader, 1):
        features = features.to(device)
        labels = labels.to(device)

        # Step 5 Audit: Verify AI data is present in training batches
        if epoch == 1 and b_idx <= 3:
            n_human = int((labels == 0).sum().item())
            n_ai = int((labels == 1).sum().item())
            print(f"  [Batch {b_idx} Dataloader Audit] Loaded: {n_human} HUMAN samples, {n_ai} AI samples (Total: {len(labels)})", flush=True)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().cpu().numpy()

        all_preds.extend(preds.flatten())
        all_targets.extend(labels.long().cpu().numpy().flatten())

        if b_idx % 15 == 0 or b_idx == total_batches:
            batch_loss = loss.item()
            running_acc = accuracy_score(all_targets, all_preds)
            print(f"  [Epoch {epoch}/{total_epochs} - Batch {b_idx}/{total_batches}] Batch Loss: {batch_loss:.4f} | Running Acc: {running_acc*100:.1f}%", flush=True)

    epoch_loss = total_loss / max(1, len(all_targets))
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    return epoch_loss, acc, prec, rec, f1


def validate_epoch(model, loader, criterion, device, clone_indices: Optional[List[int]] = None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * features.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy()

            all_logits.extend(logits.cpu().numpy().flatten())
            all_preds.extend(preds.flatten())
            all_targets.extend(labels.long().cpu().numpy().flatten())

    val_loss = total_loss / max(1, len(all_targets))
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    ai_rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    # Clone recall specifically on voice clone validation samples
    if clone_indices is not None and len(clone_indices) > 0:
        c_targets = [all_targets[i] for i in clone_indices]
        c_preds = [all_preds[i] for i in clone_indices]
        clone_rec = recall_score(c_targets, c_preds, zero_division=0)
    else:
        clone_rec = ai_rec

    # Detailed Confusion Matrix Metrics
    cm = confusion_matrix(all_targets, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / max(1, fp + tn)        # Human misclassified as AI
        fnr = fn / max(1, fn + tp)        # AI misclassified as Human
        human_rec = tn / max(1, tn + fp)  # Correct Human identification
        balanced_acc = 0.5 * (human_rec + ai_rec)
    else:
        fpr, fnr = 0.0, 0.0
        human_rec = 1.0
        balanced_acc = acc

    return val_loss, acc, prec, ai_rec, human_rec, clone_rec, balanced_acc, f1, fpr, fnr, np.array(all_logits), np.array(all_targets)


def run_training(
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 5e-4,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    augment: bool = True,
    output_model: str = str(MODELS_DIR / "voiceshield_model.pt"),
    seed: int = 42
):
    print("=" * 70)
    print("VoiceShield: Training Deepfake Classifier on Kaggle Audio Dataset")
    print("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Device selection: MPS -> CUDA -> CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[+] Compute Hardware Device: {device}")

    # Load splits
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Splits not found. Run scripts/prepare_kaggle_splits.py first.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print(f"[+] Total Available in Splits: Train={len(train_df):,}, Val={len(val_df):,}")

    # Enforce strictly balanced Real/AI sampling when capping
    if max_train_samples and max_train_samples < len(train_df):
        df_real = train_df[train_df['label'] == 0]
        df_ai = train_df[train_df['label'] == 1]
        half = max_train_samples // 2
        train_df = pd.concat([
            df_real.sample(n=min(half, len(df_real)), random_state=seed),
            df_ai.sample(n=min(half, len(df_ai)), random_state=seed)
        ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    val_cap = max_val_samples or (int(max_train_samples * 0.25) if max_train_samples else None)
    if val_cap and val_cap < len(val_df):
        v_real = val_df[val_df['label'] == 0]
        v_ai = val_df[val_df['label'] == 1]
        v_half = val_cap // 2
        val_df = pd.concat([
            v_real.sample(n=min(v_half, len(v_real)), random_state=seed),
            v_ai.sample(n=min(v_half, len(v_ai)), random_state=seed)
        ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    feature_extractor = AudioFeatureExtractor()
    train_dataset = VoiceShieldDataset(
        train_df,
        feature_extractor=feature_extractor,
        augment=augment
    )
    val_dataset = VoiceShieldDataset(
        val_df,
        feature_extractor=feature_extractor,
        augment=False
    )

    print(f"[+] Active Training Samples: {len(train_dataset)}")
    print(f"[+] Active Validation Samples: {len(val_dataset)}")

    # Preload all audio in RAM to avoid disk latency
    train_dataset.preload_audio()
    val_dataset.preload_audio()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = VoiceShieldNet(in_channels=3, num_classes=1)
    model.to(device)

    n_real = int((train_df['label'] == 0).sum())
    n_fake = int((train_df['label'] == 1).sum())
    pos_weight = torch.tensor([n_real / max(1, n_fake)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    val_clone_indices = [
        i for i, r in val_df.iterrows()
        if ('clone' in str(r.get('dataset_source', '')).lower() or 'clone' in str(r.get('filename', '')).lower() or 'synthetic_speaker' in str(r.get('filename', '')).lower())
        and r['label'] == 1
    ]

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_ai_rec = 0.0
    best_human_rec = 0.0
    best_clone_rec = 0.0
    best_val_fpr = 1.0
    best_epoch = 0
    best_val_logits = None
    best_val_targets = None

    print("\nStarting Training Execution...")
    print("-" * 105)
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^9} | {'Val Loss':^9} | {'Val Acc':^9} | {'AI Recall':^9} | {'Clone Rec':^9} | {'Human Rec':^9} | {'Val F1':^9}")
    print("-" * 105)

    history = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device, epoch=epoch, total_epochs=epochs)
        v_loss, v_acc, v_prec, v_ai_rec, v_human_rec, v_clone_rec, v_bal_acc, v_f1, v_fpr, v_fnr, v_logits, v_targets = validate_epoch(
            model, val_loader, criterion, device, clone_indices=val_clone_indices
        )
        scheduler.step()

        print(f"{epoch:^6} | {tr_loss:^10.4f} | {tr_acc*100:^8.1f}% | {v_loss:^9.4f} | {v_acc*100:^8.1f}% | {v_ai_rec*100:^8.1f}% | {v_clone_rec*100:^8.1f}% | {v_human_rec*100:^8.1f}% | {v_f1*100:^8.1f}%")

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(v_loss, 4),
            "val_acc": round(v_acc, 4),
            "val_ai_recall": round(v_ai_rec, 4),
            "val_clone_recall": round(v_clone_rec, 4),
            "val_human_recall": round(v_human_rec, 4),
            "val_balanced_acc": round(v_bal_acc, 4),
            "val_f1": round(v_f1, 4),
            "val_fpr": round(v_fpr, 4),
            "val_fnr": round(v_fnr, 4)
        })

        # Save best model checkpoint: prioritize low val loss while maintaining high balanced accuracy and clone recall
        is_better = (v_loss < best_val_loss) or (v_loss < best_val_loss * 1.05 and v_f1 > best_val_f1)
        if is_better and v_ai_rec >= 0.80 and v_human_rec >= 0.75 and v_clone_rec >= 0.80:
            best_val_loss = v_loss
            best_val_acc = v_acc
            best_ai_rec = v_ai_rec
            best_clone_rec = v_clone_rec
            best_human_rec = v_human_rec
            best_val_f1 = v_f1
            best_val_fpr = v_fpr
            best_epoch = epoch
            best_val_logits = v_logits
            best_val_targets = v_targets
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": v_loss,
                "val_acc": v_acc,
                "val_ai_recall": v_ai_rec,
                "val_clone_recall": v_clone_rec,
                "val_human_recall": v_human_rec,
                "val_f1": v_f1,
                "val_fpr": v_fpr,
                "dataset_name": "VoiceShield Enriched Anti-Spoofing Benchmark (Kaggle + Paired Speaker Clones)",
                "target_sr": 16000
            }, output_model)

    # Fallback save if strict threshold didn't trigger
    if best_epoch == 0:
        best_epoch = epochs
        best_val_loss = v_loss
        best_val_acc = v_acc
        best_ai_rec = v_ai_rec
        best_human_rec = v_human_rec
        best_val_f1 = v_f1
        best_val_fpr = v_fpr
        best_val_logits = v_logits
        best_val_targets = v_targets
        torch.save({
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": v_loss,
            "val_acc": v_acc,
            "val_ai_recall": v_ai_rec,
            "val_human_recall": v_human_rec,
            "val_f1": v_f1,
            "val_fpr": v_fpr,
            "dataset_name": "VoiceShield Enriched Anti-Spoofing Benchmark (Kaggle + Paired Speaker Clones)",
            "target_sr": 16000
        }, output_model)

    print("-" * 95)
    print(f"[✓] Best model checkpoint saved to: {output_model} (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc*100:.1f}%, AI Recall: {best_ai_rec*100:.1f}%, Human Recall: {best_human_rec*100:.1f}%, Val F1: {best_val_f1*100:.1f}%)")

    # Run temperature scaling and threshold calibration on validation set
    calibration_info = {}
    if best_val_logits is not None and len(best_val_logits) > 0:
        print("\n[+] Running Temperature Scaling & Threshold Calibration on Validation Data...")
        try:
            from ml.calibration import calibrate_model_validation
            calibration_info = calibrate_model_validation(best_val_logits, best_val_targets)
        except Exception as cal_err:
            print(f"[!] Calibration warning: {cal_err}")

    # Save training configuration
    config_data = {
        "dataset": "VoiceShield Enriched Anti-Spoofing Benchmark (Kaggle + Paired Speaker Clones)",
        "dataset_url": "https://www.kaggle.com/datasets/pawarrohitashok/fake-and-real-audio-dataset-deepfake-data",
        "model_architecture": "VoiceShieldNet (Spectro-Temporal Residual CNN)",
        "sample_rate": 16000,
        "n_mels": 64,
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 4),
        "best_val_f1": round(best_val_f1, 4),
        "best_val_fpr": round(best_val_fpr, 4),
        "batch_size": batch_size,
        "learning_rate": lr,
        "random_seed": seed,
        "split_method": "Stratified 70/15/15 Split (seed=42)",
        "augmentation_enabled": augment,
        "active_train_samples": len(train_dataset),
        "active_val_samples": len(val_dataset),
        "calibration": calibration_info,
        "training_history": history
    }

    config_path = MODELS_DIR / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"[✓] Saved training configuration to: {config_path}")

    return config_data


def main():
    parser = argparse.ArgumentParser(description="Train VoiceShield on Kaggle Audio Dataset")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Sample cap for fast smoke testing")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 100-sample smoke test")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--output", type=str, default=str(MODELS_DIR / "voiceshield_model.pt"), help="Output model path")
    args = parser.parse_args()

    max_train = 100 if args.smoke_test else args.max_samples
    max_val = 30 if args.smoke_test else None
    epochs = 3 if args.smoke_test else args.epochs

    run_training(
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=max_train,
        max_val_samples=max_val,
        augment=not args.no_augment,
        output_model=args.output
    )


if __name__ == "__main__":
    main()
