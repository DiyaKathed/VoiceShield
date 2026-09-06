"""
VoiceShield Comprehensive 12-Category Evaluation Matrix & Validation Benchmark
==============================================================================
1. Validation Threshold Sweep (0.30 - 0.70) on VALIDATION ONLY.
2. Streaming Aggregation Benchmark (Mean, Median, Top-k, Percentile, Weighted) on VALIDATION ONLY.
3. 12-Category Evaluation Matrix on HELD-OUT TEST DATA:
   A. Genuine HUMAN
   B. AI / TTS
   C. AI Voice Clone
   D. Noisy HUMAN
   E. Compressed HUMAN
   F. Microphone HUMAN
   G. Noisy AI
   H. Compressed AI
   I. Microphone AI
   J. AI Clone + Microphone
   K. AI Clone + Compression
   L. AI Clone + Noise / Reverb
4. Comparative Analysis: Old Checkpoint vs Fine-Tuned Checkpoint.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfilt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.features import TARGET_SR, load_audio, AudioFeatureExtractor
from ml.model import VoiceShieldNet

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def load_model(ckpt_path: Path, device: torch.device = torch.device("cpu")) -> VoiceShieldNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = VoiceShieldNet(in_channels=3, num_classes=1)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device)
    m.eval()
    return m


def evaluate_audio_sample(model, fe, audio, temp=1.0, thresh=0.50, aggregation="weighted"):
    """Runs sliding-window inference and applies chosen aggregation."""
    segs = fe.segment_audio(audio, overlap=0.5)
    if len(segs) == 0:
        return 0.0, 0.0, [-3.0], 0
        
    batches = [torch.from_numpy(fe.extract_features(s[2])).float() for s in segs]
    batch = torch.stack(batches)
    
    with torch.no_grad():
        raw_logits = model(batch).squeeze(-1).tolist()
        if isinstance(raw_logits, float):
            raw_logits = [raw_logits]
            
        scaled_logits = [l / max(0.01, temp) for l in raw_logits]
        probs = [float(1.0 / (1.0 + np.exp(-s))) for s in scaled_logits]
        
    p_arr = np.array(probs)
    
    if aggregation == "mean":
        agg_prob = float(np.mean(p_arr))
    elif aggregation == "median":
        agg_prob = float(np.median(p_arr))
    elif aggregation == "top_k":
        k = max(1, len(p_arr) // 2)
        agg_prob = float(np.mean(np.sort(p_arr)[-k:]))
    elif aggregation == "p80":
        agg_prob = float(np.percentile(p_arr, 80))
    elif aggregation == "weighted":
        ai_chunks = [p for p in probs if p >= thresh]
        p80 = float(np.percentile(p_arr, 80))
        mean_p = float(np.mean(p_arr))
        median_p = float(np.median(p_arr))
        
        has_repeated_ai = (len(ai_chunks) >= 2 and len(ai_chunks) >= len(probs) * 0.30)
        if has_repeated_ai:
            top_ai_mean = float(np.mean(sorted(ai_chunks, reverse=True)[:max(2, len(ai_chunks))]))
            agg_prob = float(0.60 * top_ai_mean + 0.25 * p80 + 0.15 * mean_p)
        elif len(ai_chunks) == 1 and len(probs) >= 3:
            sorted_p = sorted(probs)
            agg_prob = float(0.60 * np.mean(sorted_p[:-1]) + 0.40 * median_p)
        else:
            agg_prob = float(0.50 * mean_p + 0.50 * median_p)
    else:
        agg_prob = float(np.mean(p_arr))
        
    pred = 1 if agg_prob >= thresh else 0
    return agg_prob, 1.0 - agg_prob, raw_logits, pred


def run_validation_sweep(model, fe, val_df, temp=1.0):
    """Evaluates threshold range strictly on validation data."""
    print("=" * 80)
    print("STEP 1: VALIDATION SET THRESHOLD SWEEP (REQUIREMENT 11)")
    print("=" * 80)
    
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    # Pre-extract probabilities on validation set
    y_true = []
    y_probs = []
    is_clone = []
    
    for _, row in val_df.iterrows():
        p = Path(row["filepath"])
        if not p.exists():
            continue
        audio, sr = load_audio(str(p), apply_vad=True)
        prob, _, _, _ = evaluate_audio_sample(model, fe, audio, temp=temp, thresh=0.50, aggregation="weighted")
        y_true.append(row["label"])
        y_probs.append(prob)
        src = str(row.get("dataset_source", "")).lower()
        fn = str(row.get("filename", "")).lower()
        is_clone.append("clone" in src or "clone" in fn or "synthetic_speaker" in fn)
        
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    is_clone = np.array(is_clone)
    
    print(f"{'Threshold':^10} | {'AI Recall':^11} | {'Clone Recall':^14} | {'Human Rec':^11} | {'Human FPR':^11} | {'F1 Score':^10} | {'Youden J':^10}")
    print("-" * 88)
    
    best_thresh = 0.50
    best_j = -1.0
    
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        ai_rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
        hum_rec = recall_score(y_true, preds, pos_label=0, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / max(1, fp + tn)
        j = ai_rec - fpr
        
        # Clone recall
        clone_mask = (y_true == 1) & is_clone
        c_rec = recall_score(y_true[clone_mask], preds[clone_mask], pos_label=1, zero_division=0) if clone_mask.sum() > 0 else ai_rec
        
        if j > best_j:
            best_j = j
            best_thresh = t
            
        print(f"{t:^10.2f} | {ai_rec*100:^9.1f}% | {c_rec*100:^12.1f}% | {hum_rec*100:^9.1f}% | {fpr*100:^9.1f}% | {f1*100:^8.1f}% | {j:^10.4f}")
        
    print(f"\n[✓] Optimal Validation-Derived Threshold: {best_thresh:.2f} (Max Youden J: {best_j:.4f})")
    return best_thresh


def run_aggregation_benchmark(model, fe, val_df, temp=1.0, thresh=0.55):
    """Compares aggregation algorithms on validation set (Requirement 15)."""
    print("\n" + "=" * 80)
    print("STEP 2: STREAMING AGGREGATION BENCHMARK ON VALIDATION SET (REQUIREMENT 15)")
    print("=" * 80)
    
    methods = ["mean", "median", "top_k", "p80", "weighted"]
    
    for m in methods:
        y_true = []
        y_preds = []
        y_probs = []
        
        for _, row in val_df.iterrows():
            p = Path(row["filepath"])
            if not p.exists(): continue
            audio, sr = load_audio(str(p), apply_vad=True)
            prob, _, _, pred = evaluate_audio_sample(model, fe, audio, temp=temp, thresh=thresh, aggregation=m)
            y_true.append(row["label"])
            y_preds.append(pred)
            y_probs.append(prob)
            
        ai_rec = recall_score(y_true, y_preds, pos_label=1, zero_division=0)
        hum_rec = recall_score(y_true, y_preds, pos_label=0, zero_division=0)
        f1 = f1_score(y_true, y_preds, zero_division=0)
        cm = confusion_matrix(y_true, y_preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / max(1, fp + tn)
        
        print(f"  Method: {m:<10} | AI Recall: {ai_rec*100:5.1f}% | Human Rec: {hum_rec*100:5.1f}% | FPR: {fpr*100:5.1f}% | F1: {f1*100:5.1f}%")


def evaluate_12_categories(model, fe, test_df, temp=1.0, thresh=0.55, model_name="Model"):
    """
    Evaluates 12 specific operational categories on the held-out test split (Requirement 13).
    A. Genuine HUMAN
    B. AI/TTS
    C. AI Voice Clone
    D. Noisy HUMAN
    E. Compressed HUMAN
    F. Microphone HUMAN
    G. Noisy AI
    H. Compressed AI
    I. Microphone AI
    J. AI clone + microphone
    K. AI clone + compression
    L. AI clone + noise/reverb
    """
    print("\n" + "=" * 110)
    print(f"12-CATEGORY EVALUATION MATRIX: {model_name} (Threshold={thresh:.2f}, Temp={temp:.4f})")
    print("=" * 110)
    
    # Categorization rules
    categories = {
        "A. Genuine HUMAN":           lambda r: r["label"] == 0 and "hard" not in str(r.get("dataset_source", "")),
        "B. AI / TTS":                lambda r: r["label"] == 1 and ("kaggle" in str(r.get("dataset_source", "")) or "multilingual" in str(r.get("dataset_source", ""))) and "clone" not in str(r.get("filename", "")),
        "C. AI Voice Clone":          lambda r: r["label"] == 1 and ("clone" in str(r.get("dataset_source", "")) or "clone" in str(r.get("filename", "")) or "synthetic_speaker" in str(r.get("filename", ""))) and "hard" not in str(r.get("dataset_source", "")),
        "D. Noisy HUMAN":             lambda r: r["label"] == 0 and "noise" in str(r.get("dataset_source", "")),
        "E. Compressed HUMAN":        lambda r: r["label"] == 0 and "compression" in str(r.get("dataset_source", "")),
        "F. Microphone HUMAN":        lambda r: r["label"] == 0 and "mic" in str(r.get("dataset_source", "")),
        "G. Noisy AI":                lambda r: r["label"] == 1 and "noise" in str(r.get("dataset_source", "")),
        "H. Compressed AI":           lambda r: r["label"] == 1 and "compression" in str(r.get("dataset_source", "")),
        "I. Microphone AI":           lambda r: r["label"] == 1 and "mic" in str(r.get("dataset_source", "")),
        "J. AI Clone + Microphone":   lambda r: r["label"] == 1 and ("clone" in str(r.get("filename", "")) or "mic" in str(r.get("dataset_source", ""))),
        "K. AI Clone + Compression":  lambda r: r["label"] == 1 and ("clone" in str(r.get("filename", "")) or "compression" in str(r.get("dataset_source", ""))),
        "L. AI Clone + Noise/Reverb": lambda r: r["label"] == 1 and ("reverb" in str(r.get("dataset_source", "")) or "noise" in str(r.get("dataset_source", "")))
    }
    
    # Also evaluate sample_demo files
    demo_files = [
        (Path("data/sample_demo/sample_real_01.wav"), 0, "A. Genuine HUMAN"),
        (Path("data/sample_demo/sample_real_02.wav"), 0, "A. Genuine HUMAN"),
        (Path("data/sample_demo/sample_fake_01.wav"), 1, "B. AI / TTS"),
        (Path("data/sample_demo/sample_fake_02.wav"), 1, "B. AI / TTS"),
        (Path("data/sample_demo/sample_ai_cloned_fraud.wav"), 1, "C. AI Voice Clone"),
    ]
    
    matrix_results = {}
    
    print(f"{'Category':<28} | {'Samples':^7} | {'Correct':^7} | {'Incorrect':^9} | {'AI Rec':^8} | {'Hum Rec':^8} | {'FPR':^6} | {'FNR':^6} | {'Mean AI':^8} | {'Med AI':^8}")
    print("-" * 115)
    
    for cat_name, filter_fn in categories.items():
        sub_df = test_df[test_df.apply(filter_fn, axis=1)]
        
        y_true = []
        y_preds = []
        ai_probs = []
        
        for _, row in sub_df.iterrows():
            p = Path(row["filepath"])
            if not p.exists(): continue
            audio, sr = load_audio(str(p), apply_vad=True)
            prob, _, _, pred = evaluate_audio_sample(model, fe, audio, temp=temp, thresh=thresh, aggregation="weighted")
            y_true.append(row["label"])
            y_preds.append(pred)
            ai_probs.append(prob)
            
        # Add demo samples matching this category
        for dp, d_lbl, d_cat in demo_files:
            if d_cat == cat_name and dp.exists():
                audio, sr = load_audio(str(dp), apply_vad=True)
                prob, _, _, pred = evaluate_audio_sample(model, fe, audio, temp=temp, thresh=thresh, aggregation="weighted")
                y_true.append(d_lbl)
                y_preds.append(pred)
                ai_probs.append(prob)
                
        n_samples = len(y_true)
        if n_samples == 0:
            continue
            
        correct = sum(1 for yt, yp in zip(y_true, y_preds) if yt == yp)
        incorrect = n_samples - correct
        
        has_ai = any(yt == 1 for yt in y_true)
        has_hum = any(yt == 0 for yt in y_true)
        
        ai_rec = recall_score(y_true, y_preds, pos_label=1, zero_division=0) if has_ai else 1.0
        hum_rec = recall_score(y_true, y_preds, pos_label=0, zero_division=0) if has_hum else 1.0
        
        fpr = (sum(1 for yt, yp in zip(y_true, y_preds) if yt == 0 and yp == 1) / max(1, sum(1 for yt in y_true if yt == 0))) if has_hum else 0.0
        fnr = (sum(1 for yt, yp in zip(y_true, y_preds) if yt == 1 and yp == 0) / max(1, sum(1 for yt in y_true if yt == 1))) if has_ai else 0.0
        
        mean_ai = float(np.mean(ai_probs))
        med_ai = float(np.median(ai_probs))
        
        ai_rec_str = f"{ai_rec*100:5.1f}%" if has_ai else "N/A"
        hum_rec_str = f"{hum_rec*100:5.1f}%" if has_hum else "N/A"
        fpr_str = f"{fpr*100:4.1f}%" if has_hum else "0.0%"
        fnr_str = f"{fnr*100:4.1f}%" if has_ai else "0.0%"
        
        print(f"{cat_name:<28} | {n_samples:^7} | {correct:^7} | {incorrect:^9} | {ai_rec_str:^8} | {hum_rec_str:^8} | {fpr_str:^6} | {fnr_str:^6} | {mean_ai*100:^7.1f}% | {med_ai*100:^7.1f}%")
        
        matrix_results[cat_name] = {
            "samples": n_samples,
            "correct": correct,
            "incorrect": incorrect,
            "ai_recall": ai_rec if has_ai else None,
            "human_recall": hum_rec if has_hum else None,
            "fpr": fpr,
            "fnr": fnr,
            "mean_ai_prob": round(mean_ai, 4),
            "median_ai_prob": round(med_ai, 4)
        }
        
    return matrix_results


def main():
    fe = AudioFeatureExtractor()
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    
    old_ckpt = MODELS_DIR / "voiceshield_model_baseline_backup.pt"
    new_ckpt = MODELS_DIR / "voiceshield_model_finetuned.pt"
    
    print("Loading checkpoints for comparative evaluation...")
    m_old = load_model(old_ckpt)
    m_new = load_model(new_ckpt)
    
    # 1. Validation Threshold Sweep on New Model
    best_thresh_new = run_validation_sweep(m_new, fe, val_df, temp=0.9822)
    
    # 2. Aggregation Benchmark on New Model
    run_aggregation_benchmark(m_new, fe, val_df, temp=0.9822, thresh=best_thresh_new)
    
    # 3. 12-Category Evaluation Matrix on Old Model
    res_old = evaluate_12_categories(m_old, fe, test_df, temp=0.8927, thresh=0.5794, model_name="BASELINE MODEL (OLD)")
    
    # 4. 12-Category Evaluation Matrix on New Model
    res_new = evaluate_12_categories(m_new, fe, test_df, temp=0.9822, thresh=best_thresh_new, model_name="FINE-TUNED MODEL (NEW)")
    
    # Save results to json for reporting
    out_json = BASE_DIR / "reports" / "12_category_matrix_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "optimal_threshold_new": best_thresh_new,
            "baseline_model": res_old,
            "finetuned_model": res_new
        }, f, indent=2)
    print(f"\n[✓] Saved complete 12-category matrix evaluation to: {out_json}")


if __name__ == "__main__":
    main()
