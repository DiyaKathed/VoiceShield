import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from ml.features import AudioFeatureExtractor, load_audio
from ml.model import VoiceShieldNet

fe = AudioFeatureExtractor()
test_df = pd.read_csv("data/splits/test.csv")

m_old = VoiceShieldNet(in_channels=3, num_classes=1)
m_old.load_state_dict(torch.load("models/voiceshield_model_baseline_backup.pt", map_location="cpu", weights_only=False)["model_state_dict"])
m_old.eval()

m_new = VoiceShieldNet(in_channels=3, num_classes=1)
m_new.load_state_dict(torch.load("models/voiceshield_model_finetuned.pt", map_location="cpu", weights_only=False)["model_state_dict"])
m_new.eval()

def test_model(model, temp, thresh):
    y_true = []
    y_preds = []
    clone_true = []
    clone_preds = []
    tts_true = []
    tts_preds = []
    
    for _, row in test_df.iterrows():
        p = row["filepath"]
        audio, sr = load_audio(p, apply_vad=True)
        segs = fe.segment_audio(audio)
        if len(segs) == 0:
            pred = 0
        else:
            batches = [torch.from_numpy(fe.extract_features(s[2])).float() for s in segs]
            batch = torch.stack(batches)
            with torch.no_grad():
                raw_logits = model(batch).squeeze(-1).tolist()
                if isinstance(raw_logits, float): raw_logits = [raw_logits]
                scaled = [l / temp for l in raw_logits]
                probs = [1.0 / (1.0 + np.exp(-s)) for s in scaled]
                mean_p = float(np.mean(probs))
                pred = 1 if mean_p >= thresh else 0
            
        y_true.append(row["label"])
        y_preds.append(pred)
        
        is_c = ("clone" in str(row.get("dataset_source", "")).lower() or 
                "clone" in str(row["filename"]).lower() or 
                "synthetic_speaker" in str(row["filename"]).lower()) and row["label"] == 1
        if is_c:
            clone_true.append(1)
            clone_preds.append(pred)
            
        is_tts = row["label"] == 1 and not is_c
        if is_tts:
            tts_true.append(1)
            tts_preds.append(pred)
            
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    prec = float(precision_score(y_true, y_preds, pos_label=1, zero_division=0))
    ai_rec = float(recall_score(y_true, y_preds, pos_label=1, zero_division=0))
    hum_rec = float(recall_score(y_true, y_preds, pos_label=0, zero_division=0))
    f1 = float(f1_score(y_true, y_preds, zero_division=0))
    c_rec = float(recall_score(clone_true, clone_preds, pos_label=1, zero_division=0)) if clone_true else 0.0
    tts_rec = float(recall_score(tts_true, tts_preds, pos_label=1, zero_division=0)) if tts_true else 0.0
    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))
    
    return {
        "ai_recall": ai_rec,
        "ai_precision": prec,
        "human_recall": hum_rec,
        "f1": f1,
        "ai_fnr": fnr,
        "human_fpr": fpr,
        "clone_recall": c_rec,
        "tts_recall": tts_rec,
        "cm": [tn, fp, fn, tp],
        "total_clones": int(len(clone_true)),
        "correct_clones": int(sum(clone_preds)),
        "total_human": int(tn + fp),
        "correct_human": int(tn),
        "total_ai": int(tp + fn),
        "correct_ai": int(tp)
    }

old_res = test_model(m_old, temp=0.8927, thresh=0.5794)
new_res = test_model(m_new, temp=0.9822, thresh=0.45)

results = {
    "old_checkpoint": old_res,
    "new_checkpoint": new_res
}

with open("reports/checkpoint_comparison_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== COMPARATIVE TEST EVALUATION TABLE (REQUIREMENT 17) ===")
print(f"{'Metric':<28} | {'Old Checkpoint':^16} | {'New Checkpoint':^16}")
print("-" * 66)
print(f"{'AI Recall (All AI)':<28} | {old_res['ai_recall']*100:>15.2f}% | {new_res['ai_recall']*100:>15.2f}%")
print(f"{'AI Precision':<28} | {old_res['ai_precision']*100:>15.2f}% | {new_res['ai_precision']*100:>15.2f}%")
print(f"{'HUMAN Recall':<28} | {old_res['human_recall']*100:>15.2f}% | {new_res['human_recall']*100:>15.2f}%")
print(f"{'F1 Score':<28} | {old_res['f1']*100:>15.2f}% | {new_res['f1']*100:>15.2f}%")
print(f"{'AI False Negative Rate (FNR)':<28} | {old_res['ai_fnr']*100:>15.2f}% | {new_res['ai_fnr']*100:>15.2f}%")
print(f"{'HUMAN False Positive Rate':<28} | {old_res['human_fpr']*100:>15.2f}% | {new_res['human_fpr']*100:>15.2f}%")
print(f"{'TTS Voice Recall':<28} | {old_res['tts_recall']*100:>15.2f}% | {new_res['tts_recall']*100:>15.2f}%")
print(f"{'AI Clone Recall':<28} | {old_res['clone_recall']*100:>15.2f}% | {new_res['clone_recall']*100:>15.2f}%")
print("-" * 66)
print(f"Old Confusion Matrix (TN, FP, FN, TP): {old_res['cm']}")
print(f"New Confusion Matrix (TN, FP, FN, TP): {new_res['cm']}")
print(f"Old Clones Detected: {old_res['correct_clones']}/{old_res['total_clones']}")
print(f"New Clones Detected: {new_res['correct_clones']}/{new_res['total_clones']}")
