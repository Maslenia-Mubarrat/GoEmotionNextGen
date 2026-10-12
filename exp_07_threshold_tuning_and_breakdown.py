import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

DATA_DIR = "./emogenx_artifacts"
FIGURES_DIR = "./emogenx_artifacts/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from exp_05_train_emogenx_core import EmoGenXArchitecture, EmoGenXDataset

with open(os.path.join(DATA_DIR, "label_names.json"), "r") as f:
    label_names = json.load(f)

val_df = pd.read_parquet(os.path.join(DATA_DIR, "goemotions_val.parquet"))
test_df = pd.read_parquet(os.path.join(DATA_DIR, "goemotions_test.parquet"))
adj = torch.tensor(np.load(os.path.join(DATA_DIR, "adj_matrix.npy")), dtype=torch.float).to(device)

model = EmoGenXArchitecture(num_labels=len(label_names)).to(device)
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "emogenx_final_model.pt"), map_location=device))
model.eval()

val_loader = DataLoader(EmoGenXDataset(val_df), batch_size=32, shuffle=False)
test_loader = DataLoader(EmoGenXDataset(test_df), batch_size=32, shuffle=False)

def get_predictions(loader):
    p, t = [], []
    with torch.no_grad():
        for b in loader:
            logits, _ = model(b['input_ids'].to(device), b['attention_mask'].to(device), adj)
            p.append(torch.sigmoid(logits).cpu().numpy())
            t.append(b['labels'].numpy())
    return np.vstack(p), np.vstack(t)

val_p, val_t = get_predictions(val_loader)
test_p, test_t = get_predictions(test_loader)

# Calibrate per-class threshold
optimal_thresholds = {}
for i, name in enumerate(label_names):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        score = f1_score(val_t[:, i], (val_p[:, i] >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    optimal_thresholds[name] = best_t

thresh_arr = np.array([optimal_thresholds[name] for name in label_names])
final_test_preds = (test_p >= thresh_arr).astype(int)

macro_f1 = f1_score(test_t, final_test_preds, average='macro', zero_division=0)
micro_f1 = f1_score(test_t, final_test_preds, average='micro', zero_division=0)

# Class-wise F1 comparisons
class_f1_scores = f1_score(test_t, final_test_preds, average=None, zero_division=0)
with open(os.path.join(DATA_DIR, "optimal_thresholds.json"), "w") as f:
    json.dump(optimal_thresholds, f, indent=4)

# Generate Class-Wise Performance Plot
plt.figure(figsize=(14, 6))
indices = np.arange(len(label_names))
plt.bar(indices, class_f1_scores, color="#0ea5e9", edgecolor="none")
plt.xticks(indices, label_names, rotation=90)
plt.title(f"Per-Class F1 Profile under EmoGenX (Macro-F1: {macro_f1:.4f})")
plt.ylabel("F1-Score")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_per_class_f1.png"), dpi=300)
plt.close()

print(f"[SUCCESS] exp_07 completed. Calibrated Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f}")