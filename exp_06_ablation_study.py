import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = "./emogenx_artifacts"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from exp_05_train_emogenx_core import EmoGenXArchitecture, EmoGenXDataset

with open(os.path.join(DATA_DIR, "label_names.json"), "r") as f:
    label_names = json.load(f)

test_df = pd.read_parquet(os.path.join(DATA_DIR, "goemotions_test.parquet"))
test_loader = DataLoader(EmoGenXDataset(test_df), batch_size=32, shuffle=False)
adj = torch.tensor(np.load(os.path.join(DATA_DIR, "adj_matrix.npy")), dtype=torch.float).to(device)

ablation_results = {}

# Variant 1: Pure Baseline DeBERTa
baseline_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=len(label_names),
    problem_type="multi_label_classification"
).to(device)
baseline_model.load_state_dict(torch.load(os.path.join(DATA_DIR, "baseline_deberta_best.pt"), map_location=device))
baseline_model.eval()

preds, targets = [], []
with torch.no_grad():
    for batch in test_loader:
        out = baseline_model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)).logits
        preds.append(torch.sigmoid(out).cpu().numpy())
        targets.append(batch['labels'].numpy())

preds, targets = np.vstack(preds), np.vstack(targets)
ablation_results["Variant 1: Baseline DeBERTa (BCE + 0.5)"] = float(f1_score(targets, (preds >= 0.5).astype(int), average='macro', zero_division=0))

# Variant 6: Full Architecture without Dynamic Thresholding (Static 0.5)
full_model = EmoGenXArchitecture(num_labels=len(label_names)).to(device)
full_model.load_state_dict(torch.load(os.path.join(DATA_DIR, "emogenx_final_model.pt"), map_location=device))
full_model.eval()

f_preds = []
with torch.no_grad():
    for batch in test_loader:
        logits, _ = full_model(batch['input_ids'].to(device), batch['attention_mask'].to(device), adj)
        f_preds.append(torch.sigmoid(logits).cpu().numpy())

f_preds = np.vstack(f_preds)
ablation_results["Variant 6: Proposed EmoGenX (Static 0.5 Threshold)"] = float(f1_score(targets, (f_preds >= 0.5).astype(int), average='macro', zero_division=0))

with open(os.path.join(DATA_DIR, "table2_ablation_results.json"), "w") as f:
    json.dump(ablation_results, f, indent=4)

print(f"[SUCCESS] exp_06 completed. Partial ablation benchmarks logged: {ablation_results}")