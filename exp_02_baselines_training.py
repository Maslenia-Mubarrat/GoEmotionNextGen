import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

DATA_DIR = "./emogenx_artifacts"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df = pd.read_parquet(os.path.join(DATA_DIR, "goemotions_train.parquet"))
val_df   = pd.read_parquet(os.path.join(DATA_DIR, "goemotions_val.parquet"))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, "goemotions_test.parquet"))

with open(os.path.join(DATA_DIR, "label_names.json"), "r") as f:
    label_names = json.load(f)

num_classes = len(label_names)
benchmark_records = {}

# 1. Classical Baseline: TF-IDF + Logistic Regression (OneVsRest)
print("[INFO] Evaluating Classical Baseline: TF-IDF + Logistic Regression...")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(train_df['text'].astype(str))
X_test_vec = tfidf.transform(test_df['text'].astype(str))
y_train = train_df[label_names].values
y_test = test_df[label_names].values

clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced'))
clf.fit(X_train_vec, y_train)
lr_preds = clf.predict(X_test_vec)

benchmark_records["TF-IDF + Logistic Regression"] = {
    "Macro-F1": float(f1_score(y_test, lr_preds, average='macro', zero_division=0)),
    "Micro-F1": float(f1_score(y_test, lr_preds, average='micro', zero_division=0))
}

# 2. DeBERTa-v3 Baseline Classifier & Teacher Checkpoint Training
MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class EmotionDataset(Dataset):
    def __init__(self, df, max_len=128):
        self.texts = df['text'].values
        self.labels = df[label_names].values.astype(np.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

train_loader = DataLoader(EmotionDataset(train_df), batch_size=16, shuffle=True)
val_loader   = DataLoader(EmotionDataset(val_df), batch_size=16, shuffle=False)
test_loader  = DataLoader(EmotionDataset(test_df), batch_size=16, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    problem_type="multi_label_classification"
).to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

best_val_macro_f1 = 0.0

print("[INFO] Training Baseline DeBERTa-v3-base...")
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask=attn).logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)).logits
            val_preds.append(torch.sigmoid(logits).cpu().numpy())
            val_targets.append(batch['labels'].numpy())

    macro_f1 = f1_score(np.vstack(val_targets), (np.vstack(val_preds) >= 0.5).astype(int), average='macro', zero_division=0)
    print(f"Epoch {epoch+1}/{epochs} | Val Macro-F1: {macro_f1:.4f}")

    if macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = macro_f1
        torch.save(model.state_dict(), os.path.join(DATA_DIR, "baseline_deberta_best.pt"))

# Final Test evaluation of baseline DeBERTa
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "baseline_deberta_best.pt"), map_location=device))
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)).logits
        test_preds.append(torch.sigmoid(logits).cpu().numpy())
        test_targets.append(batch['labels'].numpy())

test_preds = np.vstack(test_preds)
test_targets = np.vstack(test_targets)
bin_preds = (test_preds >= 0.5).astype(int)

benchmark_records["DeBERTa-v3-base"] = {
    "Macro-F1": float(f1_score(test_targets, bin_preds, average='macro', zero_division=0)),
    "Micro-F1": float(f1_score(test_targets, bin_preds, average='micro', zero_division=0)),
    "Macro-Precision": float(precision_score(test_targets, bin_preds, average='macro', zero_division=0)),
    "Macro-Recall": float(recall_score(test_targets, bin_preds, average='macro', zero_division=0))
}

with open(os.path.join(DATA_DIR, "table1_baselines_results.json"), "w") as f:
    json.dump(benchmark_records, f, indent=4)

print(f"[SUCCESS] exp_02 completed. Baseline results and teacher weights saved to {DATA_DIR}")