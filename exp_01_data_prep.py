import os
import json
import numpy as np
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = "./emogenx_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Fetching GoEmotions dataset (simplified split)...")
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

label_names = dataset['train'].features['labels'].feature.names
num_classes = len(label_names)

def process_split(split_data):
    texts = split_data['text']
    raw_labels = split_data['labels']
    multi_hot = np.zeros((len(raw_labels), num_classes), dtype=np.int32)
    for idx, sample_labels in enumerate(raw_labels):
        for l in sample_labels:
            multi_hot[idx, l] = 1
    df = pd.DataFrame(multi_hot, columns=label_names)
    df.insert(0, 'text', texts)
    return df

print("[INFO] Formatting Train, Validation, and Test splits...")
train_df = process_split(dataset['train'])
val_df   = process_split(dataset['validation'])
test_df  = process_split(dataset['test'])

# Profiling class distribution and identifying minority classes (< 1.5%)
label_counts = train_df[label_names].sum().sort_values(ascending=True)
threshold_count = int(0.015 * len(train_df))
minority_classes = label_counts[label_counts < threshold_count].index.tolist()

print(f"[INFO] Detected {len(minority_classes)} Minority Classes: {minority_classes}")

train_df.to_parquet(os.path.join(OUTPUT_DIR, "goemotions_train.parquet"), index=False)
val_df.to_parquet(os.path.join(OUTPUT_DIR, "goemotions_val.parquet"), index=False)
test_df.to_parquet(os.path.join(OUTPUT_DIR, "goemotions_test.parquet"), index=False)

with open(os.path.join(OUTPUT_DIR, "label_names.json"), "w") as f:
    json.dump(label_names, f)
with open(os.path.join(OUTPUT_DIR, "minority_classes.json"), "w") as f:
    json.dump(minority_classes, f)

print(f"[SUCCESS] exp_01 completed. Dataset splits written to {OUTPUT_DIR}")