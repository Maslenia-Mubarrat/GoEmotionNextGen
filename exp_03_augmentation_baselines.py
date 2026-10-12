import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

DATA_DIR = "./emogenx_artifacts"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(DATA_DIR, "label_names.json"), "r") as f:
    label_names = json.load(f)
with open(os.path.join(DATA_DIR, "minority_classes.json"), "r") as f:
    minority_classes = json.load(f)

# Load Teacher Classifier for Confidence Filter
clf_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
clf_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=len(label_names),
    problem_type="multi_label_classification"
).to(device)
clf_model.load_state_dict(torch.load(os.path.join(DATA_DIR, "baseline_deberta_best.pt"), map_location=device))
clf_model.eval()

# Load Causal LLM for conditional synthesis
GEN_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
print(f"[INFO] Initializing generative model: {GEN_MODEL_ID}...")
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

raw_synthetic_records = []
filtered_synthetic_records = []

for label in minority_classes:
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are a linguistic expert creating training data.<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Generate 5 realistic, casual Reddit comments expressing the emotion '{label}' subtly. "
        f"Do not explicitly mention the emotion keyword '{label}'. "
        f"Output strictly one comment per line.<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_new_tokens=150, temperature=0.8, top_p=0.95, do_sample=True)
    
    generated_text = gen_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    label_idx = label_names.index(label)
    
    for line in generated_text.split("\n"):
        clean_text = line.strip(" -*0123456789.")
        if len(clean_text) > 10:
            row = {name: 0 for name in label_names}
            row[label] = 1
            row['text'] = clean_text
            raw_synthetic_records.append(row)
            
            # Purity check via Teacher Confidence Gate
            enc = clf_tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(clf_model(**enc).logits)[0, label_idx].item()
            
            if prob >= 0.80:
                row_filtered = dict(row)
                row_filtered['teacher_confidence'] = prob
                filtered_synthetic_records.append(row_filtered)

print(f"[INFO] Raw Generated Count: {len(raw_synthetic_records)} | Filtered Count (P>=0.80): {len(filtered_synthetic_records)}")

train_df = pd.read_parquet(os.path.join(DATA_DIR, "goemotions_train.parquet"))

# Save Unfiltered Variant (for ablation study)
unfiltered_df = pd.concat([train_df, pd.DataFrame(raw_synthetic_records)], ignore_index=True)
unfiltered_df.to_parquet(os.path.join(DATA_DIR, "augmented_train_unfiltered.parquet"), index=False)

# Save Filtered Variant (Proposed core dataset)
if filtered_synthetic_records:
    filtered_df = pd.concat([train_df, pd.DataFrame(filtered_synthetic_records).drop(columns=['teacher_confidence'])], ignore_index=True)
else:
    filtered_df = train_df
filtered_df.to_parquet(os.path.join(DATA_DIR, "augmented_train_llama_filtered.parquet"), index=False)

with open(os.path.join(DATA_DIR, "raw_synthetic_samples.json"), "w") as f:
    json.dump(raw_synthetic_records, f, indent=4)
with open(os.path.join(DATA_DIR, "filtered_synthetic_samples.json"), "w") as f:
    json.dump(filtered_synthetic_records, f, indent=4)

print("[SUCCESS] exp_03 completed. Augmented corpora stored.")