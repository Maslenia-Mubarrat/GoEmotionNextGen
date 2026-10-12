import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer

DATA_DIR = "./emogenx_artifacts"
FIGURES_DIR = "./emogenx_artifacts/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "filtered_synthetic_samples.json"), "r") as f:
    synthetic_samples = json.load(f)

with open(os.path.join(DATA_DIR, "minority_classes.json"), "r") as f:
    minority_classes = json.load(f)

train_df = pd.read_parquet(os.path.join(DATA_DIR, "goemotions_train.parquet"))

# 1. Distinct-1 and Distinct-2 Metrics
def calculate_distinct_n(texts, n=1):
    total_ngrams = 0
    unique_ngrams = set()
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

syn_texts = [item['text'] for item in synthetic_samples]
distinct_1 = calculate_distinct_n(syn_texts, n=1)
distinct_2 = calculate_distinct_n(syn_texts, n=2)

# 2. Self-BLEU Calculation
smoothing = SmoothingFunction().method1
bleu_scores = []
tokenized_texts = [t.lower().split() for t in syn_texts]
sample_size = min(len(tokenized_texts), 200)

for i in range(sample_size):
    references = tokenized_texts[:i] + tokenized_texts[i+1:sample_size]
    if references:
        score = sentence_bleu(references, tokenized_texts[i], smoothing_function=smoothing)
        bleu_scores.append(score)

mean_self_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0

fidelity_metrics = {
    "Distinct-1": float(distinct_1),
    "Distinct-2": float(distinct_2),
    "Self-BLEU": mean_self_bleu
}

with open(os.path.join(DATA_DIR, "table_diversity_metrics.json"), "w") as f:
    json.dump(fidelity_metrics, f, indent=4)
print(f"[INFO] Diversity Metrics calculated: {fidelity_metrics}")

# 3. Latent Feature Space t-SNE Distribution Analysis
print("[INFO] Computing t-SNE projections...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

real_minority_texts = []
for m_class in minority_classes:
    matched = train_df[train_df[m_class] == 1]['text'].head(50).tolist()
    real_minority_texts.extend(matched)

syn_subset = syn_texts[:len(real_minority_texts)]

if len(real_minority_texts) > 0 and len(syn_subset) > 0:
    all_texts = real_minority_texts + syn_subset
    labels = ["Real Minority"] * len(real_minority_texts) + ["Synthetic Filtered"] * len(syn_subset)
    
    embeddings = embedder.encode(all_texts)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_texts)-1))
    proj_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for category, color, marker in zip(["Real Minority", "Synthetic Filtered"], ["#2563eb", "#dc2626"], ["o", "^"]):
        idx = [i for i, l in enumerate(labels) if l == category]
        plt.scatter(proj_2d[idx, 0], proj_2d[idx, 1], label=category, alpha=0.6, c=color, marker=marker, edgecolors='none')
    
    plt.title("t-SNE Feature Space Distribution of Real vs Synthetic Samples")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_tsne_distribution.png"), dpi=300)
    plt.close()

print(f"[SUCCESS] exp_04 completed. Metrics and t-SNE plot generated in {FIGURES_DIR}")