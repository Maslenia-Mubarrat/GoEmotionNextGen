import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

DATA_DIR = "./emogenx_artifacts"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(DATA_DIR, "label_names.json"), "r") as f:
    label_names = json.load(f)
num_classes = len(label_names)

train_df = pd.read_parquet(os.path.join(DATA_DIR, "augmented_train_llama_filtered.parquet"))

# Construct Label Co-occurrence Adjacency Matrix
label_matrix = train_df[label_names].values
co_occur = np.dot(label_matrix.T, label_matrix)
diag = np.diag(co_occur)
adj = co_occur / np.maximum(diag[:, None], 1)
adj_tensor = torch.tensor(adj, dtype=torch.float).to(device)

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0)
        pos_loss = targets * ((1 - p) ** self.gamma_pos) * torch.log(p.clamp(min=1e-8))
        neg_loss = (1 - targets) * (p_neg ** self.gamma_neg) * torch.log((1 - p_neg).clamp(min=1e-8))
        return -torch.mean(pos_loss + neg_loss)

class MultiLabelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, targets):
        sim = torch.matmul(features, features.T) / self.temperature
        intersection = torch.matmul(targets, targets.T)
        sums = targets.sum(dim=1, keepdim=True)
        jaccard = intersection / (sums + sums.T - intersection + 1e-8)
        mask = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
        jaccard.masked_fill_(mask, 0.0)
        exp_sim = torch.exp(sim) * (~mask)
        loss = -torch.log(exp_sim / (exp_sim.sum(dim=1, keepdim=True) + 1e-8) + 1e-8)
        return (jaccard * loss).mean()

class EmoGenXArchitecture(nn.Module):
    def __init__(self, num_labels=28):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        d_model = self.encoder.config.hidden_size
        self.label_emb = nn.Parameter(torch.randn(num_labels, d_model))
        self.gat = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, 128)
        self.classifier = nn.Linear(d_model * 2, num_labels)

    def forward(self, input_ids, attention_mask, adj):
        cls_rep = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        g_rep = torch.matmul(adj, self.label_emb)
        g_rep = F.relu(self.gat(g_rep)).mean(dim=0, keepdim=True).expand(cls_rep.size(0), -1)
        fused = torch.cat([cls_rep, g_rep], dim=-1)
        logits = self.classifier(fused)
        proj_features = F.normalize(self.proj(cls_rep), dim=-1)
        return logits, proj_features

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

class EmoGenXDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].values
        self.labels = df[label_names].values.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(str(self.texts[idx]), truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx])
        }

train_loader = DataLoader(EmoGenXDataset(train_df), batch_size=16, shuffle=True)
model = EmoGenXArchitecture(num_labels=num_classes).to(device)

criterion_asl = AsymmetricLoss()
criterion_scl = MultiLabelContrastiveLoss()
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

print("[INFO] Executing Core EmoGenX Training with Joint Objectives...")
model.train()
for epoch in range(3):
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits, proj_features = model(ids, mask, adj_tensor)
        loss = criterion_asl(logits, labels) + 0.1 * criterion_scl(proj_features, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/3 | Training Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), os.path.join(DATA_DIR, "emogenx_final_model.pt"))
np.save(os.path.join(DATA_DIR, "adj_matrix.npy"), adj)
print(f"[SUCCESS] exp_05 completed. Model artifact saved at {DATA_DIR}/emogenx_final_model.pt")