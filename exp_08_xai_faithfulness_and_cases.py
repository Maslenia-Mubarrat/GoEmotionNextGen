import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from captum.attr import IntegratedGradients

DATA_DIR = "./emogenx_artifacts"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from exp_05_train_emogenx_core import EmoGenXArchitecture

with open(os.path.join(DATA_DIR, "label_names.json"), "r") as f:
    label_names = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
adj = torch.tensor(np.load(os.path.join(DATA_DIR, "adj_matrix.npy")), dtype=torch.float).to(device)

model = EmoGenXArchitecture(num_labels=len(label_names)).to(device)
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "emogenx_final_model.pt"), map_location=device))
model.eval()

def attribution_forward(inputs_embeds, attn_mask):
    cls_rep = model.encoder(inputs_embeds=inputs_embeds, attention_mask=attn_mask).last_hidden_state[:, 0, :]
    g_rep = torch.matmul(adj, model.label_emb)
    g_rep = torch.relu(model.gat(g_rep)).mean(dim=0, keepdim=True).expand(cls_rep.size(0), -1)
    fused = torch.cat([cls_rep, g_rep], dim=-1)
    return model.classifier(fused)

ig = IntegratedGradients(attribution_forward)

# Comprehensiveness Evaluation (Token Deletion)
def compute_comprehensiveness(sample_text, target_label):
    target_idx = label_names.index(target_label)
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    embeddings = model.encoder.embeddings(inputs.input_ids)
    
    with torch.no_grad():
        orig_prob = torch.sigmoid(attribution_forward(embeddings, inputs.attention_mask))[0, target_idx].item()
    
    attributions = ig.attribute(embeddings, additional_forward_args=(inputs.attention_mask,), target=target_idx)
    token_weights = attributions.sum(dim=-1).squeeze(0).abs().detach().cpu().numpy()
    
    # Identify top 20% salient tokens
    num_tokens_to_drop = max(1, int(0.20 * len(token_weights)))
    top_indices = np.argsort(token_weights)[-num_tokens_to_drop:]
    
    # Delete salient tokens
    input_ids_list = inputs.input_ids[0].tolist()
    modified_ids = [token for idx, token in enumerate(input_ids_list) if idx not in top_indices]
    modified_tensor = torch.tensor([modified_ids], device=device)
    modified_mask = torch.ones_like(modified_tensor, device=device)
    
    with torch.no_grad():
        mod_embeddings = model.encoder.embeddings(modified_tensor)
        dropped_prob = torch.sigmoid(attribution_forward(mod_embeddings, modified_mask))[0, target_idx].item()
    
    return float(orig_prob - dropped_prob)

test_sentence = "Losing my dog yesterday shattered my world entirely."
comp_score = compute_comprehensiveness(test_sentence, "grief")

xai_metrics = {
    "Comprehensiveness_Drop_Sample": comp_score,
    "Status": "Evaluated"
}

with open(os.path.join(DATA_DIR, "table_xai_faithfulness_metrics.json"), "w") as f:
    json.dump(xai_metrics, f, indent=4)

print(f"[SUCCESS] exp_08 completed. Faithfulness metric (Comprehensiveness Drop: {comp_score:.4f}) recorded.")