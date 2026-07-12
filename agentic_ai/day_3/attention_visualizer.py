import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load pretrained BERT (downloads ~440MB first time)
print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()
print("Loaded.\n")

# Real Stack Overflow question
question = "how do i reverse a list in python"
print(f"Question: {question}\n")

# Tokenize — BERT adds [CLS] at start, [SEP] at end
inputs = tokenizer(question, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(f"Tokens ({len(tokens)}): {tokens}\n")

# Run through BERT — no gradient needed (we're not training)
with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions = tuple of 12 layers
# each layer shape: (batch=1, heads=12, seq_len, seq_len)
attentions = outputs.attentions
print(f"Total layers: {len(attentions)}")
print(f"Per layer shape: {attentions[0].shape}  (batch, heads, seq, seq)\n")

# Pick layer 6 (middle — good balance of syntax + semantics)
# Average across all 12 heads
layer = 6
avg_attention = attentions[layer][0].mean(dim=0).numpy()  # (seq_len, seq_len)

# Print attention matrix
print(f"=== Attention Matrix — Layer {layer} (avg across 12 heads) ===\n")
col_width = 10

# Header row
print(f"{'':>{col_width}}", end="")
for t in tokens:
    print(f"{t:>{col_width}}", end="")
print()

# Each row = which token attends to what
for i, row_token in enumerate(tokens):
    print(f"{row_token:>{col_width}}", end="")
    for j in range(len(tokens)):
        val = avg_attention[i][j]
        bar = "█" * int(val * 10)   # visual bar scaled to 0-10
        print(f"{bar:>{col_width}}", end="")
    print()

print()

# Top 5 strongest attention pairs (excluding [CLS] and [SEP])
print("=== Top 5 Attention Pairs ===")
pairs = []
for i in range(len(tokens)):
    for j in range(len(tokens)):
        if i != j and tokens[i] not in ["[CLS]", "[SEP]"] and tokens[j] not in ["[CLS]", "[SEP]"]:
            pairs.append((avg_attention[i][j], tokens[i], tokens[j]))

pairs.sort(reverse=True)
for score, t_from, t_to in pairs[:5]:
    print(f"  {t_from:>10}  →  {t_to:<10}  ({score:.4f})")
