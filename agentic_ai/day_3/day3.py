# # Part 0: Deep Learning basics

# def neuron(inputs, weights, bias):
#     return sum(x * w for x, w in zip(inputs, weights)) + bias

# # Test
# inputs  = [0.5, 0.3, 0.8]
# weights = [0.4, -0.2, 0.7]
# bias    = 0.1

# output = neuron(inputs, weights, bias)
# print(f"Neuron output: {output:.4f}")   # should be 0.8

# # A layer = multiple neurons
# def layer(inputs, weight_matrix, biases):
#     return [neuron(inputs, weights, b) 
#             for weights, b in zip(weight_matrix, biases)]

# # 3 neurons in one layer
# weight_matrix = [
#     [0.4, -0.2, 0.7],   # neuron 1 weights
#     [0.1,  0.5, 0.3],   # neuron 2 weights
#     [-0.3, 0.8, 0.2],   # neuron 3 weights
# ]
# biases = [0.1, 0.0, -0.1]

# layer_output = layer(inputs, weight_matrix, biases)
# print(f"Layer output: {[round(x, 4) for x in layer_output]}")

# # ===============================================================================================

# import math

# def relu(x):
#     return max(0, x)

# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))

# def tanh(x):
#     return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

# # Test on neuron output
# x = 0.8
# print(f"ReLU({x})    = {relu(x):.4f}")
# print(f"Sigmoid({x}) = {sigmoid(x):.4f}")
# print(f"Tanh({x})    = {tanh(x):.4f}")

# # Test with negative
# x = -0.5
# print(f"\nReLU({x})    = {relu(x):.4f}")
# print(f"Sigmoid({x}) = {sigmoid(x):.4f}")
# print(f"Tanh({x})    = {tanh(x):.4f}")


# # ===============================================================================================

# Single Attention mode - which looks at one relationship like 'list' connection to 'python'

import numpy as np

np.random.seed(42)

# Real Stack Overflow question (shortened)
question = "how do i reverse a list in python"
words = question.split()
print("Words:", words)
print("Sequence length:", len(words))

# Step 1: Create fake word embeddings (inreality these come from the model)
# Each word = vect or of d_model numbers
d_model = 8   # small for readability (real BERT uses 768)
seq_len = len(words)

np.random.seed(42)
X = np.random.randn(seq_len, d_model)   # (7 × 8) — 7 words, 8-dim each
print("\nInput shape:", X.shape)         # (7, 8)

# Step 2: Create Q, K, V weight matrices (model learns these)
dk = 8   # key/query dimension
dv = 8   # value dimension

Wq = np.random.randn(d_model, dk)
Wk = np.random.randn(d_model, dk)
Wv = np.random.randn(d_model, dv)

# Step 3: Project inputs into Q, K, V
Q = X @ Wq   # (7×8) @ (8×8) = (7×8)
K = X @ Wk
V = X @ Wv

print("Q shape:", Q.shape)   # (7, 8)
print("K shape:", K.shape)   # (7, 8)
print("V shape:", V.shape)   # (7, 8)

# Step 4: Compute raw scores
scores = Q @ K.T             # (7×8) @ (8×7) = (7×7)
print("\nRaw scores shape:", scores.shape)   # (7, 7)

# Step 5: Scale
scores = scores / np.sqrt(dk)

# Step 6: Softmax
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

weights = softmax(scores)    # (7×7) — attention weights

# Step 7: Weighted sum of values
output = weights @ V         # (7×7) @ (7×8) = (7×8)

# Visualize attention weights
print("\n=== Attention Weights ===")
print(f"{'':12}", end="")
for w in words:
    print(f"{w:10}", end="")
print()

for i, w1 in enumerate(words):
    print(f"{w1:12}", end="")
    for j in range(len(words)):
        print(f"{weights[i][j]:.3f}{'':5}", end="")
    print()

print("\nOutput shape:", output.shape)


# =============================================================================

# Multi-Head Attention

def attention_head(X, Wq, Wk, Wv):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = Q @ K.T / np.sqrt(Wk.shape[-1])
    w = softmax(scores)
    return w @ V

def multi_head_attention(X, num_heads, d_model):
    dk = d_model // num_heads    # split dimensions across heads

    head_outputs = []
    for h in range(num_heads):
        Wq = np.random.randn(d_model, dk)
        Wk = np.random.randn(d_model, dk)
        Wv = np.random.randn(d_model, dk)

        head_out = attention_head(X, Wq, Wk, Wv)   # (seq_len × dk)
        head_outputs.append(head_out)
        print(f"  Head {h+1} output shape: {head_out.shape}")

    # Concatenate all heads along last dimension
    concat = np.concatenate(head_outputs, axis=-1)  # (seq_len × d_model)
    print(f"\nAfter concat: {concat.shape}")

    # Final linear projection
    Wo = np.random.randn(d_model, d_model)
    return concat @ Wo

print("\n=== Multi-Head Attention (4 heads) ===")
mha_output = multi_head_attention(X, num_heads=4, d_model=d_model)
print(f"Final output shape: {mha_output.shape}")    # should be (7, 8)

