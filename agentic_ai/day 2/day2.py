# Part 1: Naive tokenization problems

text = "I don't know ChatGPT's pricing, it's $20/month"

# Approach 1: split by space
word_tokens = text.split(" ")
print("Word split:", word_tokens)
print("Count:", len(word_tokens))

# Approach 2: character level
char_tokens = list(text)
print("\nChar split:", char_tokens)
print("Count:", len(char_tokens))

# The problem: "running" and "runs" are totally different tokens
vocab = ["running", "runs", "ran", "runner"]
print("\nSame meaning, different tokens:", vocab)
print("A model has to learn ALL of these separately → wasteful")

# ===========================================================================

# Part 2: BPE Algorithm from scratch

from collections import Counter

def get_pairs(vocab):
    """Count frequency of every adjacent pair across all words"""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_pair(pair, vocab):
    """Merge the best pair everywhere in vocab"""
    new_vocab = {}
    bigram = " ".join(pair)          # "l o"
    replacement = "".join(pair)      # "lo"
    
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

# Start: each word split into characters + space-separated
# Format: "c h a r s" -> frequency
vocab = {
    "l o w </w>": 3,      # </w> marks end of word
    "l o w e r </w>": 1,
    "l o w e s t </w>": 1,
    "n e w e r </w>": 1,
    "w i d e r </w>": 1,
}

print("=== BPE Training ===")
print("Initial vocab:", vocab)

num_merges = 6

for i in range(num_merges):
    pairs = get_pairs(vocab)
    best = max(pairs, key=pairs.get)   # most frequent pair
    vocab = merge_pair(best, vocab)
    print(f"\nMerge {i+1}: {best[0]} + {best[1]} → {''.join(best)}")
    print("Vocab:", vocab)


# =============================================================================

# Part 3: Real tokenizers

from transformers import GPT2Tokenizer, BertTokenizer

# GPT-2 uses BPE
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "ChatGPT's tokenization is fascinating!"

tokens = gpt_tokenizer.tokenize(text)
ids = gpt_tokenizer.encode(text)
decoded = gpt_tokenizer.decode(ids)

print("=== GPT-2 (BPE) ===")
print("Tokens:", tokens)
print("IDs:   ", ids)
print("Decoded:", decoded)

# BERT uses WordPiece
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert_tokens = bert_tokenizer.tokenize(text)
bert_ids = bert_tokenizer.encode(text)

print("\n=== BERT (WordPiece) ===")
print("Tokens:", bert_tokens)
print("IDs:   ", bert_ids)

# Compare unknown word handling
unknown = "supercalifragilistic"
print(f"\nUnknown word: '{unknown}'")
print("GPT-2:", gpt_tokenizer.tokenize(unknown))
print("BERT: ", bert_tokenizer.tokenize(unknown))


import tiktoken
enc = tiktoken.get_encoding("o200k_base")  # GPT-4
print(enc.encode("ChatGPT's tokenization"))

# ===============================================================

# Part 4: Embeddings

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good quality

sentences = [
    "I love dogs",
    "I love cats", 
    "Dogs are great pets",
    "The stock market crashed today",
    "Bitcoin price dropped 10%",
]

embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)         # (5, 384) — 5 sentences, 384 dimensions each
print("First vector (first 10 values):")
print(embeddings[0][:10])


# Top free embedding models ranked:

# Model	                                Dims	Best for
# BAAI/bge-large-en-v1.5	            1024	English, beats ada-002
# BAAI/bge-m3	                        1024	100+ languages, multilingual
# intfloat/e5-large-v2	                1024	Strong general purpose
# nomic-ai/nomic-embed-text-v1.5	    768	    Long documents (8192 tokens)
# mixedbread-ai/mxbai-embed-large-v1	1024	Very competitive quality


# For large scale production (free):

# Run locally via Ollama — nomic-embed-text model, no API cost
# HuggingFace Inference Endpoints — free tier for low volume


# For our projects:

# Learning → all-MiniLM-L6-v2 (what we're using, fast)
# Knowledge Assistant (Day 9) → BAAI/bge-large-en-v1.5 (free, production quality)


# ===========================================================================

# Part 5: Cosine similarity from scratch

import math

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def magnitude(a):
    return math.sqrt(sum(x * x for x in a))

def cosine_similarity(a, b):
    return dot_product(a, b) / (magnitude(a) * magnitude(b))

# Test with simple 2D vectors first
v1 = [1, 0]   # points right
v2 = [1, 0]   # same direction
v3 = [0, 1]   # points up (90 degrees)
v4 = [-1, 0]  # opposite direction

print(cosine_similarity(v1, v2))   # should be 1.0
print(cosine_similarity(v1, v3))   # should be 0.0
print(cosine_similarity(v1, v4))   # should be -1.0

# Now use on real embeddings
sentences = [
    "I love dogs",
    "I love cats",
    "Dogs are great pets",
    "The stock market crashed today",
    "Bitcoin price dropped 10%",
]

e = embeddings  # from Part 4

print("\n=== Similarity scores ===")
query = 0  # "I love dogs"
for i in range(len(sentences)):
    score = cosine_similarity(e[query].tolist(), e[i].tolist())
    print(f"{score:.3f} — {sentences[i]}")


# ========================================================================

print("\n=== Full similarity matrix ===")
print(f"{'':30}", end="")
for s in sentences:
    print(f"{s[:12]:14}", end="")
print()

for i, s1 in enumerate(sentences):
    print(f"{s1[:30]:30}", end="")
    for j, s2 in enumerate(sentences):
        score = cosine_similarity(embeddings[i].tolist(), embeddings[j].tolist())
        print(f"{score:.2f}{'':{9 if score < 0 else 10}}", end="")
    print()



# =======================================================================================

# Part 6: Project: Semantic Similarity Finder

# Build a CLI tool that:

# Takes a query from user input
# Embeds the query
# Finds the most similar sentence from a knowledge base
# Prints ranked results with scores


import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good quality

knowledge_base = [
    "Python is a popular programming language for AI",
    "Machine learning models learn from data",
    "Neural networks are inspired by the human brain",
    "Dogs are loyal and friendly animals",
    "cats are independent pets",
    "The stock market had a volatile week",
    "Bitcoin reached a new all time high",
    "FastAPI is great for building REST APIs",
    "LangChain helps build LLM applications",
    "Vector databases store embeddings for fast search",
]

embeddings = model.encode(knowledge_base)

def cosine_similarity(a, b) :
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = input("enter your text: ")
query_embeddings = model.encode(query)

max_score = -1.0
closest_sent = ""
scores = {}
for i, sent in enumerate(knowledge_base):
    score = cosine_similarity(query_embeddings, embeddings[i])
    scores[sent] = score

    if score >= max_score:
        closest_sent = sent
        max_score = score


ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for sent, score in ranked:
    print(f"{score:.3f} — {sent}")


print(f"{max_score:.3f} — {closest_sent}")