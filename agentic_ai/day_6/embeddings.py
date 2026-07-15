import os
import math
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

questions = pd.read_csv('../../datasets/StackOverflowDataset/Questions.csv', encoding='utf-8', encoding_errors='replace')

data = questions[:5000]

titles = data['Title']

model = SentenceTransformer("all-MiniLM-L6-v2")

if(os.path.exists('./title_embeddings.npy')):
    with open('./title_embeddings.npy', 'rb') as f:
        title_embeddings = np.load(f)
else:
    title_embeddings = model.encode(titles)
    with open('title_embeddings.npy', 'wb') as f:
        np.save(f, title_embeddings)

query = 'how to sort a list in python'

def cosine_similarity(a, b):
    dot = 0
    magA = 0
    magB = 0
    for i, j in zip(a,b):
        dot += i*j
        magA += i*i
        magB += j*j
    return dot / (math.sqrt(magA) * math.sqrt(magB))

def semantic_search(query, top_k=5):
    query_embedding = model.encode(query)

    similarity_scores = [cosine_similarity(query_embedding, embed) for embed in title_embeddings]
    scored = [(score, i) for i, score in enumerate(similarity_scores)]
    scored.sort(reverse=True)
    return scored[:top_k]

ans = semantic_search(query)
print("Top 5 Similar questions:")
for score, i in ans:
    print(f"  {score:.4f} — {titles[i]}")