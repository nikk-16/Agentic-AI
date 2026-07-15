import math

def cosine_similarity(a, b):
    dot = 0
    magA = 0
    magB = 0
    for i, j in zip(a,b):
        dot += i*j
        magA += i*i
        magB += j*j
    return dot / (math.sqrt(magA) * math.sqrt(magB))

def dot_product(a, b):
    dot = 0
    for i, j in zip(a,b):
        dot += i*j
    return dot 

def euclidean(a, b):
    dist = 0
    for i, j in zip(a,b):
        dist += (i-j) * (i-j)
    return math.sqrt(dist)

a=[1,2,3]
b=[2,4,6]

print(f"Cosine: {cosine_similarity(a,b)}")
print(f"Dot: {dot_product(a,b)}")
print(f"Euclidean: {euclidean(a,b)}")

# ====================================================================================

def normalize(v):
    return [x / mag(v) for x in v]

def mag(v):
    magA = 0
    for a in v:
        magA += a*a

    return math.sqrt(magA)

v= [3, 4]
print(f"Normalized: {normalize(v)}")

# ====================================================================================

from sentence_transformers import SentenceTransformer

embedding = SentenceTransformer("all-MiniLM-L6-v2")

query = "how to handle exceptions in Python"

documents = [
    "how do I catch exceptions in Python using try except",
    "what is the difference between a list and a tuple in Python",
    "how to reverse a string in Python",
    "Python error handling best practices with multiple exceptions",
    "how to open and read a file in Python",
    "what does the finally block do in Python exception handling",
    "how to sort a dictionary by value in Python"
]

def semantic_search(query, documents, top_k=3):
    query_embedding = embedding.encode(query)
    documents_embeddings = embedding.encode(documents)

    similarity_scores = [cosine_similarity(query_embedding, embed) for embed in documents_embeddings]
    scored = [(score, i) for i, score in enumerate(similarity_scores)]
    scored.sort(reverse=True)
    return scored[:top_k]

ans = semantic_search(query, documents)
print("Top 3 Similar docs:")
for score, i in ans:
    print(f"  {score:.4f} — {documents[i]}")
    