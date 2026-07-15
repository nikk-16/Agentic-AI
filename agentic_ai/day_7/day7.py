# import chromadb

# client = chromadb.PersistentClient(path="./chroma_db")
# collection = client.get_or_create_collection("my_docs")

# # Add
# collection.add(documents=["text1", "text2"], ids=["1", "2"])

# # we can also pass embeddings directly, passing documents to retrieve the text if needed
# # Without it, results give only id, and also it accepts numpy array. 

# # collection.add(embeddings = [[0.1, 0.2, ... 384 dims] [0.2, 0.3, ... 384 dims]], documents=["text1", "text2"], ids=["1", "2"])


# # while 
# # Query
# results = collection.query(query_texts=["search query"], n_results=5)
# print(results)

# collection.query(
#     query_embeddings=[...],
#     n_results=5,
#     where={"score": {"$gte": 50}}   # only high-voted questions
# )

# # Delete
# collection.delete(ids=["1"])


# ========================================================================================


import os
import math
import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("stack_overflow_questions", metadata={"hnsw:space": "cosine"})

model = SentenceTransformer("all-MiniLM-L6-v2")

questions = pd.read_csv('../../datasets/StackOverflowDataset/Questions.csv', encoding='utf-8', encoding_errors='replace')

data = questions[:5000]

# load csv
title_embeddings = []
with open('../day_6/title_embeddings.npy', 'rb') as f:
    title_embeddings = np.load(f)


collection.add(
    documents=data['Title'].tolist(),
    embeddings=title_embeddings,
    ids=data['Id'].astype(str).tolist(),
    metadatas=[{"score": int(s)} for s in data['Score']]
)

query = "how to sort a list in Python"

query_embedding = model.encode(query)

lesser = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5
)
      
greater = collection.query(query_embeddings=[query_embedding.tolist()],
    n_results=5,
    where={"score": {"$gte": 50}}
)


print(f"with score < 50:\n {lesser}\n")
print(f"with score > 50:\n {greater}")
