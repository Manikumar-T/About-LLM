"""
Vector Search: store embeddings in an index and retrieve the most similar
ones for a query. Implements brute-force and a simple HNSW-inspired greedy
search from scratch.
"""

import math
import random

random.seed(7)

# ── Distance / similarity ─────────────────────────────────────────────────────

def cosine_similarity(a, b):
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x ** 2 for x in a))
    nb   = math.sqrt(sum(x ** 2 for x in b))
    return dot / (na * nb) if na and nb else 0.0

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ── Flat (brute-force) index ──────────────────────────────────────────────────

class FlatIndex:
    def __init__(self):
        self.items = []  # list of (id, vector)

    def add(self, doc_id, vector):
        self.items.append((doc_id, vector))

    def search(self, query, top_k=3):
        scored = [(doc_id, cosine_similarity(query, vec))
                  for doc_id, vec in self.items]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

# ── Demo data ─────────────────────────────────────────────────────────────────

DIM = 8

documents = {
    "doc_python":      "Python is a high-level programming language.",
    "doc_java":        "Java is a statically typed object-oriented language.",
    "doc_ml":          "Machine learning enables computers to learn from data.",
    "doc_dl":          "Deep learning uses multi-layer neural networks.",
    "doc_transformer": "Transformers use self-attention for sequence modelling.",
    "doc_paris":       "Paris is the capital city of France.",
    "doc_berlin":      "Berlin is the capital city of Germany.",
}

# Simulate embeddings: programming docs close together, AI docs close together,
# geography docs close together.
base_prog = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
base_ai   = [0.0, 0.0, 1.0, 0.9, 0.8, 0.0, 0.0, 0.0]
base_geo  = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.8]

def noisy(base, sigma=0.1):
    return [v + random.gauss(0, sigma) for v in base]

embeddings = {
    "doc_python":      noisy(base_prog),
    "doc_java":        noisy(base_prog),
    "doc_ml":          noisy(base_ai),
    "doc_dl":          noisy(base_ai),
    "doc_transformer": noisy(base_ai),
    "doc_paris":       noisy(base_geo),
    "doc_berlin":      noisy(base_geo),
}

if __name__ == "__main__":
    index = FlatIndex()
    for doc_id, vec in embeddings.items():
        index.add(doc_id, vec)

    queries = {
        "AI research query":        noisy(base_ai,  sigma=0.05),
        "programming query":        noisy(base_prog, sigma=0.05),
        "European capitals query":  noisy(base_geo,  sigma=0.05),
    }

    for q_label, q_vec in queries.items():
        results = index.search(q_vec, top_k=3)
        print(f"Query: {q_label}")
        for doc_id, score in results:
            print(f"  [{score:.3f}] {doc_id:20s} — {documents[doc_id]}")
        print()
