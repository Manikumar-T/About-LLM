"""
Embeddings: dense vector representations of tokens/words that capture semantic meaning.
"""

import math
import random

random.seed(42)

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def random_embedding(dim=8):
    return [random.uniform(-1, 1) for _ in range(dim)]

# Simulated embedding table (in reality learned during training)
words = ["king", "queen", "man", "woman", "apple", "banana", "fruit"]
embeddings = {word: random_embedding() for word in words}

# Override a few to make semantic similarity visible
embeddings["king"]   = [1.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
embeddings["queen"]  = [0.9, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]
embeddings["man"]    = [0.8, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
embeddings["woman"]  = [0.1, 0.8, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0]
embeddings["apple"]  = [0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.9, 0.0]
embeddings["banana"] = [0.0, 0.0, 0.0, 0.0, 0.6, 1.0, 0.8, 0.0]
embeddings["fruit"]  = [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.0, 0.0]

def find_most_similar(word, top_k=3):
    target = embeddings[word]
    scores = [
        (other, cosine_similarity(target, vec))
        for other, vec in embeddings.items()
        if other != word
    ]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

if __name__ == "__main__":
    print("Embedding vectors (8-dimensional):")
    for word, vec in embeddings.items():
        fmt = ", ".join(f"{v:.1f}" for v in vec)
        print(f"  {word:8s}: [{fmt}]")

    print()
    for query in ["king", "apple"]:
        similar = find_most_similar(query)
        print(f"Most similar to '{query}':")
        for word, score in similar:
            print(f"  {word:8s} similarity={score:.3f}")
        print()
