"""
Retrieval-Augmented Generation (RAG): retrieve relevant documents and inject
them into the prompt so the model can answer questions grounded in real data.
"""

import math

# ── Tiny in-memory vector store ───────────────────────────────────────────────

def cosine_similarity(a, b):
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x ** 2 for x in a))
    nb   = math.sqrt(sum(x ** 2 for x in b))
    return dot / (na * nb) if na and nb else 0.0

def bag_of_words(text, vocab):
    words = text.lower().split()
    return [words.count(w) for w in vocab]

# ── Documents (knowledge base) ────────────────────────────────────────────────

documents = [
    "Paris is the capital of France and is known for the Eiffel Tower.",
    "The Python programming language was created by Guido van Rossum in 1991.",
    "Transformers are neural network architectures introduced in the paper 'Attention is All You Need'.",
    "The Great Wall of China stretches over 21,000 kilometres.",
    "RAG combines retrieval systems with generative language models.",
]

# Build a shared vocabulary from all documents
vocab = sorted(set(
    word for doc in documents for word in doc.lower().split()
))

# Embed each document
doc_vectors = [bag_of_words(doc, vocab) for doc in documents]

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query, top_k=2):
    q_vec = bag_of_words(query, vocab)
    scored = [(doc, cosine_similarity(q_vec, vec))
              for doc, vec in zip(documents, doc_vectors)]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

# ── Prompt construction ───────────────────────────────────────────────────────

def build_rag_prompt(query):
    retrieved = retrieve(query, top_k=2)
    context = "\n".join(f"- {doc}" for doc, _ in retrieved)
    return (
        f"Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "What is the capital of France?",
        "Who created Python?",
        "What is RAG in AI?",
    ]

    for query in queries:
        print(f"Query: {query}")
        retrieved = retrieve(query, top_k=2)
        print("Retrieved documents:")
        for doc, score in retrieved:
            print(f"  [{score:.3f}] {doc}")
        print("\nFinal prompt sent to LLM:")
        print(build_rag_prompt(query))
        print("─" * 60)
        print()
