"""
Temperature & Sampling: controls how creative or deterministic the model's
next-token selection is.
"""

import math
import random

random.seed(42)

def softmax(logits):
    e = [math.exp(l) for l in logits]
    s = sum(e)
    return [v / s for v in e]

def apply_temperature(logits, temperature):
    if temperature == 0:
        raise ValueError("Use argmax for temperature=0")
    return [l / temperature for l in logits]

def sample(probs, vocab):
    r = random.random()
    cumulative = 0.0
    for token, p in zip(vocab, probs):
        cumulative += p
        if r <= cumulative:
            return token
    return vocab[-1]

def top_k_filter(probs, vocab, k):
    """Keep only top-k tokens, re-normalise."""
    paired = sorted(zip(probs, vocab), reverse=True)[:k]
    top_probs, top_vocab = zip(*paired)
    total = sum(top_probs)
    return [p / total for p in top_probs], list(top_vocab)

def top_p_filter(probs, vocab, p):
    """Nucleus sampling: keep smallest set of tokens whose cumulative prob >= p."""
    paired = sorted(zip(probs, vocab), reverse=True)
    cumulative, selected = 0.0, []
    for prob, token in paired:
        cumulative += prob
        selected.append((prob, token))
        if cumulative >= p:
            break
    total = sum(pr for pr, _ in selected)
    return [pr / total for pr, _ in selected], [tok for _, tok in selected]

if __name__ == "__main__":
    vocab   = ["cat", "dog", "bird", "fish", "lion"]
    logits  = [2.0, 1.5, 0.5, 0.2, 0.1]  # raw model output

    print("Raw logits:", dict(zip(vocab, logits)))
    print()

    for temp in [0.1, 0.5, 1.0, 2.0]:
        scaled  = apply_temperature(logits, temp)
        probs   = softmax(scaled)
        p_str   = ", ".join(f"{t}:{p:.3f}" for t, p in zip(vocab, probs))
        samples = [sample(probs, vocab) for _ in range(10)]
        print(f"Temperature={temp:.1f}  probs=[{p_str}]")
        print(f"  10 samples: {samples}")
        print()

    print("Top-k=2 sampling (temp=1.0):")
    probs = softmax(logits)
    k_probs, k_vocab = top_k_filter(probs, vocab, k=2)
    print(f"  Kept tokens: {k_vocab}")
    print(f"  10 samples:  {[sample(k_probs, k_vocab) for _ in range(10)]}")
    print()

    print("Top-p=0.9 (nucleus) sampling:")
    p_probs, p_vocab = top_p_filter(probs, vocab, p=0.9)
    print(f"  Kept tokens: {p_vocab}")
    print(f"  10 samples:  {[sample(p_probs, p_vocab) for _ in range(10)]}")
