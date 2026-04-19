"""
Attention Mechanism: allows the model to focus on relevant parts of the input.
Implements scaled dot-product attention from scratch.
"""

import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(scores):
    e = [math.exp(s) for s in scores]
    total = sum(e)
    return [v / total for v in e]

def scaled_dot_product_attention(query, keys, values):
    """
    query : vector of dim d_k
    keys  : list of vectors, each dim d_k
    values: list of vectors, each dim d_v
    Returns weighted sum of values.
    """
    d_k = len(query)
    scale = math.sqrt(d_k)

    # Score each key against the query
    raw_scores = [dot(query, k) / scale for k in keys]
    weights = softmax(raw_scores)

    # Weighted sum of values
    d_v = len(values[0])
    output = [0.0] * d_v
    for w, v in zip(weights, values):
        for i in range(d_v):
            output[i] += w * v[i]

    return output, weights

if __name__ == "__main__":
    # Toy sentence: ["The", "cat", "sat"]
    # Each token has a 4-dim embedding (simplified)
    tokens = ["The", "cat", "sat"]

    keys = [
        [1.0, 0.0, 0.5, 0.2],  # The
        [0.2, 1.0, 0.3, 0.8],  # cat
        [0.5, 0.4, 1.0, 0.1],  # sat
    ]
    values = keys  # In self-attention, K and V come from the same source

    print("Scaled dot-product attention\n")
    for idx, token in enumerate(tokens):
        query = keys[idx]
        output, weights = scaled_dot_product_attention(query, keys, values)
        w_str = ", ".join(f"{w:.3f}" for w in weights)
        o_str = ", ".join(f"{v:.3f}" for v in output)
        print(f"Query='{token}'  attention weights=[{w_str}]")
        print(f"  → output vector: [{o_str}]")
        print()
