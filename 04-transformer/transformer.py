"""
Transformer building blocks: positional encoding, feed-forward layer, and a
single encoder layer assembled from scratch (no ML libraries).
"""

import math

# ── Positional Encoding ──────────────────────────────────────────────────────

def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding as in 'Attention is All You Need'."""
    pe = []
    for pos in range(seq_len):
        row = []
        for i in range(d_model):
            angle = pos / (10000 ** (2 * (i // 2) / d_model))
            row.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        pe.append(row)
    return pe

# ── Helpers ──────────────────────────────────────────────────────────────────

def add_vectors(a, b):
    return [x + y for x, y in zip(a, b)]

def layer_norm(vec, eps=1e-6):
    mean = sum(vec) / len(vec)
    var  = sum((x - mean) ** 2 for x in vec) / len(vec)
    return [(x - mean) / math.sqrt(var + eps) for x in vec]

def relu(x):
    return max(0.0, x)

def feed_forward(vec, w1, b1, w2, b2):
    """Two-layer FFN: FFN(x) = max(0, xW1+b1)W2+b2."""
    hidden = [relu(sum(vec[j] * w1[j][i] for j in range(len(vec))) + b1[i])
              for i in range(len(b1))]
    out    = [sum(hidden[j] * w2[j][i] for j in range(len(hidden))) + b2[i]
              for i in range(len(b2))]
    return out

# ── Attention (reused from concept 03) ───────────────────────────────────────

def softmax(scores):
    e = [math.exp(s - max(scores)) for s in scores]
    t = sum(e)
    return [v / t for v in e]

def attention(query, keys, values):
    scale = math.sqrt(len(query))
    scores = [sum(q * k for q, k in zip(query, key)) / scale for key in keys]
    weights = softmax(scores)
    d_v = len(values[0])
    out = [sum(weights[i] * values[i][j] for i in range(len(values))) for j in range(d_v)]
    return out

# ── Encoder Layer ─────────────────────────────────────────────────────────────

def encoder_layer(token_embeddings, ffn_w1, ffn_b1, ffn_w2, ffn_b2):
    """
    One Transformer encoder layer:
      1. Self-attention  + residual + layer norm
      2. Feed-forward    + residual + layer norm
    """
    # Self-attention (using token embeddings as Q, K, V)
    attn_out = [attention(tok, token_embeddings, token_embeddings)
                for tok in token_embeddings]

    # Add & Norm (residual connection)
    normed1 = [layer_norm(add_vectors(orig, attn))
               for orig, attn in zip(token_embeddings, attn_out)]

    # Feed-forward
    ff_out = [feed_forward(tok, ffn_w1, ffn_b1, ffn_w2, ffn_b2) for tok in normed1]

    # Add & Norm
    normed2 = [layer_norm(add_vectors(n, ff))
               for n, ff in zip(normed1, ff_out)]

    return normed2

# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seq_len, d_model = 3, 4

    # Raw token embeddings (random stand-ins)
    raw_embeddings = [
        [0.5, 0.1, 0.3, 0.9],  # token 0
        [0.2, 0.8, 0.6, 0.4],  # token 1
        [0.7, 0.3, 0.1, 0.5],  # token 2
    ]

    # Add positional encoding
    pe = positional_encoding(seq_len, d_model)
    token_embeddings = [add_vectors(e, p) for e, p in zip(raw_embeddings, pe)]

    print("Token embeddings after positional encoding:")
    for i, vec in enumerate(token_embeddings):
        print(f"  token {i}: [{', '.join(f'{v:.3f}' for v in vec)}]")

    # Tiny FFN weights (identity-like for demo)
    ffn_hidden = 8
    import random; random.seed(0)
    w1 = [[random.uniform(-0.5, 0.5) for _ in range(ffn_hidden)] for _ in range(d_model)]
    b1 = [0.0] * ffn_hidden
    w2 = [[random.uniform(-0.5, 0.5) for _ in range(d_model)] for _ in range(ffn_hidden)]
    b2 = [0.0] * d_model

    output = encoder_layer(token_embeddings, w1, b1, w2, b2)

    print("\nEncoder layer output:")
    for i, vec in enumerate(output):
        print(f"  token {i}: [{', '.join(f'{v:.3f}' for v in vec)}]")
