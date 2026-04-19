"""
Fine-tuning concepts: demonstrates the data format and training loop idea
for supervised fine-tuning (SFT) and parameter-efficient fine-tuning (LoRA).
No ML framework required — pure Python illustrations.
"""

import math
import random

random.seed(0)

# ── 1. SFT dataset format ─────────────────────────────────────────────────────

SFT_DATASET = [
    {"prompt": "What is 2 + 2?",         "completion": "4"},
    {"prompt": "Capital of Germany?",    "completion": "Berlin"},
    {"prompt": "Translate 'hello' to Spanish.", "completion": "Hola"},
]

def format_sft_example(example):
    return f"### Instruction:\n{example['prompt']}\n### Response:\n{example['completion']}"

# ── 2. Simplified cross-entropy loss ─────────────────────────────────────────

def cross_entropy_loss(logit_correct, all_logits):
    """How surprised was the model by the correct token?"""
    e = [math.exp(l) for l in all_logits]
    prob_correct = math.exp(logit_correct) / sum(e)
    return -math.log(prob_correct + 1e-9)

# ── 3. Gradient descent step (scalar demo) ───────────────────────────────────

def gradient_descent_step(param, gradient, lr=0.01):
    return param - lr * gradient

# ── 4. LoRA: Low-Rank Adaptation concept ─────────────────────────────────────

def lora_forward(x, W, A, B, alpha=1.0, r=4):
    """
    LoRA replaces W with W + (alpha/r) * B @ A.
    Only A and B are trained; W is frozen.
    x: input vector, W: frozen weight matrix
    A: (r × in), B: (out × r) — tiny trainable matrices
    """
    # Original frozen path
    Wx = [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]

    # Low-rank update: x → A → B → scale
    Ax = [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]
    BAx = [sum(B[i][j] * Ax[j] for j in range(len(Ax))) for i in range(len(B))]
    scale = alpha / r
    delta = [scale * v for v in BAx]

    return [w + d for w, d in zip(Wx, delta)]

if __name__ == "__main__":
    print("─── 1. SFT Dataset Format ───")
    for ex in SFT_DATASET:
        print(format_sft_example(ex))
        print()

    print("─── 2. Cross-Entropy Loss Demo ───")
    all_logits = [2.0, 0.5, 0.1, 0.3]   # model scores for 4 vocab tokens
    correct_logit = all_logits[0]         # correct token is token 0
    loss = cross_entropy_loss(correct_logit, all_logits)
    print(f"Logits: {all_logits}")
    print(f"Loss when correct token has highest logit: {loss:.4f}")

    all_logits2 = [0.1, 0.5, 2.0, 0.3]  # now correct token is NOT highest
    loss2 = cross_entropy_loss(correct_logit, all_logits2)
    print(f"Loss when correct token has lowest logit:  {loss2:.4f}\n")

    print("─── 3. Gradient Descent Step ───")
    param = 0.8
    grad  = 0.4
    updated = gradient_descent_step(param, grad, lr=0.1)
    print(f"param={param}, gradient={grad} → updated param={updated:.4f}\n")

    print("─── 4. LoRA Forward Pass ───")
    in_dim, out_dim, r = 4, 4, 2
    W = [[random.gauss(0, 0.1) for _ in range(in_dim)] for _ in range(out_dim)]
    A = [[random.gauss(0, 0.1) for _ in range(in_dim)] for _ in range(r)]
    B = [[0.0 for _ in range(r)] for _ in range(out_dim)]  # B init to 0
    x = [1.0, 0.5, 0.3, 0.8]

    normal_out = [sum(W[i][j] * x[j] for j in range(in_dim)) for i in range(out_dim)]
    lora_out   = lora_forward(x, W, A, B)

    print(f"Normal output: {[round(v,4) for v in normal_out]}")
    print(f"LoRA   output: {[round(v,4) for v in lora_out]}")
    print("(identical at init because B=0 — LoRA updates B and A during training)")
