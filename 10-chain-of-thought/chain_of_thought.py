"""
Chain-of-Thought (CoT): prompting the model to reason step-by-step before
answering, improving accuracy on complex tasks.
Demonstrates CoT prompt patterns and a simple rule-based "reasoning" engine.
"""

# ── Prompt builders ───────────────────────────────────────────────────────────

def standard_prompt(question):
    return f"Question: {question}\nAnswer:"

def cot_prompt(question):
    return f"Question: {question}\nLet's think step by step:\n"

def zero_shot_cot(question):
    return f"Question: {question}\nThink carefully, then provide the answer.\nReasoning:\n"

def few_shot_cot(examples, question):
    shots = []
    for ex in examples:
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(ex["steps"]))
        shots.append(
            f"Question: {ex['question']}\n"
            f"Let's think step by step:\n{steps}\n"
            f"Answer: {ex['answer']}"
        )
    body = "\n\n".join(shots)
    return f"{body}\n\nQuestion: {question}\nLet's think step by step:"

# ── Toy step-by-step arithmetic solver ───────────────────────────────────────

def solve_arithmetic_cot(expression):
    """
    Very simple: handles 'A op B op C ...' chains.
    Returns list of (step_description, running_total) pairs.
    """
    import re
    tokens = re.split(r'\s+', expression.strip())
    result = float(tokens[0])
    steps  = [f"Start with {result}"]
    i = 1
    while i < len(tokens) - 1:
        op  = tokens[i]
        rhs = float(tokens[i + 1])
        if op == '+':
            result += rhs
            steps.append(f"Add {rhs} → {result}")
        elif op == '-':
            result -= rhs
            steps.append(f"Subtract {rhs} → {result}")
        elif op == '*':
            result *= rhs
            steps.append(f"Multiply by {rhs} → {result}")
        elif op == '/':
            result /= rhs
            steps.append(f"Divide by {rhs} → {result}")
        i += 2
    return steps, result

if __name__ == "__main__":
    sep = "─" * 60

    print("1. STANDARD PROMPT (no reasoning)")
    print(sep)
    print(standard_prompt("If a train travels at 60 km/h for 2.5 hours, how far does it go?"))
    print()

    print("2. ZERO-SHOT CHAIN-OF-THOUGHT")
    print(sep)
    print(cot_prompt("If a train travels at 60 km/h for 2.5 hours, how far does it go?"))
    print()

    print("3. FEW-SHOT CHAIN-OF-THOUGHT")
    print(sep)
    examples = [
        {
            "question": "Sarah has 5 apples. She buys 3 more and eats 2. How many does she have?",
            "steps": ["Start: 5 apples", "Buy 3 → 5+3 = 8", "Eat 2 → 8-2 = 6"],
            "answer": "6",
        },
        {
            "question": "A rectangle is 4m wide and 7m long. What is its area?",
            "steps": ["Area = width × length", "Area = 4 × 7 = 28"],
            "answer": "28 m²",
        },
    ]
    print(few_shot_cot(examples, "A room is 5m by 6m. What is the floor area?"))
    print()

    print("4. STEP-BY-STEP ARITHMETIC SOLVER")
    print(sep)
    for expr in ["100 + 25 - 10 * 2", "200 / 4 + 50"]:
        steps, answer = solve_arithmetic_cot(expr)
        print(f"Expression: {expr}")
        for step in steps:
            print(f"  {step}")
        print(f"Final answer: {answer}")
        print()
