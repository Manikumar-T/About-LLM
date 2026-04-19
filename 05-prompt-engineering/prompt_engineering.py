"""
Prompt Engineering: structuring input text to guide LLM behaviour.
Demonstrates common patterns without calling a real API.
"""

def zero_shot_prompt(task, input_text):
    return f"{task}\n\nInput: {input_text}\nOutput:"

def few_shot_prompt(task, examples, input_text):
    shots = "\n\n".join(
        f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples
    )
    return f"{task}\n\n{shots}\n\nInput: {input_text}\nOutput:"

def chain_of_thought_prompt(question):
    return (
        f"Question: {question}\n"
        "Let's think step by step:\n"
    )

def system_user_prompt(system, user):
    return f"[SYSTEM]\n{system}\n\n[USER]\n{user}"

def role_prompt(role, task):
    return f"You are a {role}. {task}"

if __name__ == "__main__":
    separator = "─" * 60

    print("1. ZERO-SHOT PROMPT")
    print(separator)
    print(zero_shot_prompt(
        task="Classify the sentiment of the following review as Positive or Negative.",
        input_text="The product broke after one day. Very disappointed."
    ))
    print()

    print("2. FEW-SHOT PROMPT")
    print(separator)
    examples = [
        {"input": "I love this movie!", "output": "Positive"},
        {"input": "Terrible experience, never again.", "output": "Negative"},
    ]
    print(few_shot_prompt(
        task="Classify sentiment.",
        examples=examples,
        input_text="Pretty good, but could be better."
    ))
    print()

    print("3. CHAIN-OF-THOUGHT PROMPT")
    print(separator)
    print(chain_of_thought_prompt(
        "If a train travels 60 km/h for 2.5 hours, how far does it go?"
    ))
    print()

    print("4. SYSTEM / USER PROMPT (chat models)")
    print(separator)
    print(system_user_prompt(
        system="You are a concise assistant. Answer in one sentence.",
        user="What is the capital of France?"
    ))
    print()

    print("5. ROLE PROMPTING")
    print(separator)
    print(role_prompt(
        role="senior Python engineer with 10 years of experience",
        task="Review the following code and suggest improvements."
    ))
