"""
AI Agents: an LLM that autonomously plans, acts (via tools), observes results,
and iterates until a goal is reached — the ReAct (Reason + Act) loop.

This demo simulates a complete agent loop without a real LLM or API.
"""

import json
import math
import datetime

# ── Tools the agent can use ───────────────────────────────────────────────────

TOOLS = {
    "calculator": {
        "description": "Evaluate a math expression. Input: {'expression': str}",
        "fn": lambda expression: {
            "result": eval(
                expression,
                {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi}
            )
        },
    },
    "get_weather": {
        "description": "Get weather for a city. Input: {'city': str}",
        "fn": lambda city: {
            "city": city,
            "weather": {"london": "12°C Rainy", "tokyo": "27°C Sunny",
                        "paris": "18°C Cloudy"}.get(city.lower(), "22°C Clear"),
        },
    },
    "search": {
        "description": "Search for a fact. Input: {'query': str}",
        "fn": lambda query: {
            "result": {
                "speed of light":   "299,792,458 metres per second",
                "population india": "1.44 billion (2024)",
                "python creator":   "Guido van Rossum",
            }.get(query.lower(), "No result found.")
        },
    },
    "finish": {
        "description": "Return the final answer to the user. Input: {'answer': str}",
        "fn": lambda answer: {"final_answer": answer},
    },
}

# ── Agent memory (conversation + scratchpad) ──────────────────────────────────

class AgentMemory:
    def __init__(self):
        self.steps = []   # list of {"role", "content"} dicts

    def add(self, role: str, content: str):
        self.steps.append({"role": role, "content": content})

    def show(self):
        for s in self.steps:
            print(f"[{s['role'].upper()}] {s['content']}")

# ── Simulated LLM planner (replaces real model) ───────────────────────────────

def simulated_llm(task: str, memory: AgentMemory, step: int) -> dict:
    """
    A real agent sends the full memory (task + prior steps) to the LLM and
    gets back a thought + tool_call. Here we hard-code a decision tree to
    illustrate the pattern for three demo tasks.
    """
    task_lower = task.lower()

    # Task 1: multi-step math
    if "area" in task_lower and "circle" in task_lower:
        if step == 0:
            return {
                "thought": "I need to compute the area of a circle with radius 7. "
                           "Formula: π × r². I'll use the calculator.",
                "tool": "calculator",
                "args": {"expression": "pi * 7 ** 2"},
            }
        return {
            "thought": "I have the result. I can now give the final answer.",
            "tool": "finish",
            "args": {"answer": f"The area of a circle with radius 7 is "
                               f"{math.pi * 49:.4f} square units."},
        }

    # Task 2: weather + comparison
    if "weather" in task_lower and "london" in task_lower and "tokyo" in task_lower:
        if step == 0:
            return {"thought": "I need the weather in London first.",
                    "tool": "get_weather", "args": {"city": "London"}}
        if step == 1:
            return {"thought": "Now I need Tokyo's weather.",
                    "tool": "get_weather", "args": {"city": "Tokyo"}}
        return {
            "thought": "I have both. I can compare and answer.",
            "tool": "finish",
            "args": {"answer": "London: 12°C Rainy. Tokyo: 27°C Sunny. "
                               "Tokyo is warmer by 15°C."},
        }

    # Task 3: fact lookup + calculation
    if "speed of light" in task_lower:
        if step == 0:
            return {"thought": "I'll look up the speed of light.",
                    "tool": "search", "args": {"query": "speed of light"}}
        if step == 1:
            return {
                "thought": "Speed of light is 299,792,458 m/s. "
                           "How far in 10 seconds? distance = speed × time.",
                "tool": "calculator",
                "args": {"expression": "299792458 * 10"},
            }
        return {
            "thought": "Calculation done.",
            "tool": "finish",
            "args": {"answer": "Light travels 2,997,924,580 metres in 10 seconds."},
        }

    return {"thought": "I don't know how to handle this task.",
            "tool": "finish", "args": {"answer": "I cannot answer this question."}}

# ── ReAct agent loop ──────────────────────────────────────────────────────────

def run_agent(task: str, max_steps: int = 6):
    print(f"\n{'═' * 60}")
    print(f"  TASK: {task}")
    print(f"{'═' * 60}\n")

    memory = AgentMemory()
    memory.add("user", task)

    for step in range(max_steps):
        # 1. REASON — LLM produces a thought and chooses a tool
        decision = simulated_llm(task, memory, step)
        thought  = decision["thought"]
        tool     = decision["tool"]
        args     = decision["args"]

        print(f"Step {step + 1}")
        print(f"  Thought : {thought}")
        print(f"  Action  : {tool}({json.dumps(args)})")

        memory.add("thought", thought)
        memory.add("action",  f"{tool}({json.dumps(args)})")

        # 2. ACT — execute the tool
        tool_fn = TOOLS[tool]["fn"]
        result  = tool_fn(**args)

        print(f"  Observation: {json.dumps(result)}")
        memory.add("observation", json.dumps(result))
        print()

        # 3. CHECK — did the agent finish?
        if tool == "finish":
            print(f"Final Answer: {result['final_answer']}")
            return result["final_answer"]

    print("Max steps reached without a final answer.")
    return None

# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tasks = [
        "What is the area of a circle with radius 7?",
        "Compare the weather in London and Tokyo.",
        "How far does light travel in 10 seconds? Use the speed of light.",
    ]
    for task in tasks:
        run_agent(task)
        print()
