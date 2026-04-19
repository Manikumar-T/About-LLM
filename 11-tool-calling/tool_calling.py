"""
Tool Calling (Function Calling): the model decides WHEN to call a tool and
WITH WHAT arguments. The host program executes the tool and returns the result
so the model can continue reasoning.

This demo simulates the full loop without a real LLM API.
"""

import json
import math
import datetime

# ── 1. Define tools (the "schema" sent to the model) ─────────────────────────

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a basic math expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '2 * (3 + 4)'"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Return the current UTC date and time.",
        "parameters": {"type": "object", "properties": {}},
    },
]

# ── 2. Tool implementations (run on the HOST, not inside the model) ───────────

def get_weather(city: str, unit: str = "celsius") -> dict:
    # Simulated — a real impl would call a weather API
    fake_data = {
        "london": {"temp_c": 15, "condition": "Cloudy"},
        "tokyo":  {"temp_c": 28, "condition": "Sunny"},
        "paris":  {"temp_c": 18, "condition": "Partly cloudy"},
    }
    data = fake_data.get(city.lower(), {"temp_c": 20, "condition": "Unknown"})
    temp = data["temp_c"] if unit == "celsius" else data["temp_c"] * 9 / 5 + 32
    symbol = "°C" if unit == "celsius" else "°F"
    return {"city": city, "temperature": f"{temp}{symbol}", "condition": data["condition"]}

def calculator(expression: str) -> dict:
    # Safe eval: only allow numbers and basic operators
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters in expression"}
    try:
        result = eval(expression, {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

def get_current_time() -> dict:
    now = datetime.datetime.now(datetime.timezone.utc)
    return {"utc_time": now.strftime("%Y-%m-%d %H:%M:%S UTC")}

TOOL_REGISTRY = {
    "get_weather":    get_weather,
    "calculator":     calculator,
    "get_current_time": get_current_time,
}

# ── 3. Tool dispatcher ────────────────────────────────────────────────────────

def dispatch_tool(tool_call: dict) -> str:
    name = tool_call["name"]
    args = tool_call.get("arguments", {})
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})
    result = TOOL_REGISTRY[name](**args)
    return json.dumps(result)

# ── 4. Simulated "model" that decides to call tools ───────────────────────────

def simulated_model_response(user_message: str):
    """
    A real LLM would analyse the user message and return either:
      (a) a plain text answer, or
      (b) a tool_call object specifying which tool to invoke and with what args.
    Here we hard-code responses to illustrate the pattern.
    """
    msg = user_message.lower()
    if "weather" in msg:
        city = "London"
        for c in ["london", "tokyo", "paris"]:
            if c in msg:
                city = c.capitalize()
                break
        return {"type": "tool_call", "name": "get_weather",
                "arguments": {"city": city, "unit": "celsius"}}
    if any(op in msg for op in ["+", "-", "*", "/", "calculate", "compute", "sqrt"]):
        # Extract the expression crudely
        import re
        expr = re.search(r"[\d\s\+\-\*\/\(\)\.sqrt]+", msg)
        expression = expr.group().strip() if expr else "0"
        return {"type": "tool_call", "name": "calculator",
                "arguments": {"expression": expression}}
    if "time" in msg or "date" in msg:
        return {"type": "tool_call", "name": "get_current_time", "arguments": {}}
    return {"type": "text", "content": "I can help with weather, math, and time queries."}

# ── 5. Agent loop (model ↔ tool ↔ model) ──────────────────────────────────────

def agent_loop(user_message: str):
    print(f"User : {user_message}")
    response = simulated_model_response(user_message)

    if response["type"] == "text":
        print(f"Model: {response['content']}\n")
        return

    # Model requested a tool call
    print(f"Model: [calls tool '{response['name']}' with args {response.get('arguments', {})}]")
    tool_result = dispatch_tool(response)
    print(f"Tool : {tool_result}")

    # Model receives the result and produces a final answer
    result_data = json.loads(tool_result)
    if response["name"] == "get_weather":
        final = (f"The weather in {result_data['city']} is "
                 f"{result_data['temperature']} and {result_data['condition']}.")
    elif response["name"] == "calculator":
        final = (f"The result of {result_data.get('expression')} "
                 f"is {result_data.get('result', result_data.get('error'))}.")
    else:
        final = f"The current time is {result_data.get('utc_time', 'unknown')}."

    print(f"Model: {final}\n")

# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Tool Schema (sent to model at the start of every request) ===")
    print(json.dumps(TOOLS, indent=2))
    print("\n=== Agent Loop Demos ===\n")

    agent_loop("What's the weather like in Tokyo?")
    agent_loop("Calculate 15 * (3 + 7)")
    agent_loop("What time is it right now?")
    agent_loop("Tell me a joke")
