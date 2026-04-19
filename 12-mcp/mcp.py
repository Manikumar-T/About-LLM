"""
Model Context Protocol (MCP): a standard protocol for connecting LLMs to
external tools and resources via a client-server architecture.

This demo simulates an MCP server, client, and the full request/response
cycle without any network or real LLM.
"""

import json
import datetime

# ── MCP message types ─────────────────────────────────────────────────────────
# MCP uses JSON-RPC 2.0 over stdio, HTTP-SSE, or WebSocket.

def jsonrpc_request(method: str, params: dict, req_id: int = 1) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})

def jsonrpc_response(result, req_id: int = 1) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})

def jsonrpc_error(code: int, message: str, req_id: int = 1) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": req_id,
                       "error": {"code": code, "message": message}})

# ── MCP Server ────────────────────────────────────────────────────────────────

class MCPServer:
    """
    An MCP server exposes three capability types:
      - tools     : callable functions (like tool calling)
      - resources : readable data sources (files, DB rows, API responses)
      - prompts   : reusable prompt templates
    """

    def __init__(self, name: str, version: str = "1.0"):
        self.name    = name
        self.version = version
        self._tools     = {}
        self._resources = {}
        self._prompts   = {}

    # ── Registration helpers ──────────────────────────────────────────────────

    def tool(self, name: str, description: str, input_schema: dict):
        def decorator(fn):
            self._tools[name] = {"fn": fn, "description": description,
                                  "inputSchema": input_schema}
            return fn
        return decorator

    def add_resource(self, uri: str, name: str, description: str, content: str,
                     mime_type: str = "text/plain"):
        self._resources[uri] = {"name": name, "description": description,
                                 "content": content, "mimeType": mime_type}

    def add_prompt(self, name: str, description: str, template: str, arguments: list):
        self._prompts[name] = {"description": description,
                                "template": template, "arguments": arguments}

    # ── Protocol handlers ─────────────────────────────────────────────────────

    def handle(self, raw_message: str) -> str:
        try:
            msg = json.loads(raw_message)
        except json.JSONDecodeError:
            return jsonrpc_error(-32700, "Parse error")

        method = msg.get("method")
        params = msg.get("params", {})
        req_id = msg.get("id", 1)

        handlers = {
            "initialize":        self._handle_initialize,
            "tools/list":        self._handle_tools_list,
            "tools/call":        self._handle_tools_call,
            "resources/list":    self._handle_resources_list,
            "resources/read":    self._handle_resources_read,
            "prompts/list":      self._handle_prompts_list,
            "prompts/get":       self._handle_prompts_get,
        }

        handler = handlers.get(method)
        if not handler:
            return jsonrpc_error(-32601, f"Method not found: {method}", req_id)
        return handler(params, req_id)

    def _handle_initialize(self, params, req_id):
        return jsonrpc_response({
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": self.name, "version": self.version},
            "capabilities": {
                "tools":     {"listChanged": False},
                "resources": {"listChanged": False},
                "prompts":   {"listChanged": False},
            },
        }, req_id)

    def _handle_tools_list(self, params, req_id):
        tools = [{"name": n, "description": t["description"],
                  "inputSchema": t["inputSchema"]}
                 for n, t in self._tools.items()]
        return jsonrpc_response({"tools": tools}, req_id)

    def _handle_tools_call(self, params, req_id):
        name      = params.get("name")
        arguments = params.get("arguments", {})
        if name not in self._tools:
            return jsonrpc_error(-32602, f"Unknown tool: {name}", req_id)
        try:
            result = self._tools[name]["fn"](**arguments)
            return jsonrpc_response(
                {"content": [{"type": "text", "text": json.dumps(result)}],
                 "isError": False}, req_id)
        except Exception as e:
            return jsonrpc_response(
                {"content": [{"type": "text", "text": str(e)}],
                 "isError": True}, req_id)

    def _handle_resources_list(self, params, req_id):
        resources = [{"uri": uri, "name": r["name"],
                      "description": r["description"], "mimeType": r["mimeType"]}
                     for uri, r in self._resources.items()]
        return jsonrpc_response({"resources": resources}, req_id)

    def _handle_resources_read(self, params, req_id):
        uri = params.get("uri")
        if uri not in self._resources:
            return jsonrpc_error(-32602, f"Resource not found: {uri}", req_id)
        r = self._resources[uri]
        return jsonrpc_response(
            {"contents": [{"uri": uri, "mimeType": r["mimeType"], "text": r["content"]}]},
            req_id)

    def _handle_prompts_list(self, params, req_id):
        prompts = [{"name": n, "description": p["description"],
                    "arguments": p["arguments"]}
                   for n, p in self._prompts.items()]
        return jsonrpc_response({"prompts": prompts}, req_id)

    def _handle_prompts_get(self, params, req_id):
        name      = params.get("name")
        arguments = params.get("arguments", {})
        if name not in self._prompts:
            return jsonrpc_error(-32602, f"Unknown prompt: {name}", req_id)
        template = self._prompts[name]["template"]
        rendered = template.format(**arguments)
        return jsonrpc_response(
            {"description": self._prompts[name]["description"],
             "messages": [{"role": "user", "content": {"type": "text", "text": rendered}}]},
            req_id)


# ── MCP Client ────────────────────────────────────────────────────────────────

class MCPClient:
    """Thin client that serialises requests and deserialises responses."""

    def __init__(self, server: MCPServer):
        self._server  = server
        self._req_id  = 0

    def _call(self, method: str, params: dict = None) -> dict:
        self._req_id += 1
        raw = jsonrpc_request(method, params or {}, self._req_id)
        response_raw = self._server.handle(raw)
        return json.loads(response_raw)

    def initialize(self):
        return self._call("initialize",
                          {"protocolVersion": "2024-11-05",
                           "clientInfo": {"name": "demo-client", "version": "1.0"}})

    def list_tools(self):
        return self._call("tools/list")

    def call_tool(self, name: str, arguments: dict):
        return self._call("tools/call", {"name": name, "arguments": arguments})

    def list_resources(self):
        return self._call("resources/list")

    def read_resource(self, uri: str):
        return self._call("resources/read", {"uri": uri})

    def list_prompts(self):
        return self._call("prompts/list")

    def get_prompt(self, name: str, arguments: dict):
        return self._call("prompts/get", {"name": name, "arguments": arguments})


# ── Build a demo MCP server ───────────────────────────────────────────────────

server = MCPServer(name="demo-mcp-server")

@server.tool(
    name="get_weather",
    description="Get current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
)
def get_weather(city: str):
    data = {"london": "15°C, Cloudy", "tokyo": "28°C, Sunny", "paris": "18°C, Partly cloudy"}
    return {"city": city, "weather": data.get(city.lower(), "20°C, Unknown")}

@server.tool(
    name="get_time",
    description="Return the current UTC time.",
    input_schema={"type": "object", "properties": {}},
)
def get_time():
    return {"utc_time": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

server.add_resource(
    uri="file:///docs/llm-guide.txt",
    name="LLM Guide",
    description="Internal guide on using LLMs.",
    content="LLMs are transformer-based models trained on large text corpora. "
            "Use temperature < 0.5 for factual tasks and > 0.8 for creative ones.",
)

server.add_prompt(
    name="summarise",
    description="Summarise a piece of text.",
    template="Summarise the following text in {style} style:\n\n{text}",
    arguments=[
        {"name": "text",  "description": "Text to summarise", "required": True},
        {"name": "style", "description": "Summary style (e.g. bullet points)", "required": True},
    ],
)


# ── Demo ──────────────────────────────────────────────────────────────────────

def pretty(label, data):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print('─'*55)
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    client = MCPClient(server)

    pretty("initialize", client.initialize())
    pretty("tools/list", client.list_tools())
    pretty("tools/call  → get_weather(city=Tokyo)",
           client.call_tool("get_weather", {"city": "Tokyo"}))
    pretty("tools/call  → get_time()",
           client.call_tool("get_time", {}))
    pretty("resources/list", client.list_resources())
    pretty("resources/read  → file:///docs/llm-guide.txt",
           client.read_resource("file:///docs/llm-guide.txt"))
    pretty("prompts/list", client.list_prompts())
    pretty("prompts/get  → summarise",
           client.get_prompt("summarise", {"text": "MCP standardises LLM tool access.",
                                           "style": "bullet points"}))
