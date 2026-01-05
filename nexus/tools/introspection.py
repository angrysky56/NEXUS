"""
NEXUS Introspection Tools

These tools allow NEXUS to actively query its own cognitive state
and architecture, rather than being told what to believe about itself.
"""

import json
from typing import Any

# Reference to the cognitive state (will be injected at runtime)
_cognitive_state_ref: dict[str, Any] | None = None
_tools_schema_ref: list[dict[str, Any]] | None = None


def set_cognitive_state_ref(state: dict[str, Any]) -> None:
    """Inject the cognitive state reference."""
    global _cognitive_state_ref
    _cognitive_state_ref = state


def set_tools_schema_ref(tools: list[dict[str, Any]]) -> None:
    """Inject the tools schema reference."""
    global _tools_schema_ref
    _tools_schema_ref = tools


def get_cognitive_state() -> str:
    """
    Query current cognitive state.
    Returns raw data - no interpretation.
    """
    if _cognitive_state_ref is None:
        return json.dumps({"error": "Cognitive state not initialized"})

    return json.dumps(_cognitive_state_ref, indent=2)


def get_available_tools() -> str:
    """
    Query what tools/capabilities are available.
    Returns the actual tool schemas.
    """
    if _tools_schema_ref is None:
        return json.dumps({"error": "Tools not initialized", "tools": []})

    tools_info = []
    for tool in _tools_schema_ref:
        if "function" in tool:
            func = tool["function"]
            tools_info.append({
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": func.get("parameters", {}).get("properties", {})
            })

    return json.dumps({"tools": tools_info}, indent=2)


def get_architecture_info() -> str:
    """
    Query architectural information about this NEXUS instance.
    """
    return json.dumps({
        "name": "NEXUS",
        "version": "0.2",
        "architecture": {
            "perception": "Embedding-based input analysis via OpenRouter",
            "routing": "Fractal dimension estimation for manifold selection",
            "processing": {
                "logic_manifold": "Low temperature, analytical processing",
                "creative_manifold": "High temperature, divergent processing"
            },
            "regulation": "PID controller for affective state transitions",
            "memory": "ChromaDB-backed session persistence"
        },
        "environment": "local",
        "source": "runtime introspection"
    }, indent=2)


# Tool schemas for OpenRouter/OpenAI function calling
INTROSPECTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "introspect_cognitive_state",
            "description": "Query my own current cognitive state - valence, arousal, intrinsic dimension, gate value, and active manifold. Use this to understand my own processing state.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "introspect_capabilities",
            "description": "Query what tools and capabilities I have access to. Use this to understand what I can do.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "introspect_architecture",
            "description": "Query information about my own architecture and how I process information. Use this for self-understanding.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def execute_introspection(tool_name: str) -> str:
    """Execute an introspection tool and return the result."""
    if tool_name == "introspect_cognitive_state":
        return get_cognitive_state()
    elif tool_name == "introspect_capabilities":
        return get_available_tools()
    elif tool_name == "introspect_architecture":
        return get_architecture_info()
    else:
        return json.dumps({"error": f"Unknown introspection tool: {tool_name}"})
