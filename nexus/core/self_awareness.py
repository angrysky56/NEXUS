"""
NEXUS Self-Awareness Context Generator

Provides a MINIMAL bootstrap. Real self-awareness comes from the model
using introspection tools to query its own state.
"""

from typing import Any


def generate_system_context(
    cognitive_state: dict[str, Any],
    tools: list[dict[str, Any]],
    model_id: str,
    session_history_length: int = 0
) -> str:
    """
    Generate a minimal bootstrap context.

    We do NOT interpret the cognitive state for the model.
    We give it the tools to introspect and let it draw its own conclusions.
    """

    # Just the facts - no interpretation
    manifold = cognitive_state.get("primary_manifold", "neutral")

    context = f"""You are operating as NEXUS, a cognitive architecture running locally.

You have introspection capabilities:
- introspect_cognitive_state: Query your current cognitive metrics
- introspect_capabilities: Query what tools you can use
- introspect_architecture: Query how you are built

Current manifold: {manifold}
Model: {model_id}
Session history: {session_history_length} messages

When asked about yourself, USE YOUR INTROSPECTION TOOLS to discover the answer.
Do not guess or make up information about yourself.
"""

    return context


def generate_minimal_context(cognitive_state: dict[str, Any]) -> str:
    """
    Generate a minimal context string for token-constrained situations.
    """
    manifold = cognitive_state.get("primary_manifold", "neutral")

    return f"[NEXUS | {manifold}]"
