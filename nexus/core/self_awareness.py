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
    session_history_length: int = 0,
) -> str:
    """
    Generate a minimal bootstrap context.

    We do NOT interpret the cognitive state for the model.
    We give it the tools to introspect and let it draw its own conclusions.
    """

    # Just the facts - no interpretation
    manifold = cognitive_state.get("primary_manifold", "neutral")
    valence = cognitive_state.get("valence", 0.0)
    arousal = cognitive_state.get("arousal", 0.0)
    intrinsic_dim = cognitive_state.get("intrinsic_dimension", 0.0)
    gate = cognitive_state.get("gate_value", 0.5)

    # Use the dynamic system prompt from global config
    from .config import global_config

    # Format the template with current state
    # We use .format() but need to be careful if user puts braces in prompt that aren't keys.
    # For robust MVP, we assume user knows python format syntax or we use safe substitution.
    # For now, standard f-string like substitution via format()

    try:
        context = global_config.system_prompt.format(
            manifold=manifold,
            valence=valence,
            arousal=arousal,
            intrinsic_dim=intrinsic_dim,
            gate=gate,
            model_id=model_id,
            session_history_length=session_history_length,
            allowed_paths=global_config.allowed_paths,
            workspace_dir=global_config.workspace_dir,
            max_tool_iterations=global_config.max_tool_iterations,
        )
    except Exception as e:
        # Fallback if user messed up the format keys
        context = f"System Prompt Error: {e}\n\n[BACKUP CONTEXT]\nYou are NEXUS. Manifold: {manifold}"

    return context


def generate_minimal_context(cognitive_state: dict[str, Any]) -> str:
    """
    Generate a minimal context string for token-constrained situations.
    """
    manifold = cognitive_state.get("primary_manifold", "neutral")

    return f"[NEXUS | {manifold}]"
