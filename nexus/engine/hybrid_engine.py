
import logging
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np

from ..core import EmotionalPIDController, Manifold, TaskStateResolver
from ..core.self_awareness import generate_system_context
from ..integrations.openrouter import OpenRouterClient
from .bicameral_engine import BicameralEngine
from .processors import OpenRouterProcessor

logger = logging.getLogger(__name__)


class HybridEngine(BicameralEngine):
    """
    Hybrid Engine that combines:
    1. OpenRouter Embeddings -> Fractal Estimator / Geometric Router
    2. OpenRouter LLM -> Logic/Creative Manifolds
    3. Dynamic Self-Awareness Context Generation
    """

    def __init__(
        self,
        client: OpenRouterClient,
        logic_model: str,
        creative_model: str,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.client = client
        self.logic_model = logic_model
        self.creative_model = creative_model

        self.history_embeddings: list[np.ndarray] = []

        # Initialize Emotional Control Plane
        self.pid_controller = EmotionalPIDController()
        self.task_resolver = TaskStateResolver()

    async def process_stream(
        self,
        user_input: str,
        history: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        api_key: str | None = None,
        include_reasoning: bool = False
    ) -> AsyncGenerator[dict[str, Any], None]:

        logger.info(f"[ENGINE] Processing started - history_len={len(history)}, tools_count={len(tools or [])}, reasoning={include_reasoning}")
        logger.debug(f"[ENGINE] Input preview: {user_input[:100]}..." if len(user_input) > 100 else f"[ENGINE] Input: {user_input}")

        # 1. PERCEPTION (Embeddings)
        # We use the current input embedding + recent history to form a trajectory
        logger.info("[ENGINE] Stage 1/6: PERCEPTION - Getting embeddings...")
        try:
            current_embedding = await self.client.get_embeddings(user_input, api_key=api_key)
            logger.info(f"[ENGINE] Embedding received: dim={len(current_embedding)}")
        except Exception as e:
            logger.error(f"[ENGINE] Embedding failed: {e}")
            yield {"type": "error", "content": f"Embedding failed: {str(e)}"}
            return

        vec = np.array(current_embedding)

        # Maintain a sliding window of history for fractal analysis
        self.history_embeddings.append(vec)
        if len(self.history_embeddings) > 20:
            self.history_embeddings.pop(0)

        # Create a trajectory matrix (Time, Dimensions)
        # If history is short, we might pad or just use what we have
        if len(self.history_embeddings) < 3:
            trajectory = np.vstack([vec, vec, vec]) # minimal shape
        else:
            trajectory = np.vstack(self.history_embeddings)
        logger.debug(f"[ENGINE] Trajectory matrix: shape={trajectory.shape}, embedding_history={len(self.history_embeddings)}")

        # 2. ROUTING
        # Use trajectory to estimate ID
        logger.info("[ENGINE] Stage 2/6: ROUTING - Estimating intrinsic dimension...")
        routing = self.router.route(trajectory.T, update_stats=True)
        logger.info(f"[ENGINE] Routing result: ID={routing.intrinsic_dimension:.3f}, gate={routing.gate_value:.3f}, manifold={routing.primary_manifold.value}")

        # 3. EMOTIONAL REGULATION (PID)
        # Detect task type from input
        logger.info("[ENGINE] Stage 3/6: EMOTIONAL REGULATION - PID control...")
        task_type = self.task_resolver.detect_task_type(user_input)
        logger.debug(f"[ENGINE] Detected task type: {task_type}")

        # Determine target state for this task
        target_state = self.task_resolver.resolve(task_type)
        logger.debug(f"[ENGINE] Target emotional state: valence={target_state.valence:.2f}, arousal={target_state.arousal:.2f}")

        # Smoothly transition to target state
        current_emotional_state = self.pid_controller.compute(target_state)
        logger.info(f"[ENGINE] PID output: valence={current_emotional_state.valence:.3f}, arousal={current_emotional_state.arousal:.3f}")

        # 4. MANIFOLD SELECTION
        logger.info("[ENGINE] Stage 4/6: MANIFOLD SELECTION...")
        is_logic = routing.primary_manifold == Manifold.LOGIC
        selected_model = self.logic_model if is_logic else self.creative_model

        # Build cognitive state dict for context generation
        cognitive_state = {
            "valence": current_emotional_state.valence,
            "arousal": current_emotional_state.arousal,
            "intrinsic_dimension": routing.intrinsic_dimension,
            "gate_value": routing.gate_value,
            "primary_manifold": routing.primary_manifold.value,
            "model": selected_model
        }

        # NEXUS runs the temp if supported
        if await self.client.supports_parameter(selected_model, "temperature"):
            temperature = 0.2 if is_logic else 0.9
        else:
            # Fallback to model default if parameter not supported or not determined
            temperature = await self.client.get_model_default_temp(selected_model) or 1.0

        logger.info(f"[ENGINE] Selected: manifold={'LOGIC' if is_logic else 'CREATIVE'}, model={selected_model}, temp={temperature}")

        # 5. GENERATE DYNAMIC SELF-AWARENESS CONTEXT
        # This injects real-time cognitive state into the system prompt
        logger.info("[ENGINE] Stage 5/6: SELF-AWARENESS CONTEXT - Generating system prompt...")
        system_context = generate_system_context(
            cognitive_state=cognitive_state,
            tools=tools or [],
            model_id=selected_model,
            session_history_length=len(history)
        )
        logger.debug(f"[ENGINE] System context generated: {len(system_context)} chars")

        # Create processor with dynamic context
        target_processor = OpenRouterProcessor(
            client=self.client,
            model=selected_model,
            temperature=temperature,
            system_prompt=system_context,
            name="logic" if is_logic else "creative"
        )

        # Yield Cognitive State Update
        yield {
            "type": "cognitive_update",
            "data": cognitive_state
        }

        # 6. GENERATION
        logger.info("[ENGINE] Stage 6/6: GENERATION - Streaming LLM response...")
        chunk_count = 0
        async for chunk in target_processor.process_chat(history, user_input, tools, api_key=api_key, include_reasoning=include_reasoning):
            chunk_count += 1
            if chunk.get("type") == "tool_call_chunk":
                tc = chunk.get("tool_calls", [{}])[0]
                logger.debug(f"[ENGINE] Tool call chunk: idx={tc.get('index')}, name={tc.get('function', {}).get('name', '...')}")
            yield chunk

        logger.info(f"[ENGINE] Generation complete - yielded {chunk_count} chunks")

