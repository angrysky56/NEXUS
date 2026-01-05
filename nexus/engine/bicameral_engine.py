"""
Bicameral Engine

The core cognitive engine that integrates all NEXUS components:
- Perception Layer: Emotional state extraction
- Regulation Layer: PID-controlled emotional homeostasis
- Routing Layer: Geometric Router (ACC) for manifold selection
- Processing Layer: Logic and Creative manifold execution
- Synthesis Layer: Output blending and formatting

This is the "brain" of NEXUS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..config import DEFAULT_CONFIG, NexusConfig
from ..core import (
    EmotionalPIDController,
    EmotionalPresets,
    EmotionalState,
    FractalEstimator,
    GeometricRouter,
    Manifold,
    RoutingDecision,
    TaskStateResolver,
    blend_outputs,
)


class ManifoldProcessor(Protocol):
    """Protocol for manifold-specific processing."""

    def process(
        self,
        input_data: np.ndarray,
        emotional_state: EmotionalState,
    ) -> np.ndarray:
        """Process input through the manifold."""
        ...


@dataclass
class ProcessingResult:
    """Result of bicameral processing."""

    # Outputs from each manifold
    logic_output: np.ndarray | None = None
    creative_output: np.ndarray | None = None
    blended_output: np.ndarray | None = None

    # Routing information
    routing: RoutingDecision | None = None

    # Emotional context
    input_state: EmotionalState | None = None
    target_state: EmotionalState | None = None
    final_state: EmotionalState | None = None

    # Task detection
    detected_task: str = "default"

    # Metadata
    processing_time_ms: float = 0.0


@dataclass
class EngineStats:
    """Statistics for engine monitoring."""

    total_requests: int = 0
    logic_routes: int = 0
    creative_routes: int = 0

    average_id: float = 0.0
    average_gate: float = 0.5

    valence_history: list[float] = field(default_factory=list)
    arousal_history: list[float] = field(default_factory=list)
    id_history: list[float] = field(default_factory=list)

    max_history: int = 100


class DefaultLogicProcessor:
    """Default Logic manifold processor - precise, sparse activations."""

    def __init__(self, config: NexusConfig):
        self.temperature = config.inference.temperature_min

    def process(
        self,
        input_data: np.ndarray,
        emotional_state: EmotionalState,
    ) -> np.ndarray:
        """
        Process through Logic manifold.

        Characteristics:
        - Low temperature (deterministic)
        - Sparse activations (top-k selection)
        - Precision-focused
        """
        # Apply temperature scaling (lower = more peaked)
        scaled = input_data / self.temperature

        # Sparse activation: keep top 20%
        threshold = np.percentile(np.abs(scaled), 80)
        sparse = np.where(np.abs(scaled) >= threshold, scaled, 0.0)

        return sparse.astype(np.float32)


class DefaultCreativeProcessor:
    """Default Creative manifold processor - high entropy, exploration."""

    def __init__(self, config: NexusConfig):
        self.temperature = config.inference.temperature_max

    def process(
        self,
        input_data: np.ndarray,
        emotional_state: EmotionalState,
    ) -> np.ndarray:
        """
        Process through Creative manifold.

        Characteristics:
        - High temperature (stochastic)
        - Dense activations (all neurons active)
        - Exploration-focused with noise injection
        """
        # Apply temperature scaling (higher = more uniform)
        scaled = input_data / self.temperature

        # Add exploration noise
        noise = np.random.normal(0, 0.1, scaled.shape)
        noisy = scaled + noise

        return noisy.astype(np.float32)


class BicameralEngine:
    """
    The Bicameral Cognitive Engine.

    This engine orchestrates the full NEXUS processing pipeline:

    1. PERCEPTION: Extract emotional state from input
    2. REGULATION: Apply PID control for smooth state transitions
    3. ROUTING: Use Geometric Router (ACC) to select manifold
    4. PROCESSING: Execute through Logic or Creative manifold
    5. SYNTHESIS: Blend outputs and format response

    The engine maintains homeostasis through multiple feedback loops:
    - Emotional PID loop: Smooths state transitions
    - E-I Balance loop: Maintains target Logic/Creative ratio
    - Dopamine loop: Rewards accurate predictions
    """

    def __init__(
        self,
        config: NexusConfig | None = None,
        logic_processor: ManifoldProcessor | None = None,
        creative_processor: ManifoldProcessor | None = None,
    ):
        """
        Initialize the Bicameral Engine.

        Args:
            config: NEXUS configuration
            logic_processor: Custom Logic manifold processor
            creative_processor: Custom Creative manifold processor
        """
        self.config = config or DEFAULT_CONFIG

        # Initialize core components
        self.pid_controller = EmotionalPIDController()
        self.router = GeometricRouter(
            threshold=self.config.fractal.target_dimension,
            temperature=self.config.fractal.gate_temperature,
            target_logic_rate=self.config.fractal.target_logic_rate,
        )
        self.fractal_estimator = FractalEstimator()
        self.task_resolver = TaskStateResolver()

        # Initialize manifold processors
        self.logic_processor = logic_processor or DefaultLogicProcessor(self.config)
        self.creative_processor = creative_processor or DefaultCreativeProcessor(self.config)

        # State
        self.current_state = EmotionalPresets.NEUTRAL
        self.stats = EngineStats()
        self.training = False

    def process(
        self,
        input_data: np.ndarray,
        text_input: str = "",
        env_state: EmotionalState | None = None,
        task_type: str | None = None,
    ) -> ProcessingResult:
        """
        Process input through the bicameral engine.

        Args:
            input_data: Numeric input (e.g., embeddings, activations)
            text_input: Optional text for task detection
            env_state: Environmental emotional state
            task_type: Override task type (otherwise auto-detected)

        Returns:
            ProcessingResult with all outputs and metadata
        """
        import time
        start_time = time.time()

        # --- PERCEPTION ---
        if env_state is None:
            env_state = self._extract_emotional_state(text_input)

        # --- TASK DETECTION ---
        if task_type is None:
            task_type = self.task_resolver.detect_task_type(text_input)

        # --- STATE RESOLUTION ---
        target_state = self.task_resolver.resolve(task_type, env_state)

        # --- REGULATION (PID Control) ---
        final_state = self.pid_controller.compute(target_state)

        # --- ROUTING (Geometric Router / ACC) ---
        routing = self.router.route(input_data, update_stats=True)

        # --- MANIFOLD PROCESSING ---
        logic_output = self.logic_processor.process(input_data, final_state)
        creative_output = self.creative_processor.process(input_data, final_state)

        # --- SYNTHESIS (Blending) ---
        blended_output = blend_outputs(
            logic_output,
            creative_output,
            routing.gate_value,
        )

        # --- UPDATE STATS ---
        self._update_stats(routing, final_state)

        # --- BUILD RESULT ---
        processing_time = (time.time() - start_time) * 1000

        return ProcessingResult(
            logic_output=logic_output,
            creative_output=creative_output,
            blended_output=blended_output,
            routing=routing,
            input_state=env_state,
            target_state=target_state,
            final_state=final_state,
            detected_task=task_type,
            processing_time_ms=processing_time,
        )

    def _extract_emotional_state(self, text: str) -> EmotionalState:
        """
        Extract emotional state from text.

        In production, this would use a sentiment analysis model.
        This is a simple heuristic placeholder.
        """
        text_lower = text.lower()

        # Simple valence heuristics
        positive_words = ["good", "great", "excellent", "thanks", "please", "help", "wonderful"]
        negative_words = ["bad", "wrong", "error", "fail", "stupid", "hate", "angry"]

        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)

        if positive_count + negative_count == 0:
            valence = 0.0
        else:
            valence = (positive_count - negative_count) / (positive_count + negative_count)

        # Simple arousal heuristics
        high_arousal_markers = ["!", "?", "urgent", "now", "immediately", "asap"]
        low_arousal_markers = ["...", "slowly", "calm", "relax"]

        high_count = sum(1 for m in high_arousal_markers if m in text_lower)
        low_count = sum(1 for m in low_arousal_markers if m in text_lower)

        arousal = 0.1 * (high_count - low_count)
        arousal = max(-1.0, min(1.0, arousal))

        return EmotionalState(valence=valence, arousal=arousal)

    def _update_stats(
        self,
        routing: RoutingDecision,
        state: EmotionalState,
    ) -> None:
        """Update engine statistics."""
        self.stats.total_requests += 1

        if routing.primary_manifold == Manifold.LOGIC:
            self.stats.logic_routes += 1
        else:
            self.stats.creative_routes += 1

        # Update running averages
        n = self.stats.total_requests
        self.stats.average_id = (
            (self.stats.average_id * (n - 1) + routing.intrinsic_dimension) / n
        )
        self.stats.average_gate = (
            (self.stats.average_gate * (n - 1) + routing.gate_value) / n
        )

        # Update history
        self.stats.valence_history.append(state.valence)
        self.stats.arousal_history.append(state.arousal)
        self.stats.id_history.append(routing.intrinsic_dimension)

        # Trim history
        if len(self.stats.valence_history) > self.stats.max_history:
            self.stats.valence_history.pop(0)
            self.stats.arousal_history.pop(0)
            self.stats.id_history.pop(0)

    def get_current_state(self) -> EmotionalState:
        """Get current emotional state."""
        return self.pid_controller.get_current_state()

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive engine statistics."""
        return {
            "total_requests": self.stats.total_requests,
            "logic_routes": self.stats.logic_routes,
            "creative_routes": self.stats.creative_routes,
            "logic_ratio": (
                self.stats.logic_routes / self.stats.total_requests
                if self.stats.total_requests > 0 else 0.5
            ),
            "average_id": self.stats.average_id,
            "average_gate": self.stats.average_gate,
            "current_state": self.get_current_state(),
            "router_stats": self.router.get_routing_stats(),
            "pid_stats": self.pid_controller.get_stats(),
        }

    def reset(self) -> None:
        """Reset engine to initial state."""
        self.pid_controller.reset()
        self.router.reset_stats()
        self.router.reset_ei_balance()
        self.stats = EngineStats()
        self.current_state = EmotionalPresets.NEUTRAL

    def train(self, mode: bool = True) -> BicameralEngine:
        """Set training mode."""
        self.training = mode
        self.router.train(mode)
        return self

    def eval(self) -> BicameralEngine:
        """Set evaluation mode."""
        self.training = False
        self.router.eval()
        return self
