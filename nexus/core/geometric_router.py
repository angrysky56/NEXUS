"""
Geometric Router (Artificial Corpus Callosum)

The ACC routes information between Logic and Creative manifolds based on
the Intrinsic Dimension (ID) of the input. This implements the bio-inspired
routing from CallosalNet and the Bicameral architecture.

Key features:
- Soft gating: Smooth transition between manifolds via sigmoid gate
- E-I Balance: Homeostatic regulation of routing ratios
- NFE Integration: Uses fractal dimension for routing decisions

Reference: docs/CallosalNet: Artifical Corpus Callosum.md
Reference: core/router.py from BM-GMoE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

import numpy as np

from .fractal_estimator import FractalEstimator, FractalResult


class Manifold(Enum):
    """The two cognitive manifolds."""

    LOGIC = "logic"      # Sparse, fractal attractor for rigorous reasoning
    CREATIVE = "creative"  # High-entropy, isotropic for fluid association


class RoutingDecision(NamedTuple):
    """Result of a routing decision."""

    primary_manifold: Manifold
    gate_value: float  # 0.0 = Full Creative, 1.0 = Full Logic
    intrinsic_dimension: float
    fractal_result: FractalResult
    is_hard_route: bool  # True if gate is near 0 or 1


@dataclass
class EIBalanceState:
    """Excitatory-Inhibitory balance state for homeostatic routing."""

    # Inhibitory weight: modulates the routing threshold
    inhibitory_weight: float = 0.0

    # Exponential moving average of logic routing rate
    ema_logic_rate: float = 0.5

    # Target proportion of Logic routing
    target_rate: float = 0.5

    # Learning rate for homeostatic updates
    ei_lr: float = 0.01


@dataclass
class RouterStats:
    """Statistics for monitoring router behavior."""

    total_routes: int = 0
    logic_routes: int = 0
    creative_routes: int = 0
    average_id: float = 0.0
    average_gate: float = 0.5
    id_history: list[float] = field(default_factory=list)
    gate_history: list[float] = field(default_factory=list)
    max_history: int = 100

    @property
    def logic_ratio(self) -> float:
        """Proportion of Logic routing."""
        if self.total_routes == 0:
            return 0.5
        return self.logic_routes / self.total_routes


class GeometricRouter:
    """
    The Artificial Corpus Callosum (ACC).

    Routes inputs to Logic or Creative manifolds based on Intrinsic Dimension.
    Uses soft gating for smooth transitions and E-I balance for homeostasis.

    The routing logic:
    - ID < threshold (1.8) → Logic Manifold (sparse, precise)
    - ID >= threshold → Creative Manifold (dense, exploratory)

    The soft gate allows blending:
    - gate_value = sigmoid((threshold - ID) / temperature)
    - Output = gate * Logic + (1 - gate) * Creative
    """

    def __init__(
        self,
        threshold: float = 1.8,
        temperature: float = 1.0,
        target_logic_rate: float = 0.5,
        ei_lr: float = 0.01,
        use_ei_balance: bool = True,
    ):
        """
        Initialize the router.

        Args:
            threshold: Base ID threshold (the Fractal Bottleneck target)
            temperature: Softness of routing (lower = harder)
            target_logic_rate: Target proportion of Logic routing
            ei_lr: E-I balance learning rate
            use_ei_balance: Whether to use homeostatic adjustment
        """
        self.base_threshold = threshold
        self.temperature = temperature
        self.use_ei_balance = use_ei_balance

        self.ei_state = EIBalanceState(
            target_rate=target_logic_rate,
            ei_lr=ei_lr,
        )

        self.fractal_estimator = FractalEstimator()
        self.stats = RouterStats()
        self.training = False

    @property
    def effective_threshold(self) -> float:
        """Current threshold after E-I modulation."""
        return self.base_threshold + self.ei_state.inhibitory_weight

    def route(
        self,
        activations: np.ndarray,
        update_stats: bool = True,
    ) -> RoutingDecision:
        """
        Route activations to the appropriate manifold.

        Args:
            activations: Activation trajectory (e.g., across layers)
            update_stats: Whether to update routing statistics

        Returns:
            RoutingDecision with manifold, gate value, and ID
        """
        # Estimate Intrinsic Dimension
        fractal_result = self.fractal_estimator.estimate(activations)
        current_id = fractal_result.dimension

        # Compute effective threshold
        threshold = self.effective_threshold

        # Hard routing decision
        is_logic = current_id < threshold
        primary_manifold = Manifold.LOGIC if is_logic else Manifold.CREATIVE

        # Soft gate value (1.0 = Full Logic, 0.0 = Full Creative)
        gate_value = self._sigmoid(
            (threshold - current_id) / self.temperature
        )

        # Determine if this is a hard route (near 0 or 1)
        is_hard_route = gate_value < 0.1 or gate_value > 0.9

        # Update E-I balance during training
        if self.training and self.use_ei_balance:
            self._update_ei_balance(is_logic)

        # Update statistics
        if update_stats:
            self._update_stats(current_id, gate_value, is_logic)

        return RoutingDecision(
            primary_manifold=primary_manifold,
            gate_value=gate_value,
            intrinsic_dimension=current_id,
            fractal_result=fractal_result,
            is_hard_route=is_hard_route,
        )

    def route_simple(self, id_value: float) -> RoutingDecision:
        """
        Route based on a pre-computed ID value.

        Args:
            id_value: Pre-computed intrinsic dimension

        Returns:
            RoutingDecision
        """
        threshold = self.effective_threshold
        is_logic = id_value < threshold
        gate_value = self._sigmoid((threshold - id_value) / self.temperature)

        return RoutingDecision(
            primary_manifold=Manifold.LOGIC if is_logic else Manifold.CREATIVE,
            gate_value=gate_value,
            intrinsic_dimension=id_value,
            fractal_result=FractalResult(
                dimension=id_value,
                method="external",
                confidence=1.0,
            ),
            is_hard_route=gate_value < 0.1 or gate_value > 0.9,
        )

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)

    def _update_ei_balance(self, is_logic: bool) -> None:
        """
        Update E-I balance based on routing decision.

        Homeostatic Inhibitory Plasticity Rule:
        - If routing to Logic too often → increase threshold
        - If routing to Creative too often → decrease threshold
        """
        # Update EMA of logic rate
        current_rate = 1.0 if is_logic else 0.0
        self.ei_state.ema_logic_rate = (
            0.99 * self.ei_state.ema_logic_rate +
            0.01 * current_rate
        )

        # Homeostatic adjustment
        deviation = self.ei_state.ema_logic_rate - self.ei_state.target_rate
        self.ei_state.inhibitory_weight += self.ei_state.ei_lr * deviation

    def _update_stats(
        self,
        id_value: float,
        gate_value: float,
        is_logic: bool,
    ) -> None:
        """Update routing statistics."""
        self.stats.total_routes += 1
        if is_logic:
            self.stats.logic_routes += 1
        else:
            self.stats.creative_routes += 1

        # Update running averages
        n = self.stats.total_routes
        self.stats.average_id = (
            (self.stats.average_id * (n - 1) + id_value) / n
        )
        self.stats.average_gate = (
            (self.stats.average_gate * (n - 1) + gate_value) / n
        )

        # Update history
        self.stats.id_history.append(id_value)
        self.stats.gate_history.append(gate_value)

        if len(self.stats.id_history) > self.stats.max_history:
            self.stats.id_history.pop(0)
            self.stats.gate_history.pop(0)

    def train(self, mode: bool = True) -> GeometricRouter:
        """Set training mode (enables E-I updates)."""
        self.training = mode
        return self

    def eval(self) -> GeometricRouter:
        """Set evaluation mode (disables E-I updates)."""
        self.training = False
        return self

    def get_routing_stats(self) -> dict[str, Any]:
        """Get current routing statistics."""
        return {
            "threshold": self.base_threshold,
            "inhibitory_weight": self.ei_state.inhibitory_weight,
            "effective_threshold": self.effective_threshold,
            "ema_logic_rate": self.ei_state.ema_logic_rate,
            "target_rate": self.ei_state.target_rate,
            "temperature": self.temperature,
            "total_routes": self.stats.total_routes,
            "logic_ratio": self.stats.logic_ratio,
            "average_id": self.stats.average_id,
            "average_gate": self.stats.average_gate,
        }

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self.stats = RouterStats()

    def reset_ei_balance(self) -> None:
        """Reset E-I balance to initial state."""
        self.ei_state = EIBalanceState(
            target_rate=self.ei_state.target_rate,
            ei_lr=self.ei_state.ei_lr,
        )


def blend_outputs(
    logic_output: np.ndarray,
    creative_output: np.ndarray,
    gate_value: float,
) -> np.ndarray:
    """
    Blend outputs from Logic and Creative manifolds.

    Args:
        logic_output: Output from Logic manifold
        creative_output: Output from Creative manifold
        gate_value: Blending weight (1.0 = all Logic, 0.0 = all Creative)

    Returns:
        Blended output
    """
    return gate_value * logic_output + (1 - gate_value) * creative_output
