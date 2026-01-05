"""
Dopamine-Inspired Reward Function

Implements TD-learning inspired reward signals for NEXUS training.
Focuses on prospective prediction (forward-looking) rather than
retrospective correlation (backward-looking).

Reference: docs/A dopamine inspired reward function for an LLM.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RewardComponents:
    """Individual components of the reward signal."""

    prospective: float = 0.0      # Forward-looking accuracy
    retrospective: float = 0.0    # Retrospective penalty (negative)
    exploration: float = 0.0      # Entropy-based exploration bonus
    total: float = 0.0            # Combined reward

    td_error: float = 0.0         # Temporal Difference error (delta)


@dataclass
class DopamineConfig:
    """Configuration for dopamine reward computation."""

    gamma: float = 0.99           # Discount factor
    alpha_prospective: float = 1.0  # Weight for forward-looking reward
    alpha_retrospective: float = 0.3  # Weight for retrospective penalty
    beta_exploration: float = 0.1   # Weight for exploration bonus


class DopamineRewardFunction:
    """
    Dopamine-inspired reward function for LLM training.

    Key principle: Reward accurate PROSPECTIVE predictions while penalizing
    retrospective correlations that don't generalize.

    The reward has three components:
    1. R_prediction: log P(actual_next | context, generated) - Forward accuracy
    2. R_retrospective: -α * log P(context | actual_next) - Anti-hallucination
    3. R_exploration: β * H(next_token_distribution) - Entropy bonus

    The TD error δ = r + γV(s') - V(s) is computed for value learning.
    """

    def __init__(self, config: DopamineConfig | None = None):
        """
        Initialize the reward function.

        Args:
            config: Reward configuration parameters
        """
        self.config = config or DopamineConfig()
        self.value_estimates: dict[str, float] = {}
        self.reward_history: list[RewardComponents] = []

    def compute_reward(
        self,
        predicted_probs: np.ndarray,
        actual_token_idx: int,
        context_given_outcome_prob: float | None = None,
        current_state_value: float = 0.0,
        next_state_value: float = 0.0,
    ) -> RewardComponents:
        """
        Compute the full reward signal.

        Args:
            predicted_probs: Probability distribution over vocabulary
            actual_token_idx: Index of the actual next token
            context_given_outcome_prob: P(context | outcome) for retrospective
            current_state_value: V(s) - current state value estimate
            next_state_value: V(s') - next state value estimate

        Returns:
            RewardComponents with all reward signals
        """
        # --- Prospective Reward (Forward-Looking Accuracy) ---
        # R_prediction = log P(actual | context)
        if actual_token_idx < len(predicted_probs):
            p_actual = max(predicted_probs[actual_token_idx], 1e-10)
            r_prospective = math.log(p_actual)
        else:
            r_prospective = -10.0  # Heavy penalty for invalid prediction

        # --- Retrospective Penalty (Anti-Hallucination) ---
        # R_retro = -α * log P(context | outcome)
        # This penalizes predictions that rely on spurious correlations
        if context_given_outcome_prob is not None:
            p_retro = max(context_given_outcome_prob, 1e-10)
            r_retrospective = -self.config.alpha_retrospective * math.log(p_retro)
        else:
            r_retrospective = 0.0

        # --- Exploration Bonus (Entropy) ---
        # R_exploration = β * H(prediction_distribution)
        # Encourages diversity and prevents mode collapse
        entropy = self._compute_entropy(predicted_probs)
        r_exploration = self.config.beta_exploration * entropy

        # --- Total Immediate Reward ---
        r_immediate = (
            self.config.alpha_prospective * r_prospective +
            r_retrospective +
            r_exploration
        )

        # --- TD Error (Dopamine Signal) ---
        # δ = r + γV(s') - V(s)
        td_error = (
            r_immediate +
            self.config.gamma * next_state_value -
            current_state_value
        )

        result = RewardComponents(
            prospective=r_prospective,
            retrospective=r_retrospective,
            exploration=r_exploration,
            total=r_immediate,
            td_error=td_error,
        )

        self.reward_history.append(result)
        return result

    def compute_simple_reward(
        self,
        predicted_prob: float,
        entropy: float = 0.0,
    ) -> RewardComponents:
        """
        Simplified reward computation.

        Args:
            predicted_prob: Probability assigned to actual token
            entropy: Entropy of the prediction distribution

        Returns:
            RewardComponents
        """
        r_prospective = math.log(max(predicted_prob, 1e-10))
        r_exploration = self.config.beta_exploration * entropy
        r_total = self.config.alpha_prospective * r_prospective + r_exploration

        return RewardComponents(
            prospective=r_prospective,
            retrospective=0.0,
            exploration=r_exploration,
            total=r_total,
            td_error=r_total,  # Simplified: TD error ≈ immediate reward
        )

    def _compute_entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of a probability distribution."""
        # Filter out zeros and very small values
        probs = probs[probs > 1e-10]
        if len(probs) == 0:
            return 0.0

        # Renormalize
        probs = probs / probs.sum()

        # H = -sum(p * log(p))
        return float(-np.sum(probs * np.log(probs)))

    def compute_aha_reward(
        self,
        entropy_before: float,
        entropy_after: float,
    ) -> float:
        """
        Compute the "Aha!" reward for insight moments.

        From docs/aha_connections.md:
        Aha = ΔEntropy_before / ΔEntropy_after

        The "energy release" from collapsing high entropy to low entropy.

        Args:
            entropy_before: Entropy before the insight
            entropy_after: Entropy after the insight

        Returns:
            Aha reward (dopamine-like signal)
        """
        if entropy_after < 0.01:
            # Near-perfect insight (entropy collapsed to near zero)
            return entropy_before * 10.0

        return entropy_before / entropy_after

    def get_average_reward(self, last_n: int = 100) -> dict[str, float]:
        """Get average reward components over recent history."""
        if not self.reward_history:
            return {
                "prospective": 0.0,
                "retrospective": 0.0,
                "exploration": 0.0,
                "total": 0.0,
                "td_error": 0.0,
            }

        recent = self.reward_history[-last_n:]
        return {
            "prospective": float(np.mean([r.prospective for r in recent])),
            "retrospective": float(np.mean([r.retrospective for r in recent])),
            "exploration": float(np.mean([r.exploration for r in recent])),
            "total": float(np.mean([r.total for r in recent])),
            "td_error": float(np.mean([r.td_error for r in recent])),
        }

    def reset_history(self) -> None:
        """Clear reward history."""
        self.reward_history = []


class EligibilityTrace:
    """
    Eligibility trace for TD(λ) learning.

    Implements credit assignment decay - recent actions get more credit
    than distant ones.
    """

    def __init__(self, decay_rate: float = 0.9):
        """
        Initialize eligibility trace.

        Args:
            decay_rate: λ parameter for trace decay
        """
        self.decay_rate = decay_rate
        self.traces: dict[str, float] = {}

    def update(self, state_id: str, learning_signal: float = 1.0) -> None:
        """
        Update trace for a state.

        Args:
            state_id: Identifier for the state
            learning_signal: Signal strength (typically 1.0)
        """
        # Decay all existing traces
        for key in self.traces:
            self.traces[key] *= self.decay_rate

        # Add/update current state
        self.traces[state_id] = (
            self.traces.get(state_id, 0.0) * self.decay_rate + learning_signal
        )

    def get_trace(self, state_id: str) -> float:
        """Get current trace value for a state."""
        return self.traces.get(state_id, 0.0)

    def get_all_traces(self) -> dict[str, float]:
        """Get all active traces."""
        return {k: v for k, v in self.traces.items() if v > 0.01}

    def reset(self) -> None:
        """Clear all traces."""
        self.traces = {}
