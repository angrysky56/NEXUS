"""
Emotional State Module

Implements the Russell Circumplex Model of Affect for NEXUS.
Provides the 2D continuous state space (Valence, Arousal) that governs
dynamic hyperparameter adjustment.

Reference: docs/AI-Emotional-Context-Control-Plane.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np


class EmotionalQuadrant(Enum):
    """The four quadrants of the Russell Circumplex."""

    # High Arousal, High Valence: Excited, Elated, Happy
    ACTIVATED_PLEASANT = "activated_pleasant"

    # High Arousal, Low Valence: Tense, Angry, Stressed
    ACTIVATED_UNPLEASANT = "activated_unpleasant"

    # Low Arousal, Low Valence: Sad, Depressed, Bored
    DEACTIVATED_UNPLEASANT = "deactivated_unpleasant"

    # Low Arousal, High Valence: Calm, Relaxed, Serene
    DEACTIVATED_PLEASANT = "deactivated_pleasant"


class InferenceParams(NamedTuple):
    """LLM inference parameters derived from emotional state."""

    temperature: float
    top_p: float
    frequency_penalty: float
    refusal_threshold: float
    tone_instruction: str


@dataclass
class EmotionalState:
    """
    A point in the Russell Circumplex state space.

    Attributes:
        valence: Pleasure/displeasure dimension. Range: [-1.0, +1.0]
                 -1.0 = Existential threat, maximum defense
                 +1.0 = Optimal trust, maximum collaboration

        arousal: Activation/energy dimension. Range: [-1.0, +1.0]
                 -1.0 = Hibernation, minimal compute, deterministic
                 +1.0 = Maximum entropy, chaotic exploration
    """

    valence: float = 0.0
    arousal: float = 0.0

    def __post_init__(self) -> None:
        """Clamp values to valid range."""
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(-1.0, min(1.0, self.arousal))

    @property
    def quadrant(self) -> EmotionalQuadrant:
        """Determine which quadrant this state falls into."""
        if self.arousal >= 0:
            if self.valence >= 0:
                return EmotionalQuadrant.ACTIVATED_PLEASANT
            else:
                return EmotionalQuadrant.ACTIVATED_UNPLEASANT
        else:
            if self.valence >= 0:
                return EmotionalQuadrant.DEACTIVATED_PLEASANT
            else:
                return EmotionalQuadrant.DEACTIVATED_UNPLEASANT

    @property
    def magnitude(self) -> float:
        """Distance from the neutral origin (0, 0)."""
        return math.sqrt(self.valence ** 2 + self.arousal ** 2)

    @property
    def angle(self) -> float:
        """Angle in radians from positive valence axis."""
        return math.atan2(self.arousal, self.valence)

    def to_polar(self) -> tuple[float, float]:
        """Convert to polar coordinates (magnitude, angle)."""
        return (self.magnitude, self.angle)

    @classmethod
    def from_polar(cls, magnitude: float, angle: float) -> EmotionalState:
        """Create from polar coordinates."""
        valence = magnitude * math.cos(angle)
        arousal = magnitude * math.sin(angle)
        return cls(valence=valence, arousal=arousal)

    def blend(self, other: EmotionalState, weight: float = 0.5) -> EmotionalState:
        """
        Blend two emotional states.

        Args:
            other: The state to blend with
            weight: Weight for the other state (0.0 = self, 1.0 = other)
        """
        weight = max(0.0, min(1.0, weight))
        return EmotionalState(
            valence=self.valence * (1 - weight) + other.valence * weight,
            arousal=self.arousal * (1 - weight) + other.arousal * weight,
        )

    def distance_to(self, other: EmotionalState) -> float:
        """Euclidean distance to another state."""
        return math.sqrt(
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2
        )

    def to_inference_params(
        self,
        temp_min: float = 0.2,
        temp_max: float = 1.8,
        top_p_min: float = 0.5,
        top_p_max: float = 0.95,
    ) -> InferenceParams:
        """
        Map emotional state to LLM inference parameters.

        This implements the Actuation Layer from DECCP spec.
        """
        # Temperature: Linear scaling from Arousal
        # A = -1.0 → T = temp_min (deterministic)
        # A = +1.0 → T = temp_max (chaotic)
        temperature = temp_min + (temp_max - temp_min) * (self.arousal + 1) / 2

        # Top-P: Sigmoidal scaling from Arousal
        # Low arousal → narrow nucleus, High arousal → wide nucleus
        sigmoid_arousal = 1 / (1 + math.exp(-2 * self.arousal))
        top_p = top_p_min + (top_p_max - top_p_min) * sigmoid_arousal

        # Frequency Penalty: Positive correlation with Arousal
        # High arousal → penalize repetition (force novelty)
        frequency_penalty = max(0.0, self.arousal * 0.5)

        # Refusal Threshold: Inverse linear from Valence
        # High V (+1) → low threshold (trusting)
        # Low V (-1) → high threshold (paranoid)
        refusal_threshold = 0.5 - (0.4 * self.valence)

        # Tone instruction based on Valence
        if self.valence > 0.5:
            tone = "Adopt a warm, expansive, collaborative tone."
        elif self.valence < -0.5:
            tone = "Adopt a defensive, boundary-enforcing, concise tone."
        elif self.valence < 0.0:
            tone = "Adopt a neutral, objective, careful tone."
        else:
            tone = "Adopt a balanced, professional tone."

        return InferenceParams(
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            refusal_threshold=refusal_threshold,
            tone_instruction=tone,
        )

    def __repr__(self) -> str:
        return f"EmotionalState(V={self.valence:+.2f}, A={self.arousal:+.2f}, Q={self.quadrant.name})"


# --- Preset States ---

class EmotionalPresets:
    """Common emotional state presets for different scenarios."""

    # Neutral baseline
    NEUTRAL = EmotionalState(valence=0.0, arousal=0.0)

    # Task-optimized states (from Yerkes-Dodson)
    LOGIC_MODE = EmotionalState(valence=0.0, arousal=-0.5)   # Low entropy for precision
    CREATIVE_MODE = EmotionalState(valence=0.8, arousal=0.7)  # High entropy for exploration
    DEBUGGING = EmotionalState(valence=0.0, arousal=-0.6)     # Very low entropy
    BRAINSTORMING = EmotionalState(valence=0.8, arousal=0.8)  # Maximum creativity

    # Defensive states
    HOSTILE_USER = EmotionalState(valence=-0.6, arousal=-0.8)  # Stoic, defensive
    SYSTEM_ERROR = EmotionalState(valence=-0.3, arousal=-0.6)  # Safe mode
    ADVERSARIAL = EmotionalState(valence=-0.9, arousal=-0.9)   # Maximum defense

    # Collaborative states
    FLOW_STATE = EmotionalState(valence=1.0, arousal=0.5)     # Peak performance
    FRIENDLY_CHAT = EmotionalState(valence=0.5, arousal=0.0)  # Warm baseline


def compute_emotional_distance_matrix(
    states: list[EmotionalState]
) -> np.ndarray:
    """Compute pairwise distances between emotional states."""
    n = len(states)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = states[i].distance_to(states[j])
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix
