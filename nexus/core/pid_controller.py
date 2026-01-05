"""
PID Controller for Emotional Regulation

Implements the Virtual Prefrontal Cortex from the DECCP specification.
Uses Proportional-Integral-Derivative control to smooth transitions
between emotional states and prevent "emotional whiplash".

Reference: docs/AI-Emotional-Context-Control-Plane.md Section 5.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .emotional_state import EmotionalState, EmotionalPresets


@dataclass
class PIDGains:
    """PID controller gain parameters."""

    kp: float = 0.6  # Proportional: Reaction speed
    ki: float = 0.1  # Integral: Mood memory (accumulated state)
    kd: float = 0.3  # Derivative: Damping (prevents whiplash)


@dataclass
class PIDState:
    """Internal state of the PID controller."""

    # Previous error (for derivative calculation)
    prev_error_v: float = 0.0
    prev_error_a: float = 0.0

    # Accumulated error (for integral calculation)
    integral_v: float = 0.0
    integral_a: float = 0.0

    # Current emotional state
    current_state: EmotionalState = field(default_factory=lambda: EmotionalPresets.NEUTRAL)

    # History for analysis
    history: list[EmotionalState] = field(default_factory=list)
    max_history: int = 100


class EmotionalPIDController:
    """
    PID-based emotional regulation controller.

    The controller takes an environmental state (from perception) and a target
    state (from task resolution) and computes a smoothed transition using PID
    control. This prevents the AI from having "emotional whiplash" when inputs
    change rapidly.

    From the DECCP spec:
    - Kp (Proportional): How quickly the AI adapts to new tasks
    - Ki (Integral): "Mood memory" - if user is hostile for 10 turns, system becomes wary
    - Kd (Derivative): Damping - resists rapid changes, ensures smooth transitions
    """

    def __init__(
        self,
        gains: PIDGains | None = None,
        initial_state: EmotionalState | None = None,
        integral_clamp: float = 2.0,
    ):
        """
        Initialize the PID controller.

        Args:
            gains: PID gains (defaults to recommended values from DECCP)
            initial_state: Starting emotional state
            integral_clamp: Maximum absolute value for integral windup prevention
        """
        self.gains = gains or PIDGains()
        self.integral_clamp = integral_clamp

        self.state = PIDState(
            current_state=initial_state or EmotionalPresets.NEUTRAL
        )

    def compute(
        self,
        target_state: EmotionalState,
        dt: float = 1.0,
    ) -> EmotionalState:
        """
        Compute the next emotional state using PID control.

        Args:
            target_state: The desired target state (S_opt)
            dt: Time delta since last update (default 1.0 for turn-based)

        Returns:
            The new smoothed emotional state (S_final)
        """
        current = self.state.current_state

        # --- Valence PID ---
        error_v = target_state.valence - current.valence

        # Proportional term
        p_v = self.gains.kp * error_v

        # Integral term (with anti-windup clamping)
        self.state.integral_v += error_v * dt
        self.state.integral_v = max(
            -self.integral_clamp,
            min(self.integral_clamp, self.state.integral_v)
        )
        i_v = self.gains.ki * self.state.integral_v

        # Derivative term
        derivative_v = (error_v - self.state.prev_error_v) / dt if dt > 0 else 0.0
        d_v = self.gains.kd * derivative_v

        # Total valence adjustment
        adjustment_v = p_v + i_v + d_v
        new_valence = current.valence + adjustment_v

        # Store previous error
        self.state.prev_error_v = error_v

        # --- Arousal PID ---
        error_a = target_state.arousal - current.arousal

        # Proportional term
        p_a = self.gains.kp * error_a

        # Integral term
        self.state.integral_a += error_a * dt
        self.state.integral_a = max(
            -self.integral_clamp,
            min(self.integral_clamp, self.state.integral_a)
        )
        i_a = self.gains.ki * self.state.integral_a

        # Derivative term
        derivative_a = (error_a - self.state.prev_error_a) / dt if dt > 0 else 0.0
        d_a = self.gains.kd * derivative_a

        # Total arousal adjustment
        adjustment_a = p_a + i_a + d_a
        new_arousal = current.arousal + adjustment_a

        # Store previous error
        self.state.prev_error_a = error_a

        # Create new state
        new_state = EmotionalState(valence=new_valence, arousal=new_arousal)

        # Update current state and history
        self.state.current_state = new_state
        self.state.history.append(new_state)
        if len(self.state.history) > self.state.max_history:
            self.state.history.pop(0)

        return new_state

    def reset(self, state: EmotionalState | None = None) -> None:
        """Reset the controller to initial state."""
        self.state = PIDState(
            current_state=state or EmotionalPresets.NEUTRAL
        )

    def get_mood_trend(self) -> tuple[float, float]:
        """
        Get the accumulated mood trend from integral terms.

        Returns:
            Tuple of (valence_trend, arousal_trend)
            Positive = trending up, Negative = trending down
        """
        return (self.state.integral_v, self.state.integral_a)

    def get_current_state(self) -> EmotionalState:
        """Get the current emotional state."""
        return self.state.current_state

    def get_stats(self) -> dict[str, Any]:
        """Get controller statistics for monitoring."""
        return {
            "current_state": self.state.current_state,
            "integral_v": self.state.integral_v,
            "integral_a": self.state.integral_a,
            "prev_error_v": self.state.prev_error_v,
            "prev_error_a": self.state.prev_error_a,
            "history_length": len(self.state.history),
            "gains": {
                "kp": self.gains.kp,
                "ki": self.gains.ki,
                "kd": self.gains.kd,
            },
        }


class TaskStateResolver:
    """
    Resolves target emotional state based on task type.

    Implements the State Resolution Matrix from DECCP spec.
    Uses Yerkes-Dodson law to select optimal arousal for task complexity.
    """

    # Task type to target state mapping
    TASK_STATE_MAP: dict[str, EmotionalState] = {
        "debug": EmotionalState(valence=0.0, arousal=-0.5),
        "coding": EmotionalState(valence=0.0, arousal=-0.5),
        "logic": EmotionalState(valence=0.0, arousal=-0.5),
        "math": EmotionalState(valence=0.0, arousal=-0.6),
        "creative": EmotionalState(valence=0.8, arousal=0.7),
        "ideation": EmotionalState(valence=0.8, arousal=0.7),
        "brainstorm": EmotionalState(valence=0.8, arousal=0.8),
        "writing": EmotionalState(valence=0.6, arousal=0.5),
        "chat": EmotionalState(valence=0.5, arousal=0.0),
        "conversation": EmotionalState(valence=0.5, arousal=0.0),
        "default": EmotionalState(valence=0.0, arousal=0.0),
    }

    @classmethod
    def resolve(
        cls,
        task_type: str,
        env_state: EmotionalState | None = None,
    ) -> EmotionalState:
        """
        Resolve the optimal target state for a task.

        Args:
            task_type: The detected or specified task type
            env_state: Current environmental state (for defensive adjustments)

        Returns:
            The target emotional state (S_opt)
        """
        # Get base target from task type
        task_type_lower = task_type.lower()
        target = cls.TASK_STATE_MAP.get(
            task_type_lower,
            cls.TASK_STATE_MAP["default"]
        )

        # Apply defensive adjustment if environment is hostile
        if env_state is not None and env_state.valence < -0.5:
            # Inverse regulation: counter hostility with calm
            target = EmotionalState(
                valence=max(0.2, target.valence),  # Stay positive/neutral
                arousal=-0.8,  # Very calm, stoic
            )

        return target

    @classmethod
    def detect_task_type(cls, text: str) -> str:
        """
        Simple heuristic task type detection.

        In production, this would use a classifier.
        """
        text_lower = text.lower()

        # Code/debug patterns
        if any(kw in text_lower for kw in ["debug", "error", "fix", "bug", "traceback"]):
            return "debug"
        if any(kw in text_lower for kw in ["code", "implement", "function", "class", "python"]):
            return "coding"

        # Math/logic patterns
        if any(kw in text_lower for kw in ["calculate", "math", "equation", "prove", "solve"]):
            return "math"
        if any(kw in text_lower for kw in ["analyze", "reason", "logic", "deduce"]):
            return "logic"

        # Creative patterns
        if any(kw in text_lower for kw in ["write", "story", "poem", "creative", "imagine"]):
            return "creative"
        if any(kw in text_lower for kw in ["brainstorm", "ideas", "suggest", "possibilities"]):
            return "ideation"

        # Default to chat
        return "chat"
