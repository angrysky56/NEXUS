"""
Core module initialization.

Exports all core NEXUS components.
"""

from .emotional_state import (
    EmotionalState,
    EmotionalQuadrant,
    EmotionalPresets,
    InferenceParams,
)

from .pid_controller import (
    PIDGains,
    EmotionalPIDController,
    TaskStateResolver,
)

from .fractal_estimator import (
    FractalResult,
    FractalEstimator,
    HiguchiFractalDimension,
    CorrelationDimension,
)

from .geometric_router import (
    Manifold,
    RoutingDecision,
    GeometricRouter,
    blend_outputs,
)

from .dopamine_reward import (
    RewardComponents,
    DopamineRewardFunction,
    EligibilityTrace,
)

__all__ = [
    # Emotional State
    "EmotionalState",
    "EmotionalQuadrant",
    "EmotionalPresets",
    "InferenceParams",

    # PID Controller
    "PIDGains",
    "EmotionalPIDController",
    "TaskStateResolver",

    # Fractal Estimator
    "FractalResult",
    "FractalEstimator",
    "HiguchiFractalDimension",
    "CorrelationDimension",

    # Geometric Router
    "Manifold",
    "RoutingDecision",
    "GeometricRouter",
    "blend_outputs",

    # Dopamine Reward
    "RewardComponents",
    "DopamineRewardFunction",
    "EligibilityTrace",
]
