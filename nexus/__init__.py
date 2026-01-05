"""
NEXUS: Neuro-Epistemic eXploration and Unified Synthesis Engine

A cognitive architecture that synthesizes:
- CallosalNet (bio-inspired multi-modal integration)
- Fractal Bottleneck Hypothesis (D_H ≈ 1.8 criticality)
- Emotional Control Plane (Russell Circumplex + Yerkes-Dodson)
- Dopamine Reward Functions (TD-learning inspired)
- Meta-Matrix (Predictive Alignment)
- Triune Architecture (Reptilian/Mammalian/Neocortex)
- Causal Compression (Iron Creche)
- Aha Connections (Connection Capacity)
"""

__version__ = "0.1.0"
__author__ = "Ty"

from .config import NexusConfig, DEFAULT_CONFIG

from .core import (
    EmotionalState,
    EmotionalPresets,
    EmotionalPIDController,
    FractalEstimator,
    FractalResult,
    GeometricRouter,
    Manifold,
    RoutingDecision,
    DopamineRewardFunction,
)

from .engine import (
    BicameralEngine,
    ProcessingResult,
    Synthesizer,
)

__all__ = [
    # Version
    "__version__",

    # Config
    "NexusConfig",
    "DEFAULT_CONFIG",

    # Core Components
    "EmotionalState",
    "EmotionalPresets",
    "EmotionalPIDController",
    "FractalEstimator",
    "FractalResult",
    "GeometricRouter",
    "Manifold",
    "RoutingDecision",
    "DopamineRewardFunction",

    # Engine
    "BicameralEngine",
    "ProcessingResult",
    "Synthesizer",
]
