"""
NEXUS Configuration Module

Centralized configuration for all NEXUS components including:
- PID Controller gains
- Fractal Bottleneck thresholds
- Temperature ranges
- Safety layer parameters
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmotionalConfig:
    """Configuration for the Emotional Control Plane (DECCP)."""

    # Russell Circumplex bounds
    valence_range: tuple[float, float] = (-1.0, 1.0)
    arousal_range: tuple[float, float] = (-1.0, 1.0)

    # PID Controller gains (from DECCP spec)
    kp: float = 0.6  # Proportional: Reaction speed
    ki: float = 0.1  # Integral: Mood memory
    kd: float = 0.3  # Derivative: Damping

    # State transition smoothing
    ema_decay: float = 0.99  # Exponential moving average for rate tracking

    # Yerkes-Dodson task complexity thresholds
    task_complexity_low: float = 0.3   # Simple tasks
    task_complexity_high: float = 0.7  # Complex tasks


@dataclass
class FractalConfig:
    """Configuration for the Fractal Bottleneck / Geometric Router."""

    # The magic number: Edge of Chaos
    target_dimension: float = 1.8

    # Routing thresholds
    logic_threshold: float = 1.8   # ID < this → Logic Manifold
    creative_threshold: float = 2.0  # ID > this → Creative Manifold

    # Soft gate temperature (lower = harder routing)
    gate_temperature: float = 1.0

    # E-I Balance parameters
    target_logic_rate: float = 0.5  # 50% Logic, 50% Creative by default
    ei_learning_rate: float = 0.01  # Homeostatic adjustment speed

    # Higuchi algorithm parameters
    k_max: int = 10  # Maximum scale for Higuchi FD


@dataclass
class InferenceConfig:
    """Configuration for LLM inference parameters."""

    # Temperature ranges (mapped from Arousal)
    temperature_min: float = 0.2   # A = -1.0 → Very deterministic
    temperature_max: float = 1.8   # A = +1.0 → Very creative
    temperature_default: float = 1.0

    # Top-P ranges (nucleus sampling)
    top_p_min: float = 0.5
    top_p_max: float = 0.95
    top_p_default: float = 0.9

    # Frequency penalty (repetition control)
    frequency_penalty_max: float = 0.5

    # Beam search width (inverse of Arousal)
    beam_width_max: int = 5
    beam_width_min: int = 1


@dataclass
class SafetyConfig:
    """Configuration for safety layers and guardrails."""

    # Valence-based refusal thresholds
    refusal_threshold_base: float = 0.5
    refusal_threshold_paranoid: float = 0.9  # V < -0.8
    refusal_threshold_trusting: float = 0.1  # V > +0.8

    # Logit bias for risky tokens (applied when V < -0.5)
    risky_token_bias: float = -5.0

    # Safety trigger threshold
    safety_valence_trigger: float = -0.5


@dataclass
class CausalConfig:
    """Configuration for the Causal Bottleneck (Iron Creche)."""

    # Intervention Sensitivity Score weights
    iss_alpha: float = 0.5  # Sensitivity reward weight
    iss_beta: float = 0.5   # Invariance penalty weight

    # Causal reward mixing
    lambda_causal: float = 1.0  # Weight of causal reward in total

    # CSAI thresholds
    csai_healthy: float = 0.7  # Good causal alignment
    csai_warning: float = 0.4  # Potential spurious correlation


@dataclass
class DopamineConfig:
    """Configuration for the Dopamine-inspired reward function."""

    # TD Learning parameters
    gamma: float = 0.99  # Discount factor for future rewards

    # Reward weights
    prospective_weight: float = 1.0   # Forward-looking accuracy
    retrospective_penalty: float = 0.3  # Anti-hallucination
    exploration_bonus: float = 0.1    # Entropy-based exploration


@dataclass
class NexusConfig:
    """Master configuration for the NEXUS system."""

    emotional: EmotionalConfig = field(default_factory=EmotionalConfig)
    fractal: FractalConfig = field(default_factory=FractalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    dopamine: DopamineConfig = field(default_factory=DopamineConfig)

    # System-wide settings
    debug: bool = False
    log_level: str = "INFO"
    device: str = "auto"  # "cpu", "cuda", or "auto"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        import dataclasses
        return {
            "emotional": dataclasses.asdict(self.emotional),
            "fractal": dataclasses.asdict(self.fractal),
            "inference": dataclasses.asdict(self.inference),
            "safety": dataclasses.asdict(self.safety),
            "causal": dataclasses.asdict(self.causal),
            "dopamine": dataclasses.asdict(self.dopamine),
            "debug": self.debug,
            "log_level": self.log_level,
            "device": self.device,
        }


# Global default configuration
DEFAULT_CONFIG = NexusConfig()
