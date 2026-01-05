"""
NEXUS Core Module Tests

Tests for emotional state, PID controller, fractal estimator, and router.
"""

import pytest
import numpy as np

from nexus.core import (
    EmotionalState,
    EmotionalPresets,
    EmotionalQuadrant,
    InferenceParams,
    EmotionalPIDController,
    PIDGains,
    TaskStateResolver,
    FractalEstimator,
    FractalResult,
    HiguchiFractalDimension,
    GeometricRouter,
    Manifold,
    RoutingDecision,
    DopamineRewardFunction,
)


class TestEmotionalState:
    """Tests for EmotionalState."""

    def test_creation(self):
        """Test basic creation."""
        state = EmotionalState(valence=0.5, arousal=-0.3)
        assert state.valence == 0.5
        assert state.arousal == -0.3

    def test_clamping(self):
        """Test value clamping to [-1, 1]."""
        state = EmotionalState(valence=2.0, arousal=-3.0)
        assert state.valence == 1.0
        assert state.arousal == -1.0

    def test_quadrant_detection(self):
        """Test quadrant classification."""
        # Activated Pleasant (high V, high A)
        state = EmotionalState(valence=0.5, arousal=0.5)
        assert state.quadrant == EmotionalQuadrant.ACTIVATED_PLEASANT

        # Activated Unpleasant (low V, high A)
        state = EmotionalState(valence=-0.5, arousal=0.5)
        assert state.quadrant == EmotionalQuadrant.ACTIVATED_UNPLEASANT

        # Deactivated Unpleasant (low V, low A)
        state = EmotionalState(valence=-0.5, arousal=-0.5)
        assert state.quadrant == EmotionalQuadrant.DEACTIVATED_UNPLEASANT

        # Deactivated Pleasant (high V, low A)
        state = EmotionalState(valence=0.5, arousal=-0.5)
        assert state.quadrant == EmotionalQuadrant.DEACTIVATED_PLEASANT

    def test_magnitude(self):
        """Test magnitude calculation."""
        state = EmotionalState(valence=0.6, arousal=0.8)
        assert abs(state.magnitude - 1.0) < 0.01  # 0.6² + 0.8² = 1.0

    def test_blend(self):
        """Test state blending."""
        state1 = EmotionalState(valence=0.0, arousal=0.0)
        state2 = EmotionalState(valence=1.0, arousal=1.0)

        blended = state1.blend(state2, weight=0.5)
        assert abs(blended.valence - 0.5) < 0.01
        assert abs(blended.arousal - 0.5) < 0.01

    def test_to_inference_params(self):
        """Test mapping to inference parameters."""
        state = EmotionalState(valence=0.0, arousal=0.0)
        params = state.to_inference_params()

        assert isinstance(params, InferenceParams)
        assert 0.2 <= params.temperature <= 1.8
        assert 0.5 <= params.top_p <= 0.95


class TestPIDController:
    """Tests for EmotionalPIDController."""

    def test_initialization(self):
        """Test controller initialization."""
        controller = EmotionalPIDController()
        state = controller.get_current_state()
        assert state.valence == 0.0
        assert state.arousal == 0.0

    def test_compute_converges(self):
        """Test that controller converges to target."""
        controller = EmotionalPIDController()
        target = EmotionalState(valence=0.8, arousal=0.5)

        # Iterate several times
        for _ in range(20):
            result = controller.compute(target)

        # Should be close to target
        assert abs(result.valence - target.valence) < 0.1
        assert abs(result.arousal - target.arousal) < 0.1

    def test_reset(self):
        """Test controller reset."""
        controller = EmotionalPIDController()
        controller.compute(EmotionalState(valence=1.0, arousal=1.0))
        controller.reset()

        state = controller.get_current_state()
        assert state.valence == 0.0
        assert state.arousal == 0.0


class TestTaskStateResolver:
    """Tests for TaskStateResolver."""

    def test_resolve_known_tasks(self):
        """Test resolution of known task types."""
        state = TaskStateResolver.resolve("debug")
        assert state.arousal < 0  # Debug should be low arousal

        state = TaskStateResolver.resolve("creative")
        assert state.arousal > 0  # Creative should be high arousal

    def test_detect_task_type(self):
        """Test task type detection."""
        assert TaskStateResolver.detect_task_type("fix this bug") == "debug"
        assert TaskStateResolver.detect_task_type("write a poem") == "creative"
        assert TaskStateResolver.detect_task_type("calculate 2+2") == "math"


class TestFractalEstimator:
    """Tests for FractalEstimator."""

    def test_higuchi_on_smooth_signal(self):
        """Test Higuchi FD on a smooth (low D_H) signal."""
        higuchi = HiguchiFractalDimension()

        # Smooth sine wave
        t = np.linspace(0, 4*np.pi, 100)
        signal = np.sin(t)

        result = higuchi.compute(signal)
        assert isinstance(result, FractalResult)
        assert result.dimension < 1.5  # Smooth signals have low FD

    def test_higuchi_on_noisy_signal(self):
        """Test Higuchi FD on a noisy (high D_H) signal."""
        higuchi = HiguchiFractalDimension()

        # Pure noise
        signal = np.random.randn(100)

        result = higuchi.compute(signal)
        assert result.dimension > 1.5  # Noisy signals have high FD

    def test_estimator_interface(self):
        """Test unified FractalEstimator interface."""
        estimator = FractalEstimator()
        signal = np.random.randn(50)

        result = estimator.estimate(signal)
        assert isinstance(result, FractalResult)
        assert 1.0 <= result.dimension <= 2.5


class TestGeometricRouter:
    """Tests for GeometricRouter."""

    def test_initialization(self):
        """Test router initialization."""
        router = GeometricRouter(threshold=1.8)
        assert router.base_threshold == 1.8

    def test_route_simple(self):
        """Test simple routing by ID value."""
        router = GeometricRouter(threshold=1.8)

        # Low ID → Logic
        decision = router.route_simple(1.5)
        assert decision.primary_manifold == Manifold.LOGIC
        assert decision.gate_value > 0.5

        # High ID → Creative
        decision = router.route_simple(2.0)
        assert decision.primary_manifold == Manifold.CREATIVE
        assert decision.gate_value < 0.5

    def test_soft_gating(self):
        """Test soft gate values."""
        router = GeometricRouter(threshold=1.8, temperature=1.0)

        # At threshold, gate should be ~0.5
        decision = router.route_simple(1.8)
        assert 0.4 < decision.gate_value < 0.6

    def test_stats_accumulation(self):
        """Test that stats accumulate correctly."""
        router = GeometricRouter()

        for _ in range(10):
            signal = np.random.randn(50)
            router.route(signal)

        stats = router.get_routing_stats()
        assert stats["total_routes"] == 10


class TestDopamineReward:
    """Tests for DopamineRewardFunction."""

    def test_compute_simple_reward(self):
        """Test simple reward computation."""
        reward_fn = DopamineRewardFunction()

        # High probability → high reward
        result = reward_fn.compute_simple_reward(0.9, entropy=1.0)
        assert result.prospective > -1.0  # log(0.9) ≈ -0.1

        # Low probability → low reward
        result = reward_fn.compute_simple_reward(0.1, entropy=1.0)
        assert result.prospective < -1.0  # log(0.1) ≈ -2.3

    def test_aha_reward(self):
        """Test Aha! moment reward."""
        reward_fn = DopamineRewardFunction()

        # High entropy before, low after → big Aha!
        aha = reward_fn.compute_aha_reward(entropy_before=2.0, entropy_after=0.1)
        assert aha > 10.0  # 2.0 / 0.1 = 20.0

        # Low entropy before and after → small Aha
        aha = reward_fn.compute_aha_reward(entropy_before=0.5, entropy_after=0.5)
        assert aha == 1.0  # 0.5 / 0.5 = 1.0


class TestIntegration:
    """Integration tests across components."""

    def test_full_pipeline(self):
        """Test full processing pipeline."""
        from nexus.engine import BicameralEngine

        engine = BicameralEngine()

        # Process some input
        input_data = np.random.randn(64).astype(np.float32)
        result = engine.process(input_data, text_input="test input")

        # Check all components produced output
        assert result.logic_output is not None
        assert result.creative_output is not None
        assert result.blended_output is not None
        assert result.routing is not None
        assert result.final_state is not None

    def test_task_routing_consistency(self):
        """Test that different tasks route differently."""
        from nexus.engine import BicameralEngine

        engine = BicameralEngine()

        # Process logic-oriented input
        logic_input = np.random.randn(64).astype(np.float32) * 0.5  # Lower variance
        logic_result = engine.process(logic_input, text_input="debug this code")

        # Reset and process creative input
        engine.reset()
        creative_input = np.random.randn(64).astype(np.float32) * 2.0  # Higher variance
        creative_result = engine.process(creative_input, text_input="write a poem")

        # Creative should have different target state
        assert creative_result.target_state.arousal != logic_result.target_state.arousal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
