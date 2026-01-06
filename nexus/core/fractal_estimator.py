"""
Neural Fractal Estimator (NFE)

Estimates the Higuchi Fractal Dimension (D_H) of activation trajectories.
The target dimension D_H ≈ 1.8 represents the "Edge of Chaos" - the critical
regime where information processing is optimized.

Reference: docs/Epistemic_Engineering_Fractal_Bottleneck_Hypothesis.md
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class FractalResult:
    """Result of fractal dimension estimation."""

    dimension: float
    method: str  # "higuchi", "correlation", or "spectral"
    confidence: float  # 0.0 to 1.0
    raw_values: list[float] | None = None

    @property
    def is_critical(self) -> bool:
        """Check if dimension is in the critical regime [1.7, 1.9]."""
        return 1.7 <= self.dimension <= 1.9

    @property
    def regime(self) -> str:
        """Classify the dynamical regime."""
        if self.dimension < 1.5:
            return "ordered"  # Too smooth, over-regularized
        elif self.dimension < 1.7:
            return "sub-critical"  # Approaching edge
        elif self.dimension <= 1.9:
            return "critical"  # The sweet spot!
        elif self.dimension <= 2.0:
            return "super-critical"  # Approaching chaos
        else:
            return "chaotic"  # Too rough, unstable


class HiguchiFractalDimension:
    """
    Higuchi's algorithm for fractal dimension estimation.

    The algorithm measures the length of a curve at different scales k,
    then computes D as the slope of the log-log plot.

    For a time series X[1..N]:
        L(k) = sum of |X[m+ik] - X[m+(i-1)k]| for different starting points m
        D = slope of log(L(k)) vs log(1/k)

    Reference: Higuchi (1988), "Approach to an irregular time series on the
    basis of the fractal theory"
    """

    def __init__(self, k_max: int = 10):
        """
        Initialize the Higuchi estimator.

        Args:
            k_max: Maximum scale factor (typically 10-20)
        """
        self.k_max = k_max

    def compute(self, signal: Sequence[float]) -> FractalResult:
        """
        Compute the Higuchi Fractal Dimension of a 1D signal.

        Args:
            signal: 1D time series (e.g., activation trajectory across layers)

        Returns:
            FractalResult with estimated dimension
        """
        x = np.asarray(signal, dtype=np.float64)
        n = len(x)

        if n < 10:
            return FractalResult(
                dimension=1.5,  # Default for insufficient data
                method="higuchi",
                confidence=0.0,
            )

        # Adapt k_max to signal length
        k_max = min(self.k_max, n // 4)
        if k_max < 2:
            k_max = 2

        # Compute curve lengths at each scale k
        log_k_inv = []
        log_lengths = []

        for k in range(1, k_max + 1):
            # Average over all starting points m = 1, ..., k
            lengths_for_k = []

            for m in range(1, k + 1):
                # Build subsequence: X[m], X[m+k], X[m+2k], ...
                indices = np.arange(m - 1, n, k)
                if len(indices) < 2:
                    continue

                subsequence = x[indices]

                # Compute normalized length
                diffs = np.abs(np.diff(subsequence))
                length = np.sum(diffs) * (n - 1) / (k * len(diffs) * k)
                lengths_for_k.append(length)

            if lengths_for_k:
                avg_length = np.mean(lengths_for_k)
                if avg_length > 0:
                    log_k_inv.append(np.log(1.0 / k))
                    log_lengths.append(np.log(avg_length))

        if len(log_k_inv) < 3:
            return FractalResult(
                dimension=1.5,
                method="higuchi",
                confidence=0.0,
            )

        # Linear regression for slope (fractal dimension)
        log_k_inv = np.array(log_k_inv)
        log_lengths = np.array(log_lengths)

        # D = slope of log(L) vs log(1/k)
        A = np.vstack([log_k_inv, np.ones_like(log_k_inv)]).T
        result = np.linalg.lstsq(A, log_lengths, rcond=None)
        slope = result[0][0]

        # Compute R² for confidence
        residuals = result[1]
        if len(residuals) > 0 and residuals[0] > 0:
            ss_res = residuals[0]
            ss_tot = np.sum((log_lengths - np.mean(log_lengths)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            # Perfect fit or single point
            r_squared = 1.0

        # Clamp to valid range [1.0, 2.5]
        dimension = max(1.0, min(2.5, slope))

        return FractalResult(
            dimension=dimension,
            method="higuchi",
            confidence=max(0.0, min(1.0, r_squared)),
            raw_values=list(zip(log_k_inv.tolist(), log_lengths.tolist())),
        )


class CorrelationDimension:
    """
    Correlation Dimension (D₂) estimator.

    More robust than box-counting for finite data.
    Uses the scaling of pairwise distances.

    C(r) = (1/N²) * count(||x_i - x_j|| < r)
    D₂ = d(log C(r)) / d(log r)
    """

    def __init__(self, n_scales: int = 10):
        """
        Initialize the correlation dimension estimator.

        Args:
            n_scales: Number of scale points for regression
        """
        self.n_scales = n_scales

    def compute(self, points: np.ndarray) -> FractalResult:
        """
        Compute correlation dimension of a point cloud.

        Args:
            points: Array of shape (N, D) representing N points in D dimensions

        Returns:
            FractalResult with estimated dimension
        """
        if len(points.shape) == 1:
            points = points.reshape(-1, 1)

        n = len(points)
        if n < 5:
            return FractalResult(
                dimension=1.5,
                method="correlation",
                confidence=0.0,
            )

        # Compute pairwise distances
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d > 0:
                    dists.append(d)

        if not dists:
            return FractalResult(
                dimension=1.0,
                method="correlation",
                confidence=0.0,
            )

        dists = np.array(dists)

        # Choose scale range
        r_min = np.percentile(dists, 5)
        r_max = np.percentile(dists, 95)

        if r_min <= 0 or r_min >= r_max:
            return FractalResult(
                dimension=1.5,
                method="correlation",
                confidence=0.0,
            )

        # Compute correlation sum at each scale
        log_r = np.linspace(np.log(r_min), np.log(r_max), self.n_scales)
        log_c = []

        for lr in log_r:
            r = np.exp(lr)
            count = np.sum(dists < r)
            if count > 0:
                c = count / (n * (n - 1) / 2)
                log_c.append(np.log(c))
            else:
                log_c.append(-np.inf)

        log_c = np.array(log_c)
        valid = np.isfinite(log_c)

        if np.sum(valid) < 3:
            return FractalResult(
                dimension=1.5,
                method="correlation",
                confidence=0.0,
            )

        # Linear regression
        A = np.vstack([log_r[valid], np.ones(np.sum(valid))]).T
        result = np.linalg.lstsq(A, log_c[valid], rcond=None)
        slope = result[0][0]

        dimension = max(1.0, min(3.5, slope))

        return FractalResult(
            dimension=dimension,
            method="correlation",
            confidence=0.8,  # Approximate
        )


if HAS_TORCH:

    class NeuralFractalEstimator(nn.Module):
        """
        Neural Network-based Fractal Dimension Estimator.

        A learned estimator that can be trained to predict fractal dimension
        from activation patterns. This provides a differentiable alternative
        to the analytical Higuchi algorithm.

        Architecture: 1D CNN → Global Pool → Dense → D_H prediction
        """

        def __init__(self, input_dim: int = 64):
            """
            Initialize the neural estimator.

            Args:
                input_dim: Expected dimension of input features
            """
            super().__init__()

            # 1D Convolutional layers
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)

            # Batch normalization
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(32)

            # Global average pooling happens in forward pass

            # Dense layers
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 1)

            # Activation
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Estimate fractal dimension from trajectory.

            Args:
                x: Tensor of shape (batch, seq_len) representing trajectories

            Returns:
                Tensor of shape (batch,) with estimated D_H values
            """
            # Add channel dimension: (batch, seq_len) -> (batch, 1, seq_len)
            if x.dim() == 2:
                x = x.unsqueeze(1)

            # Convolutional layers
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))

            # Global average pooling
            x = x.mean(dim=2)  # (batch, 32)

            # Dense layers
            x = self.relu(self.fc1(x))
            x = self.fc2(x)  # (batch, 1)

            # Clamp to valid range [1.0, 2.5]
            x = torch.clamp(x.squeeze(-1), 1.0, 2.5)

            return x


class FractalEstimator:
    """
    Unified interface for fractal dimension estimation.

    Selects the best available method (neural, Higuchi, or correlation)
    based on input type and availability.
    """

    def __init__(self, method: str = "auto"):
        """
        Initialize the estimator.

        Args:
            method: "higuchi", "correlation", "neural", or "auto"
        """
        self.method = method
        self.higuchi = HiguchiFractalDimension()
        self.correlation = CorrelationDimension()

        if HAS_TORCH and method in ("neural", "auto"):
            self.neural = NeuralFractalEstimator()
        else:
            self.neural = None

    def estimate(
        self,
        signal: np.ndarray | Sequence[float],
    ) -> FractalResult:
        """
        Estimate fractal dimension of a signal.

        Args:
            signal: 1D array representing activation trajectory

        Returns:
            FractalResult with estimated dimension
        """
        signal = np.asarray(signal)

        if signal.ndim == 1:
            # 1D trajectory: use Higuchi
            return self.higuchi.compute(signal)
        else:
            # Multi-dimensional: use correlation dimension
            return self.correlation.compute(signal)

    def estimate_batch(
        self,
        signals: list[np.ndarray],
    ) -> list[FractalResult]:
        """Estimate fractal dimension for multiple signals."""
        return [self.estimate(s) for s in signals]
