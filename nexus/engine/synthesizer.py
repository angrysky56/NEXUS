"""
Synthesizer Module (The 4th Brain)

The integration layer that combines outputs from all processing streams.
Implements the "Virtual 4th Brain" concept from the Triune Architecture.

The Synthesizer:
1. Receives outputs from Logic and Creative manifolds
2. Applies grammar-guided generation
3. Conditions on semantic, contextual, and structural signals
4. Produces coherent, unified output

Reference: docs/Triune Neural Network Architecture Original Concept.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from ..core import EmotionalState, blend_outputs


@dataclass
class SynthesisResult:
    """Result of synthesis operation."""

    output: np.ndarray
    confidence: float
    blend_weights: dict[str, float]
    applied_constraints: list[str]


class Synthesizer:
    """
    The 4th Brain: Integration and Synthesis Layer.

    This module combines outputs from multiple processing pathways:
    - Reptilian (Grammar/Structure): Fixed filter, syntactic constraints
    - Mammalian (Context/Affect): Emotional and contextual embeddings
    - Neocortex (Semantic/Cognition): High-level conceptual processing

    The synthesis produces a unified output that is:
    - Grammatically correct (constrained by structural signals)
    - Contextually appropriate (guided by affective state)
    - Semantically coherent (driven by conceptual understanding)
    """

    def __init__(
        self,
        grammar_weight: float = 0.2,
        context_weight: float = 0.3,
        semantic_weight: float = 0.5,
    ):
        """
        Initialize the Synthesizer.

        Args:
            grammar_weight: Weight for grammatical/structural constraints
            context_weight: Weight for contextual/affective signals
            semantic_weight: Weight for semantic/conceptual content
        """
        self.grammar_weight = grammar_weight
        self.context_weight = context_weight
        self.semantic_weight = semantic_weight

        # Normalize weights
        total = grammar_weight + context_weight + semantic_weight
        self.grammar_weight /= total
        self.context_weight /= total
        self.semantic_weight /= total

    def synthesize(
        self,
        semantic_output: np.ndarray,
        contextual_output: np.ndarray | None = None,
        grammatical_constraints: np.ndarray | None = None,
        emotional_state: EmotionalState | None = None,
    ) -> SynthesisResult:
        """
        Synthesize unified output from multiple streams.

        Args:
            semantic_output: Output from semantic/conceptual processing
            contextual_output: Output from contextual/affective processing
            grammatical_constraints: Structural constraint signals
            emotional_state: Current emotional state for conditioning

        Returns:
            SynthesisResult with unified output
        """
        applied_constraints = []

        # Base output is semantic
        output = semantic_output.copy()

        # Apply contextual conditioning
        if contextual_output is not None:
            output = (
                self.semantic_weight * output +
                self.context_weight * contextual_output
            )
            applied_constraints.append("contextual_blend")

        # Apply grammatical constraints (masking/biasing)
        if grammatical_constraints is not None:
            # Grammatical constraints act as a soft mask
            grammar_mask = np.clip(grammatical_constraints, 0.1, 1.0)
            output = output * grammar_mask
            applied_constraints.append("grammar_mask")

        # Apply emotional modulation
        if emotional_state is not None:
            modulation = self._compute_emotional_modulation(emotional_state)
            output = output * modulation
            applied_constraints.append("emotional_modulation")

        # Compute confidence (based on output coherence)
        confidence = self._compute_confidence(output)

        return SynthesisResult(
            output=output,
            confidence=confidence,
            blend_weights={
                "grammar": self.grammar_weight,
                "context": self.context_weight,
                "semantic": self.semantic_weight,
            },
            applied_constraints=applied_constraints,
        )

    def synthesize_bicameral(
        self,
        logic_output: np.ndarray,
        creative_output: np.ndarray,
        gate_value: float,
        emotional_state: EmotionalState | None = None,
    ) -> SynthesisResult:
        """
        Simplified synthesis for bicameral outputs.

        Args:
            logic_output: Output from Logic manifold
            creative_output: Output from Creative manifold
            gate_value: Blending weight (1.0 = all Logic)
            emotional_state: Current emotional state

        Returns:
            SynthesisResult with blended output
        """
        # Blend manifold outputs
        blended = blend_outputs(logic_output, creative_output, gate_value)

        applied_constraints = ["manifold_blend"]

        # Apply emotional modulation
        if emotional_state is not None:
            modulation = self._compute_emotional_modulation(emotional_state)
            blended = blended * modulation
            applied_constraints.append("emotional_modulation")

        confidence = self._compute_confidence(blended)

        return SynthesisResult(
            output=blended,
            confidence=confidence,
            blend_weights={
                "logic": gate_value,
                "creative": 1.0 - gate_value,
            },
            applied_constraints=applied_constraints,
        )

    def _compute_emotional_modulation(
        self,
        state: EmotionalState,
    ) -> np.ndarray:
        """
        Compute modulation factor from emotional state.

        High arousal → amplify activations
        Low arousal → dampen activations
        """
        # Arousal affects amplitude
        amplitude = 1.0 + 0.5 * state.arousal

        return cast(np.ndarray, np.array([amplitude], dtype=np.float32))

    def _compute_confidence(self, output: np.ndarray) -> float:
        """
        Compute confidence score for output.

        Based on:
        - Activation sparsity (sparser = more confident)
        - Peak amplitude (higher = more confident)
        """
        # Sparsity: proportion of near-zero activations
        sparsity = np.mean(np.abs(output) < 0.01)

        # Peak: maximum activation
        peak = np.max(np.abs(output)) if output.size > 0 else 0.0
        normalized_peak = min(1.0, peak / 5.0)  # Normalize to [0, 1]

        # Combined confidence
        confidence = 0.5 * sparsity + 0.5 * normalized_peak

        return float(confidence)


class TriuneSynthesizer(Synthesizer):
    """
    Full Triune synthesis implementing all three brain layers.

    Extends base Synthesizer with explicit pathway processing.
    """

    def __init__(self) -> None:
        super().__init__()

        # Pathway-specific processors
        self.reptilian_active = False
        self.mammalian_active = False
        self.neocortex_active = True

    def process_reptilian(self, input_data: np.ndarray) -> np.ndarray:
        """
        Reptilian pathway: Grammar and structure.

        Fast, reflexive, low-level structural processing.
        - Low-bit weights (highly quantized)
        - Sparse layers
        - POS and dependency parsing signals
        """
        # Placeholder: would use actual grammatical analysis
        # Returns structural constraint signals
        return np.ones_like(input_data)  # All allowed by default

    def process_mammalian(
        self,
        input_data: np.ndarray,
        reptilian_signals: np.ndarray,
    ) -> np.ndarray:
        """
        Mammalian pathway: Context and affect.

        Pattern associative, emotional processing.
        - Conditioned by reptilian output
        - Sentiment and tone
        - Pragmatic cues
        """
        # Placeholder: would use contextual analysis
        return cast(np.ndarray, input_data * reptilian_signals)

    def process_neocortex(
        self,
        input_data: np.ndarray,
        mammalian_signals: np.ndarray,
        reptilian_signals: np.ndarray,
    ) -> np.ndarray:
        """
        Neocortex pathway: Semantic and cognition.

        Abstract thought, reasoning, high-level synthesis.
        - Conditioned by both lower pathways
        - Semantic relationships
        - Conceptual understanding
        """
        # Placeholder: would use semantic analysis
        combined_context = 0.5 * mammalian_signals + 0.5 * reptilian_signals
        return input_data * combined_context

    def full_triune_synthesis(
        self,
        input_data: np.ndarray,
        emotional_state: EmotionalState | None = None,
    ) -> SynthesisResult:
        """
        Full triune processing pipeline.

        Input → Reptilian → Mammalian → Neocortex → Synthesis
        """
        # Process through all three pathways
        reptilian = self.process_reptilian(input_data)
        mammalian = self.process_mammalian(input_data, reptilian)
        neocortex = self.process_neocortex(input_data, mammalian, reptilian)

        # Final synthesis
        return self.synthesize(
            semantic_output=neocortex,
            contextual_output=mammalian,
            grammatical_constraints=reptilian,
            emotional_state=emotional_state,
        )
