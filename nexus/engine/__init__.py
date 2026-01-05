"""
Engine module initialization.

Exports the main engine components.
"""

from .bicameral_engine import (
    BicameralEngine,
    ProcessingResult,
    ManifoldProcessor,
    DefaultLogicProcessor,
    DefaultCreativeProcessor,
)

from .synthesizer import (
    Synthesizer,
    SynthesisResult,
    TriuneSynthesizer,
)

__all__ = [
    # Bicameral Engine
    "BicameralEngine",
    "ProcessingResult",
    "ManifoldProcessor",
    "DefaultLogicProcessor",
    "DefaultCreativeProcessor",

    # Synthesizer
    "Synthesizer",
    "SynthesisResult",
    "TriuneSynthesizer",
]
