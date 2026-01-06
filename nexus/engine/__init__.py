"""
Engine module initialization.

Exports the main engine components.
"""

from .bicameral_engine import (
    BicameralEngine,
    DefaultCreativeProcessor,
    DefaultLogicProcessor,
    ManifoldProcessor,
    ProcessingResult,
)
from .synthesizer import (
    SynthesisResult,
    Synthesizer,
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
