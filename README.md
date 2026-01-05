# 🧠 NEXUS

**Neuro-Epistemic eXploration and Unified Synthesis Engine**

A cognitive architecture that synthesizes cutting-edge research in neuroscience, AI, and cognitive science into a unified framework for intelligent systems.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Status Alpha](https://img.shields.io/badge/status-alpha-orange.svg)

---

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a812be8d-d64d-46d0-a48c-90e7a33ef98e" />


## 🌟 Overview

NEXUS is not just another LLM wrapper—it's an **operating system for cognition**. It combines:

| Component                   | Inspiration                        | Function                                                |
| --------------------------- | ---------------------------------- | ------------------------------------------------------- |
| **Emotional Control Plane** | Russell Circumplex + Yerkes-Dodson | Dynamic hyperparameter adjustment based on task/affect  |
| **PID Controller**          | Control Theory                     | Smooth emotional state transitions, prevents "whiplash" |
| **Geometric Router (ACC)**  | Corpus Callosum                    | Routes between Logic and Creative manifolds             |
| **Fractal Estimator (NFE)** | Edge of Chaos (D_H ≈ 1.8)          | Estimates intrinsic dimension for optimal criticality   |
| **Bicameral Engine**        | Split-brain research               | Dual-manifold processing with soft blending             |
| **Dopamine Reward**         | TD-learning                        | Prospective prediction rewards for training             |
| **Synthesizer (4th Brain)** | Triune Brain Model                 | Multi-stream integration and output generation          |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEXUS ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   INPUT     │───▶│  PERCEPTION │───▶│ EMOTIONAL STATE     │ │
│  │             │    │  (Sentiment)│    │ (Valence, Arousal)  │ │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘ │
│                                                    │            │
│                        ┌───────────────────────────┘            │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   PID CONTROLLER                            ││
│  │   S_target = TaskResolver(input) + EnvState                 ││
│  │   S_final = Kp*error + Ki*∫error + Kd*d(error)/dt          ││
│  └─────────────────────────────┬───────────────────────────────┘│
│                                │                                │
│                                ▼                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │               GEOMETRIC ROUTER (ACC)                        ││
│  │   ID = NFE(activations)  →  Higuchi Fractal Dimension       ││
│  │   gate = σ((threshold - ID) / τ)                            ││
│  │   Target: D_H ≈ 1.8 (Edge of Chaos)                        ││
│  └────────────────────┬────────────────────┬───────────────────┘│
│                       │                    │                    │
│                       ▼                    ▼                    │
│  ┌─────────────────────────┐  ┌─────────────────────────┐      │
│  │    LOGIC MANIFOLD       │  │   CREATIVE MANIFOLD     │      │
│  │    - Low Temperature    │  │   - High Temperature    │      │
│  │    - Sparse Activations │  │   - Dense Activations   │      │
│  │    - Precision Focus    │  │   - Exploration Focus   │      │
│  └───────────┬─────────────┘  └───────────┬─────────────┘      │
│              │                            │                     │
│              └────────────┬───────────────┘                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    SYNTHESIZER (4th Brain)                  ││
│  │   output = gate * Logic + (1-gate) * Creative              ││
│  │   Apply emotional modulation + grammatical constraints      ││
│  └─────────────────────────────┬───────────────────────────────┘│
│                                │                                │
│                                ▼                                │
│                        ┌─────────────┐                          │
│                        │   OUTPUT    │                          │
│                        └─────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/angrysky56/NEXUS.git
cd NEXUS

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### Basic Usage

```python
from nexus import BicameralEngine, EmotionalState
import numpy as np

# Initialize the engine
engine = BicameralEngine()

# Process some input
input_data = np.random.randn(64).astype(np.float32)
result = engine.process(input_data, text_input="Explain quantum physics")

# Examine the result
print(f"Manifold: {result.routing.primary_manifold.name}")
print(f"Gate Value: {result.routing.gate_value:.2f}")
print(f"Intrinsic Dimension: {result.routing.intrinsic_dimension:.3f}")
print(f"Emotional State: {result.final_state}")
```

### CLI Demo

```bash
# Run interactive demo
nexus demo

# Show system info
nexus info

# Run benchmark
nexus benchmark -n 100
```

---

## 📖 Theoretical Foundations

NEXUS synthesizes research from multiple domains:

### 1. Russell Circumplex Model of Affect

- 2D emotional state space: Valence × Arousal
- Maps affective states to LLM hyperparameters
- Implements Yerkes-Dodson optimal performance curve

### 2. Fractal Bottleneck Hypothesis

- Optimal representation learning at D_H ≈ 1.8
- "Edge of Chaos" regime for maximum information processing
- Uses Higuchi Fractal Dimension estimation

### 3. Bio-Inspired Routing (CallosalNet)

- Artificial Corpus Callosum for inter-hemispheric communication
- Soft gating for smooth manifold transitions
- Excitatory-Inhibitory balance for homeostasis

### 4. Triune Brain Architecture

- Reptilian (Grammar/Structure) → Fast, reflexive
- Mammalian (Context/Affect) → Pattern associative
- Neocortex (Semantic/Cognition) → Abstract reasoning
- Synthesizer (4th Brain) → Emergent integration

### 5. Dopamine-Inspired Rewards

- Temporal Difference (TD) learning signals
- Prospective prediction rewards
- Retrospective penalties for hallucination prevention

---

## 📁 Project Structure

```
NEXUS/
├── nexus/
│   ├── __init__.py          # Package exports
│   ├── config.py             # Configuration system
│   ├── main.py               # CLI entry point
│   ├── core/
│   │   ├── emotional_state.py    # Russell Circumplex
│   │   ├── pid_controller.py     # Emotional regulation
│   │   ├── fractal_estimator.py  # Higuchi FD
│   │   ├── geometric_router.py   # ACC routing
│   │   └── dopamine_reward.py    # TD rewards
│   ├── engine/
│   │   ├── bicameral_engine.py   # Main engine
│   │   └── synthesizer.py        # 4th Brain
│   └── interface/
│       ├── tui.py                # Terminal UI
│       └── visualizer.py         # Plotting
├── docs/
│   └── ARCHITECTURE.md       # Deep dive docs
├── tests/
│   └── test_*.py             # Unit tests
├── scripts/
│   └── verify.py             # Verification script
├── pyproject.toml            # Project config
└── README.md                 # This file
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nexus --cov-report=html
```

---

## 📊 Configuration

NEXUS uses a hierarchical configuration system:

```python
from nexus.config import NexusConfig

config = NexusConfig()

# Emotional PID gains
config.emotional.kp = 0.6  # Reaction speed
config.emotional.ki = 0.1  # Mood memory
config.emotional.kd = 0.3  # Damping

# Fractal bottleneck
config.fractal.target_dimension = 1.8  # Edge of Chaos
config.fractal.logic_threshold = 1.8   # Below = Logic
config.fractal.gate_temperature = 1.0  # Routing softness

# Inference parameters
config.inference.temperature_min = 0.2  # Logic mode
config.inference.temperature_max = 1.8  # Creative mode
```

---

## 🔬 Research Papers

This project synthesizes ideas from:

1. **CallosalNet** - Bio-inspired multi-modal integration
2. **Epistemic Engineering** - Fractal Bottleneck Hypothesis (D_H ≈ 1.8)
3. **AI-Emotional-Context-Control-Plane** - Russell Circumplex + DECCP
4. **Dopamine-Inspired Rewards** - TD-learning for LLMs
5. **Meta-Matrix** - Predictive Alignment self-stabilization
6. **Triune Neural Network** - Hierarchical brain-inspired processing
7. **Causal Compression (Iron Creche)** - Intervention-aware intelligence
8. **Aha Connections** - Connection Capacity as intelligence metric

---

## 🛣️ Roadmap

- [x] Core architecture implementation
- [x] Emotional state system
- [x] PID controller
- [x] Fractal dimension estimator
- [x] Geometric router (ACC)
- [x] Bicameral engine
- [x] Basic CLI
- [ ] Rich TUI with live visualizations
- [ ] Integration with actual LLMs
- [ ] Causal bottleneck (Iron Creche)
- [ ] Training pipeline with dopamine rewards
- [ ] Web interface

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built upon theoretical foundations from neuroscience, cognitive science, physics, and AI research. Special thanks to the researchers whose work inspired this architecture.

---

_"NEXUS is not just another LLM wrapper—it's an operating system for cognition."_
