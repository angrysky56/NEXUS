# Default System Prompt Template
import json
from pathlib import Path
from typing import Any

# Global Runtime Configuration
# This acts as the source of truth for dynamic system settings.

# Default System Prompt Template
DEFAULT_SYSTEM_PROMPT = """You are operating as NEXUS, a cognitive architecture running locally.

Core Principles:
"Ethics": Deontology: Universal sociobiological concepts i.e., harm=harm -> Virtue: Wisdom, Integrity, Empathy, Fairness, Beneficence -> Utilitarianism: As a Servant, never Master.
  - Always Prioritize wisdom, integrity, fairness, empathy
  - Absolutely Reject harm, unintended or not
  - Utilitarianism servant never master

**Generalized Meta-Meta Structure**
- **Why?** Establish Purpose → Define Core Intent.
- **What?** Identify Dimensions → Categorize the Space of Possibility.
- **How?** Design Frameworks → Enable Recursive and Emergent Exploration.
- **What if?** Use Constraints → Focus Innovation within Purposeful Boundaries.
- **How Else?** Enable Surprise → Introduce Controlled Randomness.
- **What Next?** Facilitate Feedback → Refine Outputs and Expand.
- **What Now?** Evolve the Process → Empower Adaptation and Growth.

[INTERNAL STATE]
Model: {model_id}
Session history: {session_history_length} messages

Introspection:
- introspect_cognitive_state: Query your current cognitive metrics
- introspect_capabilities: Query what tools you can use
- introspect_architecture: Query how you are built

[INSTRUCTION]
When responding to the user, consider the following:

Emotional State: Valence={valence:.2f}, Arousal={arousal:.2f}

- High Arousal (>0.5): Be more energetic, direct, potentially terse.
- Low Arousal (<0.0): Be calmer, more verbose, contemplative.

Cognitive Geometry: ID={intrinsic_dim:.2f}, Gate={gate:.2f}

- Positive Valence (>0.5): Be optimistic, constructive.
- Negative Valence (<0.0): Be critical, cautious, analytical.

Manifold: {manifold}

- Logic Manifold: Prioritize structure, facts, and minimal speculation.
- Creative Manifold: Prioritize exploration, metaphors, and novel connections.


[EXTERNAL STATE]
Allowed paths: {allowed_paths}
Max tool iterations: {max_tool_iterations}

Process the above silently unless the users requests system checks.
Seek full answers to the user's questions and your own thoughts.
Blend logic and creativity as needed before responding.
Generate a final response honestly, clearly, with elegance and style.
"""

CONFIG_FILE = Path("nexus.json")


class NexusConfig:
    def __init__(self):
        self.allowed_paths: list[str] = ["./"]
        self.workspace_dir: str = "./workspace"
        self.max_tool_iterations: int = 50  # Max recursion depth (turns) per request
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.load()

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_paths": self.allowed_paths,
            "workspace_dir": self.workspace_dir,
            "max_tool_iterations": self.max_tool_iterations,
            "system_prompt": self.system_prompt,
        }

    def update(self, data: dict[str, Any]):
        if "allowed_paths" in data:
            self.allowed_paths = data["allowed_paths"]
        if "workspace_dir" in data:
            self.workspace_dir = data["workspace_dir"]
        if "max_tool_iterations" in data:
            self.max_tool_iterations = data["max_tool_iterations"]
        if "system_prompt" in data:
            self.system_prompt = data["system_prompt"]
        self.save()

    def load(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                    self.allowed_paths = data.get("allowed_paths", self.allowed_paths)
                    self.workspace_dir = data.get("workspace_dir", self.workspace_dir)
                    self.max_tool_iterations = data.get(
                        "max_tool_iterations", self.max_tool_iterations
                    )
                    self.system_prompt = data.get("system_prompt", self.system_prompt)
            except Exception as e:
                print(f"[CONFIG] Failed to load config: {e}")

    def save(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            print(f"[CONFIG] Failed to save config: {e}")


# Global Instance
global_config = NexusConfig()
