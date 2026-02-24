import math
from dataclasses import dataclass, field
from typing import Optional

# The 6 formulation elements, in order (index = layer number)
LAYER_NAMES = ["root", "type", "sets", "parameters", "variables", "objective", "constraints"]


@dataclass
class MCTSNode:
    """
    One node in the 6-layer SolverLLM MCTS tree.

    Each node represents ONE formulation element. The path from root to
    any node defines a partial formulation; a complete path (depth 6)
    defines the full formulation:

        root (layer 0)
        └── Type: LP               (layer 1)
            └── Sets: i∈{1..n}    (layer 2)
                └── Parameters    (layer 3)
                    └── Variables  (layer 4)
                        └── Objective   (layer 5)
                            └── Constraints (layer 6)  ← complete

    Dynamic Expansion: if trigger=True AND local_uncertainty > η,
    selection stops at this (non-leaf) node and expansion adds MORE
    children here — trying different options for the next layer.
    """

    problem: str           # original problem text (shared across all nodes)
    layer: int = 0         # 0=root, 1=Type … 6=Constraints
    content: str = ""      # element content for this layer ("" for root)

    parent: Optional["MCTSNode"] = None
    children: list = field(default_factory=list)

    visits: int = 0
    value: float = 0.0     # running weighted mean of rewards (Q-value)

    # Prompt Backpropagation — updated after each simulation
    trigger: bool = False               # True = this element needs revision
    local_uncertainty: float = 0.0      # predictive entropy from token logprobs

    result: dict = field(default_factory=dict)  # last code execution result

    # ── UCB ──────────────────────────────────────────────────────────────────

    def ucb1(self, exploration_constant: float = 2.0) -> float:
        """
        Paper's UCB formula:
            Q_s' + c * sqrt(2 * ln(N_parent) / N_s')
        where Q_s' is the running mean reward and c=2.

        Returns inf for unvisited nodes → always explored first.
        """
        if self.visits == 0:
            return float("inf")
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(
            2.0 * math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def is_complete(self) -> bool:
        return self.layer == 6

    def element_name(self) -> str:
        if 0 <= self.layer < len(LAYER_NAMES):
            return LAYER_NAMES[self.layer]
        return "unknown"

    def get_formulation_path(self) -> dict:
        """Walk root→self, collecting {layer_name: content} for non-root nodes."""
        path = {}
        node = self
        while node is not None:
            if node.layer > 0 and node.content:
                path[LAYER_NAMES[node.layer]] = node.content
            node = node.parent
        return path

    def format_partial_formulation(self) -> str:
        """Format the path root→self as readable text for LLM prompts."""
        path = self.get_formulation_path()
        lines = []
        for name in LAYER_NAMES[1:]:
            if name in path:
                lines.append(f"**{name.capitalize()}**: {path[name]}")
        return "\n".join(lines) if lines else "(none yet)"
