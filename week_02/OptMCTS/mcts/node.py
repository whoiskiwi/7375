import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCTSNode:
    """
    One node in the MCTS tree.

    Each node represents ONE formulation attempt for the problem.
    The tree explores many different formulations, and MCTS decides
    which ones are worth refining vs. abandoning.

    Tree structure example:
        root (no formulation yet)
        ├── node_A  (formulation attempt A)
        │   ├── node_A1  (refinement of A after it failed)
        │   └── node_A2  (another refinement of A)
        └── node_B  (completely different formulation B)
            └── node_B1  (refinement of B)
    """

    # The original problem text (same for every node in the tree)
    problem: str

    # The mathematical formulation this node holds.
    # Empty string = root node (no formulation yet).
    # Format: "Type: LP\nSets: ...\nVariables: ...\nObjective: ...\nConstraints: ..."
    formulation: str = ""

    # The Python code generated from this formulation (filled in during simulate)
    code: str = ""

    # The result of executing the code: {"success": bool, "output": str, "error": str}
    result: dict = field(default_factory=dict)

    # Tree linkage
    parent: Optional["MCTSNode"] = None
    children: list = field(default_factory=list)

    # MCTS statistics
    visits: int = 0      # how many times this node has been visited
    value: float = 0.0   # cumulative reward across all visits

    # ---------------------------------------------------------------
    # UCB1: the formula that balances exploitation vs. exploration
    # ---------------------------------------------------------------
    def ucb1(self, exploration_constant: float = 1.4) -> float:
        """
        UCB1 score — used by SELECT to choose which child to visit next.

        UCB1 = (value / visits) + C * sqrt(ln(parent.visits) / visits)
                ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                exploitation       exploration
                (how good is       (how UNDERexplored is this node?
                 this path?)        prefer nodes visited less often)

        C controls the tradeoff:
          - High C → explore more (try less-visited nodes)
          - Low C  → exploit more (keep refining the best known path)
          - 1.4 is the standard value (= sqrt(2))

        Returns infinity for unvisited nodes → they always get explored first.
        """
        if self.visits == 0:
            return float("inf")  # unvisited nodes have highest priority

        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def is_leaf(self) -> bool:
        """A leaf node has no children yet — it's a candidate for expansion."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def error_feedback(self) -> str:
        """
        Returns the error message from this node's execution result.
        Used by EXPAND to give the LLM feedback when generating a refinement.
        """
        if not self.result:
            return ""
        err = self.result.get("error", "")
        out = self.result.get("output", "")
        if err:
            return err[:400]
        if out:
            return f"Code ran but answer was wrong. Output: {out[:200]}"
        return "No output produced."
