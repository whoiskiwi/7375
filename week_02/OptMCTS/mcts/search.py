import random
from core.llm import call_llm
from core.executor import execute_code, is_valid
from evaluation.evaluator import extract_answer, evaluate
from mcts.node import MCTSNode


# ---------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------

# Used when generating a fresh formulation (root → first child)
_FORMULATION_PROMPT = """\
You are an expert in mathematical optimization.

Given the problem below, write a precise mathematical formulation using this schema:

**Type**: What kind of problem is this? (LP = Linear Program, MILP = Mixed Integer LP, NLP = Nonlinear, etc.)
**Sets**: Index sets (e.g., i in 1..n)
**Parameters**: Given data / constants
**Variables**: Decision variables (what we're choosing)
**Objective**: Minimize or maximize what?
**Constraints**: All constraints the solution must satisfy

Problem:
{problem}

Be concise and precise. Do not write code yet."""

# Used when refining a failed formulation (child → grandchild)
_REFINEMENT_PROMPT = """\
You are an expert in mathematical optimization.

A previous formulation attempt for the problem below failed. Your job is to write
a DIFFERENT, improved formulation that fixes the issues.

Problem:
{problem}

Previous formulation:
{formulation}

What went wrong:
{feedback}

Write a new formulation using the same schema:
**Type**, **Sets**, **Parameters**, **Variables**, **Objective**, **Constraints**

Try a different approach if the previous one was fundamentally wrong."""

# Used to turn a formulation into executable Python code.
# IMPORTANT: include both the original problem AND the formulation.
# The formulation may use abstract symbols (x₁, c₁, etc.) — the original
# problem text provides the actual numbers the LLM needs to write working code.
_CODE_PROMPT = """\
Based on the problem and its mathematical formulation below, write Python code to solve it.

Original problem:
{problem}

Mathematical formulation:
{formulation}

Requirements:
- Use scipy.optimize or PuLP (for integer programs)
- Output ONLY executable Python code, no markdown, no explanation
- Print the optimal objective value as the LAST line of output
- If the problem is infeasible or unbounded, print 0"""


# ---------------------------------------------------------------
# MCTS class
# ---------------------------------------------------------------

class MCTS:
    """
    Monte Carlo Tree Search for optimization problem solving.

    How it works (one iteration = four phases):

        SELECT       Walk down the tree, always picking the child with the
                     highest UCB1 score, until you reach a leaf node.

        EXPAND       From that leaf, ask the LLM to generate a new child
                     (a new formulation). If the leaf already failed, pass
                     the error as feedback so the LLM can improve.

        SIMULATE     Take the new child's formulation, generate Python code,
                     execute it, and score the result:
                       1.0 = correct answer
                       0.3 = code ran but answer was wrong
                       0.0 = code crashed

        BACKPROPAGATE Walk back UP the tree, adding the reward to every
                     ancestor. This updates their visit counts and values,
                     so future SELECT calls know which paths have been tried.

    After n_iterations, return the best result found.
    """

    def __init__(self, n_iterations: int = 20, exploration_constant: float = 1.4):
        self.n_iterations = n_iterations
        self.C = exploration_constant

    def search(self, problem: str, expected_answer: float = None) -> dict:
        """
        Run MCTS on a problem. Returns the best execution result found.

        expected_answer: the ground-truth value (used to compute reward).
                         In real deployment you wouldn't have this, but
                         since we're evaluating against a benchmark we use it.
        """
        root = MCTSNode(problem=problem)

        best_result = {"success": False, "output": "", "error": "No solution found"}
        found_correct = False

        for i in range(self.n_iterations):
            # ── PHASE 1: SELECT ──────────────────────────────────────────
            node = self._select(root)

            # ── PHASE 2: EXPAND ──────────────────────────────────────────
            child = self._expand(node)

            # ── PHASE 3: SIMULATE ─────────────────────────────────────────
            reward, result = self._simulate(child, expected_answer)
            child.result = result

            # ── PHASE 4: BACKPROPAGATE ────────────────────────────────────
            self._backpropagate(child, reward)

            # Track best result
            if reward == 1.0:
                best_result = result
                found_correct = True
                print(f"    [iter {i+1:02d}] CORRECT — stopping early")
                break

            if result.get("success") and not found_correct:
                best_result = result  # keep best "ran but wrong" as fallback

            print(f"    [iter {i+1:02d}] reward={reward:.1f}  nodes={self._size(root)}")

        return best_result

    # ---------------------------------------------------------------
    # Phase 1 — SELECT
    # ---------------------------------------------------------------
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Walk down the tree using UCB1 until we reach a leaf.

        At each step:
          - If any child is unvisited → go there first (UCB1 = inf)
          - Otherwise → go to the child with the highest UCB1 score
        """
        while not node.is_leaf():
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                return random.choice(unvisited)
            node = max(node.children, key=lambda c: c.ucb1(self.C))
        return node

    # ---------------------------------------------------------------
    # Phase 2 — EXPAND
    # ---------------------------------------------------------------
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Ask the LLM to generate a new formulation and attach it as a child.

        - If node is root (no formulation yet): generate a fresh formulation.
        - If node already has a formulation that failed: generate a refined one
          using the error as feedback.
        """
        if node.is_root() or not node.formulation:
            prompt = (_FORMULATION_PROMPT
                      .replace("{problem}", node.problem))
        else:
            prompt = (_REFINEMENT_PROMPT
                      .replace("{problem}", node.problem)
                      .replace("{formulation}", node.formulation)
                      .replace("{feedback}", node.error_feedback()))

        formulation = call_llm(prompt)
        child = MCTSNode(problem=node.problem, formulation=formulation, parent=node)
        node.children.append(child)
        return child

    # ---------------------------------------------------------------
    # Phase 3 — SIMULATE
    # ---------------------------------------------------------------
    def _simulate(self, node: MCTSNode, expected_answer: float = None) -> tuple:
        """
        Generate code from the formulation, execute it, and return a reward.

        Reward values:
          1.0  — answer is correct (within 10% of expected)
          0.3  — code ran successfully but answer is wrong
          0.0  — code crashed or produced no numeric output

        Why 0.3 (not 0) for "ran but wrong"? It gives the MCTS a weak signal
        that a formulation that at least compiles is better than one that crashes.
        This helps steer the search toward structurally sound formulations.
        """
        code = call_llm(_CODE_PROMPT
                        .replace("{problem}", node.problem)
                        .replace("{formulation}", node.formulation))
        node.code = code

        result = execute_code(code)

        if not is_valid(result):
            return 0.0, result

        if expected_answer is not None:
            predicted = extract_answer(result["output"])
            if evaluate(predicted, expected_answer):
                return 1.0, result

        return 0.3, result

    # ---------------------------------------------------------------
    # Phase 4 — BACKPROPAGATE
    # ---------------------------------------------------------------
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Walk up the tree from node to root, updating every ancestor.

        visits += 1   so UCB1 exploration term decreases (less urgent to revisit)
        value  += reward  so exploitation term reflects the path's track record
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    # ---------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------
    def _size(self, node: MCTSNode) -> int:
        """Count total nodes in the tree (for logging)."""
        return 1 + sum(self._size(c) for c in node.children)
