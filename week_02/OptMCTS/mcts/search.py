"""
SolverLLM MCTS — full paper implementation.

Three key innovations over standard MCTS:
  1. Dynamic Expansion      — expand non-leaf nodes when trigger+uncertainty signal is active
  2. Prompt Backpropagation — accumulate layer-wise guidance in a knowledge base;
                              inject into future expansion prompts
  3. Uncertainty Backpropagation — weight Q-updates by ρ = exp(-U_global) so that
                              high-uncertainty evaluations have less influence

Tree: 6-layer hierarchy per formulation element
  root → Type → Sets → Parameters → Variables → Objective → Constraints

Code generation is separate from MCTS (max 12 retries, per paper).
"""

import math
import random
from difflib import SequenceMatcher

from core.llm import call_llm
from core.executor import execute_code, is_valid
from evaluation.evaluator import extract_answer, evaluate
from mcts.node import MCTSNode, LAYER_NAMES
from mcts.evaluator import (
    sample_objective_scores,
    compute_global_uncertainty,
    get_all_reasoning_signals,
)

# ── Prompts ───────────────────────────────────────────────────────────────────

_ELEMENT_PROMPT = """\
You are an expert in mathematical optimization.

Problem:
{problem}

Formulation so far:
{partial}
{knowledge_guidance}
Generate ONLY the "{element_name}" component of the mathematical formulation.
Be concise and precise. Do not write code."""

_CODE_PROMPT = """\
Based on the problem and mathematical formulation below, write Python code to solve it.

Original problem:
{problem}

Mathematical formulation:
{formulation}

Requirements:
- Use scipy.optimize or PuLP (for integer programs)
- Output ONLY executable Python code, no markdown, no explanation
- Print the optimal objective value as the LAST line of output
- If infeasible or unbounded, print 0"""

_FIX_CODE_PROMPT = """\
The following Python code failed. Please fix it.

Error:
{error}

Original code:
{code}

Requirements:
- Output ONLY executable Python code, no markdown, no explanation
- Print the optimal objective value as the LAST line of output
- If infeasible or unbounded, print 0"""


# ── Similarity pruning ────────────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower()[:300], b.lower()[:300]).ratio()


def _prune_similar(candidates: list, threshold: float = 0.8) -> list:
    """Remove candidates that are too similar to already-kept ones."""
    pruned = []
    for c in candidates:
        if not any(_similarity(c, p) > threshold for p in pruned):
            pruned.append(c)
    return pruned if pruned else candidates[:1]


# ── Element generation ────────────────────────────────────────────────────────

def _gen_element(problem: str, partial_text: str, element_name: str,
                 knowledge_base: dict) -> str:
    """Generate one formulation element, injecting knowledge-base guidance."""
    guidance_list = knowledge_base.get(element_name, [])
    if guidance_list:
        kb_text = (
            f"\nGuidance from previous attempts for '{element_name}':\n"
            + "\n".join(f"- {g}" for g in guidance_list[-3:])
            + "\n"
        )
    else:
        kb_text = ""

    prompt = (_ELEMENT_PROMPT
              .replace("{problem}", problem)
              .replace("{partial}", partial_text or "(none yet)")
              .replace("{knowledge_guidance}", kb_text)
              .replace("{element_name}", element_name))
    return call_llm(prompt, temperature=0.7)


def _complete_formulation(problem: str, node: MCTSNode, knowledge_base: dict) -> dict:
    """
    Starting from `node`, greedily generate all remaining layers (node.layer+1 to 6).
    Returns the full formulation dict {layer_name: content}.
    These temporary layers are NOT added to the tree — simulation use only.
    """
    formulation = node.get_formulation_path().copy()

    for layer in range(node.layer + 1, 7):
        layer_name = LAYER_NAMES[layer]
        partial_text = "\n".join(
            f"**{n.capitalize()}**: {formulation[n]}"
            for n in LAYER_NAMES[1:]
            if n in formulation
        )
        content = _gen_element(problem, partial_text, layer_name, knowledge_base)
        formulation[layer_name] = content

    return formulation


def _format_formulation(formulation_dict: dict) -> str:
    lines = []
    for name in LAYER_NAMES[1:]:
        if name in formulation_dict:
            lines.append(f"**{name.capitalize()}**: {formulation_dict[name]}")
    return "\n".join(lines)


# ── Code generation (max 12 retries, per paper) ───────────────────────────────

def _generate_and_execute(problem: str, formulation: str, max_retries: int = 12) -> dict:
    code = call_llm(_CODE_PROMPT
                    .replace("{problem}", problem)
                    .replace("{formulation}", formulation))
    result = execute_code(code)

    for _ in range(max_retries):
        if is_valid(result):
            break
        code = call_llm(_FIX_CODE_PROMPT
                        .replace("{error}", (result.get("error") or "")[:500])
                        .replace("{code}", code[:1500]))
        result = execute_code(code)

    return result


# ── MCTS ──────────────────────────────────────────────────────────────────────

class MCTS:
    """
    SolverLLM MCTS with paper's three innovations.

    Hyperparameters match Table 6 of the paper:
      n_iterations=20, C=2.0, eta=0.3, max_children=5, k_samples=3
    """

    MAX_CHILDREN = 5  # max children per node (paper: max nodes per layer = 5)

    def __init__(
        self,
        n_iterations: int = 20,
        exploration_constant: float = 2.0,
        uncertainty_threshold: float = 0.3,
        n_score_samples: int = 3,
    ):
        self.n_iterations = n_iterations
        self.C = exploration_constant
        self.eta = uncertainty_threshold   # η: trigger threshold for dynamic expansion
        self.k = n_score_samples           # K: objective score samples for uncertainty

        # Per-layer knowledge base: layer_name → [guidance strings]
        self.knowledge_base: dict = {name: [] for name in LAYER_NAMES[1:]}

    def search(self, problem: str, expected_answer: float = None) -> dict:
        # Reset knowledge base per problem (paper: knowledge base is per-problem,
        # accumulated only during the MCTS iterations for a single problem)
        self.knowledge_base = {name: [] for name in LAYER_NAMES[1:]}

        root = MCTSNode(problem=problem, layer=0)

        best_result = {"success": False, "output": "", "error": "No solution found"}
        found_correct = False

        for i in range(self.n_iterations):

            # ── 1. SELECT ───────────────────────────────────────────────────
            node = self._select(root)

            # ── 2. EXPAND ───────────────────────────────────────────────────
            child = self._expand(node)
            if child is None:
                continue

            # ── 3. SIMULATE ─────────────────────────────────────────────────
            reward, result, global_u, signals = self._simulate(child, expected_answer)
            child.result = result

            # ── 4. BACKPROPAGATE ────────────────────────────────────────────
            self._backpropagate(child, reward, global_u, signals)

            # Track best result
            if reward >= 1.0 and not found_correct:
                best_result = result
                found_correct = True
                print(f"    [iter {i+1:02d}] CORRECT — stopping early")
                break

            if result.get("success") and not found_correct:
                if not best_result.get("success"):
                    best_result = result

            print(
                f"    [iter {i+1:02d}] reward={reward:.3f}  "
                f"U_global={global_u:.3f}  nodes={self._size(root)}"
            )

        return best_result

    # ── Phase 1: SELECT ──────────────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Walk down the tree using UCB1. Stop at:
          - A leaf node (no children), OR
          - A non-leaf with trigger=True AND local_uncertainty > η
            (Dynamic Expansion: add more options at this level)
        """
        while True:
            # Dynamic Expansion: triggered non-leaf (not complete) with high uncertainty
            if not node.is_root() and not node.is_leaf() and not node.is_complete():
                if node.trigger and node.local_uncertainty > self.eta:
                    return node

            if node.is_leaf():
                # If this is a complete node (layer 6) already visited, go up
                # and let the parent pick a different child next iteration
                if node.is_complete() and node.visits > 0:
                    return node  # _expand will return None → skip iteration
                return node

            # Prefer unvisited children first
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                return random.choice(unvisited)

            node = max(node.children, key=lambda c: c.ucb1(self.C))

    # ── Phase 2: EXPAND ──────────────────────────────────────────────────────

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Generate up to 3 candidate elements for layer (node.layer + 1).
        Prune semantically similar ones (threshold=0.8).
        Add non-duplicate candidates as new children (capped at MAX_CHILDREN).
        Return one newly added child.
        """
        next_layer = node.layer + 1
        if next_layer > 6:
            return None

        if len(node.children) >= self.MAX_CHILDREN:
            unvisited = [c for c in node.children if c.visits == 0]
            return random.choice(unvisited) if unvisited else random.choice(node.children)

        partial_text = node.format_partial_formulation()
        element_name = LAYER_NAMES[next_layer]

        candidates = [
            _gen_element(node.problem, partial_text, element_name, self.knowledge_base)
            for _ in range(3)
        ]
        candidates = _prune_similar(candidates, threshold=0.8)

        existing = {c.content for c in node.children}
        new_children = []
        for content in candidates:
            if len(node.children) >= self.MAX_CHILDREN:
                break
            if not any(_similarity(content, e) > 0.8 for e in existing):
                child = MCTSNode(
                    problem=node.problem,
                    layer=next_layer,
                    content=content,
                    parent=node,
                )
                node.children.append(child)
                existing.add(content)
                new_children.append(child)

        if not new_children:
            unvisited = [c for c in node.children if c.visits == 0]
            return random.choice(unvisited) if unvisited else random.choice(node.children)

        return random.choice(new_children)

    # ── Phase 3: SIMULATE ────────────────────────────────────────────────────

    def _simulate(self, node: MCTSNode, expected_answer: float = None) -> tuple:
        """
        Complete remaining layers greedily (not added to tree), generate and
        execute code, evaluate with LLM, collect reasoning signals.

        Reward (paper Table 5):
            R = 0.1·feasible + 0.8·mean_score - 0.1·error

        Overridden to 1.0 if ground-truth answer is known and matches.

        Returns: (reward, code_result, global_uncertainty, signals_dict)
        """
        formulation_dict = _complete_formulation(node.problem, node, self.knowledge_base)
        formulation_text = _format_formulation(formulation_dict)

        result = _generate_and_execute(node.problem, formulation_text)

        scores = sample_objective_scores(node.problem, formulation_text, result, k=self.k)
        global_u = compute_global_uncertainty(scores)
        mean_score = sum(scores) / len(scores) / 100.0  # normalize to [0,1]

        feasible = 1.0 if result.get("success") else 0.0
        error_flag = 0.0 if result.get("success") else 1.0
        reward = max(0.0, min(1.0, 0.1 * feasible + 0.8 * mean_score - 0.1 * error_flag))

        # Ground-truth override
        if expected_answer is not None and is_valid(result):
            predicted = extract_answer(result["output"])
            if evaluate(predicted, expected_answer):
                reward = 1.0

        signals = get_all_reasoning_signals(node.problem, formulation_dict, result)

        return reward, result, global_u, signals

    # ── Phase 4: BACKPROPAGATE ───────────────────────────────────────────────

    def _backpropagate(self, node: MCTSNode, reward: float,
                       global_u: float, signals: dict):
        """
        Walk node → root:

        1. Prompt Backpropagation: add each layer's guidance to knowledge_base
           (controls future expansion prompts).

        2. Uncertainty Backpropagation: update Q-values with confidence weight
           ρ = exp(-U_global):
               Q ← Q + ρ·(R − Q) / N   (incremental weighted mean)

        3. Update trigger and local_uncertainty on tree nodes.
        """
        # 1. Knowledge base update
        for layer_name, sig in signals.items():
            if sig.get("guidance"):
                kb = self.knowledge_base[layer_name]
                kb.append(sig["guidance"])
                if len(kb) > 20:
                    self.knowledge_base[layer_name] = kb[-10:]

        # 2 & 3. Tree update
        rho = math.exp(-global_u)
        current = node
        while current is not None:
            current.visits += 1
            current.value += rho * (reward - current.value) / current.visits

            layer_name = current.element_name()
            if layer_name in signals:
                sig = signals[layer_name]
                current.trigger = sig["trigger"]
                current.local_uncertainty = sig["local_uncertainty"]

            current = current.parent

    # ── Utility ──────────────────────────────────────────────────────────────

    def _size(self, node: MCTSNode) -> int:
        return 1 + sum(self._size(c) for c in node.children)
