"""
LLM-based evaluation utilities for SolverLLM's MCTS.

Three responsibilities:
  1. objective_score   — LLM scores solution quality 0-100 (K samples for uncertainty)
  2. reasoning_signals — LLM evaluates all 6 formulation layers in one call
                         → (trigger, explanation, guidance) per layer
  3. uncertainty       — local (predictive entropy from logprobs)
                         global (std of K objective_score samples)
"""

import math
from core.llm import call_llm_json, call_llm_json_with_logprobs

LAYER_NAMES = ["type", "sets", "parameters", "variables", "objective", "constraints"]

# ── Prompts ───────────────────────────────────────────────────────────────────

_SCORE_PROMPT = """\
You are evaluating an optimization problem solution.

Problem:
{problem}

Mathematical formulation:
{formulation}

Execution output:
{output}

Score the overall solution quality from 0 to 100:
- 0-25:   Poor  (crashes, infeasible, or completely wrong answer)
- 26-50:  Fair  (runs but answer is significantly wrong)
- 51-75:  Good  (reasonable formulation, answer is close)
- 76-100: Excellent (correct formulation, correct answer)

Return JSON: {{"score": <integer 0-100>}}"""

_SIGNALS_PROMPT = """\
You are reviewing a mathematical optimization formulation.

Problem:
{problem}

Complete formulation:
{formulation}

Execution result — success={success}, output={output}, error={error}

For EACH of the 6 formulation elements, evaluate its quality.
Return a JSON object with keys: "type", "sets", "parameters", "variables", "objective", "constraints".
Each value must have:
  "trigger":     true if this element has issues needing revision, else false
  "explanation": one sentence on quality
  "guidance":    specific improvement advice if trigger=true, else ""

JSON:"""


# ── Objective score ───────────────────────────────────────────────────────────

def sample_objective_scores(problem: str, formulation: str, code_result: dict, k: int = 3) -> list:
    """
    Sample K objective scores (0-100) from the LLM evaluator.
    Uses temperature=0.5 for diversity, enabling uncertainty estimation.
    """
    output = (code_result.get("output") or "")[:300]
    prompt = (_SCORE_PROMPT
              .replace("{problem}", problem)
              .replace("{formulation}", formulation)
              .replace("{output}", output))
    scores = []
    for _ in range(k):
        data = call_llm_json(prompt, temperature=0.5)
        try:
            scores.append(max(0, min(100, int(data.get("score", 0)))))
        except (TypeError, ValueError):
            scores.append(0)
    return scores


# ── Uncertainty ───────────────────────────────────────────────────────────────

def compute_local_uncertainty(token_logprobs: list) -> float:
    """
    Predictive entropy from token log-probabilities (per paper §3.3):
        U_local = -E[log P(a_i | context)]
                = average of -logprob across tokens

    Normalized to [0, 1] by dividing by 5.0 nats (typical range).
    """
    if not token_logprobs:
        return 0.0
    avg_neg_logprob = -sum(token_logprobs) / len(token_logprobs)
    return min(avg_neg_logprob / 5.0, 1.0)


def compute_global_uncertainty(scores: list) -> float:
    """
    Semantic uncertainty from K objective_score samples (per paper §3.3).

    Approximated as normalized standard deviation of scores.
    Max std for 0-100 range ≈ 50, so we normalize by 50.
    """
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return min(math.sqrt(variance) / 50.0, 1.0)


# ── Reasoning signals ─────────────────────────────────────────────────────────

def get_all_reasoning_signals(
    problem: str,
    formulation_dict: dict,
    code_result: dict,
) -> dict:
    """
    Ask the LLM to evaluate all 6 formulation elements in ONE call.
    Returns logprobs for local uncertainty estimation.

    Returns dict: layer_name → {trigger, explanation, guidance, local_uncertainty}
    """
    formulation_text = "\n".join(
        f"**{k.capitalize()}**: {v}"
        for k, v in formulation_dict.items()
        if k in LAYER_NAMES
    )
    output = (code_result.get("output") or "")[:200]
    error = (code_result.get("error") or "")[:200]
    success = str(code_result.get("success", False))

    prompt = (_SIGNALS_PROMPT
              .replace("{problem}", problem)
              .replace("{formulation}", formulation_text)
              .replace("{success}", success)
              .replace("{output}", output)
              .replace("{error}", error))

    data, token_logprobs = call_llm_json_with_logprobs(prompt, temperature=0.2)

    # One local uncertainty value shared across all layers (from the single call)
    local_u = compute_local_uncertainty(token_logprobs)

    signals = {}
    for layer_name in LAYER_NAMES:
        layer_data = data.get(layer_name, {})
        if not isinstance(layer_data, dict):
            layer_data = {}
        signals[layer_name] = {
            "trigger": bool(layer_data.get("trigger", False)),
            "explanation": str(layer_data.get("explanation", "")),
            "guidance": str(layer_data.get("guidance", "")),
            "local_uncertainty": local_u,
        }
    return signals
