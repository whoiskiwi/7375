"""
Output Utilities for MDP Results

This module handles printing and displaying results.
"""

from typing import Dict
import numpy as np

from models.state import State
from models.mdp import BreastCancerScreeningMDP
from algorithms.policy_iteration import PolicyIteration
from config.constants import RISK_LEVELS, AGE_GROUPS, DECISION_STATES, DAYS_PER_YEAR


def print_optimal_policy(mdp: BreastCancerScreeningMDP, policy: Dict[State, str]):
    """Print the optimal policy in a readable format."""
    print("\n" + "=" * 70)
    print("OPTIMAL POLICY \u03c0*(s)")
    print("=" * 70)

    for health in DECISION_STATES:
        print(f"\n--- {health} State ---")
        print(f"{'Risk':<10} {'30-39':<10} {'40-49':<10} {'50-59':<10} {'60+':<10}")
        print("-" * 50)

        for risk in RISK_LEVELS:
            row = f"{risk:<10}"
            for age in AGE_GROUPS:
                state = State(risk, age, health)
                action = policy.get(state, "N/A")
                row += f"{action:<10}"
            print(row)


def print_value_function(mdp: BreastCancerScreeningMDP, V: np.ndarray):
    """Print the value function in a readable format."""
    print("\n" + "=" * 70)
    print("VALUE FUNCTION V*(s) - Expected Cumulative QALY")
    print("=" * 70)

    for health in ["Healthy", "Cured", "Early-Undetected"]:
        print(f"\n--- Health Status: {health} ---")
        print(f"{'Risk':<10} {'30-39':<12} {'40-49':<12} {'50-59':<12} {'60+':<12}")
        print("-" * 58)

        for risk in RISK_LEVELS:
            row = f"{risk:<10}"
            for age in AGE_GROUPS:
                state = State(risk, age, health)
                idx = mdp.state_to_idx[state]
                value = V[idx]
                row += f"{value:<12.2f}"
            print(row)


def print_q_value_analysis(
    mdp: BreastCancerScreeningMDP,
    solver: PolicyIteration,
    policy: Dict[State, str],
):
    """
    Print Q-value comparison for Healthy states with days conversion.

    Args:
        mdp: The MDP model
        solver: The solved PolicyIteration instance
        policy: Policy dict to display in the Policy column
    """
    print("\n" + "=" * 70)
    print("Q-VALUE ANALYSIS (Healthy State)")
    print("=" * 70)
    print(f"\n{'Risk':<8} {'Age':<8} {'Q(Screen)':<12} {'Q(Wait)':<12} "
          f"{'Diff':<10} {'Days':<8} {'Policy'}")
    print("-" * 70)

    for risk in RISK_LEVELS:
        for age in AGE_GROUPS:
            state = State(risk, age, "Healthy")
            q_values = solver.get_q_values(state)
            q_screen = q_values.get("Screen", 0)
            q_wait = q_values.get("Wait", 0)
            diff = q_screen - q_wait
            diff_days = diff * DAYS_PER_YEAR
            action = policy.get(state, "N/A")
            print(f"{risk:<8} {age:<8} {q_screen:<12.4f} {q_wait:<12.4f} "
                  f"{diff:<+10.4f} {diff_days:<+8.1f} {action}")


def print_summary(policy: Dict[State, str]):
    """Print summary of policy statistics."""
    screen_count = sum(1 for s, a in policy.items()
                       if s.health in DECISION_STATES and a == "Screen")
    wait_count = sum(1 for s, a in policy.items()
                     if s.health in DECISION_STATES and a == "Wait")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nDecision States: {screen_count + wait_count}")
    print(f"  - Screen: {screen_count}")
    print(f"  - Wait: {wait_count}")

    # Per-risk breakdown
    for risk in RISK_LEVELS:
        s = sum(1 for st, a in policy.items()
                if st.risk == risk and st.health in DECISION_STATES and a == "Screen")
        w = sum(1 for st, a in policy.items()
                if st.risk == risk and st.health in DECISION_STATES and a == "Wait")
        print(f"  {risk}: Screen={s}, Wait={w}")

    print(f"""
Policy Based On:
{chr(9473) * 71}
USPSTF 2024 (JAMA 2024;331(22):1918-1930):
  "The USPSTF recommends biennial screening mammography for women
   aged 40 to 74 years." (Grade B)

NCCN Guidelines (High Risk):
  "For BRCA1/2 carriers: Annual mammography and breast MRI starting
   at age 25-30."
{chr(9473) * 71}
""")
