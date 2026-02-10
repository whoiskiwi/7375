#!/usr/bin/env python3
"""
Breast Cancer Screening MDP - Biennial Model (USPSTF 2024 Aligned)

This script runs the MDP model for breast cancer screening decisions,
using policy iteration to find the optimal screening strategy.

References:
- USPSTF 2024: JAMA 2024;331(22):1918-1930, DOI: 10.1001/jama.2024.5534
- NCCN Guidelines for High-Risk individuals
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.mdp import BreastCancerScreeningMDP
from algorithms.policy_iteration import PolicyIteration
from utils.output import (
    print_optimal_policy,
    print_value_function,
    print_q_value_analysis,
    print_summary,
)
from utils.export import save_results


def main():
    print("=" * 70)
    print("BREAST CANCER SCREENING MDP - BIENNIAL MODEL")
    print("Policy Iteration Optimal Solution")
    print("=" * 70)

    # Create MDP
    print("\nCreating MDP model (2-year decision epochs)...")
    mdp = BreastCancerScreeningMDP()
    print(f"  Time step: {mdp.time_step} years")
    print(f"  Discount factor (\u03b3): {mdp.gamma:.4f}")

    # Run Policy Iteration
    solver = PolicyIteration(mdp)
    policy, V = solver.solve(verbose=True)

    # Print results
    print_optimal_policy(mdp, policy)
    print_q_value_analysis(mdp, solver, policy)
    print_value_function(mdp, V)
    print_summary(policy)

    # Save results
    save_results(mdp, policy, V)


if __name__ == "__main__":
    main()
