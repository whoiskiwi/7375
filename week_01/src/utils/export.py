"""
Export Utilities for MDP Results

This module handles saving results to files.
"""

import csv
from typing import Dict
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.state import State
from models.mdp import BreastCancerScreeningMDP
from config.constants import RESULTS_DIR, NO_DECISION_ACTION


def save_results(
    mdp: BreastCancerScreeningMDP,
    policy: Dict[State, str],
    V: np.ndarray,
    output_dir: Path = None
):
    """
    Save policy and value function to CSV files.

    Args:
        mdp: The MDP model
        policy: Optimal policy dictionary
        V: Value function array
        output_dir: Output directory path
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save optimal policy
    policy_file = output_dir / "optimal_policy.csv"
    with open(policy_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Risk", "Age", "Health", "Optimal_Action"])
        for state, action in policy.items():
            if action != NO_DECISION_ACTION:
                writer.writerow([state.risk, state.age, state.health, action])

    # Save value function
    value_file = output_dir / "value_function.csv"
    with open(value_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Risk", "Age", "Health", "Value_QALY"])
        for i, state in enumerate(mdp.states):
            writer.writerow([state.risk, state.age, state.health, f"{V[i]:.4f}"])

    print(f"\nResults saved to:")
    print(f"  - {policy_file}")
    print(f"  - {value_file}")
