"""
Reward Calculator for Breast Cancer Screening MDP

This module handles all reward calculations.
Rewards are measured in QALY (Quality-Adjusted Life Years).
"""

from typing import Dict
from .state import State


class RewardCalculator:
    """
    Calculates rewards R(s,a) for the MDP.

    Attributes:
        rewards: Dictionary of health utility values and costs
        incidence_rates: Cancer incidence rates (for false positive calculation)
        screening_by_age: Age-specific screening parameters
    """

    def __init__(
        self,
        rewards: Dict[str, float],
        incidence_rates: Dict,
        screening_by_age: Dict[str, Dict],
        progression: Dict = None
    ):
        self.rewards = rewards
        self.incidence = incidence_rates
        self.screening_by_age = screening_by_age
        self.progression = progression or {}

    def get_reward(self, state: State, action: str) -> float:
        """
        Calculate immediate reward for a state-action pair.

        R(s, a) = State_Utility + Action_Cost + False_Positive_Cost

        Args:
            state: Current state
            action: Action taken

        Returns:
            float: Reward in QALY
        """
        health = state.health

        # Base state utility
        base_reward = self.rewards.get(health, 0.0)

        # Dead state gets 0 reward
        if health == "Dead":
            return 0.0

        # Add screening cost if screening
        if action == "Screen":
            base_reward += self.rewards["screening_cost"]

            # Add expected false positive cost for Healthy/Cured states
            if health in ["Healthy", "Cured"]:
                base_reward += self._false_positive_cost(state)

        return base_reward

    def _false_positive_cost(self, state: State) -> float:
        """
        Calculate expected false positive cost.

        E[FP_cost] = P(no cancer) × P(false positive | no cancer) × FP_cost
                   = (1 - incidence) × (1 - specificity) × cost

        Args:
            state: Current state (Healthy or Cured)

        Returns:
            float: Expected false positive cost (negative QALY)
        """
        risk, age = state.risk, state.age

        # Get incidence rate (0 for Cured, use recurrence rate instead)
        if state.health == "Cured":
            p_no_cancer = 1 - self.progression["cured_to_recurrence"]
        else:
            incidence = self.incidence[risk][age]
            p_no_cancer = 1 - incidence

        # False positive probability (age-specific)
        false_positive_rate = self.screening_by_age[age]["false_positive"]

        # Expected cost
        fp_cost = self.rewards["false_positive_cost"]
        expected_cost = p_no_cancer * false_positive_rate * fp_cost

        return expected_cost
