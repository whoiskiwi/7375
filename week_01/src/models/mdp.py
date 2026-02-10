"""
Breast Cancer Screening MDP Model - Biennial Screening

USPSTF 2024 aligned model with biennial (every 2 years) screening.

This module defines the main MDP class that combines:
- State space
- Action space
- Transition probabilities (2-year epochs)
- Reward function

References:
- USPSTF 2024: "Biennial screening mammography for women aged 40 to 74 years"
- DOI: 10.1001/jama.2024.5534
"""

from typing import Dict, List
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.constants import (
    RISK_LEVELS, AGE_GROUPS, HEALTH_STATES,
    ACTIONS, DECISION_STATES, TERMINAL_STATES, GAMMA,
    NO_DECISION_ACTION, REFERENCE_AGE_GROUP
)
from config.parameters_biennial import (
    INCIDENCE_RATE_BIENNIAL,
    NATURAL_MORTALITY_BIENNIAL,
    SCREENING_BY_AGE,
    PROGRESSION_BIENNIAL,
    REWARDS_BIENNIAL,
    TIME_STEP_YEARS
)
from .state import State
from .transitions_v2 import TransitionCalculatorV2
from .rewards import RewardCalculator


class BreastCancerScreeningMDP:
    """
    Markov Decision Process for breast cancer screening decisions.

    This MDP models biennial screening as recommended by USPSTF 2024.
    Decision epochs are 2 years, matching the recommended screening interval.

    Attributes:
        states: List of all possible states
        n_states: Number of states
        state_to_idx: Mapping from state to index
        idx_to_state: Mapping from index to state
        gamma: Discount factor (adjusted for 2-year epochs)
        time_step: Decision epoch length in years (2 for biennial)
    """

    def __init__(self):
        """Initialize the MDP with biennial screening parameters."""
        # Build state space
        self.states = self._build_state_space()
        self.n_states = len(self.states)

        # Create state indexing
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}

        # Time step and discount factor for biennial model
        self.time_step = TIME_STEP_YEARS  # 2 years
        self.gamma = GAMMA ** TIME_STEP_YEARS  # γ² for 2-year epochs

        # Initialize calculators with biennial parameters
        self._transition_calc = TransitionCalculatorV2(
            incidence_rates=INCIDENCE_RATE_BIENNIAL,
            mortality_rates=NATURAL_MORTALITY_BIENNIAL,
            screening_by_age=SCREENING_BY_AGE,
            progression=PROGRESSION_BIENNIAL
        )

        self._reward_calc = RewardCalculator(
            rewards=REWARDS_BIENNIAL,
            incidence_rates=INCIDENCE_RATE_BIENNIAL,
            screening_by_age=SCREENING_BY_AGE,
            progression=PROGRESSION_BIENNIAL
        )

    def _build_state_space(self) -> List[State]:
        """
        Build the complete state space.

        Returns:
            List of all possible states
        """
        states = []
        for risk in RISK_LEVELS:
            for age in AGE_GROUPS:
                for health in HEALTH_STATES:
                    states.append(State(risk, age, health))
        return states

    def get_actions(self, state: State) -> List[str]:
        """
        Get available actions for a given state.

        Args:
            state: Current state

        Returns:
            List of available actions
        """
        if state.health in DECISION_STATES:
            return ACTIONS  # ["Screen", "Wait"]
        else:
            return [NO_DECISION_ACTION]  # No decision required

    def get_transition_prob(self, state: State, action: str) -> Dict[State, float]:
        """
        Get transition probabilities P(s'|s,a) for 2-year epoch.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Dictionary mapping next states to probabilities
        """
        return self._transition_calc.get_transitions(state, action)

    def get_reward(self, state: State, action: str) -> float:
        """
        Get immediate reward R(s,a) for 2-year epoch.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Reward in QALY (accumulated over 2 years)
        """
        return self._reward_calc.get_reward(state, action)

    def is_terminal(self, state: State) -> bool:
        """
        Check if state is terminal.

        Args:
            state: State to check

        Returns:
            True if terminal (Dead), False otherwise
        """
        return state.health in TERMINAL_STATES

