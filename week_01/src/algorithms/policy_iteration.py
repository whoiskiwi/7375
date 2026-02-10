"""
Policy Iteration Algorithm for MDP

This module implements the Policy Iteration algorithm to find
the optimal policy that maximizes expected QALY.

Algorithm:
1. Initialize with a random policy
2. Policy Evaluation: Compute V(s) for current policy
3. Policy Improvement: Update policy using V(s)
4. Repeat until policy converges

Reference:
- Sutton & Barto, "Reinforcement Learning: An Introduction"
"""

import numpy as np
from typing import Dict, Tuple
import time
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.state import State
from models.mdp import BreastCancerScreeningMDP
from config.constants import GAMMA, CONVERGENCE_THRESHOLD, MAX_EVAL_ITERATIONS, NO_DECISION_ACTION


class PolicyIteration:
    """
    Policy Iteration solver for the MDP.

    Attributes:
        mdp: The MDP model instance
        gamma: Discount factor
        theta: Convergence threshold
        V: Value function array
        policy: Current policy dictionary
    """

    def __init__(
        self,
        mdp: BreastCancerScreeningMDP,
        gamma: float = GAMMA,
        theta: float = CONVERGENCE_THRESHOLD,
        max_eval_iterations: int = MAX_EVAL_ITERATIONS
    ):
        """
        Initialize the solver.

        Args:
            mdp: The MDP model
            gamma: Discount factor
            theta: Convergence threshold
            max_eval_iterations: Max iterations for policy evaluation
        """
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        self.max_eval_iterations = max_eval_iterations

        # Initialize value function and policy
        self.V = np.zeros(mdp.n_states)
        self.policy = self._initialize_policy()

        # Tracking
        self.iteration_count = 0
        self.policy_stable = False

    def _initialize_policy(self) -> Dict[State, str]:
        """Initialize policy with 'Wait' for all decision states."""
        policy = {}
        for state in self.mdp.states:
            actions = self.mdp.get_actions(state)
            if "Screen" in actions:
                policy[state] = "Wait"  # Conservative initialization
            else:
                policy[state] = NO_DECISION_ACTION
        return policy

    def policy_evaluation(self) -> int:
        """
        Evaluate current policy by computing V(s).

        V(s) = R(s, π(s)) + γ × Σ P(s'|s, π(s)) × V(s')

        Returns:
            Number of iterations until convergence
        """
        iterations = 0

        while iterations < self.max_eval_iterations:
            delta = 0

            for i, state in enumerate(self.mdp.states):
                if self.mdp.is_terminal(state):
                    continue

                v_old = self.V[i]
                action = self.policy[state]

                # Compute new value
                reward = self.mdp.get_reward(state, action)
                transitions = self.mdp.get_transition_prob(state, action)

                v_new = reward
                for next_state, prob in transitions.items():
                    j = self.mdp.state_to_idx[next_state]
                    v_new += self.gamma * prob * self.V[j]

                self.V[i] = v_new
                delta = max(delta, abs(v_old - v_new))

            iterations += 1

            if delta < self.theta:
                break

        return iterations

    def policy_improvement(self) -> bool:
        """
        Improve policy based on current value function.

        π(s) = argmax_a [R(s,a) + γ × Σ P(s'|s,a) × V(s')]

        Returns:
            True if policy is stable (no changes)
        """
        policy_stable = True

        for state in self.mdp.states:
            actions = self.mdp.get_actions(state)

            if len(actions) <= 1 or NO_DECISION_ACTION in actions:
                continue

            old_action = self.policy[state]
            best_action = old_action
            best_value = float('-inf')

            for action in actions:
                q_value = self._compute_q_value(state, action)

                if q_value > best_value:
                    best_value = q_value
                    best_action = action

            self.policy[state] = best_action

            if old_action != best_action:
                policy_stable = False

        return policy_stable

    def _compute_q_value(self, state: State, action: str) -> float:
        """
        Compute Q-value for a state-action pair.

        Q(s,a) = R(s,a) + γ × Σ P(s'|s,a) × V(s')
        """
        reward = self.mdp.get_reward(state, action)
        transitions = self.mdp.get_transition_prob(state, action)

        q_value = reward
        for next_state, prob in transitions.items():
            j = self.mdp.state_to_idx[next_state]
            q_value += self.gamma * prob * self.V[j]

        return q_value

    def solve(self, verbose: bool = True) -> Tuple[Dict[State, str], np.ndarray]:
        """
        Run the full Policy Iteration algorithm.

        Args:
            verbose: Whether to print progress

        Returns:
            Tuple of (optimal_policy, value_function)
        """
        if verbose:
            print("=" * 60)
            print("POLICY ITERATION - SOLVING MDP")
            print("=" * 60)
            print(f"States: {self.mdp.n_states}")
            print(f"Discount factor (γ): {self.gamma}")
            print(f"Convergence threshold: {self.theta}")
            print("-" * 60)

        start_time = time.time()

        while not self.policy_stable:
            self.iteration_count += 1

            # Policy Evaluation
            eval_iterations = self.policy_evaluation()

            # Policy Improvement
            self.policy_stable = self.policy_improvement()

            if verbose:
                screen_count = sum(1 for s, a in self.policy.items() if a == "Screen")
                print(f"Iteration {self.iteration_count}: "
                      f"Eval iterations={eval_iterations}, "
                      f"Screen actions={screen_count}, "
                      f"Stable={self.policy_stable}")

        elapsed_time = time.time() - start_time

        if verbose:
            print("-" * 60)
            print(f"Converged in {self.iteration_count} policy iterations")
            print(f"Time elapsed: {elapsed_time:.4f} seconds")
            print("=" * 60)

        return self.policy, self.V

    def get_q_values(self, state: State) -> Dict[str, float]:
        """
        Get Q-values for all actions in a given state.

        Args:
            state: The state to evaluate

        Returns:
            Dictionary mapping actions to Q-values
        """
        q_values = {}
        actions = self.mdp.get_actions(state)

        for action in actions:
            q_values[action] = self._compute_q_value(state, action)

        return q_values
