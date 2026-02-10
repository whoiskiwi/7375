"""
Transition Probability Calculator - Version 2 (Age-Stratified)

Key Update: Uses age-specific screening parameters
"""

from typing import Dict, Callable
import logging

from .state import State
from config.constants import REFERENCE_AGE_GROUP

logger = logging.getLogger(__name__)


class TransitionCalculatorV2:
    """
    Calculates transition probabilities with age-specific screening performance.
    """

    def __init__(
        self,
        incidence_rates: Dict,
        mortality_rates: Dict,
        screening_by_age: Dict[str, Dict],  # Age-specific screening params
        progression: Dict
    ):
        self.incidence = incidence_rates
        self.mortality = mortality_rates
        self.screening_by_age = screening_by_age
        self.progression = progression

    def _get_screening_params(self, age: str) -> Dict[str, float]:
        """Get screening parameters for specific age group."""
        if age not in self.screening_by_age:
            logger.warning("Age group '%s' not found, falling back to '%s'", age, REFERENCE_AGE_GROUP)
        return self.screening_by_age.get(age, self.screening_by_age[REFERENCE_AGE_GROUP])

    def get_transitions(self, state: State, action: str) -> Dict[State, float]:
        """Get transition probabilities for a state-action pair."""
        health = state.health

        if health == "Dead":
            return {state: 1.0}
        elif health == "Healthy":
            return self._healthy_transitions(state, action)
        elif health == "Early-Undetected":
            return self._early_undetected_transitions(state, action)
        elif health == "Early-Detected":
            return self._early_detected_transitions(state)
        elif health == "Cured":
            return self._cured_transitions(state, action)
        elif health == "Advanced":
            return self._advanced_transitions(state)
        else:
            raise ValueError(f"Unknown health state: {health}")

    def _healthy_transitions(self, state: State, action: str) -> Dict[State, float]:
        """Transitions from Healthy state with age-specific screening."""
        risk, age = state.risk, state.age
        incidence = self.incidence[risk][age]
        mortality = self.mortality[age]

        # Get age-specific screening parameters
        screening = self._get_screening_params(age)
        sensitivity = screening["sensitivity"]

        transitions = {}

        if action == "Screen":
            # Cancer + detected
            p_detected = incidence * sensitivity
            if p_detected > 0:
                transitions[State(risk, age, "Early-Detected")] = p_detected

            # Cancer + not detected
            p_undetected = incidence * (1 - sensitivity)
            if p_undetected > 0:
                transitions[State(risk, age, "Early-Undetected")] = p_undetected

            # No cancer, survived
            p_healthy = (1 - incidence) * (1 - mortality)
            if p_healthy > 0:
                transitions[State(risk, age, "Healthy")] = p_healthy

            # Natural death
            if mortality > 0:
                transitions[State(risk, age, "Dead")] = mortality

        else:  # Wait
            # Cancer (undetected)
            p_cancer = incidence * (1 - mortality)
            if p_cancer > 0:
                transitions[State(risk, age, "Early-Undetected")] = p_cancer

            # No cancer, survived
            p_healthy = (1 - incidence) * (1 - mortality)
            if p_healthy > 0:
                transitions[State(risk, age, "Healthy")] = p_healthy

            # Natural death
            if mortality > 0:
                transitions[State(risk, age, "Dead")] = mortality

        return transitions

    def _early_undetected_transitions(self, state: State, action: str) -> Dict[State, float]:
        """Transitions from Early-Undetected with age-specific screening."""
        risk, age = state.risk, state.age
        screening = self._get_screening_params(age)
        sensitivity = screening["sensitivity"]
        progression = self.progression["early_to_advanced"]
        mortality = self.mortality[age]

        transitions = {}

        if action == "Screen":
            # Detected
            p_detected = sensitivity * (1 - mortality)
            if p_detected > 0:
                transitions[State(risk, age, "Early-Detected")] = p_detected

            # Not detected, progressed
            p_advanced = (1 - sensitivity) * progression * (1 - mortality)
            if p_advanced > 0:
                transitions[State(risk, age, "Advanced")] = p_advanced

            # Not detected, not progressed
            p_stay = (1 - sensitivity) * (1 - progression) * (1 - mortality)
            if p_stay > 0:
                transitions[State(risk, age, "Early-Undetected")] = p_stay

            if mortality > 0:
                transitions[State(risk, age, "Dead")] = mortality

        else:  # Wait
            p_advanced = progression * (1 - mortality)
            if p_advanced > 0:
                transitions[State(risk, age, "Advanced")] = p_advanced

            p_stay = (1 - progression) * (1 - mortality)
            if p_stay > 0:
                transitions[State(risk, age, "Early-Undetected")] = p_stay

            if mortality > 0:
                transitions[State(risk, age, "Dead")] = mortality

        return transitions

    def _early_detected_transitions(self, state: State) -> Dict[State, float]:
        """Transitions from Early-Detected (treatment outcomes)."""
        risk, age = state.risk, state.age
        cure_rate = self.progression["early_detected_to_cured"]
        failure_rate = self.progression["early_detected_to_advanced"]
        mortality = self.mortality[age]

        transitions = {}

        p_cured = cure_rate * (1 - mortality)
        if p_cured > 0:
            transitions[State(risk, age, "Cured")] = p_cured

        p_advanced = failure_rate * (1 - mortality)
        if p_advanced > 0:
            transitions[State(risk, age, "Advanced")] = p_advanced

        if mortality > 0:
            transitions[State(risk, age, "Dead")] = mortality

        return transitions

    def _cured_transitions(self, state: State, action: str) -> Dict[State, float]:
        """Transitions from Cured with age-specific screening."""
        risk, age = state.risk, state.age
        recurrence = self.progression["cured_to_recurrence"]
        mortality = self.mortality[age]
        screening = self._get_screening_params(age)
        sensitivity = screening["sensitivity"]

        transitions = {}

        if action == "Screen":
            # Recurrence + detected
            p_detected = recurrence * sensitivity * (1 - mortality)
            if p_detected > 0:
                transitions[State(risk, age, "Early-Detected")] = p_detected

            # Recurrence + not detected
            p_undetected = recurrence * (1 - sensitivity) * (1 - mortality)
            if p_undetected > 0:
                transitions[State(risk, age, "Early-Undetected")] = p_undetected

            # No recurrence
            p_cured = (1 - recurrence) * (1 - mortality)
            if p_cured > 0:
                transitions[State(risk, age, "Cured")] = p_cured

            if mortality > 0:
                transitions[State(risk, age, "Dead")] = mortality

        else:  # Wait
            p_recur = recurrence * (1 - mortality)
            if p_recur > 0:
                transitions[State(risk, age, "Early-Undetected")] = p_recur

            p_cured = (1 - recurrence) * (1 - mortality)
            if p_cured > 0:
                transitions[State(risk, age, "Cured")] = p_cured

            if mortality > 0:
                transitions[State(risk, age, "Dead")] = mortality

        return transitions

    def _advanced_transitions(self, state: State) -> Dict[State, float]:
        """Transitions from Advanced state."""
        risk, age = state.risk, state.age
        cancer_death = self.progression["advanced_to_dead"]
        natural_death = self.mortality[age]

        total_death = cancer_death + natural_death - (cancer_death * natural_death)

        transitions = {}

        if total_death > 0:
            transitions[State(risk, age, "Dead")] = total_death

        p_survive = 1 - total_death
        if p_survive > 0:
            transitions[State(risk, age, "Advanced")] = p_survive

        return transitions
