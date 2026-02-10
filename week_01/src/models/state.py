"""
State Definition for Breast Cancer Screening MDP

A state represents a patient's current situation:
- Risk level (genetic/family history)
- Age group
- Health status
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    """
    Immutable state representation for the MDP.

    Attributes:
        risk: Risk level ("High", "Medium", "Low")
        age: Age group ("30-39", "40-49", "50-59", "60+")
        health: Health status ("Healthy", "Early-Undetected", etc.)
    """
    risk: str
    age: str
    health: str

    def __repr__(self) -> str:
        return f"State({self.risk}, {self.age}, {self.health})"

    def __str__(self) -> str:
        return f"({self.risk}, {self.age}, {self.health})"
