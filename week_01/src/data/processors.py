"""
Data Processors for Breast Cancer Screening MDP

This module handles data transformation and calculations.
Takes raw data and converts it to MDP-ready format.
"""

from typing import Dict
from config.constants import (
    SEER_RATE_PER_POPULATION,
    SEER_AGE_AGGREGATION,
    SSA_AGE_RANGES,
    SURVIVAL_YEARS,
)


def aggregate_to_mdp_age_groups(seer_data: Dict[str, float]) -> Dict[str, float]:
    """
    Aggregate SEER 5-year age groups to MDP model age groups.

    SEER provides: 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, etc.
    MDP model uses: 30-39, 40-49, 50-59, 60+

    Args:
        seer_data: Dictionary of SEER age groups to rates per 100,000

    Returns:
        dict: MDP age groups to annual probability (as decimal)
    """
    mdp_rates = {}

    for mdp_age, seer_ages in SEER_AGE_AGGREGATION.items():
        rates = [seer_data[ag] for ag in seer_ages if ag in seer_data]
        if rates:
            mdp_rates[mdp_age] = sum(rates) / len(rates) / SEER_RATE_PER_POPULATION

    return mdp_rates


def calculate_annual_death_rate(five_year_survival: float) -> float:
    """
    Calculate annual death rate from 5-year survival rate.

    Formula:
        annual_survival = 5yr_survival^(1/5)
        annual_death_rate = 1 - annual_survival

    Args:
        five_year_survival: 5-year survival rate (0-1)

    Returns:
        float: Annual death rate
    """
    annual_survival = five_year_survival ** (1 / SURVIVAL_YEARS)
    return 1 - annual_survival


def calculate_age_group_mortality(ssa_data: Dict[int, float]) -> Dict[str, float]:
    """
    Calculate average mortality rates for MDP age groups from SSA data.

    SSA provides: individual ages 0-119
    MDP model uses: 30-39, 40-49, 50-59, 60+

    Args:
        ssa_data: Dictionary of age to death probability

    Returns:
        dict: MDP age groups to average annual mortality rate
    """
    mdp_mortality = {}

    for mdp_age, (start, end) in SSA_AGE_RANGES.items():
        rates = [ssa_data[age] for age in range(start, end) if age in ssa_data]
        if rates:
            mdp_mortality[mdp_age] = sum(rates) / len(rates)

    return mdp_mortality
