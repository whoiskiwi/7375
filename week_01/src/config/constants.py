"""
Constants and Configuration for Breast Cancer Screening MDP

This module contains all constant values and configuration settings.
No computation or data loading happens here - just pure constants.

References:
- Sanders et al., JAMA 2016 - Discount factor (3%)
- Model structure definitions
"""

from pathlib import Path

# =============================================================================
# FILE PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data files
SEER_INCIDENCE_FILE = DATA_DIR / "explorer_download.csv"
SSA_MORTALITY_FILE = DATA_DIR / "ssa_life_table_female_2022.csv"
SEER_SURVIVAL_FILE = DATA_DIR / "seer_survival_by_stage.csv"


# =============================================================================
# MDP STRUCTURE
# =============================================================================

# State Space Components
RISK_LEVELS = ["High", "Medium", "Low"]

AGE_GROUPS = ["30-39", "40-49", "50-59", "60+"]

HEALTH_STATES = [
    "Healthy",
    "Early-Undetected",
    "Early-Detected",
    "Cured",
    "Advanced",
    "Dead"
]

# Decision states (states where actions are required)
DECISION_STATES = ["Healthy", "Early-Undetected", "Cured"]

# Terminal states
TERMINAL_STATES = ["Dead"]

# Action Space
ACTIONS = ["Screen", "Wait"]
NO_DECISION_ACTION = "None"

# Default/reference age group for fallback and FP calculation
REFERENCE_AGE_GROUP = "50-59"

# Display conversion
DAYS_PER_YEAR = 365


# =============================================================================
# DATA PROCESSING
# =============================================================================

# SEER CSV format: first N rows are headers, rates are per 100,000
SEER_HEADER_ROWS = 6
SEER_RATE_PER_POPULATION = 100_000

# SEER 5-year age groups → MDP age groups mapping
SEER_AGE_AGGREGATION = {
    "30-39": ["30-34", "35-39"],
    "40-49": ["40-44", "45-49"],
    "50-59": ["50-54", "55-59"],
    "60+": ["60-64", "65-69", "70-74", "75-79", "80-84"],
}

# SSA single-age → MDP age groups mapping (start inclusive, end exclusive)
SSA_AGE_RANGES = {
    "30-39": (30, 40),
    "40-49": (40, 50),
    "50-59": (50, 60),
    "60+": (60, 86),  # Upper bound per SSA life table coverage
}

# 5-year survival conversion
SURVIVAL_YEARS = 5


# =============================================================================
# DISCOUNT FACTOR
# =============================================================================

# Source: Haacker et al., Health Policy and Planning 2020;35(1):107
# "On discount rates for economic evaluations in global health"
# Recommendation: 3% annual discount rate (standard for US-based evaluations)
# URL: https://academic.oup.com/heapol/article/35/1/107/5591528

DISCOUNT_RATE = 0.03  # 3% annual discount rate
GAMMA = 1 / (1 + DISCOUNT_RATE)  # ≈ 0.97


# =============================================================================
# ALGORITHM SETTINGS
# =============================================================================

# Source: Sutton & Barto, "Reinforcement Learning: An Introduction", 2nd Edition
# Chapter 4.3 - Policy Iteration, Figure 4.3 (p.97)
# θ: "a small positive number determining the accuracy of estimation"
# URL: https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf

CONVERGENCE_THRESHOLD = 1e-6  # θ for policy evaluation convergence
MAX_EVAL_ITERATIONS = 1000    # Safety cap to prevent infinite loops
