"""
MDP Model Parameters - Biennial Screening Model

This module defines all medical/economic parameters for the breast cancer
screening MDP and converts annual parameters to 2-year decision epochs.

References:
- USPSTF 2024: "Biennial screening mammography for women aged 40 to 74 years"
- PMC4894487: CISNET model screening disutility values
- PMC5638217, PubMed 8667536: Screening performance by age
"""

from typing import Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loaders import (
    load_seer_incidence_data,
    load_ssa_mortality_data,
    load_seer_survival_data
)
from data.processors import (
    aggregate_to_mdp_age_groups,
    calculate_annual_death_rate,
    calculate_age_group_mortality
)
from config.constants import AGE_GROUPS


# =============================================================================
# 1. TIME STEP & CONVERSION
#    USPSTF 2024 recommends biennial screening → each decision epoch = 2 years
#    P(2-year) = 1 - (1 - P(annual))^2
# =============================================================================

TIME_STEP_YEARS = 2


def annual_to_biennial(annual_prob: float) -> float:
    """Convert annual probability to 2-year probability."""
    return 1 - (1 - annual_prob) ** TIME_STEP_YEARS


# =============================================================================
# 2. SCREENING COSTS (QALY disutility per screen)
#    Source: Mittmann et al., Health Rep 2015 (PMC4894487)
#    URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4894487/
#    Original: de Haes et al., Int J Cancer 1991;49:538-44
#
#    screening_cost: disutility 0.006 × 1 week / 52 weeks ≈ 0.000115
#    false_positive_cost: disutility 0.105 × 5 weeks / 52 weeks ≈ 0.010
# =============================================================================

SCREENING_COSTS = {
    "screening_cost": -(0.006 * 1 / 52),   # ≈ -0.000115 QALY per mammogram
    "false_positive_cost": -(0.105 * 5 / 52),  # ≈ -0.010096 QALY per false positive
}


# =============================================================================
# 3. SCREENING PERFORMANCE BY AGE
#    Source: PMC5638217, PubMed 8667536
#    Younger women have denser breast tissue → lower accuracy
# =============================================================================

SCREENING_BY_AGE = {
    "30-39": {"sensitivity": 0.75, "specificity": 0.80,
              "false_negative": 0.25, "false_positive": 0.20},
    "40-49": {"sensitivity": 0.78, "specificity": 0.85,
              "false_negative": 0.22, "false_positive": 0.15},
    "50-59": {"sensitivity": 0.85, "specificity": 0.90,
              "false_negative": 0.15, "false_positive": 0.10},
    "60+":   {"sensitivity": 0.87, "specificity": 0.92,
              "false_negative": 0.13, "false_positive": 0.08},
}


# =============================================================================
# 4. HEALTH UTILITIES (QALY per year in each health state)
#    1.00 = perfect health, 0.00 = dead
#    Biennial = annual × 2 (accumulated over 2-year epoch)
# =============================================================================

HEALTH_UTILITIES_ANNUAL = {
    "Healthy": 1.00,
    "Early-Undetected": 1.00,   # patient unaware → no quality impact
    "Early-Detected": 0.71,     # undergoing treatment (surgery/chemo)
    "Cured": 0.88,              # post-treatment residual effects
    "Advanced": 0.45,           # late-stage cancer
    "Dead": 0.00,
}

HEALTH_UTILITIES_BIENNIAL = {
    state: utility * TIME_STEP_YEARS
    for state, utility in HEALTH_UTILITIES_ANNUAL.items()
}


# =============================================================================
# 5. DISEASE PROGRESSION PROBABILITIES
#    Annual rates → converted to biennial where time-dependent
#    Treatment outcomes (cure/failure) are one-time, not converted
# =============================================================================

PROGRESSION_ANNUAL = {
    "early_to_advanced": 0.27,           # untreated early → 27%/yr advance
    "early_detected_to_cured": 0.90,     # treatment success rate
    "early_detected_to_advanced": 0.10,  # treatment failure rate
    "cured_to_recurrence": 0.016,        # recurrence 1.6%/yr
}

PROGRESSION_BIENNIAL = {
    "early_to_advanced": annual_to_biennial(PROGRESSION_ANNUAL["early_to_advanced"]),
    "early_detected_to_cured": PROGRESSION_ANNUAL["early_detected_to_cured"],
    "early_detected_to_advanced": PROGRESSION_ANNUAL["early_detected_to_advanced"],
    "cured_to_recurrence": annual_to_biennial(PROGRESSION_ANNUAL["cured_to_recurrence"]),
}


# =============================================================================
# 6. RISK STRATIFICATION
#    Low  = general population (SEER baseline)
#    Medium = family history (HR = 2.0×)
#    High = BRCA carriers (average of BRCA1 & BRCA2 relative risk)
# =============================================================================

BRCA_RELATIVE_RISK = {
    "30-39": (33, 16),    # (BRCA1×, BRCA2×)
    "40-49": (32, 9.9),
    "50-59": (18, 12),
    "60+":   (14, 11),    # from 60-69 literature data
}

FAMILY_HISTORY_HR = 2.0


# =============================================================================
# 7. PARAMETER BUILDERS
#    Load raw CSV data → aggregate → convert to biennial → output final params
# =============================================================================

def build_incidence_rates_biennial() -> Dict[str, Dict[str, float]]:
    """SEER incidence × risk multipliers → 2-year rates by risk & age."""
    seer_raw = load_seer_incidence_data()
    low_risk_annual = aggregate_to_mdp_age_groups(seer_raw)

    low_risk = {age: annual_to_biennial(rate)
                for age, rate in low_risk_annual.items()}

    medium_risk = {age: annual_to_biennial(low_risk_annual[age] * FAMILY_HISTORY_HR)
                   for age in AGE_GROUPS}

    brca_avg_rr = {age: sum(BRCA_RELATIVE_RISK[age]) / 2
                   for age in AGE_GROUPS}
    high_risk = {age: annual_to_biennial(low_risk_annual[age] * brca_avg_rr[age])
                 for age in AGE_GROUPS}

    return {
        "High": {age: round(high_risk[age], 5) for age in AGE_GROUPS},
        "Medium": {age: round(medium_risk[age], 5) for age in AGE_GROUPS},
        "Low": {age: round(low_risk[age], 5) for age in AGE_GROUPS},
    }


def build_natural_mortality_biennial() -> Dict[str, float]:
    """SSA life table → 2-year background mortality by age."""
    ssa_data = load_ssa_mortality_data()
    mortality_annual = calculate_age_group_mortality(ssa_data)
    return {age: round(annual_to_biennial(rate), 6)
            for age, rate in mortality_annual.items()}


def build_progression_rates_biennial() -> Dict[str, float]:
    """SEER survival + disease params → 2-year progression rates."""
    seer_survival = load_seer_survival_data()
    distant_5yr_survival = seer_survival["Distant"]
    advanced_death_rate_annual = calculate_annual_death_rate(distant_5yr_survival)

    progression = PROGRESSION_BIENNIAL.copy()
    progression["advanced_to_dead"] = round(annual_to_biennial(advanced_death_rate_annual), 3)
    return progression


def build_rewards_biennial() -> Dict[str, float]:
    """Health utilities + screening costs → complete reward dict."""
    rewards = HEALTH_UTILITIES_BIENNIAL.copy()
    rewards.update(SCREENING_COSTS)
    return rewards


# =============================================================================
# 8. EXPORT (computed at import time)
# =============================================================================

INCIDENCE_RATE_BIENNIAL = build_incidence_rates_biennial()
NATURAL_MORTALITY_BIENNIAL = build_natural_mortality_biennial()
PROGRESSION_BIENNIAL = build_progression_rates_biennial()
REWARDS_BIENNIAL = build_rewards_biennial()
