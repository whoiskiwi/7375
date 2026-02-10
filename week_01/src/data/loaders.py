"""
Data Loaders for Breast Cancer Screening MDP

This module handles loading raw data from CSV files.
No processing or calculations - just reading data.

Data Sources:
- SEER Cancer Statistics (2018-2022)
- US SSA Life Tables (2022)
- SEER Survival by Stage
"""

import csv
from pathlib import Path
from typing import Dict

from config.constants import (
    SEER_INCIDENCE_FILE,
    SSA_MORTALITY_FILE,
    SEER_SURVIVAL_FILE,
    SEER_HEADER_ROWS
)


def load_seer_incidence_data(filepath: Path = None) -> Dict[str, float]:
    """
    Load SEER incidence data from CSV file.

    Source: SEER Cancer Statistics 2018-2022
    URL: https://seer.cancer.gov/statistics-network/explorer/

    Args:
        filepath: Path to the SEER CSV file. Defaults to config path.

    Returns:
        dict: Age group to incidence rate per 100,000 mapping
              e.g., {"30-34": 31.9, "35-39": 66.8, ...}
    """
    if filepath is None:
        filepath = SEER_INCIDENCE_FILE

    seer_data = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)

        # Skip header rows
        for _ in range(SEER_HEADER_ROWS):
            next(reader)

        # Read data rows
        for row in reader:
            if len(row) < 2:
                continue

            age_group = row[0].strip().strip('"')
            rate_str = row[1].strip().strip('"')

            # Skip suppressed data (marked as "^") and empty rows
            if rate_str == "^" or rate_str == "" or age_group == "":
                continue

            # Stop at footnotes
            if age_group.startswith("^") or age_group.startswith("Data"):
                break

            try:
                rate = float(rate_str)
                seer_data[age_group] = rate
            except ValueError:
                continue

    return seer_data


def load_ssa_mortality_data(filepath: Path = None) -> Dict[int, float]:
    """
    Load SSA Life Table mortality data from CSV file.

    Source: US Social Security Administration Period Life Table 2022
    URL: https://www.ssa.gov/oact/STATS/table4c6.html

    Args:
        filepath: Path to the SSA CSV file. Defaults to config path.

    Returns:
        dict: Age to death probability mapping
              e.g., {30: 0.000988, 31: 0.001053, ...}
    """
    if filepath is None:
        filepath = SSA_MORTALITY_FILE

    mortality_data = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            age = int(row['Age'])
            death_prob = float(row['Death_Probability'])
            mortality_data[age] = death_prob

    return mortality_data


def load_seer_survival_data(filepath: Path = None) -> Dict[str, float]:
    """
    Load SEER 5-year survival by stage data from CSV file.

    Source: SEER Cancer Stat Facts
    URL: https://seer.cancer.gov/statfacts/html/breast.html

    Args:
        filepath: Path to the SEER survival CSV file. Defaults to config path.

    Returns:
        dict: Stage to 5-year survival rate mapping
              e.g., {"Localized": 1.0, "Regional": 0.872, "Distant": 0.326}
    """
    if filepath is None:
        filepath = SEER_SURVIVAL_FILE

    survival_data = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = row['Stage']
            survival_5yr = float(row['Five_Year_Survival'])
            survival_data[stage] = survival_5yr

    return survival_data
