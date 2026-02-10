# Code Architecture

## Overview

The codebase follows a clean architecture with clear separation of concerns:

```
week_01/
├── README.md                       # Project documentation
├── data/                           # Raw data files (auto-loaded)
│   ├── explorer_download.csv       # SEER incidence data
│   ├── ssa_life_table_female_2022.csv  # SSA mortality data
│   └── seer_survival_by_stage.csv  # SEER survival data
├── docs/                           # Documentation
│   ├── ARCHITECTURE.md             # This document
│   ├── DATA_SOURCES.md             # Parameter sources
│   ├── MODEL_DESIGN.md             # MDP model design
│   ├── REPORT.md                   # Detailed report
│   └── PRESENTATION.md            # Presentation slides
├── src/
│   ├── main.py                     # Entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── constants.py            # Pure constants (no computation)
│   │   └── parameters_biennial.py  # Computed parameters (from data + literature)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py              # CSV file readers
│   │   └── processors.py          # Data transformation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── state.py                # State class
│   │   ├── transitions_v2.py       # Transition probability calculator (age-stratified)
│   │   ├── rewards.py              # Reward calculator
│   │   └── mdp.py                  # Main MDP class
│   ├── algorithms/
│   │   ├── __init__.py
│   │   └── policy_iteration.py     # Policy iteration implementation
│   └── utils/
│       ├── __init__.py
│       ├── output.py               # Console output formatting
│       └── export.py               # CSV export functions
└── results/
    ├── optimal_policy.csv          # Output: optimal actions for each state
    └── value_function.csv          # Output: expected QALY for each state
```

---

## Module Responsibilities

### 1. `config/` - Configuration

**Purpose:** Store all constants and configuration values.

| File | Responsibility |
|------|----------------|
| `constants.py` | Pure constants (paths, state space definition, discount factor) |
| `parameters_biennial.py` | Computed parameters (loads data, applies literature values, converts to biennial) |

```python
# constants.py - No computation
GAMMA = 0.97
RISK_LEVELS = ["High", "Medium", "Low"]

# parameters_biennial.py - Builds from data
INCIDENCE_RATE_BIENNIAL = build_incidence_rates_biennial()  # Loads SEER CSV
```

### 2. `data/` - Data Layer

**Purpose:** Handle all data I/O and transformation.

| File | Responsibility |
|------|----------------|
| `loaders.py` | Read CSV files, return raw data |
| `processors.py` | Transform raw data to MDP format |

```python
# loaders.py - Just reads
def load_seer_incidence_data() -> Dict[str, float]:
    # Returns {"30-34": 31.9, "35-39": 66.8, ...}

# processors.py - Transforms
def aggregate_to_mdp_age_groups(seer_data) -> Dict[str, float]:
    # Returns {"30-39": 0.00049, "40-49": 0.00172, ...}
```

### 3. `models/` - Domain Models

**Purpose:** Define the MDP structure and calculations.

| File | Responsibility |
|------|----------------|
| `state.py` | State data class |
| `transitions_v2.py` | P(s'\|s,a) calculations (age-stratified screening) |
| `rewards.py` | R(s,a) calculations |
| `mdp.py` | Main MDP class (combines all) |

```python
# Clean separation
class TransitionCalculatorV2:
    def get_transitions(state, action) -> Dict[State, float]

class RewardCalculator:
    def get_reward(state, action) -> float

class BreastCancerScreeningMDP:
    def __init__(self):
        self._transition_calc = TransitionCalculatorV2(...)
        self._reward_calc = RewardCalculator(...)
```

### 4. `algorithms/` - Algorithms

**Purpose:** MDP solving algorithms.

| File | Responsibility |
|------|----------------|
| `policy_iteration.py` | Policy iteration implementation |

```python
class PolicyIteration:
    def solve() -> Tuple[policy, V]
    def policy_evaluation() -> int
    def policy_improvement() -> bool
    def get_q_values(state) -> Dict[str, float]
```

### 5. `utils/` - Utilities

**Purpose:** Output formatting and file export.

| File | Responsibility |
|------|----------------|
| `output.py` | Console printing functions (policy, value function, Q-values, summary) |
| `export.py` | CSV export functions |

---

## Data Flow

```
CSV Files                    Literature Values
    │                              │
    ▼                              ▼
┌─────────┐                  ┌──────────┐
│ loaders │                  │constants │
└────┬────┘                  └────┬─────┘
     │                            │
     ▼                            ▼
┌────────────┐              ┌────────────────────┐
│ processors │──────────────│ parameters_biennial │
└────────────┘              └─────────┬──────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │    models    │
                              │  (MDP, T, R) │
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │  algorithms  │
                              │     (PI)     │
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │    utils     │
                              │ (output/CSV) │
                              └──────────────┘
```

---

## Key Design Principles

### 1. Single Responsibility
Each module has ONE job:
- `loaders.py` - Only reads files
- `processors.py` - Only transforms data
- `transitions_v2.py` - Only calculates P(s'|s,a)

### 2. Dependency Injection
Calculators receive parameters, don't fetch them:
```python
# Good - receives dependencies
class TransitionCalculatorV2:
    def __init__(self, incidence_rates, mortality_rates, ...):
        self.incidence = incidence_rates
```

### 3. Immutable State
State class is frozen dataclass:
```python
@dataclass(frozen=True)
class State:
    risk: str
    age: str
    health: str
```

### 4. Clear Interfaces
Each class has a clear public interface:
```python
class BreastCancerScreeningMDP:
    def get_actions(state) -> List[str]
    def get_transition_prob(state, action) -> Dict[State, float]
    def get_reward(state, action) -> float
    def is_terminal(state) -> bool
```

---

## Running the Code

```bash
# From src/ directory
python main.py
```

---

## Extending the Code

### Add New Risk Factor
1. Update `config/constants.py` - Add to `RISK_LEVELS`
2. Update `config/parameters_biennial.py` - Add risk multiplier
3. Update `models/transitions_v2.py` - If transition logic differs

### Add New Algorithm (e.g., Value Iteration)
1. Create `algorithms/value_iteration.py`
2. Implement same interface as `PolicyIteration`
3. Update `algorithms/__init__.py`

### Add New Output Format
1. Create function in `utils/output.py` or `utils/export.py`
2. Call from `main.py`

---

## Testing

```python
from models.state import State
from models.mdp import BreastCancerScreeningMDP
from config.parameters_biennial import INCIDENCE_RATE_BIENNIAL, SCREENING_BY_AGE

# Create MDP
mdp = BreastCancerScreeningMDP()

# Test transition
state = State("High", "40-49", "Healthy")
transitions = mdp.get_transition_prob(state, "Screen")
print(transitions)
```
