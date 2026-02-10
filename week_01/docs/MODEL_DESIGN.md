# Breast Cancer Screening MDP Model Design

## 1. Problem Background

### Core Question
> **Should a woman undergo breast cancer screening in this 2-year period?**

Breast cancer is the most common cancer in women. Early detection significantly improves survival rates (early-stage 5-year survival ~100%, late-stage only 32.6%). However, screening also has costs: anxiety from false positives, unnecessary biopsies, and discomfort from the procedure itself.

### Objective
Find the optimal screening strategy that maximizes expected cumulative Quality-Adjusted Life Years (QALY), aligned with **USPSTF 2024 guidelines**.

### USPSTF 2024 Recommendation
> "The USPSTF recommends **biennial** screening mammography for women aged 40 to 74 years" (Grade B)
>
> *Source: JAMA 2024;331(22):1918-1930, DOI: 10.1001/jama.2024.5534*

### Target Population
- BRCA gene mutation carriers (High Risk)
- Women with family history (Medium Risk)
- General population (Low Risk)

---

## 2. MDP Framework (S, A, P, R, γ)

```
┌─────────────────────────────────────────────────────────────┐
│              BIENNIAL MDP Model Structure                    │
├─────────────────────────────────────────────────────────────┤
│  Decision Epoch = 2 years (biennial screening)              │
│                                                             │
│  S = State Space = (Risk Level, Age Group, Health Status)   │
│      3 × 4 × 6 = 72 states                                  │
│                                                             │
│  A = Action Space = {Screen, Wait}                          │
│                                                             │
│  P = Transition Probabilities (2-year epoch)                │
│      P(2yr) = 1 - (1 - P(annual))²                         │
│                                                             │
│  R = Reward Function (QALY accumulated over 2 years)        │
│                                                             │
│  γ = 0.97² ≈ 0.94 (Discount Factor for 2-year epoch)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. State Space S

**Definition:** S = (Risk Level, Age Group, Health Status)

### 3.1 Risk Levels (3)

| Level | Description | Data Source |
|-------|-------------|-------------|
| High | BRCA1/BRCA2 carriers | Antoniou et al., 2003 (RR: 12.5x-24.5x) |
| Medium | First-degree relative with breast cancer | Brewer et al., 2017 (HR: 2.0x) |
| Low | General population | SEER 2018-2022 baseline |

### 3.2 Age Groups (4)

| Group | Age Range | Rationale |
|-------|-----------|-----------|
| 1 | 30-39 years | Pre-guideline screening age |
| 2 | 40-49 years | Controversial screening age |
| 3 | 50-59 years | Standard screening age |
| 4 | 60+ years | Continued screening consideration |

### 3.3 Health Status (6)

| Status | Description | Decision Required |
|--------|-------------|-------------------|
| Healthy | No cancer | Yes |
| Early-Undetected | Early-stage cancer, not yet discovered | Yes |
| Early-Detected | Early-stage cancer, in treatment | No (auto-transition) |
| Cured | Successfully treated, monitoring | Yes |
| Advanced | Late-stage cancer | No (auto-transition) |
| Dead | Terminal state | No (absorbing state) |

**Total State Space Size: 3 × 4 × 6 = 72 states**

### 3.4 State Transition Diagram

```
                    ┌─────────────────────────────────────────┐
                    │              Recurrence                 │
                    │               1.6%/yr                   │
                    ▼                                         │
              ┌─────────┐  Incidence  ┌──────────────────┐    │
         ┌───►│ Healthy │────────────►│ Early-Undetected │    │
         │    └─────────┘             └────────┬─────────┘    │
         │         │                           │              │
         │         │                  Screen   │  Undetected  │
         │         │                  (78%)    │  + Progress  │
         │         │                           ▼              │
         │         │                  ┌────────────────┐      │
         │         │                  │ Early-Detected │      │
         │         │                  └───────┬────────┘      │
         │         │                          │               │
         │         │                 Success  │  Failure      │
         │         │                  (90%)   │  (10%)        │
         │         │                          ▼               │
         │    ┌────┴──┐              ┌──────────────┐         │
         └────│ Cured │◄─────────────│   Advanced   │─────────┘
              └───────┘              └──────┬───────┘
                                            │ 20.1%/yr
                                            ▼
                                       ┌────────┐
                                       │  Dead  │
                                       └────────┘
```

---

## 4. Action Space A

```
A = {Screen, Wait}
```

### 4.1 Screen Action

When screening is performed:

| Condition | Outcome | Probability |
|-----------|---------|-------------|
| Cancer present | Detected → Early-Detected | 78% (sensitivity) |
| Cancer present | Missed → Early-Undetected | 22% (false negative) |
| No cancer | Correct negative → Healthy | 78% (specificity) |
| No cancer | False positive → Healthy + cost | 22% (false positive) |

### 4.2 Wait Action

When waiting (no screening):

| Condition | Outcome | Probability |
|-----------|---------|-------------|
| Develops cancer | → Early-Undetected | Incidence rate |
| No cancer | → Healthy | 1 - Incidence rate |

### 4.3 Decision States

Actions are only required in these states:
- **Healthy** - Decide whether to screen for early detection
- **Early-Undetected** - Decide whether to screen (patient unaware of cancer)
- **Cured** - Decide whether to continue surveillance screening

Other states have automatic transitions:
- **Early-Detected** → Treatment outcomes (Cured or Advanced)
- **Advanced** → Death or continued Advanced state
- **Dead** → Absorbing state (no transition)

---

## 5. Transition Probabilities P(s'|s,a)

### 5.1 Cancer Incidence Rates

**Source:** SEER 2018-2022, Antoniou 2003, Brewer 2017

| Risk Level | 30-39 | 40-49 | 50-59 | 60+ |
|------------|-------|-------|-------|-----|
| High (BRCA) | 1.21% | 3.60% | 3.86% | 5.43% |
| Medium (Family) | 0.10% | 0.34% | 0.52% | 0.87% |
| Low (General) | 0.05% | 0.17% | 0.26% | 0.43% |

**Calculation Method:**
- Low = SEER incidence data (per 100,000 → probability)
- Medium = Low × 2.0 (Family History HR)
- High = Low × BRCA Relative Risk (age-specific)

### 5.2 Screening Performance (Age-Stratified)

**Sources:**
- PMC5638217: "sensitivity about 75% for women aged under 50, about 85% for women aged over 50"
- PubMed 8667536: "87% in women aged 60-70 years"

| Age Group | Sensitivity | Specificity | Source |
|-----------|-------------|-------------|--------|
| 30-39 | 75% | 80% | PMC5638217 (dense breasts) |
| 40-49 | 78% | 85% | PMC5638217 (transitional) |
| 50-59 | 85% | 90% | PMC5638217 |
| 60+ | 87% | 92% | PubMed 8667536 |

**Rationale:** Younger women have denser breast tissue, reducing mammography accuracy. Sensitivity increases with age as breast density decreases.

### 5.3 Disease Progression

| Transition | Probability | Source |
|------------|-------------|--------|
| Early-Undetected → Advanced | 27%/year | PMC9011255 (Wu 2022) |
| Early-Detected → Cured | 90% | PMC10314986 (Marcadis 2022) |
| Early-Detected → Advanced | 10% | 1 - Cure rate |
| Cured → Recurrence | 1.6%/year | PMC8902439 (Pedersen et al., 2021) |
| Advanced → Dead | 20.1%/year | SEER (5yr survival 32.6%) |

### 5.4 Natural Mortality

**Source:** US SSA Life Table 2022 (Female)

| Age Group | Annual Death Rate | 2-Year Rate |
|-----------|-------------------|-------------|
| 30-39 | 0.131% | 0.26% |
| 40-49 | 0.237% | 0.47% |
| 50-59 | 0.499% | 1.00% |
| 60+ | 2.781% | 5.49% |

---

## 6. Reward Function R(s,a)

### 6.1 State Rewards (QALY per year)

**Source:** PMC9189726 (Kaur et al., 2022 - Systematic review of 69 studies)

| Health State | QALY/year | Literature Range | Rationale |
|--------------|-----------|------------------|-----------|
| Healthy | +1.00 | - | Full quality of life (baseline) |
| Early-Undetected | +1.00 | - | Unaware of cancer, same QoL |
| Early-Detected | +0.71 | 0.52-0.80 | During active treatment |
| Cured | +0.88 | 0.71-0.95 | Post-treatment survivors |
| Advanced | +0.45 | 0.08-0.82 | Metastatic cancer |
| Dead | 0.00 | - | Terminal state |

### 6.2 Action Costs

**Sources:**
- PMC4894487 (Mittmann et al., 2015) - CISNET Canadian Model
- PMC6655768 (2019) - Systematic Review of Screening Disutility

| Event | QALY Cost | Literature Value | Source |
|-------|-----------|------------------|--------|
| Screening procedure | -0.000115 | 0.006 disutility × 1/52 year = 0.000115 | PMC4894487 |
| False positive result | -0.010 | 0.105 disutility × 5 weeks = 0.0101 | PMC4894487 |

**Note:** Screening cost = 0.006 disutility × 1/52 year = 0.000115 QALY (PMC4894487). PMC6655768 systematic review confirms: "QALYs loss around 0–0.0013" for screening attendance.

### 6.3 Reward Calculation

For each state-action pair:

```
R(s, a) = State_Reward + Action_Cost + False_Positive_Cost (if applicable)

Example: R(Healthy, Screen) with no cancer and false positive:
= 1.00 + (-0.000115) + (-0.010) = 0.989885 QALY
```

---

## 7. Discount Factor γ

```
γ = 1 / (1 + r) = 1 / 1.03 ≈ 0.97

where r = 0.03 (3% annual discount rate)
```

**Source:** Sanders GD, Neumann PJ, et al. (2016). "Recommendations for Conduct, Methodological Practices, and Reporting of Cost-effectiveness Analyses: Second Panel on Cost-Effectiveness in Health and Medicine." JAMA.
- URL: https://jamanetwork.com/journals/jama/fullarticle/2552214
- Recommendation: "Both costs and health effects should be discounted at an annual rate of 3%"

**Supporting Guidelines:**

| Organization | Recommended Rate | Reference |
|--------------|------------------|-----------|
| US Panel on Cost-Effectiveness | 3% | Sanders et al., JAMA 2016 |
| WHO-CHOICE | 3% | WHO Guidelines 2003 |
| NICE (UK) | 3.5% | NICE Methods Guide |
| Canadian CADTH | 1.5% | CADTH Guidelines 2017 |

**Rationale for Discounting:**
- **Time preference** - People value current health benefits more than future ones
- **Opportunity cost** - Resources could be invested for returns
- **Uncertainty** - Future outcomes are uncertain
- **Consistency** - Enables comparison across different studies

**Effect on Value Function:**
- Short-term benefits weighted more heavily
- Long-term survival still valuable but discounted
- Prevents infinite value accumulation

---

## 8. Solution Algorithm: Policy Iteration

### 8.1 Algorithm Overview

```
Algorithm: Policy Iteration

Input:  MDP = (S, A, P, R, γ)
Output: Optimal policy π*

1. INITIALIZATION
   ├── π(s) = "Wait" for all decision states
   └── V(s) = 0 for all states

2. POLICY EVALUATION
   Repeat until convergence (Δ < θ):
     For each state s:
       V(s) = R(s, π(s)) + γ × Σ P(s'|s, π(s)) × V(s')

3. POLICY IMPROVEMENT
   For each state s:
     π(s) = argmax_a [ R(s,a) + γ × Σ P(s'|s,a) × V(s') ]

4. CONVERGENCE CHECK
   If policy unchanged → return π*
   Else → go to step 2
```

### 8.2 Convergence Properties

| Metric | Value |
|--------|-------|
| Iterations to converge | 3 |
| Evaluation iterations per policy | 270-430 |
| Convergence threshold | 1e-6 |
| Total computation time | < 0.5 seconds |

### 8.3 Q-Value Calculation

For comparing actions:

```
Q(s, a) = R(s, a) + γ × Σ P(s'|s, a) × V(s')

Optimal action: π*(s) = argmax_a Q(s, a)
```

---

## 9. References

> For experimental results and analysis, see [REPORT.md](REPORT.md).
> For code architecture and file structure, see [ARCHITECTURE.md](ARCHITECTURE.md).
> For complete citations and parameter derivations, see [DATA_SOURCES.md](DATA_SOURCES.md).
