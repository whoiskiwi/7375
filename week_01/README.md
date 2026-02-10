# Breast Cancer Screening MDP Model

A Markov Decision Process (MDP) model for optimizing breast cancer screening decisions, aligned with **USPSTF 2024 guidelines**.

## 1. Problem Definition

### Objective
Determine the optimal screening strategy for women with varying breast cancer risk levels.

### Key Question
> Should a woman undergo breast cancer screening in this 2-year period?

### Target Population
- BRCA gene mutation carriers
- Women with family history of breast cancer
- General population for comparison

### Decision Frequency
**Biennial (every 2 years)** - aligned with USPSTF 2024 recommendation

> **Reference:** USPSTF 2024 recommends "biennial screening mammography for women aged 40 to 74 years" (JAMA 2024;331(22):1918-1930)

### Why Biennial (2-Year) Instead of Annual?

**Primary Reason:** USPSTF 2024 explicitly recommends biennial screening based on extensive evidence review.

| Factor | Annual Screening | Biennial Screening |
|--------|------------------|-------------------|
| **Detection Benefit** | Slightly higher | Sufficient (early-stage cancer grows slowly) |
| **Cumulative False Positives** | ~61% over 10 years | ~42% over 10 years |
| **Radiation Exposure** | 2× | Baseline |
| **Healthcare Cost** | 2× | Baseline |
| **USPSTF Recommendation** | Not recommended | Grade B |

**Evidence Summary (JAMA 2024):**
- Biennial screening provides most of the mortality reduction benefit of annual screening
- Annual screening doubles the cumulative risk of false-positive results
- The harm-benefit ratio favors biennial intervals for average-risk women

**Why Model Uses 2-Year Decision Epochs:**

> The model adopts **2-year decision epochs** aligned with **USPSTF 2024 clinical guidelines**, ensuring model outputs can be directly applied to clinical decision-making.

| Rationale | Explanation |
|-----------|-------------|
| **Clinical Basis** | USPSTF 2024 recommends biennial screening |
| **Methodological Validity** | Decision epoch matches intervention interval, ensuring action space validity |
| **Practical Applicability** | Model output π*(s) directly maps to real clinical choices |

**Parameter Conversion (Annual → Biennial):**

```
# Probability conversion
P(2-year) = 1 - (1 - P(annual))²

# Example: Annual incidence 0.5%
P(2-year) = 1 - (1 - 0.005)² = 0.998% ≈ 1.0%

# Discount factor adjustment
γ_biennial = γ_annual² = 0.97² ≈ 0.94

# QALY rewards: accumulated over 2 years
R(Healthy, 2-year) = 2.0 QALY
```

---

## 2. MDP Framework

### 2.1 State Space S

The state is defined as a tuple: **S = (Risk Level, Age Group, Health Status)**

#### Risk Levels (3)
| Level | Description |
|-------|-------------|
| High | BRCA carriers, strong family history |
| Medium | First-degree relative with breast cancer |
| Low | No special risk factors |

#### Age Groups (4)
| Group | Age Range |
|-------|-----------|
| 1 | 30-39 years |
| 2 | 40-49 years |
| 3 | 50-59 years |
| 4 | 60+ years |

#### Health Status (6)
| Status | Description |
|--------|-------------|
| Healthy | No cancer |
| Early-Undetected | Early-stage cancer, not yet discovered |
| Early-Detected | Early-stage cancer, diagnosed and in treatment |
| Cured | Successfully treated, requires monitoring |
| Advanced | Late-stage cancer |
| Dead | Terminal state |

**Total State Space Size: 3 × 4 × 6 = 72 states**

### 2.2 Action Space A

```
A = {Screen, Wait}
```

| Action | Description |
|--------|-------------|
| Screen | Perform mammography/MRI screening |
| Wait | Skip screening this year |

**Note:** Actions are only applicable in states: Healthy, Early-Undetected, Cured. Other states have automatic transitions.

### 2.3 Transition Probabilities P(s'|s,a)

#### Table 1: 2-Year Cancer Incidence Rate (Healthy → Early-Undetected)

*Based on SEER data, converted to 2-year probability: P(2yr) = 1 - (1 - P(annual))²*

| Risk Level | 30-39 | 40-49 | 50-59 | 60+ |
|------------|-------|-------|-------|-----|
| High (BRCA) | 2.40% | 7.07% | 7.58% | 10.56% |
| Medium (Family History) | 0.20% | 0.69% | 1.03% | 1.73% |
| Low (General Population) | 0.10% | 0.34% | 0.52% | 0.87% |

> **Data Calculation:**
> - **Low**: SEER 2018-2022 annual incidence → 2-year probability
> - **Medium**: Low × 2.0 (family history HR from Brewer et al.)
> - **High**: Low × BRCA relative risk (Antoniou et al.)

#### Table 2: Screening Performance (Age-Stratified)

*Sources: PMC5638217, PubMed 8667536*

| Age Group | Sensitivity | Specificity | Rationale |
|-----------|-------------|-------------|-----------|
| 30-39 | 75% | 80% | Dense breast tissue reduces accuracy |
| 40-49 | 78% | 85% | Transitional density |
| 50-59 | 85% | 90% | Post-menopausal, fattier breasts |
| 60+ | 87% | 92% | Lowest density, highest accuracy |

> **Note:** Age-stratified parameters improve alignment with USPSTF 2024 guidelines.

#### Table 3: Disease Progression

*Sources: PMC9011255, PMC10314986, PMC8902439, SEER*

| Transition | Probability | Description |
|------------|-------------|-------------|
| Early-Undetected → Advanced | 27%/year | Undetected cancer progression |
| Early-Detected → Cured | 90% | Early treatment success rate |
| Early-Detected → Advanced | 10% | Early treatment failure |
| Cured → Early-Undetected | 1.6%/year | Recurrence rate |
| Advanced → Dead | 20.1%/year | Late-stage mortality |

#### Table 4: Age-Related Natural Mortality

*Source: US SSA Actuarial Life Tables 2022 (Female)*

| Age Group | Annual Death Rate |
|-----------|-------------------|
| 30-39 | 0.131% |
| 40-49 | 0.237% |
| 50-59 | 0.499% |
| 60+ | 2.781% |

#### State Transition Diagram

```
                        Screen (detected 78%)
Healthy ──────────→ Early-Undetected ──────────→ Early-Detected
   ↑                       │                           │
   │                       │ (not detected 22%)        │ (treatment)
   │                       ↓                           ↓
   │                   Advanced ←───────────── (failure 10%)
   │                       │                           │
   │                       ↓                           ↓ (success 90%)
   │                     Dead                        Cured
   │                                                   │
   └─────────────── (recurrence 1.6%) ─────────────────┘
```

### 2.4 Reward Function R(s, a)

Rewards are measured in **Quality-Adjusted Life Years (QALY)**.

#### State Rewards (per year)

*Based on Health Utility Values from PMC9189726 (Kaur et al., 2022)*

| State | Reward (QALY) | Rationale |
|-------|---------------|-----------|
| Healthy | +1.00 | Full quality of life (baseline) |
| Cured | +0.88 | Post-treatment survivors (range: 0.71-0.95) |
| Early-Detected | +0.71 | During active treatment (range: 0.52-0.80) |
| Advanced | +0.45 | Metastatic/advanced stage (range: 0.08-0.82) |
| Dead | 0.00 | Terminal state |

#### Action Costs

*Based on PMC4894487 (Mittmann et al., 2015) - CISNET Canadian Model*

| Event | Cost (QALY) | Calculation | Source |
|-------|-------------|-------------|--------|
| Screening | -0.000115 | 0.006 × 1/52 = 0.000115 | PMC4894487 |
| False Positive | -0.010 | 0.105 × 5/52 = 0.0101 | PMC4894487 |

> **Literature Validation:**
> - PMC6655768 systematic review: "primary screening QALY loss around 0–0.0013"
> - PMC6655768: "false positive diagnosis disutility: 0–0.26"

#### Why QALY Costs Instead of Monetary Costs?

This model uses **QALY-based costs** (quality of life loss) rather than monetary costs. This is the **international standard** for health technology assessment.

**Rationale:**

| Aspect | QALY Costs | Monetary Costs |
|--------|------------|----------------|
| **Captures** | Anxiety, discomfort, time burden | Financial expenditure |
| **Patient-centered** | ✓ Reflects patient experience | ✗ System-focused |
| **Comparability** | Unified unit with health benefits | Requires conversion |
| **Equity** | Same for all income levels | Varies by income |

**What QALY Costs Capture:**

```
Screening QALY loss (-0.000115):
├── Anxiety before and during examination
├── Time off work / daily activities
└── Waiting for results (1-2 weeks stress)

False Positive QALY loss (-0.010):
├── Additional imaging and biopsies
├── Weeks of anxiety and uncertainty
└── Psychological impact of cancer scare
```

**International Standards Using QALY:**

| Organization | Country | QALY Required |
|--------------|---------|---------------|
| **NICE** | UK | ✓ Mandatory |
| **ICER** | USA | ✓ Standard |
| **CADTH** | Canada | ✓ Standard |
| **PBAC** | Australia | ✓ Standard |

**Key Reference:**

> Sanders et al., JAMA 2016 - "Second Panel on Cost-Effectiveness in Health and Medicine":
>
> "QALYs remain the recommended measure of health outcomes for cost-effectiveness analysis."
>
> URL: https://jamanetwork.com/journals/jama/fullarticle/2552214

**Why Not Include Monetary Costs?**

1. **Single optimization target**: Maximize health, not minimize spending
2. **USPSTF approach**: Guidelines focus on health benefits/harms, not costs
3. **Extensibility**: Economic analysis can be done separately if needed

### 2.5 Discount Factor

```
γ = 0.97
```

Interpretation: Future QALYs are slightly discounted, encouraging earlier health benefits.

---

## 3. Solution Algorithm: Policy Iteration

### Algorithm

```
Input:  S, A, P, R, γ
Output: Optimal policy π*

1. INITIALIZATION
   - Initialize random policy π(s) for all states
   - Initialize value function V(s) = 0 for all states

2. POLICY EVALUATION
   Repeat until convergence:
     For each state s:
       V(s) = R(s, π(s)) + γ × Σ P(s'|s, π(s)) × V(s')

3. POLICY IMPROVEMENT
   For each state s:
     π(s) = argmax_a [ R(s,a) + γ × Σ P(s'|s,a) × V(s') ]

4. CONVERGENCE CHECK
   If policy unchanged → return π*
   Else → go to step 2
```

### Convergence Criteria
- Policy stable (no changes in any state)
- Or value function change < θ (theta)

### Convergence Threshold θ = 1e-6

**Definition:** The algorithm stops policy evaluation when the maximum value change across all states is below θ:

```
max |V_new(s) - V_old(s)| < θ
```

**Why θ = 1e-6 is the standard choice:**

| Factor | Explanation |
|--------|-------------|
| **Numerical Precision** | 1e-6 avoids false convergence from floating-point noise while maintaining computational efficiency |
| **QALY Scale** | Value function V(s) ≈ 20-62 QALY; 1e-6 QALY ≈ 0.0003 days, far below any clinically meaningful difference |
| **Standard Practice** | Recommended range in RL literature: 1e-6 to 1e-8 |
| **Computational Balance** | Small enough for accuracy, large enough to avoid excessive iterations |

**Reference:**

> Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
>
> Chapter 4.1, p.75: "Policy evaluation [...] converges to V^π as k → ∞. A natural stopping condition is when the maximum value change across all states is below a small threshold θ (e.g., θ = 10⁻⁶)."
>
> URL: http://incompleteideas.net/book/RLbook2020.pdf

**Sensitivity Analysis:**

| θ Value | Result |
|---------|--------|
| 1e-4 | Same policy, faster convergence |
| 1e-6 | Current setting, stable and reliable |
| 1e-8 | Same policy, more iterations but identical result |

**Conclusion:** The choice of θ = 1e-6 does not affect the final policy because Q-value differences in this model are on the order of 0.001-0.2 QALY, which is 10³ to 10⁵ times larger than the threshold.

---

## 4. Expected Outputs

### 4.1 MDP Optimal vs USPSTF 2024 Comparison

**Comparison Table (Healthy State):**

| Risk | Age | MDP Optimal | USPSTF 2024 | Q-Diff (Days) | Match |
|------|-----|-------------|-------------|---------------|-------|
| High | 30-39 | Screen | Screen | +31.3 | ✓ |
| High | 40+ | Screen | Screen | +52.9-80.1 | ✓ |
| Medium | 30-39 | Screen | **Wait** | +1.8 | ✗ |
| Medium | 40+ | Screen | Screen | +7.2-8.4 | ✓ |
| Low | 30-39 | Screen | **Wait** | +0.5 | ✗ |
| Low | 40+ | Screen | Screen | +3.3-4.0 | ✓ |

**Agreement: 83% (10/12)**

**Final Clinical Recommendations (USPSTF-Aligned):**

| Risk | 30-39 | 40-49 | 50-59 | 60+ |
|------|-------|-------|-------|-----|
| High | **Screen** | **Screen** | **Screen** | **Screen** |
| Medium | **Wait** | **Screen** | **Screen** | **Screen** |
| Low | **Wait** | **Screen** | **Screen** | **Screen** |

**Key Findings:**
- **MDP mathematical optimal**: Screen for ALL states
- **USPSTF-aligned policy**: Wait for Low/Medium risk age 30-39
- **Disagreement reason**: Q-value difference is marginal (0.5-1.8 days)
- **High risk**: Screen at all ages (consistent with NCCN)
- **Cancer survivors**: Continue biennial screening

### 4.2 Value Function V*(s)

Expected cumulative QALY for each state under optimal policy.

---

## 5. Parameter Summary

### Complete Parameter List

```
Biennial Model Parameters:
├── Time step                  : 2 years (biennial)
├── Incidence rate matrix      : 3×4 = 12 values (2-year probability)
├── Screening sensitivity      : Age-stratified (75-87%)
├── Screening specificity      : Age-stratified (80-92%)
├── Disease progression rates  : 4 values (2-year probability)
├── Natural mortality rates    : 4 values (2-year probability)
├── QALY reward values         : 5 values (2-year accumulation)
├── Screening cost             : -0.000115 QALY per screen
├── False positive cost        : 1 value
└── Discount factor            : 1 value (0.97)

Total: ~30 parameters
```

### Data Sources

All model parameters are derived from peer-reviewed literature and official cancer statistics databases.

> **See [DATA_SOURCES.md](docs/DATA_SOURCES.md) for complete documentation of all parameter sources with full citations and calculations.**

#### 5.1 Cancer Incidence Rates

| Parameter | Value | Source | Link |
|-----------|-------|--------|------|
| General Population Incidence (Low Risk) | See Table 1 | SEER Cancer Statistics 2018-2022 | [SEER*Explorer](https://seer.cancer.gov/statistics-network/explorer/) |
| BRCA Relative Risk (High Risk) | 12.5x - 24.5x | Antoniou et al., 2003 (Table 3) | [PMC1180265](https://pmc.ncbi.nlm.nih.gov/articles/PMC1180265/) |
| Family History Risk (Medium Risk) | 2.0x | Brewer et al., 2017 (Table 2) | [PMC5511313](https://pmc.ncbi.nlm.nih.gov/articles/PMC5511313/) |

**BRCA Relative Risk by Age (Table 3 from Antoniou et al., 2003):**

| Age Group | BRCA1 RR | BRCA2 RR | Average Used |
|-----------|----------|----------|--------------|
| 30-39 | 33 | 16 | 24.5 |
| 40-49 | 32 | 9.9 | 21.0 |
| 50-59 | 18 | 12 | 15.0 |
| 60-69 | 14 | 11 | 12.5 |

**Family History Hazard Ratio (Table 2 from Brewer et al., 2017):**

| Family History Score | HR | 95% CI |
|---------------------|-----|--------|
| 0 (No history) | 1.00 | baseline |
| 20-<50 (Typical 1 FDR) | 1.72 | 1.39-2.12 |
| 50-<100 (Strong) | 2.11 | 1.52-2.92 |

> **Note:** For this model, Medium risk uses HR = 2.0 (typical for 1 first-degree relative)

#### 5.2 Screening Performance (Age-Stratified)

| Age Group | Sensitivity | Specificity | Source | Link |
|-----------|-------------|-------------|--------|------|
| <50 years | 75-78% | 80-85% | PMC5638217 - "sensitivity about 75% for women aged under 50" | [PMC5638217](https://pmc.ncbi.nlm.nih.gov/articles/PMC5638217/) |
| 50+ years | 85-87% | 90-92% | PMC5638217 - "sensitivity about 85% for women aged over 50" | [PMC5638217](https://pmc.ncbi.nlm.nih.gov/articles/PMC5638217/) |
| 60-70 years | 87% | 92% | Kerlikowske et al. - "87% in women aged 60-70 years" | [PubMed 8667536](https://pubmed.ncbi.nlm.nih.gov/8667536/) |

#### 5.2.1 Screening Disutility (QALY Costs)

| Parameter | Value | Source | Link |
|-----------|-------|--------|------|
| Screening attendance | -0.000115 QALY | PMC4894487 (CISNET): "0.006 disutility × 1 week / 52 weeks" | [PMC4894487](https://pmc.ncbi.nlm.nih.gov/articles/PMC4894487/) |
| False positive workup | -0.010 QALY | PMC4894487 (CISNET): "0.105 disutility × 5 weeks" | [PMC4894487](https://pmc.ncbi.nlm.nih.gov/articles/PMC4894487/) |

> **Validation:** PMC6655768 systematic review confirms "QALYs loss around 0–0.0013" for screening attendance.

#### 5.3 Disease Progression Probabilities

| Parameter | Value | Source | Link |
|-----------|-------|--------|------|
| Early-Undetected → Advanced | 27%/year | Wu et al., 2022 - Stage II→III median = 2.25 years | [PMC9011255](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011255/) |
| Early-Detected → Cured | 90% | Marcadis et al., 2022: Stage I 10-year DSS = 95.9% | [PMC10314986](https://pmc.ncbi.nlm.nih.gov/articles/PMC10314986/) |
| Early-Detected → Advanced | 10% | 1 - Cure rate (treatment failure) | [Calculated] |
| Cured → Recurrence | 1.6%/year | Pedersen et al., 2021 - Late recurrence 15.53 per 1000 person-years | [PMC8902439](https://pmc.ncbi.nlm.nih.gov/articles/PMC8902439/) |
| Advanced → Dead | 20.1%/year | SEER distant 5-year survival = 32.6% → annual = 20.1% | [SEER Survival](https://seer.cancer.gov/statfacts/html/breast.html) |

#### 5.4 Natural Mortality Rates

| Age Group | Annual Rate | Source | Link |
|-----------|-------------|--------|------|
| 30-39 | 0.131% | US Social Security Administration Actuarial Life Tables (2022) | [SSA Table 4C6](https://www.ssa.gov/oact/STATS/table4c6.html) |
| 40-49 | 0.237% | US SSA Life Tables - Female q(x) averaged over decade | [SSA Table 4C6](https://www.ssa.gov/oact/STATS/table4c6.html) |
| 50-59 | 0.499% | US SSA Life Tables - Female mortality rates | [SSA Table 4C6](https://www.ssa.gov/oact/STATS/table4c6.html) |
| 60+ | 2.781% | US SSA Life Tables - Average ages 60-85 | [SSA Table 4C6](https://www.ssa.gov/oact/STATS/table4c6.html) |

#### 5.5 QALY Reward Values (Health Utility Values)

| Parameter | Value | Source | Link |
|-----------|-------|--------|------|
| Healthy | 1.00 QALY/year | General population baseline | [Standard] |
| Cured | 0.88 QALY/year | Kaur et al., 2022 - Post-treatment survivors (range: 0.71-0.95) | [PMC9189726](https://pmc.ncbi.nlm.nih.gov/articles/PMC9189726/) |
| Early-Detected (in treatment) | 0.71 QALY/year | Kaur et al., 2022 - During active treatment (range: 0.52-0.80) | [PMC9189726](https://pmc.ncbi.nlm.nih.gov/articles/PMC9189726/) |
| Advanced Cancer | 0.45 QALY/year | Kaur et al., 2022 - Metastatic/advanced (range: 0.08-0.82) | [PMC9189726](https://pmc.ncbi.nlm.nih.gov/articles/PMC9189726/) |
| Dead | 0.00 QALY | Terminal state | [Standard] |

> **Note:** Health Utility Values from systematic review of 69 studies using EQ-5D and TTO instruments. Values represent weighted means from reported ranges.

#### 5.6 Screening Costs (QALY Disutility)

| Parameter | Value | Source | Link |
|-----------|-------|--------|------|
| Screening Cost | -0.000115 QALY | PMC4894487 (CISNET): 0.006 × 1/52 | [PMC4894487](https://pmc.ncbi.nlm.nih.gov/articles/PMC4894487/) |
| False Positive Cost | -0.010 QALY | PMC4894487 (CISNET): 0.105 × 5/52 | [PMC4894487](https://pmc.ncbi.nlm.nih.gov/articles/PMC4894487/) |

#### 5.7 Discount Factor

| Parameter | Value | Source | Link |
|-----------|-------|--------|------|
| Discount Factor (γ) | 0.97 | Sanders et al., JAMA 2016 - "Second Panel on Cost-Effectiveness in Health and Medicine" | [JAMA](https://jamanetwork.com/journals/jama/fullarticle/2552214) |

> **Calculation:** γ = 1/(1+r) where r = 0.03 (3% annual discount rate recommended by US Panel on Cost-Effectiveness)

---

## 6. Use Cases

This model can answer questions such as:

1. **When should BRCA carriers start screening?**
   → Check π*(High, age, Healthy) for youngest age with Screen

2. **Should low-risk women screen before 50?**
   → Compare π*(Low, 30-39, Healthy) and π*(Low, 40-49, Healthy)

3. **Should cancer survivors continue screening?**
   → Check π*(*, *, Cured)

4. **What is the expected lifetime QALY difference between strategies?**
   → Compare V*(s) under different policies

---

## 7. Project Structure

```
week_01/
├── README.md                       # This documentation
├── data/
│   ├── explorer_download.csv       # Raw SEER incidence data (auto-loaded)
│   ├── ssa_life_table_female_2022.csv  # SSA mortality data (auto-loaded)
│   └── seer_survival_by_stage.csv  # SEER survival data (auto-loaded)
├── docs/
│   ├── ARCHITECTURE.md             # Code architecture documentation
│   ├── DATA_SOURCES.md             # Complete parameter sources documentation
│   ├── MODEL_DESIGN.md             # MDP model design introduction
│   └── REPORT.md                   # Project report
├── src/
│   ├── main.py                     # Entry point
│   ├── config/
│   │   ├── constants.py            # Pure constants
│   │   └── parameters_biennial.py  # Computed biennial parameters
│   ├── data/
│   │   ├── loaders.py              # CSV file readers
│   │   └── processors.py           # Data transformation
│   ├── models/
│   │   ├── state.py                # State data class
│   │   ├── transitions_v2.py       # Transition calculator (age-stratified)
│   │   ├── rewards.py              # Reward calculator
│   │   └── mdp.py                  # Main MDP class
│   ├── algorithms/
│   │   └── policy_iteration.py     # Policy iteration algorithm
│   └── utils/
│       ├── output.py               # Console output formatting
│       └── export.py               # CSV export functions
└── results/
    ├── optimal_policy.csv          # Output: optimal actions for each state
    └── value_function.csv          # Output: expected QALY for each state
```

### Running the Model

```bash
cd src
python main.py
```

---

## 8. References

> See [DATA_SOURCES.md](docs/DATA_SOURCES.md) for complete citations, data sources, calculations, and parameter derivations.

### Key Sources

1. **SEER Cancer Statistics (2018-2022)** — [SEER*Explorer](https://seer.cancer.gov/statistics-network/explorer/)
2. **SEER Stat Facts** — [Female Breast Cancer](https://seer.cancer.gov/statfacts/html/breast.html)
3. **US SSA (2022)** — [Life Table 4C6 (Female)](https://www.ssa.gov/oact/STATS/table4c6.html)
4. **USPSTF (2024)** — JAMA 2024;331(22):1918-1930. [Full Text](https://jamanetwork.com/journals/jama/fullarticle/2818283)
5. **Antoniou et al. (2003)** — BRCA relative risk. [PMC1180265](https://pmc.ncbi.nlm.nih.gov/articles/PMC1180265/)
6. **Brewer et al. (2017)** — Family history HR. [PMC5511313](https://pmc.ncbi.nlm.nih.gov/articles/PMC5511313/)
7. **Devolli-Disha et al. (2009)** — Screening accuracy by age. [PMC5638217](https://pmc.ncbi.nlm.nih.gov/articles/PMC5638217/)
8. **Kerlikowske et al. (1996)** — Screening sensitivity in older women. [PubMed 8667536](https://pubmed.ncbi.nlm.nih.gov/8667536/)
9. **Wu et al. (2022)** — Cancer progression rates. [PMC9011255](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011255/)
10. **Marcadis et al. (2022)** — Early-stage survival. [PMC10314986](https://pmc.ncbi.nlm.nih.gov/articles/PMC10314986/)
11. **Pedersen et al. (2021)** — Recurrence rates. [PMC8902439](https://pmc.ncbi.nlm.nih.gov/articles/PMC8902439/)
12. **Kaur et al. (2022)** — Health utility values. [PMC9189726](https://pmc.ncbi.nlm.nih.gov/articles/PMC9189726/)
13. **Mittmann et al. (2015)** — Screening disutility (CISNET). [PMC4894487](https://pmc.ncbi.nlm.nih.gov/articles/PMC4894487/)
14. **Sanders et al. (2016)** — Discount rate recommendation. [JAMA](https://jamanetwork.com/journals/jama/fullarticle/2552214)
15. **Sutton & Barto (2018)** — *Reinforcement Learning: An Introduction*. [PDF](http://incompleteideas.net/book/RLbook2020.pdf)

---

