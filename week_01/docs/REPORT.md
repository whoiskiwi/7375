# Breast Cancer Screening Optimization using Markov Decision Process

## Policy Iteration Approach Aligned with USPSTF 2024 Guidelines

---

# Part 1: MDP Formulation of the Problem

## 1. Background & Clinical Significance

### 1.1 The Problem

**Breast cancer** is the most common cancer in women worldwide:
- **1 in 8 women** (12.5%) will develop breast cancer in their lifetime
- **~310,000 new cases** diagnosed annually in the U.S.
- Early-stage 5-year survival: **99%** vs. late-stage: **32.6%**

### 1.2 The Screening Dilemma

| Benefits | Harms |
|----------|-------|
| Early detection improves survival | False positives cause anxiety |
| Lower treatment costs for early-stage | Unnecessary biopsies |
| Better treatment options | Radiation exposure |

**Core Question:** *Should a woman undergo breast cancer screening in this 2-year period?*

### 1.3 USPSTF 2024 Guidelines

> "The USPSTF recommends **biennial** screening mammography for women aged 40 to 74 years." (Grade B)
>
> *Source: [JAMA 2024;331(22):1918-1930](https://jamanetwork.com/journals/jama/fullarticle/2818283)*

### 1.4 Project Objective

Find the **optimal screening policy** that maximizes expected cumulative **Quality-Adjusted Life Years (QALY)**, aligned with USPSTF 2024 and NCCN guidelines.

## 2. What is an MDP?

A **Markov Decision Process (MDP)** is a mathematical framework for sequential decision-making under uncertainty. It is defined by a 5-tuple **(S, A, P, R, γ)**:

| Symbol | Name | Meaning |
|--------|------|---------|
| **S** | State Space | All possible situations the patient can be in |
| **A** | Action Space | Decisions available at each state |
| **P** | Transition Probabilities | P(s'\|s,a) = probability of moving to state s' given current state s and action a |
| **R** | Reward Function | R(s,a) = immediate reward (in QALY) for taking action a in state s |
| **γ** | Discount Factor | Weight on future rewards (0 < γ < 1); closer to 1 = value future more |

```
┌─────────────────────────────────────────────────────────────┐
│              BIENNIAL MDP Model Structure                   │
├─────────────────────────────────────────────────────────────┤
│  Decision Epoch = 2 years (biennial screening)              │
│                                                             │
│  S = State Space = (Risk Level, Age Group, Health Status)   │
│      3 × 4 × 6 = 72 states                                  │
│                                                             │
│  A = Action Space = {Screen, Wait}                          │
│                                                             │
│  P = Transition Probabilities (2-year epoch)                │
│      P(2yr) = 1 - (1 - P(annual))²                          │
│                                                             │
│  R = Reward Function (QALY accumulated over 2 years)        │
│                                                             │
│  γ = 0.97² ≈ 0.94 (Discount Factor for 2-year epoch)        │
└─────────────────────────────────────────────────────────────┘
```

The goal is to find a **policy** π*(s) — a mapping from every state to the best action — that maximizes:

$$V^\pi(s) = E\left[\sum_{t=0}^{\infty} \gamma^t \cdot R(s_t, a_t)\right]$$

This is the **value function**: the expected total discounted QALY starting from state s.

## 3. State Space S — "Where is the patient now?"

**S = (Risk Level, Age Group, Health Status)** — Total: 3 × 4 × 6 = **72 states**

### Risk Levels (3)

| Level | Who | How It's Calculated |
|-------|-----|---------------------|
| **High** | BRCA1/BRCA2 mutation carriers | General population rate × BRCA relative risk (12.5-24.5x by age) |
| **Medium** | First-degree relative with breast cancer | General population rate × 2.0 (hazard ratio) |
| **Low** | General population (no special risk) | Baseline from SEER incidence data |

> Sources: [Antoniou et al., 2003, Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC1180265/) (BRCA RR), [Brewer et al., 2017, Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5511313/) (Family HR=2.0)

### Age Groups (4)

| Group | Range | Why This Grouping |
|-------|-------|-------------------|
| 1 | 30-39 | Pre-screening age (USPSTF does not recommend for general population) |
| 2 | 40-49 | USPSTF 2024 newly recommends starting at 40 |
| 3 | 50-59 | Standard screening age |
| 4 | 60+ | Continued screening through age 74 |

> Source: [USPSTF 2024](https://jamanetwork.com/journals/jama/fullarticle/2818283)

### Health Status (6)

| Status | Description | Decision Required? |
|--------|-------------|--------------------|
| **Healthy** | No cancer | Yes — should we screen? |
| **Early-Undetected** | Has early-stage cancer but doesn't know | Yes — should we screen? (patient is unaware) |
| **Early-Detected** | Cancer found, undergoing treatment | No — automatic transition to Cured or Advanced |
| **Cured** | Treatment successful, monitoring for recurrence | Yes — should we continue screening? |
| **Advanced** | Late-stage/metastatic cancer | No — automatic transition (survival or death) |
| **Dead** | Terminal absorbing state | No — stays here forever, V(Dead) = 0 |

### State Transition Diagram

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
         │         │                  (75-87%) │  + Progress  │
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

## 4. Action Space A — "What can we do?"

$$A = \{\text{Screen}, \text{Wait}\}$$

| Action | What Happens | When Applicable |
|--------|-------------|-----------------|
| **Screen** | Perform mammography. If cancer exists, detected with probability = sensitivity; if no cancer, false positive with probability = 1 - specificity | Healthy, Early-Undetected, Cured |
| **Wait** | Skip screening this 2-year period. If cancer develops, it remains undetected | Healthy, Early-Undetected, Cured |

**Non-decision states** (Early-Detected, Advanced, Dead) have automatic transitions — no action choice.

## 5. Transition Probabilities P(s'|s,a) — "What happens next?"

### 5.1 Complete Transition Logic

**From Healthy:**
```
Screen:
  ├── Develops cancer (p) × Detected (sensitivity)     → Early-Detected
  ├── Develops cancer (p) × Missed (1 - sensitivity)   → Early-Undetected
  ├── No cancer (1-p) × Survives (1 - mortality)       → Healthy
  └── Natural death (mortality)                          → Dead
       (False positive cost applied if no cancer & FP occurs)

Wait:
  ├── Develops cancer (p) × Survives                    → Early-Undetected
  ├── No cancer (1-p) × Survives (1 - mortality)       → Healthy
  └── Natural death (mortality)                          → Dead
```

**From Early-Undetected:**
```
Screen:
  ├── Detected (sensitivity) × Survives                → Early-Detected
  ├── Missed × Progresses (27%/yr)                     → Advanced
  ├── Missed × Not progressed                          → Early-Undetected
  └── Natural death                                     → Dead

Wait:
  ├── Progresses (27%/yr) × Survives                   → Advanced
  ├── Not progressed × Survives                        → Early-Undetected
  └── Natural death                                     → Dead
```

**From Early-Detected (automatic, no action):**
```
  ├── Treatment success (90%)                          → Cured
  ├── Treatment failure (10%)                          → Advanced
  └── Natural death                                     → Dead
```

**From Cured:**
```
Screen:
  ├── Recurrence (1.6%/yr) × Detected (sensitivity)   → Early-Detected
  ├── Recurrence × Missed                              → Early-Undetected
  ├── No recurrence × Survives                         → Cured
  └── Natural death                                     → Dead

Wait:
  ├── Recurrence (1.6%/yr) × Survives                 → Early-Undetected
  ├── No recurrence × Survives                         → Cured
  └── Natural death                                     → Dead
```

**From Advanced (automatic, no action):**
```
  ├── Cancer death (20.1%/yr) + Natural death          → Dead
  └── Survives                                          → Advanced
```

**From Dead:** stays Dead (absorbing state, probability = 1.0).

### 5.2 Complete Parameter List

**All 30 parameters used in the model:**

#### Cancer Incidence Rates (2-Year, 12 values)

Calculation: P(2yr) = 1 - (1 - P(annual))²

| Risk \ Age | 30-39 | 40-49 | 50-59 | 60+ |
|------------|-------|-------|-------|-----|
| **High** | 2.40% | 7.07% | 7.58% | 10.56% |
| **Medium** | 0.20% | 0.69% | 1.03% | 1.73% |
| **Low** | 0.10% | 0.34% | 0.52% | 0.87% |

Calculation example (Low, 40-49):
```
SEER annual incidence for ages 40-44 and 45-49 → average ≈ 172 per 100,000
Annual probability = 172 / 100,000 = 0.00172
2-year probability = 1 - (1 - 0.00172)² = 0.0034 = 0.34%

Medium = 0.34% × 2.0 (family HR) = 0.68%
High   = 0.34% × 21.0 (avg BRCA1 RR=32 + BRCA2 RR=9.9)/2 = 6.97%
```

> Sources: [SEER 2018-2022](https://seer.cancer.gov/statistics-network/explorer/) (Low), [Brewer et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5511313/) (Medium HR=2.0), [Antoniou et al., 2003, Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC1180265/) (High BRCA RR)

#### Screening Performance (Age-Stratified, 8 values)

| Age | Sensitivity | Specificity | Why Different |
|-----|-------------|-------------|---------------|
| 30-39 | 75% | 80% | Dense breast tissue reduces accuracy |
| 40-49 | 78% | 85% | Transitional density |
| 50-59 | 85% | 90% | Post-menopausal, less dense |
| 60+ | 87% | 92% | Lowest density, highest accuracy |

> Sources: [PMC5638217](https://pmc.ncbi.nlm.nih.gov/articles/PMC5638217/) — "sensitivity about 75% for women under 50, about 85% for women over 50"; [Kerlikowske et al., 1996](https://pubmed.ncbi.nlm.nih.gov/8667536/) — "87% in women aged 60-70"

#### Disease Progression (4 values)

| Transition | Annual | 2-Year | Calculation | Source |
|------------|--------|--------|-------------|--------|
| Early-Undetected → Advanced | 27% | 46.7% | 1-(1-0.27)² | [Wu et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011255/) — Stage II→III median 2.25yr |
| Early-Detected → Cured | 90% | 90% | Treatment outcome, not time-dependent | [Marcadis et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC10314986/), [CDC/OWH](https://womenshealth.gov/blog/99-percent-survival-rate-breast-cancer-caught-early) |
| Cured → Recurrence | 1.6% | 3.2% | 1-(1-0.016)² | [Pedersen et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8902439/) — 15.53 per 1000 person-years |
| Advanced → Dead | 20.1% | 36.1% | From 5yr survival=32.6% | [SEER Stat Facts](https://seer.cancer.gov/statfacts/html/breast.html) |

Advanced → Dead calculation:
```
5-year survival (Distant) = 32.6%
Annual survival = 0.326^(1/5) = 0.799
Annual death rate = 1 - 0.799 = 0.201 = 20.1%
2-year death rate = 1 - (1 - 0.201)² = 0.361 = 36.1%
```

#### Natural Mortality (Age-Stratified, 4 values)

| Age Group | Annual Rate | 2-Year Rate | Source Data |
|-----------|-------------|-------------|-------------|
| 30-39 | 0.131% | 0.26% | Average of SSA q(x) for ages 30-39 |
| 40-49 | 0.237% | 0.47% | Average of SSA q(x) for ages 40-49 |
| 50-59 | 0.499% | 1.00% | Average of SSA q(x) for ages 50-59 |
| 60+ | 2.781% | 5.49% | Average of SSA q(x) for ages 60-85 |

> Source: [SSA Actuarial Life Table 2022, Table 4C6 (Female)](https://www.ssa.gov/oact/STATS/table4c6.html)

#### Discount Factor (1 value)

```
Annual discount rate r = 3% (recommended by US Panel on Cost-Effectiveness)
γ_annual = 1 / (1 + 0.03) = 0.9709
γ_biennial = 0.9709² = 0.9426 (adjusted for 2-year decision epochs)
```

> Source: [Sanders et al., JAMA 2016](https://jamanetwork.com/journals/jama/fullarticle/2552214)

**Total: ~30 parameters** (12 incidence + 8 screening + 4 progression + 4 mortality + 1 discount + screening/FP costs)

## 6. Reward Function R(s,a) — "How good is this outcome?"

Rewards are measured in **QALY (Quality-Adjusted Life Years)**: 1 QALY = 1 year of perfect health.

### State Rewards (per 2-year epoch)

| Health State | QALY/year | × 2 years | Meaning | Source |
|-------------|-----------|-----------|---------|--------|
| Healthy | 1.00 | **2.00** | Full quality of life | Baseline |
| Early-Undetected | 1.00 | **2.00** | Unaware of cancer, same QoL as healthy | Baseline |
| Cured | 0.88 | **1.76** | Post-treatment; some ongoing effects | [Kaur et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9189726/) |
| Early-Detected | 0.71 | **1.42** | Active treatment: surgery, chemo, radiation | [Kaur et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9189726/) |
| Advanced | 0.45 | **0.90** | Metastatic cancer, severe quality reduction | [Kaur et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9189726/) |
| Dead | 0.00 | **0.00** | Terminal state | — |

### Action Costs (per screening event)

| Cost | QALY | Meaning | Source |
|------|------|---------|--------|
| Screening | **-0.000115** | Anxiety, time off work, waiting for results (~0.04 days) | [Mittmann et al., 2015 (CISNET)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4894487/) |
| False Positive | **-0.010** | Additional imaging, biopsies, weeks of anxiety (~3.7 days) | [Mittmann et al., 2015 (CISNET)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4894487/) |

> Validation: [PMC6655768 systematic review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6655768/) confirms screening QALY loss around 0-0.0013.

### Reward Calculation Example

For state (Low, 40-49, Healthy) with action Screen:
```
R(s, Screen) = State utility + Screening cost + Expected FP cost
             = 2.00 + (-0.000115) + (1 - 0.0034) × 0.15 × (-0.010)
             = 2.00 - 0.000115 - 0.00149
             = 1.99840 QALY

Where:
  2.00     = Healthy utility × 2 years
  -0.000115 = Screening procedure cost
  0.0034   = Cancer incidence (Low, 40-49)
  0.15     = False positive rate (1 - specificity for age 40-49)
  -0.010   = False positive cost
```

## 7. Policy π(s) — "What should we do?"

A **policy** π(s) is a rule that maps every state to an action:

```
π(s) : State → Action

Example:
  π(High, 30-39, Healthy) = Screen    ← "BRCA carrier age 30-39: screen (margin +31.3 days)"
  π(Low,  30-39, Healthy) = Screen    ← "Low risk age 30-39: screen (margin only +0.5 days)"
  π(Any,  Any,   Dead)    = None      ← "Dead: no action possible"
```

The **optimal policy** π*(s) is the one that maximizes V(s) for all states simultaneously. We find it using Policy Iteration.

---

# Part 2: Experimental Results from Policy Iteration

## 1. Algorithm

**Policy Iteration** finds the optimal policy through two alternating steps ([Sutton & Barto, 2018, Ch.4](http://incompleteideas.net/book/RLbook2020.pdf)):

**Value Function:**
$$V^\pi(s) = R(s, \pi(s)) + \gamma \sum_{s'} P(s'|s, \pi(s)) \cdot V^\pi(s')$$

**Q-Function:**
$$Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \cdot V(s')$$

**Optimal Policy:**
$$\pi^*(s) = \arg\max_a Q(s, a)$$

```
Step 1: INITIALIZE
  π(s) = Wait for all decision states (conservative start)
  V(s) = 0 for all states

Step 2: POLICY EVALUATION — "How good is the current policy?"
  Repeat until convergence (Δ < 10⁻⁶):
    For each state s:
      V(s) ← R(s, π(s)) + γ · Σ P(s'|s, π(s)) · V(s')
                ↑               ↑            ↑
          immediate reward   discount    future expected value

Step 3: POLICY IMPROVEMENT — "Can we do better?"
  For each decision state s:
    Q(s, Screen) = R(s, Screen) + γ · Σ P(s'|s, Screen) · V(s')
    Q(s, Wait)   = R(s, Wait)   + γ · Σ P(s'|s, Wait)   · V(s')
    π(s) ← argmax{Q(s, Screen), Q(s, Wait)}

Step 4: CONVERGENCE CHECK
  If policy unchanged → return π* (optimal!)
  Else → go back to Step 2
```

**Q-value** Q(s,a) represents the total expected QALY of taking action a in state s and then following policy π* thereafter. The optimal action is whichever has the higher Q-value.

## 2. Convergence Results

| Metric | Value |
|--------|-------|
| Policy Iterations | **2** (converged very fast) |
| Eval Iterations (Iter 1 / Iter 2) | 439 / 371 |
| Convergence Threshold (θ) | 1×10⁻⁶ |
| Computation Time | **< 0.3 seconds** |
| Total States | 72 |
| Decision States | 36 |

## 3. Optimal Policy π*(s)

### Healthy State — "Should this healthy woman get screened?"

| Risk | 30-39 | 40-49 | 50-59 | 60+ |
|------|-------|-------|-------|-----|
| **High (BRCA)** | **Screen** | **Screen** | **Screen** | **Screen** |
| **Medium (Family)** | **Screen** | **Screen** | **Screen** | **Screen** |
| **Low (General)** | **Screen** | **Screen** | **Screen** | **Screen** |

### Early-Undetected State — "Cancer exists but unknown"

| Risk | 30-39 | 40-49 | 50-59 | 60+ |
|------|-------|-------|-------|-----|
| **All** | **Screen** | **Screen** | **Screen** | **Screen** |

Screening always recommended when cancer is present (even though patient doesn't know).

### Cured State — "Cancer survivor monitoring"

| Risk | 30-39 | 40-49 | 50-59 | 60+ |
|------|-------|-------|-------|-----|
| **High** | **Screen** | **Screen** | **Screen** | **Screen** |
| **Medium** | **Screen** | **Screen** | **Screen** | **Screen** |
| **Low** | **Screen** | **Screen** | **Screen** | **Screen** |

Same pattern as Healthy — recurrence risk (1.6%/yr) warrants monitoring.

**Summary: 36 Screen / 0 Wait out of 36 decision states.**

## 4. Q-Value Analysis — "By how much is Screen better than Wait?"

| Risk | Age | Q(Screen) | Q(Wait) | Diff (QALY) | Diff (Days) | Policy |
|------|-----|-----------|---------|-------------|-------------|--------|
| High | 30-39 | 54.9403 | 54.8547 | +0.0857 | **+31.3** | Screen |
| High | 40-49 | 47.7541 | 47.5346 | +0.2195 | **+80.1** | Screen |
| High | 50-59 | 42.7013 | 42.5494 | +0.1519 | **+55.4** | Screen |
| High | 60+ | 21.7858 | 21.6410 | +0.1448 | **+52.9** | Screen |
| Medium | 30-39 | 61.9819 | 61.9769 | +0.0051 | +1.8 | Screen |
| Medium | 40-49 | 56.4239 | 56.4041 | +0.0198 | +7.2 | Screen |
| Medium | 50-59 | 48.7793 | 48.7596 | +0.0196 | +7.2 | Screen |
| Medium | 60+ | 23.5253 | 23.5024 | +0.0229 | +8.4 | Screen |
| Low | 30-39 | 62.5085 | 62.5070 | +0.0015 | +0.5 | Screen |
| Low | 40-49 | 57.7071 | 57.6980 | +0.0091 | +3.3 | Screen |
| Low | 50-59 | 49.9939 | 49.9846 | +0.0093 | +3.4 | Screen |
| Low | 60+ | 23.8627 | 23.8517 | +0.0110 | +4.0 | Screen |

**How to read:** Q(Screen) - Q(Wait) > 0 means screening adds expected QALY. "Diff (Days)" converts QALY difference to days (× 365) for clinical interpretation.

**Key insight:** High-risk patients gain **31.3-80.1 days** of quality-adjusted life per screening decision. Low-risk 30-39 gains only **0.5 days** — near the decision boundary.

## 5. Value Function V*(s) — "Expected lifetime QALY"

**Healthy State:**

| Risk | 30-39 | 40-49 | 50-59 | 60+ |
|------|-------|-------|-------|-----|
| High | 54.94 | 47.75 | 42.70 | 21.79 |
| Medium | 61.98 | 56.42 | 48.78 | 23.53 |
| Low | 62.51 | 57.71 | 49.99 | 23.86 |

**How to read:**
- A healthy Low-risk 30-year-old expects **62.51 QALY** remaining lifetime
- A healthy High-risk (BRCA) 30-year-old expects **54.94 QALY** — a loss of **7.6 QALY** (~7.6 years of perfect health) due to elevated cancer risk
- V decreases with age (fewer remaining years) and increases with lower risk

## 6. Why the Optimal Policy Is Universally "Screen"

The model yields Screen for all 36 decision states. This is not a modeling error — it reflects the inherent structure of a **QALY-only cost model**:

1. **Screening costs are extremely small in QALY terms.** The total per-screen cost is approximately 0.002 QALY (screening disutility + expected false positive disutility), equivalent to less than 1 day of quality-adjusted life. This is consistent with published QALY-based screening disutility estimates ([PMC6655768](https://pmc.ncbi.nlm.nih.gov/articles/PMC6655768/)).

2. **Even a small probability of early detection yields a positive net benefit.** Early detection leads to a 90% cure rate (utility 0.88/yr), while missed cancers progress to advanced stage at 27%/yr (utility 0.45/yr with high mortality). The expected QALY gain from catching even one cancer early outweighs the small screening disutility across all risk-age combinations.

3. **The model does not include monetary costs.** Real-world screening decisions also consider financial costs (mammogram ~$250, false positive workup ~$1,000+), over-diagnosis risk, and cumulative radiation exposure. Incorporating these via a willingness-to-pay threshold (e.g., $100,000/QALY) would increase the effective screening cost and likely produce Wait recommendations for low-risk younger women — consistent with the USPSTF 2024 recommendation to begin routine screening at age 40.

4. **The model uses static age groups (no age transitions).** Individuals do not age between decision epochs, so the model cannot evaluate the strategy of "wait now, screen when older." This simplification removes a reason to defer screening for younger women.

**However, the Q-value margins provide meaningful differentiation** even within the uniform Screen policy, revealing which populations benefit most from screening and where the policy is most sensitive to parameter changes.

## 7. MDP vs USPSTF 2024 Comparison

| Risk | Age | MDP Optimal | USPSTF 2024 | Q-Diff (Days) | Agreement |
|------|-----|-------------|-------------|---------------|-----------|
| High | 30-39 | Screen | Screen | +31.3 | ✓ |
| High | 40+ | Screen | Screen | +52.9-80.1 | ✓ |
| Medium | 30-39 | Screen | Wait | +1.8 | ✗ (marginal) |
| Medium | 40+ | Screen | Screen | +7.2-8.4 | ✓ |
| Low | 30-39 | Screen | Wait | +0.5 | ✗ (marginal) |
| Low | 40+ | Screen | Screen | +3.3-4.0 | ✓ |

**Agreement: 83% (10/12).** The 2 disagreements are exactly the cases where Q-value margins are smallest (0.5-1.8 days), indicating that these decisions are near the boundary and would likely flip to Wait if monetary costs or over-diagnosis penalties were included.

## 8. Key Findings

1. **High Risk (BRCA):** Screen at ALL ages with strong confidence — Q-value benefit +31.3 to +80.1 days per decision, consistent with [NCCN guidelines](https://www.nccn.org/guidelines/guidelines-detail?category=2&id=1416) recommending annual screening from age 25-30 for BRCA carriers.
2. **Medium/Low Risk, Ages 40+:** Screen biennially — benefit +3.3 to +8.4 days, consistent with [USPSTF 2024](https://jamanetwork.com/journals/jama/fullarticle/2818283) Grade B recommendation.
3. **Low Risk, Age 30-39:** Marginal benefit of only **0.5 days** — effectively at the decision boundary. The USPSTF does not recommend routine screening for this group, and the near-zero margin supports that position: any additional cost factor (financial, over-diagnosis, radiation) would tip the balance to Wait.
4. **Cancer survivors:** Continue screening at all ages for recurrence monitoring (recurrence rate 1.6%/yr).
5. **Q-value gradient as the key output:** While the binary policy is uniform, the 160× range in screening benefit (0.5 to 80.1 days) across risk-age groups quantifies the clinical priority for screening resource allocation.

**Conclusion:** The MDP model produces a uniform Screen policy under QALY-only costs, but the Q-value analysis reveals a clear gradient that aligns with USPSTF 2024 guidelines at an 83% agreement rate. The two disagreements (Low/Medium risk, age 30-39) occur precisely at the decision boundary where margins are < 2 days, supporting the USPSTF's position that routine screening below age 40 offers minimal net benefit for average-risk women.

## 9. Model Limitations

| Limitation | Impact | Potential Improvement |
|------------|--------|----------------------|
| No monetary costs | Screening cost underestimated; policy biased toward Screen | Include via WTP threshold ($100K/QALY) |
| No age transitions | Cannot model "wait now, screen later" strategies | Add age progression between epochs |
| No over-diagnosis | Missing a key harm of screening in younger women | Add over-diagnosis probability and treatment disutility |
| Simplified risk (3 levels) | Cannot capture continuous risk spectrum | Use polygenic risk scores |
| US data only | May not generalize internationally | Adapt with country-specific incidence/mortality |
| No radiation accumulation | Cumulative screening harm not modeled | Add dose-dependent cancer induction risk |

---

## References

> See [DATA_SOURCES.md](DATA_SOURCES.md) for complete citations, data sources, calculations, and parameter derivations.
