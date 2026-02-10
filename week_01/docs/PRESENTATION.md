# Breast Cancer Screening Optimization using Markov Decision Process

## Policy Iteration Approach Aligned with USPSTF 2024 Guidelines

---

# Slide 1: MDP Formulation of the Problem

---

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
> *Source: JAMA 2024;331(22):1918-1930, DOI: 10.1001/jama.2024.5534*

### 1.4 Project Objective

Find the **optimal screening policy** that maximizes expected cumulative **Quality-Adjusted Life Years (QALY)**, aligned with USPSTF 2024 and NCCN guidelines.

---

## 2. MDP Framework Definition

A Markov Decision Process is defined by the 5-tuple:

$$\text{MDP} = (S, A, P, R, \gamma)$$

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

---

## 3. State Space S

**Definition:** $S = (\text{Risk Level}, \text{Age Group}, \text{Health Status})$

### 3.1 Risk Levels (3)

| Level | Description | Relative Risk | Data Source |
|-------|-------------|---------------|-------------|
| **High** | BRCA1/BRCA2 mutation carriers | 12.5× - 24.5× | Antoniou et al., 2003 (PMC1180265) |
| **Medium** | First-degree relative with breast cancer | 2.0× | Dartois et al., 2017 (PMC5511313) |
| **Low** | General population (baseline) | 1.0× | SEER 2018-2022 |

### 3.2 Age Groups (4)

| Group | Age Range | Guideline Recommendation |
|-------|-----------|--------------------------|
| 1 | 30-39 years | High risk only (NCCN) |
| 2 | 40-49 years | Biennial screening (USPSTF Grade B) |
| 3 | 50-59 years | Biennial screening (USPSTF Grade B) |
| 4 | 60+ years | Biennial screening through 74 (USPSTF) |

### 3.3 Health Status (6)

| Status | Description | Decision Required? |
|--------|-------------|-------------------|
| **Healthy** | No cancer | Yes (Screen/Wait) |
| **Early-Undetected** | Localized cancer, not yet discovered | Yes (Screen/Wait) |
| **Early-Detected** | Localized cancer, in treatment | No (auto-transition) |
| **Cured** | Successfully treated, monitoring | Yes (Screen/Wait) |
| **Advanced** | Metastatic/late-stage cancer | No (auto-transition) |
| **Dead** | Terminal state | No (absorbing state) |

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

---

## 4. Action Space A

$$A = \{\text{Screen}, \text{Wait}\}$$

| Action | Description | When Applicable |
|--------|-------------|-----------------|
| **Screen** | Perform mammography screening | Healthy, Early-Undetected, Cured |
| **Wait** | Defer screening to next 2-year epoch | Healthy, Early-Undetected, Cured |

---

## 5. Transition Probabilities P(s'|s,a)

### 5.1 Data Sources

| Parameter | Data Source | Reference |
|-----------|-------------|-----------|
| Cancer incidence | SEER Database 2018-2022 | seer.cancer.gov |
| BRCA relative risk | Antoniou et al., 2003 | PMC1180265 |
| Family history HR | Dartois et al., 2017 | PMC5511313 |
| Natural mortality | SSA Life Table 2022 | ssa.gov |
| Screening performance | PMC5638217, PubMed 8667536 | Meta-analysis |
| Disease progression | PMC9011255, PMC10314986, PMC8902439 | Clinical studies |

### 5.2 Cancer Incidence Rates (2-Year)

**Conversion formula:** $P_{\text{2-year}} = 1 - (1 - P_{\text{annual}})^2$

| Risk Level | 30-39 | 40-49 | 50-59 | 60+ |
|------------|-------|-------|-------|-----|
| **High (BRCA)** | 2.40% | 6.97% | 7.56% | 10.56% |
| **Medium (Family)** | 0.20% | 0.68% | 1.04% | 1.73% |
| **Low (General)** | 0.10% | 0.34% | 0.52% | 0.86% |

### 5.3 Screening Performance (Age-Stratified)

*Source: PMC5638217*

| Age Group | Sensitivity | Specificity |
|-----------|-------------|-------------|
| 30-39 | 75% | 80% |
| 40-49 | 78% | 85% |
| 50-59 | 85% | 90% |
| 60+ | 87% | 92% |

### 5.4 Disease Progression Rates

| Transition | Annual Rate | 2-Year Rate | Source |
|------------|-------------|-------------|--------|
| Early-Undetected → Advanced | 27% | 46.7% | PMC9011255 |
| Early-Detected → Cured | 90% | 90% | PMC10314986 |
| Cured → Recurrence | 1.6% | 3.2% | PMC8902439 |
| Advanced → Dead | 20.1% | 46.4% | SEER |

### 5.5 Natural Mortality Rates

*Source: SSA Life Table 2022 (Female)*

| Age Group | 2-Year Rate |
|-----------|-------------|
| 30-39 | 0.28% |
| 40-49 | 0.42% |
| 50-59 | 0.90% |
| 60+ | 4.88% |

---

## 6. Reward Function R(s,a)

### 6.1 Health State Utilities (QALY per 2 years)

*Source: PMC9189726 (Shafie et al., 2019)*

| Health State | 2-Year QALY |
|--------------|-------------|
| **Healthy** | 2.00 |
| **Early-Undetected** | 2.00 |
| **Early-Detected** | 1.42 |
| **Cured** | 1.76 |
| **Advanced** | 0.90 |
| **Dead** | 0.00 |

### 6.2 Screening Costs (QALY)

*Source: PMC4894487 (CISNET Model)*

| Event | QALY Cost |
|-------|-----------|
| Screening procedure | -0.000115 |
| False positive result | -0.010 |

---

## 7. Discount Factor γ

*Source: Sanders et al., JAMA 2016*

$$\gamma_{\text{annual}} = \frac{1}{1.03} = 0.9709$$

$$\gamma_{\text{biennial}} = 0.9709^2 = 0.9426$$

---

# Slide 2: Experimental Results from Policy Iteration

---

## 1. Policy Iteration Algorithm

### 1.1 Bellman Equations

**Value Function:**
$$V^\pi(s) = R(s, \pi(s)) + \gamma \sum_{s'} P(s'|s, \pi(s)) \cdot V^\pi(s')$$

**Q-Function:**
$$Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \cdot V(s')$$

**Optimal Policy:**
$$\pi^*(s) = \arg\max_a Q(s, a)$$

### 1.2 Algorithm

```
POLICY ITERATION:

1. Initialize π(s) = "Wait", V(s) = 0

2. Policy Evaluation:
   Repeat until convergence:
     V(s) ← R(s, π(s)) + γ × Σ P(s'|s, π(s)) × V(s')

3. Policy Improvement:
   π(s) ← argmax_a Q(s,a)

4. If policy unchanged → return π*; else → go to step 2
```

---

## 2. Convergence Results

| Metric | Value |
|--------|-------|
| **Total Policy Iterations** | 2 |
| **Iteration 1** | 439 evaluation iterations |
| **Iteration 2** | 371 evaluation iterations |
| **Computation Time** | < 0.7 seconds |
| **Convergence Threshold** | 1×10⁻⁶ |

---

## 3. Optimal Policy π*(s)

### 3.1 Healthy State

| Risk | 30-39 | 40-49 | 50-59 | 60+ |
|------|-------|-------|-------|-----|
| **High** | Screen | Screen | Screen | Screen |
| **Medium** | Screen | Screen | Screen | Screen |
| **Low** | Screen | Screen | Screen | Screen |

### 3.2 Early-Undetected State

| Risk | 30-39 | 40-49 | 50-59 | 60+ |
|------|-------|-------|-------|-----|
| **All** | Screen | Screen | Screen | Screen |

### 3.3 Cured State

| Risk | 30-39 | 40-49 | 50-59 | 60+ |
|------|-------|-------|-------|-----|
| **High** | Screen | Screen | Screen | Screen |
| **Medium** | Screen | Screen | Screen | Screen |
| **Low** | Screen | Screen | Screen | Screen |

---

## 4. Q-Value Analysis

### 4.1 Q-Values for Healthy State

| Risk | Age | Q(Screen) | Q(Wait) | Diff (QALY) | Diff (Days) | Policy |
|------|-----|-----------|---------|-------------|-------------|--------|
| High | 30-39 | 54.97 | 54.88 | +0.087 | **+31.6** | Screen |
| High | 40-49 | 47.77 | 47.55 | +0.220 | **+80.3** | Screen |
| High | 50-59 | 42.70 | 42.55 | +0.152 | **+55.4** | Screen |
| High | 60+ | 21.78 | 21.64 | +0.145 | **+52.8** | Screen |
| Medium | 30-39 | 62.01 | 62.01 | +0.006 | +2.2 | Screen |
| Medium | 40-49 | 56.44 | 56.42 | +0.020 | +7.4 | Screen |
| Medium | 50-59 | 48.78 | 48.76 | +0.020 | +7.2 | Screen |
| Medium | 60+ | 23.52 | 23.50 | +0.023 | +8.3 | Screen |
| Low | 30-39 | 62.54 | 62.54 | +0.003 | +0.9 | Screen |
| Low | 40-49 | 57.72 | 57.71 | +0.010 | +3.5 | Screen |
| Low | 50-59 | 49.99 | 49.98 | +0.009 | +3.4 | Screen |
| Low | 60+ | 23.86 | 23.85 | +0.011 | +4.0 | Screen |

### 4.2 Policy Rationale

- **High Risk 30-39**: Q-diff = +31.6 days → **Screen** (strong benefit)
- **Medium Risk 30-39**: Q-diff = +2.2 days → **Screen** (marginal, within uncertainty)
- **Low Risk 30-39**: Q-diff = +0.9 days → **Screen** (negligible benefit)
- **All 40+**: Q-diff = +3 to +80 days → **Screen** (consistent with USPSTF)

---

## 5. Value Function V*(s)

### 5.1 Expected Lifetime QALY

**Healthy State:**

| Risk | 30-39 | 40-49 | 50-59 | 60+ |
|------|-------|-------|-------|-----|
| High | 54.97 | 47.77 | 42.70 | 21.78 |
| Medium | 62.01 | 56.44 | 48.78 | 23.52 |
| Low | 62.54 | 57.72 | 49.99 | 23.86 |

**Interpretation:**
- V decreases with age (fewer remaining life-years)
- V decreases with risk (higher cancer probability)
- High risk has lowest V despite screening (elevated baseline risk)

---

## 6. Clinical Guidelines Alignment

### 6.1 Policy Based On

**USPSTF 2024** (JAMA 2024;331(22):1918-1930):
> "The USPSTF recommends biennial screening mammography for women aged 40 to 74 years." (Grade B)

**NCCN Guidelines** (High Risk):
> "For BRCA1/2 carriers: Annual mammography and breast MRI starting at age 25-30."

### 6.2 Alignment Summary

| Guideline | Age Group | Risk | Recommendation | MDP Policy |
|-----------|-----------|------|----------------|------------|
| NCCN | 30-39 | High | Screen | Screen ✓ |
| USPSTF | 30-39 | Medium/Low | No recommendation | Screen |
| USPSTF | 40-74 | All | Biennial Screen | Screen ✓ |

---

## 7. Key Findings

### 7.1 Summary Statistics

| Metric | Value |
|--------|-------|
| Total decision states | 36 |
| Screen recommendations | 36 |
| Wait recommendations | 0 |

### 7.2 Clinical Implications

| Risk Level | Ages 30-39 | Ages 40+ |
|------------|------------|----------|
| **High (BRCA)** | Screen (+31.6-80.3 days benefit) | Screen |
| **Medium (Family)** | Screen (+2.2 days, marginal) | Screen |
| **Low (General)** | Screen (+0.9 days, negligible) | Screen |

### 7.3 Conclusions

1. **High Risk**: Screen at ALL ages (consistent with NCCN)
2. **Medium/Low Risk, 40+**: Screen biennially (consistent with USPSTF 2024)
3. **Medium/Low Risk, 30-39**: Screen (benefit +0.9-2.2 days, marginal but positive)
4. **Cancer survivors**: Follow same guidelines for recurrence monitoring

---

## 8. Model Limitations

| Limitation | Impact |
|------------|--------|
| No age progression | Underestimates long-term dynamics |
| Simplified risk (3 levels) | Cannot capture continuous risk |
| US data only | May not generalize internationally |
| Factors not modeled | Psychological harm, overdiagnosis |

---

## References

1. **USPSTF (2024)** - JAMA 2024;331(22):1918-1930
2. **Sanders et al. (2016)** - JAMA (Discount rate)
3. **Antoniou et al. (2003)** - PMC1180265 (BRCA risk)
4. **Dartois et al. (2017)** - PMC5511313 (Family history)
5. **Shafie et al. (2019)** - PMC9189726 (Health utilities)
6. **Mittmann et al. (2015)** - PMC4894487 (Screening costs)
7. **SEER Cancer Statistics** - seer.cancer.gov
8. **SSA Life Tables (2022)** - ssa.gov

---

*Model: Python Policy Iteration | Data: SEER, SSA, CISNET | Guidelines: USPSTF 2024, NCCN*
