# Breast Cancer Screening MDP - Data Sources

All model parameters are derived from peer-reviewed literature and official cancer statistics databases.

---

## 1. Cancer Incidence Rates

### 1.1 General Population Incidence (Low Risk)

| Item | Detail |
|------|--------|
| **Source** | SEER Cancer Statistics 2018-2022 |
| **Organization** | National Cancer Institute, Surveillance, Epidemiology, and End Results Program |
| **URL** | https://seer.cancer.gov/statistics-network/explorer/ |
| **Data Used** | Age-specific breast cancer incidence rates per 100,000 women |
| **Data File** | `data/explorer_download.csv` (auto-loaded) |

### 1.2 BRCA Relative Risk (High Risk)

| Item | Detail |
|------|--------|
| **Source** | Antoniou A, Pharoah PD, Narod S, et al. (2003) |
| **Title** | "Average Risks of Breast and Ovarian Cancer Associated with BRCA1 or BRCA2 Mutations Detected in Case Series Unselected for Family History: A Combined Analysis of 22 Studies" |
| **Journal** | American Journal of Human Genetics |
| **URL** | https://pmc.ncbi.nlm.nih.gov/articles/PMC1180265/ |
| **Data Used** | Table 3: Age-specific relative risks for BRCA1/BRCA2 carriers |

**BRCA Relative Risk by Age (Table 3):**

| Age Group | BRCA1 RR | BRCA2 RR | Average Used |
|-----------|----------|----------|--------------|
| 30-39 | 33 | 16 | 24.5 |
| 40-49 | 32 | 9.9 | 21.0 |
| 50-59 | 18 | 12 | 15.0 |
| 60+ | 14 | 11 | 12.5 |

**Important: These are MULTIPLIERS, not percentages.**

Relative Risk (RR) represents how many times higher the risk is compared to the general population:

| Value | Meaning |
|-------|---------|
| RR = 33 | BRCA1 carrier's risk is **33 times** higher than non-carrier |
| RR = 16 | BRCA2 carrier's risk is **16 times** higher than non-carrier |

**Calculation Example (Age 30-39):**

```
General population (Low Risk) annual incidence: 0.05%

BRCA1 carrier:  0.05% × 33 = 1.65%/year
BRCA2 carrier:  0.05% × 16 = 0.80%/year

Model uses average:
  0.05% × (33 + 16) / 2 = 0.05% × 24.5 = 1.225%/year
```

**Why use average of BRCA1 and BRCA2?**
- Clinical genetic testing often identifies BRCA1/2 without distinguishing treatment protocols
- Using average provides a single "High Risk" category for model simplicity
- Sensitivity analysis can test BRCA1-only or BRCA2-only scenarios

### 1.3 Family History Hazard Ratio (Medium Risk)

| Item | Detail |
|------|--------|
| **Source** | Brewer HR, Jones ME, Schoemaker MJ, et al. (2017) |
| **Title** | "Family history and risk of breast cancer: an analysis accounting for family structure" |
| **Journal** | Breast Cancer Research and Treatment |
| **URL** | https://pmc.ncbi.nlm.nih.gov/articles/PMC5511313/ |
| **Data Used** | Table 2: Family History Score hazard ratios |

**Family History Hazard Ratio (Table 2):**

| Family History Score | HR | 95% CI |
|---------------------|-----|--------|
| 0 (No history) | 1.00 | baseline |
| <10 | 1.61 | 1.30-2.00 |
| 10-<20 | 1.63 | 1.30-2.05 |
| 20-<50 (Typical 1 FDR) | 1.72 | 1.39-2.12 |
| 50-<100 (Strong) | 2.11 | 1.52-2.92 |
| >=100 | 3.50 | 2.07-5.92 |

> **Model uses:** HR = 2.0 for Medium risk (typical for 1 first-degree relative)

---

## 2. Screening Performance (Age-Stratified)

| Item | Detail |
|------|--------|
| **Source 1** | Devolli-Disha E, Manxhuka-Kerliu S, Ymeri H, et al. (2009) |
| **Title** | "Comparative accuracy of mammography and ultrasound in women with breast symptoms according to age and breast density" |
| **Journal** | Bosnian Journal of Basic Medical Sciences |
| **URL** | https://pmc.ncbi.nlm.nih.gov/articles/PMC5638217/ |
| **Data** | Women <50: sensitivity ~75%, specificity ~80%; Women 50+: sensitivity ~85%, specificity ~90% |

| Item | Detail |
|------|--------|
| **Source 2** | Kerlikowske K, Grady D, Barclay J, et al. (1996) |
| **Title** | "Effect of age, breast density, and family history on the sensitivity of first screening mammography" |
| **Journal** | JAMA |
| **URL** | https://pubmed.ncbi.nlm.nih.gov/8667536/ |
| **Data** | Sensitivity 87% in women aged 60-70 |

**Age-Specific Parameters Used in Model:**

| Age Group | Sensitivity | Specificity | Source |
|-----------|-------------|-------------|--------|
| 30-39 | 75% | 80% | PMC5638217 (dense breasts) |
| 40-49 | 78% | 85% | PMC5638217 (interpolated) |
| 50-59 | 85% | 90% | PMC5638217 |
| 60+ | 87% | 92% | PubMed 8667536 |

**Rationale:** Younger women have denser breast tissue, reducing mammography accuracy. Sensitivity increases with age as breast density decreases.

---

## 3. Disease Progression Rates

### 3.1 Early-Undetected to Advanced (Progression Rate)

| Item | Detail |
|------|--------|
| **Source** | Wu SG, Sun JY, et al. (2022) |
| **Title** | "The natural history of untreated breast cancer: a chronological analysis of breast cancer progression using data from the SEER database" |
| **Journal** | Annals of Translational Medicine |
| **URL** | https://pmc.ncbi.nlm.nih.gov/articles/PMC9011255/ |
| **Study Size** | n=12,687 untreated breast cancer patients |
| **Data Used** | Stage II → Stage III median time = 2.0-2.5 years |

**Calculation:**
```
Median time to progression = 2.25 years
Annual rate = 1 - 0.5^(1/2.25) ≈ 0.27 (27%)
```

### 3.2 Early-Detected Treatment Success Rate

| Item | Detail |
|------|--------|
| **Source** | Marcadis AR, Morris LGT, Marti JL (2022) |
| **Title** | "Relative Survival With Early-Stage Breast Cancer in Screened and Unscreened Populations" |
| **Journal** | Mayo Clinic Proceedings |
| **URL** | https://pmc.ncbi.nlm.nih.gov/articles/PMC10314986/ |
| **Data** | Stage I 10-year disease-specific survival = 95.9% |

> **Model uses:** 90% treatment success rate (Early-Detected → Cured)

### 3.3 Recurrence Rate (Cured to Early-Undetected)

| Item | Detail |
|------|--------|
| **Source** | Pedersen RN, Esen BO, Mellemkjaer L, et al. (2021) |
| **Title** | "The Incidence of Breast Cancer Recurrence 10-32 Years After Primary Diagnosis" |
| **Journal** | JNCI Journal of the National Cancer Institute |
| **URL** | https://pmc.ncbi.nlm.nih.gov/articles/PMC8902439/ |
| **Data Used** | Late recurrence rate: 15.53 per 1,000 person-years |

**Calculation:**
```
Annual recurrence rate = 15.53 / 1000 = 0.01553 ≈ 1.6%
```

### 3.4 Advanced Cancer Mortality Rate

| Item | Detail |
|------|--------|
| **Source** | SEER Cancer Statistics Review |
| **Title** | "Cancer Stat Facts: Female Breast Cancer" |
| **URL** | https://seer.cancer.gov/statfacts/html/breast.html |
| **Data File** | `data/seer_survival_by_stage.csv` (auto-loaded) |
| **Data Used** | Distant stage 5-year relative survival = 32.6% |

**Calculation:**
```
5-year survival = 0.326
Annual survival = 0.326^(1/5) = 0.799
Annual death rate = 1 - 0.799 = 0.201 ≈ 20.1%
```

**SEER 5-Year Survival by Stage:**

| Stage | 5-Year Survival | % of Cases | Description |
|-------|-----------------|------------|-------------|
| Localized | 99.5% | 64% | Confined to Primary Site |
| Regional | 87.2% | 28% | Spread to Regional Lymph Nodes |
| Distant | 32.6% | 6% | Cancer Has Metastasized |
| Unknown | 70.2% | 2% | Unstaged |

---

## 4. Natural Mortality Rates

| Item | Detail |
|------|--------|
| **Source** | US Social Security Administration |
| **Title** | "Actuarial Life Table - Period Life Table, 2022 (Female)" |
| **Report** | 2025 OASDI Trustees Report |
| **URL** | https://www.ssa.gov/oact/STATS/table4c6.html |
| **Data File** | `data/ssa_life_table_female_2022.csv` (auto-loaded) |
| **Data Used** | q(x) = Probability of death within one year at age x |

**Aggregated Mortality by MDP Age Group:**

| Age Group | Annual Death Rate | Calculation |
|-----------|-------------------|-------------|
| 30-39 | 0.131% | Average of SSA q(x) for ages 30-39 |
| 40-49 | 0.237% | Average of SSA q(x) for ages 40-49 |
| 50-59 | 0.499% | Average of SSA q(x) for ages 50-59 |
| 60+ | 2.781% | Average of SSA q(x) for ages 60-85 |

---

## 5. Health Utility Values (QALY)

### 5.1 Breast Cancer Health States

| Item | Detail |
|------|--------|
| **Source** | Kaur MN, Yan J, Klassen AF, et al. (2022) |
| **Title** | "A Systematic Literature Review of Health Utility Values in Breast Cancer" |
| **Journal** | Medical Decision Making |
| **URL** | https://pmc.ncbi.nlm.nih.gov/articles/PMC9189726/ |
| **Study Size** | Systematic review of 69 studies |
| **Instruments** | EQ-5D, TTO, SG |

**Health Utility Values Used:**

| Health State | Value | Range from Review | Rationale |
|--------------|-------|-------------------|-----------|
| Healthy | 1.00 | - | General population baseline |
| Early-Undetected | 1.00 | - | Unaware of cancer, same QoL |
| Early-Detected (in treatment) | 0.71 | 0.52-0.80 | Weighted mean from 69 studies |
| Cured (post-treatment) | 0.88 | 0.71-0.95 | Weighted mean from 69 studies |
| Advanced (metastatic) | 0.45 | 0.08-0.82 | Weighted mean from 69 studies |
| Dead | 0.00 | - | Terminal state |

**Value Selection Methodology:**

Kaur et al. (2022) systematic review of **69 studies** provides two types of estimates:

| Estimate Type | Description |
|---------------|-------------|
| **Range** | Minimum to maximum values across all studies |
| **Weighted Mean** | Sample-size weighted average across studies |

**This model uses Weighted Mean values**, calculated as:

```
Weighted Mean = Σ(Study Value × Sample Size) / Σ(Sample Size)
```

**Rationale for using Weighted Mean:**

1. **Sample-size weighting**: Larger studies have greater influence, reducing bias from small studies
2. **Scientific standard**: Weighted mean is the standard summary statistic in systematic reviews
3. **Conservative approach**: Avoids cherry-picking extreme values from the range

**Sensitivity Analysis Recommendation:**

To test robustness, the model can be run with boundary values:

| Scenario | Early-Detected | Cured | Advanced |
|----------|----------------|-------|----------|
| Optimistic (upper bound) | 0.80 | 0.95 | 0.82 |
| **Baseline (weighted mean)** | **0.71** | **0.88** | **0.45** |
| Pessimistic (lower bound) | 0.52 | 0.71 | 0.08 |

If optimal policy remains consistent across scenarios, results are robust to parameter uncertainty.

### 5.2 Screening Disutility

| Item | Detail |
|------|--------|
| **Source** | Mittmann N, et al. (2015) - CISNET Canadian Model |
| **Title** | "Total cost-effectiveness of mammography screening strategies" |
| **Journal** | Health Reports (Statistics Canada) |
| **URL** | https://pmc.ncbi.nlm.nih.gov/articles/PMC4894487/ |
| **Data Used** | Screening disutility: 0.006 for one week; False positive disutility: 0.105 for five weeks |

**Costs Used:**

| Event | QALY Cost | Calculation | Source |
|-------|-----------|-------------|--------|
| Screening procedure | -0.000115 | 0.006 × 1/52 | PMC4894487 (Mittmann et al., 2015) |
| False positive result | -0.010 | 0.105 × 5/52 | PMC4894487 (Mittmann et al., 2015) |

---

## 6. Discount Factor

| Item | Detail |
|------|--------|
| **Source** | Sanders GD, Neumann PJ, Basu A, et al. (2016) |
| **Title** | "Recommendations for Conduct, Methodological Practices, and Reporting of Cost-effectiveness Analyses: Second Panel on Cost-Effectiveness in Health and Medicine" |
| **Journal** | JAMA |
| **URL** | https://jamanetwork.com/journals/jama/fullarticle/2552214 |
| **Recommendation** | "Both costs and health effects should be discounted at an annual rate of 3%" |

**Calculation:**
```
Discount rate (r) = 3% = 0.03
Discount factor (γ) = 1 / (1 + r) = 1 / 1.03 ≈ 0.97
```

**Supporting Guidelines:**

| Organization | Recommended Rate | Reference |
|--------------|------------------|-----------|
| US Panel on Cost-Effectiveness in Health and Medicine | 3% | Sanders et al., JAMA 2016 |
| WHO-CHOICE | 3% | WHO Guidelines 2003 |
| NICE (UK) | 3.5% | NICE Technology Appraisal Methods Guide |
| Canadian CADTH | 1.5% | CADTH Guidelines 2017 |

**Rationale for 3% Discount Rate:**
- Time preference: People value current health benefits more than future ones
- Opportunity cost: Resources could be invested for returns
- Uncertainty: Future outcomes are uncertain
- Consistency: Enables comparison across different health economic studies

---

## 7. Biennial Screening Model

USPSTF 2024 recommends **biennial** (every 2 years) screening, not annual. The biennial model adjusts parameters accordingly.

### 7.1 USPSTF 2024 Recommendation

| Item | Detail |
|------|--------|
| **Source** | US Preventive Services Task Force (2024) |
| **Title** | "Screening for Breast Cancer: US Preventive Services Task Force Recommendation Statement" |
| **Journal** | JAMA 2024;331(22):1918-1930 |
| **DOI** | 10.1001/jama.2024.5534 |
| **JAMA Full Text** | https://jamanetwork.com/journals/jama/fullarticle/2818283 |
| **Key Quote** | "The USPSTF recommends **biennial** screening mammography for women aged 40 to 74 years (B recommendation)" |
| **Grade** | B (moderate certainty, moderate net benefit) |
| **Update** | The 2024 update lowered the starting age from 50 to 40 |

### 7.2 Probability Conversion

For biennial decision epochs, annual probabilities are converted using:

```
P(2-year) = 1 - (1 - P_annual)^2
```

**Example: Low Risk 40-49**
- Annual incidence: 0.17%
- Biennial incidence: 1 - (1 - 0.0017)² = 0.34%

### 7.3 Discount Factor Adjustment

```
γ_annual = 0.97 (3% annual discount)
γ_biennial = 0.97² ≈ 0.94 (for 2-year epochs)
```

---

## 8. Auto-Loaded Data Files

The model automatically loads real data from CSV files:

| File | Content | Source | Auto-Calculation |
|------|---------|--------|------------------|
| `data/explorer_download.csv` | Age-specific incidence rates | SEER*Explorer | Low risk incidence by age |
| `data/ssa_life_table_female_2022.csv` | Female death probability by age | SSA Life Tables | Natural mortality by age group |
| `data/seer_survival_by_stage.csv` | 5-year survival by stage | SEER Stat Facts | Advanced → Dead rate |

---

## 9. References Summary

### Data Sources
1. **SEER Cancer Statistics (2018-2022)** — Surveillance, Epidemiology, and End Results Program. [SEER*Explorer](https://seer.cancer.gov/statistics-network/explorer/)
2. **SEER Stat Facts** — "Cancer Stat Facts: Female Breast Cancer." [SEER](https://seer.cancer.gov/statfacts/html/breast.html)
3. **US SSA (2022)** — "Actuarial Life Table - Period Life Table, 2022 (Female)." [Table 4C6](https://www.ssa.gov/oact/STATS/table4c6.html)

### Clinical Guidelines
4. **USPSTF (2024)** — "Screening for Breast Cancer." JAMA 2024;331(22):1918-1930. [Full Text](https://jamanetwork.com/journals/jama/fullarticle/2818283)

### Risk Stratification
5. **Antoniou et al. (2003)** — "Average Risks of Breast and Ovarian Cancer Associated with BRCA1 or BRCA2 Mutations." Am J Hum Genet. [PMC1180265](https://pmc.ncbi.nlm.nih.gov/articles/PMC1180265/)
6. **Brewer et al. (2017)** — "Family history and risk of breast cancer: an analysis accounting for family structure." Breast Cancer Res Treat. [PMC5511313](https://pmc.ncbi.nlm.nih.gov/articles/PMC5511313/)

### Screening Performance
7. **Devolli-Disha et al. (2009)** — "Comparative accuracy of mammography and ultrasound in women with breast symptoms according to age and breast density." Bosn J Basic Med Sci. [PMC5638217](https://pmc.ncbi.nlm.nih.gov/articles/PMC5638217/)
8. **Kerlikowske et al. (1996)** — "Effect of age, breast density, and family history on the sensitivity of first screening mammography." JAMA. [PubMed 8667536](https://pubmed.ncbi.nlm.nih.gov/8667536/)

### Disease Progression & Treatment
9. **Wu et al. (2022)** — "The natural history of breast cancer." Ann Transl Med. [PMC9011255](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011255/)
10. **Marcadis et al. (2022)** — "Relative Survival With Early-Stage Breast Cancer in Screened and Unscreened Populations." Mayo Clin Proc. [PMC10314986](https://pmc.ncbi.nlm.nih.gov/articles/PMC10314986/)
11. **Pedersen et al. (2021)** — "The Incidence of Breast Cancer Recurrence 10-32 Years After Primary Diagnosis." JNCI. [PMC8902439](https://pmc.ncbi.nlm.nih.gov/articles/PMC8902439/)

### Health Utilities & Screening Costs
12. **Kaur et al. (2022)** — "A Systematic Literature Review of Health Utility Values in Breast Cancer." Med Decis Making. [PMC9189726](https://pmc.ncbi.nlm.nih.gov/articles/PMC9189726/)
13. **Mittmann et al. (2015)** — "Total cost-effectiveness of mammography screening strategies." Health Reports (CISNET). [PMC4894487](https://pmc.ncbi.nlm.nih.gov/articles/PMC4894487/)

### Economic Parameters
14. **Sanders et al. (2016)** — "Recommendations for Cost-effectiveness Analyses." JAMA. [Full Text](https://jamanetwork.com/journals/jama/fullarticle/2552214)

### Algorithm
15. **Sutton & Barto (2018)** — *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. [PDF](http://incompleteideas.net/book/RLbook2020.pdf)

---

## 10. Data Quality Notes

- All incidence and mortality data are from US national databases (SEER, SSA)
- Health utility values from systematic review of 69 studies
- Disease progression rates from large cohort studies (n>10,000)
- All values have been validated against multiple sources where possible
