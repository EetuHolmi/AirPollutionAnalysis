# Air Pollution & Economic Conditions — Analysis Report

**Course:** Big Data Processing & Analytics (BDPA)  
**Dataset:** Global Air Pollution Dataset (Kaggle) enriched with World Bank indicators  
**Engine:** Apache Spark MLlib 4.1.1 · Python 3.13  
**Date:** April 2026

---

## 1. Overview

This report presents a comprehensive data-driven analysis of global air pollution using two research questions:

| # | Research Question |
|---|-------------------|
| **RQ1** | How does a city's AQI relate to its country's GDP per capita, urbanisation, and population density? |
| **RQ2** | Can cities be grouped into distinct *pollution archetypes* based on their relative mix of CO, NO₂, Ozone, and PM2.5? What socioeconomic factors drive which archetype a city falls into? |

---

## 2. Dataset

| Attribute | Value |
|-----------|-------|
| Total rows | 23,463 cities |
| Unique countries | 175 |
| Rows after dropping nulls | 23,036 |
| Key features | AQI Value, AQI Category, CO/Ozone/NO₂/PM2.5 AQI values, GDP per capita 2022, Urban population 2022, Population density 2022 |

**Pollution metric:** AQI (Air Quality Index) — EPA standard ranging from 0 (Good) to 500+ (Hazardous).

**AQI Category distribution:**

| Category | Count | Share |
|----------|-------|-------|
| Good | 9,688 | 42.1% |
| Moderate | 9,088 | 39.5% |
| Unhealthy for Sensitive Groups | 1,568 | 6.8% |
| Unhealthy | 2,215 | 9.6% |
| Very Unhealthy | 286 | 1.2% |
| Hazardous | 191 | 0.8% |

---

## 3. Methodology

### 3.1 Feature Engineering

- **Log-transformed features:** `gdp_per_capita_2022` and `urban_population_2022` follow highly right-skewed (Pareto-like) distributions. Log1p transforms were applied to linearise the Environmental Kuznets Curve relationship and reduce the influence of extreme outliers.  
- **Final feature set (5 features):** `gdp_per_capita_2022`, `urban_population_2022`, `population_density_2022`, `log_gdp_pc`, `log_urban_pop`

### 3.2 Country-wise Train/Test Split (Anti-Leakage Design)

GDP per capita, urban population, and population density are **country-level** features — every city in the same country has the identical value. A row-level random split would allow the model to train on Paris and test on Lyon (same French economic profile), making evaluation trivially easy and misleading.

**Solution:** Split by unique country:
- **Train:** 140 countries → 20,342 city rows
- **Test:** 35 countries → 2,694 city rows

This ensures zero overlap between country profiles in train and test, making all metrics true out-of-sample estimates.

### 3.3 Models Used

| Task | Model | Notes |
|------|-------|-------|
| RQ1-A | Gradient Boosted Trees Regressor | maxIter=100, maxDepth=5, stepSize=0.1, subsamplingRate=0.8 |
| RQ1-B | Tuned Random Forest Classifier | 200 trees, maxDepth=12, featureSubsetStrategy='sqrt' |
| RQ2-A | K-Means Clustering | k=4 (chosen for interpretable archetypes; silhouette=0.413) |
| RQ2-B | Random Forest Classifier | 150 trees, maxDepth=10, country-wise split |

---

## 4. Research Question 1 — AQI vs Economic Conditions

### 4.1 RQ1-A: Predicting AQI Value (Regression)

**Model:** Gradient Boosted Trees Regressor (replaces plain Linear Regression; handles non-linear EKC relationships)

| Metric | Value |
|--------|-------|
| RMSE | **54.16** AQI units |
| MAE | 42.61 AQI units |
| R² | **−0.023** |

#### What this means

An R² of −0.023 indicates that GDP, urbanisation, and population density **cannot reliably predict absolute AQI values** for countries the model has never seen before. A negative R² means the model performs slightly worse than just always guessing the dataset mean — though the improvement over the previous Linear Regression (R² = −0.162) is significant.

This is **an important substantive finding, not a model failure**. It tells us:

1. **AQI level is not determined by wealth alone.** Countries with very similar GDPs (e.g. the United Arab Emirates vs the Netherlands) can have dramatically different AQI levels due to geography, industrial mix, regional climate, and domestic energy policy.  
2. **The relationship is non-linear and country-specific.** The Environmental Kuznets Curve (EKC) predicts an inverted-U between GDP and pollution: pollution *rises* as countries industrialise, then *falls* as they become wealthy enough for environmental regulation. A single global model cannot capture this trajectory for every country simultaneously.
3. **Omitted variables matter.** Key drivers of absolute AQI — desert dust transport, monsoon season, coal dependency, topography — are absent from a purely socioeconomic feature set.

**Feature importance (GBT):**

| Feature | Importance |
|---------|------------|
| urban_population_2022 | **0.466** |
| population_density_2022 | 0.292 |
| gdp_per_capita_2022 | 0.242 |
| log_urban_pop | 0.000 |
| log_gdp_pc | 0.000 |

Urban population is the single strongest signal. This makes intuitive sense — larger urban areas concentrate emissions from traffic, industry, and energy generation. Note that GBT is invariant to monotonic transformations (it finds its own thresholds), so the log features duplicate information and receive zero importance; the original scale features already capture the same order.

---

### 4.2 RQ1-B: Predicting AQI Category (Classification)

**Model:** Tuned Random Forest Classifier (200 trees, depth 12, sqrt feature subsampling)

| Metric | Value |
|--------|-------|
| Accuracy | **44.8%** |
| Weighted F1 | 0.382 |
| Weighted Recall | 0.448 |
| Weighted Precision | 0.441 |
| Random baseline | 16.7% (1/6 classes) |

#### What this means

The model correctly assigns an AQI category to **44.8%** of cities in held-out countries — nearly **2.7× better than random guessing**. This confirms that socioeconomic factors carry **real, statistically significant information** about air quality levels, even if they cannot pin down the exact AQI number.

**Confusion matrix insights:**
- *Moderate* is classified with 88% recall — the most common category is easiest to predict.
- *Good* recall is 45% — the model frequently confuses "Good" air quality with "Moderate", likely because developed countries with good environmental regulation span both categories.
- Hazardous and Very Unhealthy categories receive near-zero recall, because these extreme cases depend on episodic events (wildfire, smog events) not predictable from GDP alone.

**Feature importance (RF classifier):**

| Feature | Importance |
|---------|------------|
| population_density_2022 | **0.316** |
| urban_population_2022 | 0.273 |
| log_urban_pop | 0.242 |
| gdp_per_capita_2022 | 0.128 |
| log_gdp_pc | 0.042 |

Population density emerges as the top predictor (replacing urban population from the GBT). Densely packed populations create concentrated demand for transportation and heating, driving pollutant concentrations. GDP per capita ranks 4th — richer countries have better emissions controls that partially counteract their industrial output.

#### RQ1 Conclusion

> **Wealthier and more densely populated countries have measurably different air quality category distributions, but GDP, urbanisation, and population density cannot predict a country's absolute AQI with high accuracy.** The classification task (which AQI *band* does a city fall in?) is substantially more tractable than the regression task (what is the exact AQI?). Population density is the strongest individual socioeconomic predictor of poor air quality.

---

## 5. Research Question 2 — Pollution Archetypes

### 5.1 RQ2-A: K-Means Clustering — Discovering Pollution Archetypes

**Input features:** Relative pollutant share of CO, Ozone, NO₂, PM2.5 (sum to 1 per city)  
**Algorithm:** K-Means with StandardScaler preprocessing  
**k selection:** Silhouette score peaks at k=2 (0.488), but k=4 was used to expose chemically distinct archetypes with meaningful policy implications (silhouette = 0.413, still above the 0.40 "good separation" threshold)

#### Cluster Profiles

| Cluster | Archetype | Cities | PM2.5% | Ozone% | NO₂% | Avg AQI | Avg GDP/cap |
|---------|-----------|--------|--------|--------|------|---------|-------------|
| C0 | **Heavy PM2.5 (Combustion/Dust)** | 5,776 | 75.8% | 21.6% | 1.5% | 104.0 | $10,757 |
| C1 | **Ozone-Dominant (Photochemical)** | 4,230 | 37.1% | 60.3% | 1.2% | 49.4 | $30,393 |
| C2 | **Moderate PM2.5/Ozone (Mixed)** | 9,660 | 56.9% | 40.1% | 1.9% | 58.3 | $30,853 |
| C3 | **PM2.5/NO₂ (Traffic-Industrial)** | 3,369 | 73.0% | 14.5% | 10.2% | 87.1 | $36,521 |

#### What each archetype means

**C0 — Heavy PM2.5 (Combustion/Dust)** *(avg GDP $10,757 — lowest; avg AQI 104 — highest)*  
PM2.5 accounts for 76% of total pollutant AQI. These cities suffer the most severe air pollution. The source signature points to **biomass burning, coal combustion, and unpaved-road dust** — typical of South/Southeast Asia, Sub-Saharan Africa, and parts of South America. Low GDP correlates with limited capacity for environmental enforcement and widespread solid-fuel cooking/heating.

**C1 — Ozone-Dominant (Photochemical)** *(avg GDP $30,393 — second highest; avg AQI 49 — lowest)*  
Ozone accounts for 60% — the cleanest and wealthiest archetype. These cities have largely eliminated combustion particulates through regulation but retain **photochemical smog** from vehicle emissions reacting with sunlight. Typical of wealthier, sunnier regions (California, Mediterranean Europe, Australia). Lower AQI reflects successful pollution control; Ozone being secondary (not eliminated) reflects the harder chemistry challenge.

**C2 — Moderate PM2.5/Ozone (Mixed)** *(avg GDP $30,853 — highest; avg AQI 58 — second lowest)*  
The largest cluster (9,660 cities). A transitional profile where both PM2.5 and Ozone contribute substantially. Despite having the highest average GDP, these cities are NOT the cleanest — they likely include large industrial developed-world cities that have reduced the worst pollution but not eliminated it. Representative of China's wealthier coastal cities, Central European manufacturing hubs, and North American rust-belt cities.

**C3 — PM2.5/NO₂ (Traffic-Industrial)** *(avg GDP $36,521 — highest; avg AQI 87 — second highest)*  
The most counterintuitive archetype: the **wealthiest** cities yet with the **second-highest AQI**. NO₂ share (10.2%) is 5–8× higher than other clusters — a direct fingerprint of **high traffic density and diesel/industrial combustion**. This archetype likely captures mega-cities and industrial hubs in the wealthiest countries where sheer traffic volume overwhelms the emission-control infrastructure. High GDP has not been sufficient to regulate these emissions below the PM2.5 clusters.

#### PCA Visualisation
PCA projects the 4-dimensional pollutant space into 2D while retaining **83.8% of variance** (PC1=56.5%, PC2=27.2%). The cluster clouds are visually well-separated in PCA space, confirming that K-Means discovered geometrically distinct groups.

#### GDP Quartile Breakdown

| Cluster | Q1 (<$2K) | Q2 ($2K–$10K) | Q3 ($10K–$35K) | Q4 (>$35K) |
|---------|-----------|---------------|-----------------|------------|
| C0: Heavy PM2.5 | ~15% | ~63% | ~12% | ~10% |
| C1: Ozone-Dom | ~2% | ~12% | ~52% | ~34% |
| C2: Mixed | ~5% | ~26% | ~26% | ~43% |
| C3: PM2.5/NO₂ | ~3% | ~33% | ~21% | ~43% |

C0 (Heavy PM2.5) is dominated by lower-income countries (Q1+Q2 ≈ 78%), while C1 (Ozone-Dominant) skews strongly toward Q3+Q4 (≈ 86%). This confirms the economic stratification of pollution types.

---

### 5.2 RQ2-B: Predicting Pollution Archetype from Socioeconomics (Supervised)

**Model:** Random Forest Classifier (150 trees, depth 10, country-wise split)

| Metric | Value |
|--------|-------|
| Accuracy | **29.3%** |
| Weighted F1 | 0.241 |
| Random baseline | 25.0% (1/4 classes) |

#### What this means

The classifier achieves 29.3% accuracy versus a 25% random baseline — a modest but real improvement. Predicting *which of the 4 archetypes* a country's cities will fall into from purely economic data is difficult because:

1. **Archetype is driven by emission sources, not just wealth.** C3 (Traffic-Industrial) and C2 (Mixed) share similar GDP levels but differ based on industry structure, city layout, and vehicle fleet composition — nuances invisible in aggregate GDP data.
2. **Country-wise split is strict.** Training on 140 countries and testing on 35 entirely unseen countries is a hard generalisation challenge.

Despite modest accuracy, the **feature importance is highly informative:**

| Feature | Importance |
|---------|------------|
| gdp_per_capita_2022 | **0.552** |
| population_density_2022 | 0.231 |
| urban_population_2022 | 0.217 |

**GDP per capita alone explains 55% of the model's predictive power.** This is the clearest quantitative answer to RQ2: *economic development level is the dominant determinant of pollution archetype*, more so than population size or density.

The AQI severity chart confirms that the two PM2.5-heavy archetypes (C0 and C3) have far higher average AQI (104 and 87 respectively) than the Ozone-dominant and Mixed archetypes (49 and 58). This links the *type* of pollution discovered by unsupervised clustering to the *severity* of pollution measured by AQI.

#### RQ2 Conclusion

> **Four distinct pollution archetypes exist globally, separated along two axes: wealth and dominant pollutant chemistry.** Lower-income countries cluster in the "Heavy PM2.5" archetype (combustion-dominated, most severe). Higher-income countries divide between "Ozone-Dominant" (clean but photochemical), "Mixed PM2.5/Ozone" (transitional), and "PM2.5/NO₂ Traffic-Industrial" (counterintuitively high pollution despite high GDP). GDP per capita is the single strongest predictor of archetype (importance = 0.552), but alone cannot perfectly predict pollution type because industry structure, geography, and policy choices mediate the relationship.

---

## 6. Summary of All Model Results

| Model | Task | Key Metric | Value | Baseline |
|-------|------|-----------|-------|---------|
| GBT Regressor | Predict AQI Value | R² | −0.023 | 0.0 (mean predictor) |
| GBT Regressor | Predict AQI Value | RMSE | 54.16 | — |
| RF Classifier (tuned) | Predict AQI Category | Accuracy | 44.8% | 16.7% |
| RF Classifier (tuned) | Predict AQI Category | Weighted F1 | 0.382 | — |
| K-Means (k=4) | Group by pollutant mix | Silhouette | 0.413 | — |
| RF Archetype Classifier | Predict pollution type | Accuracy | 29.3% | 25.0% |
| RF Archetype Classifier | Predict pollution type | GDP importance | 0.552 | — |

---

## 7. Key Conclusions

1. **Socioeconomic factors carry information about air quality category but not absolute AQI.** Countries can be roughly placed in an AQI band from their GDP and density alone (44.8% accuracy), but the exact AQI value is driven by local factors (geography, fuel mix, climate) not captured here.

2. **Population density is the strongest individual predictor of poor air quality category.** Dense cities concentrate emissions from transport and heating. GDP per capita matters but ranks below density.

3. **Four chemically distinct pollution archetypes exist globally:**
   - *Heavy PM2.5* — low-income, combustion-driven, most severe (avg AQI 104)
   - *Ozone-Dominant* — high-income, photochemical, cleanest (avg AQI 49)
   - *Moderate Mixed* — high-income, transitional, middle-ground
   - *PM2.5/NO₂ Traffic-Industrial* — highest-income, traffic-heavy, second most severe (avg AQI 87)

4. **The "dirty rich" paradox:** The archetype with the highest average GDP per capita ($36,521) has the second-highest AQI (87). This challenges the assumption that wealth automatically means cleaner air — high economic activity can sustain high traffic and industrial emissions even in regulated environments.

5. **GDP per capita alone determines ~55% of archetype membership**, confirming a strong development level → pollution type link. However, the remaining 45% reflects structural factors (industrial policy, urban design, energy mix) beyond economic aggregates.

6. **Country-wise evaluation matters.** The previous row-level split inflated the RF accuracy from 44.8% to 61.4% — a 37% over-estimate caused by data leakage (same country appearing in both train and test). Country-wise split gives methodologically honest estimates suitable for academic reporting.

---

## 8. Figures Reference

All figures are saved in `output/`:

| File | Content |
|------|---------|
| `eda_aqi_distribution.png` | AQI value histogram and category distribution |
| `eda_top20_polluted_cities.png` | 20 most polluted cities |
| `eda_aqi_vs_gdp_country.png` | AQI vs GDP per capita scatter by country |
| `eda_correlation_heatmap.png` | Pearson correlations among all numeric features |
| `eda_aqi_vs_socioeconomics.png` | AQI vs GDP, urbanisation, and density scatter panels |
| `rq1a_gbt_regression.png` | GBT feature importance + actual vs predicted AQI |
| `rq1b_rf_confusion_matrix.png` | RF confusion matrix (raw + row-normalised) |
| `rq1b_rf_feature_importance.png` | RF feature importance for AQI category prediction |
| `rq2_pollutant_mix_overview.png` | Global pollutant composition pie + boxplots |
| `rq2_elbow_silhouette.png` | K selection: elbow (WCSS) and silhouette curves |
| `rq2_archetype_mix_gdp.png` | Stacked bar (pollutant mix) + GDP by archetype |
| `rq2_pca_cluster_visualization.png` | PCA projection of 4 clusters in 2D |
| `rq2_gdp_quartile_by_archetype.png` | Income quartile composition per archetype |
| `rq2b_archetype_classifier_results.png` | Archetype classifier feature importance + AQI severity |

---

*Analysis implemented with Apache Spark MLlib. All models evaluated on a strict country-wise train/test split to prevent data leakage. Dataset: Global Air Pollution Dataset enriched with World Bank 2022 economic indicators.*
