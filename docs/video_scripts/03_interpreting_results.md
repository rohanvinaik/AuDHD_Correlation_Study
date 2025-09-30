# Video Script: Interpreting Results (15 minutes)

## Introduction (1 minute)

"You've run the pipeline and got results. Now what? This tutorial explains how to interpret and validate your findings."

## Understanding the Report (3 minutes)

"Open report.html in your browser."

**Navigate through sections:**

### 1. Executive Summary
- Number of clusters found
- Cluster sizes
- Overall quality metrics

### 2. Cluster Visualization
- UMAP embedding plot
- Each point is a sample
- Colors represent clusters
- Tightly grouped = strong clusters
- Scattered = weak structure

### 3. Quality Metrics
- **Silhouette score** (0.56 in this example)
  - \> 0.5 = reasonable clusters
  - Higher is better
- **Stability** (ARI = 0.72)
  - How reproducible across resampling
  - \> 0.7 = stable

## Clinical Characterization (4 minutes)

"Compare clusters by clinical variables."

**Show boxplots:**

### Age Distribution
"Cluster 2 has significantly older patients (p < 0.01)"

### Severity Scores
"Cluster 1 shows highest severity (mean = 75 vs 45)"

### Diagnosis Enrichment
"Cluster 1: enriched for AuDHD (60% vs 25% overall)
Cluster 2: mostly ADHD (70%)
Cluster 3: mixed profile"

**Show statistical tests:**
- ANOVA for continuous variables
- Chi-square for categorical
- FDR correction for multiple testing

## Biological Interpretation (4 minutes)

"What makes clusters different biologically?"

### Differentially Abundant Features

**Show heatmap:**

"Each row = feature (gene, metabolite)
Each column = sample
Red = high, blue = low"

"Cluster 1 signatures:
- Upregulated: glucose metabolism genes
- Downregulated: neurotransmitter pathways"

### Pathway Enrichment

**Show enrichment table:**

"Top enriched pathways in Cluster 1:
1. Serotonin signaling (p = 1.2e-8)
2. GABA metabolism (p = 3.4e-6)
3. Inflammatory response (p = 8.9e-5)"

"This suggests neurochemical imbalance"

## Validation Checks (2 minutes)

"Always validate your results:"

### 1. Internal Validation
- Silhouette score > 0.4 ✓
- Bootstrap stability > 0.6 ✓
- Clusters larger than 20 samples ✓

### 2. Sanity Checks
- Do clusters differ clinically? ✓
- Are pathways biologically relevant? ✓
- Any batch effects remaining? Check PCA

### 3. Red Flags
- All samples in one cluster → increase sensitivity
- Many noise points (HDBSCAN) → check data quality
- No clinical associations → may not be biologically meaningful

## Common Interpretations (1 minute)

**Scenario 1:** Clusters match diagnosis
- Validates pipeline
- But limited new insight

**Scenario 2:** Clusters cut across diagnosis
- Novel subtypes discovered!
- More interesting scientifically

**Scenario 3:** No clear clusters
- Data may be continuous spectrum
- Try supervised methods instead

## Next Steps

"Now you can:
1. Validate on independent cohort
2. Predict outcomes using cluster labels
3. Design targeted interventions
4. Publish findings!

Remember: biological validation is crucial"