# Canonical Preprocessing Order

This document defines the **mandatory order** of preprocessing operations and rationale for each step.

## Preprocessing Pipeline Order

```
1. QC Filtering (remove low-quality samples/features)
   ↓
2. Transformation (modality-specific: log, CLR, etc.)
   ↓
3. Imputation (fill missing values)
   ↓
4. Scaling/Normalization (standardize ranges)
   ↓
5. Batch Correction (remove technical variation)
   ↓
6. Covariate Adjustment (mixed effects for context)
   ↓
7. Final QC & Manifest Export
```

## Detailed Steps

### 1. QC Filtering (First)

**Why first:** Remove unusable data before spending computation on it.

**Operations:**
- Remove samples with low call rate / sequencing depth
- Remove features with excessive missing values
- Flag outliers for review

**Thresholds by modality:**

```yaml
genomic:
  min_call_rate: 0.90        # Sample must have ≥90% SNPs called
  min_maf: 0.01              # SNP must have MAF ≥1%
  max_missing_per_snp: 0.10  # SNP can have ≤10% missing

clinical:
  max_missing_per_feature: 0.50  # Drop features >50% missing
  flag_age_outliers: true        # Flag age >3 SD from mean

metabolomic:
  min_detection_rate: 0.30       # Metabolite present in ≥30% samples
  max_cv_in_qc: 0.30             # CV <30% in QC samples
  remove_blank_ratio: 3.0        # Signal/blank ratio >3

microbiome:
  min_sequencing_depth: 10000    # ≥10K reads per sample
  min_prevalence: 0.10           # Taxon in ≥10% samples
```

**Rationale:** Failing to filter first wastes computation and can bias imputation.

---

### 2. Transformation (Modality-Specific)

**Why before imputation:** Many imputation methods assume normality.

**Genomic:** None (genotypes are discrete 0/1/2)

**Clinical:** None or log for skewed variables

**Metabolomic:** **ALWAYS log2-transform**
```python
metabolomic_transformed = np.log2(metabolomic + 1)
```
**Rationale:** Metabolite abundances are log-normally distributed. Log-transform stabilizes variance and makes data more normal for imputation.

**Microbiome:** **ALWAYS CLR-transform** (Centered Log-Ratio)
```python
from skbio.stats.composition import clr
microbiome_clr = clr(microbiome + pseudocount)
```
**Rationale:** Microbiome data is compositional (relative abundances sum to 1). CLR breaks the sum constraint and handles zeros.

**Property to test:** Transformation must be monotonic within each feature (preserves rank order).

---

### 3. Imputation

**Why after transformation, before scaling:** Imputation works better on normalized distributions, but before scaling so imputed values are in original distribution space.

**Methods by modality:**

```yaml
genomic:
  method: "mode"              # Most common genotype (0/1/2)
  # or "haplotype_based" if phased

clinical:
  method: "iterative"         # MICE with random forests
  max_iter: 10
  random_state: 42            # MUST set for reproducibility

metabolomic:
  method: "knn"               # K-nearest neighbors
  n_neighbors: 5
  weights: "distance"
  # or "half_min" for left-censored (below detection)

microbiome:
  method: "zero"              # Absences are meaningful
  # Do NOT impute - missing = truly absent
```

**Convergence for iterative methods:**
```python
# In impute.py
if not converged and iter >= max_iter:
    warnings.warn(f"Imputation did not converge after {max_iter} iterations")
    # Use last iteration result (partial imputation)
```

**Properties to test:**
- Non-missing values unchanged
- Imputed values within observed range
- Deterministic with fixed seed

---

### 4. Scaling/Normalization

**Why after imputation, before batch correction:** Need complete data for scaling. Batch correction works better on scaled data.

**Methods by modality:**

```yaml
genomic:
  method: "standard"          # Z-score: mean=0, std=1
  # Per-SNP scaling

clinical:
  method: "robust"            # Median=0, IQR=1
  # Resistant to outliers

metabolomic:
  method: "standard"          # After log-transform
  # or "quantile" for extreme non-normality

microbiome:
  method: "none"              # CLR already normalized
```

**Standard scaling (Z-score):**
```python
X_scaled = (X - mean) / std
```

**Robust scaling:**
```python
X_scaled = (X - median) / IQR
```

**Quantile normalization:**
```python
# Forces same distribution across samples
X_quantile = quantile_normalize(X)
```

**Properties to test:**
- Preserves rank order within feature
- Achieves target mean/median
- Achieves target std/IQR

---

### 5. Batch Correction

**Why after scaling, before covariate adjustment:** Batch effects are technical variation. Remove before modeling biological covariates.

**Methods:**

```yaml
batch_correction:
  method: "combat"            # ComBat (parametric)
  # or "combat_seq" for count data
  # or "harmony" for large datasets

  covariates_to_preserve:     # DO NOT remove these
    - age
    - sex
    - diagnosis
    - ancestry_PC1-PC10

  batch_variable: "site"      # or "batch", "plate"

  parametric: true            # False for non-normal data

  # Convergence parameters
  max_iter: 100
  conv_threshold: 0.0001
  random_state: 42
```

**ComBat implementation:**
```python
from combat.pycombat import pycombat

corrected = pycombat(
    data=scaled_data,
    batch=batch_labels,
    mod=covariate_matrix,    # Preserve these
    par_prior=True,          # Parametric priors
    mean_only=False,         # Adjust both mean and variance
    ref_batch=None           # No reference batch
)
```

**Validation (MUST test):**

1. **Batch effect reduced:**
```python
# Variance explained by batch (before vs after)
from sklearn.metrics import r2_score

# Fit batch as predictor of PC1
r2_before = r2_score(pc1_before, batch_one_hot)
r2_after = r2_score(pc1_after, batch_one_hot)

assert r2_after < r2_before * 0.5  # Batch effect reduced by ≥50%
```

2. **Biological signal preserved:**
```python
# Variance explained by diagnosis preserved
r2_diagnosis_before = r2_score(pc1_before, diagnosis_one_hot)
r2_diagnosis_after = r2_score(pc1_after, diagnosis_one_hot)

assert r2_diagnosis_after > r2_diagnosis_before * 0.8  # Retain ≥80% signal
```

3. **Silhouette score preserved:**
```python
from sklearn.metrics import silhouette_score

sil_before = silhouette_score(data_before, true_labels)
sil_after = silhouette_score(data_after, true_labels)

assert sil_after > sil_before * 0.9  # Don't over-smooth
```

**Properties to test:**
- Batch variance reduced
- Biological variance retained
- No introduction of artificial separation

---

### 6. Covariate Adjustment (Mixed Effects)

**Why last:** After removing technical variation, model biological context.

**Use case:** Adjust for context variables that are nuisance factors but not batches:

```yaml
covariate_adjustment:
  method: "mixed_effects"     # or "linear_regression"

  covariates:
    # Context variables to regress out
    - fasting_hours
    - time_of_day
    - menstrual_phase
    - sleep_hours_last_night
    - last_medication_hours

  random_effects:
    - subject_id              # For longitudinal data
    - family_id               # For family studies

  fixed_effects_to_preserve:
    - diagnosis               # Keep these in residuals
    - age
    - sex
```

**Mixed effects model:**
```python
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# For each feature:
md = MixedLM(
    endog=feature_values,
    exog=fixed_effects,       # Covariates to adjust
    groups=subject_id,        # Random intercept per subject
)

try:
    result = md.fit(method='lbfgs', maxiter=100)

    if not result.converged:
        # Fallback to simpler model
        warnings.warn(f"Mixed effects did not converge for {feature_name}, using linear regression")
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(fixed_effects, feature_values)
        residuals = feature_values - lr.predict(fixed_effects)
    else:
        residuals = result.resid

except Exception as e:
    warnings.warn(f"Mixed effects failed for {feature_name}: {e}. Using raw values.")
    residuals = feature_values  # No adjustment

# Use residuals for downstream analysis
```

**Convergence fallbacks:**
1. Try REML → BFGS → Nelder-Mead optimizers
2. If all fail → Linear regression (no random effects)
3. If that fails → No adjustment (log warning)

**Deterministic seeding:**
```python
# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
```

**Properties to test:**
- Covariate effects removed (orthogonal to covariates)
- Preserved effects unchanged
- Deterministic results with same seed

---

### 7. Final QC & Manifest Export

**Generate preprocessing manifest:**

```json
{
  "preprocessing_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "modalities": {
    "genomic": {
      "n_samples_input": 1000,
      "n_samples_output": 950,
      "n_features_input": 500000,
      "n_features_output": 50000,
      "steps": [
        {"step": "qc_filter", "params": {"min_call_rate": 0.9, "min_maf": 0.01}},
        {"step": "impute", "params": {"method": "mode"}},
        {"step": "scale", "params": {"method": "standard"}},
        {"step": "batch_correct", "params": {"method": "combat", "batch_var": "site"}}
      ],
      "qc_metrics": {
        "mean_call_rate": 0.98,
        "missing_rate_after_impute": 0.0
      }
    },
    "metabolomic": {
      "n_samples_input": 1000,
      "n_samples_output": 1000,
      "n_features_input": 300,
      "n_features_output": 250,
      "steps": [
        {"step": "qc_filter", "params": {"min_detection_rate": 0.3}},
        {"step": "transform", "params": {"method": "log2"}},
        {"step": "impute", "params": {"method": "knn", "n_neighbors": 5}},
        {"step": "scale", "params": {"method": "standard"}},
        {"step": "batch_correct", "params": {"method": "combat"}},
        {"step": "adjust_covariates", "params": {"method": "mixed_effects"}}
      ],
      "qc_metrics": {
        "mean_detection_rate": 0.85,
        "missing_rate_after_impute": 0.0,
        "batch_effect_reduction": 0.73
      }
    }
  },
  "config_file": "configs/preprocessing/standard.yaml",
  "random_seed": 42
}
```

**Export function:**
```python
def export_preprocessing_manifest(
    steps_applied: dict,
    qc_metrics: dict,
    output_path: str
) -> None:
    """Export preprocessing manifest for reproducibility"""
    manifest = {
        "preprocessing_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "modalities": {},
        "random_seed": config.seed
    }

    for modality, steps in steps_applied.items():
        manifest["modalities"][modality] = {
            "steps": steps,
            "qc_metrics": qc_metrics.get(modality, {})
        }

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
```

---

## Why This Order Matters

### ❌ Wrong Order Example 1: Scale before transform

```python
# WRONG: Scale metabolomics before log-transform
metabolomic_scaled = (metabolomic - mean) / std      # Negative values!
metabolomic_log = np.log2(metabolomic_scaled + 1)    # Biased by shift
```

**Problem:** Metabolite abundances span orders of magnitude. Scaling first creates negative values and biases the log-transform.

**Correct:**
```python
# RIGHT: Log-transform first
metabolomic_log = np.log2(metabolomic + 1)
metabolomic_scaled = (metabolomic_log - mean) / std
```

### ❌ Wrong Order Example 2: Batch correction before scaling

```python
# WRONG: ComBat on unscaled data with different variances
corrected = pycombat(metabolomic_unscaled, batch)  # Features with large variance dominate
```

**Problem:** ComBat adjusts mean and variance. If features have vastly different scales, batch correction is unbalanced.

**Correct:**
```python
# RIGHT: Scale first so all features have equal weight
metabolomic_scaled = scale(metabolomic)
corrected = pycombat(metabolomic_scaled, batch)
```

### ❌ Wrong Order Example 3: Impute after batch correction

```python
# WRONG: Batch correct with missing values
corrected = pycombat(data_with_missing, batch)  # ComBat can't handle NaN
imputed = impute(corrected)
```

**Problem:** Most batch correction methods require complete data.

**Correct:**
```python
# RIGHT: Impute first
imputed = impute(data_with_missing)
corrected = pycombat(imputed, batch)
```

---

## Configuration Template

```yaml
# configs/preprocessing/standard.yaml

preprocessing:
  # Step 1: QC Filtering
  qc_filter:
    enabled: true
    genomic:
      min_call_rate: 0.90
      min_maf: 0.01
    clinical:
      max_missing_per_feature: 0.50
    metabolomic:
      min_detection_rate: 0.30
    microbiome:
      min_sequencing_depth: 10000

  # Step 2: Transformation
  transform:
    genomic: null                # No transform
    clinical: null               # No transform (or log for skewed)
    metabolomic: "log2"          # ALWAYS log2
    microbiome: "clr"            # ALWAYS CLR

  # Step 3: Imputation
  impute:
    genomic:
      method: "mode"
    clinical:
      method: "iterative"
      max_iter: 10
      random_state: 42
    metabolomic:
      method: "knn"
      n_neighbors: 5
    microbiome:
      method: "zero"             # Don't impute

  # Step 4: Scaling
  scale:
    genomic: "standard"
    clinical: "robust"
    metabolomic: "standard"
    microbiome: null             # Already normalized by CLR

  # Step 5: Batch Correction
  batch_correct:
    method: "combat"
    batch_variable: "site"
    covariates_to_preserve:
      - age
      - sex
      - diagnosis
    parametric: true
    max_iter: 100
    random_state: 42

  # Step 6: Covariate Adjustment
  adjust_covariates:
    enabled: true
    method: "mixed_effects"
    covariates:
      - fasting_hours
      - time_of_day
    random_effects:
      - subject_id
    fallback_on_failure: "linear_regression"

  # Step 7: Export
  export_manifest: true
  manifest_path: "outputs/preprocessing_manifest.json"
```

---

## Testing Requirements

All preprocessing steps must have:

1. **Unit tests** - Each function works correctly
2. **Integration tests** - Steps run in correct order
3. **Property tests** - Invariants preserved (see below)
4. **Validation tests** - Batch correction doesn't over-shrink

See: `tests/property/test_preprocessing_properties.py`