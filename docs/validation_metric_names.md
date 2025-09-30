# Standard Validation Metric Names

This document defines the canonical naming convention for all validation metrics to ensure consistency across the codebase.

## Naming Convention

All metric names follow the pattern: `{metric}_{statistic}`

- **metric**: Short identifier (e.g., `ari`, `ami`, `silhouette`)
- **statistic**: Aggregation type (e.g., `mean`, `std`, `median`, `ci`)

## Standard Metrics

### Similarity/Agreement Metrics

| Metric | Name | Range | Interpretation |
|--------|------|-------|----------------|
| Adjusted Rand Index | `ari` | [-1, 1] | Agreement corrected for chance |
| Adjusted Mutual Information | `ami` | [0, 1] | Information overlap corrected for chance |
| Fowlkes-Mallows Score | `fmi` | [0, 1] | Geometric mean of precision/recall |
| Normalized Mutual Information | `nmi` | [0, 1] | Mutual information normalized |
| Jaccard Index | `jaccard` | [0, 1] | Intersection over union |

### Internal Quality Metrics

| Metric | Name | Range | Interpretation |
|--------|------|-------|----------------|
| Silhouette Score | `silhouette` | [-1, 1] | Separation quality (higher = better) |
| Davies-Bouldin Index | `davies_bouldin` | [0, ∞] | Compactness (lower = better) |
| Calinski-Harabasz Index | `calinski_harabasz` | [0, ∞] | Variance ratio (higher = better) |
| Dunn Index | `dunn` | [0, ∞] | Separation/compactness (higher = better) |

### Statistics

| Statistic | Suffix | Description |
|-----------|--------|-------------|
| Mean | `_mean` | Average value |
| Standard Deviation | `_std` | Variability |
| Median | `_median` | Robust central tendency |
| Minimum | `_min` | Worst case |
| Maximum | `_max` | Best case |
| Confidence Interval | `_ci` | Tuple (lower, upper) at 95% |
| Percentile | `_p{N}` | e.g., `_p25`, `_p75` |

## Standard Field Names

### StabilityResult

**Required fields:**
```python
@dataclass
class StabilityResult:
    # Raw scores (all bootstraps)
    ari_scores: np.ndarray
    ami_scores: np.ndarray
    fmi_scores: np.ndarray
    jaccard_scores: np.ndarray

    # Summary statistics
    ari_mean: float
    ari_std: float
    ari_ci: Tuple[float, float]

    ami_mean: float
    ami_std: float
    ami_ci: Tuple[float, float]

    fmi_mean: float
    fmi_std: float

    jaccard_mean: float
    jaccard_std: float

    # Overall
    stability_score: float  # 0-1, higher = more stable
    interpretation: str     # "excellent", "good", "moderate", "poor"
```

**Deprecated (DO NOT USE):**
- `mean_ari` ❌ Use `ari_mean` ✅
- `std_ari` ❌ Use `ari_std` ✅
- `confidence_interval_ari` ❌ Use `ari_ci` ✅

### InternalValidationMetrics

**Required fields:**
```python
@dataclass
class InternalValidationMetrics:
    # Primary metrics
    silhouette_mean: float
    silhouette_std: Optional[float] = None
    silhouette_per_cluster: Dict[int, float] = field(default_factory=dict)

    davies_bouldin: float
    calinski_harabasz: float
    dunn: float

    # Variance decomposition
    within_cluster_variance: float
    between_cluster_variance: float
    variance_ratio: float

    # Composite
    overall_quality: float  # 0-1, normalized composite score
```

### ExternalValidationMetrics

**Required fields:**
```python
@dataclass
class ExternalValidationMetrics:
    # Agreement with reference
    ari_external: float
    ami_external: float
    nmi_external: float

    # Replication
    replication_rate: float  # Fraction of clusters replicated
    cluster_stability: float  # How stable are assignments

    # Quality in external cohort
    silhouette_external: float

    # Effect sizes
    effect_size_correlation: Optional[float] = None
    biomarker_correlation: Optional[Dict[str, float]] = None
```

### PermutationTestResult

**Required fields:**
```python
@dataclass
class PermutationTestResult:
    # Observed statistic
    observed_ari: float

    # Null distribution
    null_ari_mean: float
    null_ari_std: float
    null_ari_distribution: np.ndarray

    # Significance
    p_value: float
    significant: bool  # p < 0.05
    effect_size: float  # (observed - null_mean) / null_std
```

## Serialization

All dataclasses must implement consistent serialization:

```python
def to_dict(self) -> Dict[str, Any]:
    """
    Convert to dictionary with standard field names

    Returns:
        Dictionary with keys matching standard naming convention
    """
    return {
        # Use standard names
        "ari_mean": self.ari_mean,
        "ari_std": self.ari_std,
        "ari_ci": list(self.ari_ci),  # Convert tuple to list for JSON
        ...
    }
```

## Migration Guide

### Updating Existing Code

**Before (inconsistent):**
```python
result = StabilityResult(
    ari_scores=ari_scores,
    ami_scores=ami_scores,
    mean_ari=np.mean(ari_scores),  # ❌ OLD
    std_ari=np.std(ari_scores),    # ❌ OLD
    confidence_interval_ari=(lower, upper),  # ❌ OLD
    ...
)
```

**After (standardized):**
```python
result = StabilityResult(
    ari_scores=ari_scores,
    ami_scores=ami_scores,
    ari_mean=np.mean(ari_scores),  # ✅ NEW
    ari_std=np.std(ari_scores),    # ✅ NEW
    ari_ci=(lower, upper),          # ✅ NEW
    ...
)
```

### Updating Tests

**Before (with fallback):**
```python
ari_key = 'mean_ari' if 'mean_ari' in result else 'ari_mean'  # ❌ FALLBACK
assert result[ari_key] > 0.5
```

**After (direct access):**
```python
assert result.ari_mean > 0.5  # ✅ STANDARD
```

## Validation

Run this check to find non-standard names:

```python
import re
from pathlib import Path

def check_naming_consistency(file_path: str):
    """Check for old naming patterns"""
    with open(file_path) as f:
        content = f.read()

    # Check for old patterns
    old_patterns = [
        r'mean_ari',
        r'mean_ami',
        r'std_ari',
        r'std_ami',
        r'confidence_interval_ari',
        r'confidence_interval_ami',
    ]

    issues = []
    for pattern in old_patterns:
        matches = re.findall(pattern, content)
        if matches:
            issues.append(f"Found {len(matches)}x '{pattern}'")

    return issues

# Run on all validation files
for file in Path("src/audhd_correlation/validation").glob("*.py"):
    issues = check_naming_consistency(file)
    if issues:
        print(f"{file.name}: {', '.join(issues)}")
```

## Complete Example

```python
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class StabilityResult:
    """Standardized stability result"""
    # Raw scores
    ari_scores: np.ndarray
    ami_scores: np.ndarray
    fmi_scores: np.ndarray

    # Summary statistics (STANDARD NAMING)
    ari_mean: float
    ari_std: float
    ari_ci: Tuple[float, float]

    ami_mean: float
    ami_std: float
    ami_ci: Tuple[float, float]

    fmi_mean: float
    fmi_std: float

    stability_score: float
    interpretation: str

    def to_dict(self):
        """Serialize with standard names"""
        return {
            # Aggregated metrics
            "ari_mean": float(self.ari_mean),
            "ari_std": float(self.ari_std),
            "ari_ci": [float(self.ari_ci[0]), float(self.ari_ci[1])],

            "ami_mean": float(self.ami_mean),
            "ami_std": float(self.ami_std),
            "ami_ci": [float(self.ami_ci[0]), float(self.ami_ci[1])],

            "fmi_mean": float(self.fmi_mean),
            "fmi_std": float(self.fmi_std),

            # Overall
            "stability_score": float(self.stability_score),
            "interpretation": self.interpretation,

            # Raw scores (optional, for inspection)
            "ari_scores": self.ari_scores.tolist(),
            "ami_scores": self.ami_scores.tolist(),
            "fmi_scores": self.fmi_scores.tolist(),
        }
```

## Summary

✅ **DO:**
- Use `{metric}_{statistic}` pattern
- Use `ari_mean`, `ari_std`, `ari_ci`
- Implement `to_dict()` with standard names
- Update tests to use standard names directly

❌ **DON'T:**
- Use `mean_ari`, `std_ari`, `confidence_interval_ari`
- Mix naming conventions
- Rely on fallback logic in tests
- Create new non-standard field names

All validation metrics should follow this standard for consistency.