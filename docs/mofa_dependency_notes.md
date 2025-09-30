# MOFA Dependency Notes

## Version Requirements

**Pinned version:** `mofapy2>=0.7.0`

This project uses a **simplified custom implementation** of MOFA rather than the official `mofapy2` package. This decision was made for:

1. **Flexibility**: Full control over hyperparameters and convergence criteria
2. **Integration**: Better integration with scikit-learn style API
3. **Simplicity**: No R dependencies (official MOFA requires R backend)
4. **Debugging**: Easier to debug and extend

## Official MOFA vs. Our Implementation

### Official mofapy2 API

```python
from mofapy2.run.entry_point import entry_point

# Initialize
ent = entry_point()
ent.set_data_options(...)
ent.set_model_options(...)
ent.set_train_options(...)

# Build and train
ent.build()
ent.run()

# Get results
factors = ent.model.nodes["Z"]["E"]
```

### Our Custom Implementation

```python
from audhd_correlation.integrate.mofa import MOFAIntegration

# Initialize
mofa = MOFAIntegration(
    n_factors=10,
    n_iterations=1000,
    sparsity_prior=True,
    ard_prior=True,
)

# Fit
mofa.fit(data_dict)

# Get results
factors = mofa.get_latent_factors()  # Returns DataFrame
loadings = mofa.get_factor_loadings()  # Returns Dict[str, DataFrame]
```

**Key differences:**
- Our implementation uses scikit-learn style `fit()` / `transform()` API
- Returns pandas DataFrames with proper sample/feature IDs
- Simplified variational inference (not full MOFA+ with groups/time)
- No R backend required

## If Using Official mofapy2

If you want to use the official `mofapy2` package instead, here's how to integrate it:

### Installation

```bash
# Requires R installation first
R -e "install.packages('reticulate')"
pip install mofapy2
```

### Integration Wrapper

```python
from mofapy2.run.entry_point import entry_point
import pandas as pd

def integrate_mofa_official(
    data_dict: Dict[str, pd.DataFrame],
    n_factors: int = 10,
    convergence_mode: str = "fast",
    **kwargs
) -> Dict[str, Any]:
    """
    Integrate using official mofapy2

    Args:
        data_dict: Dictionary of data matrices (samples × features)
        n_factors: Number of latent factors
        convergence_mode: "fast", "medium", or "slow"

    Returns:
        Dictionary with factors and loadings
    """
    # Convert to mofapy2 format
    # mofapy2 expects: Dict[view_name: np.ndarray] with shape (features, samples)
    data_for_mofa = {}
    for view_name, df in data_dict.items():
        data_for_mofa[view_name] = df.T.values  # Transpose to features × samples

    # Initialize entry point
    ent = entry_point()

    # Set data
    ent.set_data_options(
        scale_groups=False,
        scale_views=True,  # Standardize each view
    )

    # Set model options
    ent.set_model_options(
        factors=n_factors,
        spikeslab_weights=True,  # Sparsity prior
        ard_factors=True,  # ARD for factor pruning
        ard_weights=True,
    )

    # Set training options
    ent.set_train_options(
        iter=1000,
        convergence_mode=convergence_mode,
        seed=42,
        verbose=False,
    )

    # Set data
    ent.set_data_matrix(data_for_mofa, likelihoods=["gaussian"] * len(data_dict))

    # Build and train
    ent.build()
    ent.run()

    # Extract results
    factors = ent.model.nodes["Z"]["E"]  # Shape: (samples, factors)

    # Get loadings for each view
    loadings = {}
    for i, view_name in enumerate(data_dict.keys()):
        W = ent.model.nodes["W"]["E"][i]  # Shape: (features, factors)
        loadings[view_name] = pd.DataFrame(
            W,
            index=data_dict[view_name].columns,
            columns=[f"Factor{j+1}" for j in range(n_factors)]
        )

    # Create factors DataFrame
    sample_ids = data_dict[next(iter(data_dict.keys()))].index
    factors_df = pd.DataFrame(
        factors,
        index=sample_ids,
        columns=[f"Factor{i+1}" for i in range(n_factors)]
    )

    # Calculate variance explained
    variance_explained = {}
    for i, view_name in enumerate(data_dict.keys()):
        var_exp = ent.model.calculate_variance_explained(factors=[i], views=[i])
        variance_explained[view_name] = {
            f"Factor{j+1}": var_exp[i][j] for j in range(n_factors)
        }

    return {
        "factors": factors_df,
        "loadings": loadings,
        "variance_explained": variance_explained,
        "model": ent,
    }
```

## API Compatibility Check

Our custom implementation matches the expected interface:

```python
# Both implementations return the same structure:
results = {
    "factors": pd.DataFrame,           # (n_samples, n_factors)
    "loadings": Dict[str, pd.DataFrame],  # {view: (n_features, n_factors)}
    "variance_explained": Dict[str, Dict[str, float]],
    "model": object,
}

# Both follow embedding contract
embeddings = standardize_integration_output(results, method="mofa")
X = extract_primary_embedding(results, method="mofa")  # Shape: (n_samples, n_factors)
```

## Testing

Run smoke tests to verify MOFA implementation:

```bash
pytest tests/integration/test_mofa_smoke.py -v
```

Tests cover:
- ✓ Basic fit/transform
- ✓ Output format compliance with embedding contract
- ✓ 2+ modalities with different feature counts
- ✓ Missing data handling
- ✓ Convergence and ELBO improvement
- ✓ Reproducibility with random seed
- ✓ Variance explained calculation

## Known Limitations of Custom Implementation

Our simplified MOFA implementation does **not** support:

1. **Group/time modeling**: No multi-group MOFA or temporal dynamics
2. **Non-Gaussian likelihoods**: Only Gaussian (for count data, pre-transform)
3. **Sample/feature-level covariates**: Not implemented
4. **Advanced priors**: Simplified sparsity and ARD priors
5. **GPU acceleration**: CPU only

For these features, use official `mofapy2` with the wrapper above.

## When to Use Custom vs. Official

| Use Custom Implementation | Use Official mofapy2 |
|--------------------------|----------------------|
| Standard integration of continuous data | Multi-group studies (e.g., multiple cohorts) |
| Need scikit-learn style API | Temporal/longitudinal data |
| Want to avoid R dependency | Need advanced likelihood models (Poisson, etc.) |
| Sufficient for most use cases | Publishing with MOFA citation required |

## References

- Official MOFA paper: Argelaguet et al. (2018) *Molecular Systems Biology*
- MOFA+ paper: Argelaguet et al. (2020) *Genome Biology*
- Official GitHub: https://github.com/bioFAM/mofapy2
- Official docs: https://biofam.github.io/MOFA2/