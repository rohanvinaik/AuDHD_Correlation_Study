"""Multi-omics integration methods

All integration methods follow the standard embedding contract documented in
docs/embedding_contracts.md:

Returns: Dict[str, Any] with primary embedding as np.ndarray (n_samples, n_features)
"""
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.schema import AppConfig
from .mofa import MOFAIntegration


def integrate_omics(
    X: Dict[str, Any],
    cfg: AppConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Integrate multi-omics data using configured method

    All methods follow standard embedding contract:
    - MOFA: Returns {"factors": (n_samples, n_factors), ...}
    - Stack: Returns {"concatenated": (n_samples, sum(n_features)), ...}
    - Null: Returns {"concatenated": (n_samples, sum(n_features)), ...}

    See docs/embedding_contracts.md for full specification.

    Args:
        X: Dictionary of processed feature matrices by modality
        cfg: Application configuration
        output_dir: Optional output directory for plots

    Returns:
        Dictionary with integration results (follows embedding contract)
    """
    method = cfg.integrate.method.lower()

    if method == "mofa" or method == "mofa2":
        return _integrate_mofa(X, cfg, output_dir)
    elif method == "stack" or method == "concat" or method == "concatenate":
        return _integrate_stack(X, cfg)
    elif method == "null" or method == "baseline":
        # Null integration = scaled concatenation (baseline)
        return _integrate_null(X, cfg)
    elif method == "diablo":
        return _integrate_diablo(X, cfg)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def _integrate_mofa(
    X: Dict[str, Any],
    cfg: AppConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Integrate using MOFA2

    Args:
        X: Multi-omics data dictionary
        cfg: Application configuration
        output_dir: Output directory for plots

    Returns:
        Dictionary with latent factors and loadings
    """
    # Convert arrays to DataFrames if needed
    data_dict = {}
    for modality, data in X.items():
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        data_dict[modality] = data

    # Get view weights from config
    view_weights = cfg.integrate.weights if hasattr(cfg.integrate, "weights") else {}

    # Initialize MOFA
    n_factors = cfg.integrate.get("n_factors", 10) if hasattr(cfg.integrate, "get") else 10

    mofa = MOFAIntegration(
        n_factors=n_factors,
        view_weights=view_weights,
        sparsity_prior=True,
        ard_prior=True,
    )

    # Fit model
    mofa.fit(data_dict)

    # Generate plots if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Variance explained plot
        mofa.plot_variance_explained(output_dir / "mofa_variance_explained.png")

        # Factor correlation
        mofa.plot_factor_correlation(output_dir / "mofa_factor_correlation.png")

        # Loading heatmaps for each view
        for view in data_dict.keys():
            mofa.plot_factor_loadings(
                view,
                top_features=20,
                output_path=output_dir / f"mofa_loadings_{view}.png",
            )

    # Get results
    factors = mofa.get_latent_factors()
    loadings = mofa.get_factor_loadings()

    return {
        "factors": factors,
        "loadings": loadings,
        "variance_explained": mofa.variance_explained,
        "model": mofa,
    }


def _integrate_stack(
    X: Dict[str, Any], cfg: AppConfig
) -> Dict[str, pd.DataFrame]:
    """
    Simple concatenation-based integration

    Args:
        X: Multi-omics data dictionary
        cfg: Application configuration

    Returns:
        Dictionary with concatenated features
    """
    # Get view weights
    weights = cfg.integrate.weights if hasattr(cfg.integrate, "weights") else {}

    # Concatenate all views with optional weighting
    dfs = []
    for modality, data in X.items():
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Apply weight
        weight = weights.get(modality, 1.0)
        if weight != 1.0:
            data = data * weight

        dfs.append(data)

    # Concatenate horizontally
    concatenated = pd.concat(dfs, axis=1)

    return {
        "concatenated": concatenated,
    }


def _integrate_null(
    X: Dict[str, Any], cfg: AppConfig
) -> Dict[str, pd.DataFrame]:
    """
    Null integration: Simple scaled concatenation baseline

    This serves as a baseline and de-risks toolchain issues by providing
    a method that always works (no complex dependencies).

    Differences from _integrate_stack:
    - Always standardizes features (z-score per feature)
    - No weighting options (all modalities equal weight)
    - Explicitly documented as baseline method

    Args:
        X: Multi-omics data dictionary
        cfg: Application configuration

    Returns:
        Dictionary with {"concatenated": (n_samples, sum(n_features))}
    """
    from sklearn.preprocessing import StandardScaler

    # Standardize each modality independently
    dfs = []
    for modality, data in X.items():
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Standardize features (z-score: mean=0, std=1)
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )

        dfs.append(data_scaled)

    # Concatenate horizontally
    concatenated = pd.concat(dfs, axis=1)

    return {
        "concatenated": concatenated,
        "method": "null_baseline",
        "n_modalities": len(X),
        "total_features": concatenated.shape[1],
    }


def _integrate_diablo(
    X: Dict[str, Any], cfg: AppConfig
) -> Dict[str, pd.DataFrame]:
    """
    DIABLO-style integration (simplified)

    Args:
        X: Multi-omics data dictionary
        cfg: Application configuration

    Returns:
        Dictionary with integrated components
    """
    # Placeholder: DIABLO requires mixOmics R package
    # For now, use weighted concatenation similar to stack
    return _integrate_stack(X, cfg)


def save_integration_results(
    results: Dict[str, Any],
    output_dir: Path,
    method: str = "mofa",
) -> None:
    """
    Save integration results to disk

    Args:
        results: Integration results dictionary
        output_dir: Output directory
        method: Integration method name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "mofa":
        # Save latent factors
        if "factors" in results:
            results["factors"].to_csv(output_dir / "latent_factors.csv")

        # Save loadings
        if "loadings" in results:
            for view, loadings in results["loadings"].items():
                loadings.to_csv(output_dir / f"loadings_{view}.csv")

        # Save variance explained
        if "variance_explained" in results:
            var_exp_df = pd.DataFrame(results["variance_explained"]).T
            var_exp_df.to_csv(output_dir / "variance_explained.csv")

    elif method == "stack":
        # Save concatenated matrix
        if "concatenated" in results:
            results["concatenated"].to_csv(output_dir / "concatenated_features.csv")


def load_integration_results(
    input_dir: Path, method: str = "mofa"
) -> Dict[str, Any]:
    """
    Load integration results from disk

    Args:
        input_dir: Input directory
        method: Integration method name

    Returns:
        Dictionary with loaded results
    """
    input_dir = Path(input_dir)
    results = {}

    if method == "mofa":
        # Load latent factors
        factors_file = input_dir / "latent_factors.csv"
        if factors_file.exists():
            results["factors"] = pd.read_csv(factors_file, index_col=0)

        # Load variance explained
        var_exp_file = input_dir / "variance_explained.csv"
        if var_exp_file.exists():
            results["variance_explained"] = pd.read_csv(
                var_exp_file, index_col=0
            ).to_dict()

    elif method == "stack":
        # Load concatenated matrix
        concat_file = input_dir / "concatenated_features.csv"
        if concat_file.exists():
            results["concatenated"] = pd.read_csv(concat_file, index_col=0)

    return results


# ============================================================================
# Standardization Helpers (Embedding Contract)
# ============================================================================


def validate_embedding_shape(embedding: np.ndarray, expected_n_samples: int) -> None:
    """
    Validate embedding shape according to embedding contract

    Args:
        embedding: Embedding array to validate
        expected_n_samples: Expected number of samples

    Raises:
        AssertionError: If validation fails
    """
    assert embedding.ndim == 2, f"Embedding must be 2D, got {embedding.ndim}D"
    assert embedding.shape[0] == expected_n_samples, (
        f"Expected {expected_n_samples} samples, got {embedding.shape[0]}"
    )
    assert np.isfinite(embedding).all(), "Embedding contains NaN or Inf"


def validate_no_missing(embedding: np.ndarray) -> None:
    """
    Validate that embedding has no missing values

    Args:
        embedding: Embedding array to validate

    Raises:
        AssertionError: If validation fails
    """
    assert not np.isnan(embedding).any(), "Embedding contains NaN values"
    assert np.isfinite(embedding).all(), "Embedding contains Inf values"


def standardize_integration_output(
    results: Dict[str, Any],
    method: str
) -> Dict[str, np.ndarray]:
    """
    Convert integration results to standard embedding format

    Follows embedding contract documented in docs/embedding_contracts.md.

    Args:
        results: Raw integration results
        method: Integration method name

    Returns:
        Dictionary of {name: np.ndarray} embeddings with shape (n_samples, n_features)

    Raises:
        ValueError: If method unknown or no valid embeddings extracted
    """
    embeddings = {}

    if method in ["mofa", "mofa2"]:
        # Extract factors as primary embedding
        if "factors" in results:
            factors = results["factors"]
            embeddings["mofa_factors"] = (
                factors.values if isinstance(factors, pd.DataFrame) else factors
            )

    elif method in ["stack", "concat", "concatenate", "null", "baseline"]:
        # Extract concatenated features
        if "concatenated" in results:
            concat = results["concatenated"]
            embeddings["concatenated"] = (
                concat.values if isinstance(concat, pd.DataFrame) else concat
            )

    elif method == "group_specific":
        # Extract shared and group-specific embeddings
        if "shared" in results:
            embeddings["shared"] = results["shared"]
        if "group_specific" in results:
            for group, emb in results["group_specific"].items():
                embeddings[f"group_{group}"] = emb

    elif method == "adversarial":
        # Extract protected-removed embedding (preferred for clustering)
        if "protected_removed" in results:
            embeddings["adversarial"] = results["protected_removed"]
        elif "shared" in results:
            embeddings["adversarial"] = results["shared"]

    else:
        raise ValueError(f"Unknown integration method: {method}")

    # Validate all embeddings
    if not embeddings:
        raise ValueError(f"No valid embeddings extracted from {method} results")

    n_samples = next(iter(embeddings.values())).shape[0]
    for name, emb in embeddings.items():
        validate_embedding_shape(emb, n_samples)
        validate_no_missing(emb)

    return embeddings


def extract_primary_embedding(
    results: Dict[str, Any],
    method: str
) -> np.ndarray:
    """
    Extract primary embedding for clustering

    Convenience function that extracts the main embedding array from
    integration results.

    Args:
        results: Integration results
        method: Integration method name

    Returns:
        Primary embedding as np.ndarray with shape (n_samples, n_features)

    Raises:
        ValueError: If method unknown or primary embedding not found
    """
    standardized = standardize_integration_output(results, method)

    # Get primary embedding key based on method
    if method in ["mofa", "mofa2"]:
        primary_key = "mofa_factors"
    elif method in ["stack", "concat", "concatenate", "null", "baseline"]:
        primary_key = "concatenated"
    elif method == "group_specific":
        primary_key = "shared"
    elif method == "adversarial":
        primary_key = "adversarial"
    else:
        # Fallback: use first embedding
        primary_key = next(iter(standardized.keys()))

    if primary_key not in standardized:
        raise ValueError(
            f"Primary embedding key '{primary_key}' not found in standardized results. "
            f"Available keys: {list(standardized.keys())}"
        )

    return standardized[primary_key]