"""Multi-omics integration methods"""
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

    Args:
        X: Dictionary of processed feature matrices by modality
        cfg: Application configuration
        output_dir: Optional output directory for plots

    Returns:
        Dictionary with integration results
    """
    method = cfg.integrate.method.lower()

    if method == "mofa" or method == "mofa2":
        return _integrate_mofa(X, cfg, output_dir)
    elif method == "stack":
        return _integrate_stack(X, cfg)
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