"""Batch effect correction with ComBat and ComBat-seq"""
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from ..config.schema import AppConfig


def correct(X: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """
    Correct batch effects across all modalities

    Args:
        X: Dictionary of data matrices by modality
        cfg: Application configuration

    Returns:
        Dictionary of batch-corrected matrices
    """
    corrected = {}

    for modality, data in X.items():
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        if data.empty:
            corrected[modality] = data
            continue

        # Get batch information
        batch_info = _get_batch_info(data, cfg)

        if batch_info is None or len(batch_info["batch"].unique()) <= 1:
            # No batch correction needed
            corrected[modality] = data
            continue

        # Select correction method based on data type
        if modality == "microbiome":
            # Use ComBat-seq for count data
            corrected[modality] = combat_seq(
                data,
                batch=batch_info["batch"],
                covariates=batch_info.get("covariates"),
            )
        else:
            # Use standard ComBat for continuous data
            corrected[modality] = combat(
                data,
                batch=batch_info["batch"],
                covariates=batch_info.get("covariates"),
                preserve_variability=True,
            )

    return corrected


def combat(
    data: pd.DataFrame,
    batch: pd.Series,
    covariates: Optional[pd.DataFrame] = None,
    preserve_variability: bool = True,
    parametric: bool = True,
    mean_only: bool = False,
) -> pd.DataFrame:
    """
    ComBat batch correction for continuous data

    Args:
        data: DataFrame with samples as rows, features as columns
        batch: Series with batch labels for each sample
        covariates: Optional DataFrame with covariates to preserve
        preserve_variability: Preserve biological variability
        parametric: Use parametric adjustments
        mean_only: Only adjust batch means (not variance)

    Returns:
        Batch-corrected DataFrame
    """
    # Ensure batch and data have matching indices
    batch = batch.loc[data.index]

    if covariates is not None:
        covariates = covariates.loc[data.index]

    # Get batch IDs
    batch_ids = batch.unique()
    n_batch = len(batch_ids)

    if n_batch <= 1:
        return data

    # Transpose to features × samples (ComBat convention)
    data_t = data.T.values.astype(float)
    n_features, n_samples = data_t.shape

    # Build design matrix
    design = _create_design_matrix(batch, covariates)

    # Standardize data across features
    B_hat = solve(design.T @ design, design.T @ data_t.T).T
    grand_mean = B_hat @ design.T

    # Remove covariate effects to get residuals
    if covariates is not None:
        var_pooled = ((data_t - grand_mean) ** 2).sum(axis=1) / (n_samples - design.shape[1])
    else:
        var_pooled = data_t.var(axis=1, ddof=1)

    stand_mean = (data_t - grand_mean) / np.sqrt(var_pooled[:, None])

    # Get batch effect parameters
    batch_design = _get_batch_design(batch)
    n_batches = batch_design.shape[1]

    gamma_hat = solve(batch_design.T @ batch_design, batch_design.T @ stand_mean.T).T
    delta_hat = []

    for i, batch_id in enumerate(batch_ids):
        batch_mask = batch == batch_id
        delta_hat.append(stand_mean[:, batch_mask].var(axis=1, ddof=1))

    delta_hat = np.array(delta_hat).T

    # Empirical Bayes to get batch effect parameters
    if parametric:
        gamma_star, delta_star = _empirical_bayes_parametric(
            gamma_hat, delta_hat, batch_design, stand_mean
        )
    else:
        gamma_star, delta_star = _empirical_bayes_nonparametric(
            gamma_hat, delta_hat, batch_design, stand_mean
        )

    # Adjust data
    bayesdata = np.zeros_like(data_t)

    for i, batch_id in enumerate(batch_ids):
        batch_mask = batch == batch_id
        batch_idxs = np.where(batch_mask)[0]

        if mean_only:
            bayesdata[:, batch_idxs] = (
                data_t[:, batch_idxs] - grand_mean[:, batch_idxs]
            ) - gamma_star[:, i][:, None]
        else:
            bayesdata[:, batch_idxs] = (
                (data_t[:, batch_idxs] - grand_mean[:, batch_idxs])
                - gamma_star[:, i][:, None]
            ) / np.sqrt(delta_star[:, i][:, None])

        # Add back overall mean
        if not mean_only:
            bayesdata[:, batch_idxs] = (
                bayesdata[:, batch_idxs] * np.sqrt(var_pooled)[:, None]
            )

        # Add back covariate effects
        if covariates is not None:
            covariate_effects = B_hat[:, : covariates.shape[1]] @ covariates.T.values[:, batch_idxs]
            bayesdata[:, batch_idxs] += covariate_effects
        else:
            bayesdata[:, batch_idxs] += B_hat[:, 0][:, None]

    # Transpose back to samples × features
    corrected = pd.DataFrame(
        bayesdata.T, index=data.index, columns=data.columns
    )

    return corrected


def combat_seq(
    data: pd.DataFrame,
    batch: pd.Series,
    covariates: Optional[pd.DataFrame] = None,
    full_mod: bool = True,
) -> pd.DataFrame:
    """
    ComBat-seq for count data (RNA-seq, microbiome)

    Args:
        data: DataFrame with count data (samples × features)
        batch: Batch labels
        covariates: Covariates to preserve
        full_mod: Use full model (vs null model)

    Returns:
        Batch-corrected count data
    """
    # Ensure batch and data have matching indices
    batch = batch.loc[data.index]

    if covariates is not None:
        covariates = covariates.loc[data.index]

    batch_ids = batch.unique()
    n_batch = len(batch_ids)

    if n_batch <= 1:
        return data

    # Convert to array
    counts = data.values.astype(float)
    n_samples, n_features = counts.shape

    # Filter lowly expressed features
    min_count = 10
    feature_mask = counts.sum(axis=0) >= min_count
    counts_filt = counts[:, feature_mask]

    # Log transform with pseudocount
    log_counts = np.log2(counts_filt + 1)

    # Use ComBat on log-transformed counts
    log_corrected = combat(
        pd.DataFrame(log_counts, index=data.index),
        batch=batch,
        covariates=covariates,
        preserve_variability=True,
        parametric=True,
    )

    # Back-transform to counts
    corrected_counts = 2 ** log_corrected.values - 1
    corrected_counts = np.maximum(corrected_counts, 0)

    # Put back filtered features (unchanged)
    corrected_full = counts.copy()
    corrected_full[:, feature_mask] = corrected_counts

    return pd.DataFrame(corrected_full, index=data.index, columns=data.columns)


def _create_design_matrix(
    batch: pd.Series, covariates: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """Create design matrix for batch correction"""
    n_samples = len(batch)

    if covariates is not None:
        # Standardize covariates
        covar_std = (covariates - covariates.mean()) / covariates.std()
        design = np.column_stack([np.ones(n_samples), covar_std.values])
    else:
        design = np.ones((n_samples, 1))

    return design


def _get_batch_design(batch: pd.Series) -> np.ndarray:
    """Create batch design matrix"""
    batch_ids = batch.unique()
    n_samples = len(batch)
    n_batches = len(batch_ids)

    batch_design = np.zeros((n_samples, n_batches))

    for i, batch_id in enumerate(batch_ids):
        batch_design[:, i] = (batch == batch_id).astype(int)

    return batch_design


def _empirical_bayes_parametric(
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    batch_design: np.ndarray,
    stand_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parametric empirical Bayes estimation

    Args:
        gamma_hat: Estimated batch effects (features × batches)
        delta_hat: Estimated batch variances (features × batches)
        batch_design: Batch design matrix
        stand_mean: Standardized data

    Returns:
        Tuple of (gamma_star, delta_star)
    """
    n_features, n_batches = gamma_hat.shape

    # Estimate hyperparameters for gamma (mean)
    gamma_bar = gamma_hat.mean(axis=0)
    tau_bar = gamma_hat.var(axis=0, ddof=1)

    # Estimate hyperparameters for delta (variance)
    # Use method of moments
    delta_bar = delta_hat.mean(axis=0)

    # Inverse gamma parameters for variance
    # a_prior and b_prior estimation
    delta_mean = delta_hat.mean(axis=0)
    delta_var = delta_hat.var(axis=0, ddof=1)

    a_prior = (2 * delta_mean ** 2 + delta_var) / delta_var
    b_prior = (delta_mean * delta_var + delta_mean ** 3) / delta_var

    # Empirical Bayes shrinkage
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)

    for i in range(n_batches):
        # Get samples in this batch
        batch_samples = batch_design[:, i] == 1
        n_batch = batch_samples.sum()

        # Shrink gamma (additive batch effects)
        post_var = 1 / (1 / tau_bar[i] + n_batch / delta_hat[:, i])
        post_mean = post_var * (
            gamma_bar[i] / tau_bar[i] + n_batch * gamma_hat[:, i] / delta_hat[:, i]
        )
        gamma_star[:, i] = post_mean

        # Shrink delta (multiplicative batch effects)
        post_a = a_prior[i] + n_batch / 2
        post_b = b_prior[i] + 0.5 * ((stand_mean[:, batch_samples] - gamma_star[:, i][:, None]) ** 2).sum(axis=1)
        delta_star[:, i] = post_b / (post_a - 1)

    return gamma_star, delta_star


def _empirical_bayes_nonparametric(
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    batch_design: np.ndarray,
    stand_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-parametric empirical Bayes estimation

    Args:
        gamma_hat: Estimated batch effects
        delta_hat: Estimated batch variances
        batch_design: Batch design matrix
        stand_mean: Standardized data

    Returns:
        Tuple of (gamma_star, delta_star)
    """
    # Use kernel density estimation for prior
    # Simplified version: use robust statistics
    n_features, n_batches = gamma_hat.shape

    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)

    for i in range(n_batches):
        # Robust shrinkage using median
        gamma_star[:, i] = 0.5 * gamma_hat[:, i] + 0.5 * np.median(gamma_hat, axis=0)[i]
        delta_star[:, i] = 0.5 * delta_hat[:, i] + 0.5 * np.median(delta_hat, axis=0)[i]

    return gamma_star, delta_star


def _get_batch_info(data: pd.DataFrame, cfg: AppConfig) -> Optional[Dict[str, Any]]:
    """
    Extract batch information from data and config

    Args:
        data: Data DataFrame
        cfg: Application configuration

    Returns:
        Dictionary with batch and covariate information
    """
    # Try to get batch info from index or attributes
    batch = None

    # Check if data has metadata attribute
    if hasattr(data, "attrs") and "batch" in data.attrs:
        batch = pd.Series(data.attrs["batch"], index=data.index)
    elif "batch" in data.columns:
        batch = data["batch"]
        data = data.drop("batch", axis=1)

    # Get covariates to preserve
    covariates = None
    preserve_cols = ["age", "sex", "diagnosis", "site"]

    if hasattr(cfg, "preprocess") and hasattr(cfg.preprocess, "preserve_covariates"):
        preserve_cols = cfg.preprocess.preserve_covariates

    # Extract covariates if they exist
    covar_cols = [col for col in preserve_cols if col in data.columns]
    if covar_cols:
        covariates = data[covar_cols]
        data = data.drop(covar_cols, axis=1)

    if batch is None:
        return None

    return {
        "batch": batch,
        "covariates": covariates,
    }


class BatchCorrectionPipeline:
    """Comprehensive batch correction pipeline with diagnostics"""

    def __init__(
        self,
        method: str = "combat",
        preserve_covariates: Optional[List[str]] = None,
        generate_plots: bool = True,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize batch correction pipeline

        Args:
            method: Correction method ('combat', 'combat_seq', 'mixed_effects')
            preserve_covariates: List of covariates to preserve
            generate_plots: Whether to generate diagnostic plots
            output_dir: Directory for output plots and reports
        """
        self.method = method
        self.preserve_covariates = preserve_covariates or ["age", "sex", "diagnosis"]
        self.generate_plots = generate_plots
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/batch_correction")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def correct_data(
        self,
        data: pd.DataFrame,
        batch: pd.Series,
        covariates: Optional[pd.DataFrame] = None,
        modality: str = "unknown",
    ) -> pd.DataFrame:
        """
        Correct batch effects with diagnostics

        Args:
            data: Data to correct
            batch: Batch labels
            covariates: Covariates to preserve
            modality: Data modality name

        Returns:
            Corrected data
        """
        # Generate before plots
        if self.generate_plots:
            self._plot_before_correction(data, batch, modality)

        # Apply correction
        if self.method == "combat":
            corrected = combat(data, batch, covariates)
        elif self.method == "combat_seq":
            corrected = combat_seq(data, batch, covariates)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Generate after plots
        if self.generate_plots:
            self._plot_after_correction(corrected, batch, modality)
            self._plot_comparison(data, corrected, batch, modality)

        return corrected

    def _plot_before_correction(
        self, data: pd.DataFrame, batch: pd.Series, modality: str
    ) -> None:
        """Generate diagnostic plots before correction"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PCA plot
        self._plot_pca(data, batch, axes[0], title=f"{modality} - Before Correction")

        # Density plot
        self._plot_density(data, batch, axes[1], title=f"{modality} - Before")

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{modality}_before_correction.png", dpi=300)
        plt.close()

    def _plot_after_correction(
        self, data: pd.DataFrame, batch: pd.Series, modality: str
    ) -> None:
        """Generate diagnostic plots after correction"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PCA plot
        self._plot_pca(data, batch, axes[0], title=f"{modality} - After Correction")

        # Density plot
        self._plot_density(data, batch, axes[1], title=f"{modality} - After")

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{modality}_after_correction.png", dpi=300)
        plt.close()

    def _plot_comparison(
        self,
        data_before: pd.DataFrame,
        data_after: pd.DataFrame,
        batch: pd.Series,
        modality: str,
    ) -> None:
        """Generate before/after comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Before PCA
        self._plot_pca(data_before, batch, axes[0, 0], title="Before - PCA")

        # After PCA
        self._plot_pca(data_after, batch, axes[0, 1], title="After - PCA")

        # Before density
        self._plot_density(data_before, batch, axes[1, 0], title="Before - Density")

        # After density
        self._plot_density(data_after, batch, axes[1, 1], title="After - Density")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{modality}_before_after_comparison.png", dpi=300
        )
        plt.close()

    def _plot_pca(
        self, data: pd.DataFrame, batch: pd.Series, ax, title: str = "PCA"
    ) -> None:
        """Plot PCA colored by batch"""
        # Compute PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data.fillna(0))

        # Plot
        batch_ids = batch.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(batch_ids)))

        for i, batch_id in enumerate(batch_ids):
            mask = batch == batch_id
            ax.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                c=[colors[i]],
                label=f"Batch {batch_id}",
                alpha=0.6,
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(title)
        ax.legend()

    def _plot_density(
        self, data: pd.DataFrame, batch: pd.Series, ax, title: str = "Density"
    ) -> None:
        """Plot feature density distributions by batch"""
        batch_ids = batch.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(batch_ids)))

        for i, batch_id in enumerate(batch_ids):
            mask = batch == batch_id
            batch_data = data.loc[mask].values.flatten()
            batch_data = batch_data[~np.isnan(batch_data)]

            ax.hist(
                batch_data,
                bins=50,
                alpha=0.5,
                label=f"Batch {batch_id}",
                color=colors[i],
                density=True,
            )

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()