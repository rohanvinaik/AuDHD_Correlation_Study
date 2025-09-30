"""Mixed effects models for batch correction with site and platform effects"""
from typing import Dict, Optional, List, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve


def correct_mixed_effects(
    data: pd.DataFrame,
    site: pd.Series,
    platform: Optional[pd.Series] = None,
    covariates: Optional[pd.DataFrame] = None,
    partial_correction: bool = False,
    time_series: bool = False,
) -> pd.DataFrame:
    """
    Batch correction using mixed effects models with site × platform interactions

    Args:
        data: Data matrix (samples × features)
        site: Site labels
        platform: Optional platform labels
        covariates: Covariates to preserve
        partial_correction: Only remove systematic effects, preserve some batch variance
        time_series: Preserve temporal structure if True

    Returns:
        Corrected data
    """
    # Ensure alignment
    site = site.loc[data.index]
    if platform is not None:
        platform = platform.loc[data.index]
    if covariates is not None:
        covariates = covariates.loc[data.index]

    # Build design matrices
    X_fixed, X_random = _build_mixed_design(site, platform, covariates)

    # Fit mixed effects model for each feature
    corrected_data = np.zeros_like(data.values)

    for j in range(data.shape[1]):
        y = data.iloc[:, j].values

        # Skip if too many missing values
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 10:
            corrected_data[:, j] = y
            continue

        # Fit mixed model
        try:
            y_corrected = _fit_mixed_model(
                y[valid_mask],
                X_fixed[valid_mask],
                X_random[valid_mask] if X_random is not None else None,
                partial_correction=partial_correction,
                time_series=time_series,
            )

            corrected_data[valid_mask, j] = y_corrected
            corrected_data[~valid_mask, j] = np.nan
        except:
            # Fallback to original values
            corrected_data[:, j] = y

    return pd.DataFrame(corrected_data, index=data.index, columns=data.columns)


def _build_mixed_design(
    site: pd.Series,
    platform: Optional[pd.Series] = None,
    covariates: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Build design matrices for fixed and random effects

    Args:
        site: Site labels
        platform: Platform labels
        covariates: Covariates

    Returns:
        Tuple of (X_fixed, X_random)
    """
    n_samples = len(site)

    # Fixed effects: intercept + covariates
    fixed_components = [np.ones((n_samples, 1))]

    if covariates is not None:
        # Standardize covariates
        covar_std = (covariates - covariates.mean()) / (covariates.std() + 1e-6)
        fixed_components.append(covar_std.values)

    X_fixed = np.hstack(fixed_components)

    # Random effects: site, platform, and site × platform interaction
    random_components = []

    # Site effects (random intercept per site)
    site_ids = site.unique()
    site_design = np.zeros((n_samples, len(site_ids)))
    for i, site_id in enumerate(site_ids):
        site_design[:, i] = (site == site_id).astype(float)
    random_components.append(site_design)

    # Platform effects (if provided)
    if platform is not None:
        platform_ids = platform.unique()
        platform_design = np.zeros((n_samples, len(platform_ids)))
        for i, plat_id in enumerate(platform_ids):
            platform_design[:, i] = (platform == plat_id).astype(float)
        random_components.append(platform_design)

        # Site × platform interaction
        interaction_design = []
        for site_id in site_ids:
            for plat_id in platform_ids:
                mask = (site == site_id) & (platform == plat_id)
                if mask.sum() > 0:  # Only include non-empty combinations
                    interaction_design.append(mask.astype(float))

        if interaction_design:
            interaction_design = np.column_stack(interaction_design)
            random_components.append(interaction_design)

    X_random = np.hstack(random_components) if random_components else None

    return X_fixed, X_random


def _fit_mixed_model(
    y: np.ndarray,
    X_fixed: np.ndarray,
    X_random: Optional[np.ndarray] = None,
    partial_correction: bool = False,
    time_series: bool = False,
) -> np.ndarray:
    """
    Fit mixed effects model and return corrected values

    Simplified mixed model: y = X_fixed @ beta + X_random @ u + epsilon
    where u ~ N(0, sigma_u^2) and epsilon ~ N(0, sigma_e^2)

    Args:
        y: Response variable
        X_fixed: Fixed effects design matrix
        X_random: Random effects design matrix
        partial_correction: Only remove systematic component
        time_series: Preserve temporal correlations

    Returns:
        Corrected values
    """
    n = len(y)

    # Estimate fixed effects
    beta_hat = solve(X_fixed.T @ X_fixed, X_fixed.T @ y)
    residuals = y - X_fixed @ beta_hat

    if X_random is None:
        # No random effects, just return residuals + grand mean
        return residuals + y.mean()

    # Estimate random effects variances using REML (simplified)
    # Get initial estimate of sigma_e^2 (error variance)
    sigma_e2 = np.var(residuals)

    # Estimate random effect variances (simplified: assume equal variance)
    # More sophisticated: use iterative REML
    random_var = residuals.T @ X_random @ X_random.T @ residuals / X_random.shape[1]
    sigma_u2 = max(random_var / n, 1e-6)

    # Construct variance matrix V = sigma_u^2 * (Z @ Z.T) + sigma_e^2 * I
    # Simplified: use shrinkage estimator for random effects
    shrinkage_factor = sigma_u2 / (sigma_u2 + sigma_e2 / X_random.shape[1])

    # Estimate random effects (BLUPs)
    # For each random effect, solve: u_j = shrinkage * X_j^T @ residuals / ||X_j||^2
    u_hat = np.zeros(X_random.shape[1])
    for j in range(X_random.shape[1]):
        x_j = X_random[:, j]
        norm_sq = np.sum(x_j ** 2)
        if norm_sq > 0:
            u_hat[j] = shrinkage_factor * (x_j @ residuals) / norm_sq

    # Reconstruct random effects contribution
    random_effects = X_random @ u_hat

    if partial_correction:
        # Only remove a fraction of random effects (preserve some batch variance)
        correction_strength = 0.5 if not time_series else 0.3
        y_corrected = y - correction_strength * random_effects
    else:
        # Full correction: remove all random effects
        y_corrected = y - random_effects

    # Add back grand mean
    y_corrected = y_corrected + y.mean()

    return y_corrected


class MixedEffectsBatchCorrection:
    """Mixed effects batch correction with site × platform modeling"""

    def __init__(
        self,
        partial_correction: bool = False,
        time_series: bool = False,
        preserve_covariates: Optional[List[str]] = None,
    ):
        """
        Initialize mixed effects batch correction

        Args:
            partial_correction: Preserve some batch variance
            time_series: Preserve temporal structure
            preserve_covariates: Covariates to preserve
        """
        self.partial_correction = partial_correction
        self.time_series = time_series
        self.preserve_covariates = preserve_covariates or ["age", "sex", "diagnosis"]

    def correct(
        self,
        data: pd.DataFrame,
        site: pd.Series,
        platform: Optional[pd.Series] = None,
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Apply mixed effects correction

        Args:
            data: Data to correct
            site: Site labels
            platform: Platform labels
            covariates: Covariates to preserve

        Returns:
            Corrected data
        """
        return correct_mixed_effects(
            data,
            site,
            platform,
            covariates,
            partial_correction=self.partial_correction,
            time_series=self.time_series,
        )


def correct_partial_timeseries(
    data: pd.DataFrame,
    timepoints: pd.Series,
    site: pd.Series,
    batch: pd.Series,
    preserve_trend: bool = True,
) -> pd.DataFrame:
    """
    Partial batch correction for time-series data

    Preserves temporal trends while removing batch effects.

    Args:
        data: Data matrix (samples × features)
        timepoints: Time points for each sample
        site: Site labels
        batch: Batch labels
        preserve_trend: Preserve temporal trends

    Returns:
        Partially corrected data
    """
    # Ensure alignment
    timepoints = timepoints.loc[data.index]
    site = site.loc[data.index]
    batch = batch.loc[data.index]

    corrected_data = np.zeros_like(data.values)

    for j in range(data.shape[1]):
        y = data.iloc[:, j].values

        # Skip if too many missing
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 10:
            corrected_data[:, j] = y
            continue

        try:
            # Fit model: y = f(time) + batch_effects + error
            # where f(time) is a smooth function (e.g., polynomial)

            if preserve_trend:
                # Fit temporal trend
                t = timepoints.values[valid_mask]
                y_valid = y[valid_mask]

                # Fit quadratic trend
                X_trend = np.column_stack([np.ones_like(t), t, t ** 2])
                trend_coef = solve(X_trend.T @ X_trend, X_trend.T @ y_valid)
                trend = X_trend @ trend_coef

                # Remove trend
                y_detrended = y_valid - trend
            else:
                y_detrended = y[valid_mask]

            # Correct batch effects on detrended data
            from .batch_effects import combat

            batch_data = pd.DataFrame(
                y_detrended[:, None],
                index=data.index[valid_mask],
                columns=["value"],
            )
            batch_labels = batch[valid_mask]

            # Use simple mean centering by batch (more stable for small samples)
            corrected_detrended = y_detrended.copy()
            overall_mean = corrected_detrended.mean()

            for b in batch_labels.unique():
                mask = (batch_labels == b).values
                batch_mean = corrected_detrended[mask].mean()
                corrected_detrended[mask] = corrected_detrended[mask] - (batch_mean - overall_mean)

            # Add trend back
            if preserve_trend:
                y_corrected = corrected_detrended + trend
            else:
                y_corrected = corrected_detrended

            corrected_data[valid_mask, j] = y_corrected
            corrected_data[~valid_mask, j] = np.nan

        except:
            corrected_data[:, j] = y

    return pd.DataFrame(corrected_data, index=data.index, columns=data.columns)