#!/usr/bin/env python3
"""
Covariate Adjustment with Mixed Effects Models

Implements mixed effects regression with convergence fallbacks as specified
in configs/preprocessing/PREPROCESSING_ORDER.md.

Fallback chain:
1. Mixed effects with REML
2. Mixed effects with BFGS
3. Linear regression (no random effects)
4. No adjustment (raw values)
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_mixed_effects_with_fallback(
    feature_values: np.ndarray,
    fixed_effects: pd.DataFrame,
    random_effects: pd.Series,
    feature_name: str,
    max_iter: int = 100,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Fit mixed effects model with convergence fallbacks

    Fallback chain (from PREPROCESSING_ORDER.md):
    1. Try REML → BFGS → Nelder-Mead optimizers
    2. If all fail → Linear regression (no random effects)
    3. If that fails → No adjustment (log warning)

    Args:
        feature_values: Target values to adjust (n_samples,)
        fixed_effects: Fixed effect covariates to regress out (n_samples, n_covariates)
        random_effects: Random effect grouping (e.g., subject_id) (n_samples,)
        feature_name: Name of feature (for logging)
        max_iter: Maximum iterations for optimization
        random_state: Random seed for reproducibility

    Returns:
        residuals: Adjusted values with covariates removed (n_samples,)
        metadata: Dict with keys 'method', 'converged', 'message'
    """
    # Set random seed
    np.random.seed(random_state)

    # Try mixed effects model
    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM

        # Try REML first (default)
        md = MixedLM(
            endog=feature_values,
            exog=fixed_effects,
            groups=random_effects
        )

        try:
            result = md.fit(method='lbfgs', maxiter=max_iter, reml=True)
            if result.converged:
                return result.resid, {
                    'method': 'mixed_effects_reml',
                    'converged': 'true',
                    'message': 'REML converged successfully'
                }
        except Exception as e:
            warnings.warn(f"REML failed for {feature_name}: {e}")

        # Fallback 1: Try BFGS
        try:
            result = md.fit(method='bfgs', maxiter=max_iter)
            if result.converged:
                return result.resid, {
                    'method': 'mixed_effects_bfgs',
                    'converged': 'true',
                    'message': 'BFGS converged successfully'
                }
            else:
                warnings.warn(
                    f"BFGS did not converge for {feature_name}, trying linear regression"
                )
        except Exception as e:
            warnings.warn(f"BFGS failed for {feature_name}: {e}")

        # Fallback 2: Try Nelder-Mead
        try:
            result = md.fit(method='nm', maxiter=max_iter)
            if result.converged:
                return result.resid, {
                    'method': 'mixed_effects_nm',
                    'converged': 'true',
                    'message': 'Nelder-Mead converged successfully'
                }
            else:
                warnings.warn(
                    f"Nelder-Mead did not converge for {feature_name}, "
                    f"falling back to linear regression"
                )
                # Use the result anyway but mark as not converged
                return result.resid, {
                    'method': 'mixed_effects_nm',
                    'converged': 'false',
                    'message': 'Nelder-Mead did not converge, using final iteration'
                }
        except Exception as e:
            warnings.warn(f"Nelder-Mead failed for {feature_name}: {e}")

    except ImportError:
        warnings.warn(
            "statsmodels not available, falling back to linear regression"
        )
    except Exception as e:
        warnings.warn(
            f"Mixed effects model failed for {feature_name}: {e}, "
            f"falling back to linear regression"
        )

    # Fallback 3: Linear regression (no random effects)
    try:
        lr = LinearRegression()
        lr.fit(fixed_effects, feature_values)
        residuals = feature_values - lr.predict(fixed_effects)
        return residuals, {
            'method': 'linear_regression',
            'converged': 'true',
            'message': 'Linear regression fallback (no random effects)'
        }
    except Exception as e:
        warnings.warn(
            f"Linear regression failed for {feature_name}: {e}. "
            f"Using raw values (no adjustment)."
        )

    # Fallback 4: No adjustment
    return feature_values, {
        'method': 'none',
        'converged': 'false',
        'message': 'All adjustment methods failed, using raw values'
    }


def adjust_data_covariates(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    covariate_cols: List[str],
    random_effect_col: Optional[str] = None,
    max_iter: int = 100,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adjust data for covariates using mixed effects models with fallbacks

    Args:
        data: Feature matrix (n_samples, n_features)
        metadata: Sample metadata with covariates (n_samples, n_metadata)
        covariate_cols: List of covariate column names to regress out
        random_effect_col: Column name for random effect grouping (e.g., 'subject_id')
                          If None, uses linear regression only
        max_iter: Maximum iterations for optimization
        random_state: Random seed for reproducibility
        verbose: Print progress

    Returns:
        adjusted_data: Data with covariates removed (n_samples, n_features)
        adjustment_log: Log of adjustment method used per feature
    """
    # Validate inputs
    if not all(col in metadata.columns for col in covariate_cols):
        missing = [col for col in covariate_cols if col not in metadata.columns]
        raise ValueError(f"Covariate columns not found in metadata: {missing}")

    if random_effect_col is not None and random_effect_col not in metadata.columns:
        raise ValueError(f"Random effect column not found in metadata: {random_effect_col}")

    # Align data and metadata
    common_samples = data.index.intersection(metadata.index)
    if len(common_samples) == 0:
        raise ValueError("No common samples between data and metadata")

    data = data.loc[common_samples]
    metadata = metadata.loc[common_samples]

    # Prepare fixed effects matrix
    fixed_effects = metadata[covariate_cols].copy()

    # Handle categorical variables with one-hot encoding
    for col in covariate_cols:
        if fixed_effects[col].dtype == 'object' or fixed_effects[col].dtype.name == 'category':
            # One-hot encode, drop first to avoid collinearity
            dummies = pd.get_dummies(fixed_effects[col], prefix=col, drop_first=True)
            fixed_effects = pd.concat([fixed_effects.drop(col, axis=1), dummies], axis=1)

    # Add intercept
    fixed_effects.insert(0, 'intercept', 1.0)

    # Prepare random effects
    if random_effect_col is not None:
        random_effects = metadata[random_effect_col]
    else:
        random_effects = None

    # Adjust each feature
    adjusted_data = pd.DataFrame(
        index=data.index,
        columns=data.columns,
        dtype=float
    )
    adjustment_log = []

    n_features = len(data.columns)
    for i, feature_name in enumerate(data.columns):
        if verbose and (i + 1) % 100 == 0:
            print(f"Adjusting feature {i+1}/{n_features}...")

        feature_values = data[feature_name].values

        # Check for missing values
        if np.isnan(feature_values).any():
            warnings.warn(
                f"Feature {feature_name} has missing values. "
                f"Imputation should be performed before covariate adjustment."
            )
            adjusted_data[feature_name] = feature_values
            adjustment_log.append({
                'feature': feature_name,
                'method': 'none',
                'converged': 'false',
                'message': 'Feature has missing values'
            })
            continue

        # Fit model with fallbacks
        if random_effect_col is not None:
            residuals, metadata_dict = fit_mixed_effects_with_fallback(
                feature_values=feature_values,
                fixed_effects=fixed_effects,
                random_effects=random_effects,
                feature_name=feature_name,
                max_iter=max_iter,
                random_state=random_state
            )
        else:
            # Linear regression only (no random effects)
            try:
                lr = LinearRegression()
                lr.fit(fixed_effects, feature_values)
                residuals = feature_values - lr.predict(fixed_effects)
                metadata_dict = {
                    'method': 'linear_regression',
                    'converged': 'true',
                    'message': 'Linear regression (no random effects specified)'
                }
            except Exception as e:
                warnings.warn(f"Linear regression failed for {feature_name}: {e}")
                residuals = feature_values
                metadata_dict = {
                    'method': 'none',
                    'converged': 'false',
                    'message': f'Linear regression failed: {e}'
                }

        adjusted_data[feature_name] = residuals
        adjustment_log.append({
            'feature': feature_name,
            **metadata_dict
        })

    adjustment_log_df = pd.DataFrame(adjustment_log)

    if verbose:
        print("\n" + "="*60)
        print("COVARIATE ADJUSTMENT SUMMARY")
        print("="*60)
        print(f"Total features: {n_features}")
        print("\nMethods used:")
        print(adjustment_log_df['method'].value_counts())
        print("\nConvergence:")
        print(adjustment_log_df['converged'].value_counts())
        print("="*60)

    return adjusted_data, adjustment_log_df


def validate_covariate_orthogonality(
    adjusted_data: pd.DataFrame,
    metadata: pd.DataFrame,
    covariate_cols: List[str],
    alpha: float = 0.05
) -> Dict[str, bool]:
    """
    Validate that adjusted data is orthogonal to covariates

    Tests that covariates no longer explain significant variance in adjusted data.

    Args:
        adjusted_data: Data after covariate adjustment (n_samples, n_features)
        metadata: Sample metadata (n_samples, n_metadata)
        covariate_cols: List of covariate columns that were adjusted
        alpha: Significance level for statistical tests

    Returns:
        Dict with validation results
    """
    from scipy.stats import pearsonr, f_oneway

    results = {
        'all_orthogonal': True,
        'feature_failures': [],
        'summary': {}
    }

    # For each covariate
    for covariate in covariate_cols:
        cov_values = metadata[covariate]
        n_significant = 0

        # Test each feature
        for feature in adjusted_data.columns:
            feature_values = adjusted_data[feature].values

            # Remove NaN
            mask = ~np.isnan(feature_values) & ~pd.isna(cov_values)
            if mask.sum() < 3:
                continue

            feature_clean = feature_values[mask]
            cov_clean = cov_values[mask]

            # Test based on covariate type
            if cov_clean.dtype in ['object', 'category'] or len(np.unique(cov_clean)) < 10:
                # Categorical: Use ANOVA
                groups = [feature_clean[cov_clean == val] for val in np.unique(cov_clean)]
                if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                    _, p_value = f_oneway(*groups)
                else:
                    continue
            else:
                # Continuous: Use correlation
                r, p_value = pearsonr(feature_clean, cov_clean)

            # Check if still significant
            if p_value < alpha:
                n_significant += 1
                results['feature_failures'].append({
                    'feature': feature,
                    'covariate': covariate,
                    'p_value': p_value
                })

        # Summary for this covariate
        pct_orthogonal = (1 - n_significant / len(adjusted_data.columns)) * 100
        results['summary'][covariate] = {
            'n_significant': n_significant,
            'n_total': len(adjusted_data.columns),
            'pct_orthogonal': pct_orthogonal
        }

        # Expect ≥95% of features to be orthogonal
        if pct_orthogonal < 95:
            results['all_orthogonal'] = False

    return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulate data
    n_samples = 100
    n_features = 50

    # Create synthetic data with covariate effects
    age = np.random.uniform(20, 70, n_samples)
    sex = np.random.choice(['M', 'F'], n_samples)
    subject_id = np.repeat(range(20), 5)  # 20 subjects, 5 timepoints each

    # Features correlated with age and sex
    data = np.random.randn(n_samples, n_features)
    for i in range(n_features):
        data[:, i] += 0.5 * (age - 45) / 10  # Age effect
        data[:, i] += 0.3 * (sex == 'M')     # Sex effect
        data[:, i] += np.random.randn(20)[subject_id] * 0.2  # Random subject effect

    # Create DataFrames
    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    feature_ids = [f"FEATURE_{i:03d}" for i in range(n_features)]

    df_data = pd.DataFrame(data, index=sample_ids, columns=feature_ids)
    df_metadata = pd.DataFrame({
        'sample_id': sample_ids,
        'age': age,
        'sex': sex,
        'subject_id': subject_id
    }, index=sample_ids)

    # Adjust for covariates
    print("Adjusting for age and sex with random subject effects...")
    adjusted_data, adjustment_log = adjust_data_covariates(
        data=df_data,
        metadata=df_metadata,
        covariate_cols=['age', 'sex'],
        random_effect_col='subject_id',
        verbose=True
    )

    print(f"\nAdjustment log:\n{adjustment_log}")

    # Validate orthogonality
    print("\nValidating covariate orthogonality...")
    validation = validate_covariate_orthogonality(
        adjusted_data=adjusted_data,
        metadata=df_metadata,
        covariate_cols=['age', 'sex']
    )

    print("\nOrthogonality summary:")
    for covariate, summary in validation['summary'].items():
        print(f"  {covariate}: {summary['pct_orthogonal']:.1f}% orthogonal")

    if validation['all_orthogonal']:
        print("\n✓ All covariates successfully removed")
    else:
        print(f"\n✗ Some covariates still significant (n={len(validation['feature_failures'])})")