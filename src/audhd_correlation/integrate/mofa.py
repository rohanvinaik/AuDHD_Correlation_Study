"""MOFA2-based multi-omics integration"""
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class MOFAIntegration:
    """
    Multi-Omics Factor Analysis (MOFA2-style) integration

    Implements a simplified version of MOFA2 for multi-omics data integration.
    Decomposes multi-view data into shared latent factors while handling
    missing data and accounting for view-specific variance.
    """

    def __init__(
        self,
        n_factors: int = 10,
        n_iterations: int = 1000,
        convergence_threshold: float = 1e-5,
        view_weights: Optional[Dict[str, float]] = None,
        sparsity_prior: bool = True,
        ard_prior: bool = True,
    ):
        """
        Initialize MOFA integration

        Args:
            n_factors: Number of latent factors to extract
            n_iterations: Maximum iterations for optimization
            convergence_threshold: Convergence threshold for ELBO
            view_weights: Optional weights for different omics views
            sparsity_prior: Use sparse priors on factor loadings
            ard_prior: Use Automatic Relevance Determination
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.view_weights = view_weights or {}
        self.sparsity_prior = sparsity_prior
        self.ard_prior = ard_prior

        # Model parameters (fitted)
        self.Z: Optional[np.ndarray] = None  # Latent factors (samples × factors)
        self.W: Dict[str, np.ndarray] = {}  # Factor loadings (features × factors per view)
        self.tau: Dict[str, np.ndarray] = {}  # Noise precision per view
        self.alpha: Optional[np.ndarray] = None  # ARD precision per factor

        # Metadata
        self.views: List[str] = []
        self.sample_ids: List[str] = []
        self.feature_ids: Dict[str, List[str]] = {}
        self.n_samples: int = 0
        self.n_features: Dict[str, int] = {}

        # Training history
        self.elbo_history: List[float] = []
        self.variance_explained: Dict[str, Dict[str, float]] = {}

    def fit(
        self,
        data_dict: Dict[str, pd.DataFrame],
        group_labels: Optional[pd.Series] = None,
    ) -> "MOFAIntegration":
        """
        Fit MOFA model to multi-omics data

        Args:
            data_dict: Dictionary mapping view names to data matrices
                       (samples × features)
            group_labels: Optional group labels for group-specific factors

        Returns:
            Self (fitted model)
        """
        # Prepare data matrices
        Y, sample_mask = self._prepare_data(data_dict)

        # Initialize parameters
        self._initialize_parameters(Y, sample_mask)

        # Run variational inference
        self._fit_variational(Y, sample_mask)

        # Calculate variance explained
        self._calculate_variance_explained(Y, sample_mask)

        return self

    def transform(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Transform new data to latent factor space

        Args:
            data_dict: Dictionary of data matrices

        Returns:
            DataFrame with latent factors (samples × factors)
        """
        if self.Z is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # For new data, use fitted loadings to infer factors
        # Simplified: use pseudo-inverse
        Z_new = np.zeros((len(next(iter(data_dict.values()))), self.n_factors))

        for view, data in data_dict.items():
            if view in self.W:
                # Project data onto factor space
                W_view = self.W[view]
                Y_view = data.values

                # Handle missing data
                Y_view = np.nan_to_num(Y_view, 0)

                # Infer factors: Z ≈ Y @ W @ (W.T @ W)^-1
                Z_view = Y_view @ W_view @ np.linalg.pinv(W_view.T @ W_view)
                Z_new += Z_view

        Z_new /= len(data_dict)

        return pd.DataFrame(
            Z_new,
            index=next(iter(data_dict.values())).index,
            columns=[f"Factor{i+1}" for i in range(self.n_factors)],
        )

    def get_factor_loadings(
        self, view: Optional[str] = None, threshold: float = 0.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Get factor loadings for all or specific views

        Args:
            view: Optional view name to filter
            threshold: Threshold for absolute loading values

        Returns:
            Dictionary mapping view names to loading DataFrames
        """
        if not self.W:
            raise ValueError("Model not fitted")

        loadings = {}
        views_to_export = [view] if view else self.views

        for v in views_to_export:
            if v in self.W:
                W_df = pd.DataFrame(
                    self.W[v],
                    index=self.feature_ids[v],
                    columns=[f"Factor{i+1}" for i in range(self.n_factors)],
                )

                # Apply threshold
                if threshold > 0:
                    W_df = W_df[W_df.abs() > threshold]

                loadings[v] = W_df

        return loadings

    def get_latent_factors(self) -> pd.DataFrame:
        """
        Get latent factor matrix

        Returns:
            DataFrame with latent factors (samples × factors)
        """
        if self.Z is None:
            raise ValueError("Model not fitted")

        return pd.DataFrame(
            self.Z,
            index=self.sample_ids,
            columns=[f"Factor{i+1}" for i in range(self.n_factors)],
        )

    def _prepare_data(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare data matrices for integration

        Args:
            data_dict: Dictionary of data matrices

        Returns:
            Tuple of (processed data, sample masks)
        """
        # Get union of all samples
        all_samples = set()
        for df in data_dict.values():
            all_samples.update(df.index)
        self.sample_ids = sorted(list(all_samples))
        self.n_samples = len(self.sample_ids)

        # Store metadata
        self.views = list(data_dict.keys())

        # Prepare aligned data matrices
        Y = {}
        sample_mask = {}

        for view, df in data_dict.items():
            # Store feature IDs
            self.feature_ids[view] = df.columns.tolist()
            self.n_features[view] = len(df.columns)

            # Align samples
            aligned_data = np.full((self.n_samples, len(df.columns)), np.nan)
            mask = np.zeros((self.n_samples, len(df.columns)), dtype=bool)

            for i, sample in enumerate(self.sample_ids):
                if sample in df.index:
                    aligned_data[i] = df.loc[sample].values
                    mask[i] = ~np.isnan(df.loc[sample].values)

            # Standardize features within view
            scaler = StandardScaler()
            # Only fit on non-missing values
            valid_mask = ~np.isnan(aligned_data)
            if valid_mask.any():
                for j in range(aligned_data.shape[1]):
                    col_mask = valid_mask[:, j]
                    if col_mask.sum() > 1:
                        aligned_data[col_mask, j] = scaler.fit_transform(
                            aligned_data[col_mask, j].reshape(-1, 1)
                        ).ravel()

            Y[view] = aligned_data
            sample_mask[view] = mask

        return Y, sample_mask

    def _initialize_parameters(
        self, Y: Dict[str, np.ndarray], sample_mask: Dict[str, np.ndarray]
    ) -> None:
        """Initialize model parameters"""
        # Initialize factors with PCA on concatenated data
        # Fill missing with 0 for initialization
        Y_concat = np.hstack([np.nan_to_num(Y[v], 0) for v in self.views])

        try:
            U, s, Vt = svd(Y_concat, full_matrices=False)
            self.Z = U[:, : self.n_factors] @ np.diag(s[: self.n_factors])
        except:
            # Fallback: random initialization
            self.Z = np.random.randn(self.n_samples, self.n_factors) * 0.01

        # Initialize factor loadings for each view
        for view in self.views:
            n_features = self.n_features[view]

            # Initialize with small random values
            self.W[view] = np.random.randn(n_features, self.n_factors) * 0.01

            # Initialize noise precision
            self.tau[view] = np.ones(n_features)

        # Initialize ARD precision
        if self.ard_prior:
            self.alpha = np.ones(self.n_factors)

    def _fit_variational(
        self, Y: Dict[str, np.ndarray], sample_mask: Dict[str, np.ndarray]
    ) -> None:
        """
        Fit model using variational inference

        This is a simplified version of MOFA's variational Bayes approach.
        """
        for iteration in range(self.n_iterations):
            # E-step: Update factors
            self._update_factors(Y, sample_mask)

            # M-step: Update loadings
            for view in self.views:
                self._update_loadings(view, Y[view], sample_mask[view])

            # Update noise precision
            for view in self.views:
                self._update_noise_precision(view, Y[view], sample_mask[view])

            # Update ARD precision (factor pruning)
            if self.ard_prior and iteration % 10 == 0:
                self._update_ard_precision()

            # Calculate ELBO for convergence
            if iteration % 10 == 0:
                elbo = self._calculate_elbo(Y, sample_mask)
                self.elbo_history.append(elbo)

                if len(self.elbo_history) > 1:
                    improvement = (
                        self.elbo_history[-1] - self.elbo_history[-2]
                    )
                    if abs(improvement) < self.convergence_threshold:
                        print(f"Converged at iteration {iteration}")
                        break

    def _update_factors(
        self, Y: Dict[str, np.ndarray], sample_mask: Dict[str, np.ndarray]
    ) -> None:
        """Update latent factors (Z)"""
        # Simplified update: weighted average across views
        Z_new = np.zeros_like(self.Z)
        total_weight = 0

        for view in self.views:
            W = self.W[view]
            Y_view = Y[view]
            mask = sample_mask[view]
            tau = self.tau[view]

            # Weight for this view
            view_weight = self.view_weights.get(view, 1.0)

            # Update Z for observed values
            for i in range(self.n_samples):
                obs_mask = mask[i]
                if obs_mask.sum() > 0:
                    W_obs = W[obs_mask]
                    Y_obs = Y_view[i, obs_mask]
                    tau_obs = tau[obs_mask]

                    # Weighted least squares: Z = (W^T @ diag(tau) @ W)^-1 @ W^T @ diag(tau) @ Y
                    precision = W_obs.T @ np.diag(tau_obs) @ W_obs
                    # Add small regularization
                    precision += np.eye(self.n_factors) * 1e-6

                    try:
                        Z_new[i] += view_weight * (
                            np.linalg.solve(precision, W_obs.T @ (tau_obs * Y_obs))
                        )
                        total_weight += view_weight
                    except:
                        pass

        if total_weight > 0:
            self.Z = Z_new / total_weight

    def _update_loadings(
        self, view: str, Y: np.ndarray, mask: np.ndarray
    ) -> None:
        """Update factor loadings (W) for a view"""
        W_new = np.zeros_like(self.W[view])
        tau = self.tau[view]

        # Update each feature's loadings
        for j in range(self.n_features[view]):
            obs_mask = mask[:, j]
            if obs_mask.sum() > 1:
                Z_obs = self.Z[obs_mask]
                Y_obs = Y[obs_mask, j]

                # Weighted least squares: W = (Z^T @ Z + lambda*I)^-1 @ Z^T @ Y
                precision = Z_obs.T @ Z_obs

                # Add sparsity prior
                if self.sparsity_prior:
                    precision += np.eye(self.n_factors) * 0.1

                # Add ARD prior
                if self.ard_prior and self.alpha is not None:
                    precision += np.diag(self.alpha)

                try:
                    W_new[j] = np.linalg.solve(precision, Z_obs.T @ Y_obs)
                except:
                    W_new[j] = self.W[view][j]

        self.W[view] = W_new

    def _update_noise_precision(
        self, view: str, Y: np.ndarray, mask: np.ndarray
    ) -> None:
        """Update noise precision (tau) for a view"""
        tau_new = np.zeros(self.n_features[view])

        for j in range(self.n_features[view]):
            obs_mask = mask[:, j]
            n_obs = obs_mask.sum()

            if n_obs > 1:
                Y_obs = Y[obs_mask, j]
                Z_obs = self.Z[obs_mask]
                W_j = self.W[view][j]

                # Residual variance
                residual = Y_obs - Z_obs @ W_j
                variance = (residual ** 2).sum() / n_obs

                # Precision is inverse variance
                tau_new[j] = 1 / (variance + 1e-6)
            else:
                tau_new[j] = 1.0

        self.tau[view] = tau_new

    def _update_ard_precision(self) -> None:
        """Update ARD precision for factor pruning"""
        # ARD: alpha_k = D / sum_m sum_d w_{dmk}^2
        alpha_new = np.zeros(self.n_factors)

        for k in range(self.n_factors):
            weight_sum = 0
            for view in self.views:
                weight_sum += (self.W[view][:, k] ** 2).sum()

            # Total number of features across views
            D_total = sum(self.n_features.values())

            if weight_sum > 0:
                alpha_new[k] = D_total / (weight_sum + 1e-6)
            else:
                alpha_new[k] = 1e6  # Prune this factor

        self.alpha = alpha_new

    def _calculate_elbo(
        self, Y: Dict[str, np.ndarray], sample_mask: Dict[str, np.ndarray]
    ) -> float:
        """Calculate Evidence Lower Bound for convergence monitoring"""
        elbo = 0

        for view in self.views:
            mask = sample_mask[view]
            Y_view = Y[view]
            W = self.W[view]
            tau = self.tau[view]

            # Reconstruction error
            Y_pred = self.Z @ W.T
            residual = (Y_view - Y_pred) ** 2

            # Only count observed values
            residual = residual * mask

            # Weighted by precision
            elbo -= 0.5 * (residual * tau).sum()

        # Add prior terms (simplified)
        if self.ard_prior and self.alpha is not None:
            # ARD prior on loadings
            for view in self.views:
                W = self.W[view]
                for k in range(self.n_factors):
                    elbo -= 0.5 * self.alpha[k] * (W[:, k] ** 2).sum()

        return elbo

    def _calculate_variance_explained(
        self, Y: Dict[str, np.ndarray], sample_mask: Dict[str, np.ndarray]
    ) -> None:
        """Calculate variance explained by each factor in each view"""
        for view in self.views:
            mask = sample_mask[view]
            Y_view = Y[view]
            W = self.W[view]

            # Total variance
            total_var = np.nanvar(Y_view[mask])

            # Variance explained by each factor
            var_explained = {}
            for k in range(self.n_factors):
                Z_k = self.Z[:, k : k + 1]
                W_k = W[:, k : k + 1]

                Y_pred_k = Z_k @ W_k.T
                var_k = np.var(Y_pred_k[mask])

                var_explained[f"Factor{k+1}"] = var_k / total_var if total_var > 0 else 0

            self.variance_explained[view] = var_explained

    def plot_variance_explained(
        self, output_path: Optional[Path] = None
    ) -> None:
        """Plot variance explained by each factor"""
        if not self.variance_explained:
            warnings.warn("Variance explained not calculated. Run fit() first.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepare data for plotting
        factors = [f"Factor{i+1}" for i in range(self.n_factors)]

        for view in self.views:
            var_exp = [self.variance_explained[view][f] for f in factors]
            ax.plot(factors, var_exp, marker="o", label=view)

        ax.set_xlabel("Factor")
        ax.set_ylabel("Variance Explained")
        ax.set_title("Variance Explained by Each Factor")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
        else:
            plt.show()

        plt.close()

    def plot_factor_loadings(
        self,
        view: str,
        top_features: int = 20,
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot heatmap of factor loadings for a view"""
        if view not in self.W:
            raise ValueError(f"View {view} not found")

        W = self.W[view]

        # Get top features by absolute loading
        top_loading_sum = np.abs(W).sum(axis=1).argsort()[::-1][:top_features]

        W_top = W[top_loading_sum]
        feature_names = [self.feature_ids[view][i] for i in top_loading_sum]

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            W_top,
            cmap="RdBu_r",
            center=0,
            xticklabels=[f"Factor{i+1}" for i in range(self.n_factors)],
            yticklabels=feature_names,
            cbar_kws={"label": "Loading"},
            ax=ax,
        )

        ax.set_title(f"Factor Loadings - {view}")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
        else:
            plt.show()

        plt.close()

    def plot_factor_correlation(
        self, output_path: Optional[Path] = None
    ) -> None:
        """Plot correlation between factors"""
        if self.Z is None:
            raise ValueError("Model not fitted")

        # Calculate correlation between factors
        Z_corr = np.corrcoef(self.Z.T)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            Z_corr,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=[f"Factor{i+1}" for i in range(self.n_factors)],
            yticklabels=[f"Factor{i+1}" for i in range(self.n_factors)],
            cbar_kws={"label": "Correlation"},
            ax=ax,
        )

        ax.set_title("Factor Correlation Matrix")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
        else:
            plt.show()

        plt.close()