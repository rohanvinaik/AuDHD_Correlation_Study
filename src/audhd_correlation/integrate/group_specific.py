"""Group-specific factor modeling extension for MOFA"""
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

from .mofa import MOFAIntegration


class GroupSpecificMOFA:
    """
    MOFA with group-specific factors

    Decomposes factors into:
    - Shared factors (active across all groups)
    - Group-specific factors (active in specific groups only)
    """

    def __init__(
        self,
        n_shared_factors: int = 5,
        n_group_factors: int = 3,
        **mofa_kwargs,
    ):
        """
        Initialize group-specific MOFA

        Args:
            n_shared_factors: Number of shared factors
            n_group_factors: Number of group-specific factors per group
            **mofa_kwargs: Additional arguments for MOFAIntegration
        """
        self.n_shared_factors = n_shared_factors
        self.n_group_factors = n_group_factors
        self.mofa_kwargs = mofa_kwargs

        # Fitted models
        self.shared_model: Optional[MOFAIntegration] = None
        self.group_models: Dict[str, MOFAIntegration] = {}
        self.groups: List[str] = []

    def fit(
        self,
        data_dict: Dict[str, pd.DataFrame],
        group_labels: pd.Series,
    ) -> "GroupSpecificMOFA":
        """
        Fit group-specific MOFA model

        Args:
            data_dict: Dictionary of omics data matrices
            group_labels: Group labels for each sample

        Returns:
            Self (fitted)
        """
        self.groups = sorted(group_labels.unique())

        # Fit shared factors on all data
        self.shared_model = MOFAIntegration(
            n_factors=self.n_shared_factors, **self.mofa_kwargs
        )
        self.shared_model.fit(data_dict)

        # Get residuals after removing shared factors
        residual_dict = self._compute_residuals(data_dict, self.shared_model)

        # Fit group-specific factors on residuals
        for group in self.groups:
            group_mask = group_labels == group

            # Filter data for this group
            group_data = {
                view: df[group_mask] for view, df in residual_dict.items()
            }

            # Fit group-specific MOFA
            group_model = MOFAIntegration(
                n_factors=self.n_group_factors, **self.mofa_kwargs
            )
            group_model.fit(group_data)

            self.group_models[group] = group_model

        return self

    def get_all_factors(
        self, data_dict: Dict[str, pd.DataFrame], group_labels: pd.Series
    ) -> pd.DataFrame:
        """
        Get combined shared and group-specific factors

        Args:
            data_dict: Data matrices
            group_labels: Group labels

        Returns:
            DataFrame with all factors
        """
        # Get shared factors
        shared_factors = self.shared_model.get_latent_factors()

        # Get group-specific factors
        all_group_factors = {}

        for group in self.groups:
            group_mask = group_labels == group

            # Get factors for this group
            if group in self.group_models:
                group_factors = self.group_models[group].get_latent_factors()

                # Create columns for this group (fill with 0 for all samples initially)
                for i in range(self.n_group_factors):
                    col_name = f"Group_{group}_Factor{i+1}"
                    all_group_factors[col_name] = np.zeros(len(shared_factors))
                    # Fill in values for this group's samples
                    all_group_factors[col_name][group_mask] = group_factors.iloc[:, i].values

        # Combine shared and group-specific factors
        if all_group_factors:
            group_factors_df = pd.DataFrame(all_group_factors, index=shared_factors.index)
            all_factors = pd.concat([shared_factors, group_factors_df], axis=1)
        else:
            all_factors = shared_factors

        return all_factors

    def _compute_residuals(
        self, data_dict: Dict[str, pd.DataFrame], model: MOFAIntegration
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute residuals after removing model predictions

        Args:
            data_dict: Original data
            model: Fitted MOFA model

        Returns:
            Dictionary of residual matrices
        """
        residuals = {}

        Z = model.Z  # Latent factors

        for view, data in data_dict.items():
            W = model.W[view]  # Loadings

            # Predict: Y_pred = Z @ W^T
            Y_pred = Z @ W.T

            # Residuals
            residual = data.values - Y_pred

            residuals[view] = pd.DataFrame(
                residual, index=data.index, columns=data.columns
            )

        return residuals

    def plot_factor_importance(
        self, output_path: Optional[str] = None
    ) -> None:
        """Plot variance explained by shared vs group-specific factors"""
        import matplotlib.pyplot as plt

        if self.shared_model is None:
            raise ValueError("Model not fitted")

        # Get variance explained for shared factors
        var_exp_shared = {}
        for view in self.shared_model.views:
            view_var = self.shared_model.variance_explained[view]
            total_var = sum(view_var.values())
            var_exp_shared[view] = total_var

        # Get variance explained for group-specific factors
        var_exp_group = {group: {} for group in self.groups}

        for group, model in self.group_models.items():
            for view in model.views:
                view_var = model.variance_explained[view]
                total_var = sum(view_var.values())
                var_exp_group[group][view] = total_var

        # Plot
        views = list(var_exp_shared.keys())
        x = np.arange(len(views))
        width = 0.8 / (len(self.groups) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Shared factors
        shared_vals = [var_exp_shared[v] for v in views]
        ax.bar(x, shared_vals, width, label="Shared", alpha=0.8)

        # Group-specific factors
        for i, group in enumerate(self.groups):
            group_vals = [var_exp_group[group].get(v, 0) for v in views]
            ax.bar(x + (i + 1) * width, group_vals, width, label=f"Group {group}", alpha=0.8)

        ax.set_xlabel("View")
        ax.set_ylabel("Variance Explained")
        ax.set_title("Variance Explained: Shared vs Group-Specific Factors")
        ax.set_xticks(x + width * len(self.groups) / 2)
        ax.set_xticklabels(views)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
        else:
            plt.show()

        plt.close()