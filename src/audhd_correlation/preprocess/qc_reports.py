"""QC report generation for batch correction"""
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats


class BatchCorrectionQCReport:
    """Generate comprehensive QC reports for batch correction"""

    def __init__(self, output_dir: Path):
        """
        Initialize QC report generator

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: Dict[str, Dict] = {}
        self.plots: List[Path] = []

    def generate_report(
        self,
        data_before: pd.DataFrame,
        data_after: pd.DataFrame,
        batch: pd.Series,
        modality: str = "unknown",
        covariates: Optional[pd.DataFrame] = None,
    ) -> Path:
        """
        Generate comprehensive QC report

        Args:
            data_before: Data before correction
            data_after: Data after correction
            batch: Batch labels
            modality: Data modality name
            covariates: Optional covariates

        Returns:
            Path to HTML report
        """
        # Calculate metrics
        self._calculate_metrics(data_before, data_after, batch, modality)

        # Generate plots
        self._generate_pca_plots(data_before, data_after, batch, modality)
        self._generate_density_plots(data_before, data_after, batch, modality)
        self._generate_batch_statistics_plots(data_before, data_after, batch, modality)

        if covariates is not None:
            self._generate_covariate_preservation_plots(
                data_before, data_after, covariates, modality
            )

        # Generate HTML report
        report_path = self._generate_html_report(modality)

        return report_path

    def _calculate_metrics(
        self,
        data_before: pd.DataFrame,
        data_after: pd.DataFrame,
        batch: pd.Series,
        modality: str,
    ) -> None:
        """Calculate QC metrics"""
        metrics = {}

        # Batch effect size (before/after)
        metrics["batch_effect_before"] = self._calculate_batch_effect(
            data_before, batch
        )
        metrics["batch_effect_after"] = self._calculate_batch_effect(data_after, batch)
        metrics["batch_effect_reduction"] = (
            metrics["batch_effect_before"] - metrics["batch_effect_after"]
        ) / metrics["batch_effect_before"]

        # Variance metrics
        metrics["total_variance_before"] = np.var(data_before.values)
        metrics["total_variance_after"] = np.var(data_after.values)
        metrics["variance_preserved"] = (
            metrics["total_variance_after"] / metrics["total_variance_before"]
        )

        # Feature-wise correlation (preservation of structure)
        corr_before = np.corrcoef(data_before.T)
        corr_after = np.corrcoef(data_after.T)
        metrics["correlation_preservation"] = np.corrcoef(
            corr_before[np.triu_indices_from(corr_before, k=1)],
            corr_after[np.triu_indices_from(corr_after, k=1)],
        )[0, 1]

        # Sample-wise Euclidean distance preservation
        from scipy.spatial.distance import pdist, squareform

        dist_before = squareform(pdist(data_before.fillna(0)))
        dist_after = squareform(pdist(data_after.fillna(0)))
        metrics["distance_preservation"] = np.corrcoef(
            dist_before[np.triu_indices_from(dist_before, k=1)],
            dist_after[np.triu_indices_from(dist_after, k=1)],
        )[0, 1]

        self.metrics[modality] = metrics

    def _calculate_batch_effect(
        self, data: pd.DataFrame, batch: pd.Series
    ) -> float:
        """
        Calculate batch effect size (variance explained by batch)

        Args:
            data: Data matrix
            batch: Batch labels

        Returns:
            Proportion of variance explained by batch
        """
        batch = batch.loc[data.index]
        batch_ids = batch.unique()

        if len(batch_ids) <= 1:
            return 0.0

        # Calculate between-batch variance vs total variance
        total_mean = data.mean()
        total_var = ((data - total_mean) ** 2).sum().sum()

        between_var = 0
        for batch_id in batch_ids:
            batch_mask = batch == batch_id
            batch_data = data.loc[batch_mask]
            batch_mean = batch_data.mean()
            n_batch = len(batch_data)

            between_var += n_batch * ((batch_mean - total_mean) ** 2).sum()

        return between_var / total_var if total_var > 0 else 0.0

    def _generate_pca_plots(
        self,
        data_before: pd.DataFrame,
        data_after: pd.DataFrame,
        batch: pd.Series,
        modality: str,
    ) -> None:
        """Generate PCA comparison plots"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PCA before
        pca = PCA(n_components=2)
        pca_before = pca.fit_transform(data_before.fillna(0))

        batch_ids = batch.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(batch_ids)))

        for i, batch_id in enumerate(batch_ids):
            mask = batch == batch_id
            axes[0].scatter(
                pca_before[mask, 0],
                pca_before[mask, 1],
                c=[colors[i]],
                label=f"Batch {batch_id}",
                alpha=0.6,
                s=50,
            )

        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axes[0].set_title(f"{modality} - Before Correction")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # PCA after
        pca_after = pca.fit_transform(data_after.fillna(0))

        for i, batch_id in enumerate(batch_ids):
            mask = batch == batch_id
            axes[1].scatter(
                pca_after[mask, 0],
                pca_after[mask, 1],
                c=[colors[i]],
                label=f"Batch {batch_id}",
                alpha=0.6,
                s=50,
            )

        axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axes[1].set_title(f"{modality} - After Correction")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / f"{modality}_pca_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.plots.append(plot_path)

    def _generate_density_plots(
        self,
        data_before: pd.DataFrame,
        data_after: pd.DataFrame,
        batch: pd.Series,
        modality: str,
    ) -> None:
        """Generate density comparison plots"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        batch_ids = batch.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(batch_ids)))

        # Density before
        for i, batch_id in enumerate(batch_ids):
            mask = batch == batch_id
            batch_data = data_before.loc[mask].values.flatten()
            batch_data = batch_data[~np.isnan(batch_data)]

            axes[0].hist(
                batch_data,
                bins=50,
                alpha=0.5,
                label=f"Batch {batch_id}",
                color=colors[i],
                density=True,
            )

        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Density")
        axes[0].set_title(f"{modality} - Before Correction")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Density after
        for i, batch_id in enumerate(batch_ids):
            mask = batch == batch_id
            batch_data = data_after.loc[mask].values.flatten()
            batch_data = batch_data[~np.isnan(batch_data)]

            axes[1].hist(
                batch_data,
                bins=50,
                alpha=0.5,
                label=f"Batch {batch_id}",
                color=colors[i],
                density=True,
            )

        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Density")
        axes[1].set_title(f"{modality} - After Correction")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / f"{modality}_density_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.plots.append(plot_path)

    def _generate_batch_statistics_plots(
        self,
        data_before: pd.DataFrame,
        data_after: pd.DataFrame,
        batch: pd.Series,
        modality: str,
    ) -> None:
        """Generate batch-wise statistics plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        batch_ids = sorted(batch.unique())

        # Mean by batch
        means_before = [data_before.loc[batch == b].mean().mean() for b in batch_ids]
        means_after = [data_after.loc[batch == b].mean().mean() for b in batch_ids]

        x = np.arange(len(batch_ids))
        width = 0.35

        axes[0, 0].bar(x - width / 2, means_before, width, label="Before", alpha=0.7)
        axes[0, 0].bar(x + width / 2, means_after, width, label="After", alpha=0.7)
        axes[0, 0].set_xlabel("Batch")
        axes[0, 0].set_ylabel("Mean Value")
        axes[0, 0].set_title("Mean by Batch")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([str(b) for b in batch_ids])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Variance by batch
        vars_before = [data_before.loc[batch == b].var().mean() for b in batch_ids]
        vars_after = [data_after.loc[batch == b].var().mean() for b in batch_ids]

        axes[0, 1].bar(x - width / 2, vars_before, width, label="Before", alpha=0.7)
        axes[0, 1].bar(x + width / 2, vars_after, width, label="After", alpha=0.7)
        axes[0, 1].set_xlabel("Batch")
        axes[0, 1].set_ylabel("Variance")
        axes[0, 1].set_title("Variance by Batch")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([str(b) for b in batch_ids])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Sample counts
        counts = [sum(batch == b) for b in batch_ids]
        axes[1, 0].bar(x, counts, alpha=0.7)
        axes[1, 0].set_xlabel("Batch")
        axes[1, 0].set_ylabel("Sample Count")
        axes[1, 0].set_title("Samples per Batch")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([str(b) for b in batch_ids])
        axes[1, 0].grid(True, alpha=0.3)

        # Batch effect metrics
        metrics = self.metrics.get(modality, {})
        metric_names = ["Batch Effect\nBefore", "Batch Effect\nAfter", "Effect\nReduction"]
        metric_values = [
            metrics.get("batch_effect_before", 0),
            metrics.get("batch_effect_after", 0),
            metrics.get("batch_effect_reduction", 0),
        ]

        axes[1, 1].bar(range(len(metric_names)), metric_values, alpha=0.7)
        axes[1, 1].set_ylabel("Value")
        axes[1, 1].set_title("Batch Effect Metrics")
        axes[1, 1].set_xticks(range(len(metric_names)))
        axes[1, 1].set_xticklabels(metric_names)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / f"{modality}_batch_statistics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.plots.append(plot_path)

    def _generate_covariate_preservation_plots(
        self,
        data_before: pd.DataFrame,
        data_after: pd.DataFrame,
        covariates: pd.DataFrame,
        modality: str,
    ) -> None:
        """Generate plots showing covariate preservation"""
        fig, axes = plt.subplots(1, len(covariates.columns), figsize=(5 * len(covariates.columns), 5))

        if len(covariates.columns) == 1:
            axes = [axes]

        for i, covar in enumerate(covariates.columns):
            # Calculate correlation with first PC
            pca = PCA(n_components=1)
            pc1_before = pca.fit_transform(data_before.fillna(0)).ravel()
            pc1_after = pca.fit_transform(data_after.fillna(0)).ravel()

            covar_values = covariates[covar].values

            # Remove NaNs
            valid_mask = ~np.isnan(covar_values) & ~np.isnan(pc1_before) & ~np.isnan(pc1_after)

            corr_before = np.corrcoef(covar_values[valid_mask], pc1_before[valid_mask])[0, 1]
            corr_after = np.corrcoef(covar_values[valid_mask], pc1_after[valid_mask])[0, 1]

            # Plot
            x = ["Before", "After"]
            y = [abs(corr_before), abs(corr_after)]

            axes[i].bar(x, y, alpha=0.7)
            axes[i].set_ylabel("Abs. Correlation with PC1")
            axes[i].set_title(f"Preservation of {covar}")
            axes[i].set_ylim([0, 1])
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / f"{modality}_covariate_preservation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.plots.append(plot_path)

    def _generate_html_report(self, modality: str) -> Path:
        """Generate HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Batch Correction QC Report - {modality}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .metric {{ background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .metric-name {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 24px; color: #2196F3; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
        .timestamp {{ color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Batch Correction QC Report</h1>
    <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    <p><strong>Modality:</strong> {modality}</p>

    <h2>Summary Metrics</h2>
"""

        metrics = self.metrics.get(modality, {})

        for metric_name, metric_value in metrics.items():
            formatted_name = metric_name.replace("_", " ").title()
            if isinstance(metric_value, float):
                formatted_value = f"{metric_value:.4f}"
            else:
                formatted_value = str(metric_value)

            html_content += f"""
    <div class="metric">
        <div class="metric-name">{formatted_name}</div>
        <div class="metric-value">{formatted_value}</div>
    </div>
"""

        html_content += """
    <h2>Visualizations</h2>
"""

        for plot_path in self.plots:
            if modality in plot_path.name:
                html_content += f"""
    <div class="plot">
        <h3>{plot_path.stem.replace('_', ' ').title()}</h3>
        <img src="{plot_path.name}" alt="{plot_path.stem}">
    </div>
"""

        html_content += """
</body>
</html>
"""

        report_path = self.output_dir / f"{modality}_batch_correction_report.html"
        report_path.write_text(html_content)

        return report_path