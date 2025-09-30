"""Pipeline facade for convenient API access

Provides a simple Pipeline class that wraps the run_all functions.
Matches the API shown in README examples.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import warnings

from .config import load_config, AppConfig
from .pipelines import run_all
from .reporting import report_generator
from .utils.seeds import set_global_seed


class Pipeline:
    """Main pipeline orchestrator

    Convenience wrapper around modular run_all functions.
    Provides object-oriented API for running complete pipeline.

    Example:
        >>> from audhd_correlation import Pipeline
        >>> pipeline = Pipeline(config_path="config.yaml")
        >>> results = pipeline.run()
        >>> pipeline.generate_report(results, output_path="report.html")

    Args:
        config_path: Path to YAML config file
        seed: Optional random seed (overrides config)
        output_dir: Optional output directory (overrides config)
    """

    def __init__(
        self,
        config_path: str,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        """Initialize pipeline with config

        Args:
            config_path: Path to config YAML file
            seed: Optional random seed (overrides config)
            output_dir: Optional output directory (overrides config)
        """
        self.config_path = str(config_path)
        self.config = load_config(self.config_path)

        # Override settings if provided
        if seed is not None:
            self.config.seed = seed

        if output_dir is not None:
            self.config.output_dir = output_dir

        # Set global seed
        set_global_seed(self.config.seed)

        self.results = {}

    def download(self) -> None:
        """Download raw data (requires DUAs)

        Fetches raw datasets from SPARK/SSC/ABCD/UKB and reference data.

        Raises:
            DUAError: If required Data Use Agreements not accepted
        """
        run_all.download(self.config_path)

    def build_features(self) -> None:
        """Build feature matrices

        Runs QC, harmonization, imputation, batch correction, and scaling.
        Saves processed features to disk.
        """
        run_all.build_features(self.config_path)

    def integrate(self) -> Dict[str, Any]:
        """Integrate multi-omics data

        Uses MOFA/PCA/CCA to integrate multiple modalities.

        Returns:
            Dictionary with integrated factors and loadings
        """
        return run_all.integrate(self.config_path)

    def cluster(self, integration_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform clustering analysis

        Uses HDBSCAN/K-means/hierarchical clustering on integrated data.

        Args:
            integration_results: Optional integration results (if None, loads from disk)

        Returns:
            Dictionary with cluster labels, embeddings, and metrics
        """
        return run_all.cluster(self.config_path, integration_results)

    def validate(self, clustering_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate clustering results

        Computes stability metrics, silhouette scores, and permutation tests.

        Args:
            clustering_results: Optional clustering results (if None, loads from disk)

        Returns:
            Dictionary with validation metrics and significance tests
        """
        return run_all.validate(self.config_path, clustering_results)

    def interpret(
        self,
        clustering_results: Optional[Dict] = None,
        validation_results: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Perform biological interpretation

        Runs pathway enrichment, network analysis, and drug target prediction.

        Args:
            clustering_results: Optional clustering results
            validation_results: Optional validation results

        Returns:
            Dictionary with enrichment results, networks, and drug targets
        """
        return run_all.interpret(
            self.config_path,
            clustering_results,
            validation_results,
        )

    def run(
        self,
        stages: Optional[list] = None,
        skip_download: bool = True,
    ) -> Dict[str, Any]:
        """Run complete pipeline

        Executes all pipeline stages in sequence.

        Args:
            stages: Optional list of stages to run
                    (default: ["build_features", "integrate", "cluster", "validate", "interpret"])
            skip_download: Skip download stage (default: True, since requires DUAs)

        Returns:
            Dictionary with all pipeline results
        """
        if stages is None:
            stages = [
                "build_features",
                "integrate",
                "cluster",
                "validate",
                "interpret",
            ]
            if not skip_download:
                stages.insert(0, "download")

        results = {}

        # Run stages in sequence
        for stage in stages:
            if stage == "download":
                self.download()
            elif stage == "build_features":
                self.build_features()
            elif stage == "integrate":
                results["integration"] = self.integrate()
            elif stage == "cluster":
                results["clustering"] = self.cluster(
                    results.get("integration")
                )
            elif stage == "validate":
                results["validation"] = self.validate(
                    results.get("clustering")
                )
            elif stage == "interpret":
                results["interpretation"] = self.interpret(
                    results.get("clustering"),
                    results.get("validation"),
                )
            else:
                warnings.warn(f"Unknown stage: {stage}")

        self.results = results
        return results

    def generate_report(
        self,
        results: Optional[Dict] = None,
        output_path: str = "report.html",
        include_pdf: bool = False,
        include_supplementary: bool = True,
    ) -> str:
        """Generate analysis report

        Creates HTML report with figures, tables, and interpretations.
        Optionally generates PDF version.

        Args:
            results: Pipeline results dictionary (if None, uses self.results)
            output_path: Output file path
            include_pdf: Also generate PDF report
            include_supplementary: Include supplementary materials

        Returns:
            Path to generated report

        Example:
            >>> pipeline = Pipeline("config.yaml")
            >>> results = pipeline.run()
            >>> pipeline.generate_report(results, "analysis_report.html")
        """
        if results is None:
            if not self.results:
                raise ValueError(
                    "No results available. Run pipeline first with pipeline.run()"
                )
            results = self.results

        # Generate HTML report
        report_path = report_generator.generate_html_report(
            results=results,
            config=self.config,
            output_path=output_path,
            include_supplementary=include_supplementary,
        )

        # Generate PDF if requested
        if include_pdf:
            pdf_path = output_path.replace(".html", ".pdf")
            report_generator.generate_pdf_report(
                results=results,
                config=self.config,
                output_path=pdf_path,
            )

        return report_path

    def save_results(self, output_dir: Optional[str] = None) -> None:
        """Save pipeline results to disk

        Saves all results in standard format (Parquet/CSV/NPY).

        Args:
            output_dir: Output directory (default: from config)
        """
        from .utils.io import save_data

        if output_dir is None:
            output_dir = self.config.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for stage, result in self.results.items():
            save_data(
                result,
                output_dir / f"{stage}_results.parquet",
                format="parquet",
            )

    def load_results(self, results_dir: str) -> Dict[str, Any]:
        """Load previously saved results

        Args:
            results_dir: Directory with saved results

        Returns:
            Dictionary with loaded results
        """
        from .utils.io import load_data

        results_dir = Path(results_dir)
        results = {}

        for stage in ["integration", "clustering", "validation", "interpretation"]:
            result_file = results_dir / f"{stage}_results.parquet"
            if result_file.exists():
                results[stage] = load_data(result_file)

        self.results = results
        return results


# Convenience function for simple workflows
def run_pipeline(config_path: str, **kwargs) -> Dict[str, Any]:
    """Run complete pipeline with single function call

    Convenience function for simple workflows.

    Args:
        config_path: Path to config YAML
        **kwargs: Additional arguments passed to Pipeline.run()

    Returns:
        Dictionary with pipeline results

    Example:
        >>> from audhd_correlation import run_pipeline
        >>> results = run_pipeline("config.yaml")
    """
    pipeline = Pipeline(config_path)
    return pipeline.run(**kwargs)