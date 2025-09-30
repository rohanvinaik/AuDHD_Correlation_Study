#!/usr/bin/env python3
"""
Preprocessing Manifest Export

Exports preprocessing manifest (JSON) with parameters applied per modality
for reproducibility and reporting, as specified in PREPROCESSING_ORDER.md.

Manifest includes:
- Preprocessing version and timestamp
- Steps applied per modality with parameters
- Sample/feature counts before/after each step
- QC metrics (call rate, detection rate, batch effect reduction, etc.)
- Random seed for reproducibility
- Config file used
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class PreprocessingManifest:
    """
    Track preprocessing steps and generate manifest for reproducibility

    Usage:
        manifest = PreprocessingManifest(version="1.0.0", random_seed=42)

        # Record steps
        manifest.add_modality("genomic", n_samples_input=1000, n_features_input=500000)
        manifest.add_step("genomic", "qc_filter", {"min_call_rate": 0.9, "min_maf": 0.01})
        manifest.add_qc_metric("genomic", "mean_call_rate", 0.98)
        manifest.update_counts("genomic", n_samples=950, n_features=50000)

        # Export
        manifest.export("outputs/preprocessing_manifest.json")
    """

    def __init__(
        self,
        version: str = "1.0.0",
        random_seed: int = 42,
        config_file: Optional[str] = None
    ):
        self.version = version
        self.random_seed = random_seed
        self.config_file = config_file
        self.timestamp = datetime.now().isoformat()
        self.modalities: Dict[str, Dict[str, Any]] = {}

    def add_modality(
        self,
        modality: str,
        n_samples_input: int,
        n_features_input: int
    ) -> None:
        """
        Initialize tracking for a modality

        Args:
            modality: Modality name (e.g., 'genomic', 'clinical', 'metabolomic', 'microbiome')
            n_samples_input: Number of samples before preprocessing
            n_features_input: Number of features before preprocessing
        """
        self.modalities[modality] = {
            "n_samples_input": n_samples_input,
            "n_features_input": n_features_input,
            "n_samples_output": n_samples_input,  # Will be updated
            "n_features_output": n_features_input,  # Will be updated
            "steps": [],
            "qc_metrics": {}
        }

    def add_step(
        self,
        modality: str,
        step_name: str,
        params: Dict[str, Any],
        n_samples_after: Optional[int] = None,
        n_features_after: Optional[int] = None
    ) -> None:
        """
        Record a preprocessing step

        Args:
            modality: Modality name
            step_name: Step name (e.g., 'qc_filter', 'transform', 'impute', 'scale', 'batch_correct')
            params: Parameters used for this step
            n_samples_after: Number of samples after this step (if changed)
            n_features_after: Number of features after this step (if changed)
        """
        if modality not in self.modalities:
            raise ValueError(f"Modality {modality} not initialized. Call add_modality() first.")

        step_record = {
            "step": step_name,
            "params": params
        }

        # Record counts if provided
        if n_samples_after is not None:
            step_record["n_samples_after"] = n_samples_after
            self.modalities[modality]["n_samples_output"] = n_samples_after

        if n_features_after is not None:
            step_record["n_features_after"] = n_features_after
            self.modalities[modality]["n_features_output"] = n_features_after

        self.modalities[modality]["steps"].append(step_record)

    def add_qc_metric(
        self,
        modality: str,
        metric_name: str,
        value: float
    ) -> None:
        """
        Add a QC metric for a modality

        Args:
            modality: Modality name
            metric_name: Metric name (e.g., 'mean_call_rate', 'missing_rate_after_impute')
            value: Metric value
        """
        if modality not in self.modalities:
            raise ValueError(f"Modality {modality} not initialized. Call add_modality() first.")

        self.modalities[modality]["qc_metrics"][metric_name] = value

    def update_counts(
        self,
        modality: str,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None
    ) -> None:
        """
        Update final sample/feature counts for a modality

        Args:
            modality: Modality name
            n_samples: Final number of samples
            n_features: Final number of features
        """
        if modality not in self.modalities:
            raise ValueError(f"Modality {modality} not initialized.")

        if n_samples is not None:
            self.modalities[modality]["n_samples_output"] = n_samples

        if n_features is not None:
            self.modalities[modality]["n_features_output"] = n_features

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert manifest to dictionary

        Returns:
            Manifest as nested dictionary matching PREPROCESSING_ORDER.md format
        """
        return {
            "preprocessing_version": self.version,
            "timestamp": self.timestamp,
            "random_seed": self.random_seed,
            "config_file": self.config_file,
            "modalities": self.modalities
        }

    def export(self, output_path: str) -> None:
        """
        Export manifest to JSON file

        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_dict = self.to_dict()

        with open(output_path, 'w') as f:
            json.dump(manifest_dict, f, indent=2)

        print(f"Preprocessing manifest exported to {output_path}")

    def export_summary_table(self, output_path: str) -> None:
        """
        Export human-readable summary table (CSV)

        Args:
            output_path: Path to output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create summary table
        rows = []
        for modality, data in self.modalities.items():
            row = {
                "Modality": modality,
                "Samples (Input)": data["n_samples_input"],
                "Samples (Output)": data["n_samples_output"],
                "Samples (% Retained)": f"{data['n_samples_output'] / data['n_samples_input'] * 100:.1f}%",
                "Features (Input)": data["n_features_input"],
                "Features (Output)": data["n_features_output"],
                "Features (% Retained)": f"{data['n_features_output'] / data['n_features_input'] * 100:.1f}%",
                "Steps": len(data["steps"])
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        print(f"Preprocessing summary table exported to {output_path}")

    def print_summary(self) -> None:
        """
        Print human-readable summary to console
        """
        print("\n" + "="*80)
        print("PREPROCESSING MANIFEST SUMMARY")
        print("="*80)
        print(f"Version: {self.version}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Random Seed: {self.random_seed}")
        print(f"Config File: {self.config_file or 'Not specified'}")
        print()

        for modality, data in self.modalities.items():
            print(f"{'='*80}")
            print(f"Modality: {modality.upper()}")
            print(f"{'='*80}")
            print(f"  Samples:  {data['n_samples_input']:,} → {data['n_samples_output']:,} "
                  f"({data['n_samples_output']/data['n_samples_input']*100:.1f}% retained)")
            print(f"  Features: {data['n_features_input']:,} → {data['n_features_output']:,} "
                  f"({data['n_features_output']/data['n_features_input']*100:.1f}% retained)")
            print()

            if data['steps']:
                print(f"  Steps applied ({len(data['steps'])}):")
                for i, step in enumerate(data['steps'], 1):
                    print(f"    {i}. {step['step']}")
                    for param, value in step['params'].items():
                        print(f"       - {param}: {value}")
                print()

            if data['qc_metrics']:
                print(f"  QC Metrics:")
                for metric, value in data['qc_metrics'].items():
                    if isinstance(value, float):
                        print(f"    - {metric}: {value:.4f}")
                    else:
                        print(f"    - {metric}: {value}")
                print()

        print("="*80)


def load_manifest(manifest_path: str) -> PreprocessingManifest:
    """
    Load manifest from JSON file

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        PreprocessingManifest object
    """
    with open(manifest_path, 'r') as f:
        data = json.load(f)

    manifest = PreprocessingManifest(
        version=data["preprocessing_version"],
        random_seed=data["random_seed"],
        config_file=data.get("config_file")
    )
    manifest.timestamp = data["timestamp"]
    manifest.modalities = data["modalities"]

    return manifest


def compare_manifests(
    manifest1_path: str,
    manifest2_path: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare two preprocessing manifests

    Args:
        manifest1_path: Path to first manifest
        manifest2_path: Path to second manifest
        output_path: Optional path to save comparison report (JSON)

    Returns:
        Comparison report as dictionary
    """
    manifest1 = load_manifest(manifest1_path)
    manifest2 = load_manifest(manifest2_path)

    comparison = {
        "manifest1": {
            "path": manifest1_path,
            "timestamp": manifest1.timestamp,
            "version": manifest1.version
        },
        "manifest2": {
            "path": manifest2_path,
            "timestamp": manifest2.timestamp,
            "version": manifest2.version
        },
        "differences": {}
    }

    # Compare modalities
    all_modalities = set(manifest1.modalities.keys()) | set(manifest2.modalities.keys())

    for modality in all_modalities:
        if modality not in manifest1.modalities:
            comparison["differences"][modality] = {
                "status": "missing_in_manifest1"
            }
            continue
        if modality not in manifest2.modalities:
            comparison["differences"][modality] = {
                "status": "missing_in_manifest2"
            }
            continue

        # Compare steps
        steps1 = [s["step"] for s in manifest1.modalities[modality]["steps"]]
        steps2 = [s["step"] for s in manifest2.modalities[modality]["steps"]]

        if steps1 != steps2:
            comparison["differences"][modality] = {
                "status": "different_steps",
                "manifest1_steps": steps1,
                "manifest2_steps": steps2
            }

        # Compare counts
        counts1 = {
            "samples": manifest1.modalities[modality]["n_samples_output"],
            "features": manifest1.modalities[modality]["n_features_output"]
        }
        counts2 = {
            "samples": manifest2.modalities[modality]["n_samples_output"],
            "features": manifest2.modalities[modality]["n_features_output"]
        }

        if counts1 != counts2:
            if modality not in comparison["differences"]:
                comparison["differences"][modality] = {}
            comparison["differences"][modality]["counts_differ"] = {
                "manifest1": counts1,
                "manifest2": counts2
            }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison report saved to {output_path}")

    return comparison


if __name__ == "__main__":
    # Example usage
    print("Creating example preprocessing manifest...\n")

    manifest = PreprocessingManifest(
        version="1.0.0",
        random_seed=42,
        config_file="configs/preprocessing/standard.yaml"
    )

    # Genomic modality
    manifest.add_modality("genomic", n_samples_input=1000, n_features_input=500000)
    manifest.add_step("genomic", "qc_filter", {
        "min_call_rate": 0.9,
        "min_maf": 0.01,
        "max_missing_per_snp": 0.10
    }, n_samples_after=950, n_features_after=50000)
    manifest.add_step("genomic", "impute", {"method": "mode"})
    manifest.add_step("genomic", "scale", {"method": "standard"})
    manifest.add_step("genomic", "batch_correct", {
        "method": "combat",
        "batch_var": "site",
        "covariates_preserved": ["age", "sex", "diagnosis"]
    })
    manifest.add_qc_metric("genomic", "mean_call_rate", 0.98)
    manifest.add_qc_metric("genomic", "missing_rate_after_impute", 0.0)
    manifest.add_qc_metric("genomic", "batch_effect_reduction", 0.65)

    # Metabolomic modality
    manifest.add_modality("metabolomic", n_samples_input=1000, n_features_input=300)
    manifest.add_step("metabolomic", "qc_filter", {
        "min_detection_rate": 0.3,
        "max_cv_in_qc": 0.3
    }, n_features_after=250)
    manifest.add_step("metabolomic", "transform", {"method": "log2"})
    manifest.add_step("metabolomic", "impute", {
        "method": "knn",
        "n_neighbors": 5,
        "weights": "distance"
    })
    manifest.add_step("metabolomic", "scale", {"method": "standard"})
    manifest.add_step("metabolomic", "batch_correct", {
        "method": "combat",
        "parametric": True
    })
    manifest.add_step("metabolomic", "adjust_covariates", {
        "method": "mixed_effects",
        "covariates": ["fasting_hours", "time_of_day"],
        "random_effects": ["subject_id"]
    })
    manifest.add_qc_metric("metabolomic", "mean_detection_rate", 0.85)
    manifest.add_qc_metric("metabolomic", "missing_rate_after_impute", 0.0)
    manifest.add_qc_metric("metabolomic", "batch_effect_reduction", 0.73)

    # Clinical modality
    manifest.add_modality("clinical", n_samples_input=1000, n_features_input=50)
    manifest.add_step("clinical", "qc_filter", {
        "max_missing_per_feature": 0.50,
        "flag_age_outliers": True
    }, n_features_after=45)
    manifest.add_step("clinical", "impute", {
        "method": "iterative",
        "max_iter": 10,
        "random_state": 42
    })
    manifest.add_step("clinical", "scale", {"method": "robust"})
    manifest.add_qc_metric("clinical", "missing_rate_after_impute", 0.0)

    # Microbiome modality
    manifest.add_modality("microbiome", n_samples_input=1000, n_features_input=500)
    manifest.add_step("microbiome", "qc_filter", {
        "min_sequencing_depth": 10000,
        "min_prevalence": 0.10
    }, n_samples_after=980, n_features_after=200)
    manifest.add_step("microbiome", "transform", {"method": "clr", "pseudocount": 1e-6})
    manifest.add_step("microbiome", "batch_correct", {"method": "combat"})
    manifest.add_qc_metric("microbiome", "mean_sequencing_depth", 45000)
    manifest.add_qc_metric("microbiome", "batch_effect_reduction", 0.58)

    # Print summary
    manifest.print_summary()

    # Export to JSON
    output_dir = Path("outputs/preprocessing")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest.export(output_dir / "preprocessing_manifest.json")
    manifest.export_summary_table(output_dir / "preprocessing_summary.csv")

    print("\n✓ Example manifest created successfully")