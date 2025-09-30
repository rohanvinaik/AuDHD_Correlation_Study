"""
Baseline-Deviation-Topology Pipeline

Comprehensive pipeline implementing the "baseline → deviations → topology gate → clustering" approach
to prevent false positive subtype discoveries.

Usage:
    from audhd_correlation.pipelines.baseline_deviation_pipeline import run_baseline_deviation_pipeline

    results = run_baseline_deviation_pipeline(
        Z=integrated_features,  # n_samples × n_features
        control_mask=control_indices,  # Optional
        config=pipeline_config
    )
"""
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..modeling.topology import (
    BaselineManifold,
    DeviationScores,
    RotationNull,
    TopologyGate,
    SeparationDecision,
)
from ..modeling.clustering import ConsensusClusteringPipeline


@dataclass
class BaselineDeviationResults:
    """Complete results from baseline-deviation-topology pipeline"""

    # Baseline
    baseline_manifold: BaselineManifold
    baseline_indices: np.ndarray

    # Deviations
    deviation_scores: DeviationScores
    deviation_threshold: float
    deviation_method: str  # 'quantile' or 'fdr'
    deviants_mask: np.ndarray
    deviants_indices: np.ndarray
    n_deviants: int
    prevalence: float  # deviants / total

    # Null distribution
    rotation_null: Optional[RotationNull]
    null_threshold_95: Optional[float]
    null_threshold_99: Optional[float]

    # Topology decision
    topology_decision: Optional[SeparationDecision]

    # Clustering (if topology gate passed)
    clustering_results: Optional[Any]
    cluster_labels: Optional[np.ndarray]  # Full length (baseline=-1, non-deviant=-2, deviants=0,1,2,...)
    soft_memberships: Optional[np.ndarray]

    # Report
    decision: str  # 'baseline_only', 'deviants_continuous', 'deviants_clustered'
    notes: List[str] = field(default_factory=list)

    # Config
    config: Dict[str, Any] = field(default_factory=dict)


def run_baseline_deviation_pipeline(
    Z: np.ndarray,
    control_mask: Optional[np.ndarray] = None,
    sample_ids: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> BaselineDeviationResults:
    """
    Run comprehensive baseline-deviation-topology pipeline

    Args:
        Z: Integrated feature matrix (n_samples, n_features)
        control_mask: Boolean mask for controls (if None, uses unsupervised mode)
        sample_ids: Optional sample identifiers
        config: Configuration dict (uses defaults if None)
        output_dir: Optional directory to save results

    Returns:
        BaselineDeviationResults
    """
    # Default config
    if config is None:
        config = get_default_config()

    notes = []

    print("="*80)
    print("BASELINE-DEVIATION-TOPOLOGY PIPELINE")
    print("="*80)

    # STEP 0: Standardize features
    print("\n[0/6] Standardizing features...")
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z)

    # STEP 1: Learn baseline manifold
    print("\n[1/6] Learning baseline manifold...")

    if control_mask is not None:
        print(f"   Mode: CONTROL (n_controls={control_mask.sum()}/{len(Z)})")
        baseline = BaselineManifold(
            mode='control',
            n_neighbors=config['topology']['neighbors'],
            local_pca_components=config['baseline']['local_pca_components'],
        )
        baseline.fit(Z_scaled, control_mask=control_mask)
        notes.append(f"Baseline learned from {control_mask.sum()} controls")
    else:
        print(f"   Mode: UNSUPERVISED (density-based)")
        baseline = BaselineManifold(
            mode='unsupervised',
            n_neighbors=config['topology']['neighbors'],
            local_pca_components=config['baseline']['local_pca_components'],
            density_percentile=config['baseline']['density_percentile'],
        )
        baseline.fit(Z_scaled)
        notes.append(f"Baseline learned from high-density ridge ({len(baseline.baseline_indices_)} samples)")

    # STEP 2: Score deviations
    print("\n[2/6] Scoring deviations from baseline...")
    deviation_scores = baseline.score(Z_scaled)
    print(f"   Computed: orthogonal residual, MST delta, k-NN curvature")

    # STEP 3: Determine deviation threshold (using rotation null)
    print("\n[3/6] Computing deviation threshold...")

    method = config['deviation_threshold']['method']

    if method == 'null_quantile':
        print(f"   Method: Rotation null (q={config['deviation_threshold']['q']})")

        rotation_null = RotationNull(
            baseline_manifold=baseline,
            n_rotations=config['topology']['rotation_null']['n_rotations'],
            preserve_scale=config['topology']['rotation_null']['preserve_scale'],
            random_state=42
        )
        rotation_null.fit(Z_scaled)

        threshold = rotation_null.quantile(config['deviation_threshold']['q'])
        null_95 = rotation_null.quantile(0.95)
        null_99 = rotation_null.quantile(0.99)

        print(f"   Null thresholds: 95th={null_95:.3f}, 99th={null_99:.3f}")
        print(f"   Using threshold: {threshold:.3f}")

    elif method == 'fdr':
        print(f"   Method: FDR control (q={config['deviation_threshold'].get('fdr_q', 0.05)})")

        rotation_null = RotationNull(
            baseline_manifold=baseline,
            n_rotations=config['topology']['rotation_null']['n_rotations'],
            preserve_scale=config['topology']['rotation_null']['preserve_scale'],
            random_state=42
        )
        rotation_null.fit(Z_scaled)

        threshold = rotation_null.compute_fdr_threshold(
            deviation_scores.deviation_score,
            q=config['deviation_threshold'].get('fdr_q', 0.05)
        )

        null_95 = rotation_null.quantile(0.95)
        null_99 = rotation_null.quantile(0.99)

        print(f"   FDR threshold: {threshold:.3f}")

    else:
        warnings.warn(f"Unknown method '{method}', using 99th percentile")
        threshold = np.percentile(deviation_scores.deviation_score, 99)
        rotation_null = None
        null_95 = None
        null_99 = None

    # Identify deviants
    deviants_mask = deviation_scores.deviation_score >= threshold
    deviants_indices = np.where(deviants_mask)[0]
    n_deviants = deviants_mask.sum()
    prevalence = n_deviants / len(Z)

    print(f"\n   Deviants identified: {n_deviants}/{len(Z)} ({100*prevalence:.1f}%)")
    notes.append(f"Deviation threshold: {threshold:.3f} ({method})")
    notes.append(f"Deviants: {n_deviants}/{len(Z)} ({100*prevalence:.1f}%)")

    if n_deviants == 0:
        print("\n⚠️ No deviants found. All samples within baseline.")
        notes.append("⚠️ No deviants identified - all samples baseline-like")

        return BaselineDeviationResults(
            baseline_manifold=baseline,
            baseline_indices=baseline.baseline_indices_,
            deviation_scores=deviation_scores,
            deviation_threshold=threshold,
            deviation_method=method,
            deviants_mask=deviants_mask,
            deviants_indices=deviants_indices,
            n_deviants=n_deviants,
            prevalence=prevalence,
            rotation_null=rotation_null,
            null_threshold_95=null_95,
            null_threshold_99=null_99,
            topology_decision=None,
            clustering_results=None,
            cluster_labels=None,
            soft_memberships=None,
            decision='baseline_only',
            notes=notes,
            config=config
        )

    if n_deviants < 10:
        print(f"\n⚠️ Too few deviants (n={n_deviants}). Cannot reliably assess topology.")
        notes.append(f"⚠️ Too few deviants (n={n_deviants}) for reliable clustering")

        return BaselineDeviationResults(
            baseline_manifold=baseline,
            baseline_indices=baseline.baseline_indices_,
            deviation_scores=deviation_scores,
            deviation_threshold=threshold,
            deviation_method=method,
            deviants_mask=deviants_mask,
            deviants_indices=deviants_indices,
            n_deviants=n_deviants,
            prevalence=prevalence,
            rotation_null=rotation_null,
            null_threshold_95=null_95,
            null_threshold_99=null_99,
            topology_decision=None,
            clustering_results=None,
            cluster_labels=None,
            soft_memberships=None,
            decision='deviants_too_few',
            notes=notes,
            config=config
        )

    # STEP 4: Topology gate (decide: spectrum vs discrete subtypes)
    print("\n[4/6] Evaluating topology gate...")
    print(f"   Testing {n_deviants} deviants for discrete structure...")

    Z_deviants = Z_scaled[deviants_mask]

    gate = TopologyGate(
        min_separation_score=config['topology']['separation_gate']['min_score'],
        min_confidence=config['topology']['separation_gate']['ci_method'],
        n_bootstrap=config['topology']['separation_gate']['n_bootstrap'],
        random_state=42
    )

    topology_decision = gate.analyze(Z_deviants, labels=None)

    print(f"\n   Separation score: {topology_decision.separation_score:.3f}")
    print(f"   95% CI: [{topology_decision.confidence_interval[0]:.3f}, {topology_decision.confidence_interval[1]:.3f}]")
    print(f"   Decision: {topology_decision.decision.upper()}")

    for note in topology_decision.notes:
        print(f"   {note}")

    notes.extend(topology_decision.notes)

    # STEP 5: Clustering (only if topology gate passes)
    if topology_decision.subtype_claim_allowed:
        print("\n[5/6] Topology gate PASSED → Running consensus clustering on deviants...")

        # Run consensus clustering
        clustering_pipeline = ConsensusClusteringPipeline(
            use_hdbscan=True,
            use_spectral=True,
            use_bgmm=True,
            use_tda=False,
            n_bootstrap=100,
            random_state=42,
            test_null_models=config['clustering']['test_null_models'],
            evaluate_topology_gates=False,  # Already done
            enable_config_locking=config['clustering']['enable_config_locking'],
            config_dict=config if config['clustering']['enable_config_locking'] else None,
            lockfile_path=Path(output_dir) / 'config_lock.json' if output_dir else None,
        )

        clustering_pipeline.fit(Z_deviants, generate_embeddings=True)

        # Get results
        cluster_labels_deviants = clustering_pipeline.consensus_labels_

        # Create full-length labels
        # baseline = -1, non-deviant = -2, deviants = 0,1,2,...
        cluster_labels_full = np.full(len(Z), -2, dtype=int)
        cluster_labels_full[baseline.baseline_indices_] = -1
        cluster_labels_full[deviants_indices] = cluster_labels_deviants

        print(f"\n   Found {len(set(cluster_labels_deviants))} clusters among deviants")
        print(f"   Silhouette score: {clustering_pipeline.metrics_.silhouette:.3f}")

        notes.append(f"Consensus clustering: {len(set(cluster_labels_deviants))} subtypes identified")
        notes.append(f"Silhouette: {clustering_pipeline.metrics_.silhouette:.3f}")

        decision = 'deviants_clustered'

        return BaselineDeviationResults(
            baseline_manifold=baseline,
            baseline_indices=baseline.baseline_indices_,
            deviation_scores=deviation_scores,
            deviation_threshold=threshold,
            deviation_method=method,
            deviants_mask=deviants_mask,
            deviants_indices=deviants_indices,
            n_deviants=n_deviants,
            prevalence=prevalence,
            rotation_null=rotation_null,
            null_threshold_95=null_95,
            null_threshold_99=null_99,
            topology_decision=topology_decision,
            clustering_results=clustering_pipeline,
            cluster_labels=cluster_labels_full,
            soft_memberships=None,  # TODO: extract from clustering_pipeline
            decision=decision,
            notes=notes,
            config=config
        )

    else:
        print("\n[5/6] Topology gate FAILED → No discrete subtypes found")
        print("   ⚠️ Data appears to be a continuous spectrum among deviants")
        print("   Recommend: Report continuous factors, not hard subtype labels")

        notes.append("⚠️ Topology gate failed: deviants form a spectrum, not discrete subtypes")

        decision = 'deviants_continuous'

        return BaselineDeviationResults(
            baseline_manifold=baseline,
            baseline_indices=baseline.baseline_indices_,
            deviation_scores=deviation_scores,
            deviation_threshold=threshold,
            deviation_method=method,
            deviants_mask=deviants_mask,
            deviants_indices=deviants_indices,
            n_deviants=n_deviants,
            prevalence=prevalence,
            rotation_null=rotation_null,
            null_threshold_95=null_95,
            null_threshold_99=null_99,
            topology_decision=topology_decision,
            clustering_results=None,
            cluster_labels=None,
            soft_memberships=None,
            decision=decision,
            notes=notes,
            config=config
        )


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for baseline-deviation pipeline"""
    return {
        'baseline': {
            'local_pca_components': 5,
            'density_percentile': 75.0,  # For unsupervised mode
        },
        'topology': {
            'neighbors': 15,
            'rotation_null': {
                'n_rotations': 200,
                'preserve_scale': True,
            },
            'separation_gate': {
                'min_score': 0.6,
                'ci_method': 0.5,  # Min confidence
                'n_bootstrap': 200,
            },
        },
        'deviation_threshold': {
            'method': 'null_quantile',  # 'null_quantile' or 'fdr'
            'q': 0.99,  # 99th percentile
            'fdr_q': 0.05,  # For FDR method
        },
        'clustering': {
            'use_consensus': True,
            'report_soft_membership': True,
            'test_null_models': False,  # Skip (already tested in topology)
            'enable_config_locking': True,
        },
    }


def save_results(
    results: BaselineDeviationResults,
    output_dir: Union[str, Path],
    sample_ids: Optional[np.ndarray] = None,
):
    """
    Save pipeline results to disk

    Args:
        results: BaselineDeviationResults
        output_dir: Output directory
        sample_ids: Optional sample identifiers
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save deviation scores
    dev_df = pd.DataFrame({
        'sample_id': sample_ids if sample_ids is not None else np.arange(len(results.deviation_scores.deviation_score)),
        'orthogonal_residual': results.deviation_scores.orthogonal_residual,
        'mst_delta': results.deviation_scores.mst_delta,
        'knn_curvature': results.deviation_scores.knn_curvature,
        'deviation_score': results.deviation_scores.deviation_score,
        'is_deviant': results.deviants_mask,
        'cluster_label': results.cluster_labels if results.cluster_labels is not None else -2,
    })
    dev_df.to_csv(output_dir / 'deviation_scores.csv', index=False)

    # Save summary report
    with open(output_dir / 'pipeline_summary.txt', 'w') as f:
        f.write("BASELINE-DEVIATION-TOPOLOGY PIPELINE RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Decision: {results.decision}\n\n")

        f.write("Notes:\n")
        for note in results.notes:
            f.write(f"  - {note}\n")

        f.write(f"\nBaseline: {len(results.baseline_indices)} samples\n")
        f.write(f"Deviants: {results.n_deviants} samples ({100*results.prevalence:.1f}%)\n")

        if results.topology_decision:
            f.write(f"\nTopology Decision:\n")
            f.write(f"  Separation score: {results.topology_decision.separation_score:.3f}\n")
            f.write(f"  95% CI: [{results.topology_decision.confidence_interval[0]:.3f}, {results.topology_decision.confidence_interval[1]:.3f}]\n")
            f.write(f"  Decision: {results.topology_decision.decision}\n")
            f.write(f"  Subtype claim allowed: {results.topology_decision.subtype_claim_allowed}\n")

        if results.clustering_results:
            f.write(f"\nClustering Results:\n")
            f.write(f"  Number of clusters: {len(set(results.cluster_labels[results.cluster_labels >= 0]))}\n")
            f.write(f"  Silhouette score: {results.clustering_results.metrics_.silhouette:.3f}\n")

    print(f"\n✓ Results saved to {output_dir}/")
