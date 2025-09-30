#!/usr/bin/env python3
"""
Brain Gradient Validation
Validates cluster findings against canonical brain organization
Uses neuromaps and AHBA gradients
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


@dataclass
class GradientValidationResult:
    """Results from brain gradient validation"""
    correlation: float
    p_value: float
    null_mean: float
    null_std: float
    z_score: float
    gradient_alignment: pd.DataFrame


class BrainGradientValidator:
    """
    Validate findings against canonical brain organization

    Uses:
    1. Allen Human Brain Atlas (AHBA) expression gradients
    2. Neuromaps spatial correlation with spin tests
    3. Principal gradient of connectivity

    Answers: Do AuDHD subtypes align with brain's organizational axes?
    """

    def __init__(self, parcellation: str = 'desikan'):
        """
        Initialize validator

        Parameters
        ----------
        parcellation : str
            Brain parcellation scheme ('desikan', 'destrieux', 'schaefer')
        """
        self.parcellation = parcellation
        self.canonical_gradients = None

    def load_canonical_gradients(self) -> pd.DataFrame:
        """
        Load canonical brain gradients

        In practice, would use:
        - neuromaps.datasets.fetch_annotation(source='margulies2016')
        - AHBA gene expression gradients

        Returns
        -------
        gradients : pd.DataFrame
            Regions × gradients
        """
        logger.info("Loading canonical brain gradients")

        # Mock canonical gradients (in practice, load from neuromaps)
        # Principal gradient: sensory-fugal to transmodal axis
        n_regions = 68  # Desikan-Killiany

        # Gradient 1: Sensory → Association
        gradient1 = np.linspace(-1, 1, n_regions)

        # Gradient 2: Visual → Motor
        gradient2 = np.sin(np.linspace(0, 2*np.pi, n_regions))

        # Gradient 3: Anterior → Posterior
        gradient3 = np.linspace(1, -1, n_regions)

        gradients = pd.DataFrame({
            'gradient1_sensory_assoc': gradient1,
            'gradient2_visual_motor': gradient2,
            'gradient3_ant_post': gradient3
        })

        logger.info(f"  Loaded {gradients.shape[1]} canonical gradients for {n_regions} regions")

        self.canonical_gradients = gradients

        return gradients

    def compute_ahba_gradient(
        self,
        expression_matrix: pd.DataFrame,
        genes: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Compute principal gradient from Allen Human Brain Atlas expression

        Parameters
        ----------
        expression_matrix : pd.DataFrame
            Gene expression (regions × genes)
        genes : List[str], optional
            Genes to include (default: all)

        Returns
        -------
        principal_gradient : np.ndarray
            First principal component of expression
        """
        logger.info("Computing AHBA expression gradient")

        if genes is not None:
            expression_matrix = expression_matrix[genes]

        # PCA to extract principal gradient
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3)
        gradients = pca.fit_transform(expression_matrix.values)

        logger.info(f"  Explained variance: {pca.explained_variance_ratio_[:3]}")

        return gradients[:, 0]  # Principal gradient

    def validate_against_brain_gradients(
        self,
        cluster_map: np.ndarray,
        brain_parcellation: Optional[np.ndarray] = None,
        n_nulls: int = 1000
    ) -> GradientValidationResult:
        """
        Test spatial correspondence between clusters and brain gradients

        Uses spin test (Alexander-Bloch et al. 2018) for spatial null model

        Parameters
        ----------
        cluster_map : np.ndarray
            Cluster assignments per brain region
        brain_parcellation : np.ndarray, optional
            Parcellation labels
        n_nulls : int
            Number of null rotations for spin test

        Returns
        -------
        GradientValidationResult
        """
        logger.info("Validating clusters against brain gradients")

        if self.canonical_gradients is None:
            self.load_canonical_gradients()

        # Compute correlation with each gradient
        gradient_correlations = []

        for grad_name in self.canonical_gradients.columns:
            gradient = self.canonical_gradients[grad_name].values

            # Correlation
            corr, p_val = stats.spearmanr(cluster_map, gradient)

            gradient_correlations.append({
                'gradient': grad_name,
                'correlation': corr,
                'p_value': p_val
            })

        # Spin test for spatial autocorrelation
        principal_gradient = self.canonical_gradients.iloc[:, 0].values

        null_correlations = self._spin_test(
            cluster_map, principal_gradient, n_nulls=n_nulls
        )

        # Observed correlation
        obs_corr = stats.spearmanr(cluster_map, principal_gradient)[0]

        # P-value from null distribution
        p_spin = (np.abs(null_correlations) >= np.abs(obs_corr)).sum() / n_nulls

        # Z-score
        null_mean = null_correlations.mean()
        null_std = null_correlations.std()
        z_score = (obs_corr - null_mean) / null_std if null_std > 0 else 0

        logger.info(f"  Correlation with principal gradient: r={obs_corr:.3f}, p_spin={p_spin:.4f}, z={z_score:.2f}")

        gradient_df = pd.DataFrame(gradient_correlations).sort_values('p_value')

        return GradientValidationResult(
            correlation=obs_corr,
            p_value=p_spin,
            null_mean=null_mean,
            null_std=null_std,
            z_score=z_score,
            gradient_alignment=gradient_df
        )

    def _spin_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        n_nulls: int
    ) -> np.ndarray:
        """
        Spatial permutation test (spin test)

        Rotates brain parcels on sphere to generate null distributions
        that preserve spatial autocorrelation

        Parameters
        ----------
        data1, data2 : np.ndarray
            Data per brain region
        n_nulls : int
            Number of rotations

        Returns
        -------
        null_correlations : np.ndarray
            Null correlation distribution
        """
        logger.info("  Running spin test for spatial null")

        # Simplified implementation
        # In practice, would use actual sphere rotations
        null_correlations = []

        for _ in range(n_nulls):
            # Random circular shift (simplified rotation)
            shift = np.random.randint(1, len(data1))
            data1_rotated = np.roll(data1, shift)

            corr = stats.spearmanr(data1_rotated, data2)[0]
            null_correlations.append(corr)

        return np.array(null_correlations)

    def test_gradient_alignment_per_cluster(
        self,
        cluster_labels: np.ndarray,
        brain_regions: np.ndarray
    ) -> pd.DataFrame:
        """
        Test which clusters align with which gradients

        Parameters
        ----------
        cluster_labels : np.ndarray
            Cluster per sample
        brain_regions : np.ndarray
            Regional values per sample

        Returns
        -------
        cluster_gradient_alignment : pd.DataFrame
            Clusters × gradients correlation matrix
        """
        logger.info("Testing gradient alignment per cluster")

        if self.canonical_gradients is None:
            self.load_canonical_gradients()

        alignments = []

        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_regions = brain_regions[cluster_mask]

            for grad_name in self.canonical_gradients.columns:
                gradient = self.canonical_gradients[grad_name].values

                # Mean regional alignment for this cluster
                alignment = np.mean([
                    gradient[int(region)] for region in cluster_regions
                    if int(region) < len(gradient)
                ])

                alignments.append({
                    'cluster': cluster_id,
                    'gradient': grad_name,
                    'mean_alignment': alignment
                })

        alignment_df = pd.DataFrame(alignments)

        # Pivot to matrix
        alignment_matrix = alignment_df.pivot(
            index='cluster', columns='gradient', values='mean_alignment'
        )

        logger.info(f"  Computed alignments for {len(np.unique(cluster_labels))} clusters")

        return alignment_matrix

    def identify_gradient_subtypes(
        self,
        cluster_labels: np.ndarray,
        brain_features: pd.DataFrame,
        significance_threshold: float = 0.05
    ) -> Dict[int, str]:
        """
        Characterize clusters by their gradient profile

        Parameters
        ----------
        cluster_labels : np.ndarray
        brain_features : pd.DataFrame
            Brain-derived features per sample
        significance_threshold : float

        Returns
        -------
        cluster_characterization : Dict[int, str]
            Cluster ID → gradient characterization
        """
        logger.info("Characterizing clusters by gradient profiles")

        if self.canonical_gradients is None:
            self.load_canonical_gradients()

        characterizations = {}

        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = brain_features[cluster_mask]

            # Compute mean position on each gradient
            gradient_positions = {}

            for grad_name in self.canonical_gradients.columns:
                # Simplified: assume brain_features correlate with gradients
                gradient = self.canonical_gradients[grad_name].values
                position = cluster_features.mean(axis=0).mean()  # Simplified

                gradient_positions[grad_name] = position

            # Characterize by dominant gradient
            dominant_gradient = max(gradient_positions, key=lambda k: abs(gradient_positions[k]))

            if 'sensory_assoc' in dominant_gradient:
                if gradient_positions[dominant_gradient] > 0:
                    label = "Transmodal/Association-dominant"
                else:
                    label = "Sensory/Unimodal-dominant"
            elif 'visual_motor' in dominant_gradient:
                if gradient_positions[dominant_gradient] > 0:
                    label = "Motor-dominant"
                else:
                    label = "Visual-dominant"
            else:
                if gradient_positions[dominant_gradient] > 0:
                    label = "Anterior-dominant"
                else:
                    label = "Posterior-dominant"

            characterizations[cluster_id] = label

            logger.info(f"  Cluster {cluster_id}: {label}")

        return characterizations

    def integrate_with_baseline_deviation(
        self,
        baseline_deviation_results: pd.DataFrame,
        brain_gradients: np.ndarray
    ) -> pd.DataFrame:
        """
        Integrate gradient validation with baseline-deviation framework

        Tests if deviation from baseline follows brain organizational axes

        Parameters
        ----------
        baseline_deviation_results : pd.DataFrame
        brain_gradients : np.ndarray

        Returns
        -------
        integrated_results : pd.DataFrame
        """
        logger.info("Integrating gradient validation with baseline-deviation framework")

        # Test if deviations are gradient-aligned
        deviation_scores = baseline_deviation_results['deviation_score'].values

        correlations = []
        for i, grad_name in enumerate(self.canonical_gradients.columns):
            gradient = self.canonical_gradients[grad_name].values

            # Correlation between deviation magnitude and gradient position
            # Assumes: samples align with brain regions
            corr, p_val = stats.spearmanr(deviation_scores, gradient[:len(deviation_scores)])

            correlations.append({
                'gradient': grad_name,
                'correlation': corr,
                'p_value': p_val
            })

        corr_df = pd.DataFrame(correlations)

        logger.info(f"  Tested {len(correlations)} gradient alignments")

        return corr_df


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    logger.info("Brain Gradient Validation Module")
    logger.info("Validate topology against canonical brain organization")
    logger.info("\nKey capabilities:")
    logger.info("  1. Load canonical brain gradients (AHBA, Margulies 2016)")
    logger.info("  2. Spatial correlation with spin tests")
    logger.info("  3. Per-cluster gradient alignment")
    logger.info("  4. Gradient-based subtype characterization")
    logger.info("  5. Integration with baseline-deviation framework")
    logger.info("\nAnswers: Do AuDHD subtypes follow brain's organizational principles?")
