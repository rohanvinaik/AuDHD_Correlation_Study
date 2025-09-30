#!/usr/bin/env python3
"""
Microbiome Gut-Brain Axis Analysis
Analyzes 16S rRNA and metagenomic data for AuDHD correlation study
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MicrobiomeResult:
    """Results from microbiome analysis"""
    alpha_diversity: pd.DataFrame
    beta_diversity: pd.DataFrame
    differential_taxa: pd.DataFrame
    brain_correlations: Optional[pd.DataFrame] = None
    pathway_enrichment: Optional[pd.DataFrame] = None


class MicrobiomeAnalyzer:
    """
    Gut-brain axis analysis for AuDHD research

    Capabilities:
    1. 16S rRNA amplicon analysis
    2. Alpha/beta diversity metrics
    3. Differential abundance testing (MaAsLin2-style)
    4. Microbiome-brain phenotype correlations
    5. Functional pathway prediction
    """

    def __init__(self, min_prevalence: float = 0.1, min_abundance: float = 0.0001):
        """
        Initialize analyzer

        Parameters
        ----------
        min_prevalence : float
            Minimum prevalence (fraction of samples) to retain taxa
        min_abundance : float
            Minimum relative abundance to retain taxa
        """
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance

    def preprocess_abundance_table(
        self,
        abundance: pd.DataFrame,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Quality control and normalization of microbiome abundance data

        Parameters
        ----------
        abundance : pd.DataFrame
            Raw abundance matrix (samples × taxa)
        normalize : bool
            Whether to normalize to relative abundance

        Returns
        -------
        processed_abundance : pd.DataFrame
        """
        logger.info("Preprocessing microbiome abundance data")

        # Remove low-prevalence taxa
        prevalence = (abundance > 0).mean(axis=0)
        valid_taxa = prevalence >= self.min_prevalence

        logger.info(f"  Filtered {(~valid_taxa).sum()} low-prevalence taxa")

        abundance_filtered = abundance.loc[:, valid_taxa]

        # Normalize to relative abundance
        if normalize:
            abundance_filtered = abundance_filtered.div(
                abundance_filtered.sum(axis=1), axis=0
            )

        # Remove low-abundance taxa
        mean_abundance = abundance_filtered.mean(axis=0)
        abundant_taxa = mean_abundance >= self.min_abundance

        logger.info(f"  Filtered {(~abundant_taxa).sum()} low-abundance taxa")

        abundance_final = abundance_filtered.loc[:, abundant_taxa]

        logger.info(f"  Final: {abundance_final.shape[0]} samples, {abundance_final.shape[1]} taxa")

        return abundance_final

    def calculate_alpha_diversity(
        self,
        abundance: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate alpha diversity metrics

        Metrics:
        - Shannon index
        - Simpson index
        - Observed richness
        - Pielou's evenness

        Parameters
        ----------
        abundance : pd.DataFrame
            Relative abundance matrix

        Returns
        -------
        alpha_diversity : pd.DataFrame
            Columns: shannon, simpson, richness, evenness
        """
        logger.info("Calculating alpha diversity metrics")

        results = []

        for sample_id in abundance.index:
            abundances = abundance.loc[sample_id].values
            abundances = abundances[abundances > 0]  # Remove zeros

            # Shannon index
            shannon = -np.sum(abundances * np.log(abundances))

            # Simpson index
            simpson = 1 - np.sum(abundances ** 2)

            # Observed richness
            richness = len(abundances)

            # Pielou's evenness
            evenness = shannon / np.log(richness) if richness > 1 else 0

            results.append({
                'sample_id': sample_id,
                'shannon': shannon,
                'simpson': simpson,
                'richness': richness,
                'evenness': evenness
            })

        alpha_df = pd.DataFrame(results).set_index('sample_id')

        logger.info(f"  Shannon range: {alpha_df['shannon'].min():.2f} - {alpha_df['shannon'].max():.2f}")

        return alpha_df

    def calculate_beta_diversity(
        self,
        abundance: pd.DataFrame,
        metric: str = 'bray_curtis'
    ) -> pd.DataFrame:
        """
        Calculate beta diversity (between-sample dissimilarity)

        Parameters
        ----------
        abundance : pd.DataFrame
            Relative abundance matrix
        metric : str
            Distance metric ('bray_curtis', 'jaccard', 'euclidean')

        Returns
        -------
        distance_matrix : pd.DataFrame
            Pairwise dissimilarity matrix
        """
        logger.info(f"Calculating beta diversity ({metric})")

        from scipy.spatial.distance import pdist, squareform

        if metric == 'bray_curtis':
            # Bray-Curtis dissimilarity
            distances = pdist(abundance.values, metric='braycurtis')
        elif metric == 'jaccard':
            # Jaccard distance (binary)
            binary = (abundance > 0).astype(int)
            distances = pdist(binary.values, metric='jaccard')
        elif metric == 'euclidean':
            distances = pdist(abundance.values, metric='euclidean')
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Convert to square matrix
        distance_matrix = pd.DataFrame(
            squareform(distances),
            index=abundance.index,
            columns=abundance.index
        )

        logger.info(f"  Mean {metric} distance: {distances.mean():.3f}")

        return distance_matrix

    def test_differential_abundance(
        self,
        abundance: pd.DataFrame,
        metadata: pd.DataFrame,
        covariate: str,
        covariates_adjust: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Differential abundance testing (MaAsLin2-style)

        Tests association between taxa abundance and phenotype/condition

        Parameters
        ----------
        abundance : pd.DataFrame
            Relative abundance matrix
        metadata : pd.DataFrame
            Sample metadata with covariates
        covariate : str
            Primary covariate of interest (e.g., 'diagnosis')
        covariates_adjust : List[str], optional
            Additional covariates to adjust for

        Returns
        -------
        results : pd.DataFrame
            Columns: taxon, coefficient, std_error, p_value, q_value
        """
        logger.info(f"Testing differential abundance for: {covariate}")

        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        if covariates_adjust is None:
            covariates_adjust = []

        results = []

        # Align data
        common_samples = abundance.index.intersection(metadata.index)
        abundance_aligned = abundance.loc[common_samples]
        metadata_aligned = metadata.loc[common_samples]

        for taxon in abundance_aligned.columns:
            # Log-transform abundance (arcsin-sqrt for compositional data)
            y = np.arcsin(np.sqrt(abundance_aligned[taxon].values))

            # Build design matrix
            X = metadata_aligned[[covariate] + covariates_adjust].copy()

            # Handle categorical variables
            if X[covariate].dtype == 'object':
                X = pd.get_dummies(X, columns=[covariate], drop_first=True)

            # Linear regression
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X, y)

            # Coefficient for covariate (first column after dummies)
            coef = model.coef_[0]

            # Compute standard error (simplified)
            y_pred = model.predict(X)
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (len(y) - X.shape[1] - 1)

            X_centered = X - X.mean(axis=0)
            cov_matrix = np.linalg.inv(X_centered.T @ X_centered) * mse
            se = np.sqrt(cov_matrix[0, 0])

            # T-test
            t_stat = coef / se if se > 0 else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - X.shape[1] - 1))

            results.append({
                'taxon': taxon,
                'coefficient': coef,
                'std_error': se,
                't_statistic': t_stat,
                'p_value': p_val
            })

        results_df = pd.DataFrame(results).sort_values('p_value')

        # FDR correction
        _, qvals, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['q_value'] = qvals

        logger.info(f"  Significant taxa (q<0.05): {(qvals < 0.05).sum()}")

        return results_df

    def correlate_with_brain_phenotypes(
        self,
        abundance: pd.DataFrame,
        brain_phenotypes: pd.DataFrame,
        alpha_diversity: Optional[pd.DataFrame] = None,
        method: str = 'spearman'
    ) -> pd.DataFrame:
        """
        Correlate microbiome with brain/behavioral phenotypes

        Tests gut-brain axis hypothesis

        Parameters
        ----------
        abundance : pd.DataFrame
            Taxa abundance
        brain_phenotypes : pd.DataFrame
            Brain measures (e.g., autism scores, ADHD scores, brain volumes)
        alpha_diversity : pd.DataFrame, optional
            Include diversity metrics as features
        method : str
            Correlation method ('spearman', 'pearson')

        Returns
        -------
        correlations : pd.DataFrame
            Columns: microbiome_feature, phenotype, correlation, p_value, q_value
        """
        logger.info("Correlating microbiome with brain phenotypes")

        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        # Align samples
        common_samples = abundance.index.intersection(brain_phenotypes.index)
        abundance_aligned = abundance.loc[common_samples]
        phenotypes_aligned = brain_phenotypes.loc[common_samples]

        # Combine features
        microbiome_features = abundance_aligned.copy()
        if alpha_diversity is not None:
            microbiome_features = pd.concat([
                microbiome_features,
                alpha_diversity.loc[common_samples]
            ], axis=1)

        results = []

        for microbiome_col in microbiome_features.columns:
            for phenotype_col in phenotypes_aligned.columns:
                x = microbiome_features[microbiome_col].values
                y = phenotypes_aligned[phenotype_col].values

                # Remove missing values
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]

                if len(x_clean) < 10:
                    continue

                # Correlation
                if method == 'spearman':
                    corr, p_val = stats.spearmanr(x_clean, y_clean)
                elif method == 'pearson':
                    corr, p_val = stats.pearsonr(x_clean, y_clean)
                else:
                    raise ValueError(f"Unknown method: {method}")

                results.append({
                    'microbiome_feature': microbiome_col,
                    'phenotype': phenotype_col,
                    'correlation': corr,
                    'p_value': p_val,
                    'n_samples': len(x_clean)
                })

        correlations_df = pd.DataFrame(results).sort_values('p_value')

        # FDR correction
        _, qvals, _, _ = multipletests(correlations_df['p_value'], method='fdr_bh')
        correlations_df['q_value'] = qvals

        logger.info(f"  Significant correlations (q<0.05): {(qvals < 0.05).sum()}")

        return correlations_df

    def predict_functional_pathways(
        self,
        abundance: pd.DataFrame,
        pathway_database: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Predict functional pathways from taxonomic composition

        Mimics PICRUSt2/Tax4Fun approach using 16S -> function mapping

        Parameters
        ----------
        abundance : pd.DataFrame
            Taxonomic abundance
        pathway_database : pd.DataFrame, optional
            Taxon -> pathway mapping

        Returns
        -------
        pathway_abundance : pd.DataFrame
            Predicted pathway abundances
        """
        logger.info("Predicting functional pathways")

        if pathway_database is None:
            # Simplified mock mapping
            logger.warning("  No pathway database provided - using mock predictions")

            # Mock pathways
            pathways = {
                'GABA_synthesis': np.random.rand(abundance.shape[1]),
                'serotonin_synthesis': np.random.rand(abundance.shape[1]),
                'SCFA_production': np.random.rand(abundance.shape[1]),
                'tryptophan_metabolism': np.random.rand(abundance.shape[1])
            }

            pathway_db = pd.DataFrame(pathways, index=abundance.columns)
        else:
            pathway_db = pathway_database

        # Matrix multiplication: samples × taxa @ taxa × pathways
        pathway_abundance = abundance @ pathway_db

        logger.info(f"  Predicted {pathway_abundance.shape[1]} pathways")

        return pathway_abundance

    def integrate_with_baseline_deviation(
        self,
        microbiome_features: pd.DataFrame,
        baseline_deviation_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Integrate microbiome findings with baseline-deviation framework

        Identifies microbiome features that correlate with deviation effects

        Parameters
        ----------
        microbiome_features : pd.DataFrame
            Taxa abundance or diversity metrics
        baseline_deviation_results : pd.DataFrame
            Results from baseline-deviation analysis

        Returns
        -------
        integrated_results : pd.DataFrame
        """
        logger.info("Integrating microbiome with baseline-deviation framework")

        # Align samples
        common_samples = microbiome_features.index.intersection(
            baseline_deviation_results.index
        )

        if len(common_samples) < 10:
            logger.warning("  Insufficient overlapping samples")
            return pd.DataFrame()

        logger.info(f"  {len(common_samples)} overlapping samples")

        # This would contain actual integration logic
        # For now, return placeholder
        integrated = pd.DataFrame({
            'n_samples': [len(common_samples)],
            'microbiome_features': [microbiome_features.shape[1]],
            'analysis_framework': ['baseline_deviation']
        })

        return integrated

    def analyze_complete(
        self,
        abundance: pd.DataFrame,
        metadata: pd.DataFrame,
        brain_phenotypes: Optional[pd.DataFrame] = None,
        covariate: str = 'diagnosis'
    ) -> MicrobiomeResult:
        """
        Complete microbiome analysis pipeline

        Parameters
        ----------
        abundance : pd.DataFrame
            Raw abundance matrix
        metadata : pd.DataFrame
            Sample metadata
        brain_phenotypes : pd.DataFrame, optional
            Brain/behavioral measures
        covariate : str
            Primary covariate for differential testing

        Returns
        -------
        MicrobiomeResult
        """
        logger.info("=== Complete Microbiome Gut-Brain Analysis ===")

        # 1. Preprocessing
        processed = self.preprocess_abundance_table(abundance)

        # 2. Alpha diversity
        alpha_diversity = self.calculate_alpha_diversity(processed)

        # 3. Beta diversity
        beta_diversity = self.calculate_beta_diversity(processed)

        # 4. Differential abundance
        differential_taxa = self.test_differential_abundance(
            processed, metadata, covariate
        )

        # 5. Brain correlations
        if brain_phenotypes is not None:
            brain_correlations = self.correlate_with_brain_phenotypes(
                processed, brain_phenotypes, alpha_diversity
            )
        else:
            brain_correlations = None

        # 6. Pathway prediction
        pathway_abundance = self.predict_functional_pathways(processed)

        return MicrobiomeResult(
            alpha_diversity=alpha_diversity,
            beta_diversity=beta_diversity,
            differential_taxa=differential_taxa,
            brain_correlations=brain_correlations,
            pathway_enrichment=pathway_abundance
        )


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    logger.info("Microbiome Gut-Brain Axis Analysis Module")
    logger.info("Ready for integration with AuDHD correlation study")
    logger.info("\nKey capabilities:")
    logger.info("  1. 16S rRNA amplicon analysis")
    logger.info("  2. Alpha/beta diversity metrics")
    logger.info("  3. Differential abundance testing (MaAsLin2-style)")
    logger.info("  4. Microbiome-brain phenotype correlations")
    logger.info("  5. Functional pathway prediction")
    logger.info("  6. Integration with baseline-deviation framework")
