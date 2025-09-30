#!/usr/bin/env python3
"""
ABCD Imaging Metrics Extractor

Extracts and summarizes neuroimaging metrics from ABCD Study data,
focusing on ADHD/Autism-relevant brain regions and networks.

Generates:
- imaging_metrics_extracted.csv: Subject-level imaging data
- imaging_summary_statistics.csv: Descriptive statistics
- imaging_group_comparisons.csv: ADHD vs Controls, Autism vs Controls

Usage:
    # Extract from processed ABCD data
    python extract_imaging_metrics.py \
        --input data/abcd/processed/abcd_merged_baseline_year_1_arm_1.csv \
        --output data/abcd/imaging_metrics_extracted.csv

    # Include group comparisons
    python extract_imaging_metrics.py \
        --input data/abcd/processed/abcd_merged_baseline_year_1_arm_1.csv \
        --output data/abcd/imaging_metrics_extracted.csv \
        --compare-groups

    # Export network-level summaries
    python extract_imaging_metrics.py \
        --input data/abcd/processed/abcd_merged_baseline_year_1_arm_1.csv \
        --network-summary

Author: AuDHD Correlation Study Team
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

try:
    import pandas as pd
    import numpy as np
    from scipy import stats
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install pandas numpy scipy")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ADHD-relevant brain regions
ADHD_REGIONS = {
    'basal_ganglia': [
        'smri_vol_scs_caudatelh',
        'smri_vol_scs_caudaterh',
        'smri_vol_scs_putamenlh',
        'smri_vol_scs_putamenrh',
        'smri_vol_scs_pallidumlh',
        'smri_vol_scs_pallidumrh'
    ],
    'prefrontal': [
        'smri_thick_cdk_cdacatelh',  # Caudal anterior cingulate
        'smri_thick_cdk_cdacaterh',
        'smri_thick_cdk_rracatelh',  # Rostral anterior cingulate
        'smri_thick_cdk_rracaterh',
        'smri_thick_cdk_superiorfrontallh',
        'smri_thick_cdk_superiorfrontalrh'
    ],
    'cerebellum': [
        'smri_vol_scs_cbwmatterlh',  # Cerebellar white matter
        'smri_vol_scs_cbwmatterrh'
    ]
}

# Autism-relevant brain regions
AUTISM_REGIONS = {
    'social_brain': [
        'smri_vol_scs_amygdalalh',
        'smri_vol_scs_amygdalarh',
        'smri_thick_cdk_superiortemporal',
        'smri_thick_cdk_fusiform'
    ],
    'limbic': [
        'smri_vol_scs_hpuslh',  # Hippocampus
        'smri_vol_scs_hpusrh',
        'smri_vol_scs_amygdalalh',
        'smri_vol_scs_amygdalarh'
    ]
}

# Functional connectivity networks
NETWORKS = {
    'default_mode': ['dmn', 'default'],
    'salience': ['sal', 'salience'],
    'executive': ['exec', 'fpn', 'frontoparietal'],
    'attention': ['attn', 'dorsal_attention']
}


class ImagingMetricsExtractor:
    """Extract neuroimaging metrics from ABCD data"""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize extractor

        Args:
            data: Merged ABCD dataframe with imaging variables
        """
        self.data = data.copy()
        self.subject_id_col = 'src_subject_id'

        # Identify available imaging modalities
        self.has_structural = self._check_structural()
        self.has_connectivity = self._check_connectivity()
        self.has_dti = self._check_dti()

        logger.info(f"Initialized extractor: {len(data)} subjects")
        logger.info(f"Structural MRI: {self.has_structural}")
        logger.info(f"Connectivity: {self.has_connectivity}")
        logger.info(f"DTI: {self.has_dti}")

    def _check_structural(self) -> bool:
        """Check if structural MRI data is available"""
        struct_cols = [col for col in self.data.columns if 'smri' in col]
        return len(struct_cols) > 0

    def _check_connectivity(self) -> bool:
        """Check if connectivity data is available"""
        conn_cols = [col for col in self.data.columns
                    if 'rsfmri' in col or 'connectivity' in col]
        return len(conn_cols) > 0

    def _check_dti(self) -> bool:
        """Check if DTI data is available"""
        dti_cols = [col for col in self.data.columns
                   if any(x in col for x in ['dmri', 'dti', '_fa_', '_md_'])]
        return len(dti_cols) > 0

    def extract_structural_metrics(self) -> pd.DataFrame:
        """
        Extract structural MRI metrics

        Returns:
            DataFrame with structural metrics
        """
        if not self.has_structural:
            logger.warning("No structural MRI data available")
            return pd.DataFrame()

        logger.info("Extracting structural MRI metrics...")

        # Start with subject IDs
        metrics = self.data[[self.subject_id_col]].copy()

        # Global measures
        global_vars = [
            'smri_thick_cdk_mean',
            'smri_area_cdk_total',
            'smri_vol_cdk_total',
            'smri_vol_scs_intracranialv'
        ]

        for var in global_vars:
            if var in self.data.columns:
                metrics[var] = self.data[var]

        # ADHD-relevant regions
        for region_name, region_vars in ADHD_REGIONS.items():
            available_vars = [v for v in region_vars if v in self.data.columns]

            if available_vars:
                # Calculate mean for region
                metrics[f'adhd_{region_name}_mean'] = self.data[available_vars].mean(axis=1)

                # Calculate bilateral averages where applicable
                lh_vars = [v for v in available_vars if 'lh' in v]
                rh_vars = [v for v in available_vars if 'rh' in v]

                if lh_vars and rh_vars:
                    # Match hemisphere pairs
                    for lh_var in lh_vars:
                        rh_var = lh_var.replace('lh', 'rh')
                        if rh_var in rh_vars:
                            bilateral_name = lh_var.replace('lh', 'bilateral').replace('smri_', '')
                            metrics[bilateral_name] = (self.data[lh_var] + self.data[rh_var]) / 2

        # Autism-relevant regions
        for region_name, region_vars in AUTISM_REGIONS.items():
            available_vars = [v for v in region_vars if v in self.data.columns]

            if available_vars:
                metrics[f'autism_{region_name}_mean'] = self.data[available_vars].mean(axis=1)

        # Normalize volumes by intracranial volume if available
        if 'smri_vol_scs_intracranialv' in metrics.columns:
            vol_cols = [col for col in metrics.columns if 'vol' in col and col != 'smri_vol_scs_intracranialv']

            for col in vol_cols:
                normalized_col = col.replace('vol', 'vol_norm')
                metrics[normalized_col] = metrics[col] / metrics['smri_vol_scs_intracranialv']

        logger.info(f"Extracted structural metrics: {len(metrics.columns)} variables")
        return metrics

    def extract_connectivity_metrics(self) -> pd.DataFrame:
        """
        Extract functional connectivity metrics

        Returns:
            DataFrame with connectivity metrics
        """
        if not self.has_connectivity:
            logger.warning("No connectivity data available")
            return pd.DataFrame()

        logger.info("Extracting connectivity metrics...")

        # Start with subject IDs
        metrics = self.data[[self.subject_id_col]].copy()

        # Get all connectivity columns
        conn_cols = [col for col in self.data.columns
                    if 'rsfmri' in col or 'connectivity' in col]

        # Calculate network-level summaries
        for network_name, network_patterns in NETWORKS.items():
            # Find columns matching this network
            network_cols = []
            for col in conn_cols:
                if any(pattern in col.lower() for pattern in network_patterns):
                    network_cols.append(col)

            if network_cols:
                # Within-network connectivity (mean)
                metrics[f'{network_name}_connectivity_mean'] = self.data[network_cols].mean(axis=1)

                # Within-network variability (std)
                metrics[f'{network_name}_connectivity_std'] = self.data[network_cols].std(axis=1)

        # Global connectivity metrics
        if conn_cols:
            metrics['global_connectivity_mean'] = self.data[conn_cols].mean(axis=1)
            metrics['global_connectivity_std'] = self.data[conn_cols].std(axis=1)

            # Count edges with strong connectivity (|r| > 0.3)
            strong_edges = (self.data[conn_cols].abs() > 0.3).sum(axis=1)
            metrics['strong_connectivity_edges'] = strong_edges

        logger.info(f"Extracted connectivity metrics: {len(metrics.columns)} variables")
        return metrics

    def extract_dti_metrics(self) -> pd.DataFrame:
        """
        Extract DTI metrics

        Returns:
            DataFrame with DTI metrics
        """
        if not self.has_dti:
            logger.warning("No DTI data available")
            return pd.DataFrame()

        logger.info("Extracting DTI metrics...")

        # Start with subject IDs
        metrics = self.data[[self.subject_id_col]].copy()

        # Get FA and MD columns
        fa_cols = [col for col in self.data.columns if '_fa_' in col or col.endswith('_fa')]
        md_cols = [col for col in self.data.columns if '_md_' in col or col.endswith('_md')]

        # Calculate global FA/MD
        if fa_cols:
            metrics['dti_fa_global_mean'] = self.data[fa_cols].mean(axis=1)
            metrics['dti_fa_global_std'] = self.data[fa_cols].std(axis=1)

        if md_cols:
            metrics['dti_md_global_mean'] = self.data[md_cols].mean(axis=1)
            metrics['dti_md_global_std'] = self.data[md_cols].std(axis=1)

        # Tract-specific metrics (if column names allow identification)
        tracts = {
            'corpus_callosum': ['cc', 'callosum'],
            'corona_radiata': ['cr', 'corona'],
            'superior_longitudinal': ['slf', 'longitudinal'],
            'internal_capsule': ['ic', 'capsule']
        }

        for tract_name, tract_patterns in tracts.items():
            tract_fa_cols = [col for col in fa_cols
                           if any(pattern in col.lower() for pattern in tract_patterns)]
            tract_md_cols = [col for col in md_cols
                           if any(pattern in col.lower() for pattern in tract_patterns)]

            if tract_fa_cols:
                metrics[f'{tract_name}_fa_mean'] = self.data[tract_fa_cols].mean(axis=1)

            if tract_md_cols:
                metrics[f'{tract_name}_md_mean'] = self.data[tract_md_cols].mean(axis=1)

        logger.info(f"Extracted DTI metrics: {len(metrics.columns)} variables")
        return metrics

    def merge_all_metrics(self) -> pd.DataFrame:
        """
        Merge all imaging metrics

        Returns:
            Complete imaging metrics dataframe
        """
        logger.info("Merging all imaging metrics...")

        # Extract each modality
        structural = self.extract_structural_metrics()
        connectivity = self.extract_connectivity_metrics()
        dti = self.extract_dti_metrics()

        # Start with subjects
        merged = self.data[[self.subject_id_col]].copy()

        # Add diagnosis/phenotype columns if available
        pheno_cols = ['ksads_adhd_diagnosis', 'ksads_asd_diagnosis',
                     'cbcl_adhd_risk', 'cbcl_social_problems',
                     'med_adhd_any']

        for col in pheno_cols:
            if col in self.data.columns:
                merged[col] = self.data[col]

        # Add demographic variables
        demo_cols = ['age', 'sex', 'race', 'ethnicity']
        for col in demo_cols:
            matching_cols = [c for c in self.data.columns if col in c.lower()]
            if matching_cols:
                merged[col] = self.data[matching_cols[0]]

        # Merge imaging metrics
        if not structural.empty:
            merged = merged.merge(structural, on=self.subject_id_col, how='left')

        if not connectivity.empty:
            merged = merged.merge(connectivity, on=self.subject_id_col, how='left')

        if not dti.empty:
            merged = merged.merge(dti, on=self.subject_id_col, how='left')

        # Add data availability flags
        merged['has_structural_mri'] = (~structural.empty) and \
                                       (structural.iloc[:, 1:].notna().any(axis=1))
        merged['has_connectivity'] = (~connectivity.empty) and \
                                     (connectivity.iloc[:, 1:].notna().any(axis=1))
        merged['has_dti'] = (~dti.empty) and (dti.iloc[:, 1:].notna().any(axis=1))

        logger.info(f"Merged imaging metrics: {len(merged)} subjects, {len(merged.columns)} variables")
        return merged

    def calculate_summary_statistics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate summary statistics for imaging metrics

        Args:
            metrics: Imaging metrics dataframe

        Returns:
            Summary statistics dataframe
        """
        logger.info("Calculating summary statistics...")

        # Select only numeric columns (exclude subject ID and phenotypes)
        imaging_cols = metrics.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude phenotype/demo columns
        exclude_patterns = ['diagnosis', 'risk', 'problems', 'age', 'sex', 'has_']
        imaging_cols = [col for col in imaging_cols
                       if not any(pattern in col for pattern in exclude_patterns)]

        if not imaging_cols:
            return pd.DataFrame()

        # Calculate statistics
        summary = pd.DataFrame({
            'variable': imaging_cols,
            'n': metrics[imaging_cols].notna().sum().values,
            'mean': metrics[imaging_cols].mean().values,
            'std': metrics[imaging_cols].std().values,
            'median': metrics[imaging_cols].median().values,
            'min': metrics[imaging_cols].min().values,
            'max': metrics[imaging_cols].max().values,
            'q25': metrics[imaging_cols].quantile(0.25).values,
            'q75': metrics[imaging_cols].quantile(0.75).values
        })

        # Add modality labels
        summary['modality'] = summary['variable'].apply(self._classify_modality)

        return summary

    def compare_groups(self, metrics: pd.DataFrame,
                      group_col: str = 'ksads_adhd_diagnosis') -> pd.DataFrame:
        """
        Compare imaging metrics between groups (e.g., ADHD vs controls)

        Args:
            metrics: Imaging metrics dataframe
            group_col: Column defining groups (0/1)

        Returns:
            Group comparison statistics
        """
        if group_col not in metrics.columns:
            logger.warning(f"Group column not found: {group_col}")
            return pd.DataFrame()

        logger.info(f"Comparing groups: {group_col}")

        # Get imaging columns
        imaging_cols = metrics.select_dtypes(include=[np.number]).columns.tolist()
        exclude_patterns = ['diagnosis', 'risk', 'problems', 'age', 'sex', 'has_']
        imaging_cols = [col for col in imaging_cols
                       if not any(pattern in col for pattern in exclude_patterns)]

        if not imaging_cols:
            return pd.DataFrame()

        # Split into groups
        group1 = metrics[metrics[group_col] == 1]
        group0 = metrics[metrics[group_col] == 0]

        comparisons = []

        for col in imaging_cols:
            # Get valid data
            g1_data = group1[col].dropna()
            g0_data = group0[col].dropna()

            if len(g1_data) < 10 or len(g0_data) < 10:
                continue  # Need minimum sample size

            # T-test
            t_stat, p_val = stats.ttest_ind(g1_data, g0_data, equal_var=False)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((g1_data.std()**2 + g0_data.std()**2) / 2)
            cohens_d = (g1_data.mean() - g0_data.mean()) / pooled_std if pooled_std > 0 else 0

            comparisons.append({
                'variable': col,
                'group1_n': len(g1_data),
                'group1_mean': g1_data.mean(),
                'group1_std': g1_data.std(),
                'group0_n': len(g0_data),
                'group0_mean': g0_data.mean(),
                'group0_std': g0_data.std(),
                'mean_difference': g1_data.mean() - g0_data.mean(),
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'significant': p_val < 0.05
            })

        comparison_df = pd.DataFrame(comparisons)

        if not comparison_df.empty:
            # Sort by effect size
            comparison_df = comparison_df.sort_values('cohens_d', key=abs, ascending=False)

        return comparison_df

    @staticmethod
    def _classify_modality(variable_name: str) -> str:
        """Classify variable into imaging modality"""
        if 'smri' in variable_name or 'vol' in variable_name or 'thick' in variable_name:
            return 'Structural MRI'
        elif 'rsfmri' in variable_name or 'connectivity' in variable_name:
            return 'Functional Connectivity'
        elif 'dti' in variable_name or 'dmri' in variable_name or '_fa' in variable_name or '_md' in variable_name:
            return 'DTI'
        else:
            return 'Other'


def main():
    parser = argparse.ArgumentParser(
        description='Extract neuroimaging metrics from ABCD data'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file with merged ABCD data'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/abcd/imaging_metrics_extracted.csv',
        help='Output CSV file for imaging metrics'
    )

    parser.add_argument(
        '--compare-groups',
        action='store_true',
        help='Generate group comparison statistics'
    )

    parser.add_argument(
        '--network-summary',
        action='store_true',
        help='Generate network-level summary statistics'
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from: {args.input}")
    data = pd.read_csv(args.input, low_memory=False)

    # Initialize extractor
    extractor = ImagingMetricsExtractor(data)

    # Extract all metrics
    metrics = extractor.merge_all_metrics()

    # Save main output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    logger.info(f"Saved imaging metrics: {output_path}")

    # Generate summary statistics
    if args.network_summary:
        summary = extractor.calculate_summary_statistics(metrics)
        summary_path = output_path.parent / 'imaging_summary_statistics.csv'
        summary.to_csv(summary_path, index=False)
        logger.info(f"Saved summary statistics: {summary_path}")

    # Generate group comparisons
    if args.compare_groups:
        # ADHD vs controls
        if 'ksads_adhd_diagnosis' in metrics.columns:
            adhd_comparison = extractor.compare_groups(metrics, 'ksads_adhd_diagnosis')
            adhd_path = output_path.parent / 'imaging_adhd_vs_controls.csv'
            adhd_comparison.to_csv(adhd_path, index=False)
            logger.info(f"Saved ADHD comparison: {adhd_path}")

        # Autism vs controls
        if 'ksads_asd_diagnosis' in metrics.columns:
            asd_comparison = extractor.compare_groups(metrics, 'ksads_asd_diagnosis')
            asd_path = output_path.parent / 'imaging_autism_vs_controls.csv'
            asd_comparison.to_csv(asd_path, index=False)
            logger.info(f"Saved autism comparison: {asd_path}")

    logger.info("Extraction complete!")


if __name__ == '__main__':
    main()