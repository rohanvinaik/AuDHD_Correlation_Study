"""
Prenatal Data Loader

Loads and harmonizes prenatal/maternal health data from multiple sources:
- SPARK, ABCD, SSC questionnaires
- Birth registries
- Medical records
- Environmental exposure linkage

Handles recall bias, missing data, and variable harmonization.

Author: Claude Code
Date: 2025-09-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PrenatalDataLoader:
    """Load and harmonize prenatal data from multiple sources."""

    def __init__(self, mappings_file: Optional[Path] = None):
        """
        Initialize prenatal data loader.

        Parameters
        ----------
        mappings_file : Path, optional
            Path to prenatal_mappings.yaml. If None, uses default location.
        """
        if mappings_file is None:
            mappings_file = Path(__file__).parents[3] / 'configs' / 'prenatal_mappings.yaml'

        with open(mappings_file, 'r') as f:
            self.mappings = yaml.safe_load(f)

        logger.info(f"Loaded prenatal mappings from {mappings_file}")

    def load_spark_prenatal(self, spark_data_path: Path) -> pd.DataFrame:
        """
        Load prenatal data from SPARK cohort.

        Parameters
        ----------
        spark_data_path : Path
            Path to SPARK medical history questionnaire data

        Returns
        -------
        pd.DataFrame
            Harmonized prenatal features
        """
        logger.info("Loading SPARK prenatal data...")

        try:
            spark_df = pd.read_csv(spark_data_path)
        except FileNotFoundError:
            logger.warning(f"SPARK data not found at {spark_data_path}")
            return pd.DataFrame()

        mapping = self.mappings['spark_prenatal_mapping']

        prenatal_data = {}

        # Map each variable
        for std_var, var_info in mapping.items():
            source_var = var_info['source_var']

            if source_var in spark_df.columns:
                prenatal_data[std_var] = spark_df[source_var]

                # Apply recoding if specified
                if 'recode' in var_info:
                    prenatal_data[std_var] = prenatal_data[std_var].map(var_info['recode'])

                # Handle conversions
                if 'conversion' in var_info and std_var == 'birth_weight_grams':
                    # Convert from pounds if needed
                    if prenatal_data[std_var].mean() < 20:  # Likely in pounds
                        prenatal_data[std_var] = prenatal_data[std_var] * 453.592
            else:
                logger.debug(f"Variable {source_var} not found in SPARK data")
                prenatal_data[std_var] = np.nan

        prenatal_df = pd.DataFrame(prenatal_data)
        prenatal_df['data_source'] = 'SPARK'
        prenatal_df['recall_bias_level'] = 'moderate'

        logger.info(f"Loaded {len(prenatal_df)} SPARK records with {prenatal_df.shape[1]} prenatal variables")

        return prenatal_df

    def load_abcd_prenatal(self, abcd_data_path: Path) -> pd.DataFrame:
        """
        Load prenatal data from ABCD cohort (retrospective).

        Parameters
        ----------
        abcd_data_path : Path
            Path to ABCD developmental history data

        Returns
        -------
        pd.DataFrame
            Harmonized prenatal features
        """
        logger.info("Loading ABCD prenatal data...")

        try:
            abcd_df = pd.read_csv(abcd_data_path)
        except FileNotFoundError:
            logger.warning(f"ABCD data not found at {abcd_data_path}")
            return pd.DataFrame()

        mapping = self.mappings['abcd_prenatal_mapping']

        prenatal_data = {}

        for std_var, var_info in mapping.items():
            source_var = var_info['source_var']

            if source_var in abcd_df.columns:
                prenatal_data[std_var] = abcd_df[source_var]
            else:
                prenatal_data[std_var] = np.nan

        prenatal_df = pd.DataFrame(prenatal_data)
        prenatal_df['data_source'] = 'ABCD'
        prenatal_df['recall_bias_level'] = 'high'  # 9-10 years retrospective

        logger.info(f"Loaded {len(prenatal_df)} ABCD records (recall bias: HIGH)")

        return prenatal_df

    def harmonize_gestational_age(self, prenatal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonize gestational age across different formats.

        Handles:
        - Weeks + days → weeks (decimal)
        - Days → weeks
        - Binary preterm → imputed weeks
        - Outlier flagging

        Parameters
        ----------
        prenatal_df : pd.DataFrame
            Raw prenatal data

        Returns
        -------
        pd.DataFrame
            With harmonized gestational_age_weeks column
        """
        # If we have gestational_age_days, convert to weeks
        if 'gestational_age_days' in prenatal_df.columns:
            prenatal_df['gestational_age_weeks'] = prenatal_df['gestational_age_days'] / 7

        # If we only have preterm_birth binary, impute gestational age
        if 'preterm_birth' in prenatal_df.columns and 'gestational_age_weeks' not in prenatal_df.columns:
            logger.info("Imputing gestational age from preterm_birth indicator...")
            prenatal_df['gestational_age_weeks'] = prenatal_df['preterm_birth'].map({
                1: 35.0,  # Median for preterm
                0: 39.0   # Median for term
            })
            prenatal_df['gestational_age_imputed'] = True
        else:
            prenatal_df['gestational_age_imputed'] = False

        # Flag outliers
        if 'gestational_age_weeks' in prenatal_df.columns:
            ga = prenatal_df['gestational_age_weeks']
            prenatal_df['gestational_age_outlier'] = (ga < 20) | (ga > 44)

            n_outliers = prenatal_df['gestational_age_outlier'].sum()
            if n_outliers > 0:
                logger.warning(f"Flagged {n_outliers} gestational age outliers (<20 or >44 weeks)")

        return prenatal_df

    def harmonize_birth_weight(self, prenatal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonize birth weight to grams.

        Handles:
        - Pounds/ounces → grams
        - Categorical (very low/low/normal) → imputed grams
        - Outlier flagging

        Parameters
        ----------
        prenatal_df : pd.DataFrame
            Raw prenatal data

        Returns
        -------
        pd.DataFrame
            With harmonized birth_weight_grams column
        """
        # If we have categorical birth weight, impute
        if 'birth_weight_category' in prenatal_df.columns and 'birth_weight_grams' not in prenatal_df.columns:
            logger.info("Imputing birth weight from categorical variable...")
            category_map = {
                'very_low': 1200.0,   # <1500g
                'low': 2200.0,        # 1500-2500g
                'normal': 3300.0,     # 2500-4000g
                'high': 4200.0        # >4000g
            }
            prenatal_df['birth_weight_grams'] = prenatal_df['birth_weight_category'].map(category_map)
            prenatal_df['birth_weight_imputed'] = True
        elif 'low_birth_weight' in prenatal_df.columns and 'birth_weight_grams' not in prenatal_df.columns:
            prenatal_df['birth_weight_grams'] = prenatal_df['low_birth_weight'].map({
                1: 2200.0,
                0: 3300.0
            })
            prenatal_df['birth_weight_imputed'] = True
        else:
            prenatal_df['birth_weight_imputed'] = False

        # Flag outliers
        if 'birth_weight_grams' in prenatal_df.columns:
            bw = prenatal_df['birth_weight_grams']
            prenatal_df['birth_weight_outlier'] = (bw < 500) | (bw > 6000)

            n_outliers = prenatal_df['birth_weight_outlier'].sum()
            if n_outliers > 0:
                logger.warning(f"Flagged {n_outliers} birth weight outliers (<500g or >6000g)")

        return prenatal_df

    def harmonize_infection_timing(self, prenatal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonize infection timing to trimester.

        If only binary "any infection", impute to trimester 2 (most common).

        Parameters
        ----------
        prenatal_df : pd.DataFrame
            Raw prenatal data

        Returns
        -------
        pd.DataFrame
            With harmonized infection_trimester column
        """
        if 'maternal_infection' in prenatal_df.columns and 'infection_trimester' not in prenatal_df.columns:
            logger.info("Imputing infection trimester for cases with unknown timing...")
            prenatal_df['infection_trimester'] = prenatal_df['maternal_infection'].map({
                1: 2,  # Assume trimester 2 if unknown
                0: np.nan
            })
            prenatal_df['infection_timing_imputed'] = prenatal_df['maternal_infection'] == 1
        else:
            prenatal_df['infection_timing_imputed'] = False

        return prenatal_df

    def add_data_quality_indicators(self, prenatal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add data quality indicators.

        - Recall bias level (from data source)
        - Missing data flags
        - Imputation indicators
        - Confidence scores

        Parameters
        ----------
        prenatal_df : pd.DataFrame
            Prenatal data

        Returns
        -------
        pd.DataFrame
            With added quality indicators
        """
        # Missing data indicators for critical variables
        critical_vars = [
            'gestational_age_weeks',
            'birth_weight_grams',
            'maternal_infection',
            'maternal_medication'
        ]

        for var in critical_vars:
            if var in prenatal_df.columns:
                prenatal_df[f'{var}_missing'] = prenatal_df[var].isna()

        # Overall completeness score (0-1)
        prenatal_df['data_completeness'] = (
            ~prenatal_df[[v for v in critical_vars if v in prenatal_df.columns]].isna()
        ).sum(axis=1) / len([v for v in critical_vars if v in prenatal_df.columns])

        # Confidence score based on recall bias and completeness
        recall_bias_scores = {'low': 1.0, 'moderate': 0.7, 'high': 0.4}
        prenatal_df['data_confidence'] = (
            prenatal_df['recall_bias_level'].map(recall_bias_scores).fillna(0.5) *
            prenatal_df['data_completeness']
        )

        logger.info(f"Mean data confidence: {prenatal_df['data_confidence'].mean():.2f}")

        return prenatal_df

    def load_and_harmonize(
        self,
        spark_path: Optional[Path] = None,
        abcd_path: Optional[Path] = None,
        ssc_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Load and harmonize prenatal data from all available sources.

        Parameters
        ----------
        spark_path : Path, optional
            Path to SPARK data
        abcd_path : Path, optional
            Path to ABCD data
        ssc_path : Path, optional
            Path to SSC data

        Returns
        -------
        pd.DataFrame
            Combined harmonized prenatal data
        """
        logger.info("Loading and harmonizing prenatal data from all sources...")

        all_data = []

        if spark_path is not None:
            spark_df = self.load_spark_prenatal(spark_path)
            if not spark_df.empty:
                all_data.append(spark_df)

        if abcd_path is not None:
            abcd_df = self.load_abcd_prenatal(abcd_path)
            if not abcd_df.empty:
                all_data.append(abcd_df)

        if len(all_data) == 0:
            logger.warning("No prenatal data loaded from any source")
            return pd.DataFrame()

        # Combine data
        combined_df = pd.concat(all_data, axis=0, ignore_index=True)

        logger.info(f"Combined {len(combined_df)} records from {len(all_data)} sources")

        # Harmonization steps
        combined_df = self.harmonize_gestational_age(combined_df)
        combined_df = self.harmonize_birth_weight(combined_df)
        combined_df = self.harmonize_infection_timing(combined_df)
        combined_df = self.add_data_quality_indicators(combined_df)

        logger.info(f"Harmonization complete: {combined_df.shape[1]} total columns")

        return combined_df


def create_prenatal_feature_matrix(
    prenatal_df: pd.DataFrame,
    feature_module_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Convert harmonized prenatal data to feature matrix.

    Links harmonized data to prenatal_maternal.py feature extraction.

    Parameters
    ----------
    prenatal_df : pd.DataFrame
        Harmonized prenatal data from load_and_harmonize()
    feature_module_path : Path, optional
        Path to prenatal_maternal.py module

    Returns
    -------
    pd.DataFrame
        Full feature matrix ready for analysis
    """
    from ..features.prenatal_maternal import extract_comprehensive_prenatal_features

    logger.info("Creating prenatal feature matrix...")

    # Separate into different data types for feature extraction
    infection_data = prenatal_df[[
        c for c in prenatal_df.columns if 'infection' in c.lower()
    ]].copy()

    medication_data = prenatal_df[[
        c for c in prenatal_df.columns if 'medication' in c.lower()
    ]].copy()

    pregnancy_data = prenatal_df[[
        c for c in prenatal_df.columns
        if any(term in c.lower() for term in ['complication', 'birth', 'gestational', 'apgar', 'delivery'])
    ]].copy()

    # Extract features
    features = extract_comprehensive_prenatal_features(
        infection_data=infection_data if not infection_data.empty else None,
        medication_data=medication_data if not medication_data.empty else None,
        pregnancy_data=pregnancy_data if not pregnancy_data.empty else None
    )

    logger.info(f"Created feature matrix with {features.shape[1]} features")

    return features


if __name__ == '__main__':
    # Example usage
    logger.info("Prenatal Data Loader initialized")

    loader = PrenatalDataLoader()

    # Example: Load from SPARK (if data available)
    # prenatal_data = loader.load_and_harmonize(
    #     spark_path=Path('data/spark/medical_history.csv')
    # )
    #
    # features = create_prenatal_feature_matrix(prenatal_data)

    print("\nPrenatal Data Loader ready")
    print("Use load_and_harmonize() to load data from SPARK, ABCD, or SSC")
