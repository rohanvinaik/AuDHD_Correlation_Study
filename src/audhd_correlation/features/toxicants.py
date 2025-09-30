#!/usr/bin/env python3
"""
Toxicant & Biomarker Analysis Pipeline

Processes biomarker data from biological samples:
- Heavy metals (hair, blood, urine, nail)
- Organic pollutants (phthalates, BPA, pesticides, PFAS)
- Body burden indices and mixture analysis

Implements quality control, normalization, and reference range comparisons
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeavyMetalAnalyzer:
    """
    Analyze heavy metal biomarkers from hair, blood, urine, or nail samples

    Handles essential and toxic metals with age/sex-specific reference ranges
    """

    def __init__(self, matrix: str = 'hair'):
        """
        Initialize heavy metal analyzer

        Args:
            matrix: Sample type ('hair', 'blood', 'urine', 'nail')
        """
        self.matrix = matrix

        # Essential metals
        self.essential_metals = ['Zn', 'Cu', 'Se', 'Fe', 'Mn', 'Cr', 'Mo', 'I']

        # Toxic metals
        self.toxic_metals = ['Pb', 'Hg', 'Cd', 'As', 'Al', 'Tl', 'Sb', 'Ni']

        # Important ratios
        self.metal_ratios = [
            ('Cu', 'Zn'),   # Cu/Zn (elevated in inflammation)
            ('Ca', 'Mg'),   # Ca/Mg (mineral balance)
            ('Na', 'K'),    # Na/K (electrolyte balance)
            ('Hg', 'Se'),   # Hg/Se (selenium protective against mercury)
        ]

        # Reference ranges by matrix (approximate, vary by lab and age)
        self.reference_ranges = self._load_reference_ranges()

        # Sample-specific processing
        self.matrix_protocols = {
            'hair': {
                'segment_length_cm': 3,  # Proximal 3cm (~3 months)
                'washing_protocol': 'IAEA_standard',
                'analysis_method': 'ICP-MS',
                'units': 'ug/g'
            },
            'blood': {
                'sample_type': 'whole_blood',
                'analysis_method': 'ICP-MS',
                'units': 'ug/L',
                'fasting_required': False
            },
            'urine': {
                'collection': 'spot_or_24h',
                'analysis_method': 'ICP-MS',
                'units': 'ug/L',
                'creatinine_adjustment': True
            },
            'nail': {
                'collection': 'fingernail_clippings',
                'analysis_method': 'ICP-MS',
                'units': 'ug/g'
            }
        }

    def _load_reference_ranges(self) -> Dict:
        """Load reference ranges for metals by matrix"""
        # Approximate reference ranges (would load from database in production)
        ranges = {
            'hair': {
                # Units: ug/g (ppm)
                'Pb': {'low': 0, 'high': 2.0, 'concern': 5.0},
                'Hg': {'low': 0, 'high': 1.0, 'concern': 2.0},
                'Cd': {'low': 0, 'high': 0.2, 'concern': 0.5},
                'As': {'low': 0, 'high': 0.5, 'concern': 1.0},
                'Al': {'low': 0, 'high': 10.0, 'concern': 20.0},
                'Zn': {'low': 100, 'high': 200, 'concern': None},
                'Cu': {'low': 10, 'high': 40, 'concern': None},
                'Se': {'low': 0.5, 'high': 2.0, 'concern': None},
                'Fe': {'low': 10, 'high': 50, 'concern': None}
            },
            'blood': {
                # Units: ug/L (ppb)
                'Pb': {'low': 0, 'high': 5.0, 'concern': 10.0},  # CDC: <5 ug/dL in children
                'Hg': {'low': 0, 'high': 5.8, 'concern': 10.0},
                'Cd': {'low': 0, 'high': 1.0, 'concern': 5.0},
                'As': {'low': 0, 'high': 7.0, 'concern': 15.0}
            },
            'urine': {
                # Units: ug/g creatinine
                'Pb': {'low': 0, 'high': 2.0, 'concern': 5.0},
                'Hg': {'low': 0, 'high': 2.0, 'concern': 5.0},
                'Cd': {'low': 0, 'high': 1.0, 'concern': 2.0},
                'As': {'low': 0, 'high': 50, 'concern': 100}  # Total arsenic
            }
        }

        return ranges.get(self.matrix, {})

    def load_metal_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load heavy metal biomarker data

        Expected columns:
        - participant_id
        - sample_date
        - metal_name (Pb, Hg, Cd, etc.)
        - concentration
        - units
        - (creatinine_mg_dL if urine)

        Returns:
            DataFrame with metal concentrations
        """
        df = pd.read_csv(file_path)

        # Validate required columns
        required = ['participant_id', 'metal_name', 'concentration']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return df

    def handle_below_lod(self, df: pd.DataFrame, lod_col: str = 'LOD') -> pd.DataFrame:
        """
        Handle values below limit of detection (LOD)

        Uses LOD/sqrt(2) imputation (standard EPA method)

        Args:
            df: DataFrame with concentration and LOD columns
            lod_col: Name of LOD column

        Returns:
            DataFrame with imputed values
        """
        df = df.copy()

        # Identify below LOD
        if lod_col in df.columns:
            below_lod = df['concentration'] < df[lod_col]

            # Impute as LOD/sqrt(2)
            df.loc[below_lod, 'concentration'] = df.loc[below_lod, lod_col] / np.sqrt(2)
            df.loc[below_lod, 'below_lod'] = True
        else:
            df['below_lod'] = False

        return df

    def creatinine_adjust(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust urine concentrations for creatinine (dilution correction)

        Args:
            df: DataFrame with concentration and creatinine columns

        Returns:
            DataFrame with creatinine-adjusted concentrations
        """
        if self.matrix != 'urine':
            return df

        if 'creatinine_mg_dL' not in df.columns:
            logger.warning("Creatinine data not available for adjustment")
            return df

        df = df.copy()

        # Convert to ug/g creatinine
        # concentration (ug/L) / creatinine (mg/dL) * 10 = ug/g creatinine
        df['concentration_adj'] = (df['concentration'] / df['creatinine_mg_dL']) * 10

        # Flag very dilute or concentrated samples
        df['creatinine_flag'] = (
            (df['creatinine_mg_dL'] < 30) |  # Too dilute
            (df['creatinine_mg_dL'] > 300)   # Too concentrated
        )

        return df

    def apply_reference_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare measured values to reference ranges

        Categorizes as normal, elevated, or concern level

        Args:
            df: DataFrame with metal concentrations

        Returns:
            DataFrame with reference range categories
        """
        df = df.copy()

        categories = []

        for _, row in df.iterrows():
            metal = row['metal_name']
            conc = row['concentration_adj'] if 'concentration_adj' in df.columns else row['concentration']

            if metal in self.reference_ranges:
                ref = self.reference_ranges[metal]

                if pd.isna(conc):
                    category = 'missing'
                elif conc <= ref['high']:
                    category = 'normal'
                elif ref['concern'] and conc >= ref['concern']:
                    category = 'concern'
                else:
                    category = 'elevated'

                # Percentile relative to reference range
                percentile = (conc / ref['high']) * 100 if ref['high'] > 0 else np.nan

            else:
                category = 'no_reference'
                percentile = np.nan

            categories.append({
                'category': category,
                'ref_percentile': percentile
            })

        df = pd.concat([df, pd.DataFrame(categories)], axis=1)

        return df

    def calculate_metal_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate important metal ratios

        Args:
            df: DataFrame with metal concentrations (wide format)

        Returns:
            DataFrame with calculated ratios
        """
        # Convert to wide format if needed
        if 'metal_name' in df.columns:
            df_wide = df.pivot(index='participant_id', columns='metal_name',
                              values='concentration').reset_index()
        else:
            df_wide = df.copy()

        ratios = []

        for participant in df_wide['participant_id'].unique():
            p_data = df_wide[df_wide['participant_id'] == participant].iloc[0]

            ratio_dict = {'participant_id': participant}

            for metal1, metal2 in self.metal_ratios:
                if metal1 in df_wide.columns and metal2 in df_wide.columns:
                    val1 = p_data[metal1]
                    val2 = p_data[metal2]

                    if pd.notna(val1) and pd.notna(val2) and val2 > 0:
                        ratio_dict[f'{metal1}_{metal2}_ratio'] = val1 / val2
                    else:
                        ratio_dict[f'{metal1}_{metal2}_ratio'] = np.nan

            ratios.append(ratio_dict)

        return pd.DataFrame(ratios)

    def calculate_body_burden_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative toxic metal burden index

        Combines multiple toxic metals into single score

        Args:
            df: DataFrame with metal concentrations

        Returns:
            DataFrame with burden scores
        """
        # Convert to wide format
        if 'metal_name' in df.columns:
            df_wide = df.pivot(index='participant_id', columns='metal_name',
                              values='concentration').reset_index()
        else:
            df_wide = df.copy()

        burden_scores = []

        for participant in df_wide['participant_id'].unique():
            p_data = df_wide[df_wide['participant_id'] == participant].iloc[0]

            # Weight toxic metals by relative toxicity
            toxicity_weights = {
                'Pb': 3.0,  # Lead is highly neurotoxic
                'Hg': 2.5,  # Mercury also highly toxic
                'Cd': 2.0,  # Cadmium moderately toxic
                'As': 2.0,  # Arsenic moderately toxic
                'Al': 1.0,  # Aluminum lower priority
                'Tl': 3.0,  # Thallium highly toxic (rare)
            }

            burden = 0
            n_metals = 0

            for metal, weight in toxicity_weights.items():
                if metal in df_wide.columns:
                    conc = p_data[metal]

                    if pd.notna(conc) and metal in self.reference_ranges:
                        # Normalize to reference range high
                        ref_high = self.reference_ranges[metal]['high']
                        normalized = (conc / ref_high) if ref_high > 0 else 0

                        # Weight by toxicity
                        burden += weight * normalized
                        n_metals += 1

            # Average weighted burden
            if n_metals > 0:
                burden_index = burden / n_metals
            else:
                burden_index = np.nan

            # Categorize
            if pd.isna(burden_index):
                category = 'insufficient_data'
            elif burden_index < 0.5:
                category = 'low'
            elif burden_index < 1.0:
                category = 'moderate'
            elif burden_index < 2.0:
                category = 'high'
            else:
                category = 'very_high'

            burden_scores.append({
                'participant_id': participant,
                'toxic_metal_burden_index': burden_index,
                'burden_category': category
            })

        return pd.DataFrame(burden_scores)

    def process_heavy_metals(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Complete heavy metal processing pipeline

        Args:
            file_path: Path to metal biomarker data

        Returns:
            Dict with processed data, ratios, and burden scores
        """
        logger.info(f"Processing heavy metal data from {file_path}")

        # Load data
        df = self.load_metal_data(file_path)

        # Handle LOD
        df = self.handle_below_lod(df)

        # Creatinine adjustment for urine
        if self.matrix == 'urine':
            df = self.creatinine_adjust(df)

        # Apply reference ranges
        df = self.apply_reference_ranges(df)

        # Calculate ratios
        ratios = self.calculate_metal_ratios(df)

        # Calculate burden index
        burden = self.calculate_body_burden_index(df)

        logger.info(f"✓ Processed {len(df['participant_id'].unique())} participants")

        return {
            'concentrations': df,
            'ratios': ratios,
            'burden': burden
        }


class OrganicPollutantAnalyzer:
    """
    Analyze organic pollutant biomarkers

    Handles phthalates, bisphenols, pesticides, PFAS, flame retardants
    """

    def __init__(self):
        """Initialize organic pollutant analyzer"""

        # Pollutant groups
        self.phthalates = [
            'MEP', 'MBP', 'MiBP', 'MBzP', 'MCPP',
            'MEHP', 'MEOHP', 'MEHHP', 'MECPP',  # DEHP metabolites
            'MiNP', 'MOP'
        ]

        self.bisphenols = ['BPA', 'BPS', 'BPF', 'BPAF']

        self.pesticides = {
            'organophosphates': ['DMP', 'DMTP', 'DMDTP', 'DEP', 'DETP', 'DEDTP'],
            'pyrethroids': ['3PBA', '4FPBA', 'DCCA'],
            'herbicides': ['glyphosate', 'AMPA', '24D', 'atrazine'],
            'organochlorines': ['p_p_DDE', 'p_p_DDT']  # Legacy
        }

        self.flame_retardants = [
            'PBDE_47', 'PBDE_99', 'PBDE_100', 'PBDE_153',  # PBDEs
            'BDCIPP', 'DPHP'  # Organophosphate esters
        ]

        self.pfas = [
            'PFOA', 'PFOS', 'PFHxS', 'PFNA', 'PFDA', 'PFUnDA', 'PFHpA'
        ]

        # Sample matrix (most measured in urine, except PFAS in serum)
        self.matrix_by_analyte = {
            'phthalates': 'urine',
            'bisphenols': 'urine',
            'pesticides': 'urine',
            'flame_retardants': 'serum',
            'pfas': 'serum'
        }

    def load_pollutant_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load organic pollutant data

        Expected columns:
        - participant_id
        - sample_date
        - analyte_name
        - concentration
        - units
        - LOD
        - (creatinine_mg_dL if urine)
        - (specific_gravity if urine)

        Returns:
            DataFrame with pollutant concentrations
        """
        df = pd.read_csv(file_path)
        return df

    def handle_below_lod(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle values below LOD using LOD/sqrt(2)"""
        df = df.copy()

        below_lod = df['concentration'] < df['LOD']
        df.loc[below_lod, 'concentration'] = df.loc[below_lod, 'LOD'] / np.sqrt(2)
        df.loc[below_lod, 'below_lod'] = True
        df['below_lod'] = df['below_lod'].fillna(False)

        return df

    def specific_gravity_adjust(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust urine concentrations for specific gravity (alternative to creatinine)

        SG-adjusted = measured_conc * (1.024 - 1) / (SG - 1)
        Where 1.024 is population mean specific gravity

        Args:
            df: DataFrame with specific_gravity column

        Returns:
            DataFrame with SG-adjusted concentrations
        """
        if 'specific_gravity' not in df.columns:
            logger.warning("Specific gravity not available")
            return df

        df = df.copy()

        sg_ref = 1.024  # Population reference
        df['concentration_adj'] = df['concentration'] * (sg_ref - 1) / (df['specific_gravity'] - 1)

        return df

    def calculate_phthalate_metabolites(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sum DEHP metabolites and calculate molar sums

        DEHP metabolites: MEHP, MEOHP, MEHHP, MECPP
        Report as ΣDEHPm

        Args:
            df: DataFrame with individual phthalate metabolites

        Returns:
            DataFrame with summed metabolites
        """
        # Convert to wide format
        if 'analyte_name' in df.columns:
            df_wide = df.pivot(index='participant_id', columns='analyte_name',
                              values='concentration').reset_index()
        else:
            df_wide = df.copy()

        dehp_metabolites = ['MEHP', 'MEOHP', 'MEHHP', 'MECPP']

        sums = []

        for participant in df_wide['participant_id'].unique():
            p_data = df_wide[df_wide['participant_id'] == participant].iloc[0]

            # Sum DEHP metabolites
            dehp_sum = 0
            n_detected = 0

            for metab in dehp_metabolites:
                if metab in df_wide.columns:
                    val = p_data[metab]
                    if pd.notna(val):
                        dehp_sum += val
                        n_detected += 1

            sums.append({
                'participant_id': participant,
                'DEHP_sum': dehp_sum if n_detected > 0 else np.nan,
                'DEHP_n_detected': n_detected
            })

        return pd.DataFrame(sums)

    def calculate_mixture_index(self, df: pd.DataFrame,
                               analyte_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Calculate mixture exposure index across multiple pollutants

        Uses weighted quantile sum (WQS) approach

        Args:
            df: DataFrame with pollutant concentrations
            analyte_groups: Dict of pollutant groups to include

        Returns:
            DataFrame with mixture indices
        """
        # Convert to wide format
        if 'analyte_name' in df.columns:
            df_wide = df.pivot(index='participant_id', columns='analyte_name',
                              values='concentration').reset_index()
        else:
            df_wide = df.copy()

        mixture_scores = []

        for participant in df_wide['participant_id'].unique():
            p_data = df_wide[df_wide['participant_id'] == participant].iloc[0]

            # Calculate percentiles for each analyte
            total_percentile = 0
            n_analytes = 0

            for group_name, analytes in analyte_groups.items():
                for analyte in analytes:
                    if analyte in df_wide.columns:
                        # Get all values for this analyte
                        all_values = df_wide[analyte].dropna()

                        if len(all_values) > 0:
                            val = p_data[analyte]
                            if pd.notna(val):
                                # Calculate percentile
                                percentile = stats.percentileofscore(all_values, val)
                                total_percentile += percentile
                                n_analytes += 1

            # Average percentile across all pollutants
            mixture_index = total_percentile / n_analytes if n_analytes > 0 else np.nan

            mixture_scores.append({
                'participant_id': participant,
                'pollutant_mixture_index': mixture_index,
                'n_pollutants_detected': n_analytes
            })

        return pd.DataFrame(mixture_scores)

    def process_organic_pollutants(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Complete organic pollutant processing pipeline

        Args:
            file_path: Path to pollutant data

        Returns:
            Dict with processed concentrations and mixture indices
        """
        logger.info(f"Processing organic pollutant data from {file_path}")

        # Load data
        df = self.load_pollutant_data(file_path)

        # Handle LOD
        df = self.handle_below_lod(df)

        # Specific gravity adjustment for urine
        if 'specific_gravity' in df.columns:
            df = self.specific_gravity_adjust(df)

        # Calculate DEHP sum
        dehp_sums = self.calculate_phthalate_metabolites(df)

        # Calculate mixture index
        analyte_groups = {
            'phthalates': self.phthalates,
            'bisphenols': self.bisphenols,
            'pfas': self.pfas
        }
        mixture = self.calculate_mixture_index(df, analyte_groups)

        logger.info(f"✓ Processed {len(df['participant_id'].unique())} participants")

        return {
            'concentrations': df,
            'dehp_sums': dehp_sums,
            'mixture_index': mixture
        }


if __name__ == '__main__':
    # Example usage
    logger.info("Toxicant Biomarker Analyzer initialized")

    # Example: Heavy metals
    logger.info("\n=== Heavy Metal Analysis ===")
    metal_analyzer = HeavyMetalAnalyzer(matrix='hair')

    # Simulate data (use in-memory instead of file)
    metal_data = pd.DataFrame({
        'participant_id': ['P001'] * 4 + ['P002'] * 4,
        'metal_name': ['Pb', 'Hg', 'Zn', 'Cu'] * 2,
        'concentration': [3.5, 0.8, 150, 25, 5.2, 1.5, 120, 30],
        'LOD': [0.1, 0.05, 1.0, 0.5] * 2,
        'units': ['ug/g'] * 8
    })

    print(f"\nExample metal data (hair, 2 participants):")
    print(metal_data.head())

    # Calculate burden scores directly
    burden = metal_analyzer.calculate_body_burden_index(metal_data)
    print("\nToxic Metal Burden Index:")
    print(burden)

    # Example: Organic pollutants
    logger.info("\n=== Organic Pollutant Analysis ===")
    pollutant_analyzer = OrganicPollutantAnalyzer()

    print(f"\nConfigured analyte groups:")
    print(f"  Phthalates: {len(pollutant_analyzer.phthalates)} metabolites")
    print(f"  PFAS: {len(pollutant_analyzer.pfas)} compounds")
    print(f"  Pesticides: {sum(len(v) for v in pollutant_analyzer.pesticides.values())} metabolites")