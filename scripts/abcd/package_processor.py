#!/usr/bin/env python3
"""
ABCD Package Processor

Processes downloaded ABCD data packages to extract ADHD/Autism-relevant variables,
merge across timepoints, handle missing data, and prepare for analysis.

Key Features:
- Extract ADHD/autism phenotypes from CBCL, KSADS
- Process neuroimaging metrics (connectivity, structure, DTI)
- Extract biospecimen/metabolomics data
- Merge longitudinal data across timepoints
- Create analysis-ready datasets

Usage:
    # Process all downloaded packages
    python package_processor.py --input data/abcd/ --output data/abcd/processed/

    # Process specific categories
    python package_processor.py --input data/abcd/ --categories clinical,neuroimaging

    # Extract only ADHD/autism cases
    python package_processor.py --input data/abcd/ --cases-only

    # Generate imaging summary
    python package_processor.py --input data/abcd/ --extract-imaging

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

try:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install pandas numpy tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ABCD Subject ID column (consistent across all data structures)
SUBJECT_ID_COL = 'src_subject_id'
EVENT_NAME_COL = 'eventname'

# Timepoints in ABCD Study
ABCD_TIMEPOINTS = {
    'baseline_year_1_arm_1': 'Baseline (Y0)',
    '1_year_follow_up_y_arm_1': '1-Year Follow-up (Y1)',
    '2_year_follow_up_y_arm_1': '2-Year Follow-up (Y2)',
    '3_year_follow_up_y_arm_1': '3-Year Follow-up (Y3)',
    '4_year_follow_up_y_arm_1': '4-Year Follow-up (Y4)'
}


@dataclass
class ABCDPackageInfo:
    """Metadata for an ABCD data package"""
    package_id: str
    category: str
    timepoints: List[str]
    n_subjects: int
    n_variables: int
    key_variables: List[str]


class ABCDPackageProcessor:
    """Process ABCD data packages"""

    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initialize package processor

        Args:
            input_dir: Directory with downloaded NDA packages
            output_dir: Output directory for processed data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Package registry
        self.packages: Dict[str, ABCDPackageInfo] = {}

        logger.info(f"Initialized ABCD processor: {input_dir} -> {output_dir}")

    def discover_packages(self) -> List[str]:
        """
        Discover downloaded packages in input directory

        Returns:
            List of package IDs
        """
        package_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        package_ids = [d.name for d in package_dirs if d.name.startswith('abcd_')]

        logger.info(f"Discovered {len(package_ids)} packages")
        return package_ids

    def load_package(self, package_id: str) -> Optional[pd.DataFrame]:
        """
        Load a single ABCD package

        Args:
            package_id: Package ID (e.g., 'abcd_cbcls01')

        Returns:
            DataFrame with package data, or None if not found
        """
        package_dir = self.input_dir / package_id

        if not package_dir.exists():
            logger.warning(f"Package directory not found: {package_id}")
            return None

        # Look for data files (txt or csv)
        data_files = list(package_dir.glob('*.txt')) + list(package_dir.glob('*.csv'))

        if not data_files:
            logger.warning(f"No data files found in {package_id}")
            return None

        # Load first data file (usually only one per package)
        data_file = data_files[0]

        try:
            # ABCD data files are tab-delimited with header row and description row
            df = pd.read_csv(data_file, sep='\t', skiprows=[1], low_memory=False)

            # Register package
            timepoints = df[EVENT_NAME_COL].unique().tolist() if EVENT_NAME_COL in df.columns else []
            n_subjects = df[SUBJECT_ID_COL].nunique() if SUBJECT_ID_COL in df.columns else 0

            self.packages[package_id] = ABCDPackageInfo(
                package_id=package_id,
                category='unknown',
                timepoints=timepoints,
                n_subjects=n_subjects,
                n_variables=len(df.columns),
                key_variables=[]
            )

            logger.info(f"Loaded {package_id}: {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error loading {package_id}: {e}")
            return None

    def process_cbcl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process CBCL (Child Behavior Checklist) data

        Extracts:
        - ADHD subscale (attention problems)
        - Autism-related items (social problems)
        - Total problems score
        - Internalizing/externalizing scores

        Args:
            df: Raw CBCL dataframe

        Returns:
            Processed dataframe with key variables
        """
        logger.info("Processing CBCL data...")

        # Key CBCL variables
        key_vars = [SUBJECT_ID_COL, EVENT_NAME_COL]

        # CBCL syndrome scales
        syndrome_scales = [
            'cbcl_scr_syn_attention_r',  # Attention problems (ADHD-related)
            'cbcl_scr_syn_social_r',     # Social problems (autism-related)
            'cbcl_scr_syn_aggressive_r', # Aggressive behavior
            'cbcl_scr_syn_anxdep_r',     # Anxious/depressed
            'cbcl_scr_syn_withdep_r',    # Withdrawn/depressed
            'cbcl_scr_syn_somatic_r',    # Somatic complaints
            'cbcl_scr_syn_thought_r',    # Thought problems
            'cbcl_scr_syn_rulebreak_r'   # Rule-breaking behavior
        ]

        # CBCL summary scores
        summary_scales = [
            'cbcl_scr_syn_internal_r',   # Internalizing
            'cbcl_scr_syn_external_r',   # Externalizing
            'cbcl_scr_syn_totprob_r'     # Total problems
        ]

        # DSM-oriented scales
        dsm_scales = [
            'cbcl_scr_dsm5_adhd_r',      # ADHD problems
            'cbcl_scr_dsm5_depress_r',   # Depressive problems
            'cbcl_scr_dsm5_anxdisord_r', # Anxiety problems
            'cbcl_scr_dsm5_somaticpr_r', # Somatic problems
            'cbcl_scr_dsm5_conduct_r'    # Conduct problems
        ]

        all_scales = syndrome_scales + summary_scales + dsm_scales

        # Select available columns
        available_cols = [col for col in key_vars + all_scales if col in df.columns]

        processed = df[available_cols].copy()

        # Create binary ADHD risk flag (T-score >= 65 on attention or DSM ADHD scale)
        if 'cbcl_scr_syn_attention_r' in processed.columns:
            processed['cbcl_adhd_risk'] = (processed['cbcl_scr_syn_attention_r'] >= 65).astype(int)

        if 'cbcl_scr_dsm5_adhd_r' in processed.columns:
            processed['cbcl_adhd_dsm_risk'] = (processed['cbcl_scr_dsm5_adhd_r'] >= 65).astype(int)

        # Create social problems flag (autism-related)
        if 'cbcl_scr_syn_social_r' in processed.columns:
            processed['cbcl_social_problems'] = (processed['cbcl_scr_syn_social_r'] >= 65).astype(int)

        logger.info(f"Processed CBCL: {len(processed)} observations")
        return processed

    def process_ksads(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process KSADS (Kiddie Schedule for Affective Disorders and Schizophrenia)

        Extracts:
        - ADHD diagnosis (past/present)
        - ASD diagnosis
        - Comorbid diagnoses

        Args:
            df: Raw KSADS dataframe

        Returns:
            Processed dataframe with diagnosis variables
        """
        logger.info("Processing KSADS data...")

        key_vars = [SUBJECT_ID_COL, EVENT_NAME_COL]

        # ADHD diagnosis variables
        adhd_vars = [
            'ksads_adhd_raw_26_p',        # ADHD Inattentive
            'ksads_adhd_raw_27_p',        # ADHD Hyperactive-Impulsive
            'ksads_adhd_raw_28_p',        # ADHD Combined
            'ksads_14_853_p',             # ADHD present
            'ksads_14_854_p'              # ADHD past
        ]

        # Autism/ASD variables
        asd_vars = [
            'ksads_23_946_p',             # ASD present
            'ksads_23_947_p'              # ASD past
        ]

        # Common comorbidities
        comorbid_vars = [
            'ksads_1_840_p',              # Depression present
            'ksads_2_835_p',              # Mania present
            'ksads_4_826_p',              # Panic present
            'ksads_8_863_p',              # OCD present
            'ksads_15_901_p',             # Oppositional defiant present
            'ksads_16_897_p'              # Conduct disorder present
        ]

        all_dx_vars = adhd_vars + asd_vars + comorbid_vars

        # Select available columns
        available_cols = [col for col in key_vars + all_dx_vars if col in df.columns]

        processed = df[available_cols].copy()

        # Create consolidated diagnosis flags
        # ADHD (any type, present or past)
        adhd_cols = [col for col in adhd_vars if col in processed.columns]
        if adhd_cols:
            processed['ksads_adhd_diagnosis'] = processed[adhd_cols].max(axis=1)

        # ASD (present or past)
        asd_cols = [col for col in asd_vars if col in processed.columns]
        if asd_cols:
            processed['ksads_asd_diagnosis'] = processed[asd_cols].max(axis=1)

        # Comorbidity count
        comorbid_cols = [col for col in comorbid_vars if col in processed.columns]
        if comorbid_cols:
            processed['ksads_comorbidity_count'] = processed[comorbid_cols].sum(axis=1)

        logger.info(f"Processed KSADS: {len(processed)} observations")
        return processed

    def process_medications(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process medication history

        Extracts:
        - ADHD medications (stimulants, non-stimulants)
        - Psychotropic medications
        - Medication count

        Args:
            df: Raw medication dataframe

        Returns:
            Processed dataframe with medication variables
        """
        logger.info("Processing medication data...")

        key_vars = [SUBJECT_ID_COL, EVENT_NAME_COL]

        # ADHD medication variables (stimulants)
        stimulant_vars = [
            'medsy_pres_meth_stm_v2',     # Methylphenidate
            'medsy_pres_amph_stm_v2',     # Amphetamine
            'medsy_pres_dext_stm_v2',     # Dextroamphetamine
            'medsy_pres_lisd_stm_v2'      # Lisdexamfetamine
        ]

        # ADHD non-stimulant medications
        nonstim_vars = [
            'medsy_pres_atom_stm_v2',     # Atomoxetine
            'medsy_pres_guan_stm_v2',     # Guanfacine
            'medsy_pres_clon_stm_v2'      # Clonidine
        ]

        # Other psychotropics
        psychotropic_vars = [
            'medsy_pres_ssri_stm_v2',     # SSRIs
            'medsy_pres_atyp_stm_v2',     # Atypical antipsychotics
            'medsy_pres_mood_stm_v2'      # Mood stabilizers
        ]

        all_med_vars = stimulant_vars + nonstim_vars + psychotropic_vars

        # Select available columns
        available_cols = [col for col in key_vars + all_med_vars if col in df.columns]

        if not available_cols:
            # Try alternative medication column naming
            med_cols = [col for col in df.columns if 'med' in col.lower()]
            available_cols = key_vars + med_cols[:20]  # Take first 20

        processed = df[available_cols].copy()

        # Create medication flags
        stim_cols = [col for col in stimulant_vars if col in processed.columns]
        if stim_cols:
            processed['med_adhd_stimulant'] = processed[stim_cols].max(axis=1)

        nonstim_cols = [col for col in nonstim_vars if col in processed.columns]
        if nonstim_cols:
            processed['med_adhd_nonstimulant'] = processed[nonstim_cols].max(axis=1)

        # Any ADHD medication
        adhd_med_cols = stim_cols + nonstim_cols
        if adhd_med_cols:
            processed['med_adhd_any'] = processed[adhd_med_cols].max(axis=1)

        # Psychotropic medication count
        psycho_cols = [col for col in psychotropic_vars if col in processed.columns]
        if psycho_cols:
            processed['med_psychotropic_count'] = processed[psycho_cols].sum(axis=1)

        logger.info(f"Processed medications: {len(processed)} observations")
        return processed

    def process_brain_connectivity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process resting-state fMRI connectivity data

        Extracts:
        - Default Mode Network (DMN) connectivity
        - Salience Network connectivity
        - Executive Control Network connectivity

        Args:
            df: Raw connectivity dataframe

        Returns:
            Processed dataframe with connectivity metrics
        """
        logger.info("Processing brain connectivity data...")

        key_vars = [SUBJECT_ID_COL, EVENT_NAME_COL]

        # Network connectivity columns (depends on ABCD release version)
        # Select all rsfmri columns
        connectivity_cols = [col for col in df.columns if 'rsfmri' in col.lower()]

        if not connectivity_cols:
            logger.warning("No connectivity columns found")
            return df[key_vars].copy()

        available_cols = key_vars + connectivity_cols

        processed = df[available_cols].copy()

        # Calculate connectivity summaries
        # Mean connectivity for each network (if column naming allows)
        dmn_cols = [col for col in connectivity_cols if 'dmn' in col.lower()]
        if dmn_cols:
            processed['connectivity_dmn_mean'] = processed[dmn_cols].mean(axis=1)

        salience_cols = [col for col in connectivity_cols if ('sal' in col.lower() or
                                                              'salience' in col.lower())]
        if salience_cols:
            processed['connectivity_salience_mean'] = processed[salience_cols].mean(axis=1)

        executive_cols = [col for col in connectivity_cols if ('exec' in col.lower() or
                                                               'fpn' in col.lower())]
        if executive_cols:
            processed['connectivity_executive_mean'] = processed[executive_cols].mean(axis=1)

        logger.info(f"Processed connectivity: {len(processed)} observations")
        return processed

    def process_structural_mri(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process structural MRI data (FreeSurfer outputs)

        Extracts:
        - Cortical thickness (global, regional)
        - Surface area
        - Subcortical volumes (striatum, amygdala, hippocampus)

        Args:
            df: Raw structural MRI dataframe

        Returns:
            Processed dataframe with structural metrics
        """
        logger.info("Processing structural MRI data...")

        key_vars = [SUBJECT_ID_COL, EVENT_NAME_COL]

        # Global measures
        global_vars = [
            'smri_thick_cdk_mean',        # Mean cortical thickness
            'smri_area_cdk_total',        # Total surface area
            'smri_vol_cdk_total'          # Total cortical volume
        ]

        # Subcortical volumes (ADHD/autism-relevant)
        subcortical_vars = [
            'smri_vol_scs_caudatelh',     # Caudate (left)
            'smri_vol_scs_caudaterh',     # Caudate (right)
            'smri_vol_scs_putamenlh',     # Putamen (left)
            'smri_vol_scs_putamenrh',     # Putamen (right)
            'smri_vol_scs_amygdalalh',    # Amygdala (left)
            'smri_vol_scs_amygdalarh',    # Amygdala (right)
            'smri_vol_scs_hpuslh',        # Hippocampus (left)
            'smri_vol_scs_hpusrh'         # Hippocampus (right)
        ]

        # Prefrontal cortex (executive function regions)
        pfc_vars = [
            'smri_thick_cdk_cdacatelh',   # Caudal anterior cingulate
            'smri_thick_cdk_cdacaterh',
            'smri_thick_cdk_rracatelh',   # Rostral anterior cingulate
            'smri_thick_cdk_rracaterh'
        ]

        all_struct_vars = global_vars + subcortical_vars + pfc_vars

        # Select available columns
        available_cols = [col for col in key_vars + all_struct_vars if col in df.columns]

        processed = df[available_cols].copy()

        # Calculate bilateral averages
        if 'smri_vol_scs_caudatelh' in processed.columns and 'smri_vol_scs_caudaterh' in processed.columns:
            processed['smri_vol_caudate_bilateral'] = (
                processed['smri_vol_scs_caudatelh'] + processed['smri_vol_scs_caudaterh']
            ) / 2

        if 'smri_vol_scs_amygdalalh' in processed.columns and 'smri_vol_scs_amygdalarh' in processed.columns:
            processed['smri_vol_amygdala_bilateral'] = (
                processed['smri_vol_scs_amygdalalh'] + processed['smri_vol_scs_amygdalarh']
            ) / 2

        logger.info(f"Processed structural MRI: {len(processed)} observations")
        return processed

    def process_dti(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process DTI (Diffusion Tensor Imaging) data

        Extracts:
        - Fractional anisotropy (FA)
        - Mean diffusivity (MD)
        - White matter tract integrity

        Args:
            df: Raw DTI dataframe

        Returns:
            Processed dataframe with DTI metrics
        """
        logger.info("Processing DTI data...")

        key_vars = [SUBJECT_ID_COL, EVENT_NAME_COL]

        # DTI metrics columns
        dti_cols = [col for col in df.columns if any(x in col.lower()
                    for x in ['dmri', 'dti', 'fa', 'md'])]

        if not dti_cols:
            logger.warning("No DTI columns found")
            return df[key_vars].copy()

        available_cols = key_vars + dti_cols

        processed = df[available_cols].copy()

        # Calculate global FA/MD if available
        fa_cols = [col for col in dti_cols if 'fa' in col.lower()]
        if fa_cols:
            processed['dti_fa_mean'] = processed[fa_cols].mean(axis=1)

        md_cols = [col for col in dti_cols if 'md' in col.lower()]
        if md_cols:
            processed['dti_md_mean'] = processed[md_cols].mean(axis=1)

        logger.info(f"Processed DTI: {len(processed)} observations")
        return processed

    def merge_longitudinal_data(self, dataframes: Dict[str, pd.DataFrame],
                                timepoint: str = 'baseline_year_1_arm_1') -> pd.DataFrame:
        """
        Merge multiple processed dataframes for a specific timepoint

        Args:
            dataframes: Dict mapping package_id -> processed dataframe
            timepoint: Timepoint to extract (default: baseline)

        Returns:
            Merged dataframe
        """
        logger.info(f"Merging data for timepoint: {timepoint}")

        # Filter to specified timepoint
        filtered_dfs = {}
        for pkg_id, df in dataframes.items():
            if EVENT_NAME_COL in df.columns:
                df_tp = df[df[EVENT_NAME_COL] == timepoint].copy()
                df_tp = df_tp.drop(columns=[EVENT_NAME_COL])
                filtered_dfs[pkg_id] = df_tp
            else:
                filtered_dfs[pkg_id] = df

        if not filtered_dfs:
            logger.warning(f"No data found for timepoint: {timepoint}")
            return pd.DataFrame()

        # Start with first dataframe
        pkg_ids = list(filtered_dfs.keys())
        merged = filtered_dfs[pkg_ids[0]].copy()

        # Merge remaining dataframes
        for pkg_id in pkg_ids[1:]:
            df = filtered_dfs[pkg_id]

            # Merge on subject ID
            merged = merged.merge(
                df,
                on=SUBJECT_ID_COL,
                how='outer',
                suffixes=('', f'_{pkg_id}')
            )

        logger.info(f"Merged data: {len(merged)} subjects, {len(merged.columns)} variables")
        return merged

    def extract_adhd_autism_cases(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract ADHD, autism, and comorbid cases

        Args:
            df: Merged dataframe with diagnosis variables

        Returns:
            Tuple of (adhd_only, autism_only, comorbid) dataframes
        """
        logger.info("Extracting ADHD/autism cases...")

        # Identify ADHD cases
        adhd_criteria = []
        if 'ksads_adhd_diagnosis' in df.columns:
            adhd_criteria.append(df['ksads_adhd_diagnosis'] == 1)
        if 'cbcl_adhd_risk' in df.columns:
            adhd_criteria.append(df['cbcl_adhd_risk'] == 1)
        if 'med_adhd_any' in df.columns:
            adhd_criteria.append(df['med_adhd_any'] == 1)

        adhd_mask = pd.Series(False, index=df.index)
        if adhd_criteria:
            for criterion in adhd_criteria:
                adhd_mask = adhd_mask | criterion

        # Identify autism cases
        autism_criteria = []
        if 'ksads_asd_diagnosis' in df.columns:
            autism_criteria.append(df['ksads_asd_diagnosis'] == 1)
        if 'cbcl_social_problems' in df.columns:
            autism_criteria.append(df['cbcl_social_problems'] == 1)

        autism_mask = pd.Series(False, index=df.index)
        if autism_criteria:
            for criterion in autism_criteria:
                autism_mask = autism_mask | criterion

        # Extract cohorts
        adhd_only = df[adhd_mask & ~autism_mask].copy()
        autism_only = df[autism_mask & ~adhd_mask].copy()
        comorbid = df[adhd_mask & autism_mask].copy()

        logger.info(f"ADHD only: {len(adhd_only)} subjects")
        logger.info(f"Autism only: {len(autism_only)} subjects")
        logger.info(f"Comorbid: {len(comorbid)} subjects")

        return adhd_only, autism_only, comorbid

    def generate_imaging_summary(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary of imaging metrics for all subjects

        Args:
            merged_df: Merged dataframe with imaging data

        Returns:
            Summary dataframe
        """
        logger.info("Generating imaging summary...")

        # Select imaging columns
        imaging_cols = [col for col in merged_df.columns if any(
            x in col.lower() for x in ['smri', 'rsfmri', 'dmri', 'dti', 'connectivity']
        )]

        if not imaging_cols:
            logger.warning("No imaging columns found")
            return pd.DataFrame()

        summary_cols = [SUBJECT_ID_COL] + imaging_cols
        available_cols = [col for col in summary_cols if col in merged_df.columns]

        imaging_summary = merged_df[available_cols].copy()

        # Add data availability flags
        imaging_summary['has_structural_mri'] = merged_df[[
            col for col in imaging_cols if 'smri' in col
        ]].notna().any(axis=1).astype(int)

        imaging_summary['has_connectivity'] = merged_df[[
            col for col in imaging_cols if 'rsfmri' in col or 'connectivity' in col
        ]].notna().any(axis=1).astype(int)

        imaging_summary['has_dti'] = merged_df[[
            col for col in imaging_cols if 'dmri' in col or 'dti' in col
        ]].notna().any(axis=1).astype(int)

        logger.info(f"Imaging summary: {len(imaging_summary)} subjects")
        return imaging_summary


def main():
    parser = argparse.ArgumentParser(
        description='Process ABCD Study data packages',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with downloaded NDA packages'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for processed data'
    )

    parser.add_argument(
        '--categories',
        type=str,
        help='Comma-separated categories to process (clinical,neuroimaging,etc.)'
    )

    parser.add_argument(
        '--cases-only',
        action='store_true',
        help='Extract only ADHD/autism cases'
    )

    parser.add_argument(
        '--extract-imaging',
        action='store_true',
        help='Generate imaging metrics summary'
    )

    parser.add_argument(
        '--timepoint',
        type=str,
        default='baseline_year_1_arm_1',
        choices=list(ABCD_TIMEPOINTS.keys()),
        help='Timepoint to process (default: baseline)'
    )

    args = parser.parse_args()

    # Initialize processor
    processor = ABCDPackageProcessor(Path(args.input), Path(args.output))

    # Discover packages
    package_ids = processor.discover_packages()

    if not package_ids:
        logger.error("No ABCD packages found")
        sys.exit(1)

    # Process packages
    processed_dfs = {}

    for pkg_id in tqdm(package_ids, desc="Processing packages"):
        df = processor.load_package(pkg_id)

        if df is None:
            continue

        # Process based on package type
        if 'cbcl' in pkg_id:
            df_processed = processor.process_cbcl(df)
        elif 'ksad' in pkg_id:
            df_processed = processor.process_ksads(df)
        elif 'medhy' in pkg_id:
            df_processed = processor.process_medications(df)
        elif 'betnet' in pkg_id:
            df_processed = processor.process_brain_connectivity(df)
        elif 'smrip' in pkg_id:
            df_processed = processor.process_structural_mri(df)
        elif 'dmdtif' in pkg_id:
            df_processed = processor.process_dti(df)
        else:
            df_processed = df  # Keep as-is for other packages

        processed_dfs[pkg_id] = df_processed

        # Save individual processed package
        output_file = processor.output_dir / f"{pkg_id}_processed.csv"
        df_processed.to_csv(output_file, index=False)
        logger.info(f"Saved: {output_file}")

    # Merge data
    merged = processor.merge_longitudinal_data(processed_dfs, timepoint=args.timepoint)

    if not merged.empty:
        merged_file = processor.output_dir / f"abcd_merged_{args.timepoint}.csv"
        merged.to_csv(merged_file, index=False)
        logger.info(f"Saved merged data: {merged_file}")

        # Extract cases
        if args.cases_only:
            adhd_only, autism_only, comorbid = processor.extract_adhd_autism_cases(merged)

            adhd_only.to_csv(processor.output_dir / 'abcd_adhd_only.csv', index=False)
            autism_only.to_csv(processor.output_dir / 'abcd_autism_only.csv', index=False)
            comorbid.to_csv(processor.output_dir / 'abcd_comorbid.csv', index=False)

        # Generate imaging summary
        if args.extract_imaging:
            imaging_summary = processor.generate_imaging_summary(merged)
            imaging_file = processor.output_dir / 'imaging_metrics_extracted.csv'
            imaging_summary.to_csv(imaging_file, index=False)
            logger.info(f"Saved imaging summary: {imaging_file}")

    logger.info("Processing complete!")


if __name__ == '__main__':
    main()