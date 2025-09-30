#!/usr/bin/env python3
"""
Metadata Harmonizer for Cross-Study Microbiome Integration

Harmonizes and standardizes metadata across different microbiome data sources
(SRA, Qiita, MG-RAST, curatedMetagenomicData) for integrated analysis.

Challenges in microbiome metadata:
- Inconsistent variable names (age vs age_years vs participant_age)
- Different units (months vs years for age, g/day vs mg/day)
- Missing data (often >50% missingness)
- Categorical encoding (Male/Female vs M/F vs 1/2)
- Diagnosis criteria variations
- Medication name variations

Features:
- Standardize variable names and units
- Map categorical variables to consistent encoding
- Identify and flag low-quality samples
- Create unified case/control definitions
- Extract dietary and medication information
- Generate analysis-ready metadata tables

Requirements:
    pip install pandas numpy fuzzywuzzy python-Levenshtein

Usage:
    # Harmonize metadata from multiple sources
    python metadata_harmonizer.py --input data/microbiome/raw/ --output data/microbiome/harmonized/

    # Process single study
    python metadata_harmonizer.py --study sra_study_metadata.csv --output data/microbiome/

    # Apply quality filters
    python metadata_harmonizer.py --input data/ --min-reads 5000 --max-contamination 0.05

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

try:
    import pandas as pd
    import numpy as np
    from fuzzywuzzy import fuzz, process
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install pandas numpy fuzzywuzzy python-Levenshtein")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Standard variable mappings
STANDARD_VARIABLES = {
    'sample_id': ['sample_id', 'sample', 'sampleid', 'sample_name', '#SampleID'],
    'subject_id': ['subject_id', 'subject', 'participant_id', 'individual_id', 'host_subject_id'],
    'age': ['age', 'age_years', 'age_at_collection', 'participant_age', 'host_age'],
    'age_months': ['age_months', 'age_in_months'],
    'sex': ['sex', 'gender', 'participant_sex', 'host_sex'],
    'bmi': ['bmi', 'body_mass_index', 'body_mass_index_calculated'],
    'diagnosis': ['diagnosis', 'disease', 'condition', 'phenotype', 'disease_state'],
    'sample_type': ['sample_type', 'body_site', 'body_habitat', 'sample_site', 'feces_sample_type'],
    'collection_date': ['collection_date', 'date_collected', 'collection_timestamp'],
    'sequencing_depth': ['num_reads', 'read_count', 'sequencing_depth', 'library_size'],
    'country': ['country', 'geo_loc_name', 'geographic_location'],
    'medications': ['medications', 'current_medications', 'medication_history'],
    'antibiotics': ['antibiotics', 'antibiotic_usage', 'recent_antibiotics'],
    'diet': ['diet', 'diet_type', 'dietary_pattern'],
    'gi_symptoms': ['gi_symptoms', 'gastrointestinal_symptoms', 'diarrhea', 'constipation']
}

# ADHD/Autism diagnosis patterns
DIAGNOSIS_PATTERNS = {
    'adhd': [
        'adhd', 'add', 'attention deficit', 'hyperactivity',
        'hyperkinetic', 'attention-deficit'
    ],
    'autism': [
        'autism', 'asd', 'autistic', 'asperger', 'pervasive developmental',
        'pdd-nos', 'autism spectrum'
    ],
    'control': [
        'control', 'healthy', 'neurotypical', 'td', 'typically developing',
        'non-affected', 'unaffected'
    ]
}

# Sex encodings
SEX_MAPPINGS = {
    'male': ['male', 'm', '1', 'man', 'boy'],
    'female': ['female', 'f', '2', 'woman', 'girl']
}

# Medication categories
MEDICATION_CATEGORIES = {
    'stimulants': ['methylphenidate', 'ritalin', 'concerta', 'adderall', 'vyvanse',
                   'dexedrine', 'focalin', 'amphetamine', 'dextroamphetamine'],
    'non_stimulants': ['atomoxetine', 'strattera', 'guanfacine', 'intuniv',
                       'clonidine', 'kapvay'],
    'ssri': ['fluoxetine', 'prozac', 'sertraline', 'zoloft', 'escitalopram',
             'lexapro', 'citalopram', 'paroxetine', 'paxil'],
    'antipsychotics': ['risperidone', 'risperdal', 'aripiprazole', 'abilify',
                       'quetiapine', 'seroquel'],
    'antibiotics': ['amoxicillin', 'azithromycin', 'ciprofloxacin', 'cephalexin',
                    'penicillin', 'doxycycline']
}

# Quality control thresholds
QC_THRESHOLDS = {
    'min_reads': 5000,
    'max_contamination': 0.05,
    'min_features': 100,
    'min_diversity': 0.5,
    'max_missing_metadata': 0.5
}


class MetadataHarmonizer:
    """Harmonize metadata across microbiome studies"""

    def __init__(self, output_dir: Path):
        """
        Initialize harmonizer

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.harmonization_log = []

        logger.info(f"Initialized metadata harmonizer: {output_dir}")

    def harmonize_study(self, metadata_file: Path,
                       study_id: str = None) -> pd.DataFrame:
        """
        Harmonize metadata for a single study

        Args:
            metadata_file: Path to metadata CSV
            study_id: Study identifier

        Returns:
            Harmonized metadata DataFrame
        """
        logger.info(f"Harmonizing metadata: {metadata_file}")

        # Read metadata
        df = pd.read_csv(metadata_file, sep=None, engine='python')

        if study_id is None:
            study_id = metadata_file.stem

        # Add study identifier
        df['study_id'] = study_id

        # Harmonize column names
        df = self._harmonize_columns(df)

        # Standardize categorical variables
        df = self._standardize_categories(df)

        # Extract diagnosis
        df = self._extract_diagnosis(df)

        # Process medications
        df = self._process_medications(df)

        # Standardize units
        df = self._standardize_units(df)

        # Add quality flags
        df = self._add_quality_flags(df)

        logger.info(f"Harmonized {len(df)} samples from {study_id}")

        return df

    def _harmonize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map column names to standard variables"""
        logger.info("Harmonizing column names...")

        renamed = {}
        original_cols = df.columns.tolist()

        for standard_var, variants in STANDARD_VARIABLES.items():
            # Find matching column
            for col in original_cols:
                if col.lower() in [v.lower() for v in variants]:
                    renamed[col] = standard_var
                    self.harmonization_log.append({
                        'action': 'rename',
                        'original': col,
                        'standardized': standard_var
                    })
                    break
                else:
                    # Try fuzzy matching
                    for variant in variants:
                        similarity = fuzz.ratio(col.lower(), variant.lower())
                        if similarity > 85:  # High similarity threshold
                            renamed[col] = standard_var
                            self.harmonization_log.append({
                                'action': 'fuzzy_rename',
                                'original': col,
                                'standardized': standard_var,
                                'similarity': similarity
                            })
                            break

        # Rename columns
        df = df.rename(columns=renamed)

        logger.info(f"Renamed {len(renamed)} columns")
        return df

    def _standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical variables"""
        logger.info("Standardizing categorical variables...")

        # Standardize sex
        if 'sex' in df.columns:
            df['sex_standardized'] = df['sex'].apply(self._standardize_sex)

        # Standardize sample type
        if 'sample_type' in df.columns:
            df['sample_type_standardized'] = df['sample_type'].apply(
                self._standardize_sample_type
            )

        return df

    def _standardize_sex(self, value: Any) -> Optional[str]:
        """Standardize sex encoding"""
        if pd.isna(value):
            return None

        value_str = str(value).lower().strip()

        for standard_sex, variants in SEX_MAPPINGS.items():
            if value_str in variants:
                return standard_sex

        return None

    def _standardize_sample_type(self, value: Any) -> Optional[str]:
        """Standardize sample type"""
        if pd.isna(value):
            return None

        value_str = str(value).lower().strip()

        # Map to standard types
        if any(term in value_str for term in ['stool', 'fecal', 'feces']):
            return 'stool'
        elif any(term in value_str for term in ['saliva', 'oral', 'buccal']):
            return 'saliva'
        elif any(term in value_str for term in ['gut', 'intestinal', 'colonic']):
            return 'gut'
        else:
            return value_str

    def _extract_diagnosis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract standardized diagnosis"""
        logger.info("Extracting diagnosis information...")

        df['diagnosis_adhd'] = 0
        df['diagnosis_autism'] = 0
        df['diagnosis_control'] = 0
        df['diagnosis_standardized'] = 'unknown'

        if 'diagnosis' not in df.columns:
            logger.warning("No diagnosis column found")
            return df

        for idx, row in df.iterrows():
            diagnosis_str = str(row['diagnosis']).lower()

            # Check for ADHD
            if any(pattern in diagnosis_str for pattern in DIAGNOSIS_PATTERNS['adhd']):
                df.at[idx, 'diagnosis_adhd'] = 1
                df.at[idx, 'diagnosis_standardized'] = 'adhd'

            # Check for autism
            if any(pattern in diagnosis_str for pattern in DIAGNOSIS_PATTERNS['autism']):
                df.at[idx, 'diagnosis_autism'] = 1
                if df.at[idx, 'diagnosis_adhd'] == 1:
                    df.at[idx, 'diagnosis_standardized'] = 'comorbid_adhd_autism'
                else:
                    df.at[idx, 'diagnosis_standardized'] = 'autism'

            # Check for control
            if any(pattern in diagnosis_str for pattern in DIAGNOSIS_PATTERNS['control']):
                df.at[idx, 'diagnosis_control'] = 1
                if df.at[idx, 'diagnosis_adhd'] == 0 and df.at[idx, 'diagnosis_autism'] == 0:
                    df.at[idx, 'diagnosis_standardized'] = 'control'

        logger.info(f"ADHD cases: {df['diagnosis_adhd'].sum()}")
        logger.info(f"Autism cases: {df['diagnosis_autism'].sum()}")
        logger.info(f"Controls: {df['diagnosis_control'].sum()}")

        return df

    def _process_medications(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process medication information"""
        logger.info("Processing medications...")

        # Initialize medication flags
        for category in MEDICATION_CATEGORIES.keys():
            df[f'medication_{category}'] = 0

        if 'medications' not in df.columns:
            logger.warning("No medications column found")
            return df

        for idx, row in df.iterrows():
            meds_str = str(row['medications']).lower()

            if pd.isna(row['medications']) or meds_str == 'nan':
                continue

            # Check each medication category
            for category, med_list in MEDICATION_CATEGORIES.items():
                for med in med_list:
                    if med in meds_str:
                        df.at[idx, f'medication_{category}'] = 1

        return df

    def _standardize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize units for numeric variables"""
        logger.info("Standardizing units...")

        # Convert age to years if in months
        if 'age_months' in df.columns and 'age' not in df.columns:
            df['age'] = df['age_months'] / 12.0
            self.harmonization_log.append({
                'action': 'unit_conversion',
                'variable': 'age',
                'conversion': 'months_to_years'
            })

        # Ensure age is numeric
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')

        # Ensure BMI is numeric
        if 'bmi' in df.columns:
            df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

        return df

    def _add_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality control flags"""
        logger.info("Adding quality flags...")

        df['qc_pass'] = 1

        # Check sequencing depth
        if 'sequencing_depth' in df.columns:
            df['qc_low_reads'] = (
                df['sequencing_depth'] < QC_THRESHOLDS['min_reads']
            ).astype(int)
            df.loc[df['qc_low_reads'] == 1, 'qc_pass'] = 0

        # Check metadata completeness
        essential_cols = ['sample_id', 'age', 'sex_standardized', 'diagnosis_standardized']
        available_cols = [c for c in essential_cols if c in df.columns]

        if available_cols:
            df['qc_missing_metadata'] = df[available_cols].isna().sum(axis=1) / len(available_cols)
            df.loc[
                df['qc_missing_metadata'] > QC_THRESHOLDS['max_missing_metadata'],
                'qc_pass'
            ] = 0

        logger.info(f"Samples passing QC: {df['qc_pass'].sum()} / {len(df)}")

        return df

    def merge_studies(self, harmonized_files: List[Path]) -> pd.DataFrame:
        """
        Merge multiple harmonized studies

        Args:
            harmonized_files: List of harmonized metadata files

        Returns:
            Merged DataFrame
        """
        logger.info(f"Merging {len(harmonized_files)} studies...")

        dfs = []

        for file in harmonized_files:
            df = pd.read_csv(file)
            dfs.append(df)

        # Concatenate
        merged = pd.concat(dfs, ignore_index=True, sort=False)

        # Add unified sample ID
        merged['unified_sample_id'] = merged.apply(
            lambda row: f"{row.get('study_id', 'unknown')}_{row.get('sample_id', 'unknown')}",
            axis=1
        )

        logger.info(f"Merged dataset: {len(merged)} samples from {merged['study_id'].nunique()} studies")

        return merged

    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for harmonized data"""
        logger.info("Generating summary statistics...")

        stats = {
            'total_samples': len(df),
            'total_studies': df['study_id'].nunique() if 'study_id' in df.columns else 1,
            'samples_passing_qc': df['qc_pass'].sum() if 'qc_pass' in df.columns else len(df),
            'case_control_breakdown': {},
            'age_statistics': {},
            'sex_distribution': {},
            'sample_type_distribution': {},
            'medication_summary': {}
        }

        # Diagnosis breakdown
        if 'diagnosis_standardized' in df.columns:
            stats['case_control_breakdown'] = df['diagnosis_standardized'].value_counts().to_dict()

        # Age statistics
        if 'age' in df.columns:
            age_data = df['age'].dropna()
            if len(age_data) > 0:
                stats['age_statistics'] = {
                    'mean': float(age_data.mean()),
                    'std': float(age_data.std()),
                    'min': float(age_data.min()),
                    'max': float(age_data.max()),
                    'missing': int(df['age'].isna().sum())
                }

        # Sex distribution
        if 'sex_standardized' in df.columns:
            stats['sex_distribution'] = df['sex_standardized'].value_counts().to_dict()

        # Sample type distribution
        if 'sample_type_standardized' in df.columns:
            stats['sample_type_distribution'] = df['sample_type_standardized'].value_counts().to_dict()

        # Medication usage
        med_cols = [c for c in df.columns if c.startswith('medication_')]
        for col in med_cols:
            category = col.replace('medication_', '')
            stats['medication_summary'][category] = int(df[col].sum())

        return stats

    def save_harmonization_log(self) -> Path:
        """Save harmonization log"""
        log_file = self.output_dir / 'harmonization_log.json'

        with open(log_file, 'w') as f:
            json.dump(self.harmonization_log, f, indent=2)

        logger.info(f"Harmonization log saved: {log_file}")
        return log_file


def main():
    parser = argparse.ArgumentParser(
        description='Harmonize microbiome metadata across studies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Harmonize single study
  python metadata_harmonizer.py --study study1_metadata.csv --output data/microbiome/

  # Harmonize multiple studies
  python metadata_harmonizer.py --input data/microbiome/raw/ --output data/microbiome/harmonized/

  # Apply quality filters
  python metadata_harmonizer.py --input data/ --min-reads 10000 --output data/

  # Generate merged dataset
  python metadata_harmonizer.py --merge harmonized/*.csv --output data/
        """
    )

    parser.add_argument(
        '--study',
        type=str,
        help='Single metadata file to harmonize'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Directory containing multiple metadata files'
    )

    parser.add_argument(
        '--merge',
        nargs='+',
        help='Merge harmonized files'
    )

    parser.add_argument(
        '--min-reads',
        type=int,
        default=QC_THRESHOLDS['min_reads'],
        help=f'Minimum sequencing depth (default: {QC_THRESHOLDS["min_reads"]})'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/microbiome/harmonized',
        help='Output directory'
    )

    args = parser.parse_args()

    # Update QC threshold
    QC_THRESHOLDS['min_reads'] = args.min_reads

    # Initialize harmonizer
    harmonizer = MetadataHarmonizer(Path(args.output))

    # Handle single study
    if args.study:
        study_file = Path(args.study)
        harmonized_df = harmonizer.harmonize_study(study_file)

        # Save harmonized metadata
        output_file = harmonizer.output_dir / f'{study_file.stem}_harmonized.csv'
        harmonized_df.to_csv(output_file, index=False)
        print(f"\nHarmonized metadata saved: {output_file}")

        # Generate stats
        stats = harmonizer.generate_summary_stats(harmonized_df)
        print(f"\n=== Summary Statistics ===\n")
        print(json.dumps(stats, indent=2))

        # Save stats
        stats_file = harmonizer.output_dir / f'{study_file.stem}_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

    # Handle multiple studies
    elif args.input:
        input_dir = Path(args.input)
        metadata_files = list(input_dir.glob('*.csv'))

        print(f"\nFound {len(metadata_files)} metadata files")

        harmonized_files = []

        for metadata_file in metadata_files:
            print(f"\nProcessing {metadata_file.name}...")

            harmonized_df = harmonizer.harmonize_study(metadata_file)

            # Save
            output_file = harmonizer.output_dir / f'{metadata_file.stem}_harmonized.csv'
            harmonized_df.to_csv(output_file, index=False)
            harmonized_files.append(output_file)

        print(f"\n{len(harmonized_files)} studies harmonized")

        # Merge all studies
        if len(harmonized_files) > 1:
            merged_df = harmonizer.merge_studies(harmonized_files)

            merged_file = harmonizer.output_dir / 'all_studies_merged.csv'
            merged_df.to_csv(merged_file, index=False)
            print(f"\nMerged dataset saved: {merged_file}")

            # Generate merged stats
            stats = harmonizer.generate_summary_stats(merged_df)
            print(f"\n=== Merged Dataset Statistics ===\n")
            print(json.dumps(stats, indent=2))

            stats_file = harmonizer.output_dir / 'merged_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

    # Handle merge
    elif args.merge:
        merge_files = [Path(f) for f in args.merge]
        merged_df = harmonizer.merge_studies(merge_files)

        merged_file = harmonizer.output_dir / 'merged_metadata.csv'
        merged_df.to_csv(merged_file, index=False)
        print(f"\nMerged metadata saved: {merged_file}")

        stats = harmonizer.generate_summary_stats(merged_df)
        print(f"\n=== Merged Statistics ===\n")
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()
        return

    # Save harmonization log
    harmonizer.save_harmonization_log()


if __name__ == '__main__':
    main()