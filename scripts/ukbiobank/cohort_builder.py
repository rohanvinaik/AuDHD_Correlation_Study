#!/usr/bin/env python3
"""
UK Biobank Cohort Builder

Build ADHD and Autism cohorts from UK Biobank data using:
- ICD-10 diagnoses (F90.x for ADHD, F84.x for Autism)
- Self-reported conditions
- Medication history (ADHD medications)
- Mental health questionnaire

Identifies:
- ADHD cases (childhood-onset, adult-diagnosed)
- Autism cases
- Comorbid ADHD+Autism
- Controls (no psychiatric diagnosis)
- Metabolomics subset (participants with NMR data)

Usage:
    # Build cohorts from extracted data
    python cohort_builder.py --input data/ukb/ukb_diagnosis.csv --output data/ukb/

    # Build with metabolomics filtering
    python cohort_builder.py --input data/ukb/ukb_combined.csv --metabolomics-only

    # Export cohort IDs
    python cohort_builder.py --input data/ukb/ukb_combined.csv --export-ids
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy required")
    print("Install with: pip install pandas numpy")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ICD-10 codes
ICD10_CODES = {
    'ADHD': {
        'F90.0': 'Disturbance of activity and attention',
        'F90.1': 'Hyperkinetic conduct disorder',
        'F90.8': 'Other hyperkinetic disorders',
        'F90.9': 'Hyperkinetic disorder, unspecified',
    },
    'Autism': {
        'F84.0': 'Childhood autism',
        'F84.1': 'Atypical autism',
        'F84.5': 'Asperger syndrome',
        'F84.8': 'Other pervasive developmental disorders',
        'F84.9': 'Pervasive developmental disorder, unspecified',
    }
}

# Self-reported codes (field 20002)
SELF_REPORT_CODES = {
    'ADHD': [1117],  # Attention deficit hyperactivity disorder
    'Autism': [1111],  # Autism
}

# ADHD medication codes (field 20003)
ADHD_MEDICATION_CODES = {
    '1140926662': 'Methylphenidate',
    '1140888594': 'Atomoxetine',
    '1141173882': 'Dexamfetamine',
    '1140888648': 'Lisdexamfetamine',
    '1140860806': 'Guanfacine',
}


@dataclass
class CohortDefinition:
    """Definition of cohort inclusion/exclusion criteria"""
    name: str
    icd10_codes: List[str] = field(default_factory=list)
    self_report_codes: List[int] = field(default_factory=list)
    medication_codes: List[str] = field(default_factory=list)
    age_range: Tuple[int, int] = (40, 70)
    exclude_psychiatric: bool = False
    exclude_neurological: bool = False
    require_metabolomics: bool = False


@dataclass
class CohortMember:
    """Individual cohort member"""
    eid: int
    diagnosis: str
    source: List[str]  # icd10, self_report, medication
    icd10_codes: List[str] = field(default_factory=list)
    age: Optional[float] = None
    sex: Optional[int] = None
    has_metabolomics: bool = False
    comorbidities: List[str] = field(default_factory=list)


class UKBCohortBuilder:
    """Build ADHD/Autism cohorts from UK Biobank data"""

    def __init__(self, data_path: Path):
        """
        Initialize cohort builder

        Args:
            data_path: Path to extracted UK Biobank CSV
        """
        self.data_path = Path(data_path)
        self.data: Optional[pd.DataFrame] = None
        self.cohorts: Dict[str, List[CohortMember]] = {}

        # Load data
        self._load_data()

    def _load_data(self):
        """Load UK Biobank data"""
        logger.info(f"Loading data from: {self.data_path}")

        try:
            self.data = pd.read_csv(self.data_path, low_memory=False)
            logger.info(f"Loaded {len(self.data)} participants")

            # Check required columns
            required_cols = ['eid']
            missing_cols = [col for col in required_cols if col not in self.data.columns]

            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            sys.exit(1)

    def identify_adhd_cases(self) -> List[CohortMember]:
        """
        Identify ADHD cases using multiple sources

        Returns:
            List of ADHD cohort members
        """
        logger.info("Identifying ADHD cases...")

        adhd_cases = {}

        # Method 1: ICD-10 diagnoses (fields 41270, 41271)
        adhd_icd10 = self._find_icd10_cases('ADHD')
        logger.info(f"  Found {len(adhd_icd10)} ADHD cases from ICD-10")

        for eid in adhd_icd10:
            if eid not in adhd_cases:
                adhd_cases[eid] = CohortMember(
                    eid=eid,
                    diagnosis='ADHD',
                    source=['icd10'],
                    icd10_codes=adhd_icd10[eid]
                )
            else:
                adhd_cases[eid].source.append('icd10')
                adhd_cases[eid].icd10_codes.extend(adhd_icd10[eid])

        # Method 2: Self-reported (field 20002)
        adhd_self_report = self._find_self_reported_cases('ADHD')
        logger.info(f"  Found {len(adhd_self_report)} ADHD cases from self-report")

        for eid in adhd_self_report:
            if eid not in adhd_cases:
                adhd_cases[eid] = CohortMember(
                    eid=eid,
                    diagnosis='ADHD',
                    source=['self_report']
                )
            else:
                if 'self_report' not in adhd_cases[eid].source:
                    adhd_cases[eid].source.append('self_report')

        # Method 3: ADHD medications (field 20003)
        adhd_medications = self._find_medication_cases('ADHD')
        logger.info(f"  Found {len(adhd_medications)} ADHD cases from medications")

        for eid in adhd_medications:
            if eid not in adhd_cases:
                adhd_cases[eid] = CohortMember(
                    eid=eid,
                    diagnosis='ADHD',
                    source=['medication']
                )
            else:
                if 'medication' not in adhd_cases[eid].source:
                    adhd_cases[eid].source.append('medication')

        # Add demographics and metabolomics status
        for eid, member in adhd_cases.items():
            self._add_demographics(member)
            self._check_metabolomics(member)

        logger.info(f"✓ Total ADHD cases: {len(adhd_cases)}")

        return list(adhd_cases.values())

    def identify_autism_cases(self) -> List[CohortMember]:
        """
        Identify Autism cases

        Returns:
            List of Autism cohort members
        """
        logger.info("Identifying Autism cases...")

        autism_cases = {}

        # Method 1: ICD-10 diagnoses
        autism_icd10 = self._find_icd10_cases('Autism')
        logger.info(f"  Found {len(autism_icd10)} Autism cases from ICD-10")

        for eid in autism_icd10:
            if eid not in autism_cases:
                autism_cases[eid] = CohortMember(
                    eid=eid,
                    diagnosis='Autism',
                    source=['icd10'],
                    icd10_codes=autism_icd10[eid]
                )

        # Method 2: Self-reported
        autism_self_report = self._find_self_reported_cases('Autism')
        logger.info(f"  Found {len(autism_self_report)} Autism cases from self-report")

        for eid in autism_self_report:
            if eid not in autism_cases:
                autism_cases[eid] = CohortMember(
                    eid=eid,
                    diagnosis='Autism',
                    source=['self_report']
                )
            else:
                if 'self_report' not in autism_cases[eid].source:
                    autism_cases[eid].source.append('self_report')

        # Add demographics and metabolomics
        for eid, member in autism_cases.items():
            self._add_demographics(member)
            self._check_metabolomics(member)

        logger.info(f"✓ Total Autism cases: {len(autism_cases)}")

        return list(autism_cases.values())

    def identify_comorbid_cases(
        self,
        adhd_cases: List[CohortMember],
        autism_cases: List[CohortMember]
    ) -> List[CohortMember]:
        """
        Identify comorbid ADHD+Autism cases

        Args:
            adhd_cases: ADHD cohort
            autism_cases: Autism cohort

        Returns:
            List of comorbid cases
        """
        logger.info("Identifying comorbid ADHD+Autism cases...")

        adhd_eids = {m.eid for m in adhd_cases}
        autism_eids = {m.eid for m in autism_cases}

        comorbid_eids = adhd_eids & autism_eids

        comorbid_cases = []
        for eid in comorbid_eids:
            # Get ADHD member
            adhd_member = next(m for m in adhd_cases if m.eid == eid)

            # Combine sources
            member = CohortMember(
                eid=eid,
                diagnosis='ADHD+Autism',
                source=list(set(adhd_member.source)),
                icd10_codes=adhd_member.icd10_codes,
                age=adhd_member.age,
                sex=adhd_member.sex,
                has_metabolomics=adhd_member.has_metabolomics
            )
            comorbid_cases.append(member)

        logger.info(f"✓ Comorbid cases: {len(comorbid_cases)}")

        return comorbid_cases

    def identify_controls(
        self,
        n_controls: int = 10000,
        match_age_sex: bool = True,
        require_metabolomics: bool = False
    ) -> List[CohortMember]:
        """
        Identify control participants

        Args:
            n_controls: Number of controls to select
            match_age_sex: Match age/sex distribution of cases
            require_metabolomics: Require metabolomics data

        Returns:
            List of control cohort members
        """
        logger.info(f"Identifying {n_controls} controls...")

        # Exclude anyone with psychiatric diagnosis
        excluded_eids = self._get_psychiatric_eids()

        # Get potential controls
        potential_controls = self.data[~self.data['eid'].isin(excluded_eids)]

        logger.info(f"  Potential controls: {len(potential_controls)}")

        # Filter by metabolomics if required
        if require_metabolomics:
            metabolomics_eids = self._get_metabolomics_eids()
            potential_controls = potential_controls[
                potential_controls['eid'].isin(metabolomics_eids)
            ]
            logger.info(f"  With metabolomics: {len(potential_controls)}")

        # Sample controls
        if len(potential_controls) > n_controls:
            sampled = potential_controls.sample(n=n_controls, random_state=42)
        else:
            sampled = potential_controls

        # Create control members
        controls = []
        for _, row in sampled.iterrows():
            member = CohortMember(
                eid=int(row['eid']),
                diagnosis='Control',
                source=['control']
            )
            self._add_demographics(member)
            self._check_metabolomics(member)
            controls.append(member)

        logger.info(f"✓ Selected {len(controls)} controls")

        return controls

    def _find_icd10_cases(self, condition: str) -> Dict[int, List[str]]:
        """Find cases by ICD-10 code"""
        icd10_codes = ICD10_CODES[condition].keys()
        cases = {}

        # Check fields 41270 and 41271
        icd10_fields = [col for col in self.data.columns if col.startswith('41270') or col.startswith('41271')]

        for field in icd10_fields:
            for idx, value in self.data[field].items():
                if pd.isna(value):
                    continue

                # Check if any ICD-10 code matches
                for code in icd10_codes:
                    if str(value).startswith(code):
                        eid = int(self.data.loc[idx, 'eid'])
                        if eid not in cases:
                            cases[eid] = []
                        cases[eid].append(str(value))
                        break

        return cases

    def _find_self_reported_cases(self, condition: str) -> Set[int]:
        """Find cases by self-report (field 20002)"""
        codes = SELF_REPORT_CODES[condition]
        cases = set()

        # Check field 20002 (all instances)
        self_report_fields = [col for col in self.data.columns if col.startswith('20002')]

        for field in self_report_fields:
            for idx, value in self.data[field].items():
                if pd.isna(value):
                    continue

                if int(value) in codes:
                    eid = int(self.data.loc[idx, 'eid'])
                    cases.add(eid)

        return cases

    def _find_medication_cases(self, condition: str) -> Set[int]:
        """Find cases by medication (field 20003)"""
        if condition != 'ADHD':
            return set()

        med_codes = ADHD_MEDICATION_CODES.keys()
        cases = set()

        # Check field 20003 (all instances)
        med_fields = [col for col in self.data.columns if col.startswith('20003')]

        for field in med_fields:
            for idx, value in self.data[field].items():
                if pd.isna(value):
                    continue

                if str(int(value)) in med_codes:
                    eid = int(self.data.loc[idx, 'eid'])
                    cases.add(eid)

        return cases

    def _get_psychiatric_eids(self) -> Set[int]:
        """Get all participants with any psychiatric diagnosis"""
        psychiatric_codes = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']
        excluded = set()

        icd10_fields = [col for col in self.data.columns if col.startswith('41270') or col.startswith('41271')]

        for field in icd10_fields:
            for idx, value in self.data[field].items():
                if pd.isna(value):
                    continue

                value_str = str(value)
                if any(value_str.startswith(code) for code in psychiatric_codes):
                    eid = int(self.data.loc[idx, 'eid'])
                    excluded.add(eid)

        return excluded

    def _get_metabolomics_eids(self) -> Set[int]:
        """Get participants with metabolomics data"""
        # Check if any metabolomics field (23400-23649) has data
        metabolomics_fields = [col for col in self.data.columns if col.startswith('234')]

        if not metabolomics_fields:
            logger.warning("No metabolomics fields found")
            return set()

        # Participant has metabolomics if ANY field is non-null
        has_data = self.data[metabolomics_fields].notna().any(axis=1)
        metabolomics_eids = set(self.data.loc[has_data, 'eid'].astype(int))

        logger.info(f"  Participants with metabolomics: {len(metabolomics_eids)}")

        return metabolomics_eids

    def _add_demographics(self, member: CohortMember):
        """Add demographic information to member"""
        row = self.data[self.data['eid'] == member.eid].iloc[0]

        # Age (field 21003-0.0)
        age_field = '21003-0.0'
        if age_field in row and not pd.isna(row[age_field]):
            member.age = float(row[age_field])

        # Sex (field 31-0.0)
        sex_field = '31-0.0'
        if sex_field in row and not pd.isna(row[sex_field]):
            member.sex = int(row[sex_field])

    def _check_metabolomics(self, member: CohortMember):
        """Check if member has metabolomics data"""
        metabolomics_eids = self._get_metabolomics_eids()
        member.has_metabolomics = member.eid in metabolomics_eids

    def build_cohorts(
        self,
        n_controls: int = 10000,
        metabolomics_only: bool = False
    ) -> Dict[str, List[CohortMember]]:
        """
        Build all cohorts

        Args:
            n_controls: Number of controls
            metabolomics_only: Only include participants with metabolomics

        Returns:
            Dictionary of cohorts
        """
        logger.info("Building cohorts...")

        # Identify cases
        adhd_cases = self.identify_adhd_cases()
        autism_cases = self.identify_autism_cases()
        comorbid_cases = self.identify_comorbid_cases(adhd_cases, autism_cases)

        # Remove comorbid from individual groups
        comorbid_eids = {m.eid for m in comorbid_cases}
        adhd_only = [m for m in adhd_cases if m.eid not in comorbid_eids]
        autism_only = [m for m in autism_cases if m.eid not in comorbid_eids]

        # Filter by metabolomics if requested
        if metabolomics_only:
            logger.info("Filtering for metabolomics subset...")
            adhd_only = [m for m in adhd_only if m.has_metabolomics]
            autism_only = [m for m in autism_only if m.has_metabolomics]
            comorbid_cases = [m for m in comorbid_cases if m.has_metabolomics]

        # Identify controls
        controls = self.identify_controls(
            n_controls=n_controls,
            require_metabolomics=metabolomics_only
        )

        cohorts = {
            'ADHD_only': adhd_only,
            'Autism_only': autism_only,
            'ADHD+Autism': comorbid_cases,
            'Controls': controls
        }

        # Log summary
        logger.info("\nCohort Summary:")
        for name, members in cohorts.items():
            n_with_metab = sum(1 for m in members if m.has_metabolomics)
            logger.info(f"  {name}: {len(members)} ({n_with_metab} with metabolomics)")

        self.cohorts = cohorts
        return cohorts

    def export_cohort_ids(self, output_dir: Path):
        """Export cohort IDs to text files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for cohort_name, members in self.cohorts.items():
            output_file = output_dir / f"{cohort_name.lower().replace('+', '_')}_ids.txt"

            with open(output_file, 'w') as f:
                for member in members:
                    f.write(f"{member.eid}\n")

            logger.info(f"Exported {cohort_name}: {output_file}")

    def export_cohort_metadata(self, output_dir: Path):
        """Export detailed cohort metadata"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for cohort_name, members in self.cohorts.items():
            output_file = output_dir / f"{cohort_name.lower().replace('+', '_')}_metadata.csv"

            data = []
            for member in members:
                data.append({
                    'eid': member.eid,
                    'diagnosis': member.diagnosis,
                    'source': ','.join(member.source),
                    'icd10_codes': ','.join(member.icd10_codes),
                    'age': member.age,
                    'sex': member.sex,
                    'has_metabolomics': member.has_metabolomics
                })

            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)

            logger.info(f"Exported metadata: {output_file}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Build ADHD/Autism cohorts from UK Biobank data"
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input CSV file (extracted UK Biobank data)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/ukb'),
        help='Output directory'
    )

    parser.add_argument(
        '--n-controls',
        type=int,
        default=10000,
        help='Number of controls (default: 10000)'
    )

    parser.add_argument(
        '--metabolomics-only',
        action='store_true',
        help='Only include participants with metabolomics data'
    )

    parser.add_argument(
        '--export-ids',
        action='store_true',
        help='Export cohort IDs to text files'
    )

    parser.add_argument(
        '--export-metadata',
        action='store_true',
        help='Export detailed cohort metadata'
    )

    args = parser.parse_args()

    # Build cohorts
    builder = UKBCohortBuilder(args.input)

    cohorts = builder.build_cohorts(
        n_controls=args.n_controls,
        metabolomics_only=args.metabolomics_only
    )

    # Export
    if args.export_ids:
        builder.export_cohort_ids(args.output)

    if args.export_metadata:
        builder.export_cohort_metadata(args.output)

    # Summary
    print("\n" + "="*60)
    print("Cohort Building Complete")
    print("="*60)
    for name, members in cohorts.items():
        print(f"{name}: {len(members)}")
    print("="*60)


if __name__ == "__main__":
    main()