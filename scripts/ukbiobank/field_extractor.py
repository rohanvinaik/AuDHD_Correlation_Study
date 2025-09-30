#!/usr/bin/env python3
"""
UK Biobank Field Extractor

Extract specific phenotype fields from UK Biobank dataset using ukbconv.
Focuses on ADHD/Autism relevant fields including:
- Mental health diagnoses (ICD-10: F90, F84)
- Medications (ADHD medications)
- Metabolomics (Nightingale NMR platform, 249 biomarkers)
- Genetics (PCs, kinship)
- Environmental exposures
- Sleep, diet, family history

Usage:
    # Extract all ADHD/Autism relevant fields
    python field_extractor.py --ukb-file ukb12345.enc_ukb --output data/ukb/

    # Extract specific field groups
    python field_extractor.py --ukb-file ukb12345.enc_ukb --groups diagnosis,metabolomics

    # Extract custom field list
    python field_extractor.py --ukb-file ukb12345.enc_ukb --fields 20544,41270,23400-23649

Requires:
    - ukbconv (UK Biobank extraction tool)
    - UK Biobank .enc_ukb file and .key file
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set

try:
    import pandas as pd
    import yaml
except ImportError:
    print("ERROR: Required packages not installed")
    print("Install with: pip install pandas pyyaml")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# UK Biobank field definitions
UKB_FIELD_GROUPS = {
    'diagnosis': {
        'description': 'Mental health diagnoses (ICD-10)',
        'fields': [
            '20544',  # Mental health problems ever diagnosed by professional
            '41270',  # ICD-10 diagnoses (main)
            '41271',  # ICD-10 diagnoses (secondary)
            '41280',  # Date of first ICD-10 diagnosis
        ],
        'icd10_codes': ['F90.0', 'F90.1', 'F90.2', 'F90.8', 'F90.9',  # ADHD
                        'F84.0', 'F84.1', 'F84.5', 'F84.8', 'F84.9']  # Autism
    },
    'medications': {
        'description': 'Medication history',
        'fields': [
            '20003',  # Treatment/medication code
            '20004',  # Date of treatment/medication
            '6153',   # Medication for cholesterol, blood pressure or diabetes
            '6177',   # Medication for pain relief, constipation, heartburn
        ],
        'adhd_med_codes': [
            '1140926662',  # Methylphenidate
            '1140888594',  # Atomoxetine
            '1141173882',  # Dexamfetamine
            '1140888648',  # Lisdexamfetamine
        ]
    },
    'metabolomics': {
        'description': 'NMR metabolomics (Nightingale platform, 249 biomarkers)',
        'fields': list(range(23400, 23650)),  # Fields 23400-23649
        'categories': [
            'lipoproteins',
            'fatty_acids',
            'amino_acids',
            'glycolysis',
            'ketone_bodies',
            'inflammation'
        ]
    },
    'genetics': {
        'description': 'Genetic principal components and quality',
        'fields': [
            '22001',  # Genetic sex
            '22006',  # Genetic ethnic grouping
            '22009',  # Genetic principal components (1-40)
            '22020',  # Genetic kinship
            '22021',  # Genetic kinship ID
            '22000',  # Genotype batch
        ]
    },
    'environmental': {
        'description': 'Environmental exposures',
        'fields': [
            '24006',  # Air pollution (PM2.5)
            '24007',  # Air pollution (PM2.5-10)
            '24008',  # Air pollution (PM10)
            '24016',  # Air pollution (NO2)
            '24017',  # Air pollution (NOx)
            '24003',  # Noise pollution (day)
            '24004',  # Noise pollution (evening)
            '24005',  # Noise pollution (night)
        ]
    },
    'diet': {
        'description': 'Dietary intake',
        'fields': list(range(1309, 1360)) + [  # Fields 1309-1359
            '1558',  # Alcohol intake frequency
            '1568',  # Average weekly red wine intake
            '1578',  # Average weekly beer plus cider intake
            '6144',  # Oily fish intake
        ]
    },
    'sleep': {
        'description': 'Sleep patterns and disorders',
        'fields': [
            '1160',  # Sleep duration
            '1170',  # Getting up in morning
            '1180',  # Morning/evening person
            '1190',  # Nap during day
            '1200',  # Sleeplessness/insomnia
            '1210',  # Snoring
        ]
    },
    'mental_health': {
        'description': 'Mental health questionnaire scores',
        'fields': [
            '20400',  # Recent easy annoyance or irritability
            '20401',  # Recent feelings or nervousness or anxiety
            '20402',  # Recent restlessness
            '20403',  # Recent inability to stop or control worrying
            '20404',  # Recent worrying too much about different things
            '20405',  # Recent trouble relaxing
            '20406',  # Recent being so restless
            '20407',  # Recent becoming easily annoyed or irritable
            '20408',  # Recent feeling afraid
            '20409',  # Recent feelings of foreboding
            # PHQ-9 depression
            '20510',  # Recent thoughts of suicide or self-harm
            '20511',  # Recent poor appetite or overeating
            '20512',  # Recent trouble concentrating
            '20513',  # Recent feelings of tiredness or low energy
            '20514',  # Recent feelings of inadequacy
        ]
    },
    'family_history': {
        'description': 'Family history of psychiatric disorders',
        'fields': [
            '20107',  # Illness of father
            '20110',  # Illness of mother
            '20111',  # Illness of siblings
            '20074',  # Father\'s age at death
            '20075',  # Mother\'s age at death
        ]
    },
    'birth_factors': {
        'description': 'Birth and early development',
        'fields': [
            '20022',  # Birth weight
            '2744',   # Birth weight of first child
            '2754',   # Gestational age
            '1777',   # Breastfed as a baby
            '1787',   # Adopted as a child
        ]
    },
    'ses': {
        'description': 'Socioeconomic status indicators',
        'fields': [
            '738',    # Average total household income before tax
            '6138',   # Qualifications
            '845',    # Age completed full time education
            '6142',   # Current employment status
            '680',    # Job involves mainly walking or standing
            '26410',  # Forced to give up work/role due to illness
        ]
    },
    'demographics': {
        'description': 'Basic demographics',
        'fields': [
            '31',     # Sex
            '21003',  # Age at assessment
            '21000',  # Ethnic background
            '21001',  # BMI
            '54',     # Assessment centre
            '53',     # Date of assessment
        ]
    },
    'physical_health': {
        'description': 'Physical health conditions',
        'fields': [
            '20002',  # Non-cancer illness code, self-reported
            '20001',  # Cancer code, self-reported
            '20008',  # Age of cancer diagnosis
            '2443',   # Diabetes diagnosed by doctor
            '6150',   # Vascular/heart problems diagnosed by doctor
        ]
    }
}


@dataclass
class FieldExtractionJob:
    """Represents a field extraction job"""
    ukb_file: Path
    output_dir: Path
    field_groups: List[str] = field(default_factory=list)
    custom_fields: List[str] = field(default_factory=list)
    format: str = 'csv'  # csv, r, docs
    encoding: str = 'ascii'
    include_withdrawn: bool = False


class UKBFieldExtractor:
    """Extract fields from UK Biobank dataset"""

    def __init__(self, ukb_file: Path, output_dir: Path):
        """
        Initialize UK Biobank field extractor

        Args:
            ukb_file: Path to .enc_ukb file
            output_dir: Output directory for extracted data
        """
        self.ukb_file = Path(ukb_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check ukbconv availability
        self._check_ukbconv()

    def _check_ukbconv(self):
        """Check if ukbconv is available"""
        try:
            result = subprocess.run(
                ['ukbconv', '--help'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0:
                raise FileNotFoundError()

            logger.info("✓ ukbconv found")

        except FileNotFoundError:
            logger.error("ukbconv not found")
            logger.error("Download from: https://biobank.ndph.ox.ac.uk/ukb/refer.cgi?id=644")
            logger.error("Or contact UK Biobank support")
            sys.exit(1)

    def get_all_fields(self, field_groups: List[str]) -> Set[str]:
        """
        Get all field IDs for specified groups

        Args:
            field_groups: List of field group names

        Returns:
            Set of field IDs
        """
        all_fields = set()

        for group in field_groups:
            if group not in UKB_FIELD_GROUPS:
                logger.warning(f"Unknown field group: {group}")
                continue

            group_data = UKB_FIELD_GROUPS[group]
            fields = group_data['fields']

            # Handle range fields
            for f in fields:
                if isinstance(f, int):
                    all_fields.add(str(f))
                else:
                    all_fields.add(f)

        return all_fields

    def extract_fields(
        self,
        field_ids: Set[str],
        output_name: str,
        format: str = 'csv'
    ) -> Optional[Path]:
        """
        Extract specific fields using ukbconv

        Args:
            field_ids: Set of field IDs to extract
            output_name: Output file name (without extension)
            format: Output format (csv, r, docs)

        Returns:
            Path to extracted file or None if failed
        """
        logger.info(f"Extracting {len(field_ids)} fields...")

        # Create field specification file
        field_file = self.output_dir / f"{output_name}_fields.txt"
        with open(field_file, 'w') as f:
            for field_id in sorted(field_ids, key=lambda x: int(x) if x.isdigit() else x):
                f.write(f"{field_id}\n")

        logger.info(f"Field list saved to: {field_file}")

        # Build ukbconv command
        output_path = self.output_dir / output_name

        cmd = [
            'ukbconv',
            str(self.ukb_file),
            format,
            '-e', 'ascii',  # Encoding
            '-i', str(field_file),  # Field list
            '-o', str(output_path)
        ]

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                output_file = output_path.with_suffix(f'.{format}')
                logger.info(f"✓ Extraction complete: {output_file}")
                return output_file
            else:
                logger.error(f"ukbconv failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Extraction timeout (> 1 hour)")
            return None
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return None

    def extract_by_groups(
        self,
        field_groups: List[str],
        format: str = 'csv',
        separate_files: bool = False
    ) -> Dict[str, Path]:
        """
        Extract fields by groups

        Args:
            field_groups: List of field group names
            format: Output format
            separate_files: Create separate file for each group

        Returns:
            Dictionary mapping group names to output files
        """
        output_files = {}

        if separate_files:
            # Extract each group to separate file
            for group in field_groups:
                if group not in UKB_FIELD_GROUPS:
                    logger.warning(f"Skipping unknown group: {group}")
                    continue

                fields = self.get_all_fields([group])
                output_name = f"ukb_{group}"

                output_file = self.extract_fields(
                    fields,
                    output_name,
                    format
                )

                if output_file:
                    output_files[group] = output_file

        else:
            # Extract all groups to single file
            all_fields = self.get_all_fields(field_groups)
            output_name = f"ukb_{'_'.join(field_groups)}"

            output_file = self.extract_fields(
                all_fields,
                output_name,
                format
            )

            if output_file:
                output_files['combined'] = output_file

        return output_files

    def extract_custom_fields(
        self,
        field_ids: List[str],
        output_name: str,
        format: str = 'csv'
    ) -> Optional[Path]:
        """
        Extract custom field list

        Args:
            field_ids: List of field IDs
            output_name: Output file name
            format: Output format

        Returns:
            Path to output file
        """
        # Expand field ranges (e.g., "23400-23649")
        expanded_fields = set()

        for field_spec in field_ids:
            if '-' in field_spec:
                # Range specification
                start, end = field_spec.split('-')
                start = int(start)
                end = int(end)

                for field_id in range(start, end + 1):
                    expanded_fields.add(str(field_id))
            else:
                expanded_fields.add(field_spec)

        return self.extract_fields(expanded_fields, output_name, format)

    def generate_field_documentation(self, field_groups: List[str]) -> Path:
        """
        Generate documentation for extracted fields

        Args:
            field_groups: List of field groups

        Returns:
            Path to documentation file
        """
        doc_path = self.output_dir / "field_documentation.md"

        with open(doc_path, 'w') as f:
            f.write("# UK Biobank Field Documentation\n\n")
            f.write(f"**Generated**: {pd.Timestamp.now()}\n\n")

            for group in field_groups:
                if group not in UKB_FIELD_GROUPS:
                    continue

                group_data = UKB_FIELD_GROUPS[group]

                f.write(f"## {group.title()}\n\n")
                f.write(f"**Description**: {group_data['description']}\n\n")
                f.write(f"**Fields**: {len(group_data['fields'])}\n\n")

                # Field list
                f.write("### Field IDs\n\n")
                fields = group_data['fields']
                if len(fields) <= 20:
                    # Show all fields
                    for field_id in fields:
                        f.write(f"- `{field_id}`\n")
                else:
                    # Show range
                    if isinstance(fields[0], int):
                        f.write(f"- Range: `{fields[0]}` to `{fields[-1]}`\n")
                    else:
                        f.write(f"- {len(fields)} fields total\n")

                f.write("\n")

                # Special notes
                if 'icd10_codes' in group_data:
                    f.write("**ICD-10 Codes**:\n")
                    for code in group_data['icd10_codes']:
                        f.write(f"- `{code}`\n")
                    f.write("\n")

                if 'categories' in group_data:
                    f.write("**Categories**:\n")
                    for cat in group_data['categories']:
                        f.write(f"- {cat}\n")
                    f.write("\n")

        logger.info(f"Documentation saved to: {doc_path}")
        return doc_path

    def get_extraction_summary(self) -> Dict:
        """Get summary of available field groups"""
        summary = {
            'total_groups': len(UKB_FIELD_GROUPS),
            'groups': {}
        }

        for group, data in UKB_FIELD_GROUPS.items():
            summary['groups'][group] = {
                'description': data['description'],
                'n_fields': len(data['fields']),
                'fields': data['fields'][:5] if len(data['fields']) <= 5 else data['fields'][:5]
            }

        return summary


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Extract UK Biobank fields for ADHD/Autism study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all mental health related fields
  python field_extractor.py --ukb-file ukb12345.enc_ukb --groups diagnosis,mental_health,medications

  # Extract metabolomics data
  python field_extractor.py --ukb-file ukb12345.enc_ukb --groups metabolomics

  # Extract custom field list
  python field_extractor.py --ukb-file ukb12345.enc_ukb --fields 20544,41270,23400-23649

  # List available field groups
  python field_extractor.py --list-groups

  # Generate documentation
  python field_extractor.py --ukb-file ukb12345.enc_ukb --groups all --docs
        """
    )

    parser.add_argument(
        '--ukb-file',
        type=Path,
        help='UK Biobank .enc_ukb file'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/ukb'),
        help='Output directory (default: data/ukb)'
    )

    parser.add_argument(
        '--groups',
        help='Comma-separated field groups (or "all")'
    )

    parser.add_argument(
        '--fields',
        help='Comma-separated custom field IDs (supports ranges: 23400-23649)'
    )

    parser.add_argument(
        '--format',
        choices=['csv', 'r', 'docs'],
        default='csv',
        help='Output format (default: csv)'
    )

    parser.add_argument(
        '--separate',
        action='store_true',
        help='Create separate file for each group'
    )

    parser.add_argument(
        '--list-groups',
        action='store_true',
        help='List available field groups'
    )

    parser.add_argument(
        '--docs',
        action='store_true',
        help='Generate field documentation'
    )

    args = parser.parse_args()

    # List groups
    if args.list_groups:
        print("\nAvailable Field Groups:")
        print("=" * 60)
        for group, data in UKB_FIELD_GROUPS.items():
            print(f"\n{group}:")
            print(f"  Description: {data['description']}")
            print(f"  Fields: {len(data['fields'])}")
        print("\n")
        return

    # Require UKB file for extraction
    if not args.ukb_file:
        parser.error("--ukb-file required (unless using --list-groups)")

    # Initialize extractor
    extractor = UKBFieldExtractor(args.ukb_file, args.output)

    # Extract by groups
    if args.groups:
        if args.groups.lower() == 'all':
            groups = list(UKB_FIELD_GROUPS.keys())
        else:
            groups = [g.strip() for g in args.groups.split(',')]

        logger.info(f"Extracting groups: {', '.join(groups)}")

        output_files = extractor.extract_by_groups(
            groups,
            format=args.format,
            separate_files=args.separate
        )

        print("\nExtracted Files:")
        for group, file_path in output_files.items():
            print(f"  {group}: {file_path}")

        # Generate documentation
        if args.docs or args.format == 'docs':
            doc_path = extractor.generate_field_documentation(groups)
            print(f"\nDocumentation: {doc_path}")

    # Extract custom fields
    elif args.fields:
        field_list = [f.strip() for f in args.fields.split(',')]

        logger.info(f"Extracting {len(field_list)} custom fields")

        output_file = extractor.extract_custom_fields(
            field_list,
            'ukb_custom',
            format=args.format
        )

        if output_file:
            print(f"\nExtracted: {output_file}")

    else:
        parser.error("Either --groups or --fields required")


if __name__ == "__main__":
    main()