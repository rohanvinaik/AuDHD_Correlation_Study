#!/usr/bin/env python3
"""
Data Sharing Finder for Clinical Trials

Identifies clinical trials with:
- Individual Participant Data (IPD) sharing statements
- Biospecimen retention
- Data sharing timelines
- Access procedures
- Principal investigator contact information

Requirements:
    pip install requests pandas

Usage:
    # Find trials with IPD sharing
    python data_sharing_finder.py \\
        --input data/trials/trials_with_biomarkers.csv \\
        --output data/trials/

    # Generate PI contact list
    python data_sharing_finder.py \\
        --input data/trials/trials_with_biomarkers.csv \\
        --generate-contacts \\
        --output data/trials/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

try:
    import requests
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install requests pandas")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CT_API_BASE = "https://clinicaltrials.gov/api/v2"


class DataSharingFinder:
    """Find data sharing opportunities in clinical trials"""

    def __init__(self, output_dir: Path):
        """
        Initialize finder

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        logger.info(f"Initialized data sharing finder: {output_dir}")

    def get_ipd_info(self, nct_id: str) -> Dict:
        """
        Get detailed IPD sharing information

        Args:
            nct_id: NCT identifier

        Returns:
            Dictionary with IPD details
        """
        try:
            response = self.session.get(
                f"{CT_API_BASE}/studies/{nct_id}",
                params={'format': 'json'},
                timeout=30
            )

            if response.status_code != 200:
                return {}

            data = response.json()
            protocol_section = data.get('protocolSection', {})

            ipd_module = protocol_section.get('ipdSharingStatementModule', {})

            ipd_info = {
                'nct_id': nct_id,
                'ipd_sharing': ipd_module.get('ipdSharing', 'No'),
                'ipd_sharing_description': ipd_module.get('description', ''),
                'ipd_info_types': ipd_module.get('infoTypes', []),
                'ipd_time_frame': ipd_module.get('timeFrame', ''),
                'ipd_access_criteria': ipd_module.get('accessCriteria', ''),
                'ipd_url': ipd_module.get('url', '')
            }

            return ipd_info

        except Exception as e:
            logger.error(f"Error fetching IPD info for {nct_id}: {e}")
            return {}

    def analyze_trials(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze trials for data sharing opportunities

        Args:
            trials_df: DataFrame with trial information

        Returns:
            Enhanced DataFrame with IPD details
        """
        logger.info(f"Analyzing {len(trials_df)} trials for data sharing...")

        ipd_data = []

        for idx, row in trials_df.iterrows():
            nct_id = row['nct_id']

            if idx % 20 == 0:
                logger.info(f"Processed {idx}/{len(trials_df)} trials...")

            ipd_info = self.get_ipd_info(nct_id)

            if ipd_info:
                # Combine with trial info
                combined = {
                    'nct_id': nct_id,
                    'title': row.get('title', ''),
                    'status': row.get('status', ''),
                    'enrollment': row.get('enrollment', 0),
                    'sponsor': row.get('sponsor', ''),
                    **ipd_info
                }
                ipd_data.append(combined)

        result_df = pd.DataFrame(ipd_data)

        logger.info(f"Completed analysis of {len(result_df)} trials")
        return result_df

    def filter_data_sharing_trials(self, ipd_df: pd.DataFrame) -> pd.DataFrame:
        """Filter trials with active data sharing"""
        # Filter for positive IPD sharing
        sharing = ipd_df[ipd_df['ipd_sharing'].isin(['Yes', 'Undecided'])]

        logger.info(f"Found {len(sharing)} trials with data sharing plans")
        return sharing

    def generate_pi_contacts(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        """Generate PI contact list"""
        logger.info("Generating PI contact list...")

        contacts = []

        for _, row in trials_df.iterrows():
            if row.get('pi_name') and row.get('pi_name') != '':
                contact = {
                    'nct_id': row['nct_id'],
                    'trial_title': row.get('title', ''),
                    'pi_name': row.get('pi_name', ''),
                    'pi_affiliation': row.get('pi_affiliation', ''),
                    'contact_email': row.get('contact_email', ''),
                    'sponsor': row.get('sponsor', ''),
                    'status': row.get('status', ''),
                    'enrollment': row.get('enrollment', 0),
                    'has_ipd_sharing': row.get('ipd_sharing', 'No'),
                    'trial_url': row.get('url', '')
                }
                contacts.append(contact)

        contacts_df = pd.DataFrame(contacts)

        logger.info(f"Generated {len(contacts_df)} PI contacts")
        return contacts_df


def main():
    parser = argparse.ArgumentParser(
        description='Find data sharing opportunities in clinical trials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze trials for IPD sharing
  python data_sharing_finder.py \\
      --input data/trials/trials_with_biomarkers.csv \\
      --output data/trials/

  # Generate PI contact list
  python data_sharing_finder.py \\
      --input data/trials/trials_with_biomarkers.csv \\
      --generate-contacts \\
      --output data/trials/
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input trials CSV file'
    )

    parser.add_argument(
        '--generate-contacts',
        action='store_true',
        help='Generate PI contact list'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/trials',
        help='Output directory'
    )

    args = parser.parse_args()

    # Load trials
    trials_df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(trials_df)} trials")

    # Initialize finder
    finder = DataSharingFinder(Path(args.output))

    # Analyze for IPD sharing
    ipd_df = finder.analyze_trials(trials_df)

    # Filter for data sharing
    sharing_df = finder.filter_data_sharing_trials(ipd_df)

    # Export IPD data
    ipd_file = finder.output_dir / 'trials_ipd_sharing.csv'
    sharing_df.to_csv(ipd_file, index=False)

    print(f"\n=== Data Sharing Analysis ===\n")
    print(f"Total trials analyzed: {len(ipd_df)}")
    print(f"Trials with IPD sharing: {len(sharing_df)}")

    if len(sharing_df) > 0:
        print(f"\nIPD sharing breakdown:")
        print(sharing_df['ipd_sharing'].value_counts().to_string())

        print(f"\nTrials with access criteria: {sharing_df['ipd_access_criteria'].notna().sum()}")
        print(f"Trials with IPD URL: {sharing_df['ipd_url'].notna().sum()}")

    print(f"\nResults saved: {ipd_file}")

    # Generate PI contacts if requested
    if args.generate_contacts:
        contacts_df = finder.generate_pi_contacts(trials_df)

        contacts_file = finder.output_dir / 'pi_contact_list.csv'
        contacts_df.to_csv(contacts_file, index=False)

        print(f"\n=== PI Contact List ===\n")
        print(f"Total PIs: {len(contacts_df)}")
        print(f"PIs with email: {contacts_df['contact_email'].notna().sum()}")
        print(f"PIs with IPD sharing: {(contacts_df['has_ipd_sharing'] != 'No').sum()}")

        print(f"\nContact list saved: {contacts_file}")


if __name__ == '__main__':
    main()