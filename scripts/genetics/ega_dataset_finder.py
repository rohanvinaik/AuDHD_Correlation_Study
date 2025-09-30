#!/usr/bin/env python3
"""
EGA Dataset Finder for ADHD/Autism Research

Searches European Genome-phenome Archive (EGA) for relevant datasets using
their REST API and metadata search.

EGA contains:
- 2,000+ studies
- European cohorts and biobanks
- WGS, WES, RNA-seq, methylation, ChIP-seq
- Controlled access with DAC approval

Requirements:
    pip install requests pandas

Usage:
    # Search for ADHD/autism datasets
    python ega_dataset_finder.py --search --output data/genetics/

    # Get details for specific dataset
    python ega_dataset_finder.py --dataset EGAD00001000001 --output data/genetics/

    # Search by study
    python ega_dataset_finder.py --study EGAS00001000001

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
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


# EGA API endpoints
EGA_API_BASE = "https://ega-archive.org/metadata/v2"

# Search terms
SEARCH_TERMS = {
    'adhd': ['ADHD', 'attention deficit', 'hyperactivity', 'hyperkinetic'],
    'autism': ['autism', 'ASD', 'autistic', 'Asperger'],
    'neurodevelopmental': ['neurodevelopmental', 'developmental disorder']
}

# Data types
DATA_TYPES = ['WGS', 'WES', 'RNA-seq', 'methylation', 'ChIP-seq', 'genotyping']


@dataclass
class EGADataset:
    """Represents an EGA dataset"""
    dataset_id: str
    study_id: str
    title: str
    description: str
    data_types: List[str]
    num_samples: int
    genome_assembly: str
    dac_id: str
    policy_id: str
    relevance_score: float
    matched_terms: List[str]
    url: str


class EGADatasetFinder:
    """Search EGA for ADHD/Autism datasets"""

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

        logger.info(f"Initialized EGA finder: {output_dir}")

    def search_datasets(self, query: str) -> List[Dict]:
        """
        Search EGA datasets (note: API has limited search functionality)

        Args:
            query: Search query

        Returns:
            List of dataset metadata
        """
        logger.info(f"Searching EGA with query: {query}")

        # Note: EGA API doesn't have a direct search endpoint
        # Would need to iterate through datasets or use web interface
        # Here we provide structure for known datasets

        # Placeholder for API call (actual implementation would vary)
        # In practice, may need to scrape web interface or contact EGA directly

        logger.warning("EGA API has limited programmatic search. Manual curation recommended.")
        return []

    def get_dataset_details(self, dataset_id: str) -> Optional[Dict]:
        """
        Get dataset details from EGA API

        Args:
            dataset_id: EGA dataset ID (e.g., EGAD00001000001)

        Returns:
            Dataset metadata dictionary
        """
        try:
            url = f"{EGA_API_BASE}/datasets/{dataset_id}"
            response = self.session.get(url, timeout=30)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {dataset_id}: HTTP {response.status_code}")
                return None

            data = response.json()
            return data

        except Exception as e:
            logger.error(f"Error fetching {dataset_id}: {e}")
            return None

    def get_study_details(self, study_id: str) -> Optional[Dict]:
        """
        Get study details from EGA API

        Args:
            study_id: EGA study ID (e.g., EGAS00001000001)

        Returns:
            Study metadata dictionary
        """
        try:
            url = f"{EGA_API_BASE}/studies/{study_id}"
            response = self.session.get(url, timeout=30)

            if response.status_code != 200:
                return None

            data = response.json()
            return data

        except Exception as e:
            logger.error(f"Error fetching study {study_id}: {e}")
            return None

    def _calculate_relevance(self, title: str, description: str) -> tuple[float, List[str]]:
        """Calculate relevance score"""
        score = 0.0
        matched_terms = []

        searchable_text = f"{title} {description}".lower()

        # Check ADHD terms
        for term in SEARCH_TERMS['adhd']:
            if term.lower() in searchable_text:
                score += 10.0
                matched_terms.append(term)

        # Check autism terms
        for term in SEARCH_TERMS['autism']:
            if term.lower() in searchable_text:
                score += 10.0
                matched_terms.append(term)

        # Check neurodevelopmental
        for term in SEARCH_TERMS['neurodevelopmental']:
            if term.lower() in searchable_text:
                score += 5.0
                matched_terms.append(term)

        # Bonus for specific data types
        for data_type in DATA_TYPES:
            if data_type.lower() in searchable_text:
                score += 2.0

        return score, matched_terms

    def get_known_datasets(self) -> List[Dict]:
        """
        Return known ADHD/Autism datasets in EGA (manually curated)

        Returns:
            List of known dataset metadata
        """
        # Note: These are examples/placeholders
        # Actual dataset IDs would need to be discovered through EGA portal

        known_datasets = [
            {
                'dataset_id': 'EGAD00001000XXX',
                'study_id': 'EGAS00001000XXX',
                'title': 'Autism European Cohort WGS',
                'description': 'Whole genome sequencing of European autism cases and controls',
                'data_types': ['WGS'],
                'num_samples': 1500,
                'genome_assembly': 'GRCh38',
                'relevance_score': 15.0,
                'matched_terms': ['autism', 'WGS'],
                'note': 'Placeholder - actual ID to be determined'
            },
            {
                'dataset_id': 'EGAD00001000YYY',
                'study_id': 'EGAS00001000YYY',
                'title': 'ADHD Methylation Study',
                'description': 'DNA methylation profiling in ADHD patients',
                'data_types': ['methylation'],
                'num_samples': 500,
                'genome_assembly': 'GRCh38',
                'relevance_score': 13.0,
                'matched_terms': ['ADHD', 'methylation'],
                'note': 'Placeholder - actual ID to be determined'
            }
        ]

        return known_datasets

    def generate_dataset_catalog(self, datasets: List[Dict]) -> pd.DataFrame:
        """Generate catalog of datasets"""
        if not datasets:
            return pd.DataFrame()

        df = pd.DataFrame(datasets)
        if 'relevance_score' in df.columns:
            df = df.sort_values('relevance_score', ascending=False)

        return df


def main():
    parser = argparse.ArgumentParser(
        description='Search EGA for ADHD/Autism datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List known datasets
  python ega_dataset_finder.py --known-datasets

  # Get details for specific dataset
  python ega_dataset_finder.py --dataset EGAD00001000001

  # Get study details
  python ega_dataset_finder.py --study EGAS00001000001

Note: EGA requires controlled access approval.
Apply at: https://ega-archive.org/access/data-access-committee
        """
    )

    parser.add_argument(
        '--search',
        action='store_true',
        help='Search for ADHD/autism datasets (limited API support)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Get details for specific dataset (EGAD ID)'
    )

    parser.add_argument(
        '--study',
        type=str,
        help='Get details for specific study (EGAS ID)'
    )

    parser.add_argument(
        '--known-datasets',
        action='store_true',
        help='List known ADHD/autism datasets'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/genetics/ega',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize finder
    finder = EGADatasetFinder(Path(args.output))

    # Handle known datasets
    if args.known_datasets or not any([args.search, args.dataset, args.study]):
        known = finder.get_known_datasets()
        print("\n=== Known ADHD/Autism Datasets in EGA ===\n")
        print("Note: EGA has limited programmatic search. These are examples.")
        print("Please search EGA portal manually: https://ega-archive.org\n")

        for dataset in known:
            print(f"{dataset['dataset_id']}: {dataset['title']}")
            print(f"  Study: {dataset['study_id']}")
            print(f"  Data: {', '.join(dataset['data_types'])}")
            print(f"  Samples: {dataset['num_samples']}")
            print(f"  Note: {dataset.get('note', '')}")
            print()

        # Save catalog
        catalog_df = finder.generate_dataset_catalog(known)
        catalog_file = finder.output_dir / 'ega_known_datasets.csv'
        catalog_df.to_csv(catalog_file, index=False)
        print(f"Catalog saved: {catalog_file}")

        return

    # Handle specific dataset
    if args.dataset:
        details = finder.get_dataset_details(args.dataset)
        if details:
            print(f"\n=== {args.dataset} Details ===\n")
            print(json.dumps(details, indent=2))
        else:
            print(f"Dataset {args.dataset} not found or not accessible")

    # Handle specific study
    if args.study:
        details = finder.get_study_details(args.study)
        if details:
            print(f"\n=== {args.study} Details ===\n")
            print(json.dumps(details, indent=2))
        else:
            print(f"Study {args.study} not found or not accessible")


if __name__ == '__main__':
    main()