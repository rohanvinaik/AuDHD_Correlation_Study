#!/usr/bin/env python3
"""
Metabolomics Workbench API Interface

Searches NIH Metabolomics Workbench for ADHD/Autism-relevant studies and downloads
metabolite data, pathway information, and study metadata.

Metabolomics Workbench:
- ~2,500 studies (ST000001-ST002500+)
- REST API for programmatic access
- Multiple organisms and sample types
- Untargeted and targeted metabolomics

Requirements:
    pip install requests pandas tqdm

Usage:
    # Search all studies for ADHD/autism
    python workbench_api.py --search-all --output data/metabolomics/

    # Search specific study range
    python workbench_api.py --range ST001000-ST002000 --output data/metabolomics/

    # Download specific study
    python workbench_api.py --study ST001234 --download-data --output data/metabolomics/

    # Get study summary
    python workbench_api.py --study ST001234 --summary

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

try:
    import requests
    import pandas as pd
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install requests pandas tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Metabolomics Workbench REST API
WORKBENCH_API_BASE = "https://www.metabolomicsworkbench.org/rest"

# Search terms
SEARCH_TERMS = {
    'adhd': ['ADHD', 'attention deficit', 'hyperactivity', 'hyperkinetic'],
    'autism': ['autism', 'ASD', 'autistic', 'Asperger'],
    'neurodevelopmental': ['neurodevelopmental', 'developmental disorder', 'neurodevelopment'],
    'neurotransmitters': ['GABA', 'glutamate', 'dopamine', 'serotonin', 'norepinephrine']
}

# Sample types
SAMPLE_TYPES = ['Human', 'Homo sapiens', 'plasma', 'serum', 'blood', 'urine', 'CSF']


@dataclass
class WorkbenchStudy:
    """Represents a Metabolomics Workbench study"""
    study_id: str
    title: str
    summary: str
    institute: str
    last_name: str
    submit_date: str
    num_subjects: int
    species: str
    sample_type: str
    analysis_type: str
    relevance_score: float
    matched_terms: List[str]
    url: str


class WorkbenchAPI:
    """Interface to Metabolomics Workbench REST API"""

    def __init__(self, output_dir: Path):
        """
        Initialize API interface

        Args:
            output_dir: Output directory for data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        # Cache
        self.cache_dir = self.output_dir / '.cache'
        self.cache_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized Workbench API: {output_dir}")

    def get_study_summary(self, study_id: str) -> Optional[Dict]:
        """
        Get study summary from Workbench API

        Args:
            study_id: Study ID (e.g., 'ST001234')

        Returns:
            Study summary dictionary
        """
        # Check cache
        cache_file = self.cache_dir / f"{study_id}_summary.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except:
                pass

        try:
            # API endpoint: /study/study_id/{study_id}/summary
            url = f"{WORKBENCH_API_BASE}/study/study_id/{study_id}/summary"
            response = self.session.get(url, timeout=30)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {study_id}: HTTP {response.status_code}")
                return None

            # Parse response (tab-separated text)
            lines = response.text.strip().split('\n')

            if len(lines) < 2:
                return None

            # Parse header and values
            headers = lines[0].split('\t')
            values = lines[1].split('\t')

            summary = dict(zip(headers, values))

            # Cache
            with open(cache_file, 'w') as f:
                json.dump(summary, f, indent=2)

            return summary

        except Exception as e:
            logger.error(f"Error fetching {study_id}: {e}")
            return None

    def get_study_metabolites(self, study_id: str) -> Optional[pd.DataFrame]:
        """
        Get metabolite data for a study

        Args:
            study_id: Study ID

        Returns:
            DataFrame with metabolite data
        """
        try:
            # API endpoint: /study/study_id/{study_id}/metabolites
            url = f"{WORKBENCH_API_BASE}/study/study_id/{study_id}/metabolites"
            response = self.session.get(url, timeout=30)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch metabolites for {study_id}")
                return None

            # Parse tab-separated response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), sep='\t')

            logger.info(f"Retrieved {len(df)} metabolites for {study_id}")
            return df

        except Exception as e:
            logger.error(f"Error fetching metabolites for {study_id}: {e}")
            return None

    def get_study_data(self, study_id: str, analysis_id: str = None) -> Optional[pd.DataFrame]:
        """
        Get complete metabolite concentration data

        Args:
            study_id: Study ID
            analysis_id: Analysis ID (optional, will use first if not specified)

        Returns:
            DataFrame with metabolite concentrations
        """
        try:
            if analysis_id:
                url = f"{WORKBENCH_API_BASE}/study/analysis_id/{analysis_id}/data"
            else:
                # Get first analysis for study
                url = f"{WORKBENCH_API_BASE}/study/study_id/{study_id}/data"

            response = self.session.get(url, timeout=60)

            if response.status_code != 200:
                return None

            # Parse response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), sep='\t')

            logger.info(f"Retrieved data: {df.shape[0]} samples × {df.shape[1]} metabolites")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {study_id}: {e}")
            return None

    def search_study_range(self, start: int, end: int) -> List[WorkbenchStudy]:
        """
        Search studies in a range for ADHD/autism relevance

        Args:
            start: Start study number (e.g., 1000 for ST001000)
            end: End study number

        Returns:
            List of relevant studies
        """
        logger.info(f"Searching studies ST{start:06d} to ST{end:06d}...")

        relevant_studies = []

        for study_num in tqdm(range(start, end + 1), desc="Screening studies"):
            study_id = f"ST{study_num:06d}"

            # Get summary
            summary = self.get_study_summary(study_id)

            if not summary:
                continue

            # Calculate relevance
            relevance_score, matched_terms = self._calculate_relevance(summary)

            if relevance_score < 2.0:
                continue

            # Create study object
            study = WorkbenchStudy(
                study_id=study_id,
                title=summary.get('STUDY_TITLE', ''),
                summary=summary.get('STUDY_SUMMARY', ''),
                institute=summary.get('INSTITUTE', ''),
                last_name=summary.get('LAST_NAME', ''),
                submit_date=summary.get('SUBMIT_DATE', ''),
                num_subjects=int(summary.get('SUBJECT_SPECIES_NUM', 0) or 0),
                species=summary.get('SUBJECT_SPECIES', ''),
                sample_type=summary.get('SUBJECT_TYPE', ''),
                analysis_type=summary.get('ANALYSIS_TYPE', ''),
                relevance_score=relevance_score,
                matched_terms=matched_terms,
                url=f"https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID={study_id}"
            )

            relevant_studies.append(study)
            logger.info(f"Found relevant study: {study_id} (score: {relevance_score:.1f})")

            # Rate limiting
            time.sleep(0.2)

        logger.info(f"Found {len(relevant_studies)} relevant studies in range")
        return relevant_studies

    def _calculate_relevance(self, summary: Dict) -> Tuple[float, List[str]]:
        """Calculate relevance score for ADHD/autism research"""
        score = 0.0
        matched_terms = []

        # Extract searchable text
        title = summary.get('STUDY_TITLE', '').lower()
        study_summary = summary.get('STUDY_SUMMARY', '').lower()
        factors = summary.get('FACTORS', '').lower()

        searchable_text = f"{title} {study_summary} {factors}"

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

        # Check neurodevelopmental terms
        for term in SEARCH_TERMS['neurodevelopmental']:
            if term.lower() in searchable_text:
                score += 5.0
                matched_terms.append(term)

        # Check neurotransmitter terms
        for term in SEARCH_TERMS['neurotransmitters']:
            if term.lower() in searchable_text:
                score += 2.0
                matched_terms.append(term)

        # Bonus for human studies
        species = summary.get('SUBJECT_SPECIES', '').lower()
        if 'human' in species or 'homo sapiens' in species:
            score += 5.0

        # Bonus for relevant sample types
        sample_type = summary.get('SUBJECT_TYPE', '').lower()
        for st in SAMPLE_TYPES:
            if st.lower() in sample_type:
                score += 2.0
                break

        return score, matched_terms

    def search_all_studies(self, max_study: int = 2500) -> List[WorkbenchStudy]:
        """
        Search all Workbench studies (ST000001 to ST{max_study})

        Args:
            max_study: Maximum study number to check

        Returns:
            List of relevant studies
        """
        return self.search_study_range(1, max_study)

    def download_study_complete(self, study_id: str) -> Dict[str, Path]:
        """
        Download complete study data (summary, metabolites, data)

        Args:
            study_id: Study ID

        Returns:
            Dict mapping data type to file path
        """
        logger.info(f"Downloading complete data for {study_id}...")

        study_dir = self.output_dir / study_id
        study_dir.mkdir(exist_ok=True)

        downloaded = {}

        # Summary
        summary = self.get_study_summary(study_id)
        if summary:
            summary_file = study_dir / f"{study_id}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            downloaded['summary'] = summary_file
            logger.info(f"Saved summary: {summary_file}")

        # Metabolites
        metabolites_df = self.get_study_metabolites(study_id)
        if metabolites_df is not None and not metabolites_df.empty:
            metabolites_file = study_dir / f"{study_id}_metabolites.csv"
            metabolites_df.to_csv(metabolites_file, index=False)
            downloaded['metabolites'] = metabolites_file
            logger.info(f"Saved metabolites: {metabolites_file}")

        # Data
        data_df = self.get_study_data(study_id)
        if data_df is not None and not data_df.empty:
            data_file = study_dir / f"{study_id}_data.csv"
            data_df.to_csv(data_file, index=False)
            downloaded['data'] = data_file
            logger.info(f"Saved data: {data_file}")

        return downloaded

    def generate_study_catalog(self, studies: List[WorkbenchStudy]) -> pd.DataFrame:
        """Generate catalog of studies"""
        if not studies:
            return pd.DataFrame()

        catalog = [asdict(study) for study in studies]
        df = pd.DataFrame(catalog)
        df = df.sort_values('relevance_score', ascending=False)

        return df


def main():
    parser = argparse.ArgumentParser(
        description='Search Metabolomics Workbench for ADHD/Autism studies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search all studies
  python workbench_api.py --search-all --output data/metabolomics/

  # Search specific range
  python workbench_api.py --range ST001000-ST002000 --output data/metabolomics/

  # Get study summary
  python workbench_api.py --study ST001234 --summary

  # Download complete study data
  python workbench_api.py --study ST001234 --download-data --output data/metabolomics/

  # Download all relevant studies found in search
  python workbench_api.py --range ST001000-ST001500 --download-all
        """
    )

    parser.add_argument(
        '--search-all',
        action='store_true',
        help='Search all studies (ST000001-ST002500)'
    )

    parser.add_argument(
        '--range',
        type=str,
        help='Study range to search (e.g., ST001000-ST002000)'
    )

    parser.add_argument(
        '--study',
        type=str,
        help='Specific study ID (e.g., ST001234)'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print study summary'
    )

    parser.add_argument(
        '--download-data',
        action='store_true',
        help='Download complete study data'
    )

    parser.add_argument(
        '--download-all',
        action='store_true',
        help='Download all relevant studies from search'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/metabolomics/workbench',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize API
    api = WorkbenchAPI(Path(args.output))

    # Handle search
    if args.search_all or args.range:
        if args.search_all:
            relevant_studies = api.search_all_studies(max_study=2500)
        else:
            # Parse range
            match = args.range.upper().replace('ST', '').split('-')
            if len(match) != 2:
                print("Error: Range must be in format ST001000-ST002000")
                sys.exit(1)

            start = int(match[0])
            end = int(match[1])
            relevant_studies = api.search_study_range(start, end)

        if not relevant_studies:
            print("\nNo relevant studies found")
            return

        # Generate catalog
        catalog_df = api.generate_study_catalog(relevant_studies)

        # Save catalog
        catalog_file = api.output_dir / 'workbench_study_catalog.csv'
        catalog_df.to_csv(catalog_file, index=False)
        print(f"\nStudy catalog saved: {catalog_file}")

        # Print summary
        print("\n=== Metabolomics Workbench Search Results ===\n")
        print(catalog_df[['study_id', 'title', 'species', 'sample_type',
                         'relevance_score']].to_string(index=False, max_colwidth=50))

        print(f"\nTotal studies found: {len(relevant_studies)}")
        print(f"Human studies: {len([s for s in relevant_studies if 'human' in s.species.lower()])}")

        # Download all if requested
        if args.download_all:
            print("\n=== Downloading Studies ===\n")
            for study in tqdm(relevant_studies[:10], desc="Downloading"):  # Limit to top 10
                api.download_study_complete(study.study_id)
                time.sleep(1)

    # Handle specific study
    elif args.study:
        if args.summary:
            summary = api.get_study_summary(args.study)
            if summary:
                print(f"\n=== {args.study} Summary ===\n")
                for key, value in summary.items():
                    print(f"{key}: {value}")
            else:
                print(f"Study {args.study} not found")

        if args.download_data:
            downloaded = api.download_study_complete(args.study)
            print(f"\nDownloaded {len(downloaded)} files:")
            for data_type, path in downloaded.items():
                print(f"  {data_type}: {path}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()