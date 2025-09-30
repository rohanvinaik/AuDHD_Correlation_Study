#!/usr/bin/env python3
"""
MetaboLights Scraper for ADHD/Autism Studies

Searches MetaboLights database for metabolomics studies relevant to ADHD/Autism
research. Downloads study metadata, sample information, and metabolite data.

MetaboLights is the world's largest open-access metabolomics repository:
- >1,000 studies
- Multiple platforms: NMR, LC-MS, GC-MS
- Human and animal studies
- Raw data + processed results

Requirements:
    pip install requests pandas tqdm

Usage:
    # Search for ADHD/autism studies
    python metabolights_scraper.py --search

    # Download specific study
    python metabolights_scraper.py --study MTBLS1234 --output data/metabolomics/

    # Download all ADHD/autism studies
    python metabolights_scraper.py --download-all --output data/metabolomics/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging
import re

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


# MetaboLights API endpoints
METABOLIGHTS_API_BASE = "https://www.ebi.ac.uk/metabolights/ws"
METABOLIGHTS_FTP = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public"

# Search terms for ADHD/Autism studies
SEARCH_TERMS = {
    'adhd': ['ADHD', 'attention deficit', 'hyperactivity', 'hyperkinetic'],
    'autism': ['autism', 'ASD', 'autistic', 'Asperger', 'pervasive developmental'],
    'neurodevelopmental': ['neurodevelopmental', 'developmental disorder'],
    'related': ['GABA', 'glutamate', 'dopamine', 'serotonin', 'methylation']
}

# Metabolomics platforms
PLATFORMS = ['NMR', 'LC-MS', 'GC-MS', 'CE-MS', 'UHPLC-MS']

# Sample types of interest
SAMPLE_TYPES = ['plasma', 'serum', 'blood', 'urine', 'CSF', 'saliva', 'brain']


@dataclass
class MetaboLightsStudy:
    """Represents a MetaboLights study"""
    study_id: str
    title: str
    description: str
    organism: str
    platform: str
    sample_types: List[str]
    num_samples: int
    release_date: str
    relevance_score: float
    matched_terms: List[str]
    url: str
    ftp_url: str


class MetaboLightsScraper:
    """Scrape MetaboLights for ADHD/Autism metabolomics studies"""

    def __init__(self, output_dir: Path):
        """
        Initialize scraper

        Args:
            output_dir: Output directory for downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        # Cache for API responses
        self.cache_dir = self.output_dir / '.cache'
        self.cache_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized MetaboLights scraper: {output_dir}")

    def search_all_studies(self) -> List[Dict]:
        """
        Get list of all public MetaboLights studies

        Returns:
            List of study metadata dictionaries
        """
        logger.info("Fetching all public studies from MetaboLights...")

        try:
            # Get all studies
            url = f"{METABOLIGHTS_API_BASE}/studies"
            response = self.session.get(url, timeout=30)

            if response.status_code != 200:
                logger.error(f"Failed to fetch studies: HTTP {response.status_code}")
                return []

            data = response.json()

            # Extract study list
            if 'content' in data:
                studies = data['content']
            else:
                studies = data

            logger.info(f"Found {len(studies)} public studies")
            return studies

        except Exception as e:
            logger.error(f"Error fetching studies: {e}")
            return []

    def get_study_details(self, study_id: str) -> Optional[Dict]:
        """
        Get detailed metadata for a specific study

        Args:
            study_id: Study ID (e.g., 'MTBLS1234')

        Returns:
            Study metadata dictionary
        """
        # Check cache
        cache_file = self.cache_dir / f"{study_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except:
                pass

        try:
            url = f"{METABOLIGHTS_API_BASE}/studies/{study_id}"
            response = self.session.get(url, timeout=30)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {study_id}: HTTP {response.status_code}")
                return None

            data = response.json()

            # Cache response
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            return data

        except Exception as e:
            logger.error(f"Error fetching {study_id}: {e}")
            return None

    def calculate_relevance_score(self, study_data: Dict) -> tuple[float, List[str]]:
        """
        Calculate relevance score for ADHD/Autism research

        Args:
            study_data: Study metadata

        Returns:
            Tuple of (relevance_score, matched_terms)
        """
        score = 0.0
        matched_terms = []

        # Extract searchable text
        title = study_data.get('title', '').lower()
        description = study_data.get('description', '').lower()
        factors = ' '.join([f.get('factorName', '') for f in study_data.get('factors', [])]).lower()

        searchable_text = f"{title} {description} {factors}"

        # Check ADHD terms (highest weight)
        for term in SEARCH_TERMS['adhd']:
            if term.lower() in searchable_text:
                score += 10.0
                matched_terms.append(term)

        # Check autism terms (highest weight)
        for term in SEARCH_TERMS['autism']:
            if term.lower() in searchable_text:
                score += 10.0
                matched_terms.append(term)

        # Check neurodevelopmental terms (medium weight)
        for term in SEARCH_TERMS['neurodevelopmental']:
            if term.lower() in searchable_text:
                score += 5.0
                matched_terms.append(term)

        # Check related terms (low weight)
        for term in SEARCH_TERMS['related']:
            if term.lower() in searchable_text:
                score += 2.0
                matched_terms.append(term)

        # Bonus for human studies
        organism = study_data.get('organism', {}).get('organismName', '').lower()
        if 'homo sapiens' in organism or 'human' in organism:
            score += 5.0

        # Bonus for relevant sample types
        sample_types = self._extract_sample_types(study_data)
        for sample_type in SAMPLE_TYPES:
            if any(sample_type in s.lower() for s in sample_types):
                score += 2.0

        return score, matched_terms

    def _extract_sample_types(self, study_data: Dict) -> List[str]:
        """Extract sample types from study metadata"""
        sample_types = set()

        # From materials
        for material in study_data.get('materials', {}).get('samples', []):
            characteristics = material.get('characteristics', [])
            for char in characteristics:
                if 'organism part' in char.get('category', {}).get('characteristicType', '').lower():
                    sample_types.add(char.get('value', {}).get('annotationValue', ''))

        return list(sample_types)

    def _extract_platform(self, study_data: Dict) -> str:
        """Extract metabolomics platform from study metadata"""
        # Check assays
        assays = study_data.get('assays', [])
        for assay in assays:
            measurement_type = assay.get('measurementType', {}).get('annotationValue', '')
            tech_type = assay.get('technology', {}).get('annotationValue', '')

            # Combine measurement and technology
            combined = f"{measurement_type} {tech_type}".upper()

            for platform in PLATFORMS:
                if platform in combined:
                    return platform

        return 'Unknown'

    def search_adhd_autism_studies(self, min_relevance: float = 2.0) -> List[MetaboLightsStudy]:
        """
        Search for ADHD/Autism-relevant studies

        Args:
            min_relevance: Minimum relevance score threshold

        Returns:
            List of relevant studies
        """
        logger.info("Searching for ADHD/Autism-relevant studies...")

        all_studies = self.search_all_studies()

        if not all_studies:
            return []

        relevant_studies = []

        for study_summary in tqdm(all_studies, desc="Screening studies"):
            study_id = study_summary.get('accession') or study_summary.get('studyIdentifier')

            if not study_id:
                continue

            # Get detailed metadata
            study_data = self.get_study_details(study_id)

            if not study_data:
                continue

            # Calculate relevance
            relevance_score, matched_terms = self.calculate_relevance_score(study_data)

            if relevance_score < min_relevance:
                continue

            # Extract study information
            sample_types = self._extract_sample_types(study_data)
            platform = self._extract_platform(study_data)

            study = MetaboLightsStudy(
                study_id=study_id,
                title=study_data.get('title', ''),
                description=study_data.get('description', ''),
                organism=study_data.get('organism', {}).get('organismName', ''),
                platform=platform,
                sample_types=sample_types,
                num_samples=len(study_data.get('materials', {}).get('samples', [])),
                release_date=study_data.get('releaseDate', ''),
                relevance_score=relevance_score,
                matched_terms=matched_terms,
                url=f"https://www.ebi.ac.uk/metabolights/{study_id}",
                ftp_url=f"{METABOLIGHTS_FTP}/{study_id}"
            )

            relevant_studies.append(study)

            logger.info(f"Found relevant study: {study_id} (score: {relevance_score:.1f})")

            # Rate limiting
            time.sleep(0.5)

        # Sort by relevance
        relevant_studies.sort(key=lambda s: s.relevance_score, reverse=True)

        logger.info(f"Found {len(relevant_studies)} relevant studies")
        return relevant_studies

    def download_study_metadata(self, study_id: str) -> Optional[Dict]:
        """
        Download complete study metadata

        Args:
            study_id: Study ID

        Returns:
            Study metadata dictionary
        """
        logger.info(f"Downloading metadata for {study_id}...")

        study_data = self.get_study_details(study_id)

        if not study_data:
            return None

        # Save to file
        output_file = self.output_dir / f"{study_id}_metadata.json"
        with open(output_file, 'w') as f:
            json.dump(study_data, f, indent=2)

        logger.info(f"Saved metadata: {output_file}")
        return study_data

    def download_study_files(self, study_id: str, file_types: List[str] = None) -> List[Path]:
        """
        Download study data files

        Args:
            study_id: Study ID
            file_types: File types to download (e.g., ['s_', 'a_', 'm_'])
                       s_ = sample files, a_ = assay files, m_ = metabolite assignments

        Returns:
            List of downloaded file paths
        """
        if file_types is None:
            file_types = ['s_', 'a_', 'm_']

        logger.info(f"Downloading files for {study_id}...")

        study_dir = self.output_dir / study_id
        study_dir.mkdir(exist_ok=True)

        downloaded_files = []

        try:
            # Get file list from FTP
            ftp_url = f"{METABOLIGHTS_FTP}/{study_id}"

            # Download investigation file (i_Investigation.txt)
            inv_file = f"{ftp_url}/i_Investigation.txt"
            inv_response = self.session.get(inv_file, timeout=60)

            if inv_response.status_code == 200:
                inv_path = study_dir / "i_Investigation.txt"
                with open(inv_path, 'wb') as f:
                    f.write(inv_response.content)
                downloaded_files.append(inv_path)
                logger.info(f"Downloaded: i_Investigation.txt")

            # Download sample/assay/metabolite files
            # Try common filenames
            common_files = [
                's_Sample.txt',
                'a_Assay.txt',
                'm_metabolite_profiling.txt',
                'm_metabolite_profiling_mass_spectrometry.txt',
                'm_metabolite_profiling_NMR_spectroscopy.txt'
            ]

            for filename in common_files:
                if not any(filename.startswith(ft) for ft in file_types):
                    continue

                file_url = f"{ftp_url}/{filename}"
                response = self.session.get(file_url, timeout=60)

                if response.status_code == 200:
                    file_path = study_dir / filename
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    downloaded_files.append(file_path)
                    logger.info(f"Downloaded: {filename}")

        except Exception as e:
            logger.error(f"Error downloading files for {study_id}: {e}")

        logger.info(f"Downloaded {len(downloaded_files)} files for {study_id}")
        return downloaded_files

    def parse_metabolite_data(self, study_id: str) -> Optional[pd.DataFrame]:
        """
        Parse metabolite assignments/data from downloaded files

        Args:
            study_id: Study ID

        Returns:
            DataFrame with metabolite data
        """
        study_dir = self.output_dir / study_id

        if not study_dir.exists():
            logger.warning(f"Study directory not found: {study_dir}")
            return None

        # Look for metabolite assignment files
        metabolite_files = list(study_dir.glob('m_*.txt')) + list(study_dir.glob('m_*.tsv'))

        if not metabolite_files:
            logger.warning(f"No metabolite files found for {study_id}")
            return None

        # Try to parse first metabolite file
        metabolite_file = metabolite_files[0]

        try:
            # MetaboLights files are tab-separated
            df = pd.read_csv(metabolite_file, sep='\t', low_memory=False)

            logger.info(f"Parsed metabolite data: {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error parsing {metabolite_file}: {e}")
            return None

    def generate_study_catalog(self, studies: List[MetaboLightsStudy]) -> pd.DataFrame:
        """
        Generate catalog of studies

        Args:
            studies: List of MetaboLightsStudy objects

        Returns:
            DataFrame with study catalog
        """
        if not studies:
            return pd.DataFrame()

        catalog = []
        for study in studies:
            catalog.append({
                'study_id': study.study_id,
                'title': study.title,
                'organism': study.organism,
                'platform': study.platform,
                'sample_types': ', '.join(study.sample_types),
                'num_samples': study.num_samples,
                'release_date': study.release_date,
                'relevance_score': study.relevance_score,
                'matched_terms': ', '.join(study.matched_terms),
                'url': study.url,
                'ftp_url': study.ftp_url
            })

        df = pd.DataFrame(catalog)
        df = df.sort_values('relevance_score', ascending=False)

        return df


def main():
    parser = argparse.ArgumentParser(
        description='Scrape MetaboLights for ADHD/Autism metabolomics studies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for ADHD/autism studies
  python metabolights_scraper.py --search --output data/metabolomics/

  # Download specific study
  python metabolights_scraper.py --study MTBLS1234 --output data/metabolomics/

  # Download all relevant studies
  python metabolights_scraper.py --download-all --output data/metabolomics/

  # Search with custom relevance threshold
  python metabolights_scraper.py --search --min-relevance 5.0

Note: MetaboLights API may be slow. Be patient.
        """
    )

    parser.add_argument(
        '--search',
        action='store_true',
        help='Search for ADHD/Autism-relevant studies'
    )

    parser.add_argument(
        '--study',
        type=str,
        help='Download specific study by ID (e.g., MTBLS1234)'
    )

    parser.add_argument(
        '--download-all',
        action='store_true',
        help='Download all relevant studies found in search'
    )

    parser.add_argument(
        '--min-relevance',
        type=float,
        default=2.0,
        help='Minimum relevance score (default: 2.0)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/metabolomics/metabolights',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize scraper
    scraper = MetaboLightsScraper(Path(args.output))

    # Handle search
    if args.search or args.download_all:
        relevant_studies = scraper.search_adhd_autism_studies(
            min_relevance=args.min_relevance
        )

        if not relevant_studies:
            print("\nNo relevant studies found")
            return

        # Generate catalog
        catalog_df = scraper.generate_study_catalog(relevant_studies)

        # Save catalog
        catalog_file = scraper.output_dir / 'metabolights_study_catalog.csv'
        catalog_df.to_csv(catalog_file, index=False)
        print(f"\nStudy catalog saved: {catalog_file}")

        # Print summary
        print("\n=== MetaboLights Search Results ===\n")
        print(catalog_df[['study_id', 'title', 'platform', 'num_samples',
                         'relevance_score']].to_string(index=False))

        print(f"\nTotal studies found: {len(relevant_studies)}")
        print(f"Human studies: {len([s for s in relevant_studies if 'sapiens' in s.organism.lower()])}")
        print(f"NMR studies: {len([s for s in relevant_studies if s.platform == 'NMR'])}")
        print(f"LC-MS studies: {len([s for s in relevant_studies if s.platform == 'LC-MS'])}")

        # Download all if requested
        if args.download_all:
            print("\n=== Downloading Studies ===\n")
            for study in tqdm(relevant_studies[:10], desc="Downloading studies"):  # Limit to top 10
                scraper.download_study_metadata(study.study_id)
                scraper.download_study_files(study.study_id)
                time.sleep(2)  # Rate limiting

    # Handle specific study download
    elif args.study:
        scraper.download_study_metadata(args.study)
        downloaded_files = scraper.download_study_files(args.study)

        if downloaded_files:
            print(f"\nDownloaded {len(downloaded_files)} files for {args.study}")

            # Try to parse metabolite data
            metabolite_df = scraper.parse_metabolite_data(args.study)
            if metabolite_df is not None:
                print(f"\nMetabolite data: {len(metabolite_df)} rows")
                print(f"Columns: {list(metabolite_df.columns[:10])}")
        else:
            print(f"\nNo files downloaded for {args.study}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()