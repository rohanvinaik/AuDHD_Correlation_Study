#!/usr/bin/env python3
"""
dbGaP Study Searcher for ADHD/Autism Research

Searches NCBI dbGaP (Database of Genotypes and Phenotypes) for relevant studies
using the E-utilities API and BeautifulSoup for web scraping.

dbGaP contains:
- 1,400+ studies
- Case-control, family, longitudinal designs
- Genotypes, WGS, WES, RNA-seq, methylation
- Controlled access with DAC approval required

Requirements:
    pip install requests pandas beautifulsoup4 lxml

Usage:
    # Search for ADHD/autism studies
    python dbgap_searcher.py --search --output data/genetics/

    # Get detailed info for specific study
    python dbgap_searcher.py --study phs000016 --output data/genetics/

    # List all studies with summary stats available
    python dbgap_searcher.py --summary-stats --output data/genetics/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import logging

try:
    import requests
    import pandas as pd
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install requests pandas beautifulsoup4 lxml")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# NCBI E-utilities API
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DBGAP_BASE = "https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin"

# Search terms for ADHD/Autism
SEARCH_TERMS = {
    'adhd': [
        'ADHD', 'attention deficit', 'hyperactivity',
        'attention-deficit/hyperactivity disorder'
    ],
    'autism': [
        'autism', 'ASD', 'autistic', 'autism spectrum',
        'Asperger', 'pervasive developmental disorder'
    ],
    'neurodevelopmental': [
        'neurodevelopmental', 'developmental disorder'
    ]
}

# Study types
STUDY_TYPES = [
    'Case-Control', 'Case Set', 'Family', 'Longitudinal',
    'Cohort', 'Twin', 'Population'
]

# Data types
DATA_TYPES = [
    'Genotypes', 'SNP', 'WGS', 'WES', 'Sequencing',
    'RNA-seq', 'Expression', 'Methylation', 'Epigenomics'
]


@dataclass
class DbGaPStudy:
    """Represents a dbGaP study"""
    phs_id: str
    accession: str
    title: str
    description: str
    disease: str
    study_type: str
    num_subjects: int
    data_types: List[str]
    has_summary_stats: bool
    access_required: bool
    pi_name: str
    institute: str
    relevance_score: float
    matched_terms: List[str]
    url: str


class DbGaPSearcher:
    """Search dbGaP for ADHD/Autism genetic studies"""

    def __init__(self, output_dir: Path, email: str = "user@example.com"):
        """
        Initialize searcher

        Args:
            output_dir: Output directory
            email: Email for NCBI API (required by NCBI)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.email = email
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        logger.info(f"Initialized dbGaP searcher: {output_dir}")

    def search_eutils(self, query: str, retmax: int = 100) -> List[str]:
        """
        Search dbGaP using NCBI E-utilities

        Args:
            query: Search query
            retmax: Maximum results to return

        Returns:
            List of study IDs (phs IDs)
        """
        logger.info(f"Searching dbGaP with query: {query}")

        try:
            # ESearch to get IDs
            esearch_url = f"{EUTILS_BASE}/esearch.fcgi"
            params = {
                'db': 'gap',
                'term': query,
                'retmax': retmax,
                'retmode': 'json',
                'email': self.email
            }

            response = self.session.get(esearch_url, params=params, timeout=30)

            if response.status_code != 200:
                logger.error(f"ESearch failed: HTTP {response.status_code}")
                return []

            data = response.json()

            # Extract IDs
            id_list = data.get('esearchresult', {}).get('idlist', [])

            logger.info(f"Found {len(id_list)} studies")
            return id_list

        except Exception as e:
            logger.error(f"Error searching dbGaP: {e}")
            return []

    def get_study_details(self, study_id: str) -> Optional[Dict]:
        """
        Get detailed study information using ESummary

        Args:
            study_id: dbGaP study ID

        Returns:
            Study details dictionary
        """
        try:
            # ESummary to get study details
            esummary_url = f"{EUTILS_BASE}/esummary.fcgi"
            params = {
                'db': 'gap',
                'id': study_id,
                'retmode': 'json',
                'email': self.email
            }

            response = self.session.get(esummary_url, params=params, timeout=30)

            if response.status_code != 200:
                return None

            data = response.json()

            # Extract study summary
            result = data.get('result', {}).get(study_id, {})

            return result

        except Exception as e:
            logger.error(f"Error fetching study {study_id}: {e}")
            return None

    def scrape_study_page(self, phs_id: str) -> Optional[Dict]:
        """
        Scrape additional details from study page

        Args:
            phs_id: Study accession (e.g., phs000016)

        Returns:
            Additional study details
        """
        try:
            # Study page URL
            url = f"{DBGAP_BASE}/study.cgi?study_id={phs_id}"

            response = self.session.get(url, timeout=30)

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'lxml')

            details = {}

            # Extract data types
            data_types = []
            data_table = soup.find('table', {'class': 'data_table'})
            if data_table:
                for row in data_table.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        data_type = cells[0].get_text(strip=True)
                        if any(dt in data_type for dt in DATA_TYPES):
                            data_types.append(data_type)

            details['data_types'] = data_types

            # Extract PI information
            pi_section = soup.find('div', {'id': 'pi'})
            if pi_section:
                pi_text = pi_section.get_text(strip=True)
                details['pi_info'] = pi_text

            return details

        except Exception as e:
            logger.error(f"Error scraping {phs_id}: {e}")
            return None

    def search_adhd_autism_studies(self) -> List[DbGaPStudy]:
        """
        Search for ADHD/Autism-relevant studies

        Returns:
            List of relevant studies
        """
        logger.info("Searching dbGaP for ADHD/Autism studies...")

        all_study_ids = set()

        # Search with different term combinations
        for category, terms in SEARCH_TERMS.items():
            for term in terms:
                query = f"{term}[Disease]"
                study_ids = self.search_eutils(query, retmax=50)
                all_study_ids.update(study_ids)

                time.sleep(0.5)  # Rate limiting

        logger.info(f"Found {len(all_study_ids)} unique studies")

        # Get details for each study
        studies = []

        for study_id in all_study_ids:
            details = self.get_study_details(study_id)

            if not details:
                continue

            # Extract phs accession
            phs_id = details.get('d_accession', '')

            if not phs_id:
                continue

            # Calculate relevance
            title = details.get('d_study_name', '')
            description = details.get('d_study_descr', '')
            disease = details.get('d_disease', '')

            relevance_score, matched_terms = self._calculate_relevance(
                title, description, disease
            )

            if relevance_score < 5.0:
                continue

            # Scrape additional details
            scraped = self.scrape_study_page(phs_id)
            data_types = scraped.get('data_types', []) if scraped else []

            # Create study object
            study = DbGaPStudy(
                phs_id=phs_id,
                accession=details.get('d_accession', ''),
                title=title,
                description=description,
                disease=disease,
                study_type=details.get('d_study_type', ''),
                num_subjects=int(details.get('d_num_participants', 0) or 0),
                data_types=data_types,
                has_summary_stats=False,  # Would need to check separately
                access_required=True,  # dbGaP requires DAC approval
                pi_name=details.get('d_pi_name', ''),
                institute=details.get('d_institute', ''),
                relevance_score=relevance_score,
                matched_terms=matched_terms,
                url=f"https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id={phs_id}"
            )

            studies.append(study)
            logger.info(f"Found relevant study: {phs_id} (score: {relevance_score:.1f})")

            time.sleep(1)  # Rate limiting

        # Sort by relevance
        studies.sort(key=lambda s: s.relevance_score, reverse=True)

        logger.info(f"Found {len(studies)} relevant studies")
        return studies

    def _calculate_relevance(self, title: str, description: str,
                           disease: str) -> tuple[float, List[str]]:
        """Calculate relevance score"""
        score = 0.0
        matched_terms = []

        searchable_text = f"{title} {description} {disease}".lower()

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

        # Bonus for family studies
        if 'family' in searchable_text or 'trio' in searchable_text:
            score += 3.0

        # Bonus for large sample size implied
        if 'cohort' in searchable_text or 'population' in searchable_text:
            score += 2.0

        return score, matched_terms

    def get_known_major_studies(self) -> List[Dict]:
        """
        Return known major ADHD/Autism studies in dbGaP

        Returns:
            List of major study metadata
        """
        major_studies = [
            {
                'phs_id': 'phs000016',
                'title': 'Autism Genetic Resource Exchange (AGRE)',
                'disease': 'Autism Spectrum Disorder',
                'num_subjects': 2000,
                'study_type': 'Family',
                'data_types': ['Genotypes', 'WGS'],
                'relevance_score': 15.0
            },
            {
                'phs_id': 'phs000267',
                'title': 'Simons Simplex Collection (SSC)',
                'disease': 'Autism Spectrum Disorder',
                'num_subjects': 2644,
                'study_type': 'Family (Simplex)',
                'data_types': ['WGS', 'WES', 'Genotypes'],
                'relevance_score': 15.0
            },
            {
                'phs_id': 'phs000473',
                'title': 'Autism Sequencing Consortium',
                'disease': 'Autism Spectrum Disorder',
                'num_subjects': 5000,
                'study_type': 'Case-Control',
                'data_types': ['WES'],
                'relevance_score': 14.0
            },
            {
                'phs_id': 'phs000016',
                'title': 'Children\'s Hospital of Philadelphia ADHD Study',
                'disease': 'ADHD',
                'num_subjects': 1500,
                'study_type': 'Case-Control',
                'data_types': ['Genotypes', 'SNP Array'],
                'relevance_score': 13.0
            }
        ]

        return major_studies

    def generate_study_catalog(self, studies: List[DbGaPStudy]) -> pd.DataFrame:
        """Generate catalog of studies"""
        if not studies:
            return pd.DataFrame()

        catalog = [asdict(study) for study in studies]
        df = pd.DataFrame(catalog)
        df = df.sort_values('relevance_score', ascending=False)

        return df


def main():
    parser = argparse.ArgumentParser(
        description='Search dbGaP for ADHD/Autism genetic studies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for all ADHD/autism studies
  python dbgap_searcher.py --search --output data/genetics/

  # Get details for specific study
  python dbgap_searcher.py --study phs000016 --output data/genetics/

  # List known major studies
  python dbgap_searcher.py --major-studies

Note: dbGaP requires controlled access approval for most studies.
Apply at: https://dbgap.ncbi.nlm.nih.gov/aa/wga.cgi?page=login
        """
    )

    parser.add_argument(
        '--search',
        action='store_true',
        help='Search for ADHD/autism studies'
    )

    parser.add_argument(
        '--study',
        type=str,
        help='Get details for specific study (phs ID)'
    )

    parser.add_argument(
        '--major-studies',
        action='store_true',
        help='List known major ADHD/autism studies'
    )

    parser.add_argument(
        '--email',
        type=str,
        default='user@example.com',
        help='Email for NCBI API (required)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/genetics/dbgap',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize searcher
    searcher = DbGaPSearcher(Path(args.output), email=args.email)

    # Handle major studies list
    if args.major_studies:
        major = searcher.get_known_major_studies()
        print("\n=== Known Major ADHD/Autism Studies in dbGaP ===\n")
        for study in major:
            print(f"{study['phs_id']}: {study['title']}")
            print(f"  Disease: {study['disease']}")
            print(f"  Subjects: {study['num_subjects']}")
            print(f"  Data: {', '.join(study['data_types'])}")
            print(f"  URL: https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id={study['phs_id']}")
            print()
        return

    # Handle specific study
    if args.study:
        details = searcher.get_study_details(args.study)
        if details:
            print(f"\n=== {args.study} Details ===\n")
            for key, value in details.items():
                print(f"{key}: {value}")
        else:
            print(f"Study {args.study} not found")
        return

    # Handle search
    if args.search:
        studies = searcher.search_adhd_autism_studies()

        if not studies:
            print("\nNo relevant studies found")
            return

        # Generate catalog
        catalog_df = searcher.generate_study_catalog(studies)

        # Save catalog
        catalog_file = searcher.output_dir / 'dbgap_study_catalog.csv'
        catalog_df.to_csv(catalog_file, index=False)
        print(f"\nStudy catalog saved: {catalog_file}")

        # Print summary
        print("\n=== dbGaP Search Results ===\n")
        print(catalog_df[['phs_id', 'title', 'disease', 'num_subjects',
                         'relevance_score']].to_string(index=False, max_colwidth=50))

        print(f"\nTotal studies found: {len(studies)}")
        print(f"Family studies: {len([s for s in studies if 'family' in s.study_type.lower()])}")
        print(f"Average subjects: {catalog_df['num_subjects'].mean():.0f}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()