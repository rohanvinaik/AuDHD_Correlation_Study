#!/usr/bin/env python3
"""
Supplementary Material Extractor

Extracts supplementary data files from PubMed Central full-text articles.
Identifies and downloads:
- Supplementary tables (Excel, CSV, TSV)
- Supplementary data files
- Code/software links
- GitHub repositories
- Data repository links

Requirements:
    pip install requests beautifulsoup4 lxml pandas

Usage:
    # Extract from papers JSON
    python supplement_extractor.py \
        --input data/literature/papers_with_data.json \
        --output data/literature/supplements/

    # Extract from specific PMC article
    python supplement_extractor.py \
        --pmcid PMC6402513 \
        --output data/literature/supplements/

    # Extract and download files
    python supplement_extractor.py \
        --input data/literature/papers_with_data.json \
        --download \
        --output data/literature/supplements/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install requests beautifulsoup4 lxml pandas")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# PMC base URLs
PMC_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/"
PMC_OAS_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# File extensions of interest
DATA_FILE_EXTENSIONS = [
    '.xlsx', '.xls', '.csv', '.tsv', '.txt',
    '.vcf', '.bed', '.bam', '.fastq', '.fasta',
    '.zip', '.tar.gz', '.gz',
    '.json', '.xml', '.h5', '.hdf5',
    '.RData', '.rds', '.Rdata'
]

# Supplementary file patterns
SUPP_FILE_PATTERNS = [
    r'supplementary[_\s]+(data|table|file|material|information)',
    r'additional[_\s]+(file|table|data)',
    r'S\d+[_\s]+(Table|Data|File)',
    r'table[_\s]+S\d+',
    r'dataset[_\s]+S\d+',
    r'appendix',
    r'supporting[_\s]+information'
]

# External data repository patterns
EXTERNAL_REPO_PATTERNS = {
    'figshare': r'https?://(?:www\.)?figshare\.com/(?:articles|s)/[\w/]+/\d+',
    'zenodo': r'https?://(?:www\.)?zenodo\.org/record/\d+',
    'dryad': r'https?://(?:www\.)?datadryad\.org/\w+/\d+',
    'osf': r'https?://(?:www\.)?osf\.io/[\w/]+',
    'github': r'https?://(?:www\.)?github\.com/[\w-]+/[\w-]+',
    'gitlab': r'https?://(?:www\.)?gitlab\.com/[\w-]+/[\w-]+',
    'bitbucket': r'https?://(?:www\.)?bitbucket\.org/[\w-]+/[\w-]+',
    'geo': r'https?://www\.ncbi\.nlm\.nih\.gov/geo/query/acc\.cgi\?acc=GSE\d+',
    'sra': r'https?://www\.ncbi\.nlm\.nih\.gov/(?:bioproject|sra)/(?:PRJNA|SRP)\d+',
    'arrayexpress': r'https?://www\.ebi\.ac\.uk/arrayexpress/experiments/E-[\w]+-\d+',
    'ega': r'https?://ega-archive\.org/studies/EGAS\d+'
}


class SupplementExtractor:
    """Extract supplementary materials from PubMed Central articles"""

    def __init__(self, output_dir: Path):
        """
        Initialize extractor

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        logger.info(f"Initialized supplement extractor: {output_dir}")

    def extract_from_pmc(self, pmcid: str) -> Dict:
        """
        Extract supplementary materials from PMC article

        Args:
            pmcid: PubMed Central ID (with or without PMC prefix)

        Returns:
            Dictionary with extracted information
        """
        # Ensure PMC prefix
        if not pmcid.startswith('PMC'):
            pmcid = f'PMC{pmcid}'

        logger.info(f"Extracting supplements from {pmcid}...")

        result = {
            'pmcid': pmcid,
            'url': f"{PMC_BASE_URL}{pmcid}/",
            'supplementary_files': [],
            'external_repos': {},
            'github_repos': [],
            'data_availability_statement': '',
            'code_availability_statement': '',
            'full_text_extracted': False
        }

        try:
            # Fetch PMC article page
            response = self.session.get(result['url'], timeout=30)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {pmcid}: HTTP {response.status_code}")
                return result

            soup = BeautifulSoup(response.content, 'lxml')

            # Extract supplementary files
            result['supplementary_files'] = self._extract_supplementary_files(soup, pmcid)

            # Extract external repositories
            result['external_repos'] = self._extract_external_repos(soup)

            # Extract GitHub repos
            result['github_repos'] = result['external_repos'].get('github', [])

            # Extract data availability statement
            result['data_availability_statement'] = self._extract_data_availability(soup)

            # Extract code availability statement
            result['code_availability_statement'] = self._extract_code_availability(soup)

            result['full_text_extracted'] = True
            logger.info(f"Extracted {len(result['supplementary_files'])} supplementary files from {pmcid}")

        except Exception as e:
            logger.error(f"Error extracting from {pmcid}: {e}")

        return result

    def _extract_supplementary_files(self, soup: BeautifulSoup, pmcid: str) -> List[Dict]:
        """Extract supplementary file links"""
        supp_files = []

        # Look for supplementary material sections
        supp_sections = soup.find_all(['div', 'section'], class_=re.compile(r'suppl|supplement|additional', re.I))

        for section in supp_sections:
            # Find all links in supplementary sections
            for link in section.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)

                # Check if it's a data file
                if any(ext in href.lower() for ext in DATA_FILE_EXTENSIONS):
                    # Make absolute URL
                    if not href.startswith('http'):
                        href = urljoin(PMC_BASE_URL, href)

                    supp_files.append({
                        'url': href,
                        'text': text,
                        'filename': href.split('/')[-1],
                        'type': self._guess_file_type(href)
                    })

        # Also look for direct supplementary file links
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)

            # Check if link text matches supplementary patterns
            if any(re.search(pattern, text, re.I) for pattern in SUPP_FILE_PATTERNS):
                if any(ext in href.lower() for ext in DATA_FILE_EXTENSIONS):
                    if not href.startswith('http'):
                        href = urljoin(PMC_BASE_URL, href)

                    # Avoid duplicates
                    if not any(s['url'] == href for s in supp_files):
                        supp_files.append({
                            'url': href,
                            'text': text,
                            'filename': href.split('/')[-1],
                            'type': self._guess_file_type(href)
                        })

        return supp_files

    def _extract_external_repos(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract external repository links"""
        repos = {}

        # Get all text and links
        page_text = soup.get_text()
        page_html = str(soup)

        for repo_type, pattern in EXTERNAL_REPO_PATTERNS.items():
            matches = re.findall(pattern, page_html, re.IGNORECASE)
            if matches:
                repos[repo_type] = list(set(matches))

        return repos

    def _extract_data_availability(self, soup: BeautifulSoup) -> str:
        """Extract data availability statement"""
        # Look for data availability section
        data_avail_sections = soup.find_all(['div', 'section', 'p'],
                                           text=re.compile(r'data\s+availability', re.I))

        for section in data_avail_sections:
            # Get the content
            if section.name == 'p':
                return section.get_text(strip=True)
            else:
                # Get next sibling paragraphs
                text_parts = []
                for sibling in section.find_next_siblings(['p'], limit=3):
                    text_parts.append(sibling.get_text(strip=True))
                return ' '.join(text_parts)

        return ''

    def _extract_code_availability(self, soup: BeautifulSoup) -> str:
        """Extract code availability statement"""
        # Look for code/software availability
        code_sections = soup.find_all(['div', 'section', 'p'],
                                     text=re.compile(r'code\s+availability|software\s+availability', re.I))

        for section in code_sections:
            if section.name == 'p':
                return section.get_text(strip=True)
            else:
                text_parts = []
                for sibling in section.find_next_siblings(['p'], limit=3):
                    text_parts.append(sibling.get_text(strip=True))
                return ' '.join(text_parts)

        return ''

    def _guess_file_type(self, filename: str) -> str:
        """Guess file type from extension"""
        filename_lower = filename.lower()

        if any(ext in filename_lower for ext in ['.xlsx', '.xls']):
            return 'excel'
        elif any(ext in filename_lower for ext in ['.csv', '.tsv', '.txt']):
            return 'tabular'
        elif any(ext in filename_lower for ext in ['.vcf', '.bed']):
            return 'genomics'
        elif any(ext in filename_lower for ext in ['.bam', '.fastq', '.fasta']):
            return 'sequencing'
        elif any(ext in filename_lower for ext in ['.json', '.xml']):
            return 'structured'
        elif any(ext in filename_lower for ext in ['.zip', '.tar.gz', '.gz']):
            return 'archive'
        elif any(ext in filename_lower for ext in ['.RData', '.rds']):
            return 'r_data'
        else:
            return 'unknown'

    def download_file(self, url: str, output_dir: Path) -> Optional[Path]:
        """
        Download supplementary file

        Args:
            url: File URL
            output_dir: Output directory

        Returns:
            Path to downloaded file
        """
        try:
            filename = url.split('/')[-1].split('?')[0]
            output_file = output_dir / filename

            logger.info(f"Downloading: {filename}")

            response = self.session.get(url, stream=True, timeout=60)

            if response.status_code != 200:
                logger.error(f"Failed to download: HTTP {response.status_code}")
                return None

            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None

    def process_papers_json(self, json_file: Path,
                           download: bool = False) -> List[Dict]:
        """
        Process papers from JSON file

        Args:
            json_file: Path to papers JSON
            download: Whether to download files

        Returns:
            List of extraction results
        """
        logger.info(f"Processing papers from: {json_file}")

        with open(json_file) as f:
            papers = json.load(f)

        results = []

        for paper in papers:
            pmcid = paper.get('pmcid')

            if not pmcid:
                continue

            # Extract supplements
            result = self.extract_from_pmc(pmcid)

            # Download files if requested
            if download and result['supplementary_files']:
                download_dir = self.output_dir / pmcid
                download_dir.mkdir(exist_ok=True)

                for supp_file in result['supplementary_files']:
                    downloaded = self.download_file(supp_file['url'], download_dir)
                    if downloaded:
                        supp_file['local_path'] = str(downloaded)

            results.append(result)

            time.sleep(1)  # Rate limiting

        logger.info(f"Processed {len(results)} papers")
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract supplementary materials from PubMed Central articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from papers JSON
  python supplement_extractor.py \\
      --input data/literature/papers_with_data.json \\
      --output data/literature/supplements/

  # Extract from specific PMC article
  python supplement_extractor.py \\
      --pmcid PMC6402513 \\
      --output data/literature/supplements/

  # Extract and download files
  python supplement_extractor.py \\
      --input data/literature/papers_with_data.json \\
      --download \\
      --output data/literature/supplements/

  # Extract from multiple PMCIDs
  python supplement_extractor.py \\
      --pmcids PMC6402513,PMC6875795 \\
      --output data/literature/supplements/
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Input papers JSON file'
    )

    parser.add_argument(
        '--pmcid',
        type=str,
        help='Single PMC ID to extract'
    )

    parser.add_argument(
        '--pmcids',
        type=str,
        help='Comma-separated PMC IDs'
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Download supplementary files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/literature/supplements',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = SupplementExtractor(Path(args.output))

    results = []

    # Handle single PMCID
    if args.pmcid:
        result = extractor.extract_from_pmc(args.pmcid)
        results = [result]

        if args.download and result['supplementary_files']:
            download_dir = extractor.output_dir / args.pmcid
            download_dir.mkdir(exist_ok=True)

            for supp_file in result['supplementary_files']:
                extractor.download_file(supp_file['url'], download_dir)

    # Handle multiple PMCIDs
    elif args.pmcids:
        pmcids = args.pmcids.split(',')

        for pmcid in pmcids:
            result = extractor.extract_from_pmc(pmcid.strip())
            results.append(result)

            if args.download and result['supplementary_files']:
                download_dir = extractor.output_dir / pmcid
                download_dir.mkdir(exist_ok=True)

                for supp_file in result['supplementary_files']:
                    extractor.download_file(supp_file['url'], download_dir)

            time.sleep(1)

    # Handle JSON input
    elif args.input:
        results = extractor.process_papers_json(
            Path(args.input),
            download=args.download
        )

    else:
        parser.print_help()
        return

    if not results:
        print("\nNo results")
        return

    # Export results
    output_file = extractor.output_dir / 'supplementary_materials.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n=== Supplementary Material Extraction Summary ===\n")
    print(f"Total articles processed: {len(results)}")
    print(f"Articles with supplements: {sum(1 for r in results if r['supplementary_files'])}")

    total_files = sum(len(r['supplementary_files']) for r in results)
    print(f"Total supplementary files found: {total_files}")

    # File type breakdown
    file_types = {}
    for result in results:
        for supp_file in result['supplementary_files']:
            ftype = supp_file['type']
            file_types[ftype] = file_types.get(ftype, 0) + 1

    if file_types:
        print(f"\nFile type breakdown:")
        for ftype, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ftype}: {count}")

    # Repository breakdown
    repo_counts = {}
    for result in results:
        for repo_type in result['external_repos'].keys():
            repo_counts[repo_type] = repo_counts.get(repo_type, 0) + 1

    if repo_counts:
        print(f"\nExternal repositories:")
        for repo, count in sorted(repo_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {repo}: {count}")

    print(f"\nArticles with GitHub repos: {sum(1 for r in results if r['github_repos'])}")
    print(f"Articles with data availability statements: {sum(1 for r in results if r['data_availability_statement'])}")

    print(f"\nResults saved: {output_file}")


if __name__ == '__main__':
    main()