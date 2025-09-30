#!/usr/bin/env python3
"""
Extract data repository URLs from papers and download raw data
Focuses on GitHub, Zenodo, Dryad, figshare, OSF, etc.
"""

import re
import requests
from pathlib import Path
import logging
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataRepositoryExtractor:
    def __init__(self, output_dir='data/papers'):
        self.output_dir = Path(output_dir)
        self.repos_dir = self.output_dir / 'repositories'
        self.repos_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Academic Research Bot)'
        })

        # Data repository patterns
        self.repo_patterns = {
            'github': re.compile(r'github\.com/[\w\-]+/[\w\-]+', re.IGNORECASE),
            'zenodo': re.compile(r'zenodo\.org/record/\d+', re.IGNORECASE),
            'dryad': re.compile(r'datadryad\.org/stash/dataset/doi:[^\s<>"\']+', re.IGNORECASE),
            'figshare': re.compile(r'figshare\.com/articles/[^\s<>"\']+', re.IGNORECASE),
            'osf': re.compile(r'osf\.io/[\w]+', re.IGNORECASE),
            'mendeley': re.compile(r'data\.mendeley\.com/datasets/[\w]+', re.IGNORECASE),
            'ncbi_geo': re.compile(r'GSE\d+', re.IGNORECASE),
            'ncbi_sra': re.compile(r'SRP\d+|PRJNA\d+|PRJEB\d+', re.IGNORECASE),
        }

        # Track what we've found
        self.found_repos = []

    def get_pmc_fulltext_xml(self, pmc_id):
        """Download full-text XML from PMC"""
        url = f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
        params = {
            'verb': 'GetRecord',
            'identifier': f'oai:pubmedcentral.nih.gov:{pmc_id}',
            'metadataPrefix': 'pmc'
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.warning(f"Failed to get XML for PMC{pmc_id}: {e}")

        return None

    def extract_data_availability(self, xml_content):
        """Extract data availability statements from XML"""
        soup = BeautifulSoup(xml_content, 'xml')

        statements = []

        # Look for data availability sections
        for section in soup.find_all(['sec', 'supplementary-material', 'back']):
            title = section.find('title')
            if title:
                title_text = title.get_text().lower()
                if any(keyword in title_text for keyword in [
                    'data availability', 'data access', 'availability of data',
                    'code availability', 'accession', 'supplementary'
                ]):
                    statements.append(section.get_text())

        # Also check refs for data citations
        for ref in soup.find_all('ext-link'):
            href = ref.get('xlink:href', '')
            if any(domain in href.lower() for domain in [
                'github.com', 'zenodo.org', 'dryad.org', 'figshare.com',
                'osf.io', 'mendeley.com'
            ]):
                statements.append(href)

        return statements

    def extract_repos_from_text(self, text):
        """Extract repository URLs from text"""
        repos = {}

        for repo_type, pattern in self.repo_patterns.items():
            matches = pattern.findall(text)
            if matches:
                repos[repo_type] = list(set(matches))  # Deduplicate

        return repos

    def download_from_github(self, github_url):
        """Download data from GitHub repo"""
        logger.info(f"Found GitHub repo: {github_url}")

        # Convert to raw download URL if it's a specific file
        if 'blob' in github_url:
            github_url = github_url.replace('blob', 'raw')

        # Try to download as zip
        if not github_url.startswith('http'):
            github_url = f"https://{github_url}"

        # Get the archive
        if '/tree/' not in github_url and '/blob/' not in github_url:
            archive_url = f"{github_url}/archive/refs/heads/main.zip"
            repo_name = github_url.split('/')[-1]

            try:
                response = self.session.get(archive_url, timeout=60)
                if response.status_code == 200:
                    output_file = self.repos_dir / f"{repo_name}.zip"
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"✓ Downloaded {repo_name}.zip ({len(response.content)/1024/1024:.1f} MB)")
                    return output_file
                else:
                    # Try master branch
                    archive_url = f"{github_url}/archive/refs/heads/master.zip"
                    response = self.session.get(archive_url, timeout=60)
                    if response.status_code == 200:
                        output_file = self.repos_dir / f"{repo_name}.zip"
                        with open(output_file, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"✓ Downloaded {repo_name}.zip ({len(response.content)/1024/1024:.1f} MB)")
                        return output_file
            except Exception as e:
                logger.warning(f"Failed to download from GitHub: {e}")

        return None

    def download_from_zenodo(self, zenodo_url):
        """Download data from Zenodo"""
        logger.info(f"Found Zenodo record: {zenodo_url}")

        if not zenodo_url.startswith('http'):
            zenodo_url = f"https://{zenodo_url}"

        # Extract record ID
        record_id = re.search(r'record/(\d+)', zenodo_url)
        if not record_id:
            return None

        record_id = record_id.group(1)

        # Get record metadata via API
        api_url = f"https://zenodo.org/api/records/{record_id}"

        try:
            response = self.session.get(api_url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                files = data.get('files', [])

                downloaded = []
                for file_info in files[:10]:  # Limit to 10 files per record
                    file_url = file_info.get('links', {}).get('self')
                    filename = file_info.get('key', 'unknown')

                    if file_url:
                        output_file = self.repos_dir / f"zenodo_{record_id}_{filename}"

                        try:
                            file_response = self.session.get(file_url, timeout=120)
                            with open(output_file, 'wb') as f:
                                f.write(file_response.content)

                            size_mb = len(file_response.content) / 1024 / 1024
                            logger.info(f"✓ Downloaded {filename} ({size_mb:.1f} MB)")
                            downloaded.append(output_file)

                        except Exception as e:
                            logger.warning(f"Failed to download {filename}: {e}")

                        time.sleep(1)

                return downloaded

        except Exception as e:
            logger.warning(f"Failed to access Zenodo: {e}")

        return None

    def process_pmc_article(self, pmc_id):
        """Process a single PMC article to find and download data"""
        logger.info(f"Processing PMC{pmc_id}...")

        # Get full text
        xml_content = self.get_pmc_fulltext_xml(pmc_id)
        if not xml_content:
            return None

        # Extract data availability sections
        statements = self.extract_data_availability(xml_content)

        if not statements:
            return None

        # Extract repository URLs
        all_text = ' '.join(statements)
        repos = self.extract_repos_from_text(all_text)

        if not repos:
            return None

        logger.info(f"Found repositories in PMC{pmc_id}: {repos}")

        # Download from each repository
        downloaded_files = []

        for github_match in repos.get('github', []):
            file = self.download_from_github(github_match)
            if file:
                downloaded_files.append(file)
            time.sleep(2)

        for zenodo_match in repos.get('zenodo', []):
            files = self.download_from_zenodo(zenodo_match)
            if files:
                downloaded_files.extend(files)
            time.sleep(2)

        # Save metadata
        if downloaded_files:
            metadata = {
                'pmc_id': pmc_id,
                'repositories': repos,
                'files': [str(f) for f in downloaded_files]
            }

            metadata_file = self.repos_dir / f"PMC{pmc_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.found_repos.append(metadata)

        return downloaded_files


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract data repositories from papers')
    parser.add_argument('--pmc-ids', nargs='+', help='PMC IDs to process')
    parser.add_argument('--from-log', help='Extract PMC IDs from scraping log')
    parser.add_argument('--output', default='data/papers', help='Output directory')

    args = parser.parse_args()

    extractor = DataRepositoryExtractor(output_dir=args.output)

    pmc_ids = []

    if args.from_log:
        # Extract PMC IDs from log file
        with open(args.from_log) as f:
            log_content = f.read()
            # Find all PMC IDs
            pmc_matches = re.findall(r'PMC(\d+)', log_content)
            pmc_ids = list(set(pmc_matches))

        logger.info(f"Found {len(pmc_ids)} unique PMC IDs in log")

    elif args.pmc_ids:
        pmc_ids = [pid.replace('PMC', '') for pid in args.pmc_ids]

    else:
        logger.error("Must provide --pmc-ids or --from-log")
        return

    # Process each PMC ID
    total_files = 0
    for pmc_id in pmc_ids[:50]:  # Limit to first 50
        files = extractor.process_pmc_article(pmc_id)
        if files:
            total_files += len(files)
        time.sleep(3)  # Be nice to servers

    logger.info(f"\n✓ Complete! Downloaded {total_files} files from {len(extractor.found_repos)} papers")

    # Save summary
    summary_file = Path(args.output) / 'repositories' / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(extractor.found_repos, f, indent=2)

    logger.info(f"Summary saved to {summary_file}")


if __name__ == '__main__':
    main()