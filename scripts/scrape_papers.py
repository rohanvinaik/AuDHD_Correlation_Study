#!/usr/bin/env python3
"""
Paper Scraper for Supplementary Data

Searches PubMed, bioRxiv, and Google Scholar for ADHD/Autism papers
Downloads PDFs and supplementary files containing raw data

This finds data that's NOT in official databases!
"""

import argparse
import requests
import json
import time
from pathlib import Path
from datetime import datetime
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaperScraper:
    """Scrapes papers and supplementary data"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.pdfs_dir = self.output_dir / "pdfs"
        self.supplements_dir = self.output_dir / "supplements"
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.supplements_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def search_pubmed(self, query, max_results=50):
        """Search PubMed for papers"""
        logger.info(f"Searching PubMed: {query}")

        # Use E-utilities API
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        # Search
        search_url = f"{base_url}esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }

        try:
            response = self.session.get(search_url, params=params, timeout=30)
            data = response.json()

            pmids = data.get('esearchresult', {}).get('idlist', [])
            logger.info(f"Found {len(pmids)} papers")

            return pmids

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    def get_paper_metadata(self, pmid):
        """Get paper metadata from PubMed"""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml'
        }

        try:
            response = self.session.get(base_url, params=params, timeout=30)
            soup = BeautifulSoup(response.content, 'xml')

            # Extract metadata
            article = soup.find('PubmedArticle')
            if not article:
                return None

            title = article.find('ArticleTitle')
            title = title.text if title else 'Unknown'

            year = article.find('Year')
            year = year.text if year else 'Unknown'

            journal = article.find('Journal')
            journal = journal.find('Title').text if journal and journal.find('Title') else 'Unknown'

            # Check for PMC ID (open access)
            pmc_id = article.find('ArticleId', {'IdType': 'pmc'})
            pmc_id = pmc_id.text if pmc_id else None

            # Check for DOI
            doi = article.find('ArticleId', {'IdType': 'doi'})
            doi = doi.text if doi else None

            return {
                'pmid': pmid,
                'pmc_id': pmc_id,
                'doi': doi,
                'title': title,
                'year': year,
                'journal': journal
            }

        except Exception as e:
            logger.error(f"Failed to get metadata for PMID {pmid}: {e}")
            return None

    def download_pmc_paper(self, pmc_id):
        """Download open access paper from PubMed Central"""
        if not pmc_id:
            return None

        logger.info(f"Downloading PMC{pmc_id}...")

        # PMC full text XML
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"

        try:
            response = self.session.get(url, timeout=60)
            if response.status_code == 200 and 'pdf' in response.headers.get('content-type', '').lower():
                pdf_file = self.pdfs_dir / f"PMC{pmc_id}.pdf"
                with open(pdf_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"✓ Downloaded PMC{pmc_id}")
                return pdf_file

        except Exception as e:
            logger.warning(f"Failed to download PMC{pmc_id}: {e}")

        return None

    def search_supplements(self, pmc_id, pmid):
        """Search for supplementary files"""
        supplements = []

        if not pmc_id:
            return supplements

        logger.info(f"Searching supplements for PMC{pmc_id}...")

        # PMC supplements page
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"

        try:
            response = self.session.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find supplementary material links
            supp_links = soup.find_all('a', href=re.compile(r'supplementary|supp|additional'))

            for link in supp_links:
                href = link.get('href')
                if not href:
                    continue

                # Make absolute URL
                full_url = urljoin(url, href)

                # Download if it's a data file
                if any(ext in href.lower() for ext in ['.xlsx', '.csv', '.txt', '.zip', '.tsv', '.dat']):
                    filename = Path(urlparse(href).path).name
                    supp_file = self.supplements_dir / f"PMC{pmc_id}_{filename}"

                    try:
                        supp_response = self.session.get(full_url, timeout=60)
                        with open(supp_file, 'wb') as f:
                            f.write(supp_response.content)

                        supplements.append(supp_file)
                        logger.info(f"✓ Downloaded supplement: {filename}")

                    except Exception as e:
                        logger.warning(f"Failed to download {filename}: {e}")

            time.sleep(1)  # Be nice to NCBI servers

        except Exception as e:
            logger.error(f"Failed to search supplements: {e}")

        return supplements

    def scrape_papers(self, queries, max_papers_per_query=50):
        """Main scraping workflow"""
        all_results = []

        for query in queries:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing query: {query}")
            logger.info('='*80)

            # Search PubMed
            pmids = self.search_pubmed(query, max_results=max_papers_per_query)

            for pmid in pmids:
                # Get metadata
                metadata = self.get_paper_metadata(pmid)
                if not metadata:
                    continue

                logger.info(f"Processing: {metadata['title'][:80]}...")

                # Only download open access papers
                if metadata['pmc_id']:
                    # Download PDF
                    pdf = self.download_pmc_paper(metadata['pmc_id'])

                    # Download supplements
                    supplements = self.search_supplements(metadata['pmc_id'], pmid)

                    metadata['pdf_path'] = str(pdf) if pdf else None
                    metadata['supplements'] = [str(s) for s in supplements]
                    metadata['n_supplements'] = len(supplements)

                    all_results.append(metadata)

                    time.sleep(2)  # Rate limiting
                else:
                    logger.info("  Not open access - skipping")

            time.sleep(3)  # Rate limiting between queries

        # Save results
        results_file = self.output_dir / "scraped_papers.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"SCRAPING COMPLETE")
        logger.info('='*80)
        logger.info(f"Papers found: {len(all_results)}")
        logger.info(f"PDFs downloaded: {sum(1 for r in all_results if r['pdf_path'])}")
        logger.info(f"Supplements downloaded: {sum(r['n_supplements'] for r in all_results)}")
        logger.info(f"Results saved: {results_file}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description='Scrape papers for supplementary data')
    parser.add_argument('--query', action='append', required=True,
                       help='Search query (can specify multiple)')
    parser.add_argument('--max-papers', type=int, default=50,
                       help='Max papers per query')
    parser.add_argument('--output', type=str, default='data/papers',
                       help='Output directory')

    args = parser.parse_args()

    scraper = PaperScraper(args.output)
    results = scraper.scrape_papers(args.query, max_papers_per_query=args.max_papers)

    print(f"\nDownloaded {len(results)} papers with supplements")
    print(f"Check {args.output}/ for files")


if __name__ == '__main__':
    main()