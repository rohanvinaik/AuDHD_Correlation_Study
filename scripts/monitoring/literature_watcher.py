#!/usr/bin/env python3
"""
Literature Watcher for Dataset Announcements

Monitors scientific literature for:
- New dataset publications
- Data descriptor papers
- Resource announcements
- Preprints with datasets

Sources:
- PubMed/PMC
- bioRxiv/medRxiv preprints
- arXiv
- Nature Scientific Data
- GigaScience

Usage:
    python literature_watcher.py --check-all
    python literature_watcher.py --query "autism ADHD dataset"
    python literature_watcher.py --daemon --interval 86400

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

try:
    import requests
    from bs4 import BeautifulSoup
    import xml.etree.ElementTree as ET
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install requests beautifulsoup4")
    import sys
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Publication:
    """Scientific publication with dataset"""
    source: str  # pubmed, biorxiv, arxiv, etc.
    pubmed_id: Optional[str]
    doi: Optional[str]
    title: str
    authors: List[str]
    journal: str
    publication_date: str
    abstract: str
    dataset_mentions: List[str]
    data_availability: Optional[str]
    repository_links: List[str]
    accession_numbers: List[str]
    relevance_score: float
    detected_date: str


class LiteratureWatcher:
    """Monitor literature for dataset announcements"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Study-Monitor/1.0 (Research; mailto:contact@example.com)'
        })
        self.history_file = Path('data/monitoring/literature_history.json')
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        """Load publication history"""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return {'publications': []}

    def _save_history(self):
        """Save publication history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _is_seen(self, identifier: str) -> bool:
        """Check if publication already seen"""
        return any(p.get('pubmed_id') == identifier or p.get('doi') == identifier
                  for p in self.history.get('publications', []))

    def search_pubmed(
        self,
        query: str,
        days_back: int = 30,
        max_results: int = 100
    ) -> List[Publication]:
        """Search PubMed for publications"""
        publications = []

        try:
            logger.info(f"Searching PubMed: {query}")

            # PubMed E-utilities
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Search
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'datetype': 'pdat',
                'mindate': start_date.strftime('%Y/%m/%d'),
                'maxdate': end_date.strftime('%Y/%m/%d'),
                'retmode': 'json'
            }

            response = self.session.get(search_url, params=search_params, timeout=30)
            if response.status_code != 200:
                logger.error(f"PubMed search failed: {response.status_code}")
                return publications

            search_data = response.json()
            pmids = search_data.get('esearchresult', {}).get('idlist', [])

            if not pmids:
                logger.info("No PubMed results found")
                return publications

            logger.info(f"Found {len(pmids)} PubMed articles")

            # Fetch details in batches
            batch_size = 20
            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i+batch_size]

                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(batch_pmids),
                    'retmode': 'xml'
                }

                fetch_response = self.session.get(fetch_url, params=fetch_params, timeout=30)
                if fetch_response.status_code != 200:
                    continue

                # Parse XML
                root = ET.fromstring(fetch_response.content)

                for article in root.findall('.//PubmedArticle'):
                    pmid = article.findtext('.//PMID')

                    # Skip if already seen
                    if self._is_seen(pmid):
                        continue

                    # Extract article info
                    title = article.findtext('.//ArticleTitle', default='')
                    abstract_text = article.findtext('.//AbstractText', default='')

                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        lastname = author.findtext('LastName', default='')
                        forename = author.findtext('ForeName', default='')
                        if lastname:
                            authors.append(f"{forename} {lastname}".strip())

                    # Extract journal
                    journal = article.findtext('.//Journal/Title', default='')

                    # Extract publication date
                    pub_date_elem = article.find('.//PubDate')
                    if pub_date_elem is not None:
                        year = pub_date_elem.findtext('Year', default='')
                        month = pub_date_elem.findtext('Month', default='01')
                        day = pub_date_elem.findtext('Day', default='01')
                        pub_date = f"{year}-{month}-{day}"
                    else:
                        pub_date = ''

                    # Extract DOI
                    doi = None
                    for article_id in article.findall('.//ArticleId'):
                        if article_id.get('IdType') == 'doi':
                            doi = article_id.text
                            break

                    # Check for dataset indicators
                    full_text = f"{title} {abstract_text}".lower()
                    dataset_indicators = [
                        'dataset', 'data repository', 'publicly available',
                        'accession', 'dbgap', 'geo', 'ega', 'sra',
                        'data are available', 'deposited in'
                    ]

                    dataset_mentions = [ind for ind in dataset_indicators if ind in full_text]

                    if not dataset_mentions:
                        continue  # Skip if no dataset indicators

                    # Extract repository links and accessions
                    repo_links = self._extract_repository_links(abstract_text)
                    accessions = self._extract_accessions(abstract_text)

                    # Calculate relevance score
                    relevance = self._calculate_relevance(title, abstract_text, query)

                    publication = Publication(
                        source='PubMed',
                        pubmed_id=pmid,
                        doi=doi,
                        title=title,
                        authors=authors[:5],  # Limit to first 5 authors
                        journal=journal,
                        publication_date=pub_date,
                        abstract=abstract_text[:500],
                        dataset_mentions=dataset_mentions,
                        data_availability=self._extract_data_availability(abstract_text),
                        repository_links=repo_links,
                        accession_numbers=accessions,
                        relevance_score=relevance,
                        detected_date=datetime.now().isoformat()
                    )

                    publications.append(publication)
                    logger.info(f"Found dataset publication: {pmid} - {title[:50]}...")

                # Respect NCBI rate limits
                time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")

        return publications

    def search_biorxiv(
        self,
        query: str,
        days_back: int = 30,
        max_results: int = 100
    ) -> List[Publication]:
        """Search bioRxiv/medRxiv for preprints"""
        publications = []

        try:
            logger.info(f"Searching bioRxiv/medRxiv: {query}")

            # bioRxiv API
            api_url = "https://api.biorxiv.org/details/biorxiv"

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Search by date
            date_url = f"{api_url}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

            response = self.session.get(date_url, timeout=30)
            if response.status_code != 200:
                logger.error(f"bioRxiv search failed: {response.status_code}")
                return publications

            data = response.json()
            preprints = data.get('collection', [])

            logger.info(f"Found {len(preprints)} bioRxiv preprints")

            # Filter by query terms
            query_terms = query.lower().split()

            for preprint in preprints[:max_results]:
                title = preprint.get('title', '')
                abstract = preprint.get('abstract', '')
                doi = preprint.get('doi', '')

                # Skip if already seen
                if self._is_seen(doi):
                    continue

                # Check if matches query
                full_text = f"{title} {abstract}".lower()
                if not any(term in full_text for term in query_terms):
                    continue

                # Check for dataset indicators
                dataset_indicators = [
                    'dataset', 'data repository', 'publicly available',
                    'accession', 'github', 'zenodo', 'figshare'
                ]

                dataset_mentions = [ind for ind in dataset_indicators if ind in full_text]

                if not dataset_mentions:
                    continue

                # Extract info
                authors = preprint.get('authors', '').split(';')[:5]
                pub_date = preprint.get('date', '')

                repo_links = self._extract_repository_links(abstract)
                accessions = self._extract_accessions(abstract)
                relevance = self._calculate_relevance(title, abstract, query)

                publication = Publication(
                    source='bioRxiv',
                    pubmed_id=None,
                    doi=doi,
                    title=title,
                    authors=[a.strip() for a in authors],
                    journal='bioRxiv (preprint)',
                    publication_date=pub_date,
                    abstract=abstract[:500],
                    dataset_mentions=dataset_mentions,
                    data_availability=self._extract_data_availability(abstract),
                    repository_links=repo_links,
                    accession_numbers=accessions,
                    relevance_score=relevance,
                    detected_date=datetime.now().isoformat()
                )

                publications.append(publication)
                logger.info(f"Found preprint: {doi} - {title[:50]}...")

        except Exception as e:
            logger.error(f"Error searching bioRxiv: {e}")

        return publications

    def search_scientific_data(self, days_back: int = 30) -> List[Publication]:
        """Search Nature Scientific Data for data descriptors"""
        publications = []

        try:
            logger.info("Searching Nature Scientific Data")

            # Nature API or RSS feed
            rss_url = "https://www.nature.com/sdata.rss"

            response = self.session.get(rss_url, timeout=30)
            if response.status_code != 200:
                logger.error(f"Scientific Data RSS failed: {response.status_code}")
                return publications

            # Parse RSS
            import feedparser
            feed = feedparser.parse(response.content)

            cutoff_date = datetime.now() - timedelta(days=days_back)

            for entry in feed.entries:
                # Parse date
                if hasattr(entry, 'published_parsed'):
                    entry_date = datetime(*entry.published_parsed[:6])
                else:
                    entry_date = datetime.now()

                if entry_date < cutoff_date:
                    continue

                title = entry.get('title', '')
                doi = entry.get('link', '')

                # Skip if already seen
                if self._is_seen(doi):
                    continue

                summary = entry.get('summary', '')

                publication = Publication(
                    source='Scientific Data',
                    pubmed_id=None,
                    doi=doi,
                    title=title,
                    authors=[],
                    journal='Scientific Data',
                    publication_date=entry_date.isoformat(),
                    abstract=summary[:500],
                    dataset_mentions=['data descriptor'],
                    data_availability='Available in article',
                    repository_links=[],
                    accession_numbers=[],
                    relevance_score=0.8,  # High relevance for data descriptors
                    detected_date=datetime.now().isoformat()
                )

                publications.append(publication)
                logger.info(f"Found data descriptor: {title[:50]}...")

        except Exception as e:
            logger.error(f"Error searching Scientific Data: {e}")

        return publications

    def _extract_repository_links(self, text: str) -> List[str]:
        """Extract repository links from text"""
        import re

        patterns = [
            r'https?://github\.com/[\w\-/]+',
            r'https?://zenodo\.org/record/\d+',
            r'https?://figshare\.com/[\w/]+',
            r'https?://osf\.io/[\w/]+',
            r'https?://www\.ncbi\.nlm\.nih\.gov/geo/query/acc\.cgi\?acc=GSE\d+',
            r'https?://www\.ebi\.ac\.uk/arrayexpress/experiments/E-\w+-\d+',
        ]

        links = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            links.extend(matches)

        return list(set(links))

    def _extract_accessions(self, text: str) -> List[str]:
        """Extract accession numbers from text"""
        import re

        patterns = {
            'GEO': r'GSE\d+',
            'SRA': r'SRP\d+|SRR\d+',
            'dbGaP': r'phs\d{6}',
            'ArrayExpress': r'E-\w+-\d+',
            'EGA': r'EGAS\d+'
        }

        accessions = []
        for db, pattern in patterns.items():
            matches = re.findall(pattern, text)
            accessions.extend([f"{db}:{m}" for m in matches])

        return list(set(accessions))

    def _extract_data_availability(self, text: str) -> Optional[str]:
        """Extract data availability statement"""
        import re

        # Look for data availability section
        patterns = [
            r'data avail[a-z\s]+:([^\.]+)',
            r'data(?:\s+are|\s+is)?\s+available[^\.]+',
            r'deposited in[^\.]+',
            r'accessible at[^\.]+',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)[:200]

        return None

    def _calculate_relevance(self, title: str, abstract: str, query: str) -> float:
        """Calculate relevance score 0-1"""
        query_terms = set(query.lower().split())
        text = f"{title} {abstract}".lower()

        # Count query term matches
        matches = sum(1 for term in query_terms if term in text)
        term_score = matches / len(query_terms) if query_terms else 0

        # Bonus for dataset indicators
        dataset_indicators = ['dataset', 'repository', 'available']
        indicator_score = sum(0.1 for ind in dataset_indicators if ind in text)

        # Combine scores
        relevance = min(1.0, term_score * 0.7 + indicator_score)

        return round(relevance, 2)

    def watch_all_sources(
        self,
        query: str,
        days_back: int = 30
    ) -> List[Publication]:
        """Watch all literature sources"""
        all_publications = []

        # Search PubMed
        pubmed_pubs = self.search_pubmed(query, days_back)
        all_publications.extend(pubmed_pubs)

        # Search bioRxiv
        biorxiv_pubs = self.search_biorxiv(query, days_back)
        all_publications.extend(biorxiv_pubs)

        # Search Scientific Data
        scidata_pubs = self.search_scientific_data(days_back)
        all_publications.extend(scidata_pubs)

        # Sort by relevance
        all_publications.sort(key=lambda p: p.relevance_score, reverse=True)

        return all_publications

    def save_publications(self, publications: List[Publication], output_file: Path):
        """Save publications to file"""
        # Add to history
        for pub in publications:
            self.history['publications'].append(asdict(pub))

        self._save_history()

        # Save to output file
        data = {
            'generated_date': datetime.now().isoformat(),
            'total_publications': len(publications),
            'publications': [asdict(p) for p in publications]
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(publications)} publications to {output_file}")

    def run_daemon(self, query: str, interval_seconds: int = 86400):
        """Run watcher as daemon (daily)"""
        logger.info(f"Starting literature watcher daemon (interval: {interval_seconds}s)")

        while True:
            try:
                logger.info(f"Searching literature for: {query}")
                publications = self.watch_all_sources(query, days_back=7)

                if publications:
                    logger.info(f"Found {len(publications)} new publications")
                    self.save_publications(
                        publications,
                        Path('data/monitoring/new_publications.json')
                    )
                else:
                    logger.info("No new publications found")

                logger.info(f"Sleeping for {interval_seconds} seconds...")
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Daemon stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in daemon loop: {e}")
                time.sleep(3600)  # Sleep 1 hour before retrying


def main():
    parser = argparse.ArgumentParser(description='Watch literature for datasets')
    parser.add_argument('--check-all', action='store_true',
                       help='Check all sources')
    parser.add_argument('--query', default='autism ADHD dataset genomics',
                       help='Search query')
    parser.add_argument('--days-back', type=int, default=30,
                       help='Days to look back')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon')
    parser.add_argument('--interval', type=int, default=86400,
                       help='Daemon check interval (default: 24 hours)')
    parser.add_argument('--output', default='data/monitoring/new_publications.json',
                       help='Output file')

    args = parser.parse_args()

    watcher = LiteratureWatcher()

    if args.daemon:
        watcher.run_daemon(args.query, args.interval)
    elif args.check_all:
        publications = watcher.watch_all_sources(args.query, args.days_back)

        print(f"\n=== Found {len(publications)} publications ===\n")
        for pub in publications:
            print(f"[{pub.relevance_score:.2f}] {pub.source}: {pub.title}")
            print(f"  Authors: {', '.join(pub.authors[:3])}")
            print(f"  Journal: {pub.journal} ({pub.publication_date})")
            if pub.doi:
                print(f"  DOI: {pub.doi}")
            if pub.accession_numbers:
                print(f"  Accessions: {', '.join(pub.accession_numbers)}")
            if pub.repository_links:
                print(f"  Links: {pub.repository_links[0]}")
            print()

        watcher.save_publications(publications, Path(args.output))
    else:
        parser.print_help()


if __name__ == '__main__':
    import sys
    main()