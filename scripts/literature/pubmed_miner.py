#!/usr/bin/env python3
"""
PubMed Literature Miner for ADHD/Autism Research

Searches PubMed/PMC for relevant papers and extracts metadata including:
- Study types (clinical trials, observational studies, meta-analyses)
- Supplementary material links
- Dataset availability statements
- Author contact information
- GitHub repositories
- Data repository accessions

Uses NCBI E-utilities API for programmatic access to PubMed and PubMed Central.

Requirements:
    pip install biopython requests pandas beautifulsoup4 lxml

Usage:
    # Search for autism metabolomics papers
    python pubmed_miner.py --query autism metabolomics --output data/literature/

    # Search with advanced query
    python pubmed_miner.py --query-file queries/adhd_genomics.txt --years 2015:2024

    # Extract from PMIDs
    python pubmed_miner.py --pmids 30478444,30804558 --output data/literature/

    # Search with data availability filter
    python pubmed_miner.py --query adhd microbiome --has-data --output data/literature/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging
import xml.etree.ElementTree as ET

try:
    from Bio import Entrez
    import requests
    import pandas as pd
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install biopython requests pandas beautifulsoup4 lxml")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configure Entrez
Entrez.email = "your.email@institution.edu"  # Set this!
Entrez.tool = "AuDHD_Literature_Miner"

# Search query templates
QUERY_TEMPLATES = {
    'autism_metabolomics': '(autism[Title/Abstract] OR ASD[Title/Abstract] OR "autism spectrum"[Title/Abstract]) AND (metabolomics[Title/Abstract] OR metabolome[Title/Abstract] OR metabolite[Title/Abstract])',

    'adhd_genetics': '(ADHD[Title/Abstract] OR "attention deficit"[Title/Abstract] OR "attention-deficit"[Title/Abstract] OR hyperactivity[Title/Abstract]) AND (genomics[Title/Abstract] OR genetics[Title/Abstract] OR GWAS[Title/Abstract] OR "genome wide"[Title/Abstract])',

    'autism_genetics': '(autism[Title/Abstract] OR ASD[Title/Abstract]) AND (genomics[Title/Abstract] OR genetics[Title/Abstract] OR exome[Title/Abstract] OR sequencing[Title/Abstract])',

    'adhd_microbiome': '(ADHD[Title/Abstract] OR "attention deficit"[Title/Abstract]) AND (microbiome[Title/Abstract] OR microbiota[Title/Abstract] OR "gut brain"[Title/Abstract])',

    'autism_microbiome': '(autism[Title/Abstract] OR ASD[Title/Abstract]) AND (microbiome[Title/Abstract] OR microbiota[Title/Abstract] OR "gut brain"[Title/Abstract])',

    'neurodevelopmental_multiomics': '(neurodevelopmental[Title/Abstract] OR "developmental disorder"[Title/Abstract]) AND (multiomics[Title/Abstract] OR "multi-omics"[Title/Abstract] OR biomarker[Title/Abstract] OR integrative[Title/Abstract])',

    'adhd_neuroimaging': '(ADHD[Title/Abstract] OR "attention deficit"[Title/Abstract]) AND (MRI[Title/Abstract] OR neuroimaging[Title/Abstract] OR "brain imaging"[Title/Abstract] OR fMRI[Title/Abstract])',

    'autism_neuroimaging': '(autism[Title/Abstract] OR ASD[Title/Abstract]) AND (MRI[Title/Abstract] OR neuroimaging[Title/Abstract] OR "brain imaging"[Title/Abstract])'
}

# Data availability indicators
DATA_AVAILABILITY_TERMS = [
    'data are available',
    'data is available',
    'supplementary data',
    'supplementary material',
    'github.com',
    'figshare',
    'zenodo',
    'dryad',
    'GSE',  # GEO accession
    'PRJNA',  # SRA BioProject
    'EGAS',  # EGA study
    'dbGaP',
    'European Nucleotide Archive',
    'data repository',
    'publicly available',
    'openly available',
    'code availability',
    'software availability'
]

# Repository patterns
REPOSITORY_PATTERNS = {
    'geo': r'GSE\d+',
    'sra': r'PRJNA\d+|SRP\d+',
    'ega': r'EGAS\d+|EGAD\d+',
    'dbgap': r'phs\d+',
    'github': r'github\.com/[\w-]+/[\w-]+',
    'figshare': r'figshare\.com/articles/[\w/]+/\d+',
    'zenodo': r'zenodo\.org/record/\d+',
    'dryad': r'datadryad\.org/\w+/\d+'
}


@dataclass
class Paper:
    """Represents a research paper"""
    pmid: str
    pmcid: Optional[str]
    title: str
    abstract: str
    authors: List[Dict]
    journal: str
    publication_date: str
    publication_types: List[str]
    mesh_terms: List[str]
    doi: Optional[str]
    pmc_url: Optional[str]
    has_supplementary: bool
    supplementary_links: List[str]
    data_repositories: Dict[str, List[str]]
    dataset_mentions: List[str]
    github_repos: List[str]
    author_emails: List[str]
    corresponding_author: Optional[Dict]
    full_text_available: bool
    keywords: List[str]


class PubMedMiner:
    """Mine PubMed for ADHD/Autism research papers with data"""

    def __init__(self, output_dir: Path, email: str = None):
        """
        Initialize miner

        Args:
            output_dir: Output directory
            email: Email for NCBI API
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if email:
            Entrez.email = email

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        logger.info(f"Initialized PubMed miner: {output_dir}")

    def search_pubmed(self, query: str, retmax: int = 500,
                     years: Optional[str] = None,
                     has_abstract: bool = True,
                     publication_types: Optional[List[str]] = None) -> List[str]:
        """
        Search PubMed using Entrez API

        Args:
            query: Search query
            retmax: Maximum results
            years: Year range (e.g., "2015:2024")
            has_abstract: Require abstract
            publication_types: Filter by publication type

        Returns:
            List of PMIDs
        """
        logger.info(f"Searching PubMed: {query[:100]}...")

        # Build complete query
        full_query = query

        if years:
            full_query += f' AND ("{years}"[Date - Publication])'

        if has_abstract:
            full_query += ' AND hasabstract'

        if publication_types:
            pub_type_query = ' OR '.join([f'"{pt}"[Publication Type]' for pt in publication_types])
            full_query += f' AND ({pub_type_query})'

        try:
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=full_query,
                retmax=retmax,
                sort="relevance",
                usehistory="y"
            )
            record = Entrez.read(handle)
            handle.close()

            pmids = record.get("IdList", [])
            logger.info(f"Found {len(pmids)} papers")

            return pmids

        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []

    def fetch_paper_details(self, pmids: List[str]) -> List[Paper]:
        """
        Fetch detailed metadata for papers

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of Paper objects
        """
        logger.info(f"Fetching details for {len(pmids)} papers...")

        papers = []

        # Fetch in batches
        batch_size = 50
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]

            try:
                # Fetch PubMed records
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch),
                    retmode="xml"
                )

                xml_data = handle.read()
                handle.close()

                # Parse XML
                root = ET.fromstring(xml_data)

                for article in root.findall('.//PubmedArticle'):
                    paper = self._parse_pubmed_article(article)
                    if paper:
                        papers.append(paper)

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                continue

        logger.info(f"Parsed {len(papers)} papers")
        return papers

    def _parse_pubmed_article(self, article: ET.Element) -> Optional[Paper]:
        """Parse PubMed article XML"""
        try:
            # Get PMID
            pmid_elem = article.find('.//PMID')
            if pmid_elem is None:
                return None
            pmid = pmid_elem.text

            # Get title
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ''

            # Get abstract
            abstract_parts = article.findall('.//AbstractText')
            abstract = ' '.join([part.text or '' for part in abstract_parts])

            # Get authors
            authors = []
            author_list = article.find('.//AuthorList')
            if author_list is not None:
                for author in author_list.findall('.//Author'):
                    last_name = author.find('.//LastName')
                    fore_name = author.find('.//ForeName')

                    if last_name is not None:
                        author_dict = {
                            'last_name': last_name.text,
                            'first_name': fore_name.text if fore_name is not None else '',
                            'affiliations': []
                        }

                        # Get affiliations
                        for affiliation in author.findall('.//Affiliation'):
                            author_dict['affiliations'].append(affiliation.text or '')

                        authors.append(author_dict)

            # Get journal
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ''

            # Get publication date
            pub_date = article.find('.//PubDate')
            pub_date_str = ''
            if pub_date is not None:
                year = pub_date.find('.//Year')
                month = pub_date.find('.//Month')
                if year is not None:
                    pub_date_str = year.text
                    if month is not None:
                        pub_date_str += f"-{month.text}"

            # Get publication types
            pub_types = []
            for pub_type in article.findall('.//PublicationType'):
                pub_types.append(pub_type.text)

            # Get MeSH terms
            mesh_terms = []
            for mesh in article.findall('.//MeshHeading/DescriptorName'):
                mesh_terms.append(mesh.text)

            # Get DOI
            doi = None
            for article_id in article.findall('.//ArticleId'):
                if article_id.get('IdType') == 'doi':
                    doi = article_id.text

            # Get PMCID
            pmcid = None
            for article_id in article.findall('.//ArticleId'):
                if article_id.get('IdType') == 'pmc':
                    pmcid = article_id.text

            # Check for data availability
            searchable_text = f"{title} {abstract}".lower()
            has_supplementary = any(term.lower() in searchable_text for term in DATA_AVAILABILITY_TERMS)

            # Extract repository accessions
            data_repositories = {}
            for repo_type, pattern in REPOSITORY_PATTERNS.items():
                matches = re.findall(pattern, f"{title} {abstract}", re.IGNORECASE)
                if matches:
                    data_repositories[repo_type] = list(set(matches))

            # Extract GitHub repos
            github_repos = data_repositories.get('github', [])

            # Get keywords
            keywords = []
            for keyword in article.findall('.//Keyword'):
                keywords.append(keyword.text)

            paper = Paper(
                pmid=pmid,
                pmcid=pmcid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=pub_date_str,
                publication_types=pub_types,
                mesh_terms=mesh_terms,
                doi=doi,
                pmc_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else None,
                has_supplementary=has_supplementary,
                supplementary_links=[],  # Will be filled by supplement_extractor
                data_repositories=data_repositories,
                dataset_mentions=[],  # Will be filled by dataset_mention_finder
                github_repos=github_repos,
                author_emails=[],  # Will be filled if full text available
                corresponding_author=None,
                full_text_available=pmcid is not None,
                keywords=keywords
            )

            return paper

        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None

    def filter_papers_with_data(self, papers: List[Paper]) -> List[Paper]:
        """Filter papers that likely have data available"""
        filtered = []

        for paper in papers:
            # Check multiple criteria
            has_data = (
                paper.has_supplementary or
                bool(paper.data_repositories) or
                bool(paper.github_repos) or
                any(term in paper.abstract.lower() for term in ['data are available', 'data is available'])
            )

            if has_data:
                filtered.append(paper)

        logger.info(f"Filtered to {len(filtered)} papers with data availability indicators")
        return filtered

    def export_papers(self, papers: List[Paper], filename: str = 'papers_with_data.json'):
        """Export papers to JSON"""
        output_file = self.output_dir / filename

        papers_dict = [asdict(paper) for paper in papers]

        with open(output_file, 'w') as f:
            json.dump(papers_dict, f, indent=2)

        logger.info(f"Exported {len(papers)} papers: {output_file}")
        return output_file

    def generate_summary(self, papers: List[Paper]) -> Dict:
        """Generate summary statistics"""
        summary = {
            'total_papers': len(papers),
            'papers_with_pmc': sum(1 for p in papers if p.pmcid),
            'papers_with_doi': sum(1 for p in papers if p.doi),
            'papers_with_supplementary': sum(1 for p in papers if p.has_supplementary),
            'papers_with_github': sum(1 for p in papers if p.github_repos),
            'papers_with_repositories': sum(1 for p in papers if p.data_repositories),
            'repository_breakdown': {},
            'publication_types': {},
            'journals': {},
            'years': {}
        }

        # Repository breakdown
        for paper in papers:
            for repo_type in paper.data_repositories.keys():
                summary['repository_breakdown'][repo_type] = summary['repository_breakdown'].get(repo_type, 0) + 1

        # Publication types
        for paper in papers:
            for pub_type in paper.publication_types:
                summary['publication_types'][pub_type] = summary['publication_types'].get(pub_type, 0) + 1

        # Journals
        for paper in papers:
            summary['journals'][paper.journal] = summary['journals'].get(paper.journal, 0) + 1

        # Years
        for paper in papers:
            year = paper.publication_date[:4] if paper.publication_date else 'Unknown'
            summary['years'][year] = summary['years'].get(year, 0) + 1

        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Mine PubMed for ADHD/Autism research papers with data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search autism metabolomics
  python pubmed_miner.py --query autism metabolomics --output data/literature/

  # Use predefined query
  python pubmed_miner.py --query-template autism_genetics --years 2015:2024

  # Search with data filter
  python pubmed_miner.py --query adhd microbiome --has-data --output data/literature/

  # Extract from specific PMIDs
  python pubmed_miner.py --pmids 30478444,30804558 --output data/literature/

  # Clinical trials only
  python pubmed_miner.py --query autism biomarker --pub-types "Clinical Trial" --output data/literature/
        """
    )

    parser.add_argument(
        '--query',
        nargs='+',
        help='Search query terms'
    )

    parser.add_argument(
        '--query-template',
        type=str,
        choices=list(QUERY_TEMPLATES.keys()),
        help='Use predefined query template'
    )

    parser.add_argument(
        '--pmids',
        type=str,
        help='Comma-separated PMIDs to fetch'
    )

    parser.add_argument(
        '--years',
        type=str,
        help='Year range (e.g., "2015:2024")'
    )

    parser.add_argument(
        '--pub-types',
        nargs='+',
        choices=['Clinical Trial', 'Observational Study', 'Meta-Analysis', 'Review', 'Randomized Controlled Trial'],
        help='Filter by publication types'
    )

    parser.add_argument(
        '--has-data',
        action='store_true',
        help='Filter papers with data availability indicators'
    )

    parser.add_argument(
        '--max-results',
        type=int,
        default=500,
        help='Maximum results (default: 500)'
    )

    parser.add_argument(
        '--email',
        type=str,
        help='Your email for NCBI API'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/literature',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize miner
    miner = PubMedMiner(Path(args.output), email=args.email)

    papers = []

    # Handle PMIDs
    if args.pmids:
        pmids = args.pmids.split(',')
        papers = miner.fetch_paper_details(pmids)

    # Handle search
    elif args.query or args.query_template:
        # Build query
        if args.query_template:
            query = QUERY_TEMPLATES[args.query_template]
        else:
            query = ' AND '.join(args.query)

        # Search
        pmids = miner.search_pubmed(
            query,
            retmax=args.max_results,
            years=args.years,
            publication_types=args.pub_types
        )

        if pmids:
            papers = miner.fetch_paper_details(pmids)

    else:
        parser.print_help()
        return

    if not papers:
        print("\nNo papers found")
        return

    # Filter for data availability
    if args.has_data:
        papers = miner.filter_papers_with_data(papers)

    # Export papers
    output_file = miner.export_papers(papers)

    # Generate summary
    summary = miner.generate_summary(papers)

    print(f"\n=== PubMed Mining Results ===\n")
    print(f"Total papers: {summary['total_papers']}")
    print(f"Papers with PMC ID: {summary['papers_with_pmc']}")
    print(f"Papers with supplementary material: {summary['papers_with_supplementary']}")
    print(f"Papers with GitHub repos: {summary['papers_with_github']}")
    print(f"Papers with data repositories: {summary['papers_with_repositories']}")

    if summary['repository_breakdown']:
        print(f"\nRepository breakdown:")
        for repo, count in summary['repository_breakdown'].items():
            print(f"  {repo}: {count}")

    print(f"\nTop 10 journals:")
    top_journals = sorted(summary['journals'].items(), key=lambda x: x[1], reverse=True)[:10]
    for journal, count in top_journals:
        print(f"  {journal}: {count}")

    print(f"\nPublications by year:")
    for year in sorted(summary['years'].keys(), reverse=True):
        print(f"  {year}: {summary['years'][year]}")

    # Save summary
    summary_file = miner.output_dir / 'mining_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved:")
    print(f"  Papers: {output_file}")
    print(f"  Summary: {summary_file}")


if __name__ == '__main__':
    main()