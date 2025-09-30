#!/usr/bin/env python3
"""
NCBI SRA Searcher for ADHD/Autism Microbiome Research

Searches NCBI Sequence Read Archive (SRA) for 16S rRNA and metagenomic
sequencing studies related to ADHD and autism using the Entrez API.

SRA contains:
- 16S rRNA amplicon sequencing (V3-V4, V4, full-length)
- Whole metagenome shotgun sequencing (WGS)
- Metatranscriptomics (RNA-seq)
- Thousands of gut microbiome studies
- Structured metadata (BioSample, BioProject)

Key microbiome studies:
- Gut-brain axis investigations
- SCFA (short-chain fatty acid) producers
- Neurotransmitter-producing bacteria
- Dietary interventions
- Probiotic trials

Requirements:
    pip install biopython pandas requests

Usage:
    # Search for ADHD microbiome studies
    python sra_searcher.py --search adhd --output data/microbiome/

    # Search for autism studies
    python sra_searcher.py --search autism --output data/microbiome/

    # Search both with custom filters
    python sra_searcher.py --search both --min-samples 20 --output data/microbiome/

    # Get details for specific BioProject
    python sra_searcher.py --bioproject PRJNA123456

    # Download metadata for study
    python sra_searcher.py --download-metadata PRJNA123456

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
import re
import xml.etree.ElementTree as ET

try:
    from Bio import Entrez
    import pandas as pd
    import requests
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install biopython pandas requests")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# SRA API configuration
Entrez.email = "your.email@institution.edu"  # Set this!
Entrez.tool = "AuDHD_Microbiome_Pipeline"

# Search queries
SRA_QUERIES = {
    'adhd': {
        'query': '((ADHD[Title] OR "attention deficit"[Title] OR hyperactivity[Title]) '
                 'AND (16S[All Fields] OR metagenom*[All Fields] OR microbiome[All Fields]) '
                 'AND "Homo sapiens"[Organism])',
        'description': 'ADHD microbiome studies'
    },
    'autism': {
        'query': '((autism[Title] OR ASD[Title] OR "autistic disorder"[Title] OR Asperger[Title]) '
                 'AND (16S[All Fields] OR metagenom*[All Fields] OR microbiome[All Fields]) '
                 'AND "Homo sapiens"[Organism])',
        'description': 'Autism microbiome studies'
    },
    'gut_brain': {
        'query': '(("gut brain axis"[All Fields] OR "gut-brain"[All Fields]) '
                 'AND (16S[All Fields] OR metagenom*[All Fields]) '
                 'AND "Homo sapiens"[Organism])',
        'description': 'Gut-brain axis studies'
    },
    'neurodevelopmental': {
        'query': '((neurodevelopmental[All Fields] OR "developmental disorder"[All Fields]) '
                 'AND microbiome[All Fields] '
                 'AND "Homo sapiens"[Organism])',
        'description': 'Neurodevelopmental disorder microbiome'
    }
}

# Sequencing platforms and strategies
SEQUENCING_PLATFORMS = ['Illumina', 'PacBio', 'Oxford Nanopore', 'Ion Torrent']
SEQUENCING_STRATEGIES = ['AMPLICON', 'WGS', 'RNA-Seq', 'METAGENOMIC']

# Sample types of interest
SAMPLE_TYPES = [
    'stool', 'fecal', 'gut', 'intestinal',
    'saliva', 'oral', 'buccal',
    'duodenal', 'colonic', 'rectal'
]

# ADHD/Autism-relevant bacterial genera
KEY_GENERA = {
    'scfa_producers': [
        'Faecalibacterium', 'Roseburia', 'Eubacterium',
        'Butyricicoccus', 'Coprococcus', 'Anaerostipes'
    ],
    'neurotransmitter_producers': [
        'Lactobacillus', 'Bifidobacterium', 'Streptococcus',
        'Escherichia', 'Enterococcus', 'Bacillus'
    ],
    'pro_inflammatory': [
        'Clostridium', 'Desulfovibrio', 'Sutterella'
    ],
    'beneficial': [
        'Akkermansia', 'Prevotella', 'Bacteroides'
    ]
}


@dataclass
class SRAStudy:
    """Represents an SRA study"""
    bioproject_id: str
    sra_study_id: str
    title: str
    abstract: str
    organism: str
    num_samples: int
    num_runs: int
    sequencing_strategy: str
    sequencing_platform: str
    sample_type: str
    publication_date: str
    pubmed_id: Optional[str]
    relevance_score: float
    matched_terms: List[str]
    case_count: Optional[int]
    control_count: Optional[int]
    metadata_available: bool
    url: str


class SRASearcher:
    """Search NCBI SRA for ADHD/Autism microbiome studies"""

    def __init__(self, output_dir: Path, email: str = None):
        """
        Initialize searcher

        Args:
            output_dir: Output directory
            email: Your email for NCBI API (required)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if email:
            Entrez.email = email

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        logger.info(f"Initialized SRA searcher: {output_dir}")

    def search_sra(self, query: str, retmax: int = 200) -> List[str]:
        """
        Search SRA using Entrez API

        Args:
            query: Search query
            retmax: Maximum results to return

        Returns:
            List of SRA study IDs
        """
        logger.info(f"Searching SRA with query: {query[:100]}...")

        try:
            # Search SRA database
            handle = Entrez.esearch(
                db="sra",
                term=query,
                retmax=retmax,
                usehistory="y"
            )
            record = Entrez.read(handle)
            handle.close()

            id_list = record.get("IdList", [])
            logger.info(f"Found {len(id_list)} SRA records")

            return id_list

        except Exception as e:
            logger.error(f"Error searching SRA: {e}")
            return []

    def fetch_study_details(self, sra_ids: List[str]) -> List[Dict]:
        """
        Fetch detailed metadata for SRA studies

        Args:
            sra_ids: List of SRA IDs

        Returns:
            List of study metadata dictionaries
        """
        logger.info(f"Fetching details for {len(sra_ids)} studies...")

        studies = []

        # Fetch in batches
        batch_size = 20
        for i in range(0, len(sra_ids), batch_size):
            batch = sra_ids[i:i+batch_size]

            try:
                # Fetch metadata
                handle = Entrez.efetch(
                    db="sra",
                    id=",".join(batch),
                    retmode="xml"
                )

                xml_data = handle.read()
                handle.close()

                # Parse XML
                root = ET.fromstring(xml_data)

                for experiment_package in root.findall('.//EXPERIMENT_PACKAGE'):
                    study_data = self._parse_experiment_package(experiment_package)
                    if study_data:
                        studies.append(study_data)

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                continue

        logger.info(f"Successfully parsed {len(studies)} studies")
        return studies

    def _parse_experiment_package(self, exp_pkg: ET.Element) -> Optional[Dict]:
        """Parse experiment package XML"""
        try:
            # Get study info
            study = exp_pkg.find('.//STUDY')
            if study is None:
                return None

            study_accession = study.get('accession', '')

            # Get title and abstract
            descriptor = study.find('.//DESCRIPTOR')
            title = ''
            abstract = ''
            if descriptor is not None:
                study_title = descriptor.find('.//STUDY_TITLE')
                if study_title is not None:
                    title = study_title.text or ''

                study_abstract = descriptor.find('.//STUDY_ABSTRACT')
                if study_abstract is not None:
                    abstract = study_abstract.text or ''

            # Get BioProject
            bioproject_id = ''
            external_ids = study.findall('.//EXTERNAL_ID')
            for ext_id in external_ids:
                if ext_id.get('namespace') == 'BioProject':
                    bioproject_id = ext_id.text or ''
                    break

            # Get experiment info
            experiment = exp_pkg.find('.//EXPERIMENT')
            platform_elem = experiment.find('.//PLATFORM') if experiment is not None else None
            sequencing_platform = ''
            if platform_elem is not None:
                for platform in SEQUENCING_PLATFORMS:
                    if platform_elem.find(f'.//{platform.upper().replace(" ", "_")}') is not None:
                        sequencing_platform = platform
                        break

            # Get sequencing strategy
            design = experiment.find('.//DESIGN') if experiment is not None else None
            library_descriptor = design.find('.//LIBRARY_DESCRIPTOR') if design is not None else None
            strategy_elem = library_descriptor.find('.//LIBRARY_STRATEGY') if library_descriptor is not None else None
            sequencing_strategy = strategy_elem.text if strategy_elem is not None else ''

            # Get sample info
            sample = exp_pkg.find('.//SAMPLE')
            organism = ''
            sample_type = ''
            if sample is not None:
                sci_name = sample.find('.//SCIENTIFIC_NAME')
                if sci_name is not None:
                    organism = sci_name.text or ''

                # Check sample attributes for type
                for attr in sample.findall('.//SAMPLE_ATTRIBUTE'):
                    tag = attr.find('.//TAG')
                    value = attr.find('.//VALUE')
                    if tag is not None and value is not None:
                        tag_text = (tag.text or '').lower()
                        value_text = (value.text or '').lower()

                        if any(t in tag_text or t in value_text for t in ['sample_type', 'tissue', 'body_site']):
                            sample_type = value.text or ''
                            break

            # Get run info
            runs = exp_pkg.findall('.//RUN')
            num_runs = len(runs)

            # Calculate relevance
            relevance_score, matched_terms = self._calculate_relevance(
                title, abstract, sample_type
            )

            # Get publication info
            pubmed_id = None
            study_links = study.findall('.//STUDY_LINK')
            for link in study_links:
                xref_link = link.find('.//XREF_LINK')
                if xref_link is not None:
                    db = xref_link.find('.//DB')
                    if db is not None and db.text == 'pubmed':
                        id_elem = xref_link.find('.//ID')
                        if id_elem is not None:
                            pubmed_id = id_elem.text

            return {
                'bioproject_id': bioproject_id,
                'sra_study_id': study_accession,
                'title': title,
                'abstract': abstract,
                'organism': organism,
                'num_runs': num_runs,
                'sequencing_strategy': sequencing_strategy,
                'sequencing_platform': sequencing_platform,
                'sample_type': sample_type,
                'pubmed_id': pubmed_id,
                'relevance_score': relevance_score,
                'matched_terms': matched_terms,
                'url': f"https://www.ncbi.nlm.nih.gov/bioproject/{bioproject_id}"
            }

        except Exception as e:
            logger.error(f"Error parsing experiment package: {e}")
            return None

    def _calculate_relevance(self, title: str, abstract: str,
                           sample_type: str) -> Tuple[float, List[str]]:
        """Calculate relevance score for study"""
        score = 0.0
        matched_terms = []

        searchable_text = f"{title} {abstract} {sample_type}".lower()

        # Check for ADHD terms
        adhd_terms = ['adhd', 'attention deficit', 'hyperactivity', 'hyperkinetic']
        for term in adhd_terms:
            if term in searchable_text:
                score += 15.0
                matched_terms.append(f"ADHD:{term}")

        # Check for autism terms
        autism_terms = ['autism', 'asd', 'autistic', 'asperger']
        for term in autism_terms:
            if term in searchable_text:
                score += 15.0
                matched_terms.append(f"Autism:{term}")

        # Check for neurodevelopmental
        neurodev_terms = ['neurodevelopmental', 'developmental disorder']
        for term in neurodev_terms:
            if term in searchable_text:
                score += 10.0
                matched_terms.append(f"NeuroD:{term}")

        # Bonus for gut-brain axis
        if 'gut brain' in searchable_text or 'gut-brain' in searchable_text:
            score += 8.0
            matched_terms.append("Gut-brain axis")

        # Check for relevant sample types
        for sample in SAMPLE_TYPES:
            if sample in searchable_text:
                score += 5.0
                matched_terms.append(f"Sample:{sample}")
                break

        # Bonus for key bacterial genera mentioned
        for category, genera in KEY_GENERA.items():
            for genus in genera:
                if genus.lower() in searchable_text:
                    score += 3.0
                    matched_terms.append(f"Genus:{genus}")
                    break

        # Check for SCFA
        if 'scfa' in searchable_text or 'short chain fatty acid' in searchable_text:
            score += 5.0
            matched_terms.append("SCFA")

        # Bonus for intervention/trial
        if any(term in searchable_text for term in ['intervention', 'trial', 'probiotic', 'prebiotic']):
            score += 3.0
            matched_terms.append("Intervention")

        return score, matched_terms

    def get_bioproject_metadata(self, bioproject_id: str) -> Optional[Dict]:
        """
        Get detailed metadata for a BioProject

        Args:
            bioproject_id: BioProject ID (e.g., PRJNA123456)

        Returns:
            Detailed metadata dictionary
        """
        logger.info(f"Fetching metadata for {bioproject_id}...")

        try:
            # Search for BioProject
            handle = Entrez.esearch(
                db="bioproject",
                term=bioproject_id
            )
            record = Entrez.read(handle)
            handle.close()

            if not record.get("IdList"):
                logger.warning(f"BioProject {bioproject_id} not found")
                return None

            bioproject_uid = record["IdList"][0]

            # Fetch details
            handle = Entrez.efetch(
                db="bioproject",
                id=bioproject_uid,
                retmode="xml"
            )
            xml_data = handle.read()
            handle.close()

            # Parse XML
            root = ET.fromstring(xml_data)

            # Extract metadata
            metadata = {
                'bioproject_id': bioproject_id,
                'title': '',
                'description': '',
                'organism': '',
                'data_type': '',
                'num_samples': 0,
                'registration_date': '',
                'modification_date': '',
                'publications': []
            }

            project = root.find('.//Project')
            if project is not None:
                # Title
                title_elem = project.find('.//ProjectDescr/Title')
                if title_elem is not None:
                    metadata['title'] = title_elem.text or ''

                # Description
                desc_elem = project.find('.//ProjectDescr/Description')
                if desc_elem is not None:
                    metadata['description'] = desc_elem.text or ''

            return metadata

        except Exception as e:
            logger.error(f"Error fetching BioProject metadata: {e}")
            return None

    def download_study_metadata(self, bioproject_id: str) -> Optional[Path]:
        """
        Download SRA RunInfo table for a study

        Args:
            bioproject_id: BioProject ID

        Returns:
            Path to downloaded metadata CSV
        """
        logger.info(f"Downloading metadata for {bioproject_id}...")

        try:
            # Use SRA RunInfo to get metadata
            url = f"https://www.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?save=efetch&db=sra&rettype=runinfo&term={bioproject_id}"

            response = self.session.get(url, timeout=60)

            if response.status_code != 200:
                logger.error(f"Failed to download metadata: HTTP {response.status_code}")
                return None

            # Save to file
            output_file = self.output_dir / f"{bioproject_id}_RunInfo.csv"
            output_file.write_text(response.text)

            logger.info(f"Downloaded metadata: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error downloading metadata: {e}")
            return None

    def generate_study_catalog(self, studies: List[Dict],
                              min_relevance: float = 5.0) -> pd.DataFrame:
        """
        Generate catalog of relevant studies

        Args:
            studies: List of study dictionaries
            min_relevance: Minimum relevance score

        Returns:
            DataFrame with study catalog
        """
        # Filter by relevance
        relevant = [s for s in studies if s.get('relevance_score', 0) >= min_relevance]

        if not relevant:
            logger.warning("No studies meet relevance threshold")
            return pd.DataFrame()

        df = pd.DataFrame(relevant)
        df = df.sort_values('relevance_score', ascending=False)

        logger.info(f"Generated catalog with {len(df)} relevant studies")
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Search NCBI SRA for ADHD/Autism microbiome studies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for ADHD microbiome studies
  python sra_searcher.py --search adhd --output data/microbiome/

  # Search for autism studies
  python sra_searcher.py --search autism --output data/microbiome/

  # Search all neurodevelopmental
  python sra_searcher.py --search both --output data/microbiome/

  # Get details for specific BioProject
  python sra_searcher.py --bioproject PRJNA123456

  # Download metadata
  python sra_searcher.py --download-metadata PRJNA123456

Note: Set your email in the script or use --email parameter
        """
    )

    parser.add_argument(
        '--search',
        type=str,
        choices=['adhd', 'autism', 'both', 'gut_brain', 'neurodevelopmental'],
        help='Search for specific condition'
    )

    parser.add_argument(
        '--bioproject',
        type=str,
        help='Get details for specific BioProject'
    )

    parser.add_argument(
        '--download-metadata',
        type=str,
        help='Download RunInfo metadata for BioProject'
    )

    parser.add_argument(
        '--min-relevance',
        type=float,
        default=5.0,
        help='Minimum relevance score (default: 5.0)'
    )

    parser.add_argument(
        '--max-results',
        type=int,
        default=200,
        help='Maximum search results (default: 200)'
    )

    parser.add_argument(
        '--email',
        type=str,
        help='Your email for NCBI API (required)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/microbiome/sra',
        help='Output directory'
    )

    args = parser.parse_args()

    # Check email
    if not args.email and Entrez.email == "your.email@institution.edu":
        print("\nWarning: Please set your email for NCBI API")
        print("Use --email parameter or edit the script\n")

    # Initialize searcher
    searcher = SRASearcher(Path(args.output), email=args.email)

    # Handle BioProject query
    if args.bioproject:
        metadata = searcher.get_bioproject_metadata(args.bioproject)
        if metadata:
            print(f"\n=== {args.bioproject} Metadata ===\n")
            print(json.dumps(metadata, indent=2))
        else:
            print(f"Could not fetch metadata for {args.bioproject}")
        return

    # Handle metadata download
    if args.download_metadata:
        output_file = searcher.download_study_metadata(args.download_metadata)
        if output_file:
            print(f"\nMetadata downloaded: {output_file}")

            # Show preview
            df = pd.read_csv(output_file)
            print(f"\n{len(df)} samples found")
            print(f"\nColumns: {', '.join(df.columns.tolist()[:10])}...")
        return

    # Handle search
    if args.search:
        queries = []

        if args.search == 'both':
            queries = [
                ('adhd', SRA_QUERIES['adhd']['query']),
                ('autism', SRA_QUERIES['autism']['query'])
            ]
        else:
            queries = [(args.search, SRA_QUERIES[args.search]['query'])]

        all_studies = []

        for condition, query in queries:
            print(f"\n=== Searching {condition.upper()} studies ===\n")

            # Search SRA
            sra_ids = searcher.search_sra(query, retmax=args.max_results)

            if not sra_ids:
                print(f"No results found for {condition}")
                continue

            # Fetch details
            studies = searcher.fetch_study_details(sra_ids)
            all_studies.extend(studies)

            print(f"Found {len(studies)} {condition} studies")

        if all_studies:
            # Generate catalog
            catalog_df = searcher.generate_study_catalog(
                all_studies,
                min_relevance=args.min_relevance
            )

            if not catalog_df.empty:
                # Save catalog
                catalog_file = searcher.output_dir / 'sra_study_catalog.csv'
                catalog_df.to_csv(catalog_file, index=False)
                print(f"\nCatalog saved: {catalog_file}")

                # Print summary
                print(f"\n=== Study Catalog Summary ===\n")
                print(f"Total studies: {len(catalog_df)}")
                print(f"Total runs: {catalog_df['num_runs'].sum()}")

                print(f"\nSequencing strategies:")
                print(catalog_df['sequencing_strategy'].value_counts().to_string())

                print(f"\nTop 10 studies by relevance:")
                print(catalog_df[['bioproject_id', 'title', 'num_runs',
                                 'relevance_score']].head(10).to_string(index=False))

                # Save detailed JSON
                json_file = searcher.output_dir / 'sra_studies_detailed.json'
                catalog_df.to_json(json_file, orient='records', indent=2)
                print(f"\nDetailed data saved: {json_file}")
            else:
                print(f"\nNo studies found with relevance >= {args.min_relevance}")
        else:
            print("\nNo studies found")

    if not any([args.search, args.bioproject, args.download_metadata]):
        parser.print_help()


if __name__ == '__main__':
    main()