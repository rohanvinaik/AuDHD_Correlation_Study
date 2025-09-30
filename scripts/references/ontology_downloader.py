#!/usr/bin/env python3
"""
Ontology and Reference Database Downloader

Downloads and manages biomedical ontologies and reference databases for
ADHD/Autism research including clinical terminologies, drug classifications,
and standardized vocabularies.

Supported Ontologies:
- HPO (Human Phenotype Ontology): Phenotypic abnormalities
- SNOMED CT: Clinical terminology (requires UMLS license)
- ICD-10: Disease classification
- RxNorm: Drug nomenclature
- ATC: Anatomical Therapeutic Chemical classification
- FNDDS: Food and Nutrient Database

Clinical Databases:
- ClinVar: Genetic variants and clinical significance
- OMIM: Genetic disorders (requires license)
- OrphaNet: Rare disease data
- SFARI Gene: Autism gene database

Requirements:
    pip install requests pronto obonet networkx pandas pyyaml

Usage:
    # Download all open-access ontologies
    python ontology_downloader.py --download-all --output data/references/

    # Download specific ontology
    python ontology_downloader.py --ontology HPO --output data/references/

    # Update existing ontologies
    python ontology_downloader.py --update --output data/references/

    # Check versions
    python ontology_downloader.py --check-versions

Author: AuDHD Correlation Study Team
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import gzip
import shutil

try:
    import requests
    import pandas as pd
    import yaml
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install requests pandas pyyaml")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Ontology download URLs
ONTOLOGY_SOURCES = {
    'HPO': {
        'name': 'Human Phenotype Ontology',
        'url': 'https://purl.obolibrary.org/obo/hp.obo',
        'format': 'obo',
        'license': 'Open',
        'description': 'Phenotypic abnormalities in human disease',
        'relevance': 'ADHD/autism phenotypes, comorbidities',
        'terms_count': 16000,
        'version_url': 'https://hpo.jax.org/app/data/ontology',
        'annotations_url': 'https://hpo.jax.org/app/data/annotations'
    },
    'HPO_ANNOTATIONS': {
        'name': 'HPO Disease Annotations',
        'url': 'http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa',
        'format': 'tsv',
        'license': 'Open',
        'description': 'Disease-phenotype associations',
        'relevance': 'Link ADHD/autism to phenotypes'
    },
    'GO': {
        'name': 'Gene Ontology',
        'url': 'http://purl.obolibrary.org/obo/go.obo',
        'format': 'obo',
        'license': 'Open',
        'description': 'Gene functions, biological processes, cellular components',
        'relevance': 'Neurodevelopmental processes, synaptic function',
        'terms_count': 44000
    },
    'GO_ANNOTATIONS': {
        'name': 'GO Gene Annotations (Human)',
        'url': 'http://geneontology.org/gene-associations/goa_human.gaf.gz',
        'format': 'gaf.gz',
        'license': 'Open',
        'description': 'Gene-function associations'
    },
    'MONDO': {
        'name': 'Monarch Disease Ontology',
        'url': 'https://purl.obolibrary.org/obo/mondo.obo',
        'format': 'obo',
        'license': 'Open',
        'description': 'Disease classification and relationships',
        'relevance': 'ADHD, autism, comorbid conditions',
        'terms_count': 23000
    },
    'DO': {
        'name': 'Disease Ontology',
        'url': 'https://purl.obolibrary.org/obo/doid.obo',
        'format': 'obo',
        'license': 'Open',
        'description': 'Human disease classification',
        'relevance': 'Psychiatric and neurological diseases'
    },
    'UBERON': {
        'name': 'Uber Anatomy Ontology',
        'url': 'https://purl.obolibrary.org/obo/uberon.obo',
        'format': 'obo',
        'license': 'Open',
        'description': 'Anatomical structures across species',
        'relevance': 'Brain regions, gut anatomy'
    },
    'CHEBI': {
        'name': 'Chemical Entities of Biological Interest',
        'url': 'https://purl.obolibrary.org/obo/chebi.obo',
        'format': 'obo',
        'license': 'Open',
        'description': 'Chemical compounds, drugs, metabolites',
        'relevance': 'Neurotransmitters, SCFAs, medications',
        'terms_count': 61000
    },
    'RXNORM': {
        'name': 'RxNorm',
        'url': 'https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_current.zip',
        'format': 'zip',
        'license': 'Open (UMLS license)',
        'description': 'Drug nomenclature',
        'relevance': 'ADHD medications, SSRIs, antipsychotics',
        'note': 'Requires UMLS Terminology Services (UTS) account'
    },
    'ATC': {
        'name': 'Anatomical Therapeutic Chemical',
        'url': 'https://bioportal.bioontology.org/ontologies/ATC',
        'format': 'web',
        'license': 'Open',
        'description': 'Drug classification system',
        'relevance': 'Medication categorization',
        'note': 'Available through BioPortal or WHO website'
    },
    'ICD10': {
        'name': 'ICD-10',
        'url': 'https://icd.who.int/browse10/Content/statichtml/ICD10Volume2_en_2019.pdf',
        'format': 'various',
        'license': 'Open',
        'description': 'International Classification of Diseases',
        'relevance': 'F90 (ADHD), F84 (ASD) codes',
        'note': 'Multiple formats available from WHO and NIH'
    },
    'CLINVAR': {
        'name': 'ClinVar',
        'url': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz',
        'format': 'tsv.gz',
        'license': 'Open',
        'description': 'Genetic variants and clinical significance',
        'relevance': 'ADHD/autism associated variants'
    },
    'ORPHANET': {
        'name': 'Orphanet',
        'url': 'https://www.orphadata.com/data/xml/en_product1.xml',
        'format': 'xml',
        'license': 'Open',
        'description': 'Rare disease database',
        'relevance': 'Rare neurodevelopmental disorders'
    },
    'SFARI': {
        'name': 'SFARI Gene',
        'url': 'https://gene.sfari.org/database/human-gene/',
        'format': 'csv',
        'license': 'Open',
        'description': 'Autism candidate genes with scores',
        'relevance': 'Autism genetics',
        'note': 'Download from web interface'
    },
    'FNDDS': {
        'name': 'Food and Nutrient Database',
        'url': 'https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_csv_2024-10-31.zip',
        'format': 'csv.zip',
        'license': 'Open',
        'description': 'Food composition data',
        'relevance': 'Dietary interventions'
    }
}


@dataclass
class OntologyMetadata:
    """Metadata for downloaded ontology"""
    name: str
    source: str
    version: str
    download_date: str
    file_path: str
    file_size: int
    checksum: str
    format: str
    license: str
    terms_count: Optional[int] = None
    url: str = ''


class OntologyDownloader:
    """Download and manage biomedical ontologies"""

    def __init__(self, output_dir: Path):
        """
        Initialize downloader

        Args:
            output_dir: Output directory for downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        self.metadata_file = self.output_dir / 'ontology_versions.json'
        self.metadata = self._load_metadata()

        logger.info(f"Initialized ontology downloader: {output_dir}")

    def _load_metadata(self) -> Dict:
        """Load existing metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Metadata saved: {self.metadata_file}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def download_ontology(self, ontology_key: str,
                         force: bool = False) -> Optional[Path]:
        """
        Download specific ontology

        Args:
            ontology_key: Ontology identifier (e.g., 'HPO', 'GO')
            force: Force re-download even if exists

        Returns:
            Path to downloaded file
        """
        if ontology_key not in ONTOLOGY_SOURCES:
            logger.error(f"Unknown ontology: {ontology_key}")
            return None

        source = ONTOLOGY_SOURCES[ontology_key]
        logger.info(f"Downloading {source['name']}...")

        # Check if already downloaded
        if not force and ontology_key in self.metadata:
            existing_file = Path(self.metadata[ontology_key]['file_path'])
            if existing_file.exists():
                logger.info(f"Using cached version: {existing_file}")
                return existing_file

        # Determine output filename
        url = source['url']
        file_format = source['format']

        if file_format == 'web':
            logger.warning(f"{source['name']} requires manual download from: {url}")
            logger.warning(f"Note: {source.get('note', 'See documentation')}")
            return None

        # Extract filename from URL
        filename = url.split('/')[-1]
        if not filename or '?' in filename:
            filename = f"{ontology_key.lower()}.{file_format}"

        output_file = self.output_dir / filename

        try:
            # Download with progress
            logger.info(f"Downloading from: {url}")
            response = self.session.get(url, stream=True, timeout=300)

            if response.status_code != 200:
                logger.error(f"Download failed: HTTP {response.status_code}")
                return None

            # Get file size
            total_size = int(response.headers.get('content-length', 0))

            # Download
            with open(output_file, 'wb') as f:
                if total_size > 0:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
                    print()
                else:
                    f.write(response.content)

            logger.info(f"Downloaded: {output_file}")

            # Extract if compressed
            if filename.endswith('.gz') and not filename.endswith('.tar.gz'):
                extracted_file = output_file.with_suffix('')
                if not extracted_file.exists() or force:
                    logger.info(f"Extracting: {output_file}")
                    with gzip.open(output_file, 'rb') as f_in:
                        with open(extracted_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    logger.info(f"Extracted: {extracted_file}")
                    output_file = extracted_file

            # Calculate checksum
            checksum = self._calculate_checksum(output_file)

            # Count terms if OBO format
            terms_count = None
            if file_format == 'obo':
                terms_count = self._count_obo_terms(output_file)

            # Store metadata
            metadata = OntologyMetadata(
                name=source['name'],
                source=ontology_key,
                version=datetime.now().strftime('%Y-%m-%d'),
                download_date=datetime.now().isoformat(),
                file_path=str(output_file),
                file_size=output_file.stat().st_size,
                checksum=checksum,
                format=file_format,
                license=source['license'],
                terms_count=terms_count,
                url=url
            )

            self.metadata[ontology_key] = asdict(metadata)
            self._save_metadata()

            return output_file

        except Exception as e:
            logger.error(f"Error downloading {ontology_key}: {e}")
            return None

    def _count_obo_terms(self, obo_file: Path) -> int:
        """Count terms in OBO file"""
        try:
            count = 0
            with open(obo_file) as f:
                for line in f:
                    if line.startswith('[Term]'):
                        count += 1
            return count
        except Exception as e:
            logger.error(f"Error counting terms: {e}")
            return 0

    def download_all_open_access(self) -> Dict[str, Path]:
        """
        Download all open-access ontologies

        Returns:
            Dictionary of ontology_key -> file_path
        """
        logger.info("Downloading all open-access ontologies...")

        downloaded = {}

        # Ontologies that don't require special access
        open_access = [
            'HPO', 'HPO_ANNOTATIONS', 'GO', 'GO_ANNOTATIONS',
            'MONDO', 'DO', 'UBERON', 'CHEBI',
            'CLINVAR', 'ORPHANET', 'FNDDS'
        ]

        for ontology_key in open_access:
            file_path = self.download_ontology(ontology_key)
            if file_path:
                downloaded[ontology_key] = file_path

            # Rate limiting
            time.sleep(2)

        logger.info(f"Downloaded {len(downloaded)} ontologies")
        return downloaded

    def update_all(self) -> Dict[str, Path]:
        """
        Update all previously downloaded ontologies

        Returns:
            Dictionary of updated ontologies
        """
        logger.info("Updating ontologies...")

        updated = {}

        for ontology_key in self.metadata.keys():
            logger.info(f"Updating {ontology_key}...")
            file_path = self.download_ontology(ontology_key, force=True)
            if file_path:
                updated[ontology_key] = file_path
            time.sleep(2)

        return updated

    def check_versions(self) -> pd.DataFrame:
        """
        Check versions of downloaded ontologies

        Returns:
            DataFrame with version information
        """
        logger.info("Checking ontology versions...")

        versions = []

        for key, meta in self.metadata.items():
            versions.append({
                'ontology': key,
                'name': meta['name'],
                'version': meta['version'],
                'download_date': meta['download_date'],
                'file_size_mb': meta['file_size'] / (1024 * 1024),
                'terms_count': meta.get('terms_count', 'N/A'),
                'license': meta['license']
            })

        df = pd.DataFrame(versions)
        return df

    def get_adhd_autism_terms(self, ontology_key: str = 'HPO') -> List[Dict]:
        """
        Extract ADHD/autism related terms from ontology

        Args:
            ontology_key: Ontology to search

        Returns:
            List of relevant terms
        """
        if ontology_key not in self.metadata:
            logger.error(f"{ontology_key} not downloaded")
            return []

        file_path = Path(self.metadata[ontology_key]['file_path'])

        if ontology_key == 'HPO':
            return self._extract_hpo_adhd_autism_terms(file_path)
        elif ontology_key == 'GO':
            return self._extract_go_neuro_terms(file_path)
        else:
            logger.warning(f"Term extraction not implemented for {ontology_key}")
            return []

    def _extract_hpo_adhd_autism_terms(self, hpo_file: Path) -> List[Dict]:
        """Extract ADHD/autism phenotypes from HPO"""
        logger.info("Extracting ADHD/autism terms from HPO...")

        search_terms = [
            'adhd', 'attention deficit', 'hyperactivity', 'hyperkinetic',
            'autism', 'autistic', 'asperger', 'pervasive developmental',
            'intellectual disability', 'developmental delay',
            'stereotypy', 'social communication', 'restrictive behavior'
        ]

        terms = []

        try:
            current_term = {}

            with open(hpo_file) as f:
                for line in f:
                    line = line.strip()

                    if line == '[Term]':
                        if current_term:
                            # Check if relevant
                            name = current_term.get('name', '').lower()
                            if any(st in name for st in search_terms):
                                terms.append(current_term)
                        current_term = {}

                    elif line.startswith('id:'):
                        current_term['id'] = line.split(':', 1)[1].strip()
                    elif line.startswith('name:'):
                        current_term['name'] = line.split(':', 1)[1].strip()
                    elif line.startswith('def:'):
                        current_term['definition'] = line.split(':', 1)[1].strip()

            logger.info(f"Found {len(terms)} ADHD/autism related HPO terms")
            return terms

        except Exception as e:
            logger.error(f"Error extracting HPO terms: {e}")
            return []

    def _extract_go_neuro_terms(self, go_file: Path) -> List[Dict]:
        """Extract neurodevelopmental GO terms"""
        logger.info("Extracting neurodevelopmental terms from GO...")

        search_terms = [
            'neurotransmitter', 'synaptic', 'neuron', 'axon', 'dendrite',
            'dopamine', 'serotonin', 'gaba', 'glutamate',
            'brain development', 'nervous system development'
        ]

        terms = []

        try:
            current_term = {}

            with open(go_file) as f:
                for line in f:
                    line = line.strip()

                    if line == '[Term]':
                        if current_term:
                            name = current_term.get('name', '').lower()
                            if any(st in name for st in search_terms):
                                terms.append(current_term)
                        current_term = {}

                    elif line.startswith('id:'):
                        current_term['id'] = line.split(':', 1)[1].strip()
                    elif line.startswith('name:'):
                        current_term['name'] = line.split(':', 1)[1].strip()
                    elif line.startswith('namespace:'):
                        current_term['namespace'] = line.split(':', 1)[1].strip()

            logger.info(f"Found {len(terms)} neurodevelopmental GO terms")
            return terms

        except Exception as e:
            logger.error(f"Error extracting GO terms: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(
        description='Download biomedical ontologies and reference databases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all open-access ontologies
  python ontology_downloader.py --download-all --output data/references/

  # Download specific ontology
  python ontology_downloader.py --ontology HPO --output data/references/

  # Update all ontologies
  python ontology_downloader.py --update --output data/references/

  # Check versions
  python ontology_downloader.py --check-versions --output data/references/

  # Extract ADHD/autism terms
  python ontology_downloader.py --extract-terms HPO --output data/references/

Note: Some ontologies (SNOMED, RxNorm, OMIM) require license agreements.
      Register at https://uts.nlm.nih.gov for UMLS Terminology Services.
        """
    )

    parser.add_argument(
        '--download-all',
        action='store_true',
        help='Download all open-access ontologies'
    )

    parser.add_argument(
        '--ontology',
        type=str,
        choices=list(ONTOLOGY_SOURCES.keys()),
        help='Download specific ontology'
    )

    parser.add_argument(
        '--update',
        action='store_true',
        help='Update all previously downloaded ontologies'
    )

    parser.add_argument(
        '--check-versions',
        action='store_true',
        help='Check versions of downloaded ontologies'
    )

    parser.add_argument(
        '--extract-terms',
        type=str,
        choices=['HPO', 'GO'],
        help='Extract ADHD/autism related terms'
    )

    parser.add_argument(
        '--list-available',
        action='store_true',
        help='List all available ontologies'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if exists'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/references/ontologies',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = OntologyDownloader(Path(args.output))

    # Handle list available
    if args.list_available:
        print("\n=== Available Ontologies ===\n")
        for key, source in ONTOLOGY_SOURCES.items():
            print(f"{key}: {source['name']}")
            print(f"  Description: {source['description']}")
            print(f"  Relevance: {source.get('relevance', 'N/A')}")
            print(f"  License: {source['license']}")
            if 'note' in source:
                print(f"  Note: {source['note']}")
            print()
        return

    # Handle check versions
    if args.check_versions:
        versions_df = downloader.check_versions()
        if not versions_df.empty:
            print("\n=== Downloaded Ontologies ===\n")
            print(versions_df.to_string(index=False))
        else:
            print("\nNo ontologies downloaded yet")
        return

    # Handle download all
    if args.download_all:
        downloaded = downloader.download_all_open_access()
        print(f"\n=== Downloaded {len(downloaded)} ontologies ===\n")
        for key, path in downloaded.items():
            print(f"{key}: {path}")
        return

    # Handle single ontology download
    if args.ontology:
        file_path = downloader.download_ontology(args.ontology, force=args.force)
        if file_path:
            print(f"\nDownloaded {args.ontology}: {file_path}")
        return

    # Handle update
    if args.update:
        updated = downloader.update_all()
        print(f"\n=== Updated {len(updated)} ontologies ===\n")
        for key, path in updated.items():
            print(f"{key}: {path}")
        return

    # Handle term extraction
    if args.extract_terms:
        terms = downloader.get_adhd_autism_terms(args.extract_terms)
        if terms:
            print(f"\n=== {len(terms)} ADHD/Autism Related Terms ===\n")
            for term in terms[:20]:  # Show first 20
                print(f"{term['id']}: {term['name']}")
            print(f"\n... and {len(terms) - 20} more")

            # Save to file
            output_file = downloader.output_dir / f'{args.extract_terms.lower()}_adhd_autism_terms.json'
            with open(output_file, 'w') as f:
                json.dump(terms, f, indent=2)
            print(f"\nTerms saved: {output_file}")
        return

    parser.print_help()


if __name__ == '__main__':
    main()