#!/usr/bin/env python3
"""
GWAS Catalog Extractor for ADHD/Autism Research

Extracts GWAS associations from NHGRI-EBI GWAS Catalog using their REST API
and downloads summary statistics from PGC (Psychiatric Genomics Consortium).

GWAS Catalog contains:
- 500,000+ SNP-trait associations
- 6,000+ publications
- Genome-wide significant associations (p < 5e-8)
- Standardized trait ontology (EFO)

PGC provides:
- ADHD summary statistics (55,374 cases, 202,788 controls)
- Autism summary statistics (18,381 cases, 27,969 controls)
- Cross-disorder analysis

Requirements:
    pip install requests pandas

Usage:
    # Extract ADHD associations
    python gwas_catalog_extractor.py --trait ADHD --output data/genetics/

    # Extract autism associations
    python gwas_catalog_extractor.py --trait autism --output data/genetics/

    # Download PGC summary stats
    python gwas_catalog_extractor.py --pgc-download --output data/genetics/

    # Get all associations for gene
    python gwas_catalog_extractor.py --gene DRD4 --output data/genetics/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import logging

try:
    import requests
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install requests pandas")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# GWAS Catalog API
GWAS_API_BASE = "https://www.ebi.ac.uk/gwas/rest/api"

# PGC download URLs
PGC_DOWNLOADS = {
    'adhd': {
        'url': 'https://figshare.com/ndownloader/files/28169253',
        'filename': 'adhd_eur_jun2017.gz',
        'description': 'ADHD GWAS (Demontis et al. 2019)',
        'cases': 20183,
        'controls': 35191,
        'snps': 8047421,
        'citation': 'Nat Genet. 2019 Jan;51(1):63-75'
    },
    'autism': {
        'url': 'https://figshare.com/ndownloader/files/28169256',
        'filename': 'iPSYCH-PGC_ASD_Nov2017.gz',
        'description': 'Autism GWAS (Grove et al. 2019)',
        'cases': 18381,
        'controls': 27969,
        'snps': 9112386,
        'citation': 'Nat Genet. 2019 Mar;51(3):431-444'
    },
    'cross_disorder': {
        'url': 'https://figshare.com/ndownloader/files/28169274',
        'filename': 'Cross-Disorder_2013.tsv.gz',
        'description': 'Cross-Disorder GWAS (PGC 2013)',
        'disorders': ['ADHD', 'ASD', 'BIP', 'MDD', 'SCZ'],
        'cases': 33332,
        'controls': 27888,
        'citation': 'Lancet. 2013 Apr 20;381(9875):1371-9'
    }
}

# EFO trait mappings
EFO_TRAITS = {
    'adhd': [
        'attention deficit hyperactivity disorder',
        'ADHD',
        'hyperactivity'
    ],
    'autism': [
        'autism spectrum disorder',
        'autism',
        'autistic disorder',
        'Asperger syndrome'
    ]
}


class GWASCatalogExtractor:
    """Extract GWAS associations from NHGRI-EBI GWAS Catalog"""

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

        logger.info(f"Initialized GWAS Catalog extractor: {output_dir}")

    def search_associations(self, trait: str,
                          p_value_threshold: float = 5e-8) -> pd.DataFrame:
        """
        Search GWAS Catalog for trait associations

        Args:
            trait: Trait name (e.g., 'ADHD', 'autism')
            p_value_threshold: P-value threshold (default: 5e-8)

        Returns:
            DataFrame with associations
        """
        logger.info(f"Searching GWAS Catalog for: {trait}")

        all_associations = []

        # Get EFO trait terms
        trait_terms = EFO_TRAITS.get(trait.lower(), [trait])

        for term in trait_terms:
            try:
                # Search associations
                url = f"{GWAS_API_BASE}/efoTraits/{term}/associations"
                response = self.session.get(url, timeout=60)

                if response.status_code != 200:
                    logger.warning(f"Failed to fetch associations for {term}")
                    continue

                data = response.json()

                # Extract associations
                if '_embedded' in data and 'associations' in data['_embedded']:
                    associations = data['_embedded']['associations']

                    for assoc in associations:
                        # Filter by p-value
                        p_value = assoc.get('pvalue')
                        if p_value and float(p_value) <= p_value_threshold:
                            all_associations.append({
                                'rsid': assoc.get('strongestAllele', '').split('-')[0],
                                'chr': assoc.get('locations', [{}])[0].get('chromosomeName', ''),
                                'position': assoc.get('locations', [{}])[0].get('chromosomePosition', ''),
                                'risk_allele': assoc.get('strongestAllele', ''),
                                'p_value': float(p_value),
                                'or_beta': assoc.get('orPerCopyNum', ''),
                                'ci': assoc.get('range', ''),
                                'trait': assoc.get('efoTraits', [{}])[0].get('trait', ''),
                                'study': assoc.get('study', {}).get('publicationInfo', {}).get('title', ''),
                                'pubmed_id': assoc.get('study', {}).get('publicationInfo', {}).get('pubmedId', ''),
                                'sample_size': assoc.get('study', {}).get('initialSampleSize', ''),
                                'mapped_genes': ', '.join([g.get('geneName', '')
                                                          for g in assoc.get('loci', [{}])[0].get('authorReportedGenes', [])])
                            })

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Error fetching associations for {term}: {e}")

        df = pd.DataFrame(all_associations)

        if not df.empty:
            # Remove duplicates
            df = df.drop_duplicates(subset=['rsid', 'p_value'])

            # Sort by p-value
            df = df.sort_values('p_value')

        logger.info(f"Found {len(df)} associations for {trait}")
        return df

    def get_gene_associations(self, gene: str) -> pd.DataFrame:
        """
        Get all associations for a specific gene

        Args:
            gene: Gene symbol (e.g., 'DRD4')

        Returns:
            DataFrame with associations
        """
        logger.info(f"Fetching associations for gene: {gene}")

        try:
            url = f"{GWAS_API_BASE}/singleNucleotidePolymorphisms/search/findByGene?geneName={gene}"
            response = self.session.get(url, timeout=30)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch gene associations")
                return pd.DataFrame()

            data = response.json()

            associations = []

            if '_embedded' in data:
                snps = data['_embedded'].get('singleNucleotidePolymorphisms', [])

                for snp in snps:
                    associations.append({
                        'rsid': snp.get('rsId', ''),
                        'gene': gene,
                        'chr': snp.get('locations', [{}])[0].get('chromosomeName', ''),
                        'position': snp.get('locations', [{}])[0].get('chromosomePosition', ''),
                        'merged': snp.get('merged', False)
                    })

            df = pd.DataFrame(associations)
            logger.info(f"Found {len(df)} SNPs for gene {gene}")
            return df

        except Exception as e:
            logger.error(f"Error fetching gene associations: {e}")
            return pd.DataFrame()

    def download_pgc_summary_stats(self, dataset: str = 'adhd') -> Optional[Path]:
        """
        Download PGC summary statistics

        Args:
            dataset: Dataset name ('adhd', 'autism', 'cross_disorder')

        Returns:
            Path to downloaded file
        """
        if dataset not in PGC_DOWNLOADS:
            logger.error(f"Unknown PGC dataset: {dataset}")
            return None

        pgc_info = PGC_DOWNLOADS[dataset]
        url = pgc_info['url']
        filename = pgc_info['filename']

        output_file = self.output_dir / filename

        if output_file.exists():
            logger.info(f"Using cached file: {output_file}")
            return output_file

        logger.info(f"Downloading {pgc_info['description']}...")

        try:
            response = self.session.get(url, stream=True, timeout=300)

            if response.status_code != 200:
                logger.error(f"Failed to download: HTTP {response.status_code}")
                return None

            # Get file size
            total_size = int(response.headers.get('content-length', 0))

            # Download with progress
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
            return output_file

        except Exception as e:
            logger.error(f"Error downloading PGC data: {e}")
            return None

    def get_pgc_info(self) -> pd.DataFrame:
        """
        Get information about available PGC datasets

        Returns:
            DataFrame with PGC dataset info
        """
        info = []

        for dataset_id, pgc_info in PGC_DOWNLOADS.items():
            info.append({
                'dataset': dataset_id,
                'description': pgc_info['description'],
                'cases': pgc_info.get('cases', 'N/A'),
                'controls': pgc_info.get('controls', 'N/A'),
                'snps': pgc_info.get('snps', 'N/A'),
                'disorders': ', '.join(pgc_info.get('disorders', [])) if 'disorders' in pgc_info else dataset_id.upper(),
                'citation': pgc_info['citation']
            })

        return pd.DataFrame(info)


def main():
    parser = argparse.ArgumentParser(
        description='Extract GWAS associations and download PGC summary statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract ADHD associations
  python gwas_catalog_extractor.py --trait ADHD --output data/genetics/

  # Extract autism associations
  python gwas_catalog_extractor.py --trait autism --output data/genetics/

  # Download PGC ADHD summary stats
  python gwas_catalog_extractor.py --pgc-download adhd --output data/genetics/

  # Download all PGC summary stats
  python gwas_catalog_extractor.py --pgc-download all --output data/genetics/

  # Get associations for gene
  python gwas_catalog_extractor.py --gene DRD4 --output data/genetics/

  # List available PGC datasets
  python gwas_catalog_extractor.py --pgc-info
        """
    )

    parser.add_argument(
        '--trait',
        type=str,
        choices=['ADHD', 'autism'],
        help='Extract associations for trait'
    )

    parser.add_argument(
        '--gene',
        type=str,
        help='Get associations for specific gene'
    )

    parser.add_argument(
        '--pgc-download',
        type=str,
        choices=['adhd', 'autism', 'cross_disorder', 'all'],
        help='Download PGC summary statistics'
    )

    parser.add_argument(
        '--pgc-info',
        action='store_true',
        help='Show available PGC datasets'
    )

    parser.add_argument(
        '--p-threshold',
        type=float,
        default=5e-8,
        help='P-value threshold (default: 5e-8)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/genetics/gwas',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = GWASCatalogExtractor(Path(args.output))

    # Handle PGC info
    if args.pgc_info:
        pgc_info = extractor.get_pgc_info()
        print("\n=== Available PGC Summary Statistics ===\n")
        print(pgc_info.to_string(index=False))
        print("\nDownload with: --pgc-download [adhd|autism|cross_disorder|all]")
        return

    # Handle PGC downloads
    if args.pgc_download:
        if args.pgc_download == 'all':
            datasets = ['adhd', 'autism', 'cross_disorder']
        else:
            datasets = [args.pgc_download]

        for dataset in datasets:
            output_file = extractor.download_pgc_summary_stats(dataset)
            if output_file:
                print(f"\nDownloaded {dataset}: {output_file}")

                # Show info
                pgc_info = PGC_DOWNLOADS[dataset]
                print(f"  Description: {pgc_info['description']}")
                print(f"  Cases: {pgc_info.get('cases', 'N/A')}")
                print(f"  Controls: {pgc_info.get('controls', 'N/A')}")
                print(f"  SNPs: {pgc_info.get('snps', 'N/A')}")
                print(f"  Citation: {pgc_info['citation']}")

        return

    # Handle trait associations
    if args.trait:
        associations_df = extractor.search_associations(
            args.trait,
            p_value_threshold=args.p_threshold
        )

        if not associations_df.empty:
            # Save associations
            output_file = extractor.output_dir / f'{args.trait.lower()}_gwas_associations.csv'
            associations_df.to_csv(output_file, index=False)
            print(f"\nAssociations saved: {output_file}")

            # Print summary
            print(f"\n=== {args.trait} GWAS Associations ===\n")
            print(f"Total associations: {len(associations_df)}")
            print(f"Unique SNPs: {associations_df['rsid'].nunique()}")
            print(f"Unique studies: {associations_df['pubmed_id'].nunique()}")

            # Top hits
            print("\nTop 10 associations:")
            print(associations_df[['rsid', 'chr', 'position', 'p_value',
                                  'mapped_genes']].head(10).to_string(index=False))
        else:
            print(f"\nNo associations found for {args.trait}")

    # Handle gene query
    if args.gene:
        gene_df = extractor.get_gene_associations(args.gene)

        if not gene_df.empty:
            output_file = extractor.output_dir / f'{args.gene}_associations.csv'
            gene_df.to_csv(output_file, index=False)
            print(f"\nGene associations saved: {output_file}")

            print(f"\n=== {args.gene} Associations ===\n")
            print(gene_df.to_string(index=False))
        else:
            print(f"\nNo associations found for gene {args.gene}")

    if not any([args.trait, args.gene, args.pgc_download, args.pgc_info]):
        parser.print_help()


if __name__ == '__main__':
    main()