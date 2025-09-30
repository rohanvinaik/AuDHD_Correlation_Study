#!/usr/bin/env python3
"""
Master Dataset Catalog Builder

Creates and maintains a master catalog of all datasets in the study.

Usage:
    python catalog_builder.py --build
    python catalog_builder.py --search "ADHD"
    python catalog_builder.py --stats

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatalogBuilder:
    """Build and query master dataset catalog"""

    def __init__(self, catalog_dir: Path = Path('data/catalogs')):
        self.catalog_dir = catalog_dir
        self.catalog_dir.mkdir(parents=True, exist_ok=True)

        self.catalog_file = catalog_dir / 'master_catalog.json'
        self.db_file = catalog_dir / 'catalog.db'

        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for catalog"""

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()

            # Datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    full_name TEXT,
                    description TEXT,
                    data_type TEXT,
                    source TEXT,
                    url TEXT,
                    version TEXT,
                    release_date TEXT,
                    last_updated TEXT,
                    sample_size INTEGER,
                    variables INTEGER,
                    size_bytes INTEGER,
                    access_type TEXT,
                    license TEXT,
                    citation TEXT,
                    doi TEXT,
                    documentation_path TEXT,
                    data_path TEXT,
                    added_to_catalog TEXT
                )
            """)

            # File formats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_formats (
                    dataset_id TEXT,
                    format TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
                )
            """)

            # Keywords table for search
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS keywords (
                    dataset_id TEXT,
                    keyword TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
                )
            """)

            conn.commit()

    def add_dataset(self, dataset_info: Dict):
        """Add dataset to catalog"""

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()

            # Insert dataset
            cursor.execute("""
                INSERT OR REPLACE INTO datasets VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                dataset_info['dataset_id'],
                dataset_info['name'],
                dataset_info.get('full_name'),
                dataset_info.get('description'),
                dataset_info.get('data_type'),
                dataset_info.get('source'),
                dataset_info.get('url'),
                dataset_info.get('version'),
                dataset_info.get('release_date'),
                dataset_info.get('last_updated'),
                dataset_info.get('sample_size'),
                dataset_info.get('variables'),
                dataset_info.get('size_bytes'),
                dataset_info.get('access_type'),
                dataset_info.get('license'),
                dataset_info.get('citation'),
                dataset_info.get('doi'),
                dataset_info.get('documentation_path'),
                dataset_info.get('data_path'),
                datetime.now().isoformat()
            ))

            # Insert file formats
            if 'file_formats' in dataset_info:
                cursor.execute("DELETE FROM file_formats WHERE dataset_id = ?",
                             (dataset_info['dataset_id'],))
                for fmt in dataset_info['file_formats']:
                    cursor.execute("INSERT INTO file_formats VALUES (?, ?)",
                                 (dataset_info['dataset_id'], fmt))

            # Insert keywords
            if 'keywords' in dataset_info:
                cursor.execute("DELETE FROM keywords WHERE dataset_id = ?",
                             (dataset_info['dataset_id'],))
                for keyword in dataset_info['keywords']:
                    cursor.execute("INSERT INTO keywords VALUES (?, ?)",
                                 (dataset_info['dataset_id'], keyword))

            conn.commit()

        logger.info(f"Added {dataset_info['dataset_id']} to catalog")

    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset info from catalog"""

        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
            row = cursor.fetchone()

            if not row:
                return None

            dataset = dict(row)

            # Get file formats
            cursor.execute("SELECT format FROM file_formats WHERE dataset_id = ?",
                         (dataset_id,))
            dataset['file_formats'] = [r[0] for r in cursor.fetchall()]

            # Get keywords
            cursor.execute("SELECT keyword FROM keywords WHERE dataset_id = ?",
                         (dataset_id,))
            dataset['keywords'] = [r[0] for r in cursor.fetchall()]

            return dataset

    def search_datasets(
        self,
        query: Optional[str] = None,
        data_type: Optional[str] = None,
        access_type: Optional[str] = None
    ) -> List[Dict]:
        """Search datasets"""

        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            sql = "SELECT DISTINCT d.* FROM datasets d"
            conditions = []
            params = []

            if query:
                sql += " LEFT JOIN keywords k ON d.dataset_id = k.dataset_id"
                conditions.append("""(
                    d.name LIKE ? OR
                    d.full_name LIKE ? OR
                    d.description LIKE ? OR
                    k.keyword LIKE ?
                )""")
                query_param = f"%{query}%"
                params.extend([query_param] * 4)

            if data_type:
                conditions.append("d.data_type = ?")
                params.append(data_type)

            if access_type:
                conditions.append("d.access_type = ?")
                params.append(access_type)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

            cursor.execute(sql, params)
            results = [dict(row) for row in cursor.fetchall()]

            return results

    def get_all_datasets(self) -> List[Dict]:
        """Get all datasets in catalog"""

        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM datasets ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]

    def get_catalog_statistics(self) -> Dict:
        """Get catalog statistics"""

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()

            stats = {}

            # Total datasets
            cursor.execute("SELECT COUNT(*) FROM datasets")
            stats['total_datasets'] = cursor.fetchone()[0]

            # By data type
            cursor.execute("""
                SELECT data_type, COUNT(*) as count
                FROM datasets
                GROUP BY data_type
            """)
            stats['by_data_type'] = {row[0]: row[1] for row in cursor.fetchall()}

            # By access type
            cursor.execute("""
                SELECT access_type, COUNT(*) as count
                FROM datasets
                GROUP BY access_type
            """)
            stats['by_access_type'] = {row[0]: row[1] for row in cursor.fetchall()}

            # Total samples
            cursor.execute("SELECT SUM(sample_size) FROM datasets WHERE sample_size IS NOT NULL")
            stats['total_samples'] = cursor.fetchone()[0] or 0

            # Total size
            cursor.execute("SELECT SUM(size_bytes) FROM datasets WHERE size_bytes IS NOT NULL")
            total_bytes = cursor.fetchone()[0] or 0
            stats['total_size_bytes'] = total_bytes
            stats['total_size_human'] = self._format_bytes(total_bytes)

            return stats

    def export_catalog(self):
        """Export catalog to JSON"""

        datasets = self.get_all_datasets()

        # Get file formats and keywords for each
        for ds in datasets:
            ds['file_formats'] = []
            ds['keywords'] = []

            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT format FROM file_formats WHERE dataset_id = ?",
                             (ds['dataset_id'],))
                ds['file_formats'] = [r[0] for r in cursor.fetchall()]

                cursor.execute("SELECT keyword FROM keywords WHERE dataset_id = ?",
                             (ds['dataset_id'],))
                ds['keywords'] = [r[0] for r in cursor.fetchall()]

        catalog = {
            'generated_date': datetime.now().isoformat(),
            'version': '1.0',
            'statistics': self.get_catalog_statistics(),
            'datasets': datasets
        }

        with open(self.catalog_file, 'w') as f:
            json.dump(catalog, f, indent=2)

        logger.info(f"Exported catalog to {self.catalog_file}")

    def _format_bytes(self, size_bytes: int) -> str:
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"


def build_catalog():
    """Build catalog from existing datasets"""

    builder = CatalogBuilder()

    # Add datasets to catalog
    datasets = [
        {
            'dataset_id': 'PGC_ADHD_GWAS',
            'name': 'PGC ADHD GWAS',
            'full_name': 'Psychiatric Genomics Consortium ADHD Genome-Wide Association Study',
            'description': 'Large-scale GWAS meta-analysis of ADHD including 20,183 individuals with ADHD and 35,191 controls.',
            'data_type': 'genomics',
            'source': 'Psychiatric Genomics Consortium',
            'url': 'https://www.med.unc.edu/pgc/download-results/',
            'version': '2019',
            'release_date': '2019-01-15',
            'last_updated': '2019-01-15',
            'sample_size': 55374,
            'variables': 10,
            'size_bytes': 524288000,
            'access_type': 'public',
            'license': 'CC BY 4.0',
            'citation': 'Demontis D, et al. (2019). Nat Genet. 51(1):63-75.',
            'doi': '10.1038/s41588-018-0269-7',
            'documentation_path': 'data/documentation/dataset_summaries/PGC_ADHD_GWAS_README.md',
            'data_path': 'data/raw/PGC_ADHD_GWAS/',
            'file_formats': ['tsv', 'gz'],
            'keywords': ['ADHD', 'GWAS', 'genetics', 'SNP', 'association study', 'psychiatric genomics']
        },
        {
            'dataset_id': 'SPARK_phenotypes',
            'name': 'SPARK Phenotypes',
            'full_name': 'SPARK Autism Phenotype Data',
            'description': 'Detailed phenotype data from SPARK including demographics, diagnostic assessments, and behavioral measures for over 50,000 individuals with autism.',
            'data_type': 'clinical',
            'source': 'SPARK (Simons Foundation)',
            'url': 'https://sparkforautism.org/portal/',
            'version': 'v4.0',
            'release_date': '2024-03-20',
            'last_updated': '2024-03-20',
            'sample_size': 50000,
            'variables': 450,
            'size_bytes': 104857600,
            'access_type': 'controlled',
            'license': 'SPARK Data Use Agreement',
            'citation': 'SPARK Consortium (2018). Neuron, 97(3), 488-493.',
            'doi': '10.1016/j.neuron.2018.01.015',
            'documentation_path': 'data/documentation/dataset_summaries/SPARK_phenotypes_README.md',
            'data_path': 'data/raw/SPARK_phenotypes/',
            'file_formats': ['csv', 'gz'],
            'keywords': ['autism', 'ASD', 'phenotype', 'clinical', 'ADOS', 'ADI-R', 'SPARK']
        },
        {
            'dataset_id': 'ABCD_microbiome',
            'name': 'ABCD Microbiome',
            'full_name': 'ABCD Study Gut Microbiome Data',
            'description': 'Gut microbiome 16S rRNA sequencing data from the ABCD Study. Includes OTU counts and diversity metrics.',
            'data_type': 'microbiome',
            'source': 'ABCD Study',
            'url': 'https://nda.nih.gov/abcd',
            'version': '5.1',
            'release_date': '2024-06-15',
            'last_updated': '2024-06-15',
            'sample_size': 5000,
            'variables': 1250,
            'size_bytes': 2147483648,
            'access_type': 'controlled',
            'license': 'ABCD Data Use Certification',
            'citation': 'Volkow ND, et al. (2018). Dev Cogn Neurosci. 32:4-7.',
            'doi': '10.1016/j.dcn.2017.10.002',
            'documentation_path': 'data/documentation/dataset_summaries/ABCD_microbiome_README.md',
            'data_path': 'data/raw/ABCD_microbiome/',
            'file_formats': ['biom', 'tsv', 'csv'],
            'keywords': ['microbiome', 'gut', '16S', 'ABCD', 'diversity', 'OTU', 'adolescent']
        },
        {
            'dataset_id': 'EPA_AQS_neurotoxins',
            'name': 'EPA AQS Neurotoxins',
            'full_name': 'EPA Air Quality System - Neurotoxic Pollutants',
            'description': 'EPA Air Quality System data for neurotoxic air pollutants including PM2.5, benzene, lead, and other criteria pollutants.',
            'data_type': 'environmental',
            'source': 'EPA Air Quality System',
            'url': 'https://aqs.epa.gov/aqsweb/documents/data_api.html',
            'version': '2024',
            'release_date': '2024-01-01',
            'last_updated': '2024-09-01',
            'sample_size': 85000,
            'variables': 25,
            'size_bytes': 52428800,
            'access_type': 'public',
            'license': 'Public Domain (US Government)',
            'citation': 'U.S. EPA. Air Quality System Data Mart.',
            'doi': None,
            'documentation_path': 'data/documentation/dataset_summaries/EPA_AQS_neurotoxins_README.md',
            'data_path': 'data/raw/EPA_AQS_neurotoxins/',
            'file_formats': ['csv'],
            'keywords': ['environmental', 'air quality', 'PM2.5', 'ozone', 'pollution', 'EPA', 'neurotoxin']
        }
    ]

    for dataset in datasets:
        builder.add_dataset(dataset)

    # Export to JSON
    builder.export_catalog()

    # Print statistics
    stats = builder.get_catalog_statistics()
    print("\n=== Catalog Statistics ===")
    print(f"Total Datasets: {stats['total_datasets']}")
    print(f"\nBy Data Type:")
    for dtype, count in stats['by_data_type'].items():
        print(f"  - {dtype}: {count}")
    print(f"\nBy Access Type:")
    for atype, count in stats['by_access_type'].items():
        print(f"  - {atype}: {count}")
    print(f"\nTotal Samples: {stats['total_samples']:,}")
    print(f"Total Size: {stats['total_size_human']}")


def main():
    parser = argparse.ArgumentParser(description='Manage dataset catalog')
    parser.add_argument('--build', action='store_true', help='Build catalog')
    parser.add_argument('--search', help='Search datasets')
    parser.add_argument('--data-type', help='Filter by data type')
    parser.add_argument('--access-type', help='Filter by access type')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--get', help='Get specific dataset by ID')

    args = parser.parse_args()

    builder = CatalogBuilder()

    if args.build:
        build_catalog()

    elif args.search or args.data_type or args.access_type:
        results = builder.search_datasets(
            query=args.search,
            data_type=args.data_type,
            access_type=args.access_type
        )
        print(f"\n=== Search Results ({len(results)} found) ===\n")
        for ds in results:
            print(f"{ds['dataset_id']}: {ds['name']} ({ds['data_type']}, {ds['access_type']})")

    elif args.stats:
        stats = builder.get_catalog_statistics()
        print("\n=== Catalog Statistics ===")
        print(f"Total Datasets: {stats['total_datasets']}")
        print(f"Total Samples: {stats['total_samples']:,}")
        print(f"Total Size: {stats['total_size_human']}")

    elif args.get:
        dataset = builder.get_dataset(args.get)
        if dataset:
            print(json.dumps(dataset, indent=2))
        else:
            print(f"Dataset {args.get} not found")


if __name__ == '__main__':
    import sys
    main()