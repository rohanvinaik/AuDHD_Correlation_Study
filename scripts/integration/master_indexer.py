#!/usr/bin/env python3
"""
Master Sample Indexer

Builds and maintains a comprehensive registry of all samples across
ADHD/autism datasets with cross-dataset tracking and data availability.

Features:
- Cross-dataset sample deduplication
- Data availability tracking per sample
- Quality control flags
- Access status tracking
- SQLite database for efficient queries

Requirements:
    pip install pandas sqlite3

Usage:
    # Build master index from all datasets
    python master_indexer.py \\
        --build \\
        --output data/index/

    # Update existing index with new data
    python master_indexer.py \\
        --update \\
        --dataset genomics \\
        --input data/genetics/samples.csv \\
        --output data/index/

    # Query index
    python master_indexer.py \\
        --query \\
        --filter "genomics=True AND diagnosis=ASD" \\
        --output data/index/query_results.csv

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import sys
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import hashlib

try:
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install pandas")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SampleRecord:
    """Master sample record"""
    sample_id: str
    master_id: str  # Unified ID across datasets
    dataset_sources: List[str]

    # Demographics
    age: Optional[int]
    sex: Optional[str]
    ethnicity: Optional[str]
    ancestry: Optional[str]

    # Diagnoses
    primary_diagnosis: Optional[str]
    diagnoses: List[str]

    # Data availability
    has_genomics: bool
    has_metabolomics: bool
    has_microbiome: bool
    has_clinical: bool
    has_imaging: bool
    has_environmental: bool

    # Data quality
    completeness_score: float
    qc_flags: List[str]

    # Access
    access_status: str  # available, pending, restricted
    consent_restrictions: List[str]

    # Provenance
    first_seen: str
    last_updated: str
    data_sources: Dict[str, str]  # dataset_type -> source_id


class MasterIndexer:
    """Build and maintain master sample registry"""

    def __init__(self, db_path: Path):
        """
        Initialize indexer

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self._initialize_database()

        logger.info(f"Initialized master indexer: {db_path}")

    def _initialize_database(self):
        """Create database schema"""
        cursor = self.conn.cursor()

        # Main samples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                master_id TEXT PRIMARY KEY,
                sample_id TEXT,
                age INTEGER,
                sex TEXT,
                ethnicity TEXT,
                ancestry TEXT,
                primary_diagnosis TEXT,
                has_genomics INTEGER,
                has_metabolomics INTEGER,
                has_microbiome INTEGER,
                has_clinical INTEGER,
                has_imaging INTEGER,
                has_environmental INTEGER,
                completeness_score REAL,
                access_status TEXT,
                first_seen TEXT,
                last_updated TEXT
            )
        """)

        # Dataset sources table (many-to-many)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sample_datasets (
                master_id TEXT,
                dataset TEXT,
                source_id TEXT,
                date_added TEXT,
                PRIMARY KEY (master_id, dataset),
                FOREIGN KEY (master_id) REFERENCES samples(master_id)
            )
        """)

        # Diagnoses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sample_diagnoses (
                master_id TEXT,
                diagnosis TEXT,
                diagnosis_date TEXT,
                PRIMARY KEY (master_id, diagnosis),
                FOREIGN KEY (master_id) REFERENCES samples(master_id)
            )
        """)

        # QC flags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sample_qc (
                master_id TEXT,
                flag_type TEXT,
                flag_description TEXT,
                date_flagged TEXT,
                PRIMARY KEY (master_id, flag_type),
                FOREIGN KEY (master_id) REFERENCES samples(master_id)
            )
        """)

        # Consent restrictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sample_consent (
                master_id TEXT,
                restriction_type TEXT,
                restriction_description TEXT,
                PRIMARY KEY (master_id, restriction_type),
                FOREIGN KEY (master_id) REFERENCES samples(master_id)
            )
        """)

        # ID mapping table (for cross-dataset linkage)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS id_mappings (
                dataset TEXT,
                dataset_id TEXT,
                master_id TEXT,
                confidence REAL,
                mapping_method TEXT,
                date_mapped TEXT,
                PRIMARY KEY (dataset, dataset_id),
                FOREIGN KEY (master_id) REFERENCES samples(master_id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis ON samples(primary_diagnosis)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_genomics ON samples(has_genomics)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_access ON samples(access_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset ON sample_datasets(dataset)")

        self.conn.commit()
        logger.info("Database schema initialized")

    def generate_master_id(self, sample_info: Dict) -> str:
        """
        Generate unique master ID for a sample

        Args:
            sample_info: Dictionary with sample information

        Returns:
            Master ID string
        """
        # Use hash of key identifiers
        key_fields = [
            str(sample_info.get('sample_id', '')),
            str(sample_info.get('sex', '')),
            str(sample_info.get('age', '')),
            str(sample_info.get('dataset', ''))
        ]

        hash_input = '|'.join(key_fields).encode('utf-8')
        hash_value = hashlib.sha256(hash_input).hexdigest()[:16]

        return f"MASTER_{hash_value}"

    def add_sample(self, sample_record: SampleRecord) -> bool:
        """
        Add or update sample in registry

        Args:
            sample_record: SampleRecord to add

        Returns:
            True if successful
        """
        cursor = self.conn.cursor()

        try:
            # Insert or update main record
            cursor.execute("""
                INSERT OR REPLACE INTO samples (
                    master_id, sample_id, age, sex, ethnicity, ancestry,
                    primary_diagnosis, has_genomics, has_metabolomics,
                    has_microbiome, has_clinical, has_imaging, has_environmental,
                    completeness_score, access_status, first_seen, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample_record.master_id,
                sample_record.sample_id,
                sample_record.age,
                sample_record.sex,
                sample_record.ethnicity,
                sample_record.ancestry,
                sample_record.primary_diagnosis,
                int(sample_record.has_genomics),
                int(sample_record.has_metabolomics),
                int(sample_record.has_microbiome),
                int(sample_record.has_clinical),
                int(sample_record.has_imaging),
                int(sample_record.has_environmental),
                sample_record.completeness_score,
                sample_record.access_status,
                sample_record.first_seen,
                sample_record.last_updated
            ))

            # Add dataset sources
            for dataset in sample_record.dataset_sources:
                source_id = sample_record.data_sources.get(dataset, '')
                cursor.execute("""
                    INSERT OR REPLACE INTO sample_datasets
                    (master_id, dataset, source_id, date_added)
                    VALUES (?, ?, ?, ?)
                """, (sample_record.master_id, dataset, source_id, datetime.now().isoformat()))

            # Add diagnoses
            for diagnosis in sample_record.diagnoses:
                cursor.execute("""
                    INSERT OR IGNORE INTO sample_diagnoses
                    (master_id, diagnosis, diagnosis_date)
                    VALUES (?, ?, ?)
                """, (sample_record.master_id, diagnosis, datetime.now().isoformat()))

            # Add QC flags
            for flag in sample_record.qc_flags:
                cursor.execute("""
                    INSERT OR IGNORE INTO sample_qc
                    (master_id, flag_type, flag_description, date_flagged)
                    VALUES (?, ?, ?, ?)
                """, (sample_record.master_id, flag, flag, datetime.now().isoformat()))

            # Add consent restrictions
            for restriction in sample_record.consent_restrictions:
                cursor.execute("""
                    INSERT OR IGNORE INTO sample_consent
                    (master_id, restriction_type, restriction_description)
                    VALUES (?, ?, ?)
                """, (sample_record.master_id, restriction, restriction))

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error adding sample {sample_record.master_id}: {e}")
            self.conn.rollback()
            return False

    def import_dataset(
        self,
        dataset_name: str,
        data_df: pd.DataFrame,
        id_column: str,
        data_type: str,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Import samples from a dataset

        Args:
            dataset_name: Name of dataset
            data_df: DataFrame with sample data
            id_column: Column with sample IDs
            data_type: Type of data (genomics, metabolomics, etc.)
            column_mapping: Map dataset columns to standard fields

        Returns:
            Number of samples added
        """
        logger.info(f"Importing {len(data_df)} samples from {dataset_name}...")

        count = 0

        for idx, row in data_df.iterrows():
            try:
                # Generate master ID
                sample_info = {
                    'sample_id': row[id_column],
                    'dataset': dataset_name,
                    'sex': row.get('sex') if column_mapping is None else row.get(column_mapping.get('sex', 'sex')),
                    'age': row.get('age') if column_mapping is None else row.get(column_mapping.get('age', 'age'))
                }

                master_id = self.generate_master_id(sample_info)

                # Check if sample exists
                existing = self.get_sample(master_id)

                # Build sample record
                if existing:
                    # Update existing record
                    sample = existing
                    if dataset_name not in sample.dataset_sources:
                        sample.dataset_sources.append(dataset_name)

                    # Update data availability
                    if data_type == 'genomics':
                        sample.has_genomics = True
                    elif data_type == 'metabolomics':
                        sample.has_metabolomics = True
                    elif data_type == 'microbiome':
                        sample.has_microbiome = True
                    elif data_type == 'clinical':
                        sample.has_clinical = True
                    elif data_type == 'imaging':
                        sample.has_imaging = True
                    elif data_type == 'environmental':
                        sample.has_environmental = True

                    sample.last_updated = datetime.now().isoformat()

                else:
                    # Create new record
                    sample = SampleRecord(
                        sample_id=str(row[id_column]),
                        master_id=master_id,
                        dataset_sources=[dataset_name],
                        age=row.get('age') if column_mapping is None else row.get(column_mapping.get('age')),
                        sex=row.get('sex') if column_mapping is None else row.get(column_mapping.get('sex')),
                        ethnicity=row.get('ethnicity') if column_mapping is None else row.get(column_mapping.get('ethnicity')),
                        ancestry=row.get('ancestry') if column_mapping is None else row.get(column_mapping.get('ancestry')),
                        primary_diagnosis=row.get('diagnosis') if column_mapping is None else row.get(column_mapping.get('diagnosis')),
                        diagnoses=[row.get('diagnosis')] if row.get('diagnosis') else [],
                        has_genomics=(data_type == 'genomics'),
                        has_metabolomics=(data_type == 'metabolomics'),
                        has_microbiome=(data_type == 'microbiome'),
                        has_clinical=(data_type == 'clinical'),
                        has_imaging=(data_type == 'imaging'),
                        has_environmental=(data_type == 'environmental'),
                        completeness_score=0.0,
                        qc_flags=[],
                        access_status='available',
                        consent_restrictions=[],
                        first_seen=datetime.now().isoformat(),
                        last_updated=datetime.now().isoformat(),
                        data_sources={data_type: str(row[id_column])}
                    )

                # Add to database
                if self.add_sample(sample):
                    count += 1

                if count % 100 == 0:
                    logger.info(f"Imported {count}/{len(data_df)} samples...")

            except Exception as e:
                logger.error(f"Error importing sample {idx}: {e}")
                continue

        logger.info(f"Successfully imported {count} samples from {dataset_name}")
        return count

    def get_sample(self, master_id: str) -> Optional[SampleRecord]:
        """
        Retrieve sample record

        Args:
            master_id: Master ID

        Returns:
            SampleRecord or None
        """
        cursor = self.conn.cursor()

        # Get main record
        cursor.execute("SELECT * FROM samples WHERE master_id = ?", (master_id,))
        row = cursor.fetchone()

        if not row:
            return None

        # Get dataset sources
        cursor.execute("SELECT dataset FROM sample_datasets WHERE master_id = ?", (master_id,))
        datasets = [r[0] for r in cursor.fetchall()]

        # Get diagnoses
        cursor.execute("SELECT diagnosis FROM sample_diagnoses WHERE master_id = ?", (master_id,))
        diagnoses = [r[0] for r in cursor.fetchall()]

        # Get QC flags
        cursor.execute("SELECT flag_type FROM sample_qc WHERE master_id = ?", (master_id,))
        qc_flags = [r[0] for r in cursor.fetchall()]

        # Get consent restrictions
        cursor.execute("SELECT restriction_type FROM sample_consent WHERE master_id = ?", (master_id,))
        restrictions = [r[0] for r in cursor.fetchall()]

        # Get data sources
        cursor.execute("SELECT dataset, source_id FROM sample_datasets WHERE master_id = ?", (master_id,))
        data_sources = {r[0]: r[1] for r in cursor.fetchall()}

        # Build SampleRecord
        sample = SampleRecord(
            sample_id=row[1],
            master_id=row[0],
            dataset_sources=datasets,
            age=row[2],
            sex=row[3],
            ethnicity=row[4],
            ancestry=row[5],
            primary_diagnosis=row[6],
            diagnoses=diagnoses,
            has_genomics=bool(row[7]),
            has_metabolomics=bool(row[8]),
            has_microbiome=bool(row[9]),
            has_clinical=bool(row[10]),
            has_imaging=bool(row[11]),
            has_environmental=bool(row[12]),
            completeness_score=row[13],
            qc_flags=qc_flags,
            access_status=row[14],
            consent_restrictions=restrictions,
            first_seen=row[15],
            last_updated=row[16],
            data_sources=data_sources
        )

        return sample

    def query_samples(
        self,
        diagnosis: Optional[str] = None,
        has_genomics: Optional[bool] = None,
        has_metabolomics: Optional[bool] = None,
        has_microbiome: Optional[bool] = None,
        min_completeness: Optional[float] = None,
        access_status: Optional[str] = None
    ) -> List[SampleRecord]:
        """
        Query samples with filters

        Args:
            diagnosis: Filter by diagnosis
            has_genomics: Filter by genomics availability
            has_metabolomics: Filter by metabolomics availability
            has_microbiome: Filter by microbiome availability
            min_completeness: Minimum completeness score
            access_status: Filter by access status

        Returns:
            List of matching SampleRecords
        """
        cursor = self.conn.cursor()

        # Build query
        query = "SELECT master_id FROM samples WHERE 1=1"
        params = []

        if diagnosis:
            query += " AND primary_diagnosis = ?"
            params.append(diagnosis)

        if has_genomics is not None:
            query += " AND has_genomics = ?"
            params.append(int(has_genomics))

        if has_metabolomics is not None:
            query += " AND has_metabolomics = ?"
            params.append(int(has_metabolomics))

        if has_microbiome is not None:
            query += " AND has_microbiome = ?"
            params.append(int(has_microbiome))

        if min_completeness is not None:
            query += " AND completeness_score >= ?"
            params.append(min_completeness)

        if access_status:
            query += " AND access_status = ?"
            params.append(access_status)

        cursor.execute(query, params)

        # Fetch samples
        samples = []
        for row in cursor.fetchall():
            sample = self.get_sample(row[0])
            if sample:
                samples.append(sample)

        logger.info(f"Query returned {len(samples)} samples")
        return samples

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for registry"""
        cursor = self.conn.cursor()

        stats = {}

        # Total samples
        cursor.execute("SELECT COUNT(*) FROM samples")
        stats['total_samples'] = cursor.fetchone()[0]

        # By diagnosis
        cursor.execute("""
            SELECT primary_diagnosis, COUNT(*)
            FROM samples
            GROUP BY primary_diagnosis
        """)
        stats['by_diagnosis'] = dict(cursor.fetchall())

        # By data type
        cursor.execute("SELECT COUNT(*) FROM samples WHERE has_genomics = 1")
        stats['has_genomics'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM samples WHERE has_metabolomics = 1")
        stats['has_metabolomics'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM samples WHERE has_microbiome = 1")
        stats['has_microbiome'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM samples WHERE has_clinical = 1")
        stats['has_clinical'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM samples WHERE has_imaging = 1")
        stats['has_imaging'] = cursor.fetchone()[0]

        # Multi-omics samples
        cursor.execute("""
            SELECT COUNT(*) FROM samples
            WHERE has_genomics = 1 AND has_metabolomics = 1
        """)
        stats['genomics_metabolomics'] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM samples
            WHERE has_genomics = 1 AND has_microbiome = 1
        """)
        stats['genomics_microbiome'] = cursor.fetchone()[0]

        # By access status
        cursor.execute("""
            SELECT access_status, COUNT(*)
            FROM samples
            GROUP BY access_status
        """)
        stats['by_access'] = dict(cursor.fetchall())

        # Average completeness
        cursor.execute("SELECT AVG(completeness_score) FROM samples")
        stats['avg_completeness'] = cursor.fetchone()[0]

        return stats

    def export_to_csv(self, output_file: Path) -> Path:
        """Export master index to CSV"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM samples ORDER BY master_id
        """)

        columns = [
            'master_id', 'sample_id', 'age', 'sex', 'ethnicity', 'ancestry',
            'primary_diagnosis', 'has_genomics', 'has_metabolomics',
            'has_microbiome', 'has_clinical', 'has_imaging', 'has_environmental',
            'completeness_score', 'access_status', 'first_seen', 'last_updated'
        ]

        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=columns)

        df.to_csv(output_file, index=False)
        logger.info(f"Exported {len(df)} samples to {output_file}")

        return output_file

    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Build and maintain master sample registry',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--build',
        action='store_true',
        help='Build master index from scratch'
    )

    parser.add_argument(
        '--update',
        action='store_true',
        help='Update existing index'
    )

    parser.add_argument(
        '--query',
        action='store_true',
        help='Query index'
    )

    parser.add_argument(
        '--dataset',
        help='Dataset name for import'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Input file for import'
    )

    parser.add_argument(
        '--data-type',
        choices=['genomics', 'metabolomics', 'microbiome', 'clinical', 'imaging', 'environmental'],
        help='Type of data'
    )

    parser.add_argument(
        '--diagnosis',
        help='Filter by diagnosis'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/index',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize indexer
    db_path = Path(args.output) / 'master_sample_registry.db'
    indexer = MasterIndexer(db_path)

    if args.build or args.update:
        if args.input and args.dataset and args.data_type:
            # Import dataset
            df = pd.read_csv(args.input)
            indexer.import_dataset(
                dataset_name=args.dataset,
                data_df=df,
                id_column='sample_id',
                data_type=args.data_type
            )
        else:
            print("Error: --input, --dataset, and --data-type required for import")
            sys.exit(1)

    elif args.query:
        # Query samples
        samples = indexer.query_samples(diagnosis=args.diagnosis)

        # Export results
        output_file = Path(args.output) / 'query_results.csv'
        results_df = pd.DataFrame([asdict(s) for s in samples])
        results_df.to_csv(output_file, index=False)

        print(f"Query returned {len(samples)} samples")
        print(f"Results saved to {output_file}")

    # Print summary
    print("\n=== Master Index Summary ===\n")
    stats = indexer.get_summary_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Export to CSV
    csv_file = Path(args.output) / 'master_sample_registry.csv'
    indexer.export_to_csv(csv_file)

    indexer.close()


if __name__ == '__main__':
    main()