#!/usr/bin/env python3
"""
Data Completeness Calculator

Calculates completeness scores and generates data availability reports
for samples across all datasets.

Usage:
    python completeness_calculator.py \\
        --database data/index/master_sample_registry.db \\
        --output data/index/

Author: AuDHD Correlation Study Team
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Dict
import logging

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install pandas numpy")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompletenessCalculator:
    """Calculate data completeness metrics"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))

    def calculate_completeness_score(self, sample_data: Dict) -> float:
        """Calculate completeness score (0-1)"""
        total_fields = 6  # genomics, metabolomics, microbiome, clinical, imaging, environmental
        available_fields = sum([
            sample_data.get('has_genomics', 0),
            sample_data.get('has_metabolomics', 0),
            sample_data.get('has_microbiome', 0),
            sample_data.get('has_clinical', 0),
            sample_data.get('has_imaging', 0),
            sample_data.get('has_environmental', 0)
        ])

        return available_fields / total_fields

    def update_all_completeness_scores(self):
        """Update completeness scores for all samples"""
        cursor = self.conn.cursor()

        cursor.execute("SELECT master_id, has_genomics, has_metabolomics, has_microbiome, has_clinical, has_imaging, has_environmental FROM samples")

        for row in cursor.fetchall():
            master_id = row[0]
            score = sum(row[1:]) / 6.0

            cursor.execute("UPDATE samples SET completeness_score = ? WHERE master_id = ?", (score, master_id))

        self.conn.commit()
        logger.info("Updated completeness scores")

    def generate_availability_matrix(self, output_path: Path):
        """Generate data availability matrix CSV"""
        query = """
            SELECT master_id, primary_diagnosis,
                   has_genomics, has_metabolomics, has_microbiome,
                   has_clinical, has_imaging, has_environmental,
                   completeness_score, access_status
            FROM samples
            ORDER BY completeness_score DESC
        """

        df = pd.read_sql_query(query, self.conn)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported availability matrix: {output_path}")
        return df

    def close(self):
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Calculate data completeness')
    parser.add_argument('--database', required=True, help='Path to master registry database')
    parser.add_argument('--output', default='data/index', help='Output directory')

    args = parser.parse_args()

    calc = CompletenessCalculator(Path(args.database))
    calc.update_all_completeness_scores()

    output_file = Path(args.output) / 'data_availability_matrix.csv'
    df = calc.generate_availability_matrix(output_file)

    print(f"\n=== Completeness Summary ===")
    print(f"Mean completeness: {df['completeness_score'].mean():.2%}")
    print(f"Samples with all data types: {(df['completeness_score'] == 1.0).sum()}")
    print(f"Samples with 50%+ completeness: {(df['completeness_score'] >= 0.5).sum()}")

    calc.close()


if __name__ == '__main__':
    main()