#!/usr/bin/env python3
"""
ID Mapper for Cross-Dataset Identifier Resolution

Maps sample IDs across different datasets using fuzzy matching,
demographic matching, and manual curation.

Requirements:
    pip install pandas fuzzywuzzy python-Levenshtein

Usage:
    # Map IDs between two datasets
    python id_mapper.py \\
        --dataset1 data/genetics/samples.csv \\
        --dataset2 data/metabolomics/samples.csv \\
        --id1-column sample_id \\
        --id2-column participant_id \\
        --output data/index/id_mappings.csv

Author: AuDHD Correlation Study Team
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

try:
    import pandas as pd
    from fuzzywuzzy import fuzz
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install pandas fuzzywuzzy python-Levenshtein")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IDMapper:
    """Map IDs across datasets"""

    def __init__(self):
        self.mappings = []

    def fuzzy_match_ids(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        id1_col: str,
        id2_col: str,
        threshold: int = 80
    ) -> pd.DataFrame:
        """Fuzzy match IDs using string similarity"""
        matches = []

        for idx1, row1 in df1.iterrows():
            id1 = str(row1[id1_col])
            best_match = None
            best_score = 0

            for idx2, row2 in df2.iterrows():
                id2 = str(row2[id2_col])
                score = fuzz.ratio(id1, id2)

                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = id2

            if best_match:
                matches.append({
                    'id1': id1,
                    'id2': best_match,
                    'confidence': best_score / 100.0,
                    'method': 'fuzzy_match'
                })

        return pd.DataFrame(matches)

    def demographic_match(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        demo_cols: List[str] = ['age', 'sex']
    ) -> pd.DataFrame:
        """Match using demographic information"""
        # Join on demographics
        merged = df1.merge(df2, on=demo_cols, how='inner', suffixes=('_1', '_2'))

        matches = []
        for _, row in merged.iterrows():
            matches.append({
                'id1': row['sample_id_1'],
                'id2': row['sample_id_2'],
                'confidence': 0.9,
                'method': 'demographic_match'
            })

        return pd.DataFrame(matches)


def main():
    parser = argparse.ArgumentParser(description='Map IDs across datasets')

    parser.add_argument('--dataset1', required=True, help='First dataset')
    parser.add_argument('--dataset2', required=True, help='Second dataset')
    parser.add_argument('--id1-column', default='sample_id', help='ID column in dataset 1')
    parser.add_argument('--id2-column', default='sample_id', help='ID column in dataset 2')
    parser.add_argument('--output', required=True, help='Output mappings file')

    args = parser.parse_args()

    df1 = pd.read_csv(args.dataset1)
    df2 = pd.read_csv(args.dataset2)

    mapper = IDMapper()
    mappings = mapper.fuzzy_match_ids(df1, df2, args.id1_column, args.id2_column)

    mappings.to_csv(args.output, index=False)
    print(f"Generated {len(mappings)} ID mappings")


if __name__ == '__main__':
    main()