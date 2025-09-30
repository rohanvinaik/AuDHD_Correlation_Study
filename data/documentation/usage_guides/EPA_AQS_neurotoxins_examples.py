#!/usr/bin/env python3
"""
Sample Usage Examples for EPA AQS Neurotoxins

This script demonstrates how to load and work with the EPA AQS Neurotoxins dataset.

Author: AuDHD Correlation Study Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Dataset paths
DATASET_DIR = Path('data/raw/EPA_AQS_neurotoxins')
DATA_FILE = DATASET_DIR / 'data' / 'EPA_AQS_neurotoxins_data.csv'  # Adjust filename as needed
DICT_FILE = Path('data/documentation/data_dictionaries/EPA_AQS_neurotoxins_dictionary.json')


def load_dataset():
    """Load the EPA AQS Neurotoxins dataset"""

    print(f"Loading {metadata.name}...")

    # Read data
    df = pd.read_csv(DATA_FILE)

    print(f"Loaded {len(df):,} records")
    print(f"Variables: {len(df.columns)}")

    return df


def load_data_dictionary():
    """Load variable descriptions from data dictionary"""

    with open(DICT_FILE) as f:
        data_dict = json.load(f)

    return data_dict


def explore_dataset(df):
    """Explore dataset structure and contents"""

    print("\n=== Dataset Overview ===")
    print(df.info())

    print("\n=== First Few Records ===")
    print(df.head())

    print("\n=== Summary Statistics ===")
    print(df.describe())

    print("\n=== Missing Data ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': missing_pct
    })
    print(missing_df[missing_df['missing_count'] > 0].sort_values('missing_count', ascending=False))


def filter_complete_cases(df):
    """Filter to complete cases only (no missing data)"""

    complete = df.dropna()
    print(f"\nComplete cases: {len(complete):,} / {len(df):,} ({len(complete)/len(df)*100:.1f}%)")

    return complete


def basic_analysis(df):
    """Perform basic exploratory analysis"""

    print("\n=== Basic Analysis ===")

    # Example analyses (customize based on actual variables)

    # Numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric variables ({len(numeric_cols)}):")
        for col in numeric_cols[:5]:  # Show first 5
            print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")

    # Categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\nCategorical variables ({len(categorical_cols)}):")
        for col in categorical_cols[:5]:  # Show first 5
            print(f"  {col}: {df[col].nunique()} unique values")


def save_processed_data(df, output_file='processed_data.csv'):
    """Save processed dataset"""

    output_path = DATASET_DIR / 'processed' / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")


def main():
    """Main analysis pipeline"""

    # Load dataset
    df = load_dataset()

    # Load data dictionary
    data_dict = load_data_dictionary()
    print(f"\nData dictionary: {data_dict['total_variables']} variables documented")

    # Explore dataset
    explore_dataset(df)

    # Filter to complete cases
    df_complete = filter_complete_cases(df)

    # Basic analysis
    basic_analysis(df_complete)

    # Save processed data
    save_processed_data(df_complete)

    print("\n=== Analysis Complete ===")


if __name__ == '__main__':
    main()
