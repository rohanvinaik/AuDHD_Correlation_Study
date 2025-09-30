#!/usr/bin/env python3
"""
Dataset Documentation Generator

Auto-generates comprehensive documentation for all acquired datasets:
- README.md with access instructions
- data_dictionary.json with variable descriptions
- quality_report.html with QC metrics
- sample_code.py with usage examples
- citations.bib with proper references

Usage:
    python dataset_documenter.py --dataset PGC_ADHD_GWAS --generate-all
    python dataset_documenter.py --generate-all-datasets

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for a dataset"""
    dataset_id: str
    name: str
    full_name: str
    description: str
    data_type: str  # genomics, clinical, microbiome, metabolomics, imaging, environmental
    source: str
    url: str
    version: Optional[str]
    release_date: Optional[str]
    last_updated: Optional[str]
    sample_size: Optional[int]
    variables: Optional[int]
    file_format: List[str]
    size_bytes: Optional[int]
    access_type: str  # public, controlled, restricted
    application_url: Optional[str]
    contact_email: Optional[str]
    license: Optional[str]
    citation: str
    doi: Optional[str]


@dataclass
class VariableMetadata:
    """Metadata for a single variable"""
    variable_name: str
    display_name: str
    description: str
    data_type: str  # numeric, categorical, binary, text, date
    unit: Optional[str]
    valid_range: Optional[Dict]
    categories: Optional[List[str]]
    missing_codes: Optional[List]
    required: bool
    primary_key: bool
    foreign_key: Optional[str]


@dataclass
class QualityMetrics:
    """Quality control metrics"""
    total_records: int
    complete_records: int
    completeness_rate: float
    missing_rate: float
    duplicate_records: int
    outliers_detected: int
    validation_errors: int
    quality_score: float  # 0-100
    qc_date: str
    qc_notes: List[str]


class DatasetDocumenter:
    """Generate comprehensive dataset documentation"""

    def __init__(self, output_dir: Path = Path('data/documentation')):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.summary_dir = output_dir / 'dataset_summaries'
        self.dict_dir = output_dir / 'data_dictionaries'
        self.quality_dir = output_dir / 'quality_reports'
        self.usage_dir = output_dir / 'usage_guides'

        for d in [self.summary_dir, self.dict_dir, self.quality_dir, self.usage_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def generate_readme(self, metadata: DatasetMetadata) -> str:
        """Generate README.md for dataset"""

        readme = f"""# {metadata.full_name}

## Dataset Information

**Dataset ID**: `{metadata.dataset_id}`
**Name**: {metadata.name}
**Type**: {metadata.data_type}
**Source**: {metadata.source}
**Version**: {metadata.version or 'N/A'}
**Release Date**: {metadata.release_date or 'N/A'}
**Last Updated**: {metadata.last_updated or datetime.now().strftime('%Y-%m-%d')}

## Description

{metadata.description}

## Dataset Characteristics

- **Sample Size**: {metadata.sample_size or 'N/A'} records
- **Variables**: {metadata.variables or 'N/A'} variables
- **File Format**: {', '.join(metadata.file_format)}
- **Size**: {self._format_bytes(metadata.size_bytes) if metadata.size_bytes else 'N/A'}

## Access Information

**Access Type**: {metadata.access_type}

"""

        if metadata.access_type == 'public':
            readme += f"""### Public Access

This dataset is publicly available and can be accessed without restrictions.

**Download URL**: {metadata.url}

"""
        elif metadata.access_type == 'controlled':
            readme += f"""### Controlled Access

This dataset requires application approval before access is granted.

**Application URL**: {metadata.application_url or 'Contact data provider'}
**Contact**: {metadata.contact_email or 'See dataset website'}

#### Application Process

1. Register for an account on the data portal
2. Submit Data Access Request with research proposal
3. Wait for Data Access Committee review (typically 4-6 weeks)
4. Sign Data Use Agreement upon approval
5. Download data using provided credentials

"""
        else:  # restricted
            readme += f"""### Restricted Access

This dataset requires direct collaboration with the data provider.

**Contact**: {metadata.contact_email or 'See dataset website'}

"""

        readme += f"""## License

{metadata.license or 'See dataset provider for license terms'}

## Citation

{metadata.citation}

"""

        if metadata.doi:
            readme += f"""**DOI**: [{metadata.doi}](https://doi.org/{metadata.doi})

"""

        readme += f"""## Data Dictionary

See `data_dictionaries/{metadata.dataset_id}_dictionary.json` for complete variable descriptions.

## Quality Report

See `quality_reports/{metadata.dataset_id}_quality.html` for quality control metrics.

## Usage Examples

See `usage_guides/{metadata.dataset_id}_examples.py` for code examples.

## File Structure

```
data/raw/{metadata.dataset_id}/
├── README.md                    # This file
├── data/                        # Raw data files
├── metadata/                    # Metadata and documentation
└── processed/                   # Processed/derived data
```

## Support

For questions or issues with this dataset:
- Check the data dictionary for variable definitions
- Review the quality report for known issues
- Contact: {metadata.contact_email or metadata.url}

---

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Documentation Version**: 1.0
"""

        return readme

    def generate_data_dictionary(
        self,
        dataset_id: str,
        variables: List[VariableMetadata]
    ) -> Dict:
        """Generate data dictionary JSON"""

        dictionary = {
            'dataset_id': dataset_id,
            'generated_date': datetime.now().isoformat(),
            'version': '1.0',
            'total_variables': len(variables),
            'variables': []
        }

        for var in variables:
            var_dict = asdict(var)
            # Add computed fields
            var_dict['is_categorical'] = var.data_type == 'categorical'
            var_dict['is_numeric'] = var.data_type == 'numeric'
            var_dict['has_missing'] = bool(var.missing_codes)

            dictionary['variables'].append(var_dict)

        # Add summary statistics
        dictionary['summary'] = {
            'numeric_vars': sum(1 for v in variables if v.data_type == 'numeric'),
            'categorical_vars': sum(1 for v in variables if v.data_type == 'categorical'),
            'binary_vars': sum(1 for v in variables if v.data_type == 'binary'),
            'text_vars': sum(1 for v in variables if v.data_type == 'text'),
            'date_vars': sum(1 for v in variables if v.data_type == 'date'),
            'required_vars': sum(1 for v in variables if v.required),
            'primary_keys': sum(1 for v in variables if v.primary_key)
        }

        return dictionary

    def generate_quality_report(
        self,
        dataset_id: str,
        metrics: QualityMetrics
    ) -> str:
        """Generate quality report HTML"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Report - {dataset_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .metric-unit {{
            font-size: 0.8em;
            color: #999;
        }}
        .quality-score {{
            font-size: 3em;
            text-align: center;
            margin: 20px 0;
        }}
        .score-excellent {{ color: #10b981; }}
        .score-good {{ color: #3b82f6; }}
        .score-fair {{ color: #f59e0b; }}
        .score-poor {{ color: #ef4444; }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }}
        .progress-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        .fill-excellent {{ background: #10b981; }}
        .fill-good {{ background: #3b82f6; }}
        .fill-fair {{ background: #f59e0b; }}
        .fill-poor {{ background: #ef4444; }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .notes-list {{
            list-style: none;
            padding: 0;
        }}
        .notes-list li {{
            padding: 10px;
            margin: 5px 0;
            background: #f9fafb;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }}
        .alert {{
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .alert-warning {{
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
        }}
        .alert-error {{
            background: #fee2e2;
            border-left: 4px solid #ef4444;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Quality Control Report</h1>
        <p><strong>Dataset:</strong> {dataset_id}</p>
        <p><strong>QC Date:</strong> {metrics.qc_date}</p>
    </div>

    <div class="section">
        <h2>Overall Quality Score</h2>
        <div class="quality-score {self._get_score_class(metrics.quality_score)}">
            {metrics.quality_score:.1f}/100
        </div>
        <div class="progress-bar">
            <div class="progress-fill {self._get_fill_class(metrics.quality_score)}"
                 style="width: {metrics.quality_score}%"></div>
        </div>
        <p style="text-align: center; margin-top: 10px; color: #666;">
            {self._get_quality_rating(metrics.quality_score)}
        </p>
    </div>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{metrics.total_records:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Complete Records</div>
            <div class="metric-value">{metrics.complete_records:,}</div>
            <div class="metric-unit">{metrics.completeness_rate:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Missing Rate</div>
            <div class="metric-value">{metrics.missing_rate:.1f}<span class="metric-unit">%</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Duplicate Records</div>
            <div class="metric-value">{metrics.duplicate_records:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Outliers Detected</div>
            <div class="metric-value">{metrics.outliers_detected:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Validation Errors</div>
            <div class="metric-value">{metrics.validation_errors:,}</div>
        </div>
    </div>

    <div class="section">
        <h2>Completeness Analysis</h2>
        <p>Completeness rate: <strong>{metrics.completeness_rate:.2f}%</strong></p>
        <div class="progress-bar">
            <div class="progress-fill fill-good" style="width: {metrics.completeness_rate}%"></div>
        </div>
        <p style="margin-top: 15px;">
            <strong>{metrics.complete_records:,}</strong> out of <strong>{metrics.total_records:,}</strong>
            records are complete with no missing values.
        </p>
    </div>

    <div class="section">
        <h2>Data Quality Issues</h2>
"""

        if metrics.duplicate_records > 0:
            html += f"""
        <div class="alert alert-warning">
            <strong>⚠ Duplicates Found:</strong> {metrics.duplicate_records:,} duplicate records detected.
            Consider deduplication before analysis.
        </div>
"""

        if metrics.outliers_detected > 0:
            html += f"""
        <div class="alert alert-warning">
            <strong>⚠ Outliers Detected:</strong> {metrics.outliers_detected:,} potential outliers identified.
            Review outliers to determine if they are errors or valid extreme values.
        </div>
"""

        if metrics.validation_errors > 0:
            html += f"""
        <div class="alert alert-error">
            <strong>✗ Validation Errors:</strong> {metrics.validation_errors:,} validation errors found.
            These should be corrected before using the dataset.
        </div>
"""

        if metrics.duplicate_records == 0 and metrics.outliers_detected == 0 and metrics.validation_errors == 0:
            html += """
        <p style="color: #10b981;">✓ No major data quality issues detected.</p>
"""

        html += """
    </div>

    <div class="section">
        <h2>QC Notes</h2>
        <ul class="notes-list">
"""

        for note in metrics.qc_notes:
            html += f"""            <li>{note}</li>\n"""

        html += """
        </ul>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ol>
"""

        # Generate recommendations based on metrics
        recommendations = []

        if metrics.missing_rate > 20:
            recommendations.append("High missing rate detected. Consider imputation or removal of sparse variables.")

        if metrics.duplicate_records > 0:
            recommendations.append(f"Remove {metrics.duplicate_records:,} duplicate records before analysis.")

        if metrics.outliers_detected > 0:
            recommendations.append("Review outliers to determine if they are data entry errors or valid extreme values.")

        if metrics.validation_errors > 0:
            recommendations.append("Correct validation errors before proceeding with analysis.")

        if metrics.completeness_rate >= 95:
            recommendations.append("Dataset has excellent completeness. Ready for analysis.")

        if not recommendations:
            recommendations.append("Dataset quality is good. No major issues to address.")

        for rec in recommendations:
            html += f"""            <li>{rec}</li>\n"""

        html += """
        </ol>
    </div>

    <div class="footer">
        <p>Generated by AuDHD Correlation Study Dataset Documentation System</p>
        <p>Report Version 1.0</p>
    </div>
</body>
</html>
"""

        return html

    def generate_sample_code(
        self,
        dataset_id: str,
        metadata: DatasetMetadata
    ) -> str:
        """Generate sample usage code"""

        code = f'''#!/usr/bin/env python3
"""
Sample Usage Examples for {metadata.name}

This script demonstrates how to load and work with the {metadata.name} dataset.

Author: AuDHD Correlation Study Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Dataset paths
DATASET_DIR = Path('data/raw/{dataset_id}')
DATA_FILE = DATASET_DIR / 'data' / '{dataset_id}_data.csv'  # Adjust filename as needed
DICT_FILE = Path('data/documentation/data_dictionaries/{dataset_id}_dictionary.json')


def load_dataset():
    """Load the {metadata.name} dataset"""

    print(f"Loading {{metadata.name}}...")

    # Read data
'''

        # Add loading code based on file format
        if 'csv' in metadata.file_format or 'tsv' in metadata.file_format:
            code += '''    df = pd.read_csv(DATA_FILE)
'''
        elif 'parquet' in metadata.file_format:
            code += '''    df = pd.read_parquet(DATA_FILE)
'''
        elif 'hdf5' in metadata.file_format or 'h5' in metadata.file_format:
            code += '''    df = pd.read_hdf(DATA_FILE, key='data')
'''
        else:
            code += '''    # Adjust loading method based on actual file format
    df = pd.read_csv(DATA_FILE)
'''

        code += f'''
    print(f"Loaded {{len(df):,}} records")
    print(f"Variables: {{len(df.columns)}}")

    return df


def load_data_dictionary():
    """Load variable descriptions from data dictionary"""

    with open(DICT_FILE) as f:
        data_dict = json.load(f)

    return data_dict


def explore_dataset(df):
    """Explore dataset structure and contents"""

    print("\\n=== Dataset Overview ===")
    print(df.info())

    print("\\n=== First Few Records ===")
    print(df.head())

    print("\\n=== Summary Statistics ===")
    print(df.describe())

    print("\\n=== Missing Data ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({{
        'missing_count': missing,
        'missing_percent': missing_pct
    }})
    print(missing_df[missing_df['missing_count'] > 0].sort_values('missing_count', ascending=False))


def filter_complete_cases(df):
    """Filter to complete cases only (no missing data)"""

    complete = df.dropna()
    print(f"\\nComplete cases: {{len(complete):,}} / {{len(df):,}} ({{len(complete)/len(df)*100:.1f}}%)")

    return complete


def basic_analysis(df):
    """Perform basic exploratory analysis"""

    print("\\n=== Basic Analysis ===")

    # Example analyses (customize based on actual variables)

    # Numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\\nNumeric variables ({{len(numeric_cols)}}):")
        for col in numeric_cols[:5]:  # Show first 5
            print(f"  {{col}}: mean={{df[col].mean():.2f}}, std={{df[col].std():.2f}}")

    # Categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\\nCategorical variables ({{len(categorical_cols)}}):")
        for col in categorical_cols[:5]:  # Show first 5
            print(f"  {{col}}: {{df[col].nunique()}} unique values")


def save_processed_data(df, output_file='processed_data.csv'):
    """Save processed dataset"""

    output_path = DATASET_DIR / 'processed' / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\\nSaved processed data to {{output_path}}")


def main():
    """Main analysis pipeline"""

    # Load dataset
    df = load_dataset()

    # Load data dictionary
    data_dict = load_data_dictionary()
    print(f"\\nData dictionary: {{data_dict['total_variables']}} variables documented")

    # Explore dataset
    explore_dataset(df)

    # Filter to complete cases
    df_complete = filter_complete_cases(df)

    # Basic analysis
    basic_analysis(df_complete)

    # Save processed data
    save_processed_data(df_complete)

    print("\\n=== Analysis Complete ===")


if __name__ == '__main__':
    main()
'''

        return code

    def generate_citations_bib(self, datasets: List[DatasetMetadata]) -> str:
        """Generate BibTeX citations file"""

        bib = """% Bibliography for AuDHD Correlation Study Datasets
% Generated by Dataset Documentation System
%
% To cite a dataset in LaTeX:
% \\cite{dataset_id}

"""

        for ds in datasets:
            # Generate BibTeX entry
            bib_id = ds.dataset_id.lower().replace('_', '')

            # Determine entry type
            entry_type = 'dataset' if ds.doi else 'misc'

            bib += f"""@{entry_type}{{{bib_id},
    title = {{{ds.full_name}}},
    author = {{{ds.source}}},
    year = {{{ds.release_date[:4] if ds.release_date else datetime.now().year}}},
"""

            if ds.doi:
                bib += f"""    doi = {{{ds.doi}}},
"""

            if ds.url:
                bib += f"""    url = {{{ds.url}}},
"""

            if ds.version:
                bib += f"""    version = {{{ds.version}}},
"""

            bib += f"""    note = {{{ds.description[:100]}...}}
}}

"""

        return bib

    def _format_bytes(self, size_bytes: int) -> str:
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def _get_score_class(self, score: float) -> str:
        """Get CSS class for quality score"""
        if score >= 90:
            return 'score-excellent'
        elif score >= 75:
            return 'score-good'
        elif score >= 60:
            return 'score-fair'
        else:
            return 'score-poor'

    def _get_fill_class(self, score: float) -> str:
        """Get CSS class for progress bar fill"""
        if score >= 90:
            return 'fill-excellent'
        elif score >= 75:
            return 'fill-good'
        elif score >= 60:
            return 'fill-fair'
        else:
            return 'fill-poor'

    def _get_quality_rating(self, score: float) -> str:
        """Get quality rating text"""
        if score >= 90:
            return 'Excellent Quality'
        elif score >= 75:
            return 'Good Quality'
        elif score >= 60:
            return 'Fair Quality'
        else:
            return 'Poor Quality - Requires Attention'

    def document_dataset(
        self,
        metadata: DatasetMetadata,
        variables: List[VariableMetadata],
        quality_metrics: QualityMetrics
    ):
        """Generate all documentation for a dataset"""

        logger.info(f"Generating documentation for {metadata.dataset_id}")

        # Generate README
        readme = self.generate_readme(metadata)
        readme_path = self.summary_dir / f"{metadata.dataset_id}_README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        logger.info(f"  ✓ Generated README: {readme_path}")

        # Generate data dictionary
        data_dict = self.generate_data_dictionary(metadata.dataset_id, variables)
        dict_path = self.dict_dir / f"{metadata.dataset_id}_dictionary.json"
        with open(dict_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
        logger.info(f"  ✓ Generated data dictionary: {dict_path}")

        # Generate quality report
        quality_html = self.generate_quality_report(metadata.dataset_id, quality_metrics)
        quality_path = self.quality_dir / f"{metadata.dataset_id}_quality.html"
        with open(quality_path, 'w') as f:
            f.write(quality_html)
        logger.info(f"  ✓ Generated quality report: {quality_path}")

        # Generate sample code
        sample_code = self.generate_sample_code(metadata.dataset_id, metadata)
        code_path = self.usage_dir / f"{metadata.dataset_id}_examples.py"
        with open(code_path, 'w') as f:
            f.write(sample_code)
        logger.info(f"  ✓ Generated sample code: {code_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate dataset documentation')
    parser.add_argument('--dataset', help='Dataset ID to document')
    parser.add_argument('--generate-all-datasets', action='store_true',
                       help='Generate documentation for all datasets')
    parser.add_argument('--output-dir', default='data/documentation',
                       help='Output directory for documentation')

    args = parser.parse_args()

    documenter = DatasetDocumenter(Path(args.output_dir))

    if args.generate_all_datasets:
        logger.info("Generating documentation for all datasets...")
        # This would integrate with the master catalog
        logger.info("Use catalog system to generate for all datasets")
    elif args.dataset:
        logger.info(f"Generating documentation for {args.dataset}")
        # This would load metadata from catalog and generate docs
    else:
        print("Specify --dataset or --generate-all-datasets")


if __name__ == '__main__':
    import sys
    main()