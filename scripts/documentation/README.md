# Dataset Documentation System

Comprehensive auto-generated documentation for all datasets in the AuDHD Correlation Study.

## Overview

This system provides automated generation of:
- **README files** with access instructions and dataset descriptions
- **Data dictionaries** with complete variable metadata
- **Quality reports** with interactive QC metrics visualization
- **Usage examples** with sample code for data loading and analysis
- **Provenance tracking** for complete data lineage and history
- **Master catalog** for unified dataset discovery and search
- **Citation management** with BibTeX format for publications

## Features

### 1. Auto-Generated Documentation

The `dataset_documenter.py` module automatically generates four types of documentation for each dataset:

- **README.md**: Comprehensive dataset overview including:
  - Dataset characteristics (sample size, variables, formats)
  - Access information and application process
  - License and usage restrictions
  - Citation information
  - File structure and organization

- **data_dictionary.json**: Structured variable metadata with:
  - Variable names and descriptions
  - Data types and units
  - Valid ranges and categories
  - Missing value codes
  - Primary/foreign key relationships
  - Summary statistics by data type

- **quality_report.html**: Interactive quality control dashboard with:
  - Overall quality score (0-100)
  - Completeness metrics and visualizations
  - Missing data analysis
  - Duplicate detection
  - Outlier identification
  - Validation error reports
  - QC recommendations

- **sample_code.py**: Ready-to-use Python examples for:
  - Loading datasets (multiple formats supported)
  - Reading data dictionaries
  - Basic exploratory analysis
  - Handling missing data
  - Filtering and processing
  - Saving processed outputs

### 2. Provenance Tracking

The `provenance_tracker.py` module maintains complete data lineage:

- **Acquisition history**: Original source, download date, version
- **Processing pipeline**: All transformations and operations applied
- **Quality control**: QC steps and results
- **Access logging**: Who accessed data and when
- **Lineage chains**: Parent-child dataset relationships
- **Checksums**: File integrity verification at each step

### 3. Master Catalog

The `catalog_builder.py` module creates a searchable catalog of all datasets:

- **SQLite database**: Structured storage with full-text search
- **JSON export**: Portable catalog format
- **Search capabilities**: By keyword, data type, access level
- **Statistics**: Aggregate metrics across all datasets
- **Metadata standardization**: Consistent schema for all datasets

### 4. Access Tracking

The `access_tracker.md` document provides:

- Current access status for all datasets
- Application procedures and timelines
- Required documents and approvals
- Contact information for Data Access Committees
- Budget tracking for paid datasets
- Compliance notes and restrictions

## Directory Structure

```
data/
├── documentation/
│   ├── dataset_summaries/        # README files for each dataset
│   │   ├── PGC_ADHD_GWAS_README.md
│   │   ├── SPARK_phenotypes_README.md
│   │   ├── ABCD_microbiome_README.md
│   │   └── EPA_AQS_neurotoxins_README.md
│   ├── data_dictionaries/        # Variable metadata
│   │   ├── PGC_ADHD_GWAS_dictionary.json
│   │   ├── SPARK_phenotypes_dictionary.json
│   │   ├── ABCD_microbiome_dictionary.json
│   │   └── EPA_AQS_neurotoxins_dictionary.json
│   ├── quality_reports/          # Interactive QC dashboards
│   │   ├── PGC_ADHD_GWAS_quality.html
│   │   ├── SPARK_phenotypes_quality.html
│   │   ├── ABCD_microbiome_quality.html
│   │   └── EPA_AQS_neurotoxins_quality.html
│   ├── usage_guides/             # Sample code
│   │   ├── PGC_ADHD_GWAS_examples.py
│   │   ├── SPARK_phenotypes_examples.py
│   │   ├── ABCD_microbiome_examples.py
│   │   └── EPA_AQS_neurotoxins_examples.py
│   └── provenance/               # Data lineage tracking
│       ├── PGC_ADHD_GWAS_provenance.json
│       ├── SPARK_phenotypes_provenance.json
│       ├── ABCD_microbiome_provenance.json
│       └── EPA_AQS_neurotoxins_provenance.json
└── catalogs/
    ├── master_catalog.json       # Complete dataset catalog
    ├── catalog.db                # SQLite searchable database
    ├── access_tracker.md         # Access status and applications
    └── citations.bib             # BibTeX citations
```

## Usage

### Generate Documentation for All Datasets

```bash
python scripts/documentation/generate_all_docs.py
```

This will:
1. Generate README, data dictionary, quality report, and sample code for each dataset
2. Create provenance records
3. Log initial acquisition events

### Build Master Catalog

```bash
python scripts/documentation/catalog_builder.py --build
```

### Search Catalog

```bash
# Search by keyword
python scripts/documentation/catalog_builder.py --search "ADHD"

# Filter by data type
python scripts/documentation/catalog_builder.py --data-type genomics

# Filter by access type
python scripts/documentation/catalog_builder.py --access-type public

# Get specific dataset
python scripts/documentation/catalog_builder.py --get PGC_ADHD_GWAS

# Show statistics
python scripts/documentation/catalog_builder.py --stats
```

### Track Provenance

```bash
# Create provenance record
python scripts/documentation/provenance_tracker.py \
    --dataset PGC_ADHD_GWAS \
    --create \
    --source "Psychiatric Genomics Consortium" \
    --url "https://www.med.unc.edu/pgc/download-results/"

# Log processing event
python scripts/documentation/provenance_tracker.py \
    --dataset PGC_ADHD_GWAS \
    --log-event "Applied QC filters" \
    --event-type processing \
    --actor researcher1

# Show event history
python scripts/documentation/provenance_tracker.py \
    --dataset PGC_ADHD_GWAS \
    --show-history

# Generate provenance report
python scripts/documentation/provenance_tracker.py \
    --dataset PGC_ADHD_GWAS \
    --generate-report
```

### Document Individual Dataset

```python
from scripts.documentation.dataset_documenter import (
    DatasetDocumenter, DatasetMetadata, VariableMetadata, QualityMetrics
)

# Define metadata
metadata = DatasetMetadata(
    dataset_id='my_dataset',
    name='My Dataset',
    full_name='My Full Dataset Name',
    description='Dataset description',
    data_type='genomics',
    source='Data Provider',
    url='https://example.com/data',
    version='1.0',
    release_date='2024-01-01',
    last_updated='2024-01-01',
    sample_size=1000,
    variables=50,
    file_format=['csv'],
    size_bytes=10485760,
    access_type='public',
    application_url=None,
    contact_email='contact@example.com',
    license='CC BY 4.0',
    citation='Author et al. (2024). Journal. 10:100-110.',
    doi='10.1234/example'
)

# Define variables
variables = [
    VariableMetadata(
        variable_name='sample_id',
        display_name='Sample ID',
        description='Unique sample identifier',
        data_type='text',
        unit=None,
        valid_range=None,
        categories=None,
        missing_codes=None,
        required=True,
        primary_key=True,
        foreign_key=None
    ),
    # ... more variables
]

# Define quality metrics
quality = QualityMetrics(
    total_records=1000,
    complete_records=950,
    completeness_rate=95.0,
    missing_rate=5.0,
    duplicate_records=0,
    outliers_detected=10,
    validation_errors=0,
    quality_score=92.0,
    qc_date='2024-01-01',
    qc_notes=['High quality dataset', 'Passed all QC checks']
)

# Generate documentation
documenter = DatasetDocumenter()
documenter.document_dataset(metadata, variables, quality)
```

## Documentation Standards

### README Format

All READMEs follow a consistent structure:
1. Dataset Information (name, type, source, version, dates)
2. Description
3. Dataset Characteristics (size, variables, formats)
4. Access Information (type, application process)
5. License and Citation
6. Links to data dictionary, quality report, and usage examples

### Data Dictionary Schema

```json
{
  "dataset_id": "string",
  "generated_date": "ISO8601",
  "version": "string",
  "total_variables": "integer",
  "variables": [
    {
      "variable_name": "string",
      "display_name": "string",
      "description": "string",
      "data_type": "numeric|categorical|binary|text|date",
      "unit": "string|null",
      "valid_range": {"min": "number", "max": "number"}|null,
      "categories": ["string"]|null,
      "missing_codes": ["value"]|null,
      "required": "boolean",
      "primary_key": "boolean",
      "foreign_key": "string|null"
    }
  ],
  "summary": {
    "numeric_vars": "integer",
    "categorical_vars": "integer",
    "binary_vars": "integer",
    "text_vars": "integer",
    "date_vars": "integer",
    "required_vars": "integer",
    "primary_keys": "integer"
  }
}
```

### Quality Report Metrics

Quality reports include:
- **Overall Score** (0-100): Weighted combination of all QC metrics
- **Completeness Rate** (%): Percentage of records with no missing values
- **Missing Rate** (%): Percentage of data points that are missing
- **Duplicate Records**: Number of duplicate entries identified
- **Outliers**: Number of potential outliers detected
- **Validation Errors**: Number of values that fail validation rules

Quality score calculation:
```
score = (completeness_rate * 0.4) +
        ((100 - duplicate_pct) * 0.3) +
        ((100 - outlier_pct) * 0.2) +
        ((100 - validation_error_pct) * 0.1)
```

Rating scale:
- 90-100: Excellent Quality
- 75-89: Good Quality
- 60-74: Fair Quality
- <60: Poor Quality (requires attention)

### Provenance Event Types

Standardized event types:
- `acquisition`: Initial data download/acquisition
- `processing`: Data transformation or processing
- `qc`: Quality control procedures
- `transformation`: Format conversion or restructuring
- `access`: Data access by researchers
- `export`: Data export or sharing

### Citation Format

All citations follow standard academic format:
- Journal articles: Author et al. (Year). Title. Journal. Volume(Issue):Pages.
- Datasets: Source Organization (Year). Dataset Name. Version. DOI/URL.
- Resources: Organization (Year). Resource Name. URL.

## Quality Assurance

### Documentation QA Checklist

Before committing documentation:
- [ ] All four documentation types generated for each dataset
- [ ] Data dictionaries include all variables
- [ ] Quality reports load without errors
- [ ] Sample code runs without errors
- [ ] Provenance records created
- [ ] Citations included in BibTeX file
- [ ] Master catalog updated
- [ ] Access tracker current

### Automated Checks

The system includes automated validation:
- Required fields presence check
- Data type consistency validation
- File format verification
- URL accessibility testing
- Checksum calculation
- Duplicate detection

## Integration with Other Systems

### Link with Download Pipeline

After downloading data:
```python
from scripts.pipeline.download_manager import DownloadManager
from scripts.documentation.provenance_tracker import ProvenanceTracker

# Download data
manager = DownloadManager()
# ... download process ...

# Log provenance
tracker = ProvenanceTracker()
tracker.log_event(
    dataset_id='PGC_ADHD_GWAS',
    event_type='acquisition',
    description='Downloaded from PGC website',
    actor='download_pipeline',
    output_files=['data/raw/PGC_ADHD_GWAS/'],
    software_version='1.0'
)
```

### Link with Master Registry

After documenting dataset:
```python
from scripts.integration.master_indexer import MasterIndexer
from scripts.documentation.catalog_builder import CatalogBuilder

# Add to catalog
builder = CatalogBuilder()
builder.add_dataset({
    'dataset_id': 'PGC_ADHD_GWAS',
    # ... metadata ...
})

# Register in master index
indexer = MasterIndexer()
indexer.import_dataset(
    dataset_name='PGC_ADHD_GWAS',
    data_df=df,
    id_column='sample_id',
    data_type='genomics'
)
```

## Best Practices

### Documentation Maintenance

1. **Update regularly**: Refresh documentation when data is updated
2. **Version control**: Track documentation versions with data versions
3. **Automated generation**: Use scripts rather than manual documentation
4. **Consistent formatting**: Follow templates for all datasets
5. **Quality checks**: Validate all generated documentation

### Provenance Tracking

1. **Log everything**: Record all operations on data
2. **Include parameters**: Document processing parameters
3. **Calculate checksums**: Verify data integrity at each step
4. **Track actors**: Record who performed each operation
5. **Maintain lineage**: Link derived datasets to sources

### Access Management

1. **Track applications**: Document all access requests
2. **Monitor deadlines**: Set reminders for approval dates
3. **Maintain compliance**: Follow all data use agreements
4. **Regular reviews**: Update access status weekly
5. **Document restrictions**: Clearly note usage limitations

## Troubleshooting

### Common Issues

**Missing documentation files**:
```bash
# Regenerate all documentation
python scripts/documentation/generate_all_docs.py
```

**Quality report not displaying**:
- Check browser console for JavaScript errors
- Verify HTML file is complete
- Open in different browser

**Catalog search not working**:
```bash
# Rebuild catalog database
python scripts/documentation/catalog_builder.py --build
```

**Provenance events not saving**:
- Check provenance directory exists and is writable
- Verify dataset_id matches existing record
- Create new provenance record if needed

## Future Enhancements

Planned additions:
- **Automated documentation updates**: Detect data changes and regenerate
- **Web dashboard**: Browse all documentation in web interface
- **API access**: RESTful API for programmatic access to catalog
- **Version comparison**: Compare data dictionary versions
- **Collaborative annotations**: Allow researchers to add notes
- **Integration testing**: Automated tests for documentation pipeline
- **Metadata validation**: JSON schema validation
- **DOI minting**: Automatic DOI generation for datasets

## Support

For questions or issues:
1. Check this README
2. Review example documentation in `data/documentation/`
3. Run with `--help` flag for command-line tools
4. Check logs in `logs/documentation.log`
5. Open GitHub issue with detailed description

---

**Version**: 1.0
**Last Updated**: 2025-09-30
**Maintained by**: AuDHD Correlation Study Team