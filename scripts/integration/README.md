# Data Integration and Master Sample Registry

Comprehensive system for tracking samples across all ADHD/autism datasets with cross-dataset ID mapping, data availability tracking, and interactive dashboards.

## Overview

This system provides:
1. **Master Sample Registry**: Unified database tracking all samples across datasets
2. **ID Mapping**: Cross-dataset identifier resolution using fuzzy matching
3. **Completeness Tracking**: Data availability metrics per sample
4. **Interactive Dashboard**: Web-based data explorer for visualization

### Key Components

1. **Master Indexer** (`master_indexer.py`)
   - Builds SQLite database with all samples
   - Tracks data availability across 6 data types
   - Manages cross-dataset sample deduplication
   - Calculates completeness scores

2. **ID Mapper** (`id_mapper.py`)
   - Fuzzy string matching for ID resolution
   - Demographic-based matching
   - Confidence scoring for mappings

3. **Completeness Calculator** (`completeness_calculator.py`)
   - Calculates data availability scores
   - Generates completeness reports
   - Exports availability matrices

4. **Data Explorer Dashboard** (`data_explorer.html`)
   - Interactive web interface
   - Real-time filtering and search
   - Visual data availability charts
   - Sample-level detail views

## Installation

```bash
# Python 3.8+
pip install pandas sqlite3 fuzzywuzzy python-Levenshtein
```

## Usage

### 1. Build Master Registry

```bash
# Import genomics data
python scripts/integration/master_indexer.py \\
    --build \\
    --dataset SPARK \\
    --input data/genetics/spark_samples.csv \\
    --data-type genomics \\
    --output data/index/

# Import metabolomics data
python scripts/integration/master_indexer.py \\
    --update \\
    --dataset MetabolomicsCore \\
    --input data/metabolomics/samples.csv \\
    --data-type metabolomics \\
    --output data/index/

# Import microbiome data
python scripts/integration/master_indexer.py \\
    --update \\
    --dataset MicrobiomeProject \\
    --input data/microbiome/samples.csv \\
    --data-type microbiome \\
    --output data/index/
```

**Output:** `data/index/master_sample_registry.db`

### 2. Map IDs Across Datasets

```bash
# Map sample IDs between datasets
python scripts/integration/id_mapper.py \\
    --dataset1 data/genetics/samples.csv \\
    --dataset2 data/metabolomics/samples.csv \\
    --id1-column sample_id \\
    --id2-column participant_id \\
    --output data/index/id_mappings.csv
```

**Output:** CSV with ID mappings and confidence scores

### 3. Calculate Completeness Scores

```bash
# Update completeness scores for all samples
python scripts/integration/completeness_calculator.py \\
    --database data/index/master_sample_registry.db \\
    --output data/index/

# Generate data availability matrix
python scripts/integration/completeness_calculator.py \\
    --database data/index/master_sample_registry.db \\
    --output data/index/
```

**Output:**
- `data/index/data_availability_matrix.csv`
- Updated completeness scores in database

### 4. View Dashboard

```bash
# Open in browser
open dashboards/data_explorer.html

# Or serve with Python
cd dashboards
python -m http.server 8000
# Navigate to http://localhost:8000/data_explorer.html
```

## Database Schema

### samples table

| Column | Type | Description |
|--------|------|-------------|
| master_id | TEXT | Unique master identifier |
| sample_id | TEXT | Original sample ID |
| age | INTEGER | Age at enrollment |
| sex | TEXT | Sex (Male/Female/Other) |
| ethnicity | TEXT | Self-reported ethnicity |
| ancestry | TEXT | Genetic ancestry |
| primary_diagnosis | TEXT | Primary diagnosis (ASD/ADHD/Control) |
| has_genomics | INTEGER | Genomics data available (0/1) |
| has_metabolomics | INTEGER | Metabolomics data available (0/1) |
| has_microbiome | INTEGER | Microbiome data available (0/1) |
| has_clinical | INTEGER | Clinical data available (0/1) |
| has_imaging | INTEGER | Imaging data available (0/1) |
| has_environmental | INTEGER | Environmental data available (0/1) |
| completeness_score | REAL | Data completeness (0.0-1.0) |
| access_status | TEXT | available/pending/restricted |
| first_seen | TEXT | Date first added |
| last_updated | TEXT | Date last updated |

### sample_datasets table

Links samples to their source datasets:

| Column | Type | Description |
|--------|------|-------------|
| master_id | TEXT | Master sample ID |
| dataset | TEXT | Dataset name |
| source_id | TEXT | Original ID in dataset |
| date_added | TEXT | Date added to registry |

### sample_diagnoses table

Tracks all diagnoses per sample:

| Column | Type | Description |
|--------|------|-------------|
| master_id | TEXT | Master sample ID |
| diagnosis | TEXT | Diagnosis label |
| diagnosis_date | TEXT | Date diagnosed |

### id_mappings table

Cross-dataset ID mappings:

| Column | Type | Description |
|--------|------|-------------|
| dataset | TEXT | Dataset name |
| dataset_id | TEXT | ID in dataset |
| master_id | TEXT | Master sample ID |
| confidence | REAL | Mapping confidence (0-1) |
| mapping_method | TEXT | Method used (fuzzy/demographic/manual) |
| date_mapped | TEXT | Date mapped |

## Complete Workflow

### End-to-End Integration

```bash
# Step 1: Create master registry database
python scripts/integration/master_indexer.py \\
    --build \\
    --output data/index/

# Step 2: Import all datasets
# Genomics
python scripts/integration/master_indexer.py \\
    --update \\
    --dataset SPARK \\
    --input data/genetics/spark_samples.csv \\
    --data-type genomics \\
    --output data/index/

# Metabolomics
python scripts/integration/master_indexer.py \\
    --update \\
    --dataset MetabolomicsCore \\
    --input data/metabolomics/samples.csv \\
    --data-type metabolomics \\
    --output data/index/

# Microbiome
python scripts/integration/master_indexer.py \\
    --update \\
    --dataset SRA \\
    --input data/microbiome/samples.csv \\
    --data-type microbiome \\
    --output data/index/

# Clinical
python scripts/integration/master_indexer.py \\
    --update \\
    --dataset ClinicalTrials \\
    --input data/trials/participants.csv \\
    --data-type clinical \\
    --output data/index/

# Environmental
python scripts/integration/master_indexer.py \\
    --update \\
    --dataset Environmental \\
    --input data/environmental/geocoded.csv \\
    --data-type environmental \\
    --output data/index/

# Step 3: Calculate completeness scores
python scripts/integration/completeness_calculator.py \\
    --database data/index/master_sample_registry.db \\
    --output data/index/

# Step 4: View dashboard
open dashboards/data_explorer.html
```

## Output Files

### master_sample_registry.db

SQLite database with complete sample registry:
- 8 example samples (4 ASD, 3 ADHD, 1 Control)
- Data availability across 6 data types
- Cross-dataset source tracking
- Completeness scores

### data_availability_matrix.csv

| master_id | primary_diagnosis | has_genomics | has_metabolomics | has_microbiome | has_clinical | has_imaging | has_environmental | completeness_score | access_status |
|-----------|-------------------|--------------|------------------|----------------|--------------|-------------|-------------------|-------------------|---------------|
| MASTER_control001 | Control | 1 | 1 | 1 | 1 | 1 | 1 | 1.00 | available |
| MASTER_asd001 | ASD | 1 | 1 | 1 | 1 | 0 | 1 | 0.83 | available |
| MASTER_adhd001 | ADHD | 1 | 1 | 0 | 1 | 1 | 1 | 0.83 | available |

### registry_summary.json

```json
{
  "generated_date": "2025-09-30",
  "total_samples": 8,
  "by_diagnosis": {
    "ASD": 4,
    "ADHD": 3,
    "Control": 1
  },
  "by_completeness": {
    "complete (100%)": 1,
    "high (>80%)": 3,
    "medium (50-80%)": 3
  },
  "data_availability": {
    "genomics": 8,
    "metabolomics": 5,
    "microbiome": 5,
    "clinical": 8,
    "imaging": 3,
    "environmental": 6
  },
  "multi_omics": {
    "all_6_types": 1,
    "5_types": 3,
    "4_types": 2
  }
}
```

## Data Availability Metrics

### Completeness Score Calculation

```python
completeness = (has_genomics + has_metabolomics + has_microbiome +
                has_clinical + has_imaging + has_environmental) / 6.0
```

**Categories:**
- **Complete (100%)**: All 6 data types available
- **High (>80%)**: 5 data types available
- **Medium (50-80%)**: 3-4 data types available
- **Low (<50%)**: 1-2 data types available

### Data Type Priorities

**Tier 1 (Core):**
- Genomics
- Clinical phenotypes

**Tier 2 (Multi-omics):**
- Metabolomics
- Microbiome

**Tier 3 (Contextual):**
- Imaging
- Environmental exposures

## Querying the Registry

### Python API

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('data/index/master_sample_registry.db')

# Query samples with all data types
query = '''
SELECT master_id, sample_id, primary_diagnosis, completeness_score
FROM samples
WHERE completeness_score = 1.0
'''

df = pd.read_sql_query(query, conn)
print(f"Found {len(df)} samples with complete data")

# Query ASD samples with genomics and metabolomics
query = '''
SELECT master_id, sample_id, access_status
FROM samples
WHERE primary_diagnosis = 'ASD'
  AND has_genomics = 1
  AND has_metabolomics = 1
'''

df = pd.read_sql_query(query, conn)
print(f"Found {len(df)} ASD samples with genomics+metabolomics")

conn.close()
```

### Command Line

```bash
# Query all available ASD samples
sqlite3 data/index/master_sample_registry.db \\
    "SELECT master_id, completeness_score FROM samples
     WHERE primary_diagnosis = 'ASD' AND access_status = 'available';"

# Count samples by diagnosis
sqlite3 data/index/master_sample_registry.db \\
    "SELECT primary_diagnosis, COUNT(*) FROM samples GROUP BY primary_diagnosis;"

# Find high-completeness samples
sqlite3 data/index/master_sample_registry.db \\
    "SELECT master_id, completeness_score FROM samples
     WHERE completeness_score >= 0.8 ORDER BY completeness_score DESC;"
```

## Integration with Analysis Pipelines

### Example: Multi-omics Integration

```python
import pandas as pd
import sqlite3

# Load master registry
conn = sqlite3.connect('data/index/master_sample_registry.db')

# Get samples with genomics + metabolomics + microbiome
query = '''
SELECT master_id, sample_id
FROM samples
WHERE has_genomics = 1
  AND has_metabolomics = 1
  AND has_microbiome = 1
  AND access_status = 'available'
'''

samples = pd.read_sql_query(query, conn)
print(f"Found {len(samples)} samples for multi-omics analysis")

# Load respective datasets
genomics = pd.read_csv('data/genetics/genotypes.csv')
metabolomics = pd.read_csv('data/metabolomics/metabolite_levels.csv')
microbiome = pd.read_csv('data/microbiome/taxa_abundance.csv')

# Merge on master_id
integrated = samples.merge(genomics, on='master_id')
integrated = integrated.merge(metabolomics, on='master_id')
integrated = integrated.merge(microbiome, on='master_id')

print(f"Integrated {len(integrated)} samples across all omics layers")

conn.close()
```

### Example: Phenotype-Genotype Association

```python
# Query samples with phenotype and genomics
query = '''
SELECT s.master_id, s.primary_diagnosis, s.age, s.sex
FROM samples s
WHERE s.has_genomics = 1 AND s.has_clinical = 1
'''

samples = pd.read_sql_query(query, conn)

# Load genetic variants
variants = pd.read_csv('data/genetics/variants.vcf', sep='\t')

# Load clinical phenotypes
phenotypes = pd.read_csv('data/clinical/assessments.csv')

# Merge for GWAS
gwas_data = samples.merge(variants, on='master_id')
gwas_data = gwas_data.merge(phenotypes, on='master_id')

# Run association analysis
# ... GWAS pipeline ...
```

## Dashboard Features

### Interactive Data Explorer

**Key Features:**
- Real-time sample filtering by diagnosis, data type, completeness
- Visual data availability charts
- Sample-level detail table with completeness progress bars
- Multi-omics coverage statistics
- Access status indicators

**Filters:**
- Diagnosis (ASD/ADHD/Control)
- Data type (genomics, metabolomics, etc.)
- Completeness level (high/medium/low)

**Visualizations:**
- Data availability bar charts
- Multi-omics coverage distribution
- Completeness score heat maps
- Access status pie charts

### Usage

```bash
# Open dashboard
open dashboards/data_explorer.html

# Or serve locally
python -m http.server 8000 --directory dashboards
# Navigate to http://localhost:8000/data_explorer.html
```

## Sample Statistics (Example Data)

**Total Samples:** 8

**By Diagnosis:**
- ASD: 4 (50%)
- ADHD: 3 (37.5%)
- Control: 1 (12.5%)

**Data Availability:**
- Genomics: 8/8 (100%)
- Clinical: 8/8 (100%)
- Environmental: 6/8 (75%)
- Metabolomics: 5/8 (62.5%)
- Microbiome: 5/8 (62.5%)
- Imaging: 3/8 (37.5%)

**Multi-omics:**
- All 6 types: 1 sample
- 5 types: 3 samples
- 4 types: 2 samples
- 3 types: 2 samples

**Completeness:**
- Mean: 70%
- Complete (100%): 1 sample
- High (>80%): 3 samples
- Medium (50-80%): 3 samples
- Low (<50%): 1 sample

**Access Status:**
- Available: 6 samples
- Restricted: 1 sample
- Pending: 1 sample

## Best Practices

### Sample Deduplication

1. **Primary Matching:**
   - Exact ID match across datasets
   - Demographic concordance (age, sex, ethnicity)
   - Study enrollment dates

2. **Fuzzy Matching:**
   - String similarity (>80% threshold)
   - Partial ID matches
   - Cross-reference with published datasets

3. **Manual Curation:**
   - Review ambiguous matches
   - Confirm with dataset PIs
   - Document mapping decisions

### Data Quality Control

**QC Flags to Track:**
- Missing critical demographics
- Inconsistent diagnoses across datasets
- Outlier completeness scores
- Failed ID mappings
- Access restrictions

**Implementation:**

```python
# Add QC flag
from master_indexer import MasterIndexer

indexer = MasterIndexer('data/index/master_sample_registry.db')

# Flag samples with missing demographics
cursor = indexer.conn.cursor()
cursor.execute('''
INSERT INTO sample_qc (master_id, flag_type, flag_description, date_flagged)
SELECT master_id, 'missing_demographics', 'Age or sex missing', datetime('now')
FROM samples
WHERE age IS NULL OR sex IS NULL
''')
```

### Access Management

**Track access status:**
- `available`: Data publicly available or approved
- `pending`: Access request in progress
- `restricted`: Requires special approval

**Consent tracking:**

```python
# Add consent restriction
cursor.execute('''
INSERT INTO sample_consent (master_id, restriction_type, restriction_description)
VALUES (?, 'commercial_use_prohibited', 'No commercial use per consent')
''', (master_id,))
```

## Troubleshooting

### Database Locked

```python
# If database is locked, check for open connections
import sqlite3

# Use shorter timeout
conn = sqlite3.connect('master_sample_registry.db', timeout=10)
```

### ID Mapping Failures

```bash
# Lower fuzzy match threshold for difficult cases
python id_mapper.py \\
    --dataset1 data/genetics/samples.csv \\
    --dataset2 data/metabolomics/samples.csv \\
    --threshold 70 \\
    --output mappings.csv
```

### Missing Completeness Scores

```bash
# Recalculate all scores
python completeness_calculator.py \\
    --database data/index/master_sample_registry.db \\
    --output data/index/
```

## Future Enhancements

Potential additions:
- **Real-time dashboard**: Connect dashboard to live database
- **API endpoint**: RESTful API for programmatic access
- **Automated ID mapping**: Machine learning for ID resolution
- **Data lineage tracking**: Full provenance of all samples
- **Version control**: Track data updates and changes
- **Export formats**: GED ICOM, OMOP CDM compliance
- **Access request workflow**: Integrated approval system

## Support

For questions or issues:
1. Check database schema and example queries
2. Review sample statistics and availability
3. Test with example data first
4. Open GitHub issue with detailed description

---

**Last updated**: 2025-09-30
**Version**: 1.0
**Maintained by**: AuDHD Correlation Study Team