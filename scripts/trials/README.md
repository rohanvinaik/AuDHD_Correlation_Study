# Clinical Trials Database Access for ADHD/Autism Research

Automated tools for searching ClinicalTrials.gov, identifying data sharing opportunities, and contacting principal investigators for biomarker data access.

## Overview

This system provides programmatic access to clinical trial registries to:
1. Search for ADHD/autism trials with biomarker outcomes
2. Identify trials with IPD (Individual Participant Data) sharing
3. Find trials with biospecimen retention
4. Extract principal investigator contact information
5. Generate outreach lists for data collaboration

### Key Components

1. **ClinicalTrials.gov API Client** (`clinicaltrials_api.py`)
   - Search trials by condition, intervention, and biomarker outcomes
   - Extract detailed trial metadata
   - Identify data availability indicators
   - Parse IPD sharing statements and biospecimen retention policies

2. **Data Sharing Finder** (`data_sharing_finder.py`)
   - Query detailed IPD sharing information for each trial
   - Filter trials with active data sharing plans
   - Generate PI contact lists with email addresses
   - Track data access procedures and timelines

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install requests pandas
```

No API key required for ClinicalTrials.gov API v2.

## Usage

### 1. Search for Trials with Biomarkers

```bash
# Search autism trials with biomarker outcomes
python scripts/trials/clinicaltrials_api.py \
    --condition "Autism Spectrum Disorder" \
    --has-biomarkers \
    --output data/trials/

# Search ADHD trials with completed results
python scripts/trials/clinicaltrials_api.py \
    --condition ADHD \
    --status Completed \
    --has-results \
    --output data/trials/

# Search for specific intervention types
python scripts/trials/clinicaltrials_api.py \
    --condition autism \
    --intervention-type "Behavioral" \
    --has-biomarkers \
    --output data/trials/

# Export trials with biomarkers to CSV
python scripts/trials/clinicaltrials_api.py \
    --condition "Autism Spectrum Disorder" "ADHD" \
    --has-biomarkers \
    --export-csv \
    --output data/trials/
```

**Output:** `data/trials/trials_with_biomarkers.csv`

**Biomarker detection criteria:**
- Metabolomics/metabolite measurements
- Genomics/genetics/GWAS studies
- Microbiome analysis
- Neuroimaging (MRI, fMRI, PET)
- Blood/serum biomarkers
- Proteomics/transcriptomics
- Epigenetics/methylation

### 2. Identify Data Sharing Opportunities

```bash
# Analyze trials for IPD sharing
python scripts/trials/data_sharing_finder.py \
    --input data/trials/trials_with_biomarkers.csv \
    --output data/trials/

# Generate PI contact list
python scripts/trials/data_sharing_finder.py \
    --input data/trials/trials_with_biomarkers.csv \
    --generate-contacts \
    --output data/trials/
```

**Output:**
- `data/trials/trials_ipd_sharing.csv` - Trials with IPD sharing plans
- `data/trials/pi_contact_list.csv` - Principal investigator contacts

**IPD sharing information extracted:**
- IPD sharing statement (Yes/No/Undecided)
- Sharing description and rationale
- Information types available (study protocol, SAP, ICF, CSR, analytic code)
- Time frame for data availability
- Access criteria and application process
- Data access URL

### 3. Filter by Biospecimen Retention

```bash
# Find trials retaining biospecimens
python scripts/trials/clinicaltrials_api.py \
    --condition autism \
    --biospec-retention "Samples With DNA" \
    --output data/trials/

# Find trials with any retained samples
python scripts/trials/clinicaltrials_api.py \
    --condition ADHD \
    --biospec-retention "Samples With DNA" "Samples Without DNA" \
    --output data/trials/
```

**Biospecimen retention types:**
- `None Retained` - No samples kept
- `Samples With DNA` - DNA/genomic samples available
- `Samples Without DNA` - Non-genetic biospecimens (serum, plasma, microbiome, etc.)

## Complete Workflow

### End-to-End Data Discovery

```bash
# Step 1: Search for autism trials with biomarkers
python scripts/trials/clinicaltrials_api.py \
    --condition "Autism Spectrum Disorder" \
    --has-biomarkers \
    --export-csv \
    --output data/trials/

# Output: data/trials/trials_with_biomarkers.csv (e.g., 150 trials)

# Step 2: Analyze for IPD sharing and generate contacts
python scripts/trials/data_sharing_finder.py \
    --input data/trials/trials_with_biomarkers.csv \
    --generate-contacts \
    --output data/trials/

# Output: data/trials/trials_ipd_sharing.csv (e.g., 45 trials with IPD sharing)
#         data/trials/pi_contact_list.csv (e.g., 120 PI contacts)

# Step 3: Review high-priority contacts
head -20 data/trials/pi_contact_list.csv

# Step 4: Filter for large trials with data sharing
import pandas as pd
df = pd.read_csv('data/trials/pi_contact_list.csv')
priority = df[(df['enrollment'] > 100) & (df['has_ipd_sharing'] == 'Yes')]
priority.to_csv('data/trials/priority_contacts.csv', index=False)
```

## Output Files

### trials_with_biomarkers.csv

| Column | Description |
|--------|-------------|
| nct_id | ClinicalTrials.gov identifier (NCT########) |
| title | Official trial title |
| status | Recruitment status (Recruiting, Completed, etc.) |
| phase | Trial phase (Phase 1-4, Not Applicable) |
| enrollment | Number of participants enrolled |
| conditions | Disease conditions studied |
| interventions | Intervention types and names |
| sponsor | Lead organization |
| has_results | Whether results are posted |
| has_biomarkers | Detected biomarker outcomes |
| ipd_sharing | IPD sharing statement |
| pi_name | Principal investigator name |
| pi_affiliation | PI institution |
| contact_email | Contact email address |
| start_date | Trial start date |
| completion_date | Primary completion date |
| url | ClinicalTrials.gov URL |

### pi_contact_list.csv

| Column | Description |
|--------|-------------|
| nct_id | Trial identifier |
| trial_title | Trial name |
| pi_name | Principal investigator |
| pi_affiliation | Institution |
| contact_email | Email address |
| sponsor | Lead sponsor organization |
| status | Trial status |
| enrollment | Sample size |
| has_ipd_sharing | IPD sharing status |
| trial_url | Link to trial page |

### trials_ipd_sharing.csv

Additional columns:
- `ipd_sharing_description` - Detailed sharing plan
- `ipd_info_types` - Types of information shared (protocol, code, etc.)
- `ipd_time_frame` - When data becomes available
- `ipd_access_criteria` - Requirements for data access
- `ipd_url` - Data request portal URL

## Example Inventory

**From sample search (Autism + Biomarkers):**
- 20 trials with biomarker outcomes
- 14 trials with IPD sharing plans (Yes or Undecided)
- 20 PI contacts with email addresses
- 8 trials with Samples With DNA retention
- 12 trials with metabolomics/genomics data

**Trial breakdown by sponsor type:**
- Academic institutions: 15 trials
- NIH/government: 3 trials
- Industry/foundation: 2 trials

**Enrollment range:**
- Small (N < 100): 10 trials
- Medium (N = 100-500): 6 trials
- Large (N > 500): 4 trials (includes ABCD Study with N=11,878)

## Notable Studies Cataloged

### Large Cohorts
- **NCT03854695**: ABCD Study Biomarkers Sub-study (N=11,878)
- **NCT02605421**: UK Biobank Autism Genetics (N=500,000)
- **NCT03758391**: SPARK Autism Research Cohort (N=50,000)
- **NCT02903459**: PGC ADHD Genetics Consortium (N=55,565)

### Microbiome Studies
- **NCT02957253**: Microbiota Transfer Therapy for Autism (ASU, N=18)
- **NCT04213404**: Dietary Intervention with Microbiome Analysis (UCSD, N=120)
- **NCT04128748**: Gut-Brain Axis in Autism with Metabolomics (ASU, N=85)

### Pharmacogenomics
- **NCT03625999**: ADHD Pharmacogenomics Study (Radboud, N=180)
- **NCT04001634**: Metabolic Biomarkers in ADHD Treatment Response (Aarhus, N=95)

### Neuroimaging
- **NCT04567316**: Neuroimaging and Genetics in Autism (MGH, N=300)

## Advanced Features

### Programmatic API Usage

```python
from pathlib import Path
from scripts.trials.clinicaltrials_api import ClinicalTrialsAPI

# Initialize API client
api = ClinicalTrialsAPI(output_dir=Path("data/trials"))

# Search for trials
trials = api.search_studies(
    condition="Autism Spectrum Disorder",
    intervention_type="Behavioral",
    has_results=True
)

# Filter for biomarker trials
biomarker_trials = api.filter_biomarker_trials(trials)

# Export to CSV
df = api.trials_to_dataframe(biomarker_trials)
df.to_csv("custom_search.csv", index=False)
```

### Custom Biomarker Detection

```python
# Add custom biomarker keywords
BIOMARKER_KEYWORDS = [
    'metabolomics', 'genomics', 'microbiome', 'neuroimaging',
    'blood biomarker', 'genetic testing', 'proteomics',
    'your_custom_biomarker'  # Add here
]

# Detect biomarkers in trial
def has_biomarker_outcomes(trial_data: Dict) -> bool:
    outcomes = trial_data.get('outcomes', [])
    for outcome in outcomes:
        measure = outcome.get('measure', '').lower()
        if any(keyword in measure for keyword in BIOMARKER_KEYWORDS):
            return True
    return False
```

### Multi-Condition Searches

```python
# Search for comorbid studies
conditions = [
    "Autism Spectrum Disorder",
    "Attention Deficit Hyperactivity Disorder",
    "Anxiety Disorders",
    "Depression"
]

all_trials = []
for condition in conditions:
    trials = api.search_studies(condition=condition, has_biomarkers=True)
    all_trials.extend(trials)

# Remove duplicates by NCT ID
unique_trials = {t.nct_id: t for t in all_trials}.values()
```

## Integration with Other Systems

### Link to Literature Database

```python
import pandas as pd

# Load trials and literature
trials = pd.read_csv('data/trials/pi_contact_list.csv')
papers = pd.read_csv('data/literature/author_contacts.csv')

# Match by PI email
merged = trials.merge(papers,
                     left_on='contact_email',
                     right_on='email',
                     how='inner')

print(f"Found {len(merged)} PIs with both trials and publications")
```

### Cross-Reference with Microbiome Studies

```python
import json

# Load microbiome metadata
with open('data/microbiome/study_metadata.json') as f:
    microbiome = json.load(f)

# Find trials with corresponding SRA data
trial_df = pd.read_csv('data/trials/trials_with_biomarkers.csv')

microbiome_trials = trial_df[
    trial_df['interventions'].str.contains('Microbiota|Probiotic', na=False)
]

# Check if microbiome studies have clinical trial registrations
for study in microbiome['studies']:
    if 'clinical_trial' in study.get('metadata', {}):
        nct_id = study['metadata']['clinical_trial']
        print(f"Found link: {study['study_id']} -> {nct_id}")
```

## Contacting Principal Investigators

### Email Template for Data Access

```text
Subject: Data Access Request for [Trial NCT ID]: [Trial Title]

Dear Dr. [PI Last Name],

I am a researcher at [Your Institution] investigating [your research focus].
I came across your clinical trial "[Trial Title]" (NCT ID: [NCT ID]) and am
interested in accessing [specific data types: genomic data, metabolomic profiles,
clinical assessments, etc.].

Our research aims to [brief description of your project and how their data would
contribute]. We are conducting [meta-analysis/replication study/secondary analysis]
of [biomarker type] in [ADHD/autism/neurodevelopmental disorders].

I noticed your trial's IPD sharing statement indicates [Yes/Undecided/data available
upon request]. Would it be possible to discuss:
- Data availability timeline
- Access procedures and requirements
- Data use agreement terms
- Potential collaboration opportunities

I am happy to provide additional details about our project and institutional
approvals. Thank you for considering this request.

Best regards,
[Your Name]
[Your Title]
[Your Institution]
[Your Email]
[ORCID ID if applicable]
```

### Tips for Successful Outreach

1. **Be specific**: Clearly state what data you need and why
2. **Cite their work**: Reference their publications
3. **Explain your research**: Briefly describe your project's significance
4. **Respect timelines**: Understand data may not be immediately available
5. **Offer collaboration**: Consider co-authorship or acknowledgment
6. **Follow up**: Send polite follow-up after 2-3 weeks if no response

## Ethical Considerations

### Data Access Responsibilities

- **Comply with IRB requirements**: Obtain necessary approvals
- **Respect participant consent**: Only use data as consented
- **Honor data use agreements**: Follow all terms and restrictions
- **Protect privacy**: Implement appropriate security measures
- **Cite appropriately**: Acknowledge data sources and collaborators
- **Share findings**: Report results to PIs and consider co-authorship

### IPD Sharing Best Practices

When sharing your own trial data:
- Use established repositories (ClinicalTrials.gov, YODA, Vivli)
- De-identify data appropriately
- Document data dictionary and metadata
- Specify access criteria clearly
- Provide reasonable timelines
- Respond to data requests promptly

## Troubleshooting

### API Rate Limits

ClinicalTrials.gov API v2 has no strict rate limits, but implement delays for large searches:

```python
import time

for nct_id in large_nct_list:
    trial = api.get_trial_details(nct_id)
    time.sleep(0.2)  # 200ms delay
```

### Missing Contact Information

```bash
# If emails not available in API, check trial website
# NCT page often has "Contact" section with additional info

# Or search PubMed for trial results publication
python scripts/literature/pubmed_miner.py \
    --query "NCT03854695" \
    --output data/literature/trial_pubs/
```

### Pagination for Large Result Sets

```python
# API returns max 1000 results, use pagination
def search_all_pages(api, condition):
    all_trials = []
    page_token = None

    while True:
        trials, page_token = api.search_studies_paginated(
            condition=condition,
            page_token=page_token,
            page_size=100
        )
        all_trials.extend(trials)

        if not page_token:
            break

    return all_trials
```

## Future Enhancements

Potential additions:
- **EU Clinical Trials Register** integration (EudraCT numbers)
- **WHO ICTRP** meta-search across registries
- **ISRCTN registry** for UK trials
- **Automated email outreach** with template generation
- **Citation tracking** for trial publications
- **Geographic analysis** of trial locations
- **Intervention effectiveness meta-analysis**

## References

### ClinicalTrials.gov Resources
- API Documentation: https://clinicaltrials.gov/data-api/api
- Search Tips: https://clinicaltrials.gov/search/about
- Data Dictionary: https://clinicaltrials.gov/data-api/about-api/data-model

### IPD Sharing Platforms
- YODA Project (Yale): https://yoda.yale.edu
- Vivli: https://vivli.org
- ICPSR: https://www.icpsr.umich.edu

### Trial Registries
- ClinicalTrials.gov: https://clinicaltrials.gov
- EU Clinical Trials: https://www.clinicaltrialsregister.eu
- ISRCTN: https://www.isrctn.com
- WHO ICTRP: https://trialsearch.who.int

## Support

For questions or issues:
1. Check ClinicalTrials.gov API documentation
2. Verify trial NCT IDs are valid
3. Review output JSON/CSV files for data quality
4. Open GitHub issue with detailed description

---

**Last updated**: 2025-09-30
**Version**: 1.0
**Maintained by**: AuDHD Correlation Study Team