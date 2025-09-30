# Literature Mining Infrastructure for Data Discovery

Automated tools for mining PubMed/PMC to discover ADHD/Autism research datasets, supplementary materials, and collaboration opportunities.

## Overview

This system uses NCBI E-utilities API and web scraping to:
1. Search PubMed for relevant papers
2. Extract supplementary data files from PMC
3. Identify dataset mentions using NLP
4. Build author contact database
5. Find unpublished data and collaboration opportunities

### Key Components

1. **PubMed Miner** (`pubmed_miner.py`)
   - Search PubMed with complex queries
   - Extract paper metadata
   - Identify data availability indicators
   - Export JSON with structured data

2. **Supplement Extractor** (`supplement_extractor.py`)
   - Parse PMC full-text HTML
   - Extract supplementary file links
   - Identify external repositories (GitHub, Figshare, Zenodo)
   - Download data files

3. **Dataset Mention Finder** (`dataset_mention_finder.py`)
   - Extract sample sizes (N = ...)
   - Identify known cohorts (UK Biobank, ABCD, etc.)
   - Find unpublished data mentions
   - Detect collaboration opportunities
   - NLP entity extraction with spaCy

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install biopython requests pandas beautifulsoup4 lxml

# Optional: spaCy for NLP
pip install spacy
python -m spacy download en_core_web_sm
```

### NCBI API Setup

Register your email for Entrez API:

```python
# In pubmed_miner.py
Entrez.email = "your.email@institution.edu"

# Or use command line
python pubmed_miner.py --email your.email@institution.edu --query autism metabolomics
```

**Rate limits**: 3 requests/second without API key, 10/second with API key.
Register for API key at: https://www.ncbi.nlm.nih.gov/account/settings/

## Usage

### 1. Search PubMed for Papers

```bash
# Search autism metabolomics papers
python scripts/literature/pubmed_miner.py \\
    --query autism metabolomics \\
    --years 2015:2024 \\
    --has-data \\
    --output data/literature/

# Use predefined query template
python scripts/literature/pubmed_miner.py \\
    --query-template autism_genetics \\
    --max-results 500 \\
    --output data/literature/

# Search for clinical trials only
python scripts/literature/pubmed_miner.py \\
    --query adhd biomarker \\
    --pub-types "Clinical Trial" "Observational Study" \\
    --output data/literature/

# Extract specific PMIDs
python scripts/literature/pubmed_miner.py \\
    --pmids 30478444,30804558,32493969 \\
    --output data/literature/
```

**Predefined query templates:**
- `autism_metabolomics`: Autism + metabolomics/metabolome
- `adhd_genetics`: ADHD + genomics/genetics/GWAS
- `autism_genetics`: Autism + genomics/sequencing
- `adhd_microbiome`: ADHD + microbiome/gut-brain
- `autism_microbiome`: Autism + microbiome
- `neurodevelopmental_multiomics`: Neurodevelopmental + multi-omics
- `adhd_neuroimaging`: ADHD + MRI/fMRI
- `autism_neuroimaging`: Autism + neuroimaging

**Output:** `data/literature/papers_with_data.json`

### 2. Extract Supplementary Materials

```bash
# Extract from papers JSON
python scripts/literature/supplement_extractor.py \\
    --input data/literature/papers_with_data.json \\
    --output data/literature/supplements/

# Extract from specific PMC article
python scripts/literature/supplement_extractor.py \\
    --pmcid PMC6402513 \\
    --output data/literature/supplements/

# Extract and download files
python scripts/literature/supplement_extractor.py \\
    --input data/literature/papers_with_data.json \\
    --download \\
    --output data/literature/supplements/

# Extract from multiple PMCIDs
python scripts/literature/supplement_extractor.py \\
    --pmcids PMC6402513,PMC6454429,PMC7264946 \\
    --download \\
    --output data/literature/supplements/
```

**Extracted information:**
- Supplementary file links (Excel, CSV, TSV, etc.)
- External repository URLs (GitHub, Figshare, Zenodo, Dryad, OSF)
- GEO/SRA/EGA accessions
- Data availability statements
- Code availability statements

**Output:** `data/literature/supplements/supplementary_materials.json`

### 3. Find Dataset Mentions

```bash
# Analyze papers for dataset mentions
python scripts/literature/dataset_mention_finder.py \\
    --input data/literature/papers_with_data.json \\
    --output data/literature/

# Extract sample sizes
python scripts/literature/dataset_mention_finder.py \\
    --input data/literature/papers_with_data.json \\
    --extract-sample-sizes \\
    --output data/literature/

# Find collaboration opportunities
python scripts/literature/dataset_mention_finder.py \\
    --input data/literature/papers_with_data.json \\
    --find-collaborations \\
    --output data/literature/
```

**Extracted information:**
- Sample sizes (N = 142 participants)
- Known cohorts (UK Biobank, ABCD Study, etc.)
- Data collection descriptions
- Unpublished data indicators
- Contact information
- Data types (genomics, metabolomics, imaging, etc.)

**Output:**
- `data/literature/dataset_mentions.json`
- `data/literature/sample_sizes.csv`
- `data/literature/collaboration_opportunities.csv`

## Complete Workflow

### End-to-End Data Discovery

```bash
# Step 1: Search PubMed for autism metabolomics papers
python scripts/literature/pubmed_miner.py \\
    --query-template autism_metabolomics \\
    --years 2015:2024 \\
    --has-data \\
    --max-results 500 \\
    --email your.email@institution.edu \\
    --output data/literature/

# Output: data/literature/papers_with_data.json (47 papers)

# Step 2: Extract supplementary materials
python scripts/literature/supplement_extractor.py \\
    --input data/literature/papers_with_data.json \\
    --download \\
    --output data/literature/supplements/

# Output: data/literature/supplements/supplementary_materials.json
#         Downloaded files in data/literature/supplements/{PMCID}/

# Step 3: Analyze for dataset mentions
python scripts/literature/dataset_mention_finder.py \\
    --input data/literature/papers_with_data.json \\
    --extract-sample-sizes \\
    --find-collaborations \\
    --output data/literature/

# Output: data/literature/dataset_mentions.json
#         data/literature/sample_sizes.csv
#         data/literature/collaboration_opportunities.csv

# Step 4: Review results
cat data/literature/mining_summary.json | jq '.papers_with_github'
cat data/literature/sample_sizes.csv | head -20
```

### Targeted Search for Specific Studies

```bash
# Search for ABCD Study papers
python scripts/literature/pubmed_miner.py \\
    --query "ABCD Study" OR "Adolescent Brain Cognitive Development" \\
    --years 2015:2024 \\
    --output data/literature/abcd/

# Search for PGC papers
python scripts/literature/pubmed_miner.py \\
    --query "Psychiatric Genomics Consortium" OR "PGC" \\
    --pub-types "Meta-Analysis" \\
    --output data/literature/pgc/

# Search for UK Biobank ADHD/autism papers
python scripts/literature/pubmed_miner.py \\
    --query "UK Biobank" AND \(ADHD OR autism\) \\
    --output data/literature/ukbiobank/
```

## Output Files

### papers_with_data.json

```json
{
  "metadata": {
    "generated_date": "2025-09-30",
    "search_queries": ["autism metabolomics"],
    "total_papers": 47,
    "papers_with_pmc": 38,
    "papers_with_github": 12
  },
  "papers": [
    {
      "pmid": "30478444",
      "pmcid": "PMC6381348",
      "title": "Discovery of the first genome-wide significant risk loci for ADHD",
      "abstract": "...",
      "authors": [...],
      "journal": "Nature Genetics",
      "publication_date": "2019-01",
      "doi": "10.1038/s41588-018-0269-7",
      "has_supplementary": true,
      "supplementary_links": ["..."],
      "data_repositories": {"figshare": ["..."]},
      "github_repos": [],
      "author_emails": ["ditte.demontis@clin.au.dk"],
      "full_text_available": true
    }
  ]
}
```

### author_contacts.csv

| pmid | corresponding_author | email | institution | research_area | collaboration_interest |
|------|---------------------|-------|-------------|---------------|----------------------|
| 30478444 | Demontis D | ditte.demontis@clin.au.dk | Aarhus University | ADHD genetics | data_available |
| 29795809 | Knight R | robknight@ucsd.edu | UC San Diego | Microbiome | data_available |

### Sample Data Inventory (Example)

**From pubmed_miner output:**
- 47 papers with data availability indicators
- 38 papers with PMC full text
- 12 papers with GitHub repositories
- 31 papers with data repository accessions (GEO, SRA, Figshare, etc.)
- 8 papers indicating collaboration opportunities

**Repository breakdown:**
- Figshare: 18 papers
- GitHub: 12 papers
- SRA/BioProject: 14 papers
- GEO: 8 papers
- Zenodo: 4 papers

## Data Availability Indicators

### What the System Detects

**Repository accessions:**
- GEO: `GSE123456`
- SRA: `PRJNA123456`, `SRP123456`
- EGA: `EGAS000001`, `EGAD000001`
- dbGaP: `phs000123`
- ArrayExpress: `E-MTAB-1234`

**External platforms:**
- GitHub: `github.com/username/repo`
- Figshare: `figshare.com/articles/.../12345`
- Zenodo: `zenodo.org/record/12345`
- Dryad: `datadryad.org/stash/dataset/12345`
- OSF: `osf.io/abc123`

**Data availability statements:**
- "Data are available at..."
- "Supplementary data"
- "Code is available at..."
- "Upon reasonable request"
- "Collaboration is welcome"

## Known Cohorts Detected

The system recognizes 30+ major cohorts:
- ABCD (Adolescent Brain Cognitive Development)
- UK Biobank
- All of Us
- Generation R
- ALSPAC (Avon Longitudinal Study)
- IMAGEN
- iPSYCH
- SPARK (Simons Foundation)
- SSC (Simons Simplex Collection)
- AGRE (Autism Genetic Resource Exchange)
- American Gut Project
- GTEx
- ENCODE
- TOPMed
- gnomAD

## Advanced Features

### Custom Search Queries

```python
# Build custom query programmatically
from scripts.literature.pubmed_miner import PubMedMiner

miner = PubMedMiner(Path("data/literature"), email="your@email.edu")

# Complex query
query = '''
(autism[Title/Abstract] OR ASD[Title/Abstract])
AND (metabolomics[Title/Abstract] OR metabolite[Title/Abstract])
AND (SCFA[Title/Abstract] OR "short chain fatty acid"[Title/Abstract])
AND ("2020"[Date - Publication] : "2024"[Date - Publication])
'''

pmids = miner.search_pubmed(query, retmax=200)
papers = miner.fetch_paper_details(pmids)
papers_with_data = miner.filter_papers_with_data(papers)
```

### Extract Specific File Types

```python
# Filter for Excel/CSV files only
from scripts.literature.supplement_extractor import SupplementExtractor

extractor = SupplementExtractor(Path("data/literature/supplements"))
result = extractor.extract_from_pmc("PMC6402513")

# Filter supplementary files
excel_files = [f for f in result['supplementary_files'] if f['type'] == 'excel']
tabular_files = [f for f in result['supplementary_files'] if f['type'] == 'tabular']
```

### Sample Size Analysis

```python
import json
import pandas as pd

# Load dataset mentions
with open('data/literature/dataset_mentions.json') as f:
    mentions = json.load(f)

# Extract all sample sizes
sample_sizes = []
for paper in mentions:
    for size in paper['sample_sizes']:
        sample_sizes.append({
            'pmid': paper['pmid'],
            'title': paper['title'],
            'sample_size': size['sample_size'],
            'context': size['context']
        })

df = pd.DataFrame(sample_sizes)
print(f"Median sample size: {df['sample_size'].median()}")
print(f"Largest cohort: {df['sample_size'].max()}")
```

## Integration with Other Tools

### Link to Genetics Data

```python
# Cross-reference papers with genetic databases
import json

with open('data/literature/papers_with_data.json') as f:
    papers = json.load(f)['papers']

with open('data/genetics/available_studies.json') as f:
    genetic_studies = json.load(f)['studies']

# Match by PMID
genetic_paper_pmids = {s['pmid'] for s in genetic_studies if 'pmid' in s}
literature_pmids = {p['pmid'] for p in papers}

overlap = genetic_paper_pmids & literature_pmids
print(f"Found {len(overlap)} papers in both databases")
```

### Build Contact Network

```python
import pandas as pd
import networkx as nx

# Load author contacts
contacts = pd.read_csv('data/literature/author_contacts.csv')

# Build collaboration network
G = nx.Graph()

for _, row in contacts.iterrows():
    G.add_node(row['corresponding_author'],
               email=row['email'],
               institution=row['institution'],
               research_area=row['research_area'])

# Add edges for co-authorship (would need to parse author lists)
# This is a simplified example
```

## Troubleshooting

### NCBI API Rate Limits

```python
# If you hit rate limits, add delays
import time

for pmid in pmids:
    paper = fetch_paper_details([pmid])
    time.sleep(0.5)  # 500ms delay
```

### PMC Access Issues

```bash
# Some articles are not in PMC
# Check if PMCID exists before extracting

python supplement_extractor.py --pmcid PMC123456
# If fails: "PMC article not available"
```

### Large Result Sets

```bash
# For >1000 results, use batching
python pubmed_miner.py --query autism genetics --max-results 500
# Then search again with different date range
python pubmed_miner.py --query autism genetics --years 2020:2024 --max-results 500
```

### Missing spaCy Model

```bash
# If spaCy not installed
pip install spacy
python -m spacy download en_core_web_sm

# Or run without NLP features
# dataset_mention_finder will work without spaCy but with limited entity extraction
```

## Citation Network Analysis

To build citation networks (future feature):
- Use NCBI PubMed Link API
- Query Semantic Scholar API
- Parse PMC references section

```bash
# Example: Get papers citing a key study
# This would require additional API integration
python citation_analyzer.py --pmid 30478444 --depth 2
```

## Data Privacy and Ethics

### When Contacting Authors

- **Always be professional** in initial contact
- **Explain your research** and data needs clearly
- **Respect data usage agreements**
- **Cite their work** appropriately
- **Offer collaboration** if appropriate

### Email Template

```
Subject: Request for [Dataset Name] Data Access

Dear Dr. [Last Name],

I am a researcher at [Your Institution] studying [your research topic].
I came across your publication "[Paper Title]" (PMID: [PMID]) and am
interested in accessing the [specific dataset/supplementary data].

[Briefly explain your research and why you need the data]

I understand the data may be subject to usage agreements and am happy
to comply with any requirements. Would it be possible to discuss data
access?

Thank you for your consideration.

Best regards,
[Your Name]
[Your Institution]
[Your Email]
```

## References

### PubMed/PMC Resources
- PubMed: https://pubmed.ncbi.nlm.nih.gov/
- PMC: https://www.ncbi.nlm.nih.gov/pmc/
- E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- Entrez API docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/

### Data Repositories
- Figshare: https://figshare.com
- Zenodo: https://zenodo.org
- Dryad: https://datadryad.org
- OSF: https://osf.io
- GEO: https://www.ncbi.nlm.nih.gov/geo/
- SRA: https://www.ncbi.nlm.nih.gov/sra

## Support

For questions or issues:
1. Check this README and output JSON files
2. Verify NCBI API access with your email
3. Review rate limits and quotas
4. Open GitHub issue with detailed description

---

**Last updated**: 2025-09-30
**Version**: 1.0
**Maintained by**: AuDHD Correlation Study Team