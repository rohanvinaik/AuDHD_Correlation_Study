# Enhanced Paper Scraping System

Comprehensive data lead extraction from scientific literature with **proper citation tracking**.

**NO DATA THEFT** - All extracted information is properly attributed to source papers.

## Overview

The enhanced scraping system extracts, validates, and scores **every** actionable lead to external data artifacts (datasets, code, models) from scientific papers - even if only mentioned in prose.

### Pipeline Architecture

```
Ingest → Structure → Extract → Normalize → Validate →
Classify → Score → Track Citations → Output
```

## Key Features

### 1. Text Structuring
- Segments papers into labeled sections (Title, Abstract, Methods, Data Availability, etc.)
- Normalizes typography (fixes soft hyphens, ligatures, line breaks)
- Handles column-order issues from PDF extraction
- Repairs broken identifiers from line wraps

### 2. Comprehensive Pattern Library

**Genomics:**
- GEO (GSE/GSM/GPL)
- SRA (SRR/SRX/SRP/SRS and international variants)
- BioProject (PRJNA/PRJEB/PRJDB)
- dbGaP (phs...)
- EGA (EGAS/EGAD)
- ArrayExpress (E-*)
- BioSample (SAMN/SAMEA/SAMD)

**Proteomics:**
- PRIDE/ProteomeXchange (PXD)
- MassIVE (MSV)

**Metabolomics:**
- Metabolomics Workbench (ST...)
- MetaboLights (MTBLS...)

**Imaging/Neuro:**
- OpenNeuro (ds...)
- NeuroVault
- NITRC, HCP, UK Biobank patterns

**General Repositories:**
- Zenodo, Figshare, Dryad, OSF, Mendeley Data
- DataCite/Crossref DOIs

**Code:**
- GitHub, GitLab, Bitbucket repos
- Release DOIs (Zenodo linkbacks)

**Other:**
- Addgene plasmids
- PDB, UniProt
- BioStudies, EMPIAR

### 3. Lexical Triggers

Context words that indicate data availability:
- **Deposit triggers:** "deposited in", "submitted to", "available at", "accessible at"
- **Accession triggers:** "accession", "under accession", "study ID", "project ID"
- **Request triggers:** "upon reasonable request", "from the corresponding author"

### 4. Validation & Enrichment

**API-first validation** (not scraping):
- NCBI E-utilities (GEO/SRA/BioProject)
- DataCite API for data DOIs
- GitHub API for code repos
- URL resolution for general repos

**Post-publication discovery:**
- DataCite relation queries to find datasets minted after publication

### 5. Classification System

**Access Levels:**
- `verified_public` - API confirmed, publicly accessible
- `mentioned_resolvable` - URL/DOI reachable but not API-verified
- `restricted` - dbGaP/EGA/UK Biobank (requires application)
- `request_only` - "Upon reasonable request"
- `dead_link` - 404/timeout/failed validation
- `ambiguous` - Looks like ID but failed validation

**Lead Types:**
- `dataset` - Raw data repositories
- `code` - Code repositories
- `model` - Materials/models (Addgene, etc.)

### 6. Confidence Scoring

```python
confidence = (
    section_weight * 0.3 +      # Data Availability > Methods > elsewhere
    context_weight * 0.3 +       # Has deposit/accession triggers?
    validation_weight * 0.4      # API verified?
)
```

Scores range 0-1, leads sorted by confidence.

### 7. Provenance Tracking

**Full audit trail:**
- Text snippet with section label
- Page/line offsets (when available)
- Context words (triggers found)
- Original mention vs. normalized ID
- API responses and redirect chains
- Paper variant (preprint, accepted, VoR)
- Extraction date

### 8. Citation Tracking

**Mandatory attribution for all extracted data:**
- Full paper citation (APA format)
- DOI/PMID/PMC links
- Direct quote from Data Availability Statement
- List of all extracted identifiers
- License information (when available)
- Extraction date

## Usage

### Basic Analysis

```python
from scripts.scrape_papers_enhanced import EnhancedPaperScraper

# Initialize scraper
scraper = EnhancedPaperScraper(output_dir="data/papers_enhanced")

# Analyze paper
paper_metadata = {
    'pmid': '12345678',
    'doi': '10.1234/journal.2024.001',
    'title': 'Example Paper',
    'authors': ['Smith J', 'Doe A'],
    'year': '2024',
    'journal': 'Nature Methods'
}

# With full text
with open('paper.txt') as f:
    paper_text = f.read()

analysis = scraper.analyze_paper(paper_text, paper_metadata)

# Results
print(f"Found {len(analysis.leads)} data leads")
for lead in analysis.leads:
    print(f"  {lead.identifier} ({lead.repository})")
    print(f"    Access: {lead.access_level}")
    print(f"    Confidence: {lead.confidence:.2f}")
    print(f"    Status: {lead.validation_status}")

# Save results
scraper.save_analysis(analysis)
```

### Command Line

```bash
# Analyze from text file
python scripts/scrape_papers_enhanced.py \
  --text-file paper.txt \
  --output data/papers_enhanced

# Audit mode (extract and validate only)
python scripts/scrape_papers_enhanced.py \
  --text-file paper.txt \
  --audit-only
```

## Output Structure

```
data/papers_enhanced/
├── leads/
│   ├── 12345678_leads.json       # Complete analysis
│   └── ...
├── citations/
│   ├── 12345678_citation.txt     # Human-readable attribution
│   └── ...
├── cache/
│   └── [API response cache]
└── ATTRIBUTIONS.md               # Master attribution file
```

### Lead JSON Format

```json
{
  "paper_id": "12345678",
  "citation": {
    "paper_doi": "10.1234/journal.2024.001",
    "paper_pmid": "12345678",
    "paper_title": "Example Paper",
    "authors": ["Smith J", "Doe A"],
    "citation_text": "Smith J, Doe A (2024). Example Paper. Nature Methods. https://doi.org/10.1234/journal.2024.001",
    "data_statement": "All data deposited in GEO under accession GSE123456...",
    "extracted_leads": ["GSE123456", "PRJNA123456"],
    "extraction_date": "2025-09-30T14:00:00"
  },
  "leads": [
    {
      "identifier": "GSE123456",
      "lead_type": "dataset",
      "repository": "GEO",
      "access_level": "verified_public",
      "validation_status": "verified_public",
      "confidence": 0.95,
      "evidence": [
        {
          "text_snippet": "...data deposited in Gene Expression Omnibus under accession GSE123456...",
          "section": "data_availability",
          "context_words": ["deposited in", "accession"]
        }
      ],
      "title": "RNA-seq of autism brain samples",
      "api_metadata": {"api_response": "GEO API confirmed"},
      "original_mention": "GSE123456",
      "paper_variant": "VoR",
      "discovered_date": "2025-09-30T14:00:00"
    }
  ],
  "sections_analyzed": ["abstract", "methods", "data_availability", "references"],
  "scrape_date": "2025-09-30T14:00:00",
  "version": "1.0"
}
```

## Citation Output

Each paper gets a citation file for proper attribution:

```
Paper: Example Paper
Citation: Smith J, Doe A (2024). Example Paper. Nature Methods. https://doi.org/10.1234/journal.2024.001

Data Availability Statement:
All RNA-seq data have been deposited in the Gene Expression Omnibus under accession GSE123456.
Code is available at github.com/smithlab/analysis-pipeline.

Data Leads Extracted (2):
- GSE123456 (GEO) - verified_public
- github.com/smithlab/analysis-pipeline (GitHub) - verified_public
```

## Integration with Main Pipeline

After paper scraping, integrate discovered datasets:

```python
from scripts.scrape_papers_enhanced import EnhancedPaperScraper

# 1. Scrape papers
scraper = EnhancedPaperScraper("data/papers_enhanced")
# ... analyze papers ...

# 2. Extract verified public datasets
analyses = []  # Load all analyses
verified_datasets = []

for analysis in analyses:
    for lead in analysis.leads:
        if lead.access_level == 'verified_public' and lead.lead_type == 'dataset':
            verified_datasets.append({
                'identifier': lead.identifier,
                'repository': lead.repository,
                'source_paper': analysis.citation.citation_text,
                'confidence': lead.confidence
            })

# 3. Download verified datasets
for dataset in verified_datasets:
    if dataset['repository'] == 'GEO':
        # Use existing GEO download logic
        download_geo_dataset(dataset['identifier'])
    elif dataset['repository'] == 'SRA':
        download_sra_dataset(dataset['identifier'])
    # ... etc.

# 4. Log attribution
with open('data/papers_enhanced/USED_DATASETS.md', 'w') as f:
    f.write("# Datasets Used in Analysis\n\n")
    for dataset in verified_datasets:
        f.write(f"## {dataset['identifier']}\n")
        f.write(f"**Source:** {dataset['source_paper']}\n")
        f.write(f"**Repository:** {dataset['repository']}\n")
        f.write(f"**Confidence:** {dataset['confidence']:.2f}\n\n")
```

## Advanced Features

### Section Weighting

Different sections have different reliability:

| Section | Weight | Rationale |
|---------|--------|-----------|
| Data Availability | 1.0 | Explicit statements |
| Code Availability | 1.0 | Explicit statements |
| Methods | 0.8 | Often contains accessions |
| Supplementary | 0.7 | Extended details |
| Figure Captions | 0.5 | Source Data mentions |
| Results | 0.6 | Sometimes has inline refs |
| Acknowledgements | 0.3 | May mention repos |
| References | 0.2 | Sometimes DOIs embedded |

### Identifier Normalization

Handles common issues:
- Line breaks splitting IDs: `GSE 12345` → `GSE12345`
- Hyphenation: `PRJ-NA123456` → `PRJNA123456`
- Spaces in project IDs: `PRJN A 123456` → `PRJNA123456`
- Case inconsistency: `gse12345` → `GSE12345`
- Unicode issues: Em dashes, ligatures, etc.

### Restricted Data Detection

Automatically classifies restricted-access repositories:
- **dbGaP** (phs...) - NIH controlled access
- **EGA** (EGAS/EGAD) - EBI controlled access
- **UK Biobank** - Application required
- **NDA/NDAR** - NIMH Data Archive

Still validates study IDs but marks as `restricted` with application info.

### Request-Only Detection

Identifies papers with "upon reasonable request" clauses:
- Marks as `request_only`
- Lower confidence score (0.3-0.5)
- Includes contact info when extractable

## Common Extraction Scenarios

### Scenario 1: Data Availability Statement

```
Input:
"Data Availability: All sequencing data have been deposited in the
Gene Expression Omnibus under accession GSE123456."

Output:
- Identifier: GSE123456
- Repository: GEO
- Access: verified_public (API confirmed)
- Confidence: 0.95
- Section: data_availability
- Triggers: ["deposited in", "accession"]
```

### Scenario 2: Methods Section Reference

```
Input:
"Raw reads were downloaded from SRA (BioProject PRJNA123456)
and processed using..."

Output:
- Identifier: PRJNA123456
- Repository: BioProject
- Access: verified_public
- Confidence: 0.82
- Section: methods
- Triggers: ["downloaded from"]
```

### Scenario 3: Figure Caption

```
Input:
"Figure 1. Source Data available at zenodo.org/record/123456"

Output:
- Identifier: zenodo.org/record/123456
- Repository: Zenodo
- Access: verified_public
- Confidence: 0.68
- Section: figure_captions
- Triggers: ["available at"]
```

### Scenario 4: Restricted Access

```
Input:
"Genotype data are available through dbGaP under accession phs001234"

Output:
- Identifier: phs001234
- Repository: dbGaP
- Access: restricted
- Confidence: 0.75
- Note: "Requires access application to dbGaP"
```

### Scenario 5: Dead Link

```
Input:
"Data available at http://labwebsite.edu/data/dataset1"
[URL returns 404]

Output:
- Identifier: http://labwebsite.edu/data/dataset1
- Repository: Unknown
- Access: dead_link
- Confidence: 0.2
- Validation: "HTTP 404"
```

## Evaluation

### Test Set Construction

Build gold standard with ~50 papers:
- Varied publishers (Nature, Cell, PLOS, PubMed Central, etc.)
- Different access types (public, restricted, request-only)
- Various repositories (GEO, SRA, Zenodo, GitHub, etc.)
- Edge cases (broken IDs, dead links, post-pub DOIs)

### Metrics

**Precision/Recall on:**
1. Detecting any lead (vs. no lead)
2. Producing verified link (vs. mentioned only)
3. Correct access classification

**Expected Performance:**
- Precision: >0.90 (few false positives)
- Recall: >0.85 (most leads found)
- Verification rate: >0.70 (most leads API-confirmed)

## Legal and Ethical Considerations

### ✅ Proper Attribution

**REQUIRED for all extracted data:**
- Full paper citation
- Direct quote from Data Availability Statement
- List of extracted identifiers
- Clear provenance trail

### ✅ Respecting Access Restrictions

**System behavior:**
- Identifies restricted-access repositories
- Does NOT attempt to bypass access controls
- Provides application information
- Marks as `restricted` with clear notes

### ✅ Rate Limiting

**API usage:**
- NCBI: 3 requests/second (respects guidelines)
- GitHub: API key recommended for heavy use
- 1-second delays between papers
- Caching to minimize repeat requests

### ⚠️ Important Notes

1. **NO DATA THEFT** - This system extracts **metadata** about where data is, not the data itself
2. **Proper Citations** - Always include source paper citations when using discovered datasets
3. **Check Licenses** - Some datasets have specific usage restrictions
4. **Respect Restrictions** - Don't attempt to access restricted data without proper authorization
5. **Give Credit** - When publishing, cite both the dataset AND the paper that led you to it

## Future Enhancements

### Near-term
- [ ] Full PMC integration (XML parsing)
- [ ] Preprint vs VoR comparison
- [ ] DataCite relation queries
- [ ] Wayback Machine for dead links
- [ ] Multi-version tracking

### Long-term
- [ ] GROBID integration for better structure
- [ ] Supplementary PDF parsing
- [ ] Multi-language support
- [ ] Publisher-specific handlers (Nature, Cell, etc.)
- [ ] Automated dataset downloading

## References

- **NCBI E-utilities:** https://www.ncbi.nlm.nih.gov/books/NBK25501/
- **DataCite API:** https://support.datacite.org/docs/api
- **Scholexplorer:** https://scholexplorer.openaire.eu/
- **Repository Pattern Examples:** See RepositoryPatterns class in code

## Citation of This Tool

When using this tool in research, please cite:

```
Enhanced Paper Scraper (2025). AuDHD Correlation Study Pipeline.
https://github.com/rohanvinaik/AuDHD_Correlation_Study
```

---

**Remember:** This tool helps you discover data. Always properly attribute source papers and respect access restrictions!
