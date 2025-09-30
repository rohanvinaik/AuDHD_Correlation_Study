# Reference Database Management System

Comprehensive tools for downloading, indexing, and managing biomedical ontologies, pathway databases, and interaction networks for ADHD/Autism research.

## Overview

This system provides automated access to 15+ biomedical ontologies and databases, enabling standardized annotation, pathway enrichment, drug-gene mapping, and network analysis.

### Key Components

1. **Ontology Downloader** (`ontology_downloader.py`)
   - HPO, GO, MONDO, DO, CHEBI, UBERON
   - ClinVar, Orphanet, SFARI Gene
   - RxNorm, ATC, ICD-10, FNDDS

2. **Database Indexer** (`database_indexer.py`)
   - Reactome, WikiPathways, KEGG pathways
   - STRING, BioGRID protein interactions
   - DGIdb, STITCH drug-gene interactions

3. **Unified SQLite Database** (`pathway_database.db`)
   - Pathway membership
   - Protein interaction networks
   - Drug-gene relationships
   - Pathway hierarchies

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install requests pandas pyyaml networkx

# For OBO format parsing (optional)
pip install pronto obonet

# For large file handling
pip install tqdm
```

### Configuration

Edit your email for NCBI API access:

```bash
# In ontology_downloader.py
Entrez.email = "your.email@institution.edu"
```

For UMLS-protected resources (RxNorm, SNOMED), register at:
https://uts.nlm.nih.gov/uts/signup-login

## Usage

### 1. Download Ontologies

```bash
# Download all open-access ontologies
python scripts/references/ontology_downloader.py \
    --download-all \
    --output data/references/ontologies/

# Download specific ontology
python scripts/references/ontology_downloader.py \
    --ontology HPO \
    --output data/references/ontologies/

# Update existing ontologies
python scripts/references/ontology_downloader.py \
    --update \
    --output data/references/ontologies/

# Check versions
python scripts/references/ontology_downloader.py \
    --check-versions \
    --output data/references/ontologies/

# List available ontologies
python scripts/references/ontology_downloader.py --list-available
```

### 2. Extract ADHD/Autism Terms

```bash
# Extract relevant HPO terms
python scripts/references/ontology_downloader.py \
    --extract-terms HPO \
    --output data/references/ontologies/

# Extract relevant GO terms
python scripts/references/ontology_downloader.py \
    --extract-terms GO \
    --output data/references/ontologies/
```

### 3. Index Pathway Databases

```bash
# Download and index all databases
python scripts/references/database_indexer.py \
    --download-all \
    --output data/references/

# Index specific database
python scripts/references/database_indexer.py \
    --database REACTOME \
    --output data/references/

python scripts/references/database_indexer.py \
    --database STRING \
    --output data/references/

python scripts/references/database_indexer.py \
    --database DGIDB \
    --output data/references/

# Show database statistics
python scripts/references/database_indexer.py \
    --stats \
    --database data/references/pathway_database.db
```

### 4. Query Database

```bash
# Query pathways for a gene
python scripts/references/database_indexer.py \
    --query-gene DRD4 \
    --database data/references/pathway_database.db

# Query protein interactions
python scripts/references/database_indexer.py \
    --query-gene SLC6A3 \
    --database data/references/pathway_database.db

# Query drug targets
python scripts/references/database_indexer.py \
    --query-drug methylphenidate \
    --database data/references/pathway_database.db
```

## Available Ontologies

### Clinical Phenotypes

#### Human Phenotype Ontology (HPO)
- **Terms**: 16,743 phenotypic abnormalities
- **Key terms**: Hyperactivity (HP:0000752), ADHD (HP:0007018), Autism (HP:0000717)
- **Use cases**: Phenotype annotation, clinical similarity, comorbidity analysis
- **License**: Open
- **Update**: Monthly

```python
# Example: Extract ADHD phenotypes
import json
with open('data/references/ontologies/hpo_adhd_autism_terms.json') as f:
    hpo_terms = json.load(f)

adhd_terms = [t for t in hpo_terms if 'adhd' in t['name'].lower()]
print(f"Found {len(adhd_terms)} ADHD-related phenotypes")
```

### Gene Function

#### Gene Ontology (GO)
- **Terms**: 44,945 (BP: 30,363, MF: 12,144, CC: 4,438)
- **Key terms**: Synaptic transmission, neurotransmitter secretion, learning, memory
- **Use cases**: Pathway enrichment, functional annotation
- **License**: CC BY 4.0
- **Update**: Monthly

```python
# Example: GO enrichment analysis
from scipy.stats import hypergeom

def go_enrichment(gene_list, go_term_genes, total_genes):
    """Hypergeometric test for GO enrichment"""
    k = len(set(gene_list) & set(go_term_genes))  # Overlap
    M = total_genes  # Total genes
    n = len(go_term_genes)  # Genes in GO term
    N = len(gene_list)  # Genes in query

    p_value = hypergeom.sf(k-1, M, n, N)
    return p_value
```

### Disease Classification

#### Monarch Disease Ontology (MONDO)
- **Terms**: 23,417 diseases
- **ADHD**: MONDO:0007743
- **Autism**: MONDO:0005258
- **Use cases**: Disease mapping, comorbidity networks
- **License**: CC BY 4.0

### Chemicals and Drugs

#### Chemical Entities of Biological Interest (CHEBI)
- **Terms**: 61,480 (12,000 drugs, 23,000 metabolites)
- **Key compounds**: Dopamine, serotonin, GABA, methylphenidate, butyrate
- **Use cases**: Drug annotation, metabolite identification
- **License**: CC BY 4.0
- **Size**: 156 MB (large file)

```python
# Example: Map drug to CHEBI ID
drug_mappings = {
    'methylphenidate': 'CHEBI:49575',
    'amphetamine': 'CHEBI:2679',
    'atomoxetine': 'CHEBI:127342',
    'dopamine': 'CHEBI:18243',
    'serotonin': 'CHEBI:28790',
    'gaba': 'CHEBI:16865'
}
```

### Clinical Databases

#### SFARI Gene
- **Genes**: 1,318 autism candidate genes
- **Score 1**: 214 high-confidence genes
- **Key genes**: CHD8, SCN2A, ADNP, ARID1B, SHANK3
- **Use cases**: Autism gene prioritization
- **License**: Open
- **Note**: Manual download from https://gene.sfari.org

#### ClinVar
- **Variants**: 2,438,916 total
- **ADHD variants**: 347
- **Autism variants**: 1,248
- **Use cases**: Variant clinical significance
- **License**: Open
- **Size**: 1.8 GB

## Pathway Databases

### Reactome
- **Pathways**: 2,877 curated pathways
- **Reactions**: 11,389
- **Proteins**: 11,388
- **License**: CC BY 4.0

**Key pathways for ADHD/Autism:**
```
R-HSA-112316: Neuronal System
R-HSA-112315: Transmission across Chemical Synapses
R-HSA-209931: Serotonin and melatonin biosynthesis
R-HSA-209968: Dopamine catabolism
R-HSA-888590: GABA synthesis, release, reuptake and degradation
```

**Download and query:**
```python
import sqlite3

# Query Reactome pathways
conn = sqlite3.connect('data/references/pathway_database.db')
cursor = conn.cursor()

# Get pathways for DRD4
cursor.execute("""
    SELECT p.pathway_name, pg.evidence_code
    FROM pathways p
    JOIN pathway_genes pg ON p.pathway_id = pg.pathway_id
    WHERE pg.gene_symbol = 'DRD4'
""")

for pathway, evidence in cursor.fetchall():
    print(f"{pathway} ({evidence})")
```

### KEGG
- **Pathways**: 372 human pathways
- **License**: Academic use free, commercial requires license
- **Access**: REST API with rate limits

**Key pathways:**
```
hsa04728: Dopaminergic synapse
hsa04726: Serotonergic synapse
hsa04727: GABAergic synapse
hsa04724: Glutamatergic synapse
hsa04360: Axon guidance
hsa00380: Tryptophan metabolism
```

**API usage:**
```python
import requests

# Get pathway info
kegg_api = "https://rest.kegg.jp"
pathway_id = "hsa04728"  # Dopaminergic synapse

# Get pathway genes
response = requests.get(f"{kegg_api}/link/genes/{pathway_id}")
genes = response.text.strip().split('\n')
print(f"Pathway contains {len(genes)} genes")
```

## Interaction Databases

### STRING Protein-Protein Interactions
- **Proteins**: 19,354 human proteins
- **Interactions**: 11,759,454 edges
- **Confidence**: 150-999 (recommend >400)
- **Evidence**: Experimental, database, coexpression, text mining
- **License**: CC BY 4.0
- **Size**: 1.8 GB

**Query interactions:**
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/references/pathway_database.db')

# Get DRD4 interaction partners
query = """
    SELECT protein_b as partner, confidence_score
    FROM protein_interactions
    WHERE protein_a = 'DRD4' AND confidence_score > 0.7
    ORDER BY confidence_score DESC
"""

df = pd.read_sql_query(query, conn)
print(f"DRD4 has {len(df)} high-confidence interactors")
print(df.head(10))
```

### DGIdb Drug-Gene Interactions
- **Interactions**: 106,797
- **Drugs**: 13,350
- **Genes**: 9,572
- **Types**: Inhibitor, agonist, antagonist, modulator
- **License**: CC0 1.0

**Query ADHD drug targets:**
```python
adhd_drugs = ['methylphenidate', 'amphetamine', 'atomoxetine', 'guanfacine']

for drug in adhd_drugs:
    cursor.execute("""
        SELECT gene_symbol, interaction_type
        FROM drug_gene_interactions
        WHERE drug_name LIKE ?
    """, (f'%{drug}%',))

    targets = cursor.fetchall()
    print(f"\n{drug.title()} targets:")
    for gene, int_type in targets:
        print(f"  {gene}: {int_type}")
```

## Workflow Examples

### 1. Annotate GWAS Genes with Pathways

```python
import sqlite3
import pandas as pd

# Load GWAS significant genes
gwas_genes = ['DRD4', 'SLC6A3', 'HTR2A', 'ADGRL3', 'SNAP25']

# Query pathways
conn = sqlite3.connect('data/references/pathway_database.db')

results = []
for gene in gwas_genes:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT p.pathway_name, p.database_source
        FROM pathways p
        JOIN pathway_genes pg ON p.pathway_id = pg.pathway_id
        WHERE pg.gene_symbol = ?
    """, (gene,))

    for pathway, source in cursor.fetchall():
        results.append({
            'gene': gene,
            'pathway': pathway,
            'database': source
        })

df = pd.DataFrame(results)
df.to_csv('data/integrated/gwas_genes_pathways.csv', index=False)
print(f"Annotated {len(gwas_genes)} genes with {len(df)} pathway associations")
```

### 2. Build Protein Interaction Network

```python
import networkx as nx
import sqlite3

# Get interactions for gene list
genes = ['DRD4', 'SLC6A3', 'DRD2', 'DRD1', 'SLC6A2']

conn = sqlite3.connect('data/references/pathway_database.db')
cursor = conn.cursor()

# Query interactions
G = nx.Graph()

for gene in genes:
    cursor.execute("""
        SELECT protein_a, protein_b, confidence_score
        FROM protein_interactions
        WHERE (protein_a = ? OR protein_b = ?) AND confidence_score > 0.7
    """, (gene, gene))

    for p1, p2, score in cursor.fetchall():
        G.add_edge(p1, p2, weight=score)

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Compute centrality
centrality = nx.degree_centrality(G)
top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 hub genes:")
for gene, cent in top_hubs:
    print(f"  {gene}: {cent:.3f}")

# Export for visualization
nx.write_gexf(G, 'data/networks/dopamine_transporter_network.gexf')
```

### 3. Map Clinical Phenotypes to Genes

```python
import json

# Load HPO ADHD/autism terms
with open('data/references/ontologies/hpo_adhd_autism_terms.json') as f:
    hpo_terms = json.load(f)

# Load HPO annotations (disease-phenotype associations)
import pandas as pd

annotations = pd.read_csv(
    'data/references/ontologies/phenotype.hpoa',
    sep='\t',
    comment='#'
)

# Filter ADHD/autism
hpo_ids = [t['id'] for t in hpo_terms]
adhd_autism_annotations = annotations[
    annotations['HPO_ID'].isin(hpo_ids)
]

print(f"Found {len(adhd_autism_annotations)} disease-phenotype associations")

# Get associated genes
gene_phenotypes = {}
for _, row in adhd_autism_annotations.iterrows():
    hpo_id = row['HPO_ID']
    disease = row['DatabaseID']

    # Map disease to genes (requires additional OMIM/Orphanet gene data)
    # This is a simplified example
    print(f"{hpo_id}: {disease}")
```

### 4. Drug Mechanism Analysis

```python
import sqlite3

# Query all ADHD medication targets
adhd_drugs = {
    'stimulants': ['methylphenidate', 'amphetamine', 'dextroamphetamine'],
    'non_stimulants': ['atomoxetine', 'guanfacine', 'clonidine']
}

conn = sqlite3.connect('data/references/pathway_database.db')

mechanisms = {}

for category, drugs in adhd_drugs.items():
    mechanisms[category] = {}

    for drug in drugs:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT gene_symbol, interaction_type
            FROM drug_gene_interactions
            WHERE drug_name LIKE ?
        """, (f'%{drug}%',))

        targets = cursor.fetchall()
        mechanisms[category][drug] = targets

# Print summary
print("ADHD Medication Mechanisms:\n")
for category, drugs in mechanisms.items():
    print(f"\n{category.upper()}:")
    for drug, targets in drugs.items():
        print(f"\n  {drug.title()}:")
        for gene, int_type in targets:
            print(f"    {gene}: {int_type}")
```

## Database Schema

### SQLite Database Structure

```sql
-- Pathways
CREATE TABLE pathways (
    pathway_id TEXT PRIMARY KEY,
    pathway_name TEXT,
    database_source TEXT,
    category TEXT,
    description TEXT,
    organism TEXT DEFAULT 'Homo sapiens',
    gene_count INTEGER,
    url TEXT
);

-- Pathway membership
CREATE TABLE pathway_genes (
    pathway_id TEXT,
    gene_symbol TEXT,
    gene_id TEXT,
    evidence_code TEXT,
    FOREIGN KEY (pathway_id) REFERENCES pathways(pathway_id)
);

-- Protein interactions
CREATE TABLE protein_interactions (
    protein_a TEXT,
    protein_b TEXT,
    database_source TEXT,
    interaction_type TEXT,
    confidence_score REAL,
    evidence TEXT,
    PRIMARY KEY (protein_a, protein_b, database_source)
);

-- Drug-gene interactions
CREATE TABLE drug_gene_interactions (
    drug_name TEXT,
    drug_chembl_id TEXT,
    gene_symbol TEXT,
    gene_entrez_id TEXT,
    interaction_type TEXT,
    database_source TEXT,
    pmid TEXT
);

-- Pathway hierarchy
CREATE TABLE pathway_hierarchy (
    parent_pathway_id TEXT,
    child_pathway_id TEXT,
    FOREIGN KEY (parent_pathway_id) REFERENCES pathways(pathway_id),
    FOREIGN KEY (child_pathway_id) REFERENCES pathways(pathway_id)
);
```

## Integration with Analysis Pipeline

### With Genetics Data

```python
# Annotate GWAS results with pathways
import pandas as pd
import sqlite3

# Load GWAS results
gwas_df = pd.read_csv('data/genetics/adhd_gwas_associations.csv')

# Query pathways for each gene
conn = sqlite3.connect('data/references/pathway_database.db')

for idx, row in gwas_df.iterrows():
    gene = row['mapped_genes'].split(',')[0]  # First mapped gene

    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(DISTINCT pathway_id)
        FROM pathway_genes
        WHERE gene_symbol = ?
    """, (gene,))

    pathway_count = cursor.fetchone()[0]
    gwas_df.at[idx, 'num_pathways'] = pathway_count

gwas_df.to_csv('data/genetics/adhd_gwas_annotated.csv', index=False)
```

### With Metabolomics Data

```python
# Map metabolites to CHEBI and pathways
metabolite_mappings = {
    'dopamine': 'CHEBI:18243',
    'serotonin': 'CHEBI:28790',
    'gaba': 'CHEBI:16865',
    'butyrate': 'CHEBI:17968',
    'tryptophan': 'CHEBI:16828'
}

# Query pathways involving these metabolites
# (Requires KEGG COMPOUND or MetaCyc integration)
```

### With Microbiome Data

```python
# Link bacterial functions to metabolite pathways
bacterial_metabolites = {
    'Faecalibacterium prausnitzii': ['butyrate'],
    'Lactobacillus': ['GABA', 'serotonin'],
    'Bifidobacterium': ['GABA'],
    'Clostridium': ['4-ethylphenylsulfate']
}

# Map to human metabolic pathways
```

## Maintenance and Updates

### Update Schedule

```bash
# Monthly updates (recommended)
crontab -e

# Add line:
0 0 1 * * cd /path/to/project && python scripts/references/ontology_downloader.py --update --output data/references/ontologies/
```

### Check for Updates

```python
import requests
from datetime import datetime

def check_ontology_updates():
    """Check if new versions available"""

    # Load current versions
    import json
    with open('data/references/ontology_versions.json') as f:
        versions = json.load(f)

    # Check each ontology (simplified)
    for ont_key, ont_data in versions['ontologies'].items():
        print(f"Checking {ont_data['name']}...")
        print(f"  Current version: {ont_data['current_version']}")
        print(f"  Last download: {ont_data.get('last_local_download', 'Never')}")

check_ontology_updates()
```

### Backup Strategy

```bash
# Create backup before updates
mkdir -p data/references/backups/$(date +%Y%m%d)
cp -r data/references/ontologies data/references/backups/$(date +%Y%m%d)/
cp data/references/pathway_database.db data/references/backups/$(date +%Y%m%d)/
```

## Storage Requirements

**Total storage needed**: ~8.5 GB

**Breakdown:**
- HPO: 45 MB
- GO: 39 MB
- CHEBI: 156 MB
- ClinVar: 1.8 GB
- STRING: 1.8 GB
- FNDDS: 842 MB
- Reactome: 156 MB
- Other ontologies: ~300 MB
- SQLite database: ~500 MB
- Cache and downloads: ~2-3 GB

**Recommendations:**
- Minimum 10 GB free space
- SSD for faster database queries
- Regular cleanup of old versions

## Troubleshooting

### Large File Downloads

```python
# For very large files, use streaming with progress bar
import requests
from tqdm import tqdm

url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
output_file = "data/references/interactions/string_links.txt.gz"

response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(output_file, 'wb') as f:
    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
```

### OBO Parsing Issues

```bash
# If pronto fails, use obonet
pip install obonet

# Parse OBO
import obonet
graph = obonet.read_obo('data/references/ontologies/hp.obo')
print(f"Loaded {len(graph)} terms")
```

### Database Locking

```python
# Set timeout for busy database
import sqlite3

conn = sqlite3.connect('data/references/pathway_database.db', timeout=30.0)
```

### API Rate Limits

```python
import time

# KEGG API rate limiting
def kegg_api_call(endpoint):
    response = requests.get(f"https://rest.kegg.jp/{endpoint}")
    time.sleep(1)  # 1 second delay
    return response
```

## References

### Ontologies
- HPO: Köhler et al. (2021) Nucleic Acids Res. PMID: 33264411
- GO: Gene Ontology Consortium (2023) Nucleic Acids Res. PMID: 33290552
- MONDO: Vasilevsky et al. (2022) Med. PMID: 36458984
- CHEBI: Hastings et al. (2016) Nucleic Acids Res. PMID: 26467479

### Pathways
- KEGG: Kanehisa et al. (2023) Nucleic Acids Res. PMID: 36300620
- Reactome: Gillespie et al. (2022) Nucleic Acids Res. PMID: 34788843
- WikiPathways: Agrawal et al. (2023) Nucleic Acids Res. PMID: 37941846

### Interactions
- STRING: Szklarczyk et al. (2023) Nucleic Acids Res. PMID: 36370105
- BioGRID: Oughtred et al. (2021) Nucleic Acids Res. PMID: 33290552
- DGIdb: Freshour et al. (2021) Nucleic Acids Res. PMID: 33237278

### Clinical
- ClinVar: Landrum et al. (2018) Nucleic Acids Res. PMID: 29165669
- SFARI Gene: Abrahams et al. (2013) Nat Neurosci. PMID: 24162654

## Support

For questions or issues:
1. Check this README and ontology_versions.json
2. Review configuration in configs/reference_paths.yaml
3. Open GitHub issue with detailed description
4. Contact repository maintainers

---

**Last updated**: 2025-09-30
**Version**: 1.0
**Maintained by**: AuDHD Correlation Study Team