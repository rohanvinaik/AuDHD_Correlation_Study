# Patient Registry and Biobank Access for ADHD/Autism Research

Comprehensive catalog of patient registries and biobanks with biological samples for ADHD/autism research, including brain tissue repositories, DNA collections, and cell line resources.

## Overview

This system catalogs and facilitates access to:
1. **Patient Registries**: SPARK, IAN, AGRE, SSC, EU-AIMS, PGC, iPSYCH, ABCD
2. **Brain Tissue Biobanks**: Autism BrainNet, NIH NeuroBioBank, UK Brain Banks
3. **DNA/Cell Line Collections**: NIMH Repository, Coriell, SPARK Biobank
4. **iPSC Resources**: Autism-derived stem cell lines for neuronal models

### Key Components

1. **Registry Crawler** (`registry_crawler.py`)
   - Catalogs 17 major ADHD/autism patient registries
   - Filters by condition, data type, access level
   - Tracks enrollment status and sample sizes
   - Provides contact and application information

2. **Biobank Finder** (`biobank_finder.py`)
   - Inventories 13 biobanks with biological samples
   - Identifies brain tissue repositories
   - Catalogs DNA, cell line, and iPSC collections
   - Details access procedures and turnaround times

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install requests pandas
```

No API keys required - this is a catalog/reference system.

## Usage

### 1. Search Patient Registries

```bash
# Catalog all registries
python scripts/registries/registry_crawler.py \\
    --output data/registries/

# Find autism registries with genomics
python scripts/registries/registry_crawler.py \\
    --condition autism \\
    --data-type genomics \\
    --output data/registries/

# Find active registries with open data access
python scripts/registries/registry_crawler.py \\
    --access-type open \\
    --active-only \\
    --output data/registries/

# Find large cohorts (N > 10,000)
python scripts/registries/registry_crawler.py \\
    --min-sample-size 10000 \\
    --output data/registries/
```

**Output:**
- `data/registries/patient_registries.json` - Complete registry catalog
- `data/registries/patient_registries.csv` - Tabular format

### 2. Find Biobank Samples

```bash
# Catalog all biobanks
python scripts/registries/biobank_finder.py \\
    --output data/registries/

# Find brain tissue biobanks
python scripts/registries/biobank_finder.py \\
    --sample-type brain_tissue \\
    --output data/registries/

# Find DNA biobanks with autism samples
python scripts/registries/biobank_finder.py \\
    --sample-type dna \\
    --condition autism \\
    --output data/registries/

# Find iPSC repositories
python scripts/registries/biobank_finder.py \\
    --sample-type ipsc \\
    --output data/registries/

# Find large biobanks (> 10,000 samples)
python scripts/registries/biobank_finder.py \\
    --min-samples 10000 \\
    --output data/registries/
```

**Output:**
- `data/registries/biobank_inventory.json` - Complete biobank catalog
- `data/registries/biobank_inventory.csv` - Tabular format

## Complete Workflow

### Identifying Data Sources for Your Research

```bash
# Step 1: Identify relevant registries
python scripts/registries/registry_crawler.py \\
    --condition autism \\
    --data-type genomics \\
    --output data/registries/

# Review results
cat data/registries/patient_registries.json | jq '.registries[] | {name, sample_size, data_types, access_type}'

# Step 2: Find complementary biobank resources
python scripts/registries/biobank_finder.py \\
    --condition autism \\
    --sample-type dna \\
    --output data/registries/

# Review biobank options
cat data/registries/biobank_inventory.json | jq '.biobanks[] | {name, autism_samples, sample_types, access_type}'

# Step 3: Check comprehensive inventory
cat data/registries/available_samples.json | jq '.priority_targets'
```

## Output Files

### patient_registries.json

Comprehensive catalog of 17 ADHD/autism registries:

```json
{
  "metadata": {
    "total_registries": 17,
    "autism_registries": 11,
    "adhd_registries": 8,
    "comorbid_registries": 2,
    "active_enrollment": 10,
    "total_samples": 1131235
  },
  "registries": [
    {
      "registry_id": "SPARK",
      "name": "SPARK",
      "full_name": "Simons Foundation Powering Autism Research",
      "sample_size": 50000,
      "data_types": ["genomics", "phenotype", "medical_records"],
      "access_type": "controlled",
      "application_url": "https://base.sfari.org",
      "contact_email": "spark@simonsfoundation.org"
    }
  ]
}
```

### biobank_inventory.json

Inventory of 13 biobanks with biological samples:

```json
{
  "metadata": {
    "total_biobanks": 13,
    "brain_tissue_biobanks": 5,
    "dna_biobanks": 7,
    "ipsc_biobanks": 2,
    "total_autism_samples": 91250,
    "total_adhd_samples": 34550
  },
  "biobanks": [
    {
      "biobank_id": "AUTISM_BRAINNET",
      "name": "Autism BrainNet",
      "sample_types": ["brain_tissue", "frozen_tissue", "fixed_tissue"],
      "autism_samples": 1800,
      "brain_regions": ["prefrontal cortex", "cerebellum", "hippocampus"],
      "access_type": "controlled",
      "typical_turnaround": "8-12 weeks"
    }
  ]
}
```

### available_samples.json

Comprehensive inventory with access procedures and priority targets.

## Major Registries Cataloged

### Autism Registries

#### SPARK (N=50,000)
- **Data**: Whole exome sequencing, phenotype, medical records
- **Access**: Controlled via SFARI Base
- **URL**: https://sparkforautism.org
- **Contact**: spark@simonsfoundation.org
- **Notes**: Largest autism genetic study. Free genetic testing for families.

#### Simons Simplex Collection (N=2,600)
- **Data**: WES, genotypes, comprehensive phenotyping
- **Access**: Controlled via SFARI Base and dbGaP (phs000473)
- **URL**: https://www.sfari.org/resource/simons-simplex-collection/
- **Notes**: Gold standard simplex families. Foundational dataset.

#### IAN (N=52,000)
- **Data**: Phenotype, medical records, treatment history
- **Access**: Collaboration only
- **URL**: https://iancommunity.org
- **Contact**: ianresearch@kennedykrieger.org
- **Notes**: Largest phenotype registry. No biological samples.

#### AGRE (N=1,800 families)
- **Data**: Genotypes, phenotypes, DNA/cell lines available
- **Access**: Controlled via AGRE portal
- **URL**: https://research.agre.org
- **Notes**: Historic multiplex family collection. Enrollment closed.

#### EU-AIMS LEAP (N=800)
- **Data**: MRI, fMRI, EEG, eye tracking, genomics, biomarkers
- **Access**: Collaboration only
- **URL**: https://www.eu-aims.eu
- **Notes**: Deep phenotyping cohort. Multi-modal biomarkers.

#### ABC-CT (N=600)
- **Data**: MRI, EEG, eye tracking, clinical assessments
- **Access**: Controlled via NDAR
- **URL**: https://www.autismbiomarkersconsortium.org
- **Notes**: Clinical trial-ready biomarkers. NCT04119687.

### ADHD Registries

#### PGC ADHD (N=55,374)
- **Data**: GWAS summary statistics, genotypes
- **Access**: Open (summary stats), Controlled (individual data)
- **URL**: https://www.med.unc.edu/pgc/download-results/adhd/
- **Notes**: Largest ADHD GWAS. Summary statistics publicly available.

#### iPSYCH ADHD (N=20,183)
- **Data**: Whole genome sequencing, Danish health registers
- **Access**: Collaboration with Danish sites
- **URL**: https://ipsych.dk
- **Notes**: Exceptional registry linkage. Neonatal screening cohort.

#### ABCD Study (N=11,878)
- **Data**: Genomics, MRI, cognitive, longitudinal
- **Access**: Controlled via NDA
- **URL**: https://abcdstudy.org
- **Notes**: ~2,000 with ADHD symptoms. Richest multi-modal dataset.

### Large Multi-Disorder Cohorts

#### UK Biobank (N=500,000)
- **Data**: Genomics, imaging, EHR, metabolomics, microbiome
- **Access**: Controlled via application (3-4 week approval)
- **URL**: https://www.ukbiobank.ac.uk
- **Notes**: ~2,000 autism, ~4,000 ADHD. Massive multi-omics resource.

#### All of Us (N=413,000)
- **Data**: Whole genome sequencing, EHR, surveys, wearables
- **Access**: Controlled via Researcher Workbench
- **URL**: https://allofus.nih.gov
- **Notes**: Searchable by ICD codes. Registered tier access available.

## Brain Tissue Biobanks

### Autism BrainNet (N=1,800 autism, 1,200 control)
- **Sample Types**: Frozen tissue, fixed tissue, RNAlater preserved
- **Brain Regions**: Prefrontal cortex, cerebellum, hippocampus, amygdala, temporal cortex
- **Access**: Controlled, application required
- **Application**: https://www.autismbrainnet.org/resources/request-tissue/
- **Turnaround**: 8-12 weeks
- **Contact**: abn@psych.uic.edu
- **Notes**: Premier autism brain tissue resource. Excellent clinical documentation.

**Application requirements:**
- Research proposal (3-5 pages)
- IRB approval
- PI credentials
- Scientific Advisory Committee review

### NIH NeuroBioBank (N=450 autism, 280 ADHD)
- **Sample Types**: Frozen, fixed, RNAlater, OCT-embedded
- **Brain Regions**: All major regions available
- **Access**: Controlled, free for NIH-funded researchers
- **Application**: https://neurobiobank.nih.gov/researchers/
- **Turnaround**: 6-8 weeks
- **Contact**: nbbinfo@nih.gov
- **Notes**: Searchable online catalog. Broader psychiatric collection.

### UK Brain Banks Network (N=200 autism, 50 ADHD)
- **Sample Types**: Frozen, fixed
- **Access**: Collaboration with UK sites required
- **Turnaround**: 12-16 weeks
- **Contact**: brainbanks@mrc.ac.uk
- **Notes**: Multiple sites (Edinburgh, London, Manchester, Oxford).

## DNA and Cell Line Biobanks

### SPARK Biobank (N=50,000 autism + families)
- **Sample Types**: DNA from saliva
- **Data**: Whole exome sequencing for all
- **Access**: Controlled via SFARI Base
- **Turnaround**: 8-12 weeks
- **Notes**: Rapidly growing. Most comprehensive autism WES resource.

### NIMH Repository (N=12,000 autism, 8,000 ADHD)
- **Sample Types**: DNA, cell lines, plasma, serum
- **Access**: Controlled, free for NIH-funded research
- **Application**: https://www.nimhgenetics.org
- **Turnaround**: 4-6 weeks
- **Contact**: nimhgenetics@rutgers.edu
- **Notes**: Fast turnaround. Searchable catalog. Major psychiatric biomaterial resource.

### iPSYCH Biobank (N=18,000 autism, 20,000 ADHD)
- **Sample Types**: DNA, plasma, dried blood spots
- **Data**: Whole genome sequencing, Danish health registers
- **Access**: Collaboration with Danish sites required
- **Turnaround**: 16-24 weeks
- **Notes**: Exceptional longitudinal registry data. Neonatal screening samples.

### AGRE Biobank (N=3,500 autism + families)
- **Sample Types**: DNA, lymphoblastoid cell lines
- **Access**: Controlled via AGRE portal
- **Turnaround**: 6-8 weeks
- **Contact**: agre@autismspeaks.org
- **Notes**: Multiplex families. Linked to genotype and phenotype data in NDAR.

### Coriell Institute (N=2,500 autism, 200 ADHD)
- **Sample Types**: Lymphoblastoid cell lines, DNA, iPSC (some)
- **Access**: Open (commercial purchase)
- **Application**: Online ordering
- **Turnaround**: 1-2 weeks
- **URL**: https://www.coriell.org
- **Notes**: Fast commercial access. Includes samples from AGRE, SSC. Fee-based.

## iPSC Resources

### Autism iPSC Repository (N=300 autism lines)
- **Sample Types**: Induced pluripotent stem cells
- **Uses**: Neuronal differentiation, organoid models
- **Access**: Controlled, contact originating labs
- **Turnaround**: 8-12 weeks
- **Notes**: From labs of Geschwind (UCLA), Pașca (Stanford), others. Check StemBANCC, CIRM databases.

**Quality control:**
- Pluripotency marker confirmation
- Karyotyping (normal karyotype)
- Mycoplasma testing (negative)
- Differentiation protocols available

## Access Procedures

### Controlled Access (11 registries, 10 biobanks)

**Typical requirements:**
1. Completed research application (2-10 pages)
2. IRB/ethics approval from your institution
3. Data Use Agreement (DUA) or Material Transfer Agreement (MTA)
4. Scientific review by advisory committee
5. PI credentials verification

**Typical timeline:**
- Application preparation: 1-2 weeks
- Committee review: 4-8 weeks
- Agreement execution: 1-2 weeks
- Sample/data delivery: 2-4 weeks
- **Total**: 8-16 weeks

**Examples:**
- SPARK: https://base.sfari.org
- NIH NeuroBioBank: https://neurobiobank.nih.gov
- UK Biobank: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access
- Autism BrainNet: https://www.autismbrainnet.org/resources/request-tissue/

### Open Access (1 registry: PGC ADHD)

**Typical requirements:**
1. User registration
2. Acceptance of terms of use
3. Citation agreement

**Typical timeline:**
- Registration: < 1 hour
- Data download: Immediate
- **Total**: Same day

**Examples:**
- PGC ADHD summary statistics: https://www.med.unc.edu/pgc/download-results/adhd/
- Coriell commercial cell lines: https://www.coriell.org (fee-based)

### Collaboration Only (5 registries, 2 biobanks)

**Typical requirements:**
1. Collaborative research proposal
2. Co-authorship or acknowledgment agreement
3. Local site approval
4. Data sharing agreement
5. Often requires co-PI from data-holding institution

**Typical timeline:**
- Collaboration negotiation: 4-12 weeks
- Proposal development: 2-4 weeks
- Approvals: 4-8 weeks
- Data/sample access: 2-4 weeks
- **Total**: 12-28 weeks

**Examples:**
- IAN: Contact ianresearch@kennedykrieger.org
- EU-AIMS: Contact info@eu-aims.eu
- iPSYCH: Requires Danish site collaboration
- UK Brain Banks: Requires UK site collaboration

## Integration with Other Data Sources

### Link to Clinical Trials

```python
import pandas as pd
import json

# Load registries and trials
with open('data/registries/patient_registries.json') as f:
    registries = json.load(f)

trials_df = pd.read_csv('data/trials/trials_with_biomarkers.csv')

# Find registries linked to clinical trials
# Example: ABCD Study (NCT03854695), SPARK (NCT03758391)
for reg in registries['registries']:
    if reg['name'] in ['ABCD ADHD', 'SPARK', 'ABC-CT']:
        print(f"{reg['name']}: {reg['sample_size']} participants, {reg['access_type']} access")
```

### Cross-Reference with Literature

```python
# Load literature contacts
papers_df = pd.read_csv('data/literature/author_contacts.csv')
registries_df = pd.read_csv('data/registries/patient_registries.csv')

# Find PIs with both publications and registry access
# Example: Demontis (PGC ADHD), Chung (SPARK), Geschwind (multiple)
```

### Link to Genetic Data

```python
# Load genetic studies
with open('data/genetics/gwas_catalog.json') as f:
    gwas = json.load(f)

# Match to registries with genomic data
# Example: PGC ADHD GWAS from iPSYCH, SSC autism exomes in dbGaP
```

## Priority Access Targets

### For Replication Studies

1. **UK Biobank** (N=500,000)
   - Application: 3-4 weeks
   - Cost: £0 for approved research
   - Data: WES/WGS, imaging, EHR, metabolomics

2. **PGC ADHD** (N=55,374)
   - Access: Immediate (summary stats)
   - Cost: Free
   - Data: GWAS summary statistics

3. **All of Us** (N=413,000)
   - Application: Registered tier available
   - Cost: Free
   - Data: WGS, EHR, surveys

### For Discovery Studies

1. **SPARK** (N=50,000)
   - Application: 8-12 weeks
   - Data: WES + phenotype
   - Notes: Largest autism WES resource

2. **iPSYCH** (N=38,183)
   - Collaboration: Danish sites required
   - Data: WGS + health registers
   - Notes: Best register linkage

3. **ABCD Study** (N=11,878)
   - Application: Via NDA
   - Data: Genomics + imaging + longitudinal
   - Notes: Richest multi-modal data

### For Brain Tissue Studies

1. **Autism BrainNet** (N=1,800 autism)
   - Application: 8-12 weeks
   - Notes: Premier autism tissue resource

2. **NIH NeuroBioBank** (N=450 autism, 280 ADHD)
   - Application: 6-8 weeks
   - Notes: Free for NIH-funded research

3. **Harvard Brain Bank** (N=120 autism)
   - Application: 8-10 weeks
   - Notes: Excellent clinical documentation

### For Cell Model Studies

1. **Autism iPSC lines** (N=300)
   - Contact: Originating labs
   - Notes: For neuronal differentiation

2. **Coriell cell lines** (N=2,500 autism)
   - Access: Immediate purchase
   - Notes: Fast commercial option

## Application Tips

### Writing Strong Proposals

1. **Be specific**: Clearly state sample requirements (tissue type, quantity, quality)
2. **Show feasibility**: Demonstrate you can complete the project
3. **Explain impact**: Emphasize scientific and clinical significance
4. **Respect guidelines**: Follow repository-specific requirements exactly
5. **Budget time**: Applications take 2-12 weeks minimum

### Common Application Components

**Research Plan (2-10 pages):**
- Background and significance
- Specific aims
- Research approach and methods
- Sample requirements (type, quantity, quality)
- Expected outcomes and impact
- Data sharing plan
- Timeline

**Supporting Documents:**
- PI CV/biosketch
- IRB approval (or pending letter)
- Institutional support letter
- Budget (if fees apply)
- Data/material use plan

### Negotiating Access

**For collaboration-only resources:**
- Identify potential collaborators early
- Offer clear value proposition (expertise, resources, co-authorship)
- Be flexible on authorship order and data use
- Consider visiting the site to build relationships
- Allow extra time for negotiation (3-6 months)

## Sample Request Template

```text
Subject: Biospecimen Request for [Your Project Title]

Dear [Biobank Contact],

I am writing to request access to [specific samples] from [Biobank Name]
for our research project investigating [brief description].

Project Details:
- Title: [Project Title]
- PI: [Your Name], [Your Institution]
- Funding: [Grant number or pending]
- IRB Approval: [Number or pending]

Sample Requirements:
- Sample type: [e.g., frozen brain tissue]
- Condition: [e.g., autism diagnosed, N=20]
- Brain regions: [e.g., prefrontal cortex, 100mg per sample]
- Control samples: [e.g., age/sex matched, N=20]
- Quality requirements: [e.g., PMI < 24h, RIN > 6]

Research Aims:
[Brief 2-3 sentence description of what you will do with the samples
and why it is important]

Timeline:
- Anticipated start date: [Date]
- Sample analysis: [Duration]
- Expected completion: [Date]

I have reviewed your access policies and am prepared to:
- Submit formal application
- Provide IRB approval
- Execute Material Transfer Agreement
- Acknowledge biobank in publications
- Share results with biobank

Please advise on the application process and timeline. I am happy to
provide additional information or discuss this request further.

Thank you for considering this request.

Best regards,
[Your Name]
[Title]
[Institution]
[Email]
[Phone]
```

## Ethical Considerations

### Data and Sample Use

- **Respect consent**: Only use data/samples consistent with participant consent
- **Protect privacy**: Implement appropriate security measures
- **Follow agreements**: Comply with all DUA/MTA terms
- **Acknowledge sources**: Cite registries and biobanks appropriately
- **Share findings**: Report results to participants and data holders when requested

### Publication Requirements

Most registries require:
- Acknowledgment in publications
- Citation of registry in methods
- Notification of publications
- Sharing of manuscripts
- Co-authorship for some (collaboration-only access)

**Standard acknowledgment:**
"Data/samples were provided by [Registry Name] ([URL]). We thank the families who participated in [Registry Name] for their contributions."

## Troubleshooting

### Application Rejected

**Common reasons:**
- Insufficient justification
- Feasibility concerns
- Inadequate sample size calculation
- Missing required documents
- Proposal outside repository scope

**Solutions:**
- Request reviewer feedback
- Address concerns and resubmit
- Consider pilot study first
- Seek collaboration with experienced users

### Long Turnaround Times

**Strategies:**
- Apply early in project planning (6-12 months ahead)
- Have IRB approval ready
- Respond quickly to queries
- Consider multiple repositories simultaneously
- Build in buffer time to grant timelines

### High Costs

**Options for reducing fees:**
- Check for NIH-funded researcher waivers
- Apply for repository-specific grants
- Start with smaller pilot study
- Consider collaboration instead of direct access
- Use publicly available data first (e.g., PGC summary stats)

## Future Enhancements

Potential additions:
- **Real-time availability tracking** via API integration
- **Automated application generators** with templates
- **Sample quality filters** (PMI, RIN, pH, etc.)
- **Cost estimators** for biospecimen requests
- **Publication tracking** from each registry
- **Network analysis** of data sharing collaborations

## References

### Registry Resources
- SPARK: https://sparkforautism.org
- SFARI Base: https://base.sfari.org
- IAN: https://iancommunity.org
- AGRE: https://research.agre.org
- iPSYCH: https://ipsych.dk
- PGC: https://www.med.unc.edu/pgc/
- ABCD Study: https://abcdstudy.org
- UK Biobank: https://www.ukbiobank.ac.uk

### Biobank Resources
- Autism BrainNet: https://www.autismbrainnet.org
- NIH NeuroBioBank: https://neurobiobank.nih.gov
- NIMH Repository: https://www.nimhgenetics.org
- Coriell: https://www.coriell.org
- UK Brain Banks: https://www.mrc.ac.uk/research/facilities-and-resources-for-researchers/brain-banks/

### Data Sharing Platforms
- NDAR (National Database for Autism Research): https://ndar.nih.gov
- SFARI Base: https://base.sfari.org
- NDA (NIMH Data Archive): https://nda.nih.gov
- dbGaP: https://www.ncbi.nlm.nih.gov/gap/

## Support

For questions or issues:
1. Check registry/biobank website for application guidelines
2. Contact repository directly with specific questions
3. Review successful grant applications for models
4. Consult with colleagues who have obtained access
5. Open GitHub issue for tool-related problems

---

**Last updated**: 2025-09-30
**Version**: 1.0
**Maintained by**: AuDHD Correlation Study Team