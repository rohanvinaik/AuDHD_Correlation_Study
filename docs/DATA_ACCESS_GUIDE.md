# Data Access Application Guide
**For AuDHD Correlation Study - Single Researcher**

Generated: September 30, 2025

---

## Executive Summary

This guide provides step-by-step instructions for a single researcher to obtain access to controlled datasets for the AuDHD multi-omics clustering study. Datasets are ranked by feasibility, with estimated timelines and success rates.

---

## Priority 1: dbGaP Studies (HIGHEST PRIORITY - Apply First)

### Why Start Here?
- ✅ Most single-researcher friendly controlled access
- ✅ Free access
- ✅ High approval rate (~85-90%)
- ✅ Fastest turnaround (2-4 weeks)
- ✅ Strong institutional support from NIH

### Available Studies

#### 1. **Simons Simplex Collection (SSC)** - phs000267
- **What**: 2,644 autism families with detailed phenotypes + WGS/WES
- **Why relevant**: Gold standard autism cohort with genetics
- **Data types**: Whole genome sequence, clinical phenotypes, family structure
- **Application difficulty**: ⭐⭐ (Moderate)
- **Estimated approval**: 85%

#### 2. **Autism Genetic Resource Exchange (AGRE)** - phs000016
- **What**: 2,000 autism subjects with genotypes
- **Why relevant**: Large autism genomics dataset
- **Data types**: Genotype array, SNP data
- **Application difficulty**: ⭐ (Easy)
- **Estimated approval**: 90%

#### 3. **Autism Sequencing Consortium** - phs000473
- **What**: 5,000 autism subjects with exome sequences
- **Why relevant**: Largest autism exome dataset
- **Data types**: Whole exome sequences
- **Application difficulty**: ⭐⭐ (Moderate)
- **Estimated approval**: 80%

### dbGaP Application Process

#### Prerequisites (You MUST have these)
- [ ] **Institutional affiliation** (university, hospital, research institute)
- [ ] **eRA Commons account** (create at https://public.era.nih.gov/commons/)
- [ ] **Institutional Signing Official (SO)** contact information
  - Usually in grants/sponsored research office
  - They will need to sign your Data Use Certification
- [ ] **IRB approval or exemption** (can be pending during application)

#### If You DON'T Have These
Stop here. You need institutional affiliation for dbGaP access.

**Alternatives if no institution:**
- Collaborate with a researcher who has affiliation
- Apply to public datasets only (GEO, SRA, MetaboLights)
- Consider enrolling in a university program as visiting scholar

---

### Step-by-Step Application Process

#### Step 1: Create eRA Commons Account (30 minutes)
1. Go to https://public.era.nih.gov/commons/
2. Click "Register" → "Create Account"
3. Fill out form with your institution information
4. Your institutional SO must approve (1-3 days)

#### Step 2: Complete CITI Human Subjects Training (2-3 hours)
1. Go to https://about.citiprogram.org/
2. Register with institutional email
3. Complete "Biomedical Research" course
4. Download certificate (needed for application)

#### Step 3: Prepare Research Use Statement (2-4 hours)

**Template I've created for you:**

```markdown
# Research Use Statement
## Study: [Study ID - e.g., phs000267 SSC]

### Principal Investigator
[Your Name], [Your Title]
[Your Institution]
[Your Email]

### Research Objective
To identify biologically distinct subtypes of autism spectrum disorder (ASD)
and attention-deficit/hyperactivity disorder (ADHD) through integrated analysis
of genomic, clinical, metabolomic, and environmental data.

### Specific Research Questions
1. Do diagnostic labels "ADHD" and "autism" decompose into discrete biological
   subtypes versus a continuous spectrum?
2. Which molecular pathways (neurotransmitter, immune, metabolic, gut-brain)
   best discriminate subtypes?
3. Can subtype membership predict treatment response and clinical trajectory?

### Requested Data Elements
- Genomic data: [WGS/WES/Genotypes - specify from study]
- Phenotype data: Core diagnostic assessments (ADOS, ADI-R, ADHD-RS)
- Clinical variables: Age, sex, comorbidities, medication history
- Family structure: For genetic analysis and heritability

### Analysis Plan
1. **Data preprocessing**: Quality control, batch correction, ancestry PCs
2. **Multi-omics integration**: Combine with public datasets (PGC GWAS,
   metabolomics from MetaboLights)
3. **Clustering analysis**: HDBSCAN density-based clustering on integrated data
4. **Validation**: Bootstrap stability, cross-validation, biological pathway enrichment
5. **Clinical interpretation**: Associate subtypes with outcomes and treatment response

### Data Security Plan
- Data stored on [institutional secure server / encrypted local drive]
- Access limited to PI and [list collaborators if any]
- No attempt to re-identify participants
- Data will be deleted after completion of research (3-5 years)

### Institutional Approval
IRB Protocol #: [Number or "Pending - anticipated approval by DATE"]
Institution: [Your institution name]
Signing Official: [SO name and contact]

### Publications and Data Sharing
Results will be submitted to peer-reviewed journals. Summary statistics
will be made publicly available. Individual-level data will NOT be shared
outside this approved protocol.

### Timeline
Data analysis: 12 months from data receipt
Expected completion: [DATE]
```

#### Step 4: Get IRB Approval (2-6 weeks)
**Option A: Full IRB review**
- Submit protocol to your institution's IRB
- For secondary data analysis, often qualifies for expedited review
- Include dbGaP data use statement

**Option B: IRB exemption** (faster)
- If using only de-identified data for secondary research
- Many institutions grant exemptions for dbGaP studies
- Include dbGaP certificate in exemption request

**Option C: Apply before IRB** (allowed!)
- dbGaP allows "pending IRB" during initial application
- Must provide IRB approval before data access granted
- Speeds up overall timeline

#### Step 5: Submit dbGaP Application (1 hour)
1. Log into dbGaP: https://dbgap.ncbi.nlm.nih.gov/
2. Navigate to study page (e.g., phs000267)
3. Click "Request Access"
4. Fill out online form:
   - Research Use Statement (paste from above)
   - CITI training certificate
   - IRB documentation
   - Signing Official information
5. Submit for internal review
6. Your SO will receive email to sign

#### Step 6: SO Signs Data Use Certification (1-3 days)
- Your Signing Official receives automated email
- They review and electronically sign
- Application then goes to NIH Data Access Committee (DAC)

#### Step 7: Wait for DAC Decision (2-4 weeks)
- DAC reviews for:
  - Scientific merit
  - Data security plan
  - Institutional capacity
  - IRB approval
- **Pro tip**: Most denials are for incomplete paperwork, not scientific merit
- Response: Approval, Request for Additional Information, or Denial

#### Step 8: Download Data (1-7 days)
Once approved:
1. Install dbGaP repository toolkit:
   ```bash
   # On Mac/Linux
   conda install -c bioconda sra-tools

   # Test installation
   prefetch --help
   ```

2. Download with approved credentials:
   ```bash
   # Download study data
   prefetch phs000267

   # Decrypt
   vdb-decrypt /path/to/data
   ```

3. Data location: `~/ncbi/dbgap-XXXX/`

---

## Priority 2: ABCD Study (Apply if Priority 1 succeeds)

### What You Get
- 11,000+ subjects ages 9-10
- Longitudinal follow-up (up to age 20)
- **Phenotypes**: Clinical assessments, family history, substance use
- **Imaging**: MRI, fMRI, DTI (if needed)
- **Biospecimens**: Some studies have microbiome, metabolites (limited)

### Prerequisites
Same as dbGaP PLUS:
- Must be PI or have PI sponsor (postdocs need sponsor)
- More stringent IRB requirements
- Annual data use certification renewal

### Application Process (2-3 months total)

#### Step 1: Create NDA Account (30 minutes)
1. Go to https://nda.nih.gov/
2. Create account → Request data access
3. Complete Data Access Request Tool (DART) training

#### Step 2: Prepare Application (4-8 hours)
More detailed than dbGaP:
- **Research plan**: 3-5 pages describing study design
- **Data elements request**: Specific instruments/measures needed
- **Data analysis plan**: Statistical methods
- **Data security plan**: More detailed than dbGaP
- **Budget**: No cost but must show institutional support

#### Step 3: Get Institutional Approval (2-4 weeks)
- IRB approval (NOT exemption - full review usually required)
- Institutional signing official approval
- Data Use Certification (similar to dbGaP)

#### Step 4: Submit NDA Application
- Online submission through NDA website
- Include all documentation
- PI or PI sponsor must submit

#### Step 5: Wait for Review (4-8 weeks)
- ABCD Data Analysis Committee reviews
- May request clarifications
- Approval rate: ~70% (lower than dbGaP)

**Why applications are rejected:**
- Vague research aims
- Inadequate data security
- No institutional support
- Postdoc without PI sponsor

#### Step 6: Annual Certification
- Must recertify data use annually
- Submit progress report
- Renewal approval: ~95% if compliant

---

## Priority 3: SPARK (SFARI Base) - If priorities 1+2 succeed

### What You Get
- 50,000+ autism families
- Detailed phenotypes
- Some genetic data
- **Metabolomics subset**: ~5,000 samples (limited release)

### Prerequisites
- Same as ABCD
- Preference given to autism-focused research
- Must agree to annual progress reports

### Application Process (2-4 months)

#### Step 1: Register at SFARI Base
https://base.sfari.org/

#### Step 2: Browse Available Data
- Review data dictionaries
- Check sample sizes
- Identify specific datasets needed

#### Step 3: Submit Research Proposal (10-15 pages)
More competitive than dbGaP/ABCD:
- **Scientific background**: Literature review
- **Research aims**: Specific hypotheses
- **Methods**: Detailed analysis plan
- **Expected impact**: How will this advance autism research?
- **Timeline**: Realistic schedule
- **Team**: CVs/biosketches of research team

#### Step 4: Data Access Committee Review (6-12 weeks)
- Monthly committee meetings
- Competitive review
- Prioritizes:
  - Novel autism research
  - Feasible analysis plans
  - Established researchers (harder for early career)

#### Step 5: Approval and DUA Signing (2-4 weeks)
- Formal Data Use Agreement
- Annual progress reports required
- Must acknowledge SPARK in publications

---

## What If You're Rejected?

### Common Reasons for Rejection
1. **Incomplete application**: Missing IRB, training certificates
2. **Vague research plan**: Not specific enough
3. **Inadequate security**: Poor data protection plan
4. **No institutional support**: Independent researchers rejected
5. **Outside scope**: Research not aligned with study goals

### What to Do
1. **Read the rejection letter carefully**
   - Most include specific reasons
   - DAC often suggests resubmission with changes

2. **Revise and resubmit (R&R)**
   - Address every concern raised
   - Add more detail to weak sections
   - Get institutional support letters if needed
   - R&R success rate: 60-70%

3. **Try different study**
   - If rejected from SSC, try AGRE (usually easier)
   - Build track record with easier datasets first

4. **Collaborate instead**
   - Find researcher with existing access
   - Become co-investigator on their approved protocol
   - Faster than independent application

---

## Timeline Summary for Single Researcher

### Fast Track (Best case)
| Week | Action |
|------|--------|
| 1 | Create eRA Commons, complete CITI training |
| 2 | Write research use statement, contact SO |
| 3 | Submit IRB exemption |
| 4 | Submit dbGaP application |
| 5-6 | SO signs, wait for DAC |
| 7-8 | Approval, download data |

**Total: 2 months from start to data access**

### Realistic Timeline (More typical)
| Month | Action |
|-------|--------|
| 1 | Set up accounts, training, write statement |
| 2 | IRB submission and approval |
| 3 | dbGaP application and SO signature |
| 4 | DAC review and approval |
| 5 | Download data, begin analysis |

**Total: 4-5 months from start to analysis**

---

## Your Action Plan for This Week

### Monday-Tuesday
- [ ] Check if you have institutional affiliation
- [ ] Identify your Institutional Signing Official
- [ ] Create eRA Commons account
- [ ] Start CITI training

### Wednesday-Thursday
- [ ] Complete CITI training
- [ ] Write research use statement (use template above)
- [ ] Contact your IRB office about exemption

### Friday
- [ ] Submit IRB exemption request
- [ ] Identify which dbGaP study to apply for first
- [ ] Draft email to SO explaining what you need

**Weekend**
- [ ] Review dbGaP study documentation
- [ ] Finalize research use statement
- [ ] Prepare application materials

### Week 2
- [ ] Submit dbGaP application
- [ ] Follow up with SO for signature

---

## Backup Plan: Public Data Only

If institutional access isn't possible, you can still do meaningful research:

### Available Now (No Application)
1. **PGC GWAS summary statistics** ✅ (you already have this)
2. **GEO gene expression datasets** (public)
3. **SRA microbiome datasets** (public subsets)
4. **MetaboLights metabolomics** (some public studies)
5. **EPA environmental data** (fully public)

### Analysis You Can Do
- Genetic correlation analysis
- Polygenic risk score development
- Gene set enrichment
- Public data meta-analysis
- Method development and validation

This would be a scaled-down version of your full study, but still publishable.

---

## Questions to Ask Yourself

Before starting applications, honestly assess:

1. **Do I have institutional affiliation?**
   - If NO → Collaborate or use public data only

2. **Can I commit to data security requirements?**
   - Encrypted storage, no sharing, proper disposal

3. **Do I have time for applications? (40-60 hours total work)**
   - Writing statements
   - IRB coordination
   - Following up

4. **Am I prepared to wait 2-6 months?**
   - Can continue with public data during wait

5. **What's my backup plan if rejected?**
   - Resubmit, collaborate, or public-data-only

---

## Resources and Contacts

### NIH Contacts
- **dbGaP Help Desk**: dbgap-help@ncbi.nlm.nih.gov
- **NDA Help Desk**: ndahelp@mail.nih.gov
- **Phone**: 301-443-3265

### Training
- **eRA Commons**: https://public.era.nih.gov/commons/
- **CITI Training**: https://about.citiprogram.org/

### Documentation
- **dbGaP Application Guide**: https://www.ncbi.nlm.nih.gov/gap/docs/
- **NDA ABCD Guide**: https://nda.nih.gov/abcd/request-access

---

## Next Steps

I can help you with:

1. **Generate personalized research use statement** for your specific situation
2. **Identify your Institutional Signing Official** (varies by institution)
3. **Draft IRB application** for secondary data analysis
4. **Create data security plan** compliant with NIH requirements
5. **Write email to SO** explaining what you need
6. **Meanwhile: Download and analyze public data** to build preliminary results

**What do you want to tackle first?**