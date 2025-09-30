# ASD/ADHD Biobank Discovery Report

Generated: 2025-09-30

## Summary

**Total biobanks identified:** 7

- **With genetic samples:** 3
- **With fluid samples:** 4
- **With tissue samples:** 2

## Biobank Catalog

### NIMH Repository and Genomics Resource (RGR)

**URL:** https://www.nimhgenetics.org/

**Focus:** Genetic studies of mental health disorders

**Available Sample Types:**
- **Dna:** ~10000 samples
- **Lymphoblastoid Cell Lines:** ~5000 samples
- **Plasma:** ~Variable samples
- **Serum:** ~Variable samples

**Access Process:**
- Application required: True
- IRB approval: True
- Cost: Variable, typically $50-500 per sample
- Turnaround: 4-8 weeks
- Apply: https://www.nimhgenetics.org/available_data/request_biomaterials/

**Feasibility for Key Assays:**
- Cortisol Rhythm: LOW - Fresh samples needed
- Heavy Metals: NONE - No hair/nail samples
- Proteomics: MEDIUM - Limited plasma/serum
- Metabolomics: LOW - Sample age/storage concerns
- Genomics: HIGH - Primary resource
- Transcriptomics: MEDIUM - Cell lines available

---

### Autism BrainNet (Brain Tissue Repository)

**URL:** https://www.autismbrainnet.org/

**Focus:** Postmortem brain tissue for autism research

**Available Sample Types:**
- **Brain Tissue:** ~200 samples
- **Csf:** ~100 samples, Variable µL
- **Blood:** ~150 samples
- **Urine:** ~50 samples

**Access Process:**
- Application required: True
- IRB approval: True
- Cost: Tissue processing and shipping fees (~$500-2000)
- Turnaround: 8-16 weeks
- Apply: https://www.autismbrainnet.org/for-researchers/

**Feasibility for Key Assays:**
- Cortisol Rhythm: NONE - Postmortem samples
- Heavy Metals: LOW - Postmortem, but hair may be available
- Proteomics: MEDIUM - CSF is valuable
- Metabolomics: LOW - Postmortem changes
- Brain Proteomics: HIGH - Primary resource
- Neuropathology: HIGH - Unique resource

---

### NICHD Data and Specimen Hub (DASH)

**URL:** https://dash.nichd.nih.gov/

**Focus:** Repository for NICHD-funded studies

**Available Sample Types:**
- **Biospecimens:**  (Varies by contributing study)
- **Dna:** 
- **Serum Plasma:** 

**Access Process:**
- Application required: True
- IRB approval: True
- Cost: Variable by study
- Turnaround: 4-12 weeks
- Apply: https://dash.nichd.nih.gov/study

**Feasibility for Key Assays:**
- Note: Must check individual studies - EARLI and SEED most promising
- Environmental Biomarkers: HIGH - Environmental health focus
- Maternal Samples: HIGH - Prenatal/perinatal focus

---

### NIH NeuroBioBank

**URL:** https://neurobiobank.nih.gov/

**Focus:** Brain tissue and associated biospecimens

**Available Sample Types:**
- **Brain Tissue:** ~100 samples
- **Csf:** ~50 samples, Variable µL
- **Blood Derivatives:** ~80 samples

**Access Process:**
- Application required: True
- IRB approval: True
- Cost: Tissue processing fees (~$500-1500)
- Turnaround: 6-12 weeks
- Apply: https://neurobiobank.nih.gov/researchers/

**Feasibility for Key Assays:**
- Cortisol Rhythm: NONE - Postmortem
- Brain Markers: HIGH - Primary resource
- Proteomics Csf: HIGH - CSF available
- Genomics: HIGH - DNA available

---

### SPARK (Simons Foundation Powering Autism Research)

**URL:** https://sparkforautism.org/

**Focus:** Large-scale autism genetics and phenotyping

**Available Sample Types:**
- **Saliva Dna:** ~100000 samples
- **Dna:** ~100000 samples
- **Potential Recall:**  (Can contact participants for additional samples)

**Access Process:**
- Application required: True
- IRB approval: True
- Cost: Free for approved projects
- Turnaround: 8-12 weeks
- Apply: https://base.sfari.org/spark

**Feasibility for Key Assays:**
- Genomics: HIGH - Primary resource
- Saliva Biomarkers: MEDIUM - Saliva samples stored
- Recall For New Samples: HIGH - Active cohort
- Cortisol Saliva: POTENTIAL - Could propose recall study

---

### ABCD Study Biospecimen Repository

**URL:** https://abcdstudy.org/

**Focus:** Adolescent Brain Cognitive Development (includes ADHD)

**Available Sample Types:**
- **Saliva:** ~11000 samples
- **Hair:** ~5000 samples (Subset of participants)
- **Baby Teeth:** ~2000 samples (For environmental exposure)

**Access Process:**
- Application required: True
- IRB approval: True
- Cost: Free data access; biospecimen costs TBD
- Turnaround: 8-16 weeks
- Apply: https://nda.nih.gov/abcd

**Feasibility for Key Assays:**
- Genomics: HIGH - Saliva DNA
- Heavy Metals Hair: HIGH - Hair samples available
- Developmental Exposures: HIGH - Baby teeth
- Salivary Biomarkers: MEDIUM - Depends on storage
- Cortisol Retrospective: HIGH - Hair cortisol possible

---

### All of Us Research Program Biobank

**URL:** https://allofus.nih.gov/

**Focus:** Precision medicine cohort (1M+ participants)

**Available Sample Types:**
- **Blood:** ~5000 samples (Self-reported ASD/ADHD)
- **Urine:** ~5000 samples
- **Saliva:** ~2000 samples

**Access Process:**
- Application required: True
- IRB approval: Unknown
- Cost: Free for approved researchers
- Turnaround: 4-8 weeks
- Apply: https://www.researchallofus.org/

**Feasibility for Key Assays:**
- Proteomics: HIGH - Fresh plasma/serum
- Metabolomics: HIGH - Urine and blood
- Genomics: HIGH - WGS available
- Inflammatory Markers: HIGH - Plasma available
- Note: Smaller ASD/ADHD N than specialized cohorts

---

## Assay Requirements Reference

### Cortisol Rhythm Saliva

**Required sample type:** saliva

**Cost per sample:** $25

**Vendor:** Salimetrics


### Cortisol Hair

**Required sample type:** hair

**Cost per sample:** $50

**Vendor:** Multiple labs


### Heavy Metals Icp-Ms

**Compatible sample types:** hair, nail, blood, urine

**Cost per sample:** $150

**Vendor:** Mayo Clinic Labs, LabCorp

**Hair requirements:**
  - length_cm: 3
  - weight_mg: 50

**Nail requirements:**
  - weight_mg: 100

**Blood requirements:**
  - volume_ul: 500

**Urine requirements:**
  - volume_ml: 10


### Proteomics Somascan

**Compatible sample types:** serum, plasma, csf

**Cost per sample:** $750

**Vendor:** SomaLogic

**Serum requirements:**
  - volume_ul: 150

**Plasma requirements:**
  - volume_ul: 150

**Csf requirements:**
  - volume_ul: 100


### Metabolomics Broad

**Compatible sample types:** serum, plasma, urine

**Cost per sample:** $400

**Vendor:** Broad Institute, Metabolon

**Serum requirements:**
  - volume_ul: 100

**Plasma requirements:**
  - volume_ul: 100

**Urine requirements:**
  - volume_ml: 5


### Inflammatory Markers

**Compatible sample types:** serum, plasma

**Cost per sample:** $150

**Vendor:** R&D Systems, Meso Scale Discovery


### Microbiome 16S

**Compatible sample types:** stool, saliva

**Cost per sample:** $100

**Vendor:** Microbiome centers

**Stool requirements:**
  - weight_mg: 200

**Saliva requirements:**
  - volume_ml: 2


### Exosomes

**Compatible sample types:** plasma, urine, csf

**Cost per sample:** $300

**Vendor:** Exosome isolation + proteomics

**Plasma requirements:**
  - volume_ul: 500

**Urine requirements:**
  - volume_ml: 10

**Csf requirements:**
  - volume_ul: 250


