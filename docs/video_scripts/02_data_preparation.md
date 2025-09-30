# Video Script: Data Preparation (10 minutes)

## Introduction (1 minute)

"In this tutorial, we'll prepare your multi-omics data for analysis. We'll cover data format requirements, quality control, and harmonization."

## Genomic Data (2 minutes)

"Let's start with genomic data in VCF format."

**Show example VCF file**

"Your VCF must include:
- Chromosome, position, SNP ID
- Reference and alternate alleles
- GT (genotype) field
- One column per sample"

"Apply quality filters:"

```python
from audhd_correlation.data import load_vcf

genotypes = load_vcf(
    "genotypes.vcf",
    min_call_rate=0.95,  # 95% of samples must have data
    min_maf=0.01,         # Minor allele frequency > 1%
)
```

**Show before/after statistics**

## Clinical Data (2 minutes)

"Clinical data should be in CSV format with required columns:"

**Show spreadsheet**

- sample_id (matching genomic data)
- age, sex, diagnosis
- Optional: severity_score, iq, bmi, site

"Validate your data:"

```python
from audhd_correlation.data import load_clinical_csv

clinical = load_clinical_csv(
    "phenotypes.csv",
    validate_ranges=True
)
```

**Show validation report**

## Metabolomic & Microbiome Data (2 minutes)

"Metabolomic data: CSV with samples in rows, metabolites in columns"

```python
metabolomic = load_metabolomics(
    "metabolites.csv",
    log_transform=True,
    min_detection_rate=0.5
)
```

"Microbiome data: TSV or BIOM format"

```python
microbiome = load_microbiome(
    "abundances.tsv",
    min_prevalence=0.1,
    relative_abundance=True
)
```

## Harmonization (2 minutes)

"Align samples across modalities:"

```python
from audhd_correlation.data import align_multiomics

data = {
    'genomic': genotypes,
    'clinical': clinical,
    'metabolomic': metabolomic
}

aligned = align_multiomics(data)

print(f"Common samples: {len(aligned['genomic'])}")
```

**Show Venn diagram of sample overlap**

## Batch Correction (1 minute)

"If data from multiple sites:"

```python
from audhd_correlation.data import correct_batch_effects

corrected = correct_batch_effects(
    aligned,
    batch_column='site',
    method='combat'
)
```

**Show before/after PCA plots**

## Summary

"Your data is now ready for analysis!

Key points:
✓ Quality control at loading
✓ Align sample IDs across modalities
✓ Correct for batch effects

Next: Preprocessing and integration"