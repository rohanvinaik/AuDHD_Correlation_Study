Data Loading Guide
==================

This guide covers loading and harmonizing multi-omics data.

Supported Data Modalities
--------------------------

The pipeline supports four data modalities:

1. **Genomic Data** - SNP genotypes from VCF files
2. **Clinical Data** - Phenotypic measurements from CSV files
3. **Metabolomic Data** - Metabolite abundances from various formats
4. **Microbiome Data** - Taxonomic abundances from TSV/BIOM files

Genomic Data
------------

Format Requirements
~~~~~~~~~~~~~~~~~~~

Genomic data should be in VCF format (v4.1 or v4.2):

.. code-block:: text

    ##fileformat=VCFv4.2
    ##contig=<ID=chr1,length=248956422>
    #CHROM  POS     ID      REF ALT QUAL    FILTER  INFO    FORMAT  SAMPLE001   SAMPLE002
    chr1    1000    rs123   A   G   .       PASS    .       GT      0/1         1/1
    chr1    2000    rs456   C   T   .       PASS    .       GT      0/0         0/1

**Required columns:**

* CHROM - Chromosome
* POS - Position
* ID - SNP identifier
* REF - Reference allele
* ALT - Alternate allele
* FORMAT - Genotype format (must include GT)
* Sample columns - One per sample

Loading Genomic Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.data import load_vcf

    # Basic loading
    genotypes = load_vcf("data/genomic/genotypes.vcf")

    # With QC filtering
    genotypes = load_vcf(
        "data/genomic/genotypes.vcf",
        min_call_rate=0.95,      # Min % of samples with calls
        min_maf=0.01,             # Min minor allele frequency
        max_missing=0.05,         # Max missing rate per SNP
        exclude_indels=True       # Remove insertions/deletions
    )

**Output format:**

Returns a pandas DataFrame with:

* Rows: Samples
* Columns: SNP IDs
* Values: Genotypes (0, 1, 2 for reference, heterozygous, alternate)

Quality Control
~~~~~~~~~~~~~~~

Apply standard QC filters:

.. code-block:: python

    from audhd_correlation.data.genomic_loader import apply_qc_filters

    filtered = apply_qc_filters(
        genotypes,
        min_call_rate=0.95,
        min_maf=0.01,
        hwe_threshold=1e-6,        # Hardy-Weinberg equilibrium p-value
        ld_threshold=0.8,          # LD pruning threshold
        remove_related=True,       # Remove related samples (PI_HAT > 0.25)
    )

    print(f"Filtered from {genotypes.shape[1]} to {filtered.shape[1]} SNPs")

Clinical Data
-------------

Format Requirements
~~~~~~~~~~~~~~~~~~~

Clinical data should be in CSV format with specific columns:

.. code-block:: text

    sample_id,age,sex,diagnosis,severity_score,iq,site,bmi
    SAMPLE001,25,M,ASD,65.5,105,Site1,23.4
    SAMPLE002,32,F,ADHD,45.2,110,Site1,26.1
    SAMPLE003,28,M,AuDHD,75.8,98,Site2,22.9

**Required columns:**

* sample_id - Unique sample identifier
* age - Age in years
* sex - Sex (M/F or Male/Female)
* diagnosis - Diagnosis category (ASD, ADHD, AuDHD, Control)

**Optional columns:**

* severity_score - Clinical severity (0-100)
* iq - Intelligence quotient
* site - Study site for batch effect correction
* bmi - Body mass index
* medication - Current medications
* comorbidities - Additional diagnoses

Loading Clinical Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.data import load_clinical_csv

    # Basic loading
    clinical = load_clinical_csv("data/clinical/phenotypes.csv")

    # With validation
    clinical = load_clinical_csv(
        "data/clinical/phenotypes.csv",
        required_columns=['sample_id', 'age', 'sex', 'diagnosis'],
        validate_ranges=True,        # Check age, BMI ranges
        handle_missing='impute',     # 'drop', 'impute', or 'keep'
    )

**Output format:**

Returns a pandas DataFrame with:

* sample_id as index
* Numeric columns for continuous variables
* Categorical columns for discrete variables

Data Validation
~~~~~~~~~~~~~~~

Validate clinical data:

.. code-block:: python

    from audhd_correlation.data.clinical_loader import validate_clinical_data

    validation_report = validate_clinical_data(
        clinical,
        age_range=(18, 90),
        bmi_range=(15, 50),
        check_outliers=True,
    )

    # Print validation results
    print(validation_report)

    # Example output:
    # ✓ All required columns present
    # ✓ No duplicate sample IDs
    # ✓ Age range: 18-68 years
    # ⚠ 3 outliers detected in BMI
    # ⚠ 5 samples missing severity_score

Metabolomic Data
----------------

Format Requirements
~~~~~~~~~~~~~~~~~~~

Metabolomic data can be in CSV, TSV, or Excel format:

.. code-block:: text

    sample_id,metabolite_001,metabolite_002,metabolite_003
    SAMPLE001,1250.5,890.2,2340.1
    SAMPLE002,1180.3,920.5,2210.8
    SAMPLE003,1320.8,850.7,2450.3

**Requirements:**

* First column: sample_id
* Remaining columns: Metabolite abundances
* Values: Non-negative real numbers
* Missing values: NA, NaN, or empty cells

Loading Metabolomic Data
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.data import load_metabolomics

    # Basic loading
    metabolites = load_metabolomics("data/metabolomic/metabolites.csv")

    # With preprocessing
    metabolites = load_metabolomics(
        "data/metabolomic/metabolites.csv",
        log_transform=True,           # Log transform abundances
        remove_low_variance=True,     # Remove low-variance metabolites
        min_detection_rate=0.5,       # Min % samples with detection
        normalize_method='quantile',  # Normalization method
    )

**Output format:**

Returns a pandas DataFrame with:

* Rows: Samples
* Columns: Metabolite IDs
* Values: Abundances (optionally log-transformed and normalized)

Missing Value Handling
~~~~~~~~~~~~~~~~~~~~~~~

Handle missing metabolite measurements:

.. code-block:: python

    from audhd_correlation.data.metabolomic_loader import handle_missing_metabolites

    # Impute missing values
    imputed = handle_missing_metabolites(
        metabolites,
        method='knn',           # 'knn', 'half-min', 'zero', 'median'
        n_neighbors=5,          # For KNN imputation
        missing_threshold=0.5   # Drop metabolites with >50% missing
    )

Microbiome Data
---------------

Format Requirements
~~~~~~~~~~~~~~~~~~~

Microbiome data can be in TSV or BIOM format:

**TSV format:**

.. code-block:: text

    OTU_ID  SAMPLE001   SAMPLE002   SAMPLE003
    OTU_001 1250        890         2340
    OTU_002 3400        2100        3800
    OTU_003 560         890         420

**BIOM format** (v2.1): JSON-based format with taxonomy annotations.

Loading Microbiome Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.data import load_microbiome

    # Load from TSV
    microbiome = load_microbiome(
        "data/microbiome/abundances.tsv",
        format='tsv'
    )

    # Load from BIOM
    microbiome = load_microbiome(
        "data/microbiome/table.biom",
        format='biom'
    )

    # With filtering
    microbiome = load_microbiome(
        "data/microbiome/abundances.tsv",
        min_prevalence=0.1,          # Present in >10% samples
        min_abundance=10,            # Min total count
        relative_abundance=True,     # Convert to proportions
        normalize_samples=True,      # Normalize by sample total
    )

**Output format:**

Returns a pandas DataFrame with:

* Rows: Samples
* Columns: OTU/ASV IDs
* Values: Abundances (counts or relative abundances)

Taxonomic Aggregation
~~~~~~~~~~~~~~~~~~~~~~

Aggregate to higher taxonomic levels:

.. code-block:: python

    from audhd_correlation.data.microbiome_loader import aggregate_taxonomy

    # Aggregate to genus level
    genus_level = aggregate_taxonomy(
        microbiome,
        taxonomy_map="data/microbiome/taxonomy.tsv",
        level='genus'  # 'phylum', 'class', 'order', 'family', 'genus', 'species'
    )

Data Harmonization
------------------

After loading all modalities, harmonize sample IDs:

Aligning Samples
~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.data import align_multiomics

    # Load all data
    data = {
        'genomic': load_vcf("data/genomic/genotypes.vcf"),
        'clinical': load_clinical_csv("data/clinical/phenotypes.csv"),
        'metabolomic': load_metabolomics("data/metabolomic/metabolites.csv"),
        'microbiome': load_microbiome("data/microbiome/abundances.tsv"),
    }

    # Align samples (keep only common samples)
    aligned_data = align_multiomics(data)

    # Check alignment
    for modality, df in aligned_data.items():
        print(f"{modality}: {len(df)} samples")

    # Example output:
    # genomic: 150 samples
    # clinical: 150 samples
    # metabolomic: 150 samples
    # microbiome: 150 samples

Handling Missing Modalities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If not all samples have all modalities:

.. code-block:: python

    from audhd_correlation.data import align_multiomics

    # Keep samples with at least 2 modalities
    aligned_data = align_multiomics(
        data,
        min_modalities=2,
        strategy='union'  # 'intersection', 'union', or 'majority'
    )

Batch Effect Correction
~~~~~~~~~~~~~~~~~~~~~~~

Correct for site/batch effects:

.. code-block:: python

    from audhd_correlation.data import correct_batch_effects

    # Correct using ComBat
    corrected_data = correct_batch_effects(
        aligned_data,
        batch_column='site',
        method='combat',           # 'combat', 'limma', or 'harmony'
        preserve_covariates=['age', 'sex']  # Variables to preserve
    )

Complete Example
----------------

Here's a complete data loading workflow:

.. code-block:: python

    from audhd_correlation.data import (
        load_vcf,
        load_clinical_csv,
        load_metabolomics,
        load_microbiome,
        align_multiomics,
        correct_batch_effects,
    )

    # 1. Load each modality
    genomic = load_vcf(
        "data/genomic/genotypes.vcf",
        min_maf=0.01,
        min_call_rate=0.95
    )

    clinical = load_clinical_csv(
        "data/clinical/phenotypes.csv",
        validate_ranges=True
    )

    metabolomic = load_metabolomics(
        "data/metabolomic/metabolites.csv",
        log_transform=True,
        normalize_method='quantile'
    )

    microbiome = load_microbiome(
        "data/microbiome/abundances.tsv",
        min_prevalence=0.1,
        relative_abundance=True
    )

    # 2. Combine and align
    data = {
        'genomic': genomic,
        'clinical': clinical,
        'metabolomic': metabolomic,
        'microbiome': microbiome,
    }

    aligned_data = align_multiomics(data)

    # 3. Batch correction
    final_data = correct_batch_effects(
        aligned_data,
        batch_column='site',
        method='combat'
    )

    print(f"Final dataset: {len(final_data['genomic'])} samples")
    print(f"Modalities: {list(final_data.keys())}")

    # Save harmonized data
    for modality, df in final_data.items():
        df.to_hdf(f"data/harmonized/{modality}.h5", key='data')

Next Steps
----------

* :doc:`preprocessing` - Preprocess and normalize data
* :doc:`integration` - Integrate multi-omics data
* :doc:`../data_dictionaries/genomic` - Genomic data dictionary
* :doc:`../troubleshooting` - Troubleshooting data loading issues