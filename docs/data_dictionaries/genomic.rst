Genomic Data Dictionary
=======================

Specification for genomic data formats.

VCF Format
----------

**File Type:** Variant Call Format (VCF) v4.1 or v4.2

**Required Columns:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - CHROM
     - String
     - Chromosome identifier (e.g., chr1, chr2, ... chrX, chrY)
   * - POS
     - Integer
     - 1-based position on chromosome
   * - ID
     - String
     - SNP identifier (e.g., rs123456, or . if unknown)
   * - REF
     - String
     - Reference allele (A, C, G, or T)
   * - ALT
     - String
     - Alternate allele(s), comma-separated if multiple
   * - QUAL
     - Float
     - Phred-scaled quality score (or . if unavailable)
   * - FILTER
     - String
     - PASS or filter codes (or . if unavailable)
   * - INFO
     - String
     - Semicolon-separated annotations
   * - FORMAT
     - String
     - Genotype format fields (must include GT)
   * - Sample columns
     - String
     - One column per sample with genotype calls

**Genotype Encoding:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Genotype
     - Description
   * - 0/0 or 0|0
     - Homozygous reference
   * - 0/1 or 0|1
     - Heterozygous
   * - 1/1 or 1|1
     - Homozygous alternate
   * - ./. or .|.
     - Missing genotype

**Internal Representation:**

After loading, genotypes are encoded as:

* 0 = Homozygous reference (REF/REF)
* 1 = Heterozygous (REF/ALT)
* 2 = Homozygous alternate (ALT/ALT)
* NaN = Missing

**Example VCF:**

.. code-block:: text

    ##fileformat=VCFv4.2
    ##reference=GRCh38
    #CHROM POS     ID      REF ALT QUAL  FILTER INFO FORMAT SAMPLE001 SAMPLE002
    chr1   100 rs001   A   G   .     PASS   .    GT     0/1       1/1
    chr1   200 rs002   C   T   .     PASS   .    GT     0/0       0/1
    chr2   150 rs003   G   A   .     PASS   .    GT     1/1       0/1

Quality Control Metrics
-----------------------

**Call Rate:**

Percentage of samples with non-missing genotype:

.. math::

    \\text{Call Rate} = \\frac{\\text{# non-missing}}{\\text{# samples}}

**Minor Allele Frequency (MAF):**

Frequency of the less common allele:

.. math::

    \\text{MAF} = \\min(p, 1-p)

where :math:`p` is the allele frequency.

**Hardy-Weinberg Equilibrium (HWE):**

Chi-square test for deviation from HWE:

.. math::

    \\chi^2 = \\frac{(O_{het} - E_{het})^2}{E_{het}}

Expected heterozygosity:

.. math::

    E_{het} = 2pq

**Recommended Filters:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Metric
     - Threshold
     - Rationale
   * - Call rate
     - ≥ 0.95
     - Remove low-quality variants
   * - MAF
     - ≥ 0.01
     - Rare variants have low power
   * - HWE p-value
     - ≥ 1e-6
     - Remove genotyping errors
   * - Sample call rate
     - ≥ 0.90
     - Remove low-quality samples

Annotation Fields
-----------------

**Common INFO fields:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - AC
     - Integer
     - Allele count
   * - AF
     - Float
     - Allele frequency
   * - AN
     - Integer
     - Total number of alleles
   * - DP
     - Integer
     - Total read depth
   * - GENE
     - String
     - Gene symbol
   * - CSQ
     - String
     - Variant consequence (from VEP)

Genomic Coordinates
-------------------

**Reference Genome:** GRCh38/hg38 (default) or GRCh37/hg19

**Chromosome Naming:**

* Use "chr" prefix: chr1, chr2, ..., chrX, chrY, chrM
* Or without: 1, 2, ..., X, Y, MT

**Position:** 1-based coordinate system

File Size Expectations
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Dataset
     - File Size
     - Notes
   * - 100 samples, 1M SNPs
     - ~500 MB
     - Uncompressed VCF
   * - 100 samples, 1M SNPs
     - ~50 MB
     - Compressed (.vcf.gz)
   * - 1000 samples, 1M SNPs
     - ~5 GB
     - Uncompressed
   * - 1000 samples, 1M SNPs
     - ~500 MB
     - Compressed

Notes
-----

* Phased genotypes (0|1) vs unphased (0/1) are treated identically
* Multi-allelic variants should be split into biallelic records
* Indels are typically filtered out for clustering analysis
* X chromosome requires special handling for males/females