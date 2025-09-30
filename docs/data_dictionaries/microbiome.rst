Microbiome Data Dictionary
==========================

Specification for microbiome data formats.

File Formats
------------

**TSV Format (Tab-Separated Values):**

.. code-block:: text

    OTU_ID  SAMPLE001   SAMPLE002   SAMPLE003
    OTU001  1250        890         2340
    OTU002  3400        2100        3800

**BIOM Format:** JSON-based hierarchical format with taxonomy

Required Structure
------------------

* Rows: OTUs/ASVs (Operational Taxonomic Units / Amplicon Sequence Variants)
* Columns: Samples
* Values: Read counts (integers ≥ 0)
* Optional: Taxonomy annotations

Taxonomic Levels
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Level
     - Example
   * - Kingdom
     - Bacteria
   * - Phylum
     - Firmicutes
   * - Class
     - Clostridia
   * - Order
     - Clostridiales
   * - Family
     - Lachnospiraceae
   * - Genus
     - Blautia
   * - Species
     - Blautia producta

Abundance Types
---------------

**Raw Counts:** Integer counts from sequencing

**Relative Abundance:** Proportions (0-1) or percentages (0-100)

.. math::

    RA_i = \\frac{count_i}{\\sum counts}

**CLR-transformed:** Centered log-ratio

.. math::

    CLR(x) = \\log\\left(\\frac{x}{g(x)}\\right)

where :math:`g(x)` is geometric mean.

Quality Filters
---------------

* **Prevalence:** Present in ≥10% of samples
* **Abundance:** Total count ≥10 across all samples
* **Rarefaction:** Subsample to even depth (optional)

Example TSV
-----------

.. code-block:: text

    OTU_ID  taxonomy    SAMPLE001   SAMPLE002
    OTU001  k__Bacteria;p__Firmicutes;c__Clostridia 1250    890
    OTU002  k__Bacteria;p__Bacteroidetes;c__Bacteroidia 3400    2100

Notes
-----

* Use 16S rRNA or shotgun metagenomic data
* Minimum sequencing depth: 10,000 reads per sample
* Remove contaminants and chimeras before analysis