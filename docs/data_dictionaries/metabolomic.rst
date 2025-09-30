Metabolomic Data Dictionary
===========================

Specification for metabolomic data formats.

File Format
-----------

**Supported Formats:** CSV, TSV, Excel (.xlsx)

**Layout:** Features in columns, samples in rows

Required Structure
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Element
     - Description
   * - First column
     - sample_id matching other modalities
   * - Remaining columns
     - Metabolite abundances with unique IDs
   * - Values
     - Non-negative real numbers
   * - Missing
     - NA, NaN, or empty cells

Metabolite Naming
-----------------

**Preferred Format:**

* Use standard identifiers (HMDB, KEGG, PubChem)
* Example: HMDB0000001, C00001, CID:5793

**Alternative:**

* Common names: glucose, lactate, creatinine
* m/z_rt format: M123.456_RT2.34

Value Ranges
------------

* **Minimum:** 0 (no negative values)
* **Typical range:** 10² to 10⁶ (platform-dependent)
* **Units:** Concentration (µM, nM) or arbitrary units (AU)
* **Missing:** 0-30% per metabolite typical

Preprocessing
-------------

**Log transformation recommended:**

.. math::

    x_{transformed} = \\log_2(x + 1)

**Normalization methods:**

* Quantile normalization
* Probabilistic quotient normalization (PQN)
* Total ion current (TIC) normalization

Example File
------------

.. code-block:: text

    sample_id,HMDB0000001,HMDB0000002,HMDB0000122
    SAMPLE001,1250.5,890.2,2340.1
    SAMPLE002,1180.3,920.5,2210.8
    SAMPLE003,NA,850.7,2450.3

Quality Metrics
---------------

* **Detection rate:** Percentage of samples with measurement
* **CV (Coefficient of Variation):** Within-batch variability
* **Blank ratio:** Signal-to-blank ratio > 3

Platform Information
--------------------

Include in metadata:

* Platform: LC-MS, GC-MS, NMR
* Ionization mode: Positive/Negative (for MS)
* Resolution: High/Low
* Normalization: Method used