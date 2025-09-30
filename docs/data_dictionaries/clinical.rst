Clinical Data Dictionary
========================

Specification for clinical phenotype data.

File Format
-----------

**File Type:** CSV (Comma-Separated Values)

**Encoding:** UTF-8

**Column Separator:** Comma (,)

**Missing Values:** Empty cells, "NA", "N/A", or "NaN"

Required Columns
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Column
     - Type
     - Required
     - Description
   * - sample_id
     - String
     - Yes
     - Unique sample identifier matching other modalities
   * - age
     - Float
     - Yes
     - Age in years (integer or decimal)
   * - sex
     - String
     - Yes
     - Biological sex: "M" or "F" (or "Male"/"Female")
   * - diagnosis
     - String
     - Yes
     - Primary diagnosis category

Optional Columns
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - severity_score
     - Float
     - Clinical severity score (0-100 scale)
   * - iq
     - Integer
     - Intelligence quotient
   * - bmi
     - Float
     - Body mass index (kg/m²)
   * - site
     - String
     - Study site identifier for batch correction
   * - ethnicity
     - String
     - Self-reported ethnicity
   * - education_years
     - Integer
     - Years of education completed
   * - medication
     - String
     - Current medications (semicolon-separated)
   * - comorbidities
     - String
     - Additional diagnoses (semicolon-separated)
   * - family_history
     - Boolean
     - Family history of condition (True/False)
   * - symptom_onset_age
     - Float
     - Age at symptom onset
   * - treatment_duration
     - Float
     - Duration of treatment in months

Diagnosis Categories
--------------------

**Primary Categories:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Category
     - Description
   * - ASD
     - Autism Spectrum Disorder
   * - ADHD
     - Attention-Deficit/Hyperactivity Disorder
   * - AuDHD
     - Co-occurring ASD and ADHD
   * - Control
     - Neurotypical control

**Subtypes (optional):**

* ASD_High_Functioning
* ASD_Moderate
* ADHD_Inattentive
* ADHD_Hyperactive
* ADHD_Combined

Value Ranges
------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Variable
     - Valid Range
     - Notes
   * - age
     - 0-120
     - Typically 5-80 for study
   * - severity_score
     - 0-100
     - Higher = more severe
   * - iq
     - 40-160
     - Population mean = 100, SD = 15
   * - bmi
     - 12-50
     - Underweight < 18.5, Normal 18.5-24.9
   * - education_years
     - 0-25
     - Years of formal education

Example File
------------

.. code-block:: text

    sample_id,age,sex,diagnosis,severity_score,iq,bmi,site,ethnicity
    SAMPLE001,25,M,ASD,65.5,105,23.4,Site1,Caucasian
    SAMPLE002,32,F,ADHD,45.2,110,26.1,Site1,Hispanic
    SAMPLE003,28,M,AuDHD,75.8,98,22.9,Site2,Caucasian
    SAMPLE004,45,F,Control,NA,115,24.5,Site2,Asian
    SAMPLE005,19,M,ASD,58.3,92,21.8,Site1,African American

Data Quality Checks
-------------------

**Automatic Validation:**

1. **Sample ID uniqueness**

   * All sample_ids must be unique
   * Sample_ids must match other modalities

2. **Age range**

   * Warning if age < 5 or age > 90
   * Error if age < 0 or age > 120

3. **Sex encoding**

   * Accepted: M, F, Male, Female
   * Converted to: M, F

4. **Diagnosis category**

   * Must match predefined categories
   * Case-insensitive matching

5. **Missing values**

   * Report percentage missing per column
   * Error if > 50% missing in required columns

**Outlier Detection:**

Variables checked for outliers (> 3 SD from mean):

* age
* severity_score
* iq
* bmi

Severity Score Definitions
---------------------------

**General Scale (0-100):**

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Range
     - Classification
     - Description
   * - 0-25
     - Minimal
     - Minimal impairment in daily functioning
   * - 26-50
     - Mild
     - Some impairment, mostly independent
   * - 51-75
     - Moderate
     - Significant impairment, needs support
   * - 76-100
     - Severe
     - Major impairment, needs substantial support

**ASD-Specific (ADOS-2 based):**

* Score 1-3: Minimal/no evidence
* Score 4-7: Mild-moderate
* Score 8-10: Moderate-severe

**ADHD-Specific (ADHD Rating Scale):**

* Score 0-18: No symptoms
* Score 19-36: Mild
* Score 37-54: Moderate
* Score 55-72: Severe

Clinical Assessment Tools
--------------------------

**Recommended Assessment Batteries:**

**For ASD:**

* ADOS-2 (Autism Diagnostic Observation Schedule)
* ADI-R (Autism Diagnostic Interview-Revised)
* SRS-2 (Social Responsiveness Scale)

**For ADHD:**

* CAARS (Conners' Adult ADHD Rating Scales)
* ASRS (Adult ADHD Self-Report Scale)
* DIVA (Diagnostic Interview for ADHD in adults)

**Cognitive Assessment:**

* WAIS-IV (Wechsler Adult Intelligence Scale)
* WISC-V (for children)

Data Privacy
------------

**Protected Health Information (PHI):**

The following must NOT be included:

* Full names
* Dates of birth (use age instead)
* Addresses
* Phone numbers
* Medical record numbers
* Social security numbers
* Email addresses

**De-identification:**

* Use pseudonymous sample_id (e.g., SAMPLE001, SUBJ_A01)
* Remove dates (use intervals: "6 months since diagnosis")
* Aggregate rare values (e.g., rare ethnicity → "Other")

Batch Effects
-------------

**Site Variable:**

If data collected at multiple sites, include site identifier:

.. code-block:: text

    site,description,n_samples,collection_period
    Site1,University Hospital A,150,2020-2021
    Site2,Research Center B,120,2021-2022
    Site3,Clinical Practice C,80,2020-2022

**Batch Correction:**

The pipeline will automatically correct for site effects using ComBat.

Notes
-----

* Age should be at time of sample collection
* Sex refers to biological sex assigned at birth
* Diagnosis should be DSM-5 based if possible
* Use consistent units across all samples
* Document any study-specific scales in metadata