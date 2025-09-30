Best Practices
==============

Guidelines for optimal use of the pipeline.

Data Preparation
----------------

Sample Size
~~~~~~~~~~~

**Minimum recommendations:**

* Overall: ≥ 50 samples
* Per cluster: ≥ 20 samples
* Optimal: 100-500 samples

**Power calculations:**

For detecting medium effect size (Cohen's d = 0.5):

* 2 clusters: 64 samples needed (80% power)
* 3 clusters: 156 samples needed
* 4 clusters: 276 samples needed

Data Quality
~~~~~~~~~~~~

**Before analysis:**

1. Remove duplicates
2. Check for sample swaps
3. Verify sample IDs match across modalities
4. Check missing data patterns
5. Identify and investigate outliers

**Quality thresholds:**

* Missing rate: < 30% per feature
* Sample call rate: > 90%
* Batch size: ≥ 10 samples per batch

Preprocessing
-------------

Order of Operations
~~~~~~~~~~~~~~~~~~~

**Correct order:**

1. Load data
2. Quality control filtering
3. Sample alignment
4. Missing value imputation
5. Transformation (log, CLR)
6. Normalization/scaling
7. Feature selection
8. Batch correction

**Why order matters:**

* Batch correction works better on non-scaled data
* Imputation before transformation prevents log(0)
* Feature selection after scaling ensures fair comparison

Method Selection
~~~~~~~~~~~~~~~~

**By modality:**

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Modality
     - Transform
     - Scale
   * - Genomic
     - None
     - Standard
   * - Clinical
     - None
     - Robust
   * - Metabolomic
     - Log2
     - Standard
   * - Microbiome
     - CLR
     - None

Integration
-----------

Number of Factors
~~~~~~~~~~~~~~~~~

**Selection criteria:**

* Aim for 70-80% variance explained
* Use elbow method
* Typical range: 10-30 factors

**Too few factors:**

* Miss important variation
* Poor clustering

**Too many factors:**

* Overfitting
* Increased noise
* Slower computation

Method Selection
~~~~~~~~~~~~~~~~

+-------------+----------------------+-------------------------+
| Data Size   | Recommended Method   | Alternative             |
+=============+======================+=========================+
| < 100       | PCA                  | NMF                     |
+-------------+----------------------+-------------------------+
| 100-500     | MOFA                 | PCA                     |
+-------------+----------------------+-------------------------+
| > 500       | PCA                  | MOFA (if resources)     |
+-------------+----------------------+-------------------------+

Clustering
----------

Method Selection
~~~~~~~~~~~~~~~~

**HDBSCAN when:**

* Don't know number of clusters
* Expect noise points
* Variable cluster sizes
* Hierarchical structure

**K-means when:**

* Know number of clusters
* Want even-sized clusters
* Need speed
* Spherical clusters

Parameter Tuning
~~~~~~~~~~~~~~~~

**HDBSCAN min_cluster_size:**

* Start with: sample_size / 5
* Minimum: 20
* Increase if too many small clusters
* Decrease if only finding 1 cluster

**K-means n_clusters:**

* Use silhouette analysis
* Try range: 2-10
* Consider domain knowledge
* Multiple runs with different k

Validation
----------

Essential Checks
~~~~~~~~~~~~~~~~

**Always compute:**

1. Silhouette score (> 0.4 acceptable)
2. Bootstrap stability (ARI > 0.6)
3. Cluster sizes (≥ 20 samples)

**If low quality:**

* Try different clustering method
* Adjust parameters
* Improve preprocessing
* Check for batch effects
* May indicate continuous variation

Statistical Testing
~~~~~~~~~~~~~~~~~~~

**Multiple testing correction:**

* Always use FDR correction
* Bonferroni too conservative
* Report both raw and adjusted p-values

**Sample size considerations:**

* Small samples: Non-parametric tests
* Large samples: Parametric tests acceptable
* Always check assumptions

Interpretation
--------------

Clinical Validation
~~~~~~~~~~~~~~~~~~~

**Check associations with:**

* Age, sex, diagnosis
* Severity measures
* Treatment response
* Outcomes

**Interpreting results:**

* Clusters matching diagnosis: Validates method but limited novelty
* Clusters cutting across diagnosis: Potential novel subtypes
* No associations: Check data quality, may be technical artifacts

Biological Validation
~~~~~~~~~~~~~~~~~~~~~

**Pathway enrichment:**

* Use multiple databases
* FDR < 0.05 recommended
* Look for biological coherence
* Validate key pathways experimentally

**External validation:**

* Test on independent cohort
* Replication is crucial
* Document differences

Reporting
---------

What to Report
~~~~~~~~~~~~~~

**Methods:**

* All preprocessing steps with parameters
* Integration method and settings
* Clustering algorithm and parameters
* Validation approaches
* Software versions

**Results:**

* Sample sizes (total and per cluster)
* Quality metrics (silhouette, stability)
* Clinical characteristics per cluster
* Top differentiating features
* Enriched pathways

**Figures:**

* UMAP/t-SNE plot with clusters
* Heatmap of cluster profiles
* Clinical variable distributions
* Validation metrics

Reproducibility
~~~~~~~~~~~~~~~

**Ensure reproducibility:**

1. Set random seeds
2. Document all parameters
3. Record software versions
4. Share code and configuration
5. Archive data with appropriate access

**Version control:**

* Git for code
* DVC for data
* Docker for environment
* Zenodo for archiving

Common Pitfalls
---------------

**❌ Don't:**

* Skip quality control
* Use default parameters blindly
* Ignore batch effects
* Over-interpret weak clusters
* Cherry-pick results
* Skip validation
* Forget multiple testing correction

**✓ Do:**

* Perform thorough QC
* Optimize parameters
* Correct for batches
* Validate rigorously
* Report all findings
* Use appropriate statistics
* Document everything

Performance Optimization
------------------------

For Large Datasets
~~~~~~~~~~~~~~~~~~

**Speed up analysis:**

1. Use PCA instead of MOFA
2. Feature selection (top 1000 features)
3. Parallel processing (n_jobs=-1)
4. K-means instead of HDBSCAN
5. Reduce bootstrap iterations

**Memory optimization:**

1. Process in batches
2. Use sparse matrices where possible
3. Enable checkpointing
4. Clear intermediate results

Quality Assurance
-----------------

Checklist
~~~~~~~~~

**Before running:**

☐ Data formats validated
☐ Sample IDs aligned
☐ Missing data < 30%
☐ No obvious outliers
☐ Batch variables identified

**After preprocessing:**

☐ Missing values imputed
☐ Features scaled appropriately
☐ Batch effects corrected
☐ QC metrics computed

**After clustering:**

☐ Cluster sizes reasonable (≥ 20)
☐ Silhouette score > 0.25
☐ Bootstrap stability > 0.5
☐ Visual inspection of embedding

**Before publication:**

☐ Results validated on independent data
☐ Biological interpretation makes sense
☐ All code and data documented
☐ Reproducibility confirmed

Resources
---------

**Further reading:**

* MOFA: https://biofam.github.io/MOFA2/
* HDBSCAN: https://hdbscan.readthedocs.io/
* Multi-omics reviews: Huang et al. (2021)
* Clustering validation: Hennig (2007)