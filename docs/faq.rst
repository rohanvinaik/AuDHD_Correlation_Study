Frequently Asked Questions
==========================

General Questions
-----------------

**Q: What is the AuDHD Correlation Study pipeline?**

A: A comprehensive analysis pipeline for identifying patient subtypes using multi-omics data (genomic, clinical, metabolomic, microbiome) with focus on ASD-ADHD correlations.

**Q: What programming language is the pipeline written in?**

A: Python 3.9+, with dependencies on scientific computing libraries (NumPy, pandas, scikit-learn).

**Q: Is the pipeline open source?**

A: Yes, released under MIT License.

**Q: Can I use this for other diseases besides AuDHD?**

A: Yes! The pipeline is designed to be general-purpose for any multi-omics clustering analysis.

Installation & Setup
--------------------

**Q: Which Python version should I use?**

A: Python 3.9, 3.10, or 3.11. Python 3.10 is recommended.

**Q: Do I need a GPU?**

A: No, but GPU acceleration (via CUDA) speeds up some operations.

**Q: How much RAM do I need?**

A: Minimum 8 GB, recommended 16 GB. For large datasets (> 1000 samples), 32 GB+ recommended.

**Q: Can I run this on Windows?**

A: Yes, via Windows Subsystem for Linux (WSL2). Native Windows support is experimental.

**Q: How long does installation take?**

A: 5-10 minutes including all dependencies.

Data Requirements
-----------------

**Q: What data formats are supported?**

A:

* Genomic: VCF (v4.1/v4.2)
* Clinical: CSV
* Metabolomic: CSV, TSV, Excel
* Microbiome: TSV, BIOM

**Q: How many samples do I need?**

A: Minimum 50 samples recommended, 100+ preferred for robust clustering.

**Q: Can I analyze just one data modality?**

A: Yes, but multi-omics integration provides richer insights.

**Q: What if samples have missing data?**

A: The pipeline handles missing values through imputation. Missing < 30% per feature recommended.

**Q: Do all samples need all modalities?**

A: No. Use ``min_modalities`` parameter to specify minimum required (e.g., 2 out of 4).

Analysis Questions
------------------

**Q: How do I choose the number of clusters?**

A: Use silhouette analysis or elbow method:

.. code-block:: python

    from audhd_correlation.modeling import select_n_clusters_silhouette
    n_clusters = select_n_clusters_silhouette(data, max_clusters=10)

Or let HDBSCAN determine automatically.

**Q: What's the difference between HDBSCAN and K-means?**

A:

* **HDBSCAN**: Finds variable-sized clusters, handles noise, auto-determines k
* **K-means**: Fixed k, all points assigned, faster

**Q: How long does analysis take?**

A:

* Small (100 samples): 5-10 minutes
* Medium (500 samples): 30-60 minutes
* Large (1000+ samples): 2-4 hours

**Q: Can I parallelize the analysis?**

A: Yes:

.. code-block:: yaml

    processing:
      n_jobs: -1  # Use all CPU cores

**Q: What does a good silhouette score mean?**

A:

* \> 0.7: Strong clusters
* 0.5-0.7: Reasonable structure
* 0.25-0.5: Weak structure
* < 0.25: No clear structure

**Q: My clusters don't match clinical diagnosis. Is that wrong?**

A: No! Unsupervised clustering discovers data-driven subtypes that may not align with diagnostic categories. This can reveal novel subtypes.

Integration Methods
-------------------

**Q: Should I use MOFA or PCA?**

A:

* **MOFA**: Better for understanding shared/specific variation, slower
* **PCA**: Faster, simpler, good for initial exploration

**Q: How many factors/components should I use?**

A: Start with 10-20. Use elbow plot or cumulative variance explained (aim for 70-80%).

**Q: What's the difference between factors and components?**

A:

* **Factors** (MOFA): Can be modality-specific or shared
* **Components** (PCA): Linear combinations of all features

Results Interpretation
----------------------

**Q: How do I interpret cluster assignments?**

A: Compare clusters by:

1. Clinical variables (age, severity, diagnosis)
2. Differentially abundant features
3. Pathway enrichment
4. Clinical outcomes

**Q: What if I only find one cluster?**

A: Possible reasons:

* Data too homogeneous
* Need more samples
* Try different preprocessing
* Adjust clustering parameters

**Q: Can clusters predict clinical outcomes?**

A: The pipeline identifies subtypes. For outcome prediction, train a supervised model using cluster assignments as features.

**Q: How do I validate my results?**

A:

1. Internal validation (silhouette, stability)
2. External validation (independent cohort)
3. Clinical validation (association with outcomes)
4. Biological validation (pathway enrichment)

Troubleshooting
---------------

**Q: I get "Out of Memory" errors. What should I do?**

A:

1. Reduce batch size
2. Enable checkpointing
3. Use PCA instead of MOFA
4. Feature selection to reduce dimensions

**Q: Clustering is too slow. How can I speed it up?**

A:

1. Use K-means instead of HDBSCAN
2. Enable parallel processing (``n_jobs=-1``)
3. Reduce data size through feature selection
4. Use PCA for quick integration

**Q: Results aren't reproducible across runs. Why?**

A: Set random seed:

.. code-block:: python

    np.random.seed(42)
    result = perform_clustering(data, method='kmeans', random_state=42)

**Q: Where can I find the log files?**

A: Check ``logs/pipeline.log`` or ``logs/pipeline_YYYYMMDD.log``.

Advanced Usage
--------------

**Q: Can I customize the pipeline?**

A: Yes! Use the Python API:

.. code-block:: python

    from audhd_correlation.data import load_multiomics
    from audhd_correlation.preprocess import impute_missing
    # ... custom steps ...

**Q: Can I add my own integration method?**

A: Yes, implement the integration interface:

.. code-block:: python

    from audhd_correlation.integrate import BaseIntegrator

    class MyIntegrator(BaseIntegrator):
        def fit(self, data):
            # Your method here
            pass

**Q: How do I create custom visualizations?**

A: Access the raw data:

.. code-block:: python

    import matplotlib.pyplot as plt

    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels)
    # Customize as needed

**Q: Can I export results to R?**

A: Yes:

.. code-block:: python

    # Save as CSV
    results.to_csv('results.csv')

    # Or use feather format
    import pyarrow.feather as feather
    feather.write_feather(results, 'results.feather')

Citation & Publication
----------------------

**Q: How do I cite this pipeline?**

A::

    AuDHD Correlation Study Pipeline (2024)
    https://github.com/your-repo/AuDHD_Correlation_Study

**Q: Can I use this for my publication?**

A: Yes, the pipeline is freely available under MIT License.

**Q: Are there example publications using this pipeline?**

A: See the documentation for case studies and example publications.

Support & Community
-------------------

**Q: Where can I get help?**

A:

* Documentation: https://audhd-pipeline.readthedocs.io
* GitHub Issues: https://github.com/your-repo/issues
* Discussions: https://github.com/your-repo/discussions

**Q: How do I report a bug?**

A: Create an issue on GitHub with:

* Error message
* Minimal reproducible example
* System information (OS, Python version)

**Q: Can I contribute to the project?**

A: Yes! See :doc:`contributing` for guidelines.

**Q: Is there a mailing list?**

A: Join our discussions at https://github.com/your-repo/discussions

Didn't find your question?
---------------------------

* Check :doc:`troubleshooting` for common issues
* Search GitHub issues: https://github.com/your-repo/issues
* Ask on GitHub Discussions: https://github.com/your-repo/discussions