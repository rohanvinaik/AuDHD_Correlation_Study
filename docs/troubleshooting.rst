Troubleshooting Guide
=====================

Common issues and solutions.

Installation Issues
-------------------

**Problem: ImportError after installation**

Solution:

.. code-block:: bash

    pip install -e .
    python -c "import audhd_correlation"

**Problem: CUDA errors**

Solution: Install matching cupy version:

.. code-block:: bash

    nvcc --version  # Check CUDA version
    pip install cupy-cuda11x

Data Loading Issues
-------------------

**Problem: VCF file not loading**

Causes:

* Invalid VCF format
* Missing header
* Incorrect file path

Solution:

.. code-block:: python

    from audhd_correlation.data import validate_vcf

    errors = validate_vcf("data/genotypes.vcf")
    for error in errors:
        print(error)

**Problem: Sample ID mismatch between modalities**

Solution:

.. code-block:: python

    # Check sample IDs
    for modality, df in data.items():
        print(f"{modality}: {df.index[:5]}")

    # Force alignment
    aligned = align_multiomics(data, strategy='intersection')

**Problem: Too many missing values**

Solution:

.. code-block:: python

    # Check missing rate
    missing_rate = data.isna().sum().sum() / data.size
    print(f"Missing: {missing_rate:.1%}")

    # Filter features with > 50% missing
    data_filtered = data.loc[:, data.isna().mean() < 0.5]

Memory Issues
-------------

**Problem: Out of memory during integration**

Solutions:

1. Reduce batch size:

.. code-block:: yaml

    processing:
      batch_size: 100

2. Enable checkpointing:

.. code-block:: yaml

    processing:
      enable_checkpointing: true

3. Use PCA instead of MOFA:

.. code-block:: python

    integrated = integrate_multiomics(data, method='pca')

**Problem: Memory error during clustering**

Solutions:

1. Reduce number of samples:

.. code-block:: python

    # Sample randomly
    sample_idx = np.random.choice(len(data), size=1000, replace=False)
    data_subset = data.iloc[sample_idx]

2. Use mini-batch K-means:

.. code-block:: python

    from sklearn.cluster import MiniBatchKMeans

    clusterer = MiniBatchKMeans(n_clusters=3, batch_size=100)
    labels = clusterer.fit_predict(data)

Clustering Issues
-----------------

**Problem: HDBSCAN finds only one cluster**

Causes:

* min_cluster_size too large
* Data too homogeneous
* Need dimension reduction

Solutions:

.. code-block:: python

    # Try smaller min_cluster_size
    result = perform_clustering(data, method='hdbscan', min_cluster_size=10)

    # Or use K-means with fixed k
    result = perform_clustering(data, method='kmeans', n_clusters=3)

**Problem: Too many noise points in HDBSCAN**

Solutions:

.. code-block:: python

    # Adjust cluster_selection_epsilon
    result = perform_clustering(
        data,
        method='hdbscan',
        min_cluster_size=20,
        cluster_selection_epsilon=0.5  # Increase to include more points
    )

**Problem: Clusters not stable across runs**

Causes:

* Random initialization
* Insufficient data
* Poor data quality

Solutions:

.. code-block:: python

    # Set random seed
    result = perform_clustering(data, method='kmeans', random_state=42)

    # Use consensus clustering
    from audhd_correlation.modeling import consensus_clustering
    result = consensus_clustering(data, n_iterations=100)

Validation Issues
-----------------

**Problem: Low silhouette score**

Interpretation: Silhouette < 0.25 indicates weak clusters

Solutions:

1. Try different number of clusters
2. Improve preprocessing
3. Use different clustering method

**Problem: Unstable bootstrap results**

Causes:

* Too few bootstrap iterations
* Small sample size
* Weak clustering structure

Solutions:

.. code-block:: python

    # Increase bootstrap iterations
    stability = bootstrap_stability(data, labels, n_bootstrap=500)

    # Check sample size per cluster
    cluster_sizes = np.bincount(labels[labels >= 0])
    print(f"Cluster sizes: {cluster_sizes}")
    # Aim for ≥20 samples per cluster

Performance Issues
------------------

**Problem: Pipeline too slow**

Solutions:

1. Use parallel processing:

.. code-block:: yaml

    processing:
      n_jobs: -1  # Use all cores

2. Reduce data size:

.. code-block:: python

    # Feature selection
    data_filtered = select_features(
        data,
        max_features=1000
    )

3. Use faster methods:

.. code-block:: python

    # PCA instead of MOFA
    integrated = integrate_multiomics(data, method='pca')

    # K-means instead of HDBSCAN
    clusters = perform_clustering(data, method='kmeans')

Visualization Issues
--------------------

**Problem: Figures not displaying in Jupyter**

Solution:

.. code-block:: python

    %matplotlib inline
    import matplotlib.pyplot as plt

**Problem: Plot too crowded**

Solutions:

.. code-block:: python

    # Adjust figure size
    plt.figure(figsize=(15, 10))

    # Reduce point size
    plt.scatter(x, y, s=10, alpha=0.5)

    # Sample points
    n_display = 1000
    idx = np.random.choice(len(data), n_display, replace=False)
    plt.scatter(x[idx], y[idx])

Result Interpretation Issues
-----------------------------

**Problem: Clusters don't match clinical diagnosis**

Interpretation: This is expected! Unsupervised clustering discovers data-driven subtypes, not diagnostic categories.

Analysis:

.. code-block:: python

    # Check diagnosis distribution per cluster
    pd.crosstab(labels, clinical['diagnosis'], normalize='index')

    # Look for enrichment
    from scipy.stats import fisher_exact
    for cluster_id in range(n_clusters):
        # Test each cluster vs others
        pass

**Problem: No significant pathway enrichment**

Causes:

* Insufficient statistical power
* Wrong pathway database
* Clusters not biologically distinct

Solutions:

.. code-block:: python

    # Try different databases
    enrichment = pathway_enrichment(
        signatures,
        databases=['KEGG', 'GO_BP', 'Reactome', 'WikiPathways']
    )

    # Use more lenient threshold
    enrichment = pathway_enrichment(
        signatures,
        fdr_threshold=0.1  # Less stringent
    )

Configuration Issues
--------------------

**Problem: Config file not loading**

Solution:

.. code-block:: bash

    # Validate YAML syntax
    python -c "import yaml; yaml.safe_load(open('config.yaml'))"

**Problem: Parameters not taking effect**

Check parameter precedence:

1. Command-line arguments (highest)
2. Config file
3. Default values (lowest)

.. code-block:: bash

    # Override config
    audhd-pipeline run --config config.yaml --n-clusters 5

Getting Help
------------

If issues persist:

1. Check logs: ``logs/pipeline.log``
2. Enable debug mode:

.. code-block:: yaml

    logging:
      level: DEBUG

3. Report issues: https://github.com/your-repo/issues

Include:

* Error message
* Full traceback
* Config file
* Python version
* OS version