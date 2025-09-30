Quick Start Guide
=================

Get started with the AuDHD Correlation Study pipeline in 5 minutes.

Basic Usage
-----------

1. **Prepare Your Data**

Create a data directory with your input files:

.. code-block:: bash

    data/
    ├── genomic/
    │   └── genotypes.vcf
    ├── clinical/
    │   └── phenotypes.csv
    ├── metabolomic/
    │   └── metabolites.csv
    └── microbiome/
        └── abundances.tsv

2. **Create Configuration**

Create ``config.yaml``:

.. code-block:: yaml

    data:
      input_dir: "data/"
      output_dir: "outputs/"

    processing:
      modalities:
        - genomic
        - clinical
        - metabolomic

    clustering:
      method: "hdbscan"
      min_cluster_size: 20

3. **Run Analysis**

.. code-block:: bash

    audhd-pipeline run --config config.yaml

That's it! Results will be in ``outputs/``.

Example with Sample Data
-------------------------

We provide sample data for testing:

.. code-block:: bash

    # Download sample data
    audhd-pipeline download-sample-data

    # Run with sample data
    audhd-pipeline run --config configs/sample_analysis.yaml

This runs a complete analysis on synthetic data (~2 minutes).

Python API
----------

You can also use the Python API:

.. code-block:: python

    from audhd_correlation import Pipeline

    # Create pipeline
    pipeline = Pipeline(config_path="config.yaml")

    # Load data
    data = pipeline.load_data()

    # Run analysis
    results = pipeline.run()

    # Generate report
    pipeline.generate_report(results, output_path="report.html")

Step-by-Step Example
---------------------

Here's a complete example with all steps:

.. code-block:: python

    import pandas as pd
    from audhd_correlation.data import load_multiomics
    from audhd_correlation.preprocess import preprocess_pipeline
    from audhd_correlation.integrate import integrate_multiomics
    from audhd_correlation.modeling import perform_clustering
    from audhd_correlation.validation import validate_clusters
    from audhd_correlation.viz import plot_results

    # 1. Load data
    data = load_multiomics(
        genomic_path="data/genomic/genotypes.vcf",
        clinical_path="data/clinical/phenotypes.csv",
        metabolomic_path="data/metabolomic/metabolites.csv"
    )

    # 2. Preprocess
    preprocessed = preprocess_pipeline(
        data,
        impute_method="knn",
        scale_method="standard",
        batch_correct=True
    )

    # 3. Integrate
    integrated = integrate_multiomics(
        preprocessed,
        method="mofa",
        n_factors=15
    )

    # 4. Cluster
    clusters = perform_clustering(
        integrated,
        method="hdbscan",
        min_cluster_size=20
    )

    # 5. Validate
    validation = validate_clusters(
        integrated,
        clusters['labels'],
        n_bootstrap=100
    )

    # 6. Visualize
    plot_results(
        integrated,
        clusters,
        validation,
        output_dir="outputs/figures"
    )

    print(f"Found {clusters['n_clusters']} clusters")
    print(f"Silhouette score: {validation['silhouette']:.3f}")

Command-Line Interface
----------------------

The CLI provides convenient access to all features:

**Full pipeline:**

.. code-block:: bash

    audhd-pipeline run --config config.yaml

**Individual stages:**

.. code-block:: bash

    # Load and preprocess only
    audhd-pipeline preprocess --config config.yaml

    # Clustering only (requires preprocessed data)
    audhd-pipeline cluster --input outputs/integrated.h5

    # Validation only
    audhd-pipeline validate --input outputs/clusters.h5

**Utilities:**

.. code-block:: bash

    # Check data integrity
    audhd-pipeline check-data --input data/

    # Generate report from existing results
    audhd-pipeline report --input outputs/results.h5

Configuration Options
---------------------

Key configuration options:

.. code-block:: yaml

    # Minimal configuration
    data:
      input_dir: "data/"
      output_dir: "outputs/"

    processing:
      modalities: ["genomic", "clinical", "metabolomic"]
      impute_method: "knn"
      scale_method: "standard"

    integration:
      method: "mofa"
      n_factors: 15

    clustering:
      method: "hdbscan"
      min_cluster_size: 20
      embedding_method: "umap"

    validation:
      n_bootstrap: 100
      compute_stability: true

See :doc:`configuration` for all options.

Output Files
------------

After running the pipeline, you'll find:

.. code-block:: text

    outputs/
    ├── preprocessed/
    │   ├── genomic_preprocessed.h5
    │   ├── clinical_preprocessed.csv
    │   └── metabolomic_preprocessed.h5
    ├── integrated/
    │   ├── factors.csv
    │   └── weights.csv
    ├── clusters/
    │   ├── labels.csv
    │   ├── embedding.csv
    │   └── cluster_stats.csv
    ├── validation/
    │   ├── metrics.json
    │   └── stability_results.csv
    ├── biological/
    │   ├── pathway_enrichment.csv
    │   └── cluster_signatures.csv
    ├── figures/
    │   ├── embedding_plot.png
    │   ├── heatmaps.png
    │   └── validation_plots.png
    └── report.html

Viewing Results
---------------

Open the HTML report:

.. code-block:: bash

    # macOS
    open outputs/report.html

    # Linux
    xdg-open outputs/report.html

    # Windows
    start outputs/report.html

Or use the interactive dashboard:

.. code-block:: bash

    audhd-dashboard --results outputs/results.h5

This opens an interactive Dash app at http://localhost:8050.

Next Steps
----------

* :doc:`user_guide/data_loading` - Learn about data format requirements
* :doc:`tutorials/complete_workflow` - Detailed tutorial with example data
* :doc:`configuration` - Complete configuration reference
* :doc:`api/data` - API documentation