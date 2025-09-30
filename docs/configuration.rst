Configuration Reference
=======================

Complete reference for pipeline configuration.

Configuration File
------------------

The pipeline uses YAML format for configuration.

Basic Structure
~~~~~~~~~~~~~~~

.. code-block:: yaml

    # Data paths
    data:
      input_dir: "data/"
      output_dir: "outputs/"

    # Processing options
    processing:
      modalities: ["genomic", "clinical", "metabolomic"]
      n_jobs: -1

    # Integration method
    integration:
      method: "mofa"
      n_factors: 15

    # Clustering
    clustering:
      method: "hdbscan"
      min_cluster_size: 20

Complete Reference
------------------

Data Section
~~~~~~~~~~~~

.. code-block:: yaml

    data:
      input_dir: "data/"              # Input data directory
      output_dir: "outputs/"          # Output directory
      genomic_path: null              # Override genomic file path
      clinical_path: null             # Override clinical file path
      metabolomic_path: null          # Override metabolomic file path
      microbiome_path: null           # Override microbiome file path
      cache_dir: ".cache/"            # Cache directory

Processing Section
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    processing:
      modalities:                     # Modalities to include
        - genomic
        - clinical
        - metabolomic
        - microbiome

      batch_size: 1000                # Batch size for processing
      n_jobs: -1                      # Number of parallel jobs (-1 = all cores)
      enable_checkpointing: true      # Enable checkpointing
      checkpoint_interval: 5          # Checkpoint every N steps

      # Imputation
      impute_method: "knn"            # 'knn', 'mean', 'median', 'iterative'
      knn_neighbors: 5                # K for KNN imputation

      # Scaling
      scale_method: "standard"        # 'standard', 'robust', 'minmax'

      # Feature selection
      variance_threshold: 0.01        # Remove features with low variance
      correlation_threshold: 0.95     # Remove highly correlated features
      max_features: null              # Maximum features to keep (null = all)

      # Batch correction
      batch_correct: true             # Enable batch correction
      batch_column: "site"            # Column containing batch variable
      batch_method: "combat"          # 'combat', 'limma', 'harmony'

Integration Section
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    integration:
      method: "mofa"                  # 'mofa', 'pca', 'cca', 'nmf'
      n_factors: 15                   # Number of factors/components

      # MOFA-specific
      mofa_convergence: "fast"        # 'fast' or 'slow'
      mofa_sparsity: 0.5              # Sparsity penalty (0-1)

      # PCA-specific
      pca_whiten: false               # Whiten components

      # Validation
      validate_integration: true      # Compute validation metrics

Clustering Section
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    clustering:
      method: "hdbscan"               # 'hdbscan', 'kmeans', 'hierarchical', 'gmm'

      # HDBSCAN-specific
      min_cluster_size: 20            # Minimum samples per cluster
      min_samples: 5                  # Core point threshold
      cluster_selection_epsilon: 0.5  # Selection threshold
      metric: "euclidean"             # Distance metric

      # K-means-specific
      n_clusters: 3                   # Number of clusters (for kmeans)
      kmeans_n_init: 100              # Number of initializations
      kmeans_max_iter: 500            # Maximum iterations

      # Embedding
      embedding_method: "umap"        # 'umap', 'tsne', 'pca'
      umap_n_neighbors: 15            # UMAP neighbors
      umap_min_dist: 0.1              # UMAP minimum distance
      tsne_perplexity: 30             # t-SNE perplexity

Validation Section
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    validation:
      compute_internal: true          # Compute internal metrics
      compute_stability: true         # Compute bootstrap stability
      n_bootstrap: 100                # Bootstrap iterations
      bootstrap_fraction: 0.8         # Fraction of samples per bootstrap

      # Permutation test
      permutation_test: true          # Run permutation test
      n_permutations: 1000            # Permutation iterations

Biological Analysis Section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    biological:
      compute_signatures: true        # Compute cluster signatures
      signature_method: "limma"       # 'limma', 't-test', 'wilcoxon'
      fdr_threshold: 0.05             # FDR threshold for significance

      # Pathway enrichment
      pathway_enrichment: true        # Run pathway enrichment
      pathway_databases:              # Databases to use
        - "KEGG"
        - "GO_Biological_Process"
        - "Reactome"
      pathway_fdr: 0.05               # FDR threshold for pathways

      # Network analysis
      build_networks: false           # Build biological networks
      network_confidence: 0.7         # Confidence threshold

Visualization Section
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    visualization:
      create_plots: true              # Generate plots
      plot_format: "png"              # 'png', 'pdf', 'svg'
      plot_dpi: 300                   # Resolution (DPI)
      figsize: [10, 8]                # Figure size [width, height]

      # Colors
      color_palette: "tab10"          # Matplotlib colormap

      # Interactive
      create_dashboard: true          # Create interactive dashboard

Reporting Section
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    reporting:
      generate_report: true           # Generate HTML report
      report_template: "default"      # Report template
      include_qc: true                # Include QC metrics
      include_validation: true        # Include validation results
      include_biological: true        # Include biological interpretation

Logging Section
~~~~~~~~~~~~~~~

.. code-block:: yaml

    logging:
      level: "INFO"                   # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
      log_file: "logs/pipeline.log"  # Log file path
      console_output: true            # Log to console

Random Seeds
~~~~~~~~~~~~

.. code-block:: yaml

    random_state: 42                  # Master random seed

Example Configurations
----------------------

Minimal Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    data:
      input_dir: "data/"
      output_dir: "outputs/"

    processing:
      modalities: ["genomic", "clinical"]

Quick Analysis
~~~~~~~~~~~~~~

.. code-block:: yaml

    data:
      input_dir: "data/"
      output_dir: "outputs/"

    integration:
      method: "pca"                   # Fast
      n_factors: 10

    clustering:
      method: "kmeans"                # Fast
      n_clusters: 3

High-Quality Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    data:
      input_dir: "data/"
      output_dir: "outputs/"

    processing:
      impute_method: "iterative"      # Best quality
      enable_checkpointing: true

    integration:
      method: "mofa"                  # Best for multi-omics
      n_factors: 20
      mofa_convergence: "slow"        # Higher quality

    clustering:
      method: "hdbscan"               # Auto-determines k
      min_cluster_size: 30            # Larger, more stable clusters

    validation:
      n_bootstrap: 500                # More iterations
      n_permutations: 5000

    biological:
      pathway_enrichment: true
      build_networks: true

Command-Line Overrides
----------------------

Override config values from command line:

.. code-block:: bash

    audhd-pipeline run \\
      --config config.yaml \\
      --n-clusters 5 \\
      --output-dir custom_outputs/

Environment Variables
---------------------

Set via environment:

.. code-block:: bash

    export AUDHD_DATA_DIR=/data/
    export AUDHD_N_JOBS=16

    audhd-pipeline run --config config.yaml