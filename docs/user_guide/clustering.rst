Clustering Guide
================

Identify patient subtypes through unsupervised clustering.

Overview
--------

Clustering groups samples with similar multi-omics profiles:

* **Input**: Integrated factors from multiple modalities
* **Output**: Cluster labels and low-dimensional embeddings
* **Goal**: Discover biologically meaningful patient subtypes

Clustering Methods
------------------

HDBSCAN (Recommended)
~~~~~~~~~~~~~~~~~~~~~

Density-based clustering that automatically determines number of clusters:

.. code-block:: python

    from audhd_correlation.modeling import perform_clustering

    result = perform_clustering(
        integrated_data,
        method='hdbscan',
        min_cluster_size=20,        # Min samples per cluster
        min_samples=5,               # Core point threshold
        cluster_selection_epsilon=0.5,
        metric='euclidean'
    )

    labels = result['labels']      # -1 indicates noise points
    n_clusters = result['n_clusters']
    probabilities = result['probabilities']  # Cluster membership

K-means
~~~~~~~

Partition-based clustering with fixed number of clusters:

.. code-block:: python

    result = perform_clustering(
        integrated_data,
        method='kmeans',
        n_clusters=3,
        n_init=100,          # Run 100 times with different initializations
        max_iter=500,
        random_state=42
    )

Hierarchical Clustering
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    result = perform_clustering(
        integrated_data,
        method='hierarchical',
        n_clusters=3,
        linkage='ward',      # 'ward', 'complete', 'average', 'single'
        distance_metric='euclidean'
    )

Gaussian Mixture Models
~~~~~~~~~~~~~~~~~~~~~~~

Probabilistic clustering:

.. code-block:: python

    result = perform_clustering(
        integrated_data,
        method='gmm',
        n_components=3,
        covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
        n_init=10
    )

Determining Number of Clusters
-------------------------------

Elbow Method
~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.modeling import select_n_clusters_elbow

    n_clusters = select_n_clusters_elbow(
        integrated_data,
        method='kmeans',
        max_clusters=10
    )

Silhouette Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.modeling import select_n_clusters_silhouette

    n_clusters, scores = select_n_clusters_silhouette(
        integrated_data,
        method='kmeans',
        min_clusters=2,
        max_clusters=10
    )

    print(f"Optimal clusters: {n_clusters}")
    print(f"Silhouette score: {scores[n_clusters]:.3f}")

Consensus Clustering
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.modeling import consensus_clustering

    consensus_result = consensus_clustering(
        integrated_data,
        method='kmeans',
        max_clusters=10,
        n_iterations=100
    )

    # Plot consensus matrix
    from audhd_correlation.modeling.viz import plot_consensus_matrix
    plot_consensus_matrix(consensus_result, output_path='figures/consensus.png')

Dimensionality Reduction
-------------------------

Create 2D embeddings for visualization:

UMAP (Recommended)
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.modeling import create_embedding

    embedding = create_embedding(
        integrated_data,
        method='umap',
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean'
    )

t-SNE
~~~~~

.. code-block:: python

    embedding = create_embedding(
        integrated_data,
        method='tsne',
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000
    )

Complete Clustering Workflow
-----------------------------

.. code-block:: python

    from audhd_correlation.modeling import (
        select_n_clusters_silhouette,
        perform_clustering,
        create_embedding,
        compute_cluster_statistics
    )

    # 1. Determine number of clusters
    n_clusters = select_n_clusters_silhouette(
        integrated_data,
        method='kmeans',
        max_clusters=10
    )

    print(f"Selected {n_clusters} clusters")

    # 2. Perform clustering
    cluster_result = perform_clustering(
        integrated_data,
        method='hdbscan',
        min_cluster_size=20
    )

    labels = cluster_result['labels']

    # 3. Create embedding
    embedding = create_embedding(
        integrated_data,
        method='umap',
        n_neighbors=15
    )

    # 4. Compute statistics
    stats = compute_cluster_statistics(
        integrated_data,
        labels,
        clinical_data
    )

    print(f"Cluster sizes: {stats['sizes']}")
    print(f"Cluster centers shape: {stats['centers'].shape}")

    # 5. Save results
    import pandas as pd
    pd.DataFrame({
        'sample_id': integrated_data.index,
        'cluster': labels,
        'umap1': embedding[:, 0],
        'umap2': embedding[:, 1]
    }).to_csv('outputs/clusters.csv', index=False)

Cluster Characterization
-------------------------

Identify defining features of each cluster:

.. code-block:: python

    from audhd_correlation.modeling import characterize_clusters

    characterization = characterize_clusters(
        data=integrated_data,
        labels=labels,
        clinical_data=clinical_data,
        original_features=preprocessed_data,
        top_n=20
    )

    # Differentially abundant features
    diff_features = characterization['differential_features']

    # Clinical associations
    clinical_assoc = characterization['clinical_associations']

    # Cluster profiles
    for cluster_id, profile in characterization['profiles'].items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {profile['size']}")
        print(f"  Mean age: {profile['mean_age']:.1f}")
        print(f"  Diagnosis distribution: {profile['diagnosis_dist']}")

Visualization
-------------

.. code-block:: python

    from audhd_correlation.modeling.viz import (
        plot_clusters,
        plot_cluster_heatmap,
        plot_cluster_boxplots
    )

    # Scatter plot
    plot_clusters(
        embedding,
        labels,
        output_path='figures/clusters_umap.png',
        color_by=clinical_data['diagnosis']
    )

    # Heatmap of cluster profiles
    plot_cluster_heatmap(
        integrated_data,
        labels,
        top_features=50,
        output_path='figures/cluster_heatmap.png'
    )

    # Clinical variable distributions
    plot_cluster_boxplots(
        labels,
        clinical_data[['age', 'severity_score', 'iq']],
        output_path='figures/cluster_clinical.png'
    )

Best Practices
--------------

1. **Scale data**: Ensure features are on comparable scales
2. **Try multiple methods**: HDBSCAN, k-means, hierarchical
3. **Validate stability**: Use bootstrap or cross-validation
4. **Clinical relevance**: Check associations with phenotypes
5. **Biological interpretation**: Perform pathway enrichment

Common Pitfalls
---------------

❌ **Too few samples per cluster**: Use min_cluster_size ≥ 20
❌ **Not checking stability**: Clusters should be reproducible
❌ **Ignoring noise points**: HDBSCAN's -1 labels are meaningful
❌ **Over-interpreting**: Small clusters may be artifacts

Next Steps
----------

* :doc:`validation` - Validate cluster quality
* :doc:`biological_analysis` - Interpret clusters biologically