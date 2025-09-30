Validation Guide
================

Validate clustering quality and stability.

Internal Validation Metrics
----------------------------

Silhouette Score
~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.validation import compute_silhouette

    score = compute_silhouette(integrated_data, labels)
    # Range: [-1, 1], higher is better
    # > 0.5: Strong clusters
    # 0.25-0.5: Reasonable clusters
    # < 0.25: Weak clusters

Calinski-Harabasz Index
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.validation import compute_calinski_harabasz

    score = compute_calinski_harabasz(integrated_data, labels)
    # Higher is better, no fixed range

Davies-Bouldin Index
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.validation import compute_davies_bouldin

    score = compute_davies_bouldin(integrated_data, labels)
    # Lower is better, minimum is 0

All Metrics
~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.validation import compute_internal_metrics

    metrics = compute_internal_metrics(integrated_data, labels)

    print(f"Silhouette: {metrics['silhouette']:.3f}")
    print(f"Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}")
    print(f"Davies-Bouldin: {metrics['davies_bouldin']:.3f}")

Stability Analysis
------------------

Bootstrap Stability
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.validation import bootstrap_stability

    stability = bootstrap_stability(
        integrated_data,
        labels,
        n_bootstrap=100,
        sample_fraction=0.8
    )

    print(f"Mean ARI: {stability['mean_ari']:.3f}")
    print(f"Std ARI: {stability['std_ari']:.3f}")
    # ARI > 0.8: Highly stable
    # ARI 0.6-0.8: Moderately stable
    # ARI < 0.6: Unstable

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.validation import cross_validate_clustering

    cv_result = cross_validate_clustering(
        integrated_data,
        method='hdbscan',
        n_folds=5,
        min_cluster_size=20
    )

    print(f"Mean stability: {cv_result['mean_stability']:.3f}")

Statistical Significance
-------------------------

Permutation Test
~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.validation import permutation_test

    p_value = permutation_test(
        integrated_data,
        labels,
        metric='silhouette',
        n_permutations=1000
    )

    print(f"p-value: {p_value:.4f}")
    # p < 0.05: Clusters are significantly better than random

Gap Statistic
~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.validation import compute_gap_statistic

    gap_stat = compute_gap_statistic(
        integrated_data,
        labels,
        n_references=50
    )

    print(f"Gap: {gap_stat['gap']:.3f}")
    # Positive gap: Clustering structure exists

Complete Validation
-------------------

.. code-block:: python

    from audhd_correlation.validation import validate_clusters

    validation = validate_clusters(
        integrated_data,
        labels,
        n_bootstrap=100,
        n_permutations=1000
    )

    # Print report
    print("=== Clustering Validation Report ===")
    print(f"Silhouette: {validation['silhouette']:.3f}")
    print(f"Stability (ARI): {validation['stability_ari']:.3f}")
    print(f"Permutation p-value: {validation['permutation_pvalue']:.4f}")

    # Save report
    import json
    with open('outputs/validation.json', 'w') as f:
        json.dump(validation, f, indent=2)

Visualization
-------------

.. code-block:: python

    from audhd_correlation.validation.viz import (
        plot_silhouette_analysis,
        plot_stability_results
    )

    plot_silhouette_analysis(
        integrated_data,
        labels,
        output_path='figures/silhouette.png'
    )

    plot_stability_results(
        stability,
        output_path='figures/stability.png'
    )

Next Steps
----------

* :doc:`biological_analysis` - Interpret clusters
* :doc:`visualization` - Create publication-ready figures