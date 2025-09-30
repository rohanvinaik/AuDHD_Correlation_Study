Visualization Guide
===================

Create publication-ready figures and interactive visualizations.

Cluster Visualizations
-----------------------

UMAP/t-SNE Plots
~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.viz import plot_embedding

    plot_embedding(
        embedding,
        labels,
        color_by=clinical_data['diagnosis'],
        output_path='figures/clusters.png',
        figsize=(10, 8),
        point_size=50,
        alpha=0.7,
        title='Patient Clusters'
    )

Heatmaps
~~~~~~~~

.. code-block:: python

    from audhd_correlation.viz import plot_heatmap

    plot_heatmap(
        data=integrated_data,
        labels=labels,
        cluster_samples=True,
        cluster_features=True,
        output_path='figures/heatmap.png',
        cmap='RdBu_r'
    )

Validation Plots
----------------

.. code-block:: python

    from audhd_correlation.viz import plot_validation_summary

    plot_validation_summary(
        validation_results,
        output_path='figures/validation.png'
    )

Interactive Dashboards
----------------------

.. code-block:: python

    from audhd_correlation.viz import create_interactive_dashboard

    create_interactive_dashboard(
        results,
        output_path='outputs/dashboard.html'
    )

Next Steps
----------

* :doc:`../troubleshooting` - Common visualization issues