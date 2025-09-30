Multi-Omics Integration Guide
==============================

This guide covers integrating multiple data modalities into a unified representation.

Overview
--------

Integration combines information from multiple omics layers:

* **Goal**: Create low-dimensional representation capturing shared variation
* **Input**: Preprocessed data from multiple modalities
* **Output**: Integrated factors/components for downstream analysis

Integration Methods
-------------------

MOFA (Multi-Omics Factor Analysis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best for**: Identifying shared and modality-specific variation

.. code-block:: python

    from audhd_correlation.integrate import integrate_mofa

    integrated = integrate_mofa(
        data={'genomic': genomic, 'metabolomic': metabolomic},
        n_factors=15,              # Number of latent factors
        convergence_mode='fast',   # 'fast' or 'slow'
        sparsity=0.5,             # Sparsity penalty (0-1)
        seed=42
    )

    # Access results
    factors = integrated['factors']        # Sample × factors
    weights = integrated['weights']        # Features × factors per modality
    variance = integrated['variance']      # Variance explained per modality

PCA-Based Integration
~~~~~~~~~~~~~~~~~~~~~

**Best for**: Quick integration, large datasets

.. code-block:: python

    from audhd_correlation.integrate import integrate_pca

    integrated = integrate_pca(
        data={'genomic': genomic, 'metabolomic': metabolomic},
        n_components=20,
        concatenate_first=True,  # Concatenate then PCA
        normalize=True
    )

    # Result: Sample × components DataFrame

Canonical Correlation Analysis (CCA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best for**: Finding correlations between two modalities

.. code-block:: python

    from audhd_correlation.integrate import integrate_cca

    integrated = integrate_cca(
        data1=genomic,
        data2=metabolomic,
        n_components=10,
        regularization=0.1
    )

Non-Negative Matrix Factorization (NMF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best for**: Count data, interpretable parts-based decomposition

.. code-block:: python

    from audhd_correlation.integrate import integrate_nmf

    integrated = integrate_nmf(
        data={'metabolomic': metabolomic, 'microbiome': microbiome},
        n_components=15,
        init='nndsvda',
        max_iter=500
    )

Method Selection Guide
----------------------

Choose integration method based on your goals:

+------------------+------------------+--------------------+-------------------+
| Method           | Use Case         | Pros               | Cons              |
+==================+==================+====================+===================+
| MOFA             | General purpose  | Shared/specific    | Slow for large    |
|                  |                  | variation, sparse  | datasets          |
+------------------+------------------+--------------------+-------------------+
| PCA              | Quick analysis   | Fast, simple       | No sparsity       |
+------------------+------------------+--------------------+-------------------+
| CCA              | Two modalities   | Correlation focus  | Only 2 modalities |
+------------------+------------------+--------------------+-------------------+
| NMF              | Count data       | Interpretable      | Non-negative only |
+------------------+------------------+--------------------+-------------------+

Complete Integration Workflow
------------------------------

.. code-block:: python

    from audhd_correlation.integrate import (
        integrate_multiomics,
        select_n_factors,
        interpret_factors
    )

    # 1. Select number of factors
    n_factors = select_n_factors(
        data,
        method='mofa',
        max_factors=30,
        criteria='elbow'  # 'elbow', 'variance_threshold', or 'cross_validation'
    )

    print(f"Selected {n_factors} factors")

    # 2. Run integration
    result = integrate_multiomics(
        data,
        method='mofa',
        n_factors=n_factors
    )

    # 3. Interpret factors
    interpretation = interpret_factors(
        result,
        feature_names=data,
        top_n=20  # Top features per factor
    )

    # 4. Save results
    result['factors'].to_csv('outputs/integrated_factors.csv')

Validation
----------

Assess integration quality:

.. code-block:: python

    from audhd_correlation.integrate import validate_integration

    validation = validate_integration(
        original_data=data,
        integrated=result,
        method='mofa'
    )

    print(f"Variance explained: {validation['variance_explained']:.1%}")
    print(f"Reconstruction error: {validation['reconstruction_error']:.3f}")
    print(f"Cross-modality correlation: {validation['cross_correlation']:.3f}")

Visualization
-------------

.. code-block:: python

    from audhd_correlation.integrate.viz import (
        plot_variance_explained,
        plot_factor_correlation,
        plot_feature_weights
    )

    # Variance explained per modality
    plot_variance_explained(result, output_path='figures/variance.png')

    # Factor correlations with clinical variables
    plot_factor_correlation(
        result['factors'],
        clinical_data,
        output_path='figures/factor_clinical.png'
    )

    # Top features per factor
    plot_feature_weights(
        result['weights'],
        top_n=20,
        output_path='figures/weights.png'
    )

Next Steps
----------

* :doc:`clustering` - Cluster integrated data
* :doc:`validation` - Validate results