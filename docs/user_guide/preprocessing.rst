Preprocessing Guide
===================

This guide covers preprocessing and normalization of multi-omics data.

Overview
--------

Preprocessing transforms raw data into a format suitable for integration and analysis:

1. **Missing value imputation** - Fill in missing measurements
2. **Normalization** - Scale features to comparable ranges
3. **Feature selection** - Remove low-quality or uninformative features
4. **Batch correction** - Remove technical variation

Preprocessing Pipeline
----------------------

The standard preprocessing workflow:

.. code-block:: python

    from audhd_correlation.preprocess import preprocess_pipeline

    preprocessed = preprocess_pipeline(
        data,
        impute_method='knn',
        scale_method='standard',
        feature_selection=True,
        batch_correct=True
    )

This applies all preprocessing steps in the correct order.

Missing Value Imputation
-------------------------

Handle missing values before downstream analysis.

Available Methods
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.preprocess import impute_missing

    # K-Nearest Neighbors (KNN) - recommended for metabolomics
    imputed = impute_missing(data, method='knn', n_neighbors=5)

    # Mean imputation - simple, fast
    imputed = impute_missing(data, method='mean')

    # Median imputation - robust to outliers
    imputed = impute_missing(data, method='median')

    # Half-minimum - for left-censored data (e.g., below detection limit)
    imputed = impute_missing(data, method='half-min')

    # Iterative imputation - uses all features to predict missing
    imputed = impute_missing(data, method='iterative', max_iter=10)

    # Random Forest - most accurate, slowest
    imputed = impute_missing(data, method='rf', n_estimators=100)

Method Selection
~~~~~~~~~~~~~~~~

Choose imputation method based on data characteristics:

**Metabolomics:** Use 'knn' or 'half-min'

.. code-block:: python

    metabolomic_imputed = impute_missing(
        metabolomic,
        method='knn',
        n_neighbors=5,
        weights='distance'  # Weight by distance
    )

**Clinical:** Use 'iterative' or 'mean'

.. code-block:: python

    clinical_imputed = impute_missing(
        clinical,
        method='iterative',
        max_iter=10,
        random_state=42
    )

**Microbiome:** Use 'zero' (absences are meaningful)

.. code-block:: python

    microbiome_imputed = impute_missing(
        microbiome,
        method='zero'
    )

Validation
~~~~~~~~~~

Validate imputation quality:

.. code-block:: python

    from audhd_correlation.preprocess import validate_imputation

    # Hold out 10% of values to test imputation
    validation = validate_imputation(
        data,
        impute_method='knn',
        test_fraction=0.1,
        n_iterations=5
    )

    print(f"Mean Absolute Error: {validation['mae']:.3f}")
    print(f"R² Score: {validation['r2']:.3f}")

Normalization and Scaling
--------------------------

Scale features to comparable ranges.

Scaling Methods
~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.preprocess import scale_features

    # Standard scaling (z-score) - recommended for most cases
    scaled = scale_features(data, method='standard')
    # Result: mean=0, std=1

    # Min-max scaling (0-1 range)
    scaled = scale_features(data, method='minmax')
    # Result: min=0, max=1

    # Robust scaling - resistant to outliers
    scaled = scale_features(data, method='robust')
    # Uses median and IQR

    # Quantile normalization - forces same distribution
    scaled = scale_features(data, method='quantile')

Method Selection
~~~~~~~~~~~~~~~~

**Genomics:** Use 'standard'

.. code-block:: python

    genomic_scaled = scale_features(genomic, method='standard')

**Clinical:** Use 'robust' (resistant to outliers)

.. code-block:: python

    clinical_scaled = scale_features(clinical, method='robust')

**Metabolomics:** Log-transform first, then 'standard'

.. code-block:: python

    import numpy as np

    # Log2 transform
    metabolomic_log = np.log2(metabolomic + 1)

    # Then scale
    metabolomic_scaled = scale_features(metabolomic_log, method='standard')

**Microbiome:** Use 'CLR' (centered log-ratio)

.. code-block:: python

    from audhd_correlation.preprocess import clr_transform

    microbiome_clr = clr_transform(microbiome)

Feature-Specific Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~

Scale each modality separately:

.. code-block:: python

    from audhd_correlation.preprocess import scale_features

    processed_data = {}

    # Genomic: standard scaling
    processed_data['genomic'] = scale_features(
        data['genomic'],
        method='standard'
    )

    # Clinical: robust scaling
    processed_data['clinical'] = scale_features(
        data['clinical'],
        method='robust'
    )

    # Metabolomic: log + standard
    metabolomic_log = np.log2(data['metabolomic'] + 1)
    processed_data['metabolomic'] = scale_features(
        metabolomic_log,
        method='standard'
    )

    # Microbiome: CLR transform
    processed_data['microbiome'] = clr_transform(data['microbiome'])

Feature Selection
-----------------

Remove uninformative or low-quality features.

Variance-Based Selection
~~~~~~~~~~~~~~~~~~~~~~~~

Remove low-variance features:

.. code-block:: python

    from audhd_correlation.preprocess import select_features_variance

    # Remove features with variance < threshold
    selected = select_features_variance(
        data,
        threshold=0.01,  # Keep features with variance > 0.01
        percentile=None  # Or use percentile (e.g., 10 keeps top 90%)
    )

    print(f"Selected {selected.shape[1]} of {data.shape[1]} features")

Correlation-Based Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove highly correlated features:

.. code-block:: python

    from audhd_correlation.preprocess import remove_correlated_features

    # Remove one of each highly correlated pair
    selected = remove_correlated_features(
        data,
        threshold=0.95,  # Correlation threshold
        method='pearson' # 'pearson' or 'spearman'
    )

Statistical Selection
~~~~~~~~~~~~~~~~~~~~~

Select features associated with outcome:

.. code-block:: python

    from audhd_correlation.preprocess import select_features_statistical

    # For continuous outcome
    selected = select_features_statistical(
        data,
        y=clinical['severity_score'],
        method='correlation',
        threshold=0.3,  # Absolute correlation > 0.3
        fdr_correction=True
    )

    # For categorical outcome
    selected = select_features_statistical(
        data,
        y=clinical['diagnosis'],
        method='anova',
        alpha=0.05,  # Significance level
        fdr_correction=True
    )

Combined Selection
~~~~~~~~~~~~~~~~~~

Apply multiple criteria:

.. code-block:: python

    from audhd_correlation.preprocess import select_features

    selected = select_features(
        data,
        variance_threshold=0.01,      # Remove low variance
        correlation_threshold=0.95,   # Remove high correlation
        statistical_test=True,        # Statistical association
        outcome=clinical['diagnosis'],
        max_features=1000,            # Keep top 1000
    )

Batch Effect Correction
-----------------------

Remove technical variation while preserving biological signal.

ComBat Method
~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.preprocess import correct_batch_combat

    corrected = correct_batch_combat(
        data,
        batch=clinical['site'],              # Batch variable
        covariates=clinical[['age', 'sex']], # Preserve these
        parametric=True                      # Parametric adjustment
    )

Harmony Method
~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.preprocess import correct_batch_harmony

    corrected = correct_batch_harmony(
        data,
        batch=clinical['site'],
        covariates=clinical[['age', 'sex']],
        theta=2.0  # Diversity clustering penalty
    )

Limma Method
~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.preprocess import correct_batch_limma

    corrected = correct_batch_limma(
        data,
        batch=clinical['site'],
        design_matrix=clinical[['age', 'sex', 'diagnosis']]
    )

Validation
~~~~~~~~~~

Validate batch correction:

.. code-block:: python

    from audhd_correlation.preprocess import validate_batch_correction

    validation = validate_batch_correction(
        original=data,
        corrected=corrected,
        batch=clinical['site'],
        biological_signal=clinical['diagnosis']
    )

    print(f"Batch effect reduction: {validation['batch_variance_reduction']:.1%}")
    print(f"Signal preservation: {validation['signal_preservation']:.1%}")

Quality Control
---------------

Assess preprocessing quality.

QC Metrics
~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.preprocess import compute_qc_metrics

    qc = compute_qc_metrics(
        original=data_original,
        preprocessed=data_preprocessed
    )

    print(f"Missing values: {qc['missing_rate']:.1%}")
    print(f"Outlier samples: {qc['n_outlier_samples']}")
    print(f"Outlier features: {qc['n_outlier_features']}")
    print(f"Mean correlation: {qc['mean_correlation']:.3f}")

Visualization
~~~~~~~~~~~~~

Visualize preprocessing effects:

.. code-block:: python

    from audhd_correlation.preprocess.viz import plot_preprocessing_effects

    plot_preprocessing_effects(
        original=data_original,
        preprocessed=data_preprocessed,
        output_dir='figures/preprocessing/'
    )

    # Creates:
    # - distribution_comparison.png
    # - pca_comparison.png
    # - correlation_comparison.png
    # - missing_pattern.png

Complete Preprocessing Example
-------------------------------

Here's a complete preprocessing workflow:

.. code-block:: python

    from audhd_correlation.preprocess import (
        impute_missing,
        scale_features,
        select_features,
        correct_batch_combat,
        compute_qc_metrics,
    )
    import numpy as np

    def preprocess_modality(data, modality_type, clinical):
        """Preprocess a single modality"""

        # 1. Handle missing values
        if modality_type == 'metabolomic':
            imputed = impute_missing(data, method='knn', n_neighbors=5)
        elif modality_type == 'clinical':
            imputed = impute_missing(data, method='iterative')
        else:
            imputed = impute_missing(data, method='mean')

        # 2. Transform if needed
        if modality_type == 'metabolomic':
            transformed = np.log2(imputed + 1)
        elif modality_type == 'microbiome':
            from audhd_correlation.preprocess import clr_transform
            transformed = clr_transform(imputed)
        else:
            transformed = imputed

        # 3. Scale
        if modality_type == 'clinical':
            scaled = scale_features(transformed, method='robust')
        else:
            scaled = scale_features(transformed, method='standard')

        # 4. Feature selection
        selected = select_features(
            scaled,
            variance_threshold=0.01,
            correlation_threshold=0.95,
            max_features=1000 if modality_type == 'genomic' else None
        )

        # 5. Batch correction
        if 'site' in clinical.columns:
            corrected = correct_batch_combat(
                selected,
                batch=clinical['site'],
                covariates=clinical[['age', 'sex']]
            )
        else:
            corrected = selected

        return corrected

    # Process each modality
    preprocessed = {}
    for modality in ['genomic', 'clinical', 'metabolomic', 'microbiome']:
        if modality in data:
            print(f"Preprocessing {modality}...")
            preprocessed[modality] = preprocess_modality(
                data[modality],
                modality,
                clinical_data
            )

            # QC
            qc = compute_qc_metrics(data[modality], preprocessed[modality])
            print(f"  Missing: {qc['missing_rate']:.1%}")
            print(f"  Features: {data[modality].shape[1]} → {preprocessed[modality].shape[1]}")

    # Save preprocessed data
    for modality, df in preprocessed.items():
        df.to_hdf(f'data/preprocessed/{modality}.h5', key='data')

    print("Preprocessing complete!")

Best Practices
--------------

1. **Order matters**: Impute → Transform → Scale → Select → Batch correct
2. **Modality-specific**: Use appropriate methods for each data type
3. **Validate**: Check preprocessing effects on data distribution
4. **Document**: Record all preprocessing parameters
5. **Reproducibility**: Use fixed random seeds

Common Pitfalls
---------------

❌ **Scaling before batch correction**
   Batch correction works better on non-scaled data

❌ **Feature selection before imputation**
   Missing values affect variance calculations

❌ **Forgetting log-transform for metabolomics**
   Metabolite abundances are log-normally distributed

❌ **Over-aggressive feature selection**
   May remove biologically relevant but low-variance features

❌ **Not preserving biological covariates in batch correction**
   Can remove signal of interest

Next Steps
----------

* :doc:`integration` - Integrate preprocessed multi-omics data
* :doc:`clustering` - Perform clustering analysis
* :doc:`../troubleshooting` - Troubleshooting preprocessing issues