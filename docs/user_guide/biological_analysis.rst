Biological Analysis Guide
=========================

Interpret clusters through pathway enrichment and biological context.

Pathway Enrichment Analysis
----------------------------

Gene Set Enrichment
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.biological import pathway_enrichment

    # For each cluster
    enrichment = pathway_enrichment(
        features=cluster_signatures,
        modality='genomic',
        databases=['KEGG', 'GO_Biological_Process', 'Reactome'],
        fdr_threshold=0.05
    )

    # Top pathways
    for pathway in enrichment['significant_pathways'][:10]:
        print(f"{pathway['name']}: p={pathway['pvalue']:.2e}")

Metabolic Pathway Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from audhd_correlation.biological import metabolite_pathway_enrichment

    met_enrichment = metabolite_pathway_enrichment(
        metabolites=cluster_metabolites,
        databases=['HMDB', 'KEGG'],
        method='ora'  # Over-representation analysis
    )

Cluster Signatures
------------------

.. code-block:: python

    from audhd_correlation.biological import compute_cluster_signatures

    signatures = compute_cluster_signatures(
        data=preprocessed_data,
        labels=labels,
        modalities=['genomic', 'metabolomic'],
        method='limma',
        fdr_threshold=0.05
    )

    # Significant features per cluster
    for cluster_id, sig_features in signatures.items():
        print(f"Cluster {cluster_id}: {len(sig_features)} signatures")

Network Analysis
----------------

.. code-block:: python

    from audhd_correlation.biological import build_biological_network

    network = build_biological_network(
        signatures,
        databases=['STRING', 'BioGRID'],
        confidence_threshold=0.7
    )

    # Identify modules
    from audhd_correlation.biological import detect_network_modules

    modules = detect_network_modules(network, method='louvain')

Next Steps
----------

* :doc:`visualization` - Visualize results