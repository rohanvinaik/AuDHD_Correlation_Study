AuDHD Correlation Study Documentation
======================================

Welcome to the AuDHD Correlation Study documentation. This pipeline provides comprehensive multi-omics analysis for identifying ASD-ADHD correlation patterns.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   configuration

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   user_guide/data_loading
   user_guide/preprocessing
   user_guide/integration
   user_guide/clustering
   user_guide/validation
   user_guide/biological_analysis
   user_guide/visualization

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/complete_workflow
   tutorials/data_preparation
   tutorials/custom_analysis
   tutorials/advanced_techniques

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/data
   api/preprocess
   api/integrate
   api/modeling
   api/validation
   api/biological
   api/visualization
   api/reporting

.. toctree::
   :maxdepth: 2
   :caption: Data Reference

   data_dictionaries/genomic
   data_dictionaries/clinical
   data_dictionaries/metabolomic
   data_dictionaries/microbiome

.. toctree::
   :maxdepth: 2
   :caption: Help

   troubleshooting
   faq
   best_practices
   performance_optimization

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   changelog
   contributing
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
========

The AuDHD Correlation Study pipeline is designed to:

* Load and harmonize multi-omics data (genomic, clinical, metabolomic, microbiome)
* Preprocess and integrate data across modalities
* Identify patient subtypes through unsupervised clustering
* Validate clusters using statistical methods
* Perform biological interpretation and pathway enrichment
* Generate comprehensive reports and visualizations

Key Features
------------

* **Multi-omics Integration**: Supports genomic, clinical, metabolomic, and microbiome data
* **Flexible Preprocessing**: Multiple imputation and normalization methods
* **Advanced Clustering**: HDBSCAN, K-means, hierarchical clustering with UMAP embedding
* **Statistical Validation**: Silhouette analysis, bootstrap stability, permutation tests
* **Biological Context**: Pathway enrichment, gene set analysis, network analysis
* **Reproducibility**: Comprehensive logging, checkpointing, and configuration management
* **Production-Ready**: Docker support, batch processing, parallel execution

Quick Links
-----------

* :doc:`installation` - Get started installing the pipeline
* :doc:`quickstart` - Run your first analysis in 5 minutes
* :doc:`tutorials/complete_workflow` - Complete walkthrough with example data
* :doc:`api/data` - API reference for data loading
* :doc:`troubleshooting` - Solutions to common problems
* :doc:`faq` - Frequently asked questions

Citation
--------

If you use this pipeline in your research, please cite::

    AuDHD Correlation Study Pipeline (2024)
    https://github.com/your-repo/AuDHD_Correlation_Study

Support
-------

* **Issues**: Report bugs at https://github.com/your-repo/issues
* **Discussions**: Ask questions at https://github.com/your-repo/discussions
* **Email**: Contact the team at support@example.com