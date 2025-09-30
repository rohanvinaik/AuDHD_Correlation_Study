Installation Guide
==================

This guide covers different installation methods for the AuDHD Correlation Study pipeline.

System Requirements
-------------------

**Minimum Requirements:**

* Python 3.9 or higher
* 8 GB RAM
* 10 GB free disk space

**Recommended:**

* Python 3.10+
* 16 GB RAM
* 50 GB free disk space (for large datasets)
* CUDA-capable GPU (optional, for acceleration)

Operating Systems
~~~~~~~~~~~~~~~~~

The pipeline is tested on:

* Ubuntu 20.04+
* macOS 11.0+
* Windows 10+ (via WSL2)

Installation Methods
--------------------

Method 1: pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install from PyPI (when published):

.. code-block:: bash

    pip install audhd-correlation-study

Or install from source:

.. code-block:: bash

    git clone https://github.com/your-repo/AuDHD_Correlation_Study.git
    cd AuDHD_Correlation_Study
    pip install -e .

Method 2: conda
~~~~~~~~~~~~~~~

Create a conda environment:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate audhd-study

Or manually:

.. code-block:: bash

    conda create -n audhd-study python=3.10
    conda activate audhd-study
    pip install -e .

Method 3: Docker
~~~~~~~~~~~~~~~~

Pull the pre-built image:

.. code-block:: bash

    docker pull your-repo/audhd-study:latest

Or build from source:

.. code-block:: bash

    git clone https://github.com/your-repo/AuDHD_Correlation_Study.git
    cd AuDHD_Correlation_Study
    docker build -t audhd-study .

Run the container:

.. code-block:: bash

    docker run -v $(pwd)/data:/data -v $(pwd)/outputs:/outputs audhd-study

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

The pipeline requires:

* **Data Processing**: pandas, numpy, scipy
* **Machine Learning**: scikit-learn, umap-learn, hdbscan
* **Statistical Analysis**: statsmodels, pingouin
* **Visualization**: matplotlib, seaborn, plotly
* **File I/O**: pyyaml, h5py

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For enhanced functionality:

* **MOFA**: mofapy2 (multi-omics factor analysis)
* **GPU Acceleration**: cupy, rapids
* **Pathway Analysis**: gseapy, enrichr
* **Network Analysis**: networkx, igraph

Install optional dependencies:

.. code-block:: bash

    # For MOFA integration
    pip install mofapy2

    # For GPU acceleration
    pip install cupy-cuda11x

    # For pathway analysis
    pip install gseapy

Development Installation
------------------------

For development and testing:

.. code-block:: bash

    git clone https://github.com/your-repo/AuDHD_Correlation_Study.git
    cd AuDHD_Correlation_Study

    # Install in editable mode with dev dependencies
    pip install -e ".[dev]"

    # Install pre-commit hooks
    pre-commit install

Run tests:

.. code-block:: bash

    pytest tests/

Verification
------------

Verify your installation:

.. code-block:: bash

    python -c "import audhd_correlation; print(audhd_correlation.__version__)"

Or run the test suite:

.. code-block:: bash

    audhd-test

Expected output::

    AuDHD Correlation Study v0.1.0
    All dependencies installed successfully.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'audhd_correlation'**

Solution: Ensure you installed the package:

.. code-block:: bash

    pip install -e .

**CUDA errors with GPU acceleration**

Solution: Ensure CUDA toolkit is installed and matches cupy version:

.. code-block:: bash

    # Check CUDA version
    nvcc --version

    # Install matching cupy
    pip install cupy-cuda11x  # Replace 11x with your CUDA version

**Memory errors during analysis**

Solution: Reduce batch size or enable checkpointing in config:

.. code-block:: yaml

    processing:
      batch_size: 100
      enable_checkpointing: true

For more troubleshooting, see :doc:`troubleshooting`.

Next Steps
----------

* :doc:`quickstart` - Run your first analysis
* :doc:`configuration` - Configure the pipeline for your data
* :doc:`tutorials/complete_workflow` - Complete walkthrough