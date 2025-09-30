# Documentation

This directory contains the Sphinx documentation for the AuDHD Correlation Study pipeline.

## Building Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

Output will be in `docs/_build/html/`. Open `index.html` in your browser:

```bash
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

### Build PDF Documentation

```bash
cd docs
make latexpdf
```

Requires LaTeX installation.

### Clean Build

```bash
cd docs
make clean
```

## Documentation Structure

```
docs/
├── index.rst                    # Main documentation index
├── installation.rst             # Installation guide
├── quickstart.rst               # Quick start tutorial
├── configuration.rst            # Configuration reference
├── user_guide/                  # User guides for each phase
│   ├── data_loading.rst
│   ├── preprocessing.rst
│   ├── integration.rst
│   ├── clustering.rst
│   ├── validation.rst
│   ├── biological_analysis.rst
│   └── visualization.rst
├── tutorials/                   # Tutorial documents
├── api/                         # API reference (auto-generated)
│   ├── data.rst
│   ├── preprocess.rst
│   ├── integrate.rst
│   ├── modeling.rst
│   ├── validation.rst
│   ├── biological.rst
│   ├── visualization.rst
│   └── reporting.rst
├── data_dictionaries/           # Data format specifications
│   ├── genomic.rst
│   ├── clinical.rst
│   ├── metabolomic.rst
│   └── microbiome.rst
├── video_scripts/               # Video tutorial scripts
│   ├── 01_quick_start.md
│   ├── 02_data_preparation.md
│   └── 03_interpreting_results.md
├── troubleshooting.rst          # Troubleshooting guide
├── faq.rst                      # Frequently asked questions
└── conf.py                      # Sphinx configuration
```

## Jupyter Notebooks

Tutorial notebooks are in `notebooks/`:

```
notebooks/
└── 01_complete_workflow.ipynb   # Complete analysis walkthrough
```

Notebooks are included in documentation via nbsphinx.

## Contributing to Documentation

### Style Guide

- Use reStructuredText (.rst) for documentation
- Use Markdown (.md) for video scripts
- Include code examples with proper syntax highlighting
- Add cross-references using `:doc:` directive
- Use admonitions for notes, warnings, tips

### Adding New Pages

1. Create `.rst` file in appropriate directory
2. Add to table of contents in `index.rst` or parent document
3. Rebuild documentation to verify

### API Documentation

API docs are auto-generated from docstrings using Sphinx autodoc.

**Docstring format:**

```python
def my_function(param1, param2):
    """Short description.

    Longer description with more details.

    Args:
        param1 (type): Description
        param2 (type): Description

    Returns:
        type: Description

    Example:
        >>> result = my_function(1, 2)
        >>> print(result)
        3
    """
    return param1 + param2
```

### Testing Documentation

Check for broken links:

```bash
cd docs
make linkcheck
```

Check for spelling errors:

```bash
pip install sphinxcontrib-spelling
make spelling
```

## Deployment

Documentation can be deployed to:

- **Read the Docs**: Connect GitHub repository
- **GitHub Pages**: Use `gh-pages` branch
- **Self-hosted**: Copy `_build/html/` to web server

### Read the Docs Setup

1. Import project at https://readthedocs.org
2. Connect GitHub repository
3. Configure webhook for auto-builds
4. Documentation will be available at `https://your-project.readthedocs.io`

### GitHub Pages

```bash
# Build docs
cd docs
make html

# Copy to gh-pages branch
git checkout gh-pages
cp -r _build/html/* .
git add .
git commit -m "Update documentation"
git push origin gh-pages
```

Documentation available at: `https://your-username.github.io/your-repo/`

## Video Tutorials

Scripts for video tutorials are in `video_scripts/`:

- **01_quick_start.md**: 5-minute quick start guide
- **02_data_preparation.md**: 10-minute data preparation tutorial
- **03_interpreting_results.md**: 15-minute results interpretation

### Recording Videos

Recommended tools:

- **Screen recording**: OBS Studio, QuickTime, ScreenFlow
- **Terminal recording**: asciinema
- **Editing**: DaVinci Resolve, iMovie, Premiere Pro
- **Hosting**: YouTube, Vimeo

## Support

For documentation issues:

- Open issue: https://github.com/your-repo/issues
- Email: docs@example.com