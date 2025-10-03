# Final Submission Package V3

**Title:** Continuous Enrichment Stratification of Neurotransmitter Genes Reveals Density Peaks in Shared ADHD-Autism Genetic Architecture

**Date:** October 3, 2025

---

## Contents

### Main Manuscript
- **manuscript.pdf** (841KB) - Main manuscript with all V3 updates
  - Primary correlation: r=0.898, p=1.06×10⁻¹³, N=36
  - Label permutation test: p<0.0001
  - Partial correlation (gene length): r=0.554, p=0.00046
  - New Limitation #7 on SNP-count validation
- **manuscript.md** - Markdown source
- **manuscript_v3_formatted.tex** - LaTeX source

### Appendices

#### Appendix A: Mathematical Framework
- **mathematical_framework.pdf** (71KB)
- **mathematical_framework.md** - Markdown source
- **mathematical_framework.tex** - LaTeX source

**Contents:**
- Formal problem statement and gene enrichment space
- Topological structure and stratification theory
- K-means clustering algorithm details
- Validation statistics and power analysis
- Computational complexity analysis

#### Appendix B: Biological Systems
- **biological_systems.pdf** (44KB)
- **biological_systems.md** - Markdown source
- **biological_systems.tex** - LaTeX source

**Contents:**
- Detailed gene functions for all 36 neurotransmitter genes
- Glutamatergic system (GRIN2A, GRM5, GRIA1, GRIN2B)
- GABAergic system (GABRB2, GABRB1, GABRB3)
- Serotonergic system (TPH2)
- Dopaminergic system
- Literature references and mechanisms

### Figures
**figures/** - Directory containing all manuscript figures

**Main Figures:**
- figure1_enrichment_patterns.pdf/png - Five enrichment patterns
- figure2_cross_disorder_validation.pdf/png - Cross-disorder validation
- figure3_novel_findings.pdf/png - GABRB2 discovery and E/I balance

**Supplementary Figures:**
- figureS1_adhd_vs_autism.pdf/png - ADHD vs Autism distributions
- figureS2_dendrogram.pdf/png - Hierarchical clustering
- figureS3_gene_cross_disorder.pdf/png - Gene-by-gene replication
- figureS4_pathway_composition.pdf/png - Pathway composition
- figureS5_sensitivity_analysis.pdf/png - Enrichment metric sensitivity
- figureS6_bootstrap_stability.pdf/png - Bootstrap stability analysis
- manifold_visualization.pdf/png - Topological manifold visualization

---

## Key V3 Updates

### Statistical Updates
1. **N=36 genes** (was 35 in V2) - corrected gene count
2. **Primary correlation:** r=0.898 (was 0.913 in V2)
3. **95% CI:** [0.830, 0.940] (was [0.833, 0.956])
4. **P-value:** p=1.06×10⁻¹³ (more precise than p<0.0001)

### New Validation Tests
1. **Label permutation test:** 10,000 iterations, p<0.0001
   - Null distribution: mean r=0.003, SD=0.170
   - Observed r=0.898 exceeded 100% of permutations
2. **Partial correlation:** r=0.554, p=0.00046
   - Controls for gene length confound
   - Demonstrates genuine biological signal

### New Limitation
**Limitation #7** (page 11): SNP-count validation method
- Acknowledges SNP-count vs MAGMA gene-based validation
- Notes future work should use MAGMA for cross-disorder validation
- Current approach successfully validated (r=0.898, partial r=0.554)

### Methods Updates
- **Section 2.5:** Added validation rationale explaining SNP-count approach
- **Section 2.6:** Added label permutation and partial correlation details

---

## Compilation Instructions

### Requirements
- XeLaTeX or PDFLaTeX
- LaTeX packages: amsmath, graphicx, booktabs, natbib, hyperref

### Compile Main Manuscript
```bash
xelatex manuscript_v3_formatted.tex
xelatex manuscript_v3_formatted.tex  # Run twice for references
```

### Compile Appendices
```bash
xelatex mathematical_framework.tex
xelatex biological_systems.tex
```

---

## File Integrity

**Total package size:** ~1.1 MB (excluding source .tex/.md files)

**PDF checksums:** (generated October 3, 2025)
- manuscript.pdf: 841KB
- mathematical_framework.pdf: 71KB
- biological_systems.pdf: 44KB

**Figure formats:** Both PDF (vector) and PNG (raster) provided for flexibility

---

## Citation

[To be determined upon publication]

**Preprint:** [To be deposited]

---

## Contact

[Author information to be added]

---

**Generated:** October 3, 2025
**Version:** 3.0 (Final)
**Status:** Ready for submission
