# GREA：Knowledge-driven annotation for gene interaction enrichment analysis

## 🗺️ Overview

![workflow](./docs/figure/workflow.png)

Gene Set Enrichment Analysis (GSEA) is a cornerstone for interpreting gene expression data, yet traditional approaches overlook gene interactions by focusing solely on individual genes, limiting their ability to detect subtle or complex pathway signals. To overcome this, we present GREA (Gene Interaction Enrichment Analysis), a novel framework that incorporates gene interaction data into enrichment analysis. GREA replaces the binary gene hit indicator with an interaction overlap ratio, capturing the degree of overlap between gene sets and gene interactions to enhance sensitivity and biological interpretability. It supports three enrichment metrics: Enrichment Score (ES), Enrichment Score Difference (ESD) from a Kolmogorov-Smirnov-based statistic, and Area Under the Curve (AUC) from a recovery curve. GREA evaluates statistical significance using both permutation testing and gamma distribution modeling. Benchmarking on transcriptomic datasets related to respiratory viral infections shows that GREA consistently outperforms existing tools such as blitzGSEA and GSEApy, identifying more relevant pathways with greater stability and reproducibility. By integrating gene interactions into pathway analysis, GREA offers a powerful and flexible tool for uncovering biologically meaningful insights in complex datasets.

## 📥 Installation

**From GitHub (recommended for the latest version)**

```
pip install git+https://github.com/compbioclub/GREA.git@dev
```

## 🚀 Quick Start

### 1️⃣ Running the GREA Pipeline

The following is an example of how to use GREA in a complete workflow:

```python
from grea import GREA
```

## 📑 Citation

If you use GREA in your research, please cite the following paper:

APA format:

```
Liu, X., Jiang, A., Lyu, C., & Chen, L. (2025). Knowledge-driven annotation for gene interaction enrichment analysis. bioRxiv, 2025-04. https://doi.org/10.1101/2025.04.15.649030
```

BibTeX format:

```bibtex
@article{grea,
  title={Knowledge-driven annotation for gene interaction enrichment analysis},
  author={Liu, Xiaoyu and Jiang, Anna and Lyu, Chengshang and Chen, Lingxi},
  journal={bioRxiv},
  pages={2025--04},
  year={2025},
  doi={10.1101/2025.04.15.649030},
  publisher={Cold Spring Harbor Laboratory}
}
```
