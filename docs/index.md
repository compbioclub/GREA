# GREA

Welcome to the official documentation of **GREA**!

## Overview

![workflow](./figure/workflow.png)

Gene Set Enrichment Analysis (GSEA) is a cornerstone for interpreting gene expression data, yet traditional approaches overlook gene interactions by focusing solely on individual genes, limiting their ability to detect subtle or complex pathway signals. To overcome this, we present GREA (Gene Interaction Enrichment Analysis), a novel framework that incorporates gene interaction data into enrichment analysis. GREA replaces the binary gene hit indicator with an interaction overlap ratio, capturing the degree of overlap between gene sets and gene interactions to enhance sensitivity and biological interpretability. It supports three enrichment metrics: Enrichment Score (ES), Enrichment Score Difference (ESD) from a Kolmogorov-Smirnov-based statistic, and Area Under the Curve (AUC) from a recovery curve. GREA evaluates statistical significance using both permutation testing and gamma distribution modeling. Benchmarking on transcriptomic datasets related to respiratory viral infections shows that GREA consistently outperforms existing tools such as blitzGSEA and GSEApy, identifying more relevant pathways with greater stability and reproducibility. By integrating gene interactions into pathway analysis, GREA offers a powerful and flexible tool for uncovering biologically meaningful insights in complex datasets.

## Getting Started

Want to start using it immediately? Check out the [Installation Guide](installation.md).


## Tutorial Guide
The followings are tutorials of how to use GREA in different scenarios:

-   [Phenotype-level prerank enrichment for genes](tutorial/gene-pheno_prerank_enrich.ipynb)
-   [Phenotype-level prerank enrichment for gene interactions](tutorial/interaction-pheno_prerank_enrich.ipynb)
-   [Observation-level prerank enrichment for genes](tutorial/gene-obs_prerank_enrich.ipynb)
-   [Observation-level prerank enrichment for gene interactions](tutorial/interaction-obs_prerank_enrich.ipynb)

## Citation

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


<div style="display:none;">
<script type='text/javascript' id='clustrmaps' src='//cdn.clustrmaps.com/map_v2.js?cl=ffffff&w=a&t=n&d=s_zp3a_kJX2eHlUNurnH4Jti8lf7sMFpJyhQQnn21MU'></script>
<script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "aec36b862a47431a979dc263a1f98d74"}'></script>
</div>