# Tabular-DC-Bench

This is the anonymous repository accompanying the submission **"A Meta-Dataset for Deep Clustering on Tabular Data"**.

The repository contains the code used to reproduce all experiments, figures, and tables in the paper.

---

## Dataset

The meta-dataset is available at the following anonymous Zenodo link: (https://zenodo.org/records/20195685)

It contains 375,000 individual model evaluations across 100 tabular datasets and 5 deep clustering algorithms, including ground-truth performance metrics (ARI, NMI, UCA), internal validation proxies (Silhouette, Davies-Bouldin), and full training loss curves.

---

## Repository Structure

```
.
├── bench/              # Code to run the benchmark and build the meta-dataset from raw runs
├── scripts/            # Scripts to reproduce all figures and tables in the paper
├── insights_and_analysis.ipynb  # Main notebook to reproduce all paper figures
└── README.md
```

---

## Reproducing the Analysis

After downloading the dataset from Zenodo, update the data path at the top of `insights_and_analysis.ipynb` and run all cells to reproduce the figures in the paper.
