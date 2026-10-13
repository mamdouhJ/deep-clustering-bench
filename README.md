# Tabular-DC-Bench

Code and analysis for **"Tabular-DC-Bench: A Large-Scale Meta-Dataset for Hyperparameter Optimization in Deep Clustering"** (AutoML 2026).

🏆 Runner-up, ABCD track, AutoML 2026.

Deep clustering can outperform K-Means on tabular data, but only with the right hyperparameter configuration — and without labels there is no reliable way to find it. Tabular-DC-Bench provides the empirical data needed to study that problem: 375,000 deep clustering runs with ground truth recorded for every one.

---

## Dataset

The meta-dataset is available on Zenodo: https://zenodo.org/records/20195685

| | |
|---|---|
| Datasets | 100 (OpenML, numerical features, categorical target) |
| Algorithms | 5 (IDEC, VaDE, DipEncoder, DeepECT, DDC) |
| Configurations | ~150 per dataset–algorithm pair (75,000 total) |
| Seeds | 5 per configuration |
| **Total runs** | **375,000** |

Each run records:

- the hyperparameter configuration (network depth, width, bottleneck size, batch size, pretraining and clustering learning rates, early stopping)
- external metrics: ARI, NMI, UCA
- internal metrics: Silhouette, Davies–Bouldin
- the full training loss curve

Datasets span three regimes by sample size, from under 1,000 to over 45,000 samples, including 17 imbalanced datasets (normalized entropy < 0.75).

---

## Repository structure

```
.
├── bench/                       # Run the benchmark and build the meta-dataset from raw runs
├── scripts/                     # Reproduce all figures and tables in the paper
├── insights_and_analysis.ipynb  # Main analysis notebook
└── README.md
```

---

## Reproducing the analysis

```bash
git clone https://github.com/mamdouhJ/deep-clustering-bench
cd deep-clustering-bench
pip install -r requirements.txt
```

Download the meta-dataset from Zenodo, set the data path at the top of `insights_and_analysis.ipynb`, and run all cells to reproduce the figures and tables in the paper.


---

## License

See `LICENSE`.

## Contributing

Issues and pull requests are welcome, particularly extensions to further algorithms, datasets, or data types. Updated versions of the meta-dataset are released on Zenodo.

## Contact

Mamdouh Aljoud — aljoud@dbs.ifi.lmu.de
