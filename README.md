# A comparative study of DP-SGD and Private Evolution for differentially private synthetic data

View the full report [here](https://www.overleaf.com/read/frpngrjrrqkt#80a851).

| Name | Role | GitHub |
| --- | --- | --- |
| Mehak Kapur<br><sub>mekapur@ucsd.edu</sub> | Student | [@mekapur](https://github.com/mekapur) |
| Hana Tjendrawasi<br><sub>htjendrawasi@ucsd.edu</sub> | Student | [@hanajuliatj](https://github.com/hanajuliatj) |
| Jason Tran<br><sub>jat037@ucsd.edu</sub> | Student | [@jktrn](https://github.com/jktrn) |
| Phuc Tran<br><sub>pct001@ucsd.edu</sub> | Student | [@21phuctran](https://github.com/21phuctran) |
| Yu-Xiang Wang<br><sub>yuxiangw@ucsd.edu</sub> | Advisor | [@yuxiangw](https://github.com/yuxiangw) |

---

## Abstract

Differentially private synthetic data generation enables the release of realistic datasets while rigorously protecting the privacy of individuals in the source data. Two methodologies dominate contemporary research: (1) differentially private stochastic gradient descent (DP-SGD), which privatizes model training through carefully calibrated noise injection, and (2) Private Evolution (PE), which achieves privacy guarantees through inference-only access to pre-trained foundation models. We implement both approaches on realistic datasets, evaluating their privacy-utility trade-offs under varying $\varepsilon$-budgets and computational constraints. Our objective is to assess whether the PE paradigm can rival the fidelity of DP-SGD while obviating the latter's computational and implementation complexity.

---

### Overview

| Path                                        | Purpose                                                                                                                 |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `data/`                                     | Input telemetry CSV and the latest synthetic dataset produced by our generators.                                        |
| `docs/`                                     | Project report sources and supporting references (TeX, bibliography, exported figures).                                 |
| `notebooks/01-exploration-deprecated.ipynb` | Early exploratory analysis (kept for provenance, not part of the workflow).                                             |
| `notebooks/02-dp-sgd-deprecated.ipynb`      | Prototype notebook experimenting with DP-SGD-based generators.                                                          |
| `notebooks/03-validation.ipynb`             | Validation and comparison of real vs. synthetic telemetry (distributions, KS test, logistic-regression utility checks). |
| `scripts/dp-sgd-deprecated.py`              | Script equivalent of the deprecated DP-SGD notebook.                                                                    |
| `scripts/dp-vae.py`                         | Current synthetic-data generator (DP-VAE with Opacus).                                                                  |

The assets marked as `-deprecated` remain in the repository for traceability but are not part of the primary pipeline.

---

### Requirements

- Python 3.10+ (tested with 3.12)
- Recommended packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `scipy`
  - `matplotlib`
  - `seaborn`
  - `torch` (GPU optional)
  - `opacus`
  - `pgfplots`/`tikz` (LaTeX packages) if you plan to recompile the report

You can install the Python dependencies via:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn torch opacus
```

---

### Usage

1. **Prepare data**
   - Place the telemetry CSV at `data/telemetry.csv`.
   - The generator writes `data/synthetic.csv` by default (existing files will be overwritten).

2. **Generate synthetic data**
   ```bash
   python scripts/dp-vae.py
   ```
   The script trains a differentially private VAE using Opacus and saves the sampled synthetic dataset.

3. **Validate utility**
   - Open `notebooks/03-validation.ipynb` and run all cells to compare real vs. synthetic distributions, compute KS statistics, and evaluate downstream logistic regression performance.
