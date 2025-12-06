# Differentially private synthetic data generation via DP-VAE

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

Differentially private synthetic data generation enables the release of realistic datasets while rigorously protecting the privacy of individuals in the source data. We implement a differentially private variational autoencoder (DP-VAE) trained using DP-SGD, which privatizes model training through per-sample gradient clipping and calibrated Gaussian noise injection. Applying this approach to a realistic telemetry dataset, we demonstrate that DP-VAE can generate high-fidelity synthetic records under a modest privacy budget (ε = 4.0).

---

## Project structure

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

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for package management.

1. If you don't have uv installed:

   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or with Homebrew
   brew install uv

   # Or with pip
   pip install uv
   ```

2. Create your virtual environment:

   ```bash
   uv venv
   uv sync
   ```

   This creates a `.venv` directory and installs all dependencies from `pyproject.toml`.

3. Activate the environment:

   ```bash
   # macOS/Linux
   source .venv/bin/activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

   # Windows (cmd)
   .venv\Scripts\activate.bat
   ```

4. To run the notebooks in Jupyter/VS Code/Cursor, register the virtual environment as a Jupyter kernel. Make sure the `.venv` is activated, then:

   ```bash
   python -m ipykernel install --user --name dsc-180a-q1 --display-name "DSC 180A Q1"
   ```

   Now you can select "DSC 180A Q1" as the kernel when opening notebooks.

---

## Usage

1. Place the telemetry CSV at `data/telemetry.csv`. The generator writes `data/synthetic.csv` by default (existing files will be overwritten).

2. Generate synthetic data:

   ```bash
   # With venv
   python scripts/dp-vae.py

   # Without venv
   uv run python scripts/dp-vae.py
   ```

   The script trains a differentially private VAE using Opacus and saves the sampled synthetic dataset.

3. To validate utility, open `notebooks/03-validation.ipynb` and run all cells to:
   - Compare real vs. synthetic distributions
   - Compute KS statistics and z-score metrics
   - Evaluate downstream logistic regression performance
