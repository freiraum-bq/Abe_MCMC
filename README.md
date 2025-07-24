# Abe (2009) Pareto/NBD Model Implementation (Modular Python Version)

This repository contains a modular Python implementation of the hierarchical Bayesian Pareto/NBD model from Abe (2009), applied to the CDNOW dataset. It supports model estimation, customer-level forecasts, and evaluation of model fit and predictive accuracy, reproducing Tables 1–4 and key figures from the original paper.
Furthermore, it contains the three parameter extension, suggested by Abe 2015. 
Finally, it extends Abe's analysis by incorporating more customer characteristics for the CDNOW dataset as well as a comparison of the four models.

## Repository Structure

```
Abe_MCMC/
├── data/
│   ├── raw/                  # Raw input data in Elog
│   └── processed/            # Processed data to CBS
├── outputs/
│   ├── excel/                # Generated Excel summaries and tables
│   ├── figures/              # Generated figures and plots
│   │   ├── abe_replication/  # Plots for Abe 2009 replication
│   │   ├── extension/        # Plots for extension / Bi & Tri separately
│   │   └── x_comparison_four_models/  # Plots for four model comparison
│   └── pickles/              # Saved MCMC draws and model outputs
├── src/
│   ├── data_processing/
│   │   ├── 1A_cdnow_fetchRaw_abe.py   # Extraction of original Abe dataset
│   │   ├── 1B_cdnow_fetchRaw_full.py  # Processing the full CDNOW dataset
│   │   ├── 2A_cdnow_elog2cbs_abe.py   # Convert Abe event log to CBS
│   │   └── 2B_cdnow_elog2cbs_full.py  # Convert full event log to CBS
│   ├── models/
│   │   ├── bivariate/
│   │   │   ├── analysis_abe.py            # Bivariate model analysis | Abe 2009 Replication
│   │   │   ├── analysis_extensions.py     # Bivariate model analysis | Extending with more customer characteristics
│   │   │   ├── mcmc.py                    # Bivariate MCMC routines
│   │   │   ├── run_mcmc_abe.py            # Script to run MCMC | Abe Replication
│   │   │   └── run_mcmc_extensions.py     # Script to run MCMC | Extension via customer characteristics
│   │   ├── trivariate/
│   │   │   ├── analysis_extensions.py     # Trivariate model analysis | Extension
│   │   │   ├── mcmc.py                    # Trivariate MCMC routines
│   │   │   └── run_mcmc_extensions.py     # Script to run Trivariate MCMC
│   │   └── utils/
│   │       ├── analysis_bi_dynamic.py     # Helper for generating parameter names and labels for bivariate hierarchical models with arbitrary covariates.
│   │       ├── analysis_bi_helpers.py     # Functions for analysis
│   │       ├── analysis_display_helper.py # Helper display function
│   │       ├── analysis_tri_helpers.py    # Functions for analysis
│   │       └── elog2cbs2param.py          # Converting Elog to CBS
│   ├── full_analysis.ipynb   # Archived analysis for four models (Extension) (based on full_analysis.py)
│   └── full_analysis.py      # Analysis for four models (Extension)
└── README.md                 
```

## Prerequisites

- Python 3.8 or higher
- Required packages:
  ```bash
  pip install numpy pandas scipy matplotlib seaborn jupyter openpyxl lifetimes arviz
  ```

## Usage
### 0. .py scripts
Scripts can be easily run in an interactive window; sometimes restart of VS Code is required due to bugs.

### 1. Data Preparation

- Place the raw CDNOW event log files in `data/raw/`:
  - Abe subset: `cdnow_abeElog.csv`
  - Full dataset: `cdnow_fullElog.csv`
- Convert event logs to CBS format using the provided scripts in `src/data_processing/`:
  - For Abe subset:
    ```bash
    python src/data_processing/1A_cdnow_fetchRaw_abe.py
    python src/data_processing/2A_cdnow_elog2cbs_abe.py
    ```
    This will output `cdnow_abeCBS.csv` in `data/processed/`.
  - For full dataset:
    ```bash
    python src/data_processing/1B_cdnow_fetchRaw_full.py
    python src/data_processing/2B_cdnow_elog2cbs_full.py
    ```
    This will output `cdnow_fullCBS.csv` in `data/processed/`.

#### Choosing the Analysis Subset
- For the **Abe subset** (1/10th of the full data, as in Abe 2009), use:
  - `data/processed/cdnow_abeCBS.csv` (CBS)
  - `data/raw/cdnow_abeElog.csv` (event log)
- For the **full dataset** analysis, use:
  - `data/processed/cdnow_fullCBS.csv` (CBS)
  - `data/raw/cdnow_fullElog.csv` (event log)
- Make sure the correct files are loaded in your analysis scripts (see debug prints in scripts for confirmation).

### 2. Model Estimation and Analysis

#### Bivariate Model (Abe 2009 and Extensions)
- **Run MCMC for Abe 2009 replication:**
  ```bash
  python src/models/bivariate/run_mcmc_abe.py
  ```
  - Outputs: `abe_bi_m1.pkl`, `abe_bi_m2.pkl` in `outputs/pickles/`
  - Runtimes: `outputs/excel/mcmc_runtimes.csv`
- **Run MCMC for bivariate extension (with more covariates):**
  ```bash
  python src/models/bivariate/run_mcmc_extensions.py
  ```
  - Outputs: `ext_bi_m1.pkl`, `ext_bi_m2.pkl` in `outputs/pickles/`
  - Runtimes: `outputs/excel/mcmc_runtimes.csv`
- **Analysis and plotting:**
  - For Abe 2009 replication:
    ```bash
    python src/models/bivariate/analysis_abe.py
    ```
    - Outputs: Excel summaries in `outputs/excel/abe_replication.xlsx`, figures in `outputs/figures/abe_replication/`
  - For bivariate extension:
    ```bash
    python src/models/bivariate/analysis_extensions.py
    ```
    - Outputs: Excel summaries in `outputs/excel/abe_extension.xlsx`, figures in `outputs/figures/extension/`

#### Trivariate Model (Abe 2015-style Extension)
- **Run MCMC for trivariate extension:**
  ```bash
  python src/models/trivariate/run_mcmc_extensions.py
  ```
  - Outputs: `ext_tri_m1.pkl`, `ext_tri_m2.pkl` in `outputs/pickles/`
  - Runtimes: `outputs/excel/mcmc_runtimes.csv`
- **Analysis and plotting:**
  ```bash
  python src/models/trivariate/analysis_extensions.py
  ```
  - Outputs: figures in `outputs/figures/extension/`

### 3. Dynamic Covariate Handling

- Covariate and parameter names are handled dynamically using helpers in `src/models/utils/analysis_bi_dynamic.py` and related scripts.
- Specify your covariate list in the MCMC runner scripts; parameter names and summary labels will update automatically throughout the analysis and plotting scripts.
- This makes it easy to extend to models with any number of covariates.

### 4. Interactive Exploration
- Launch a Jupyter notebook for custom analysis:
  ```bash
  jupyter notebook
  ```
  and open any notebook in the `notebooks/` directory (if present).

## Results

- **Excel summaries**: All tables (Tables 1–4) are saved in `outputs/excel/` (e.g., `abe_replication.xlsx`, `abe_extension.xlsx`, `x_comparison_four_models.xlsx`).
- **Pickled MCMC draws**: Saved in `outputs/pickles/` (e.g., `abe_bi_m1.pkl`, `ext_bi_m2.pkl`, `ext_tri_m1.pkl`) for reproducibility and further analysis. **Note:** These pickled MCMC draw files are not included in the repository by default.
- **Figures**: All plots and figures are saved in `outputs/figures/` and subfolders:
  - `outputs/figures/abe_replication/` (Abe 2009 replication)
  - `outputs/figures/extension/` (bivariate and trivariate extensions)
  - `outputs/figures/x_comparison_four_models/` (model comparison)

## Accessing Pickled MCMC Draw Files

If you require the pickled MCMC draw files for your own analysis or reproduction of results, please contact the repository owner directly (see Contact section). The pickle files are stored on a cloud service and are not distributed with this repository due to their large size and storage constraints. Message the repository owner to request access to these files.

## Contact

Email: Jannik.J.Schneider@stud.leuphana.de