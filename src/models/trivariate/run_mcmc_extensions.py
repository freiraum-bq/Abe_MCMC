# %% 1. Import necessary libraries & set project root & custom modules & helper function
# -- 1. Import necessary libraries & set project root & custom modules & helper function --
# ------------------------------------------------------------------

import os
import sys
# ------------------------------------------------------------------
# Find project root (folder containing "src") )
# ------------------------------------------------------------------
cwd = os.getcwd()
while not os.path.isdir(os.path.join(cwd, 'src')):
    parent = os.path.dirname(cwd)
    if parent == cwd:
        break  # Reached the root of the filesystem
    cwd = parent
project_root = cwd
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------

# Import rest of libraries
import numpy as np
import pandas as pd
import pickle
import time

# For interactive table display
from IPython.display import display

# Import custom modules for model estimation and prediction
from src.models.trivariate.mcmc import mcmc_draw_parameters_rfm_m
# ------------------------------------------------------------------------
# %% 2. Load full CDNOW CBS with customer covariates ---------------------------
# -- 2. Load full CDNOW CBS with customer covariates ---------------------------

# ------------------------------------------------------------------------
# Define the relative or absolute path to the data directory.
# Adjust if your notebook sits elsewhere.
# ------------------------------------------------------------------------

# CHANGE TO cdnow_fullCBS, once our analysis script is done to rerun on actual full dataset
FILE_CBS = os.path.join(project_root, "data", "processed", "cdnow_abeCBS.csv")

# ------------------------------------------------------------------------
# Read the CSV.  If the column "first" (first purchase date) exists,
# read it as datetime; everything else stays numeric / categorical.
# ------------------------------------------------------------------------
parse_cols = ["first"] if "first" in pd.read_csv(FILE_CBS, nrows=0).columns else None
cbs_df = pd.read_csv(FILE_CBS, parse_dates=parse_cols)

# assume `sales` is total spend in calibration 
# and `x` is # of repeat transactions (i.e. excluding the first purchase)
# so (x+1) = total # of purchases in calib window
cbs_df["log_s"] = np.log( cbs_df["sales"] / (cbs_df["x"] + 1) )

# clean up infinities / NaNs (customers with zero spend)
cbs_df["log_s"] = (
    cbs_df["log_s"]
    .replace(-np.inf, 0.0)
    .fillna(0.0)
)

# Quick sanity checks
print("Shape:", cbs_df.shape)       # rows, columns
display(cbs_df.head())              # first five rows
display(cbs_df.describe(include="all").T)  # summary stats


# %% 3. Running MCMC for model estimation M1
# -- 3. Running MCMC for model estimation M1
# ------------------------------------------------------------------
# Estimate Model M1 (no covariates)
start_m1 = time.time()
draws_3pI = mcmc_draw_parameters_rfm_m(
    cal_cbs    = cbs_df,
    covariates = None,      # intercept‐only
    mcmc       =    4000,      # 4000 total iterations
    burnin     =    10000,   # discard the first 10 000 as warm-up
    thin       =    1,         # keep every draw; 4 000 draws after burn-in
    chains     =    4,
    seed       =    42,
    trace      =    1000,
    n_mh_steps =    40,
)
m1_duration = time.time() - start_m1
print(f"Model M1 runtime: {m1_duration:.2f} seconds")

# Ensure the pickles directory exists at the project root
pickles_dir = os.path.join(project_root, "outputs", "pickles")
os.makedirs(pickles_dir, exist_ok=True)
with open(os.path.join(pickles_dir, "full_tri_m1.pkl"), "wb") as f:
    pickle.dump(draws_3pI, f)

# %% 4. Running MCMC for model estimation M2 (age & gender as cov)
# -- 4. Running MCMC for model estimation M2 (age & gender as cov)
# ------------------------------------------------------------------
covariate_cols = ["age_scaled", "gender_binary"]
print("Data shape:", cbs_df.shape)
print("Using covariates:", covariate_cols)

# Estimate Model M2 
# Track runtime for Model M2
start_m2 = time.time()
draws_3pII = mcmc_draw_parameters_rfm_m(
    cal_cbs    = cbs_df,
    covariates  = covariate_cols,
    mcmc    =   4000,      # 4000 total iterations
    burnin  =   10000,   # discard the first 10 000 as warm-up
    thin    =   1,         # keep every draw; 4 000 draws after burn-in
    chains  =   4,
    seed    =   42,
    trace   =   1000,
    n_mh_steps = 40,
)
m2_duration = time.time() - start_m2
print(f"Model M2 runtime: {m2_duration:.2f} seconds")

# Saving result M2
# Ensure the pickles directory exists at the project root
pickles_dir = os.path.join(project_root, "outputs", "pickles")
os.makedirs(pickles_dir, exist_ok=True)
with open(os.path.join(pickles_dir, "full_tri_m2.pkl"), "wb") as f:
    pickle.dump(draws_3pII, f)
# ------------------------------------------------------------------

# Print runtimes
runtimes = pd.DataFrame({
    "model":   ["M1",           "M2"],
    "runtime": [m1_duration,    m2_duration]
})

# %% 5. Save runtimes to CSV and update
# -- 5. Save runtimes to CSV and update
csv_path = os.path.join(project_root, "outputs", "excel", "mcmc_runtimes.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
if os.path.exists(csv_path):
    df_runs = pd.read_csv(csv_path)
else:
    df_runs = pd.DataFrame(columns=["model", "runtime"])
for model_name, runtime in [
    ("full_tri_M1", m1_duration),
    ("full_tri_M2", m2_duration)
]:
    # Remove any old entry for this model
    df_runs = df_runs[df_runs.model != model_name]
    # Append the new timing
    df_runs = pd.concat(
        [df_runs, pd.DataFrame([{"model": model_name, "runtime": runtime}])],
        ignore_index=True
    )
df_runs.to_csv(csv_path, index=False)
print(f"Saved runtimes to {csv_path}")
# %%
