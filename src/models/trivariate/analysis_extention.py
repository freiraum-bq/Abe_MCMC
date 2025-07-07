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
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import arviz as az

# custom helper
from src.models.utils.analysis_display_helper import _fmt
from src.models.utils.analysis_tri_helpers import (summarize_level2, 
                                                  post_mean_lambdas, 
                                                  post_mean_mus,
                                                  post_mean_etas,
                                                  mape_aggregate, 
                                                  extract_correlation, 
                                                  chain_total_loglik, 
                                                  compute_table4)

# For interactive table display
from IPython.display import display

# ------------------------------------------------------------------------

# %% 2. Load estimates, data and set paths
# -- 2. Load estimates, data and set paths

# --- Load Pre-computed Results ---
pickles_dir = os.path.join(project_root, "outputs", "pickles")

# Set folder to save Figures 
figure_path = os.path.join(project_root, "outputs", "figures", "full_extention")
os.makedirs(figure_path, exist_ok=True)   # create the folder if it doesn’t exist

# Load MCMC draws
with open(os.path.join(pickles_dir, "full_tri_m1.pkl"), "rb") as f:
    draws_3pI = pickle.load(f)
with open(os.path.join(pickles_dir, "full_tri_m2.pkl"), "rb") as f:
    draws_3pII = pickle.load(f)

# Load CBS data --> CHANGE TO FULL LATER
cbs_path = os.path.join(project_root, "data", "processed", "cdnow_abeCBS.csv")
cbs_df = pd.read_csv(cbs_path, dtype={"cust": str}, parse_dates=["first"])

# Load Elog data --> CHANGE TO FULL LATER
data_path = os.path.join(project_root, "data", "raw", "cdnow_abeElog.csv")
cdnowElog = pd.read_csv(data_path)
# Convert date column to datetime
cdnowElog["date"] = pd.to_datetime(cdnowElog["date"])
# ensure the same key type
cdnowElog["cust"] = cdnowElog["cust"].astype(str)

cbs_df["log_s"] = np.log( cbs_df["sales"] / (cbs_df["x"] + 1) )
# clean up infinities / NaNs (customers with zero spend)
cbs_df["log_s"] = (
    cbs_df["log_s"]
    .replace(-np.inf, 0.0)
    .fillna(0.0)
)
# ------------------------------------------------------------------

# %% 3. Table 2 – HB RFM model-fit metrics (no covariates vs. gender + age) --
# -- 3. Table 2 – HB RFM model-fit metrics (no covariates vs. gender + age) --

# Prepare weekly actual repeat counts
first_date = cdnowElog["date"].min()
cdnowElog["week"] = ((cdnowElog["date"] - first_date) // pd.Timedelta("7D")).astype(int) + 1
cdnowElog_sorted = cdnowElog.sort_values(by=["cust", "week"])
cdnowElog_sorted["txn_order"] = cdnowElog_sorted.groupby("cust").cumcount()
repeat_txns = cdnowElog_sorted[cdnowElog_sorted["txn_order"] >= 1]
max_week = cdnowElog["week"].max()
weekly_actual = (
    repeat_txns.groupby("week")["cust"]
    .count()
    .reindex(range(1, max_week + 1), fill_value=0)
    .to_numpy()
)

times = np.arange(1, max_week + 1)
weeks_cal_mask = (times >= 1) & (times <= 39)
weeks_val_mask = (times >= 40) & (times <= 78)

# Compute birth week for each customer
birth_week = (
    cdnowElog.groupby("cust")["week"].min()
    .reindex(cbs_df["cust"])
    .to_numpy()
)

# Helper to simulate weekly increments from draws
def simulate_weekly(draws):
    inc = np.zeros_like(times, dtype=float)
    chains = draws["level_1"]
    total = sum(chain.shape[0] for chain in chains)
    for chain_idx, chain in enumerate(chains):
        for draw_idx in range(chain.shape[0]):
            lam = chain[draw_idx, :, 0]
            mu  = chain[draw_idx, :, 1]
            tau = chain[draw_idx, :, 2]
            rng = np.random.default_rng(chain_idx * chain.shape[0] + draw_idx)
            for t_idx, t in enumerate(times):
                active = (t > birth_week) & (t <= birth_week + tau)
                inc[t_idx] += rng.poisson(lam * active).sum()
    return inc / total

# Simulate increments for both models
inc_m1 = simulate_weekly(draws_3pI)
inc_m2 = simulate_weekly(draws_3pII)

weeks_cal_mask = (times >= 1)  & (times <= 39)   # calibration window
weeks_val_mask = (times >= 40) & (times <= 78)   # validation window

# -----------------------------------------------------------------------
# 3) Metric helper ------------------------------------------------------
# -----------------------------------------------------------------------
def compute_metrics(draws: dict, label: str) -> dict[str, float]:
    """Return correlation / MSE / MAPE metrics for RFM-M draws."""
    all_d = np.concatenate(draws["level_1"], axis=0)       # (D, N, 5)
    lam   = all_d[:, :, 0].mean(axis=0)
    mu    = all_d[:, :, 1].mean(axis=0)
    z     = all_d[:, :, 3].mean(axis=0)                    # P(alive)

    t_star = 39
    xstar_pred = z * (lam / mu) * (1 - np.exp(-mu * t_star))

    corr_val = np.corrcoef(cbs_df["x_star"], xstar_pred)[0, 1]
    mse_val  = np.mean((cbs_df["x_star"] - xstar_pred) ** 2)

    calib_pred = (lam / mu) * (1 - np.exp(-mu * cbs_df["T_cal"]))
    corr_cal = np.corrcoef(cbs_df["x"], calib_pred)[0, 1]
    mse_cal  = np.mean((cbs_df["x"] - calib_pred) ** 2)

    # Posterior‑predictive weekly increments for this model
    inc_weekly = simulate_weekly(draws)
    # weekly_actual is already a numpy array
    weekly_arr = weekly_actual

    def mape(a, p):
        cum_a = np.cumsum(a)
        cum_p = np.cumsum(p)
        return np.abs(cum_p - cum_a).mean() / cum_a[-1] * 100

    return {
        "label":     label,
        "corr_val":  corr_val,
        "corr_cal":  corr_cal,
        "mse_val":   mse_val,
        "mse_cal":   mse_cal,
        "mape_val":  mape(weekly_arr[weeks_val_mask], inc_weekly[weeks_val_mask]),
        "mape_cal":  mape(weekly_arr[weeks_cal_mask], inc_weekly[weeks_cal_mask]),
        "mape_pool": mape(weekly_arr, inc_weekly)
    }

# -----------------------------------------------------------------------
# 4) Compute metrics for both models ------------------------------------
stats_0 = compute_metrics(draws_3pI,  "HB RFM (no cov)")
stats_1 = compute_metrics(draws_3pII, "HB RFM (+ gender & age)")

# -----------------------------------------------------------------------
# 5) Assemble two-column Table 2 ----------------------------------------
table2 = pd.DataFrame({
    stats_0["label"]: [
        stats_0["corr_val"], stats_0["corr_cal"],
        stats_0["mse_val"],  stats_0["mse_cal"],
        stats_0["mape_val"], stats_0["mape_cal"], stats_0["mape_pool"]
    ],
    stats_1["label"]: [
        stats_1["corr_val"], stats_1["corr_cal"],
        stats_1["mse_val"],  stats_1["mse_cal"],
        stats_1["mape_val"], stats_1["mape_cal"], stats_1["mape_pool"]
    ]
}, index=[
    "Correlation (Validation)", "Correlation (Calibration)",
    "MSE (Validation)",         "MSE (Calibration)",
    "MAPE (Validation)",        "MAPE (Calibration)", "MAPE (Pooled)"
]).round(2)

row_order = [
    "Disaggregate measure",
    "Correlation (Validation)", "Correlation (Calibration)", "",
    "MSE (Validation)",         "MSE (Calibration)",         "",
    "Aggregate measure", "Time-series MAPE (%)",
    "MAPE (Validation)",        "MAPE (Calibration)",        "MAPE (Pooled)"
]
table2 = table2.reindex(row_order)

# Print / display
table2_disp = table2.reset_index().rename(columns={"index": ""})
print("\nTable 2. Model-fit - Trivariates - HB RFM, CDNOW dataset")
display(table2_disp)

#--------------------------------------------------------------------------------
# Make sure the loaded est. data is correct / does actually differ (althought just so slightly)
stats_0 = compute_metrics(draws_3pI,  "HB RFM (no cov)")
stats_1 = compute_metrics(draws_3pII, "HB RFM (+ gender & age)")

# Optional | heads up: takes a while to print...
# print("To verify that the loaded estimates are correct, and do differ (slightly):")
# print("Raw stats M1:", stats_0)
# print("Raw stats M2:", stats_1)


# %% 4. Figure 2 – Weekly cumulative repeat transactions
# -- 4. Figure 2 – Weekly cumulative repeat transactions

# Plots three curves:
#   • Actual cumulative repeats
#   • HB RFM “no covariates”      (draws_3pI)
#   • HB RFM “+ gender & age”     (draws_3pII)
#
# Prerequisites already in memory:
#   • times, weekly_actual, birth_week   (helper block)
#   • draws_3pI   – intercept-only RFM–M
#   • draws_3pII  – gender_F + age_scaled RFM–M
# ----------------------------------------------------
def posterior_cumulative(draws: dict, label: str) -> np.ndarray:
    """
    Return posterior-mean cumulative repeat transactions per week.
    """
    inc_weekly = np.zeros_like(times, dtype=float)

    n_chains        = len(draws["level_1"])
    draws_per_chain = len(draws["level_1"][0])
    n_total_draws   = n_chains * draws_per_chain

    for d in range(n_total_draws):
        ch, idx = divmod(d, draws_per_chain)
        lam_d = draws["level_1"][ch][idx, :, 0]
        mu_d  = draws["level_1"][ch][idx, :, 1]
        eta_d = draws["level_1"][ch][idx, :, 2]

        rng = np.random.default_rng(d)            # draw-specific seed
        for t_idx, t in enumerate(times):
            active = (t > birth_week) & (t <= birth_week + eta_d)
            inc_weekly[t_idx] += rng.poisson(lam=lam_d * active).sum()

    inc_weekly /= n_total_draws                   # posterior mean
    return np.cumsum(inc_weekly)                  # cumulative curve

# -------------------------------------------------------------------------
# 1) Compute cumulative curves for both HB models
# -------------------------------------------------------------------------
cum_rfm_noCov = posterior_cumulative(draws_3pI,  "HB RFM (no cov)")
cum_rfm_cov   = posterior_cumulative(draws_3pII, "HB RFM (+ gender & age)")

# Actual cumulative repeats
cum_actual = weekly_actual.cumsum()

# -------------------------------------------------------------------------
# 2) Plot
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(times, cum_actual,      lw=2, color="tab:blue",  label="Actual")
plt.plot(times, cum_rfm_noCov,   lw=2, linestyle="--", color="tab:orange",
         label="HB RFM (no cov)")
plt.plot(times, cum_rfm_cov,     lw=2, linestyle=":",  color="tab:green",
         label="HB RFM (+ gender & age)")

plt.axvline(x=39, color="k", linestyle="--")     # calibration / validation split
plt.xlabel("Week")
plt.ylabel("Cumulative repeat transactions")
plt.title("Figure 2 - Weekly Time-Series Tracking (HB RFM)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figure_path, "Figure2_tri_weekly_tracking.png"), dpi=300, bbox_inches='tight')
plt.show()

# %% 5. Scatterplot - Predictions of both models
# -- 5. Scatterplot - Predictions of both models
cbs = cbs_df
t_star = 39.0
mean_lambda_m1 = post_mean_lambdas(draws_3pI)
mean_mu_m1     = post_mean_mus(draws_3pI)
mean_lambda_m2 = post_mean_lambdas(draws_3pII)
mean_mu_m2     = post_mean_mus(draws_3pII)
cbs["xstar_m1_pred"] = (mean_lambda_m1/mean_mu_m1) * (1 - np.exp(-mean_mu_m1 * t_star))
cbs["xstar_m2_pred"] = (mean_lambda_m2/mean_mu_m2) * (1 - np.exp(-mean_mu_m2 * t_star))


sns.set(style="whitegrid")

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# Scatterplot for Model M1
axes[0].scatter(cbs["x_star"], cbs["xstar_m1_pred"], alpha=0.4, color="tab:blue")
axes[0].plot([0, cbs["x_star"].max()], [0, cbs["x_star"].max()], 'r--')
axes[0].set_title("M1: Without Covariates")
axes[0].set_xlabel("Actual x_star")
axes[0].set_ylabel("Predicted x_star")

# Scatterplot for Model M2
axes[1].scatter(cbs["x_star"], cbs["xstar_m2_pred"], alpha=0.4, color="tab:green")
axes[1].plot([0, cbs["x_star"].max()], [0, cbs["x_star"].max()], 'r--')
axes[1].set_title("M2: With age, gender")
axes[1].set_xlabel("Actual x_star")

# Remove grid from both subplots
for ax in axes:
    ax.grid(False)

plt.tight_layout()
plt.savefig(os.path.join(figure_path, "Scatter_tri_M1_M2.png"), dpi=300, bbox_inches='tight')
plt.show()

# %% 6. Alive vs churned customers
# -- 6. Alive vs churned customers

# Add a new column for predicted alive status based on xstar_m2_pred
cbs = cbs_df
cbs["is_alive_pred"] = np.where(cbs["xstar_m2_pred"] >= 1, 1, 0)

# Prepare data
counts = cbs["is_alive_pred"].value_counts().sort_index()
labels = ["Churned (z = 0)", "Alive (z = 1)"]
colors = ["#d3d3d3", "#4a90e2"]  # Light grey and business blue

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 4))

# Plot bar chart
bars = ax.bar(labels, counts, color=colors, width=0.5)

# Set axis labels and title
ax.set_ylabel("Number of customers", fontsize=11)
ax.set_title("Predicted Alive vs. Churned\n(Last Draw of MCMC Chain)", fontsize=13)


# Annotate each bar with its value
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        f"{int(height)}",
        ha='center',
        va='bottom',
        fontsize=10
    )

# Disable grid lines so they don’t appear behind annotations
ax.grid(False)

# Clean up the axis appearance
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Hide the vertical left spine so it doesn’t bisect the first bar
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#999999')
ax.tick_params(axis='y', colors='#444444')
ax.tick_params(axis='x', colors='#444444')

# Final layout adjustment
plt.tight_layout()
plt.savefig(os.path.join(figure_path, "Alive_vs_Churned_bi.png"), dpi=300, bbox_inches='tight')
plt.show()

# %% 7. Poserior distribution and Traceplots
# -- 7. Poserior distribution and Traceplots

# Convert M1 to InferenceData
idata_m1 = az.from_dict(
    posterior={"level_2": np.array(draws_3pI["level_2"])},  # shape: (chains, draws, dims)
    coords={"param": [  # labels for better plots
        "log_lambda (intercept)", 
        "log_mu (intercept)",
        "log_eta (intercept)", 
        "var_log_lambda", 
        "cov_log_lambda_mu",
        "cov_log_lambda_eta", 
        "var_log_mu",
        "cov_log_mu_eta",
        "var_log_eta"
    ]},
    dims={"level_2": ["param"]}
)
# Convert M2 to InferenceData
idata_m2 = az.from_dict(
    posterior={"level_2": np.array(draws_3pII["level_2"])},
    coords={"param": [
    "log_lambda (intercept)",
    "log_lambda (age_scaled)",
    "log_lambda (gender_binary)",
    "log_mu (intercept)",
    "log_mu (age_scaled)",
    "log_mu (gender_binary)",
    "log_eta (intercept)",
    "log_eta (age_scaled)",
    "log_eta (gender_binary)",
    "var_log_lambda",
    "var_log_mu",
    "var_log_eta",
    "cov_log_lambda_mu",
    "cov_log_lambda_eta",
    "cov_log_mu_eta",
    ]},
    dims={"level_2": ["param"]}
)

# Plot traceplots for both models
az.plot_trace(idata_m1, var_names=["level_2"], figsize=(12, 6))
plt.suptitle("Traceplot - M1", fontsize=14)
plt.tight_layout()
plt.show()

az.plot_trace(idata_m2, var_names=["level_2"], figsize=(12, 10))
plt.suptitle("Traceplot - M2", fontsize=14)
plt.tight_layout()
plt.show()

# %%
