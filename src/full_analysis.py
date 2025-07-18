# ------------------------------------------------------------------
# --------------- this script contains -----------------------------
# --------- the analysis on the CDNOW dataset ----------------------
# ------- with both bivariate and trivariate models ----------------
# ------------------------------------------------------------------
# -----------------------------------------------------------------

# %% 1. Import necessary libraries
# -- 1. Import necessary libraries --
#import os
import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import xarray as xr
import arviz as az
from IPython.display import display

# Add lifetimes ParetoNBDFitter for MLE baseline
from lifetimes import ParetoNBDFitter

# Set up the project root directory
cwd = os.getcwd()
while not os.path.isdir(os.path.join(cwd, 'src')):
    parent = os.path.dirname(cwd)
    if parent == cwd:
        break  # Reached the root of the filesystem
    cwd = parent
project_root = cwd
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.bivariate.mcmc import draw_future_transactions
from src.models.utils.analysis_bi_helpers import (summarize_level2, 
                                                  post_mean_lambdas, 
                                                  post_mean_mus, 
                                                  mape_aggregate, 
                                                  extract_correlation, 
                                                  chain_total_loglik, 
                                                  compute_table4)
from src.models.utils.analysis_display_helper import _fmt

# Path to save the figures:
save_figures_path = os.path.join(project_root, "outputs", "figures", "x_comparison_four_models")
plots_path = os.path.join(save_figures_path, "plots")
os.makedirs(plots_path, exist_ok=True)

# ---------------------------------------------------------------------

# %% 2. Load estimated parameters, data and set file path
# -- 2. Load estimated parameters, data and set file path

# Set up the directory for pickles
pickles_dir = os.path.join(project_root, "outputs", "pickles")

# Set Excel output path
excel_path = os.path.join(project_root, "outputs", "excel", "x_comparison_four_models.xlsx")
os.makedirs(os.path.dirname(excel_path), exist_ok=True)

# Set folder to save Figures 
figure_path = os.path.join(project_root, "outputs", "figures", "x_comparison_four_models")
os.makedirs(figure_path, exist_ok=True)

# Load Estimates
# Bivariate
with open(os.path.join(pickles_dir, "ext_bi_m1.pkl"), "rb") as f:
    bi_m1 = pickle.load(f)
with open(os.path.join(pickles_dir, "ext_bi_m2.pkl"), "rb") as f:
    bi_m2 = pickle.load(f)

# Trivariate
with open(os.path.join(pickles_dir, "ext_tri_m1.pkl"), "rb") as f:
    tri_m1 = pickle.load(f)
with open(os.path.join(pickles_dir, "ext_tri_m2.pkl"), "rb") as f:
    tri_m2 = pickle.load(f)

# CBS data
cbs_path = os.path.join(project_root, "data", "processed", "cdnow_abeCBS.csv")
cbs = pd.read_csv(cbs_path, dtype={"cust": str}, parse_dates=["first"])

# Load Elog data
data_path = os.path.join(project_root, "data", "raw", "cdnow_abeElog.csv")
cdnowElog = pd.read_csv(data_path)
# Convert date column to datetime
cdnowElog["date"] = pd.to_datetime(cdnowElog["date"])
# ensure the same key type
cdnowElog["cust"] = cdnowElog["cust"].astype(str)
# -------------------------------------------------

# %% 3. Descriptive statistics
# -- 3. Descriptive statistics --
table1_stats = pd.DataFrame(
    {
        "Mean": [
            cbs["x"].mean(),
            cbs["T_cal"].mean() * 7,  # weeks to days
            (cbs["T_cal"] - cbs["t_x"]).mean() * 7,  # weeks to days
            cdnowElog.groupby("cust")["sales"].first().mean()
        ],
        "Std. dev.": [
            cbs["x"].std(),
            cbs["T_cal"].std() * 7,
            (cbs["T_cal"] - cbs["t_x"]).std() * 7,
            cdnowElog.groupby("cust")["sales"].first().std()
        ],
        "Min": [
            cbs["x"].min(),
            cbs["T_cal"].min() * 7,
            (cbs["T_cal"] - cbs["t_x"]).min() * 7,
            cdnowElog.groupby("cust")["sales"].first().min()
        ],
        "Max": [
            cbs["x"].max(),
            cbs["T_cal"].max() * 7,
            (cbs["T_cal"] - cbs["t_x"]).max() * 7,
            cdnowElog.groupby("cust")["sales"].first().max()
        ],
    },
    index=[
        "Number of repeats",
        "Observation duration T (days)",
        "Recency (T - t) (days)",
        "Amount of initial purchase ($)"
    ]
)

print("Table 1. Descriptive Statistics for CDNOW dataset")
print(table1_stats.round(2))
display(table1_stats)

# Save the DataFrame to the Excel file
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
    table1_stats.to_excel(writer, sheet_name="Table_1_DescriptStats", index=True)


# %% 4. Posterior Summary
# -- 4. Posterior Summary --

# Function to summarize level 2 draws
def summarize_level2(draws_level2: np.ndarray, param_names: list[str], decimals: int = 2) -> pd.DataFrame:
    quantiles = np.percentile(draws_level2, [2.5, 50, 97.5], axis=0)
    summary = pd.DataFrame(quantiles.T, columns=["2.5%", "50%", "97.5%"], index=param_names)
    return summary.round(decimals)

# BIVARIATE
# Parameter names for Model 1 (M1): no covariates
param_names_bi_m1 = [
    "log_lambda (intercept)",
    "log_mu (intercept)",
    "var_log_lambda",
    "var_log_mu",
    "cov_log_lambda_mu"
]

# Parameter names for Model 2 (M2): with covariate "first.sales"
param_names_bi_m2 = [
    "log_lambda (intercept)",
    "log_lambda (first.sales)",
    "log_lambda (gender)",
    "log_lambda (age)",
    "log_mu (intercept)",
    "log_mu (first.sales)",
    "log_mu (gender)",
    "log_mu (age)",
    "var_log_lambda",
    "var_log_mu",
    "cov_log_lambda_mu"
]
# TRIVARIATE 
# Parameter names for Model 1 (M1): no covariates
param_names_tri_m1 = [
    "log_lambda (intercept)",
    "log_mu (intercept)",
    "log_eta (intercept)",
    "var_log_lambda",
    "var_log_mu",
    "var_log_eta",
    "cov_log_lambda_mu",
    "cov_log_lambda_eta",
    "cov_log_mu_eta"
]

# Parameter names for Model 2 (M2): with covariate "first.sales"
param_names_tri_m2 = [
    "log_lambda (intercept)",
    "log_lambda (age)",
    "log_lambda (gender)",
    "log_mu (intercept)",
    "log_mu (gender)",
    "log_mu (age)",
    "log_eta (intercept)",
    "log_eta (gender)",
    "log_eta (age)",
    "var_log_lambda",
    "var_log_mu",
    "var_log_eta",
    "cov_log_lambda_mu",
    "cov_log_lambda_eta",
    "cov_log_mu_eta"
]

# Compute summaries
summary_bi_m1 = summarize_level2(bi_m1["level_2"][0], param_names=param_names_bi_m1)
summary_bi_m2 = summarize_level2(bi_m2["level_2"][0], param_names=param_names_bi_m2)

summary_tri_m1 = summarize_level2(tri_m1["level_2"][0], param_names=param_names_tri_m1)
summary_tri_m2 = summarize_level2(tri_m2["level_2"][0], param_names=param_names_tri_m2)

# Drop "MAE" row if present
summary_bi_m1 = summary_bi_m1.drop(index="MAE", errors="ignore")
summary_bi_m2 = summary_bi_m2.drop(index="MAE", errors="ignore")

summary_tri_m1 = summary_tri_m1.drop(index="MAE", errors="ignore")
summary_tri_m2 = summary_tri_m2.drop(index="MAE", errors="ignore")

# Rename indices to match Table 3 from the paper
summary_bi_m1.index = [
    "Purchase rate log(λ) - Intercept",
    "Dropout rate log(μ) - Intercept",
    "sigma^2_λ = var[log λ]",
    "sigma^2_μ = var[log μ]",
    "sigma_λ_μ = cov[log λ, log μ]"
] # type: ignore
summary_bi_m2.index = [
    "Purchase rate log(λ) - Intercept",
    "Purchase rate log(λ) - Initial amount ($ 10^-3)",
    "Purchase rate log(λ) - Gender [1 = Male]",
    "Purchase rate log(λ) - Age (scaled)",
    "Dropout rate log(μ) - Intercept",
    "Dropout rate log(μ) - Initial amount ($ 10^-3)",
    "Dropout rate log(μ) - Gender [1 = Male]",
    "Dropout rate log(μ) - Age (scaled)",
    "sigma^2_λ = var[log λ]",
    "sigma^2_μ = var[log μ]",
    "sigma_λ_μ = cov[log λ, log μ]"
] # type: ignore
summary_tri_m1.index = [
    "Purchase rate log(λ) - Intercept",
    "Dropout rate log(μ) - Intercept",
    "Spending log(η) - Intercept",
    "sigma^2_λ = var[log λ]",
    "sigma^2_μ = var[log μ]",
    "sigma^2_η = var[log η]",
    "sigma_λ_μ = cov[log λ, log μ]",
    "sigma_λ_η = cov[log λ, log η]",
    "sigma_μ_η = cov[log μ, log η]",
] # type: ignore
summary_tri_m2.index = [
    "Purchase rate log(λ) - Intercept",
    "Purchase rate log(λ) - Gender [1 = Male]",
    "Purchase rate log(λ) - Age (scaled)",
    "Dropout rate log(μ) - Intercept",
    "Dropout rate log(μ) - Gender [1 = Male]",
    "Dropout rate log(μ) - Age (scaled)",
    "Spending log(η) - Intercept",
    "Spending log(η) - Gender [1 = Male]",
    "Spending log(η) - Age (scaled)",
    "sigma^2_λ = var[log λ]",
    "sigma^2_μ = var[log μ]",
    "sigma^2_η = var[log η]",
    "sigma_λ_μ = cov[log λ, log μ]",
    "sigma_λ_η = cov[log λ, log η]",
    "sigma_μ_η = cov[log μ, log η]",
] # type: ignore

# ------------------------------------------------------------------

# Compute posterior means of λ and μ
def post_mean_lambdas(draws):
    all_draws = np.concatenate(draws["level_1"], axis=0)
    return all_draws[:, :, 0].mean(axis=0)

def post_mean_mus(draws):
    all_draws = np.concatenate(draws["level_1"], axis=0)
    return all_draws[:, :, 1].mean(axis=0)

def post_mean_etas(draws):
    all_draws = np.concatenate(draws["level_1"], axis=0)
    return all_draws[:, :, 2].mean(axis=0)

# Closed-form expected x_star for validation
t_star = 39.0
mean_lambda_bi_m1 = post_mean_lambdas(bi_m1)
mean_mu_bi_m1     = post_mean_mus(bi_m1)
mean_lambda_bi_m2 = post_mean_lambdas(bi_m2)
mean_mu_bi_m2     = post_mean_mus(bi_m2)
mean_lambda_tri_m1 = post_mean_lambdas(tri_m1)
mean_mu_tri_m1     = post_mean_mus(tri_m1)
mean_lambda_tri_m2 = post_mean_lambdas(tri_m2)
mean_mu_tri_m2     = post_mean_mus(tri_m2)

cbs["xstar_bi_m1_pred"] = (mean_lambda_bi_m1/mean_mu_bi_m1) * (1 - np.exp(-mean_mu_bi_m1 * t_star))
cbs["xstar_bi_m2_pred"] = (mean_lambda_bi_m2/mean_mu_bi_m2) * (1 - np.exp(-mean_mu_bi_m2 * t_star))
cbs["xstar_tri_m1_pred"] = (mean_lambda_tri_m1/mean_mu_tri_m1) * (1 - np.exp(-mean_mu_tri_m1 * t_star))
cbs["xstar_tri_m2_pred"] = (mean_lambda_tri_m2/mean_mu_tri_m2) * (1 - np.exp(-mean_mu_tri_m2 * t_star))

# Compare MAE
mae_bi_m1 = np.mean(np.abs(cbs["x_star"] - cbs["xstar_bi_m1_pred"]))
mae_bi_m2 = np.mean(np.abs(cbs["x_star"] - cbs["xstar_bi_m2_pred"]))

mae_tri_m1 = np.mean(np.abs(cbs["x_star"] - cbs["xstar_tri_m1_pred"]))
mae_tri_m2 = np.mean(np.abs(cbs["x_star"] - cbs["xstar_tri_m2_pred"]))

# Display both
print("Posterior Summary - Bivariate Model M1 (no covariates):")
print(summary_bi_m1)

print("Posterior Summary - Bivariate Model M2 (with covariates):")
print(summary_bi_m2)

print("Posterior Summary - Trivariate Model M1 (no covariates):")
print(summary_tri_m1)

print("Posterior Summary - Trivariate Model M2 (with covariates):")
print(summary_tri_m2)

# Save posterior summaries to separate sheets in the Excel file
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    summary_bi_m1.to_excel(writer, sheet_name="PostSummary_BI_M1")
    summary_bi_m2.to_excel(writer, sheet_name="PostSummary_BI_M2")
    summary_tri_m1.to_excel(writer, sheet_name="PostSummary_TRI_M1")
    summary_tri_m2.to_excel(writer, sheet_name="PostSummary_TRI_M2")


# %% 5. Model Fit Evaluation Bivariate Models
# -- 5. Model Fit Evaluation Bivariate Models --
# BI VARIATE MODEL
mean_lambda_m1 = post_mean_lambdas(bi_m1)
mean_mu_m1     = post_mean_mus(bi_m1)
mean_lambda_m2 = post_mean_lambdas(bi_m2)
mean_mu_m2     = post_mean_mus(bi_m2)

cbs["xstar_bi_m1_pred"] = (mean_lambda_m1/mean_mu_m1) * (1 - np.exp(-mean_mu_m1 * t_star))
cbs["xstar_bi_m2_pred"] = (mean_lambda_m2/mean_mu_m2) * (1 - np.exp(-mean_mu_m2 * t_star))

# Compute marginal log-likelihood for bivariate models
ll_bi_m1 = chain_total_loglik(bi_m1["level_1"], cbs)
ll_bi_m2 = chain_total_loglik(bi_m2["level_1"], cbs)

# Prepare weekly index and counts
first_date = cdnowElog["date"].min()
cdnowElog["week"] = ((cdnowElog["date"] - first_date) // pd.Timedelta("7D")).astype(int) + 1

# Set the time range for the analysis
max_week = cdnowElog["week"].max()

times = np.arange(1, max_week + 1)

# Sort the data by customer and week
cdnowElog_sorted = cdnowElog.sort_values(by=["cust","week"])
cdnowElog_sorted["txn_order"] = cdnowElog_sorted.groupby("cust").cumcount()

repeat_txns = cdnowElog_sorted[cdnowElog_sorted["txn_order"] >= 1]

weekly_actual = (
    repeat_txns.groupby("week")["cust"].count()
    .reindex(range(1, max_week+1), fill_value = 0))
# -----------------------------------------------------------------------

# TRI VARIATE MODEL
cbs_df = cbs
birth_week = (
    cdnowElog.groupby("cust")["week"].min()
    .reindex(cbs["cust"])
    .to_numpy()
)
weeks_cal_mask = (times >= 1)  & (times <= 39)
weeks_val_mask = (times >= 40) & (times <= 78)

# Compute marginal log-likelihood for trivariate models
ll_tri_m1 = chain_total_loglik(tri_m1["level_1"], cbs_df)
ll_tri_m2 = chain_total_loglik(tri_m2["level_1"], cbs_df)
print(f"Marginal log-likelihood Trivariate M1: {ll_tri_m1}")
print(f"Marginal log-likelihood Trivariate M2: {ll_tri_m2}")

# -----------------------------------------------------------------------
# 3) Metric helper ------------------------------------------------------
# -----------------------------------------------------------------------
def compute_metrics(draws: dict, label: str) -> dict[str, float]:
    """Return correlation / MSE / MAPE metrics for RFM–M draws."""
    all_d = np.concatenate(draws["level_1"], axis=0)       # (D, N, 5)
    lam   = all_d[:, :, 0].mean(axis=0)
    mu    = all_d[:, :, 1].mean(axis=0)
    # Models with an explicit 'alive probability' store it at index 3.
    # Bi‑variate models don’t have that parameter, so default to 1.
    if all_d.shape[2] > 3:
        z = all_d[:, :, 3].mean(axis=0)
    else:
        z = np.ones_like(lam)

    t_star = 39
    xstar_pred = z * (lam / mu) * (1 - np.exp(-mu * t_star))

    corr_val = np.corrcoef(cbs_df["x_star"], xstar_pred)[0, 1]
    mse_val  = np.mean((cbs_df["x_star"] - xstar_pred) ** 2)

    calib_pred = (lam / mu) * (1 - np.exp(-mu * cbs_df["T_cal"]))
    corr_cal = np.corrcoef(cbs_df["x"], calib_pred)[0, 1]
    mse_cal  = np.mean((cbs_df["x"] - calib_pred) ** 2)

    # weekly posterior-mean increments
    inc_weekly = np.zeros_like(times, dtype=float)
    n_draws, draws_per_chain = all_d.shape[0], len(draws["level_1"][0])

    for d in range(n_draws):
        ch, idx = divmod(d, draws_per_chain)
        lam_d = draws["level_1"][ch][idx, :, 0]
        mu_d  = draws["level_1"][ch][idx, :, 1]
        tau_d = draws["level_1"][ch][idx, :, 2]

        rng = np.random.default_rng(d)
        for t_idx, t in enumerate(times):
            active = (t > birth_week) & (t <= (birth_week + tau_d))
            inc_weekly[t_idx] += rng.poisson(lam=lam_d * active).sum()

    inc_weekly /= n_draws
    weekly_arr  = weekly_actual.to_numpy()

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
stats_bi1  = compute_metrics(bi_m1,  "Bivariate HB-M1")
stats_bi2  = compute_metrics(bi_m2,  "Bivariate HB-M2")
stats_tri1 = compute_metrics(tri_m1, "Trivariate HB-M1")
stats_tri2 = compute_metrics(tri_m2, "Trivariate HB-M2")

# -----------------------------------------------------------------------
# 5) Assemble two-column Table 2 ----------------------------------------
table2 = pd.DataFrame({
    stats_bi1["label"]: [
        ll_bi_m1,
        stats_bi1["corr_val"], stats_bi1["corr_cal"],
        stats_bi1["mse_val"], stats_bi1["mse_cal"],
        stats_bi1["mape_val"], stats_bi1["mape_cal"], stats_bi1["mape_pool"]
    ],
    stats_bi2["label"]: [
        ll_bi_m2,
        stats_bi2["corr_val"], stats_bi2["corr_cal"],
        stats_bi2["mse_val"], stats_bi2["mse_cal"],
        stats_bi2["mape_val"], stats_bi2["mape_cal"], stats_bi2["mape_pool"]
    ],
    stats_tri1["label"]: [
        ll_tri_m1,
        stats_tri1["corr_val"], stats_tri1["corr_cal"],
        stats_tri1["mse_val"], stats_tri1["mse_cal"],
        stats_tri1["mape_val"], stats_tri1["mape_cal"], stats_tri1["mape_pool"]
    ],
    stats_tri2["label"]: [
        ll_tri_m2,
        stats_tri2["corr_val"], stats_tri2["corr_cal"],
        stats_tri2["mse_val"], stats_tri2["mse_cal"],
        stats_tri2["mape_val"], stats_tri2["mape_cal"], stats_tri2["mape_pool"]
    ]
}, index=[
    "Marginal log-likelihood",
    "Correlation (Validation)", "Correlation (Calibration)",
    "MSE (Validation)",         "MSE (Calibration)",
    "MAPE (Validation)",        "MAPE (Calibration)", "MAPE (Pooled)"
]).round(2)

row_order = [
    "Disaggregate measure",
    "Correlation (Validation)", "Correlation (Calibration)", "",
    "MSE (Validation)",         "MSE (Calibration)",         "",
    "Aggregate measure", "Time-series MAPE (%)",
    "MAPE (Validation)",        "MAPE (Calibration)",        "MAPE (Pooled)",
        "Marginal log-likelihood"
]
table2 = table2.reindex(row_order)

table2_disp = _fmt(table2.reset_index().rename(columns={"index": ""}), 2)
print("\nModel Fit Metrics - Bivariate *and* Trivariate")
display(table2_disp)

with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    table2_disp.to_excel(writer, sheet_name="ModelFit_Table", index=False, float_format="%.2f")
# ------------------------------------------------------------------

# %% 6. Weekly-Series Tracking
# -- 6. Weekly-Series Tracking --
cum_actual = weekly_actual.cumsum()

 # --- Birth‑aligned Pareto/NBD baseline (MLE) -----------------
# first purchase week for each customer
birth_week = (
    cdnowElog.groupby("cust")["week"].min()
    .reindex(cbs["cust"])
    .to_numpy()
)

# Fit classical Pareto/NBD by maximum likelihood
pnbd_mle = ParetoNBDFitter(penalizer_coef=0.0)
pnbd_mle.fit(
    frequency=cbs["x"],
    recency=cbs["t_x"],
    T=cbs["T_cal"]
)
# Classical Pareto/NBD (MLE) expected future repeats for the next 39 weeks
exp_xstar_m1 = pnbd_mle.conditional_expected_number_of_purchases_up_to_time(
    t_star,
    cbs["x"],
    cbs["t_x"],
    cbs["T_cal"]
)

# --- Compute classical Pareto/NBD cumulative curve if missing ----
if "cum_pnbd_ml" not in globals():
    # birth week of each customer (first purchase)
    birth_week = (
        cdnowElog.groupby("cust")["week"].min()
        .reindex(cbs["cust"])
        .to_numpy()
    )
    cum_pnbd_ml = np.zeros_like(times, dtype=float)
    for t_idx, t in enumerate(times):
        rel_t = np.clip(t - birth_week, 0, None)
        exp_per_cust = pnbd_mle.expected_number_of_purchases_up_to_time(rel_t)
        cum_pnbd_ml[t_idx] = exp_per_cust.sum()
        
# Weekly PNB (MLE) increments
inc_pnbd_weekly = np.empty_like(times, dtype=float)
inc_pnbd_weekly[0] = cum_pnbd_ml[0]
inc_pnbd_weekly[1:] = np.diff(cum_pnbd_ml)
for t_idx, t in enumerate(times):
    # time since first purchase (≥0) for each customer
    rel_t = np.clip(t - birth_week, 0, None)
    exp_per_cust = pnbd_mle.expected_number_of_purchases_up_to_time(rel_t)
    cum_pnbd_ml[t_idx] = exp_per_cust.sum()

# --- Posterior‑predictive HB curve -----------------------------------------
# Reset inc_hb_weekly to zero for Figure 2 calculation
inc_hb_weekly_bi_m1 = np.zeros_like(times, dtype=float)
inc_hb_weekly_bi_m2 = np.zeros_like(times, dtype=float)
inc_hb_weekly_tri_m1 = np.zeros_like(times, dtype=float)
inc_hb_weekly_tri_m2 = np.zeros_like(times, dtype=float)

# Predictive draws for validation period
xstar_bi_m1_draws = draw_future_transactions(cbs, bi_m1, T_star=t_star, seed=42)
xstar_bi_m2_draws = draw_future_transactions(cbs, bi_m2, T_star=t_star, seed=42)
xstar_tri_m1_draws = draw_future_transactions(cbs, tri_m1, T_star=t_star, seed=42)
xstar_tri_m2_draws = draw_future_transactions(cbs, tri_m2, T_star=t_star, seed=42)

n_bi_m1_draws = len(xstar_bi_m1_draws)
n_bi_m2_draws = len(xstar_bi_m2_draws)
n_tri_m1_draws = len(xstar_tri_m1_draws)
n_tri_m2_draws = len(xstar_tri_m2_draws)

for d in range(n_bi_m1_draws):
    # map flat draw index `d` to (chain, draw) indices
    draws_per_chain = len(bi_m1["level_1"][0])
    chain = d // draws_per_chain
    idx   = d % draws_per_chain
    lam_d = bi_m1["level_1"][chain][idx, :, 0]
    mu_d  = bi_m1["level_1"][chain][idx, :, 1]
    tau_d = bi_m1["level_1"][chain][idx, :, 2]

    rng_d = np.random.default_rng(d)  # reproducible per draw
    for t_idx, t in enumerate(times):
        dt = 1.0
        active = (t > birth_week) & (t <= (birth_week + tau_d))   # after first purchase, before churn
        inc = rng_d.poisson(lam=lam_d * dt * active)
        inc_hb_weekly_bi_m1[t_idx] += inc.sum()
# average across draws and take cumulative
inc_hb_weekly_bi_m1 /= n_bi_m1_draws
cum_hb_bi_m1 = np.cumsum(inc_hb_weekly_bi_m1)

for d in range(n_bi_m2_draws):
    # map flat draw index `d` to (chain, draw) indices
    draws_per_chain = len(bi_m2["level_1"][0])
    chain = d // draws_per_chain
    idx   = d % draws_per_chain
    lam_d = bi_m2["level_1"][chain][idx, :, 0]
    mu_d  = bi_m2["level_1"][chain][idx, :, 1]
    tau_d = bi_m2["level_1"][chain][idx, :, 2]

    rng_d = np.random.default_rng(d)  # reproducible per draw
    for t_idx, t in enumerate(times):
        dt = 1.0
        active = (t > birth_week) & (t <= (birth_week + tau_d))   # after first purchase, before churn
        inc = rng_d.poisson(lam=lam_d * dt * active)
        inc_hb_weekly_bi_m2[t_idx] += inc.sum()
# average across draws and take cumulative
inc_hb_weekly_bi_m2 /= n_bi_m2_draws
cum_hb_bi_m2 = np.cumsum(inc_hb_weekly_bi_m2)

for d in range(n_tri_m1_draws):
    # map flat draw index `d` to (chain, draw) indices
    draws_per_chain = len(tri_m1["level_1"][0])
    chain = d // draws_per_chain
    idx   = d % draws_per_chain
    lam_d = tri_m1["level_1"][chain][idx, :, 0]
    mu_d  = tri_m1["level_1"][chain][idx, :, 1]
    tau_d = tri_m1["level_1"][chain][idx, :, 2]

    rng_d = np.random.default_rng(d)  # reproducible per draw
    for t_idx, t in enumerate(times):
        dt = 1.0
        active = (t > birth_week) & (t <= (birth_week + tau_d))   # after first purchase, before churn
        inc = rng_d.poisson(lam=lam_d * dt * active)
        inc_hb_weekly_tri_m1[t_idx] += inc.sum()
# average across draws and take cumulative
inc_hb_weekly_tri_m1 /= n_tri_m1_draws
cum_hb_tri_m1 = np.cumsum(inc_hb_weekly_tri_m1)

for d in range(n_tri_m2_draws):
    # map flat draw index `d` to (chain, draw) indices
    draws_per_chain = len(tri_m2["level_1"][0])
    chain = d // draws_per_chain
    idx   = d % draws_per_chain
    lam_d = tri_m2["level_1"][chain][idx, :, 0]
    mu_d  = tri_m2["level_1"][chain][idx, :, 1]
    tau_d = tri_m2["level_1"][chain][idx, :, 2]

    rng_d = np.random.default_rng(d)  # reproducible per draw
    for t_idx, t in enumerate(times):
        dt = 1.0
        active = (t > birth_week) & (t <= (birth_week + tau_d))   # after first purchase, before churn
        inc = rng_d.poisson(lam=lam_d * dt * active)
        inc_hb_weekly_tri_m2[t_idx] += inc.sum()
# average across draws and take cumulative
inc_hb_weekly_tri_m2 /= n_tri_m2_draws
cum_hb_tri_m2 = np.cumsum(inc_hb_weekly_tri_m2)

plt.figure(figsize=(8,5))
plt.plot(times, cum_actual, '-', color='tab:blue', linewidth=2, label="Actual")
plt.plot(times, cum_pnbd_ml, '--', color='tab:orange', linewidth=2, label="Pareto/NBD (MLE)")
plt.plot(times, cum_hb_bi_m1, ':', color='tab:olive', linewidth=2, label="HB Bivariate M1")
plt.plot(times, cum_hb_bi_m2, ':', color='tab:green', linewidth=2, label="HB Bivariate M2")
plt.plot(times, cum_hb_tri_m1, '-.', color='tab:purple', linewidth=2, label="HB Trivariate M1")
plt.plot(times, cum_hb_tri_m2, '*-', color='tab:pink', linewidth=2, label="HB Trivariate M2")
plt.axvline(x=int(t_star), color='k', linestyle='--')
plt.xlabel("Week")
plt.ylabel("Cumulative repeat transactions")
plt.title("Figure 2: Weekly Time-Series Tracking for CDNOW Data")
plt.legend()
plt.savefig(os.path.join(plots_path, "Weekly_Tracking.png"), dpi=300, bbox_inches='tight')
plt.show()
# ------------------------------------------------------------------

# %% 7. Conditional expectation of future transactions
# -- 7. Conditional expectation of future transactions

# Group by number of calibration transactions (0–7+)
# Use analytical expectations, with different formulas for Pareto/NBD (M1) and HB (M2)

# Expected future repeats for Figure 3:
all_draws_bi_m1 = np.concatenate(bi_m1["level_1"], axis=0)
all_draws_bi_m2 = np.concatenate(bi_m2["level_1"], axis=0)
all_draws_tri_m1 = np.concatenate(tri_m1["level_1"], axis=0)
all_draws_tri_m2 = np.concatenate(tri_m2["level_1"], axis=0)

mean_lambda_bi_m1_cust = all_draws_bi_m1[:, :, 0].mean(axis=0)
mean_mu_bi_m1_cust     = all_draws_bi_m1[:, :, 1].mean(axis=0)
mean_z_bi_m1_cust      = all_draws_bi_m1[:, :, 3].mean(axis=0)

mean_lambda_bi_m2_cust = all_draws_bi_m2[:, :, 0].mean(axis=0)
mean_mu_bi_m2_cust     = all_draws_bi_m2[:, :, 1].mean(axis=0)
mean_z_bi_m2_cust      = all_draws_bi_m2[:, :, 3].mean(axis=0)

mean_lambda_tri_m1_cust = all_draws_tri_m1[:, :, 0].mean(axis=0)
mean_mu_tri_m1_cust     = all_draws_tri_m1[:, :, 1].mean(axis=0)
mean_z_tri_m1_cust      = all_draws_tri_m1[:, :, 3].mean(axis=0)

mean_lambda_tri_m2_cust = all_draws_tri_m2[:, :, 0].mean(axis=0)
mean_mu_tri_m2_cust     = all_draws_tri_m2[:, :, 1].mean(axis=0)
mean_z_tri_m2_cust      = all_draws_tri_m2[:, :, 3].mean(axis=0)


# HB expectation (Model M2) – include posterior P(alive)
exp_xstar_bi_m1 = mean_z_bi_m1_cust * (mean_lambda_bi_m1_cust / mean_mu_bi_m1_cust) * (1 - np.exp(-mean_mu_bi_m1_cust * t_star))
exp_xstar_bi_m2 = mean_z_bi_m2_cust * (mean_lambda_bi_m2_cust / mean_mu_bi_m2_cust) * (1 - np.exp(-mean_mu_bi_m2_cust * t_star))

exp_xstar_tri_m1 = mean_z_tri_m1_cust * (mean_lambda_tri_m1_cust / mean_mu_tri_m1_cust) * (1 - np.exp(-mean_mu_tri_m1_cust * t_star))
exp_xstar_tri_m2 = mean_z_tri_m2_cust * (mean_lambda_tri_m2_cust / mean_mu_tri_m2_cust) * (1 - np.exp(-mean_mu_tri_m2_cust * t_star))

# Classical Pareto/NBD (MLE) expected future repeats for the next 39 weeks
exp_xstar_m1 = pnbd_mle.conditional_expected_number_of_purchases_up_to_time(
    t_star,
    cbs["x"],
    cbs["t_x"],
    cbs["T_cal"]
)

df = pd.DataFrame({
    "x":      cbs["x"],
    "actual": cbs["x_star"],
    "pnbd":   exp_xstar_m1,   # Pareto/NBD expectation (no P(alive))
    "hb_bi_m1":     exp_xstar_bi_m1, 
    "hb_bi_m2":     exp_xstar_bi_m2,    
    "hb_tri_m1":     exp_xstar_tri_m1,   
    "hb_tri_m2":     exp_xstar_tri_m2         
})
groups = []
for k in range(7):
    grp = df[df["x"]==k]
    groups.append((
        str(k),
        grp["actual"].mean(),
        grp["pnbd"].mean(),
        grp["hb_bi_m1"].mean(),
        grp["hb_bi_m2"].mean(),
        grp["hb_tri_m1"].mean(),
        grp["hb_tri_m2"].mean()
    ))
grp7 = df[df["x"]>=7]
groups.append((
    "7+",
    grp7["actual"].mean(),
    grp7["pnbd"].mean(),
    grp7["hb_bi_m1"].mean(),
    grp7["hb_bi_m2"].mean(),
    grp7["hb_tri_m1"].mean(),
    grp7["hb_tri_m2"].mean(),
))
cond_df = pd.DataFrame(groups, columns=["x","Actual","Pareto/NBD", "HB Bivariate M1","HB Bivariate M2", "HB Trivariate M1","HB Trivariate M2" ]).set_index("x")

plt.figure(figsize=(8,5))
plt.plot(cond_df.index, cond_df["Actual"], '-', color='tab:blue', linewidth=2, label="Actual")
plt.plot(cond_df.index, cond_df["Pareto/NBD"], marker='*', linestyle='--', color='tab:orange', linewidth=2, label="Pareto/NBD")
plt.plot(cond_df.index, cond_df["HB Bivariate M1"], marker='x', linestyle=':', color='tab:olive', linewidth=2, label="HB Bivariate M1")
plt.plot(cond_df.index, cond_df["HB Bivariate M2"], marker='+', linestyle=':', color='tab:green', linewidth=2, label="HB Bivariate M2")
plt.plot(cond_df.index, cond_df["HB Trivariate M1"], marker='*', linestyle='-', color='tab:purple', linewidth=2, label="HB Bivariate M1")
plt.plot(cond_df.index, cond_df["HB Trivariate M2"], marker='', linestyle='-', color='tab:pink', linewidth=2, label="HB Bivariate M2")
plt.xlabel("Number of transactions in weeks 1-39")
plt.ylabel("Average transactions in weeks 40-78")
plt.title("Conditional Expectation of Future Transactions for CDNOW Data")
plt.legend()
plt.savefig(os.path.join(plots_path, "Conditional_Expectation.png"), dpi=300, bbox_inches='tight')
plt.show()
# ------------------------------------------------------------------

# %% 8. Scatterplot - Prediction of all models
# -- 8. Scatterplot - Prediction of all models
import seaborn as sns
sns.set(style="whitegrid")

# --- posterior mean alive‑probabilities for trivariate models
mean_z_tri_m1 = np.concatenate(tri_m1["level_1"], axis=0)[:, :, 3].mean(axis=0)
mean_z_tri_m2 = np.concatenate(tri_m2["level_1"], axis=0)[:, :, 3].mean(axis=0)

cbs["xstar_bi_m1_pred"]  = (mean_lambda_bi_m1 / mean_mu_bi_m1)  * (1 - np.exp(-mean_mu_bi_m1  * t_star))
cbs["xstar_bi_m2_pred"]  = (mean_lambda_bi_m2 / mean_mu_bi_m2)  * (1 - np.exp(-mean_mu_bi_m2  * t_star))
cbs["xstar_tri_m1_pred"] = mean_z_tri_m1 * (mean_lambda_tri_m1 / mean_mu_tri_m1) * (1 - np.exp(-mean_mu_tri_m1 * t_star))
cbs["xstar_tri_m2_pred"] = mean_z_tri_m2 * (mean_lambda_tri_m2 / mean_mu_tri_m2) * (1 - np.exp(-mean_mu_tri_m2 * t_star))

# Create a figure with four subplots (flattened 2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

# BI VARIATE ------------+
# Scatterplot for Model M1
axes[0].scatter(cbs["x_star"], cbs["xstar_bi_m1_pred"], alpha=0.4, color="tab:blue")
axes[0].plot([0, cbs["x_star"].max()], [0, cbs["x_star"].max()], 'r--')
axes[0].set_title("Bi M1: Without Covariates")
axes[0].set_xlabel("Actual x_star")
axes[0].set_ylabel("Predicted x_star")

# Scatterplot for Model M2
axes[1].scatter(cbs["x_star"], cbs["xstar_bi_m2_pred"], alpha=0.4, color="tab:green")
axes[1].plot([0, cbs["x_star"].max()], [0, cbs["x_star"].max()], 'r--')
axes[1].set_title("Bi M2: With first.sales")
axes[1].set_xlabel("Actual x_star")
# -----------

# TRI VARIATE -----------+
# Scatterplot for Model M1
axes[2].scatter(cbs["x_star"], cbs["xstar_tri_m1_pred"], alpha=0.4, color="tab:blue")
axes[2].plot([0, cbs["x_star"].max()], [0, cbs["x_star"].max()], 'r--')
axes[2].set_title("Tri M1: Without Covariates")
axes[2].set_xlabel("Actual x_star")
axes[2].set_ylabel("Predicted x_star")

# Scatterplot for Model M2
axes[3].scatter(cbs["x_star"], cbs["xstar_tri_m2_pred"], alpha=0.4, color="tab:green")
axes[3].plot([0, cbs["x_star"].max()], [0, cbs["x_star"].max()], 'r--')
axes[3].set_title("Tri M2: With age & gender")
axes[3].set_xlabel("Actual x_star")

# Remove grid from both subplots
for ax in axes:
    ax.grid(False)

plt.tight_layout()
plt.savefig(os.path.join(plots_path, "Scatter_Prediction.png"), dpi=300, bbox_inches='tight')
plt.show()

# %% 9. Alive vs Churned customers
# -- 9. Alive vs Churned customers
# ------------------------------------------------------------------
# Alive vs. Churned – four HB models in one 2×2 barplot grid
# ------------------------------------------------------------------
# Binary alive prediction (≥ 1 repeat forecast in weeks 40–78)
cbs["bi_m1_alive_pred"]  = (cbs["xstar_bi_m1_pred"]  >= 1).astype(int)
cbs["bi_m2_alive_pred"]  = (cbs["xstar_bi_m2_pred"]  >= 1).astype(int)
cbs["tri_m1_alive_pred"] = (cbs["xstar_tri_m1_pred"] >= 1).astype(int)
cbs["tri_m2_alive_pred"] = (cbs["xstar_tri_m2_pred"] >= 1).astype(int)

model_cols = {
    "Bivariate M1": "bi_m1_alive_pred",
    "Bivariate M2": "bi_m2_alive_pred",
    "Trivariate M1": "tri_m1_alive_pred",
    "Trivariate M2": "tri_m2_alive_pred",
}

labels  = ["Churned (z = 0)", "Alive (z = 1)"]
colors  = ["#d3d3d3", "#4a90e2"]   # light grey / business blue

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
axes = axes.flatten()

for ax, (title, col) in zip(axes, model_cols.items()):
    counts = cbs[col].value_counts().sort_index()
    bars = ax.bar(labels, counts, color=colors, width=0.55)

    # Axis styling
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Number of customers")
    ax.grid(False)
    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#999999')
    ax.tick_params(axis='y', colors='#444444')
    ax.tick_params(axis='x', colors='#444444')

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + max(counts) * 0.02,
                f"{int(height)}",
                ha='center', va='bottom', fontsize=9)

plt.suptitle("Predicted Alive vs. Churned Customers - Four HB Models",
             fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(plots_path, "Alive_vs_Churned.png"),
            dpi=300, bbox_inches="tight")
plt.show()
# -------------------------------------------------------------------

# %% 10. Traceplots
# -- 10. Traceplots
plots_path = os.path.join(save_figures_path, "SamplingPlots")

# Convert Bivariate M1 to InferenceData with full dims
lvl2_bi_m1 = np.array(bi_m1["level_2"])
idata_bi_m1 = az.from_dict(
    posterior={"level_2": lvl2_bi_m1},
    coords={
        "chain": np.arange(lvl2_bi_m1.shape[0]),
        "draw":  np.arange(lvl2_bi_m1.shape[1]),
        "param": param_names_bi_m1
    },
    dims={"level_2": ["chain", "draw", "param"]}
)

# Convert Bivariate M2 to InferenceData with full dims
lvl2_bi_m2 = np.array(bi_m2["level_2"])
idata_bi_m2 = az.from_dict(
    posterior={"level_2": lvl2_bi_m2},
    coords={
        "chain": np.arange(lvl2_bi_m2.shape[0]),
        "draw":  np.arange(lvl2_bi_m2.shape[1]),
        "param": param_names_bi_m2
    },
    dims={"level_2": ["chain", "draw", "param"]}
)

# Convert Trivariate M1 to InferenceData with full dims
lvl2_tri_m1 = np.array(tri_m1["level_2"])
idata_tri_m1 = az.from_dict(
    posterior={"level_2": lvl2_tri_m1},
    coords={
        "chain": np.arange(lvl2_tri_m1.shape[0]),
        "draw":  np.arange(lvl2_tri_m1.shape[1]),
        "param": param_names_tri_m1
    },
    dims={"level_2": ["chain", "draw", "param"]}
)

# Convert Trivariate M2 to InferenceData with full dims
lvl2_tri_m2 = np.array(tri_m2["level_2"])
idata_tri_m2 = az.from_dict(
    posterior={"level_2": lvl2_tri_m2},
    coords={
        "chain": np.arange(lvl2_tri_m2.shape[0]),
        "draw":  np.arange(lvl2_tri_m2.shape[1]),
        "param": param_names_tri_m2
    },
    dims={"level_2": ["chain", "draw", "param"]}
)

# Traceplots
for idata, label in [
    (idata_bi_m1, "HB Bivariate M1"),
    (idata_bi_m2, "HB Bivariate M2"),
    (idata_tri_m1, "HB Trivariate M1"),
    (idata_tri_m2, "HB Trivariate M2"),
]:
    plt.close('all')
    n_params   = len(idata.posterior["level_2"].coords["param"])
    fig_height = max(4, n_params * 1.5)
    fig_width  = max(12, n_params * 2)   # keep width ≥ height for readability
    az.plot_trace(idata, var_names=["level_2"], figsize=(fig_width, fig_height))
    plt.suptitle(f"Traceplot - {label}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"Traceplot_{label.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Autocorrelation plots
for idata, label in [
    (idata_bi_m1, "HB Bivariate M1"),
    (idata_bi_m2, "HB Bivariate M2"),
    (idata_tri_m1, "HB Trivariate M1"),
    (idata_tri_m2, "HB Trivariate M2"),
]:
    plt.close('all')
    az.plot_autocorr(idata, var_names=["level_2"], figsize=(12, max(4, len(idata.posterior["level_2"].coords["param"]) * 1.5)))
    plt.suptitle(f"Autocorrelation - {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"Autocorr_{label.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Posterior distributions
for idata, label in [
    (idata_bi_m1, "HB Bivariate M1"),
    (idata_bi_m2, "HB Bivariate M2"),
    (idata_tri_m1, "HB Trivariate M1"),
    (idata_tri_m2, "HB Trivariate M2"),
]:
    plt.close('all')
    n_params = len(idata.posterior["level_2"].coords["param"])
    az.plot_posterior(
        idata,
        var_names=["level_2"],
        figsize=(8, n_params * 2),
        hdi_prob=0.95,
        kind='kde',
        grid=(n_params, 1)
    )
    plt.suptitle(f"Posterior Distributions - {label}", fontsize=16, y=1.02)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(plots_path, f"PosteriorDist_{label.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
# ------------------------------------------------------------------

# %% 11. BIVARIATE Convergence 
# -- 11. BIVARIATE Convergence 

def level2_summary(draws: dict, label: str = "m1") -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1) ensure NumPy array
    # ------------------------------------------------------------------
    lvl2 = np.asarray(draws["level_2"])    # list → ndarray
    if lvl2.ndim == 2:                     # single chain: (draws, P)
        lvl2 = lvl2[None, ...]             # → (1, draws, P)

    chains, draws_n, P = lvl2.shape
    # ------------------------------------------------------------------
    # 2) build dynamic parameter names (unchanged below)
    # ------------------------------------------------------------------
    n_cov = max((P - 5) // 2, 0)
    param_names = ["log_lambda (intercept)", "log_mu (intercept)"]
    for i in range(n_cov):
        param_names.extend([f"beta_lambda[{i}]", f"beta_mu[{i}]"])
    param_names.extend(
        ["var_log_lambda", "cov_log_lambda_mu", "var_log_mu"]
    )

    da = xr.DataArray(
        lvl2,
        dims=("chain", "draw", "param"),
        coords={"param": param_names},
    )

    summary = az.summary(
        az.from_dict(posterior={"level_2": da}),
        var_names=["level_2"],
        round_to=4,
    )
    summary.index = [f"level_2[{p}]" for p in param_names]
    return summary

summary_m1 = level2_summary(bi_m1, label="m1")
print(summary_m1)

# Always compute and print Bivariate M2 convergence
summary_m2 = level2_summary(bi_m2, label="m2")
print("\n" + "-" * 60)
print(summary_m2)

# Save bivariate convergence summaries to separate sheets
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    summary_m1.to_excel(writer, sheet_name="Bi_Convergence_M1", index=True)
    summary_m2.to_excel(writer, sheet_name="Bi_Convergence_M2", index=True)
# ------------------------------------------------------------------

# %% 12. TRIVARIATE Convergence
# -- 12. TRIVARIATE Convergence

def level2_summary_trivar(draws: dict, label: str = "M1") -> pd.DataFrame:
    """
    Return az.summary table (mean | sd | hdi_3% | … | r_hat) for the
    3-intercept + 6 unique covariance parameters in the tri-variate model.

    The function ignores any extra β rows that belong to covariates.
    """
    # Convert list of per-chain arrays into a single 3D array
    lvl2 = np.stack(draws["level_2"], axis=0)
    n_chain, n_draw, P = lvl2.shape
    # Determine number of covariates: each adds 3 β parameters
    n_cov = max((P - 9) // 3, 0)
    # Build dynamic parameter names: three intercepts, βs, then covariance elements
    param_names = [
        "log_lambda (intercept)",
        "log_mu (intercept)",
        "log_eta (intercept)"
    ]
    for i in range(n_cov):
        param_names.extend([
            f"beta_lambda[{i}]",
            f"beta_mu[{i}]",
            f"beta_eta[{i}]"
        ])
    param_names.extend([
        "var_log_lambda",
        "cov_log_lambda_mu",
        "cov_log_lambda_eta",
        "var_log_mu",
        "cov_log_mu_eta",
        "var_log_eta"
    ])
    da = xr.DataArray(
        lvl2,
        dims=("chain", "draw", "param"),
        coords={"param": param_names},
    )
    idata = az.from_dict(posterior={"level_2": da})
    summary = az.summary(idata, var_names=["level_2"], round_to=4)
    # Align row labels with parameter naming
    summary.index = [f"level_2[{p}]" for p in param_names]
    return summary

# ------- run for both models ---------------------------------------------
summary_3pI  = level2_summary_trivar(tri_m1,  label="M1")
summary_3pII = level2_summary_trivar(tri_m2, label="M2")

print("\nArviZ level-2 summary - Model 1 (intercept only)")
print(summary_3pI)
print("\nArviZ level-2 summary - Model 2 (with covariates)")
print(summary_3pII)

# Save trivariate convergence summaries to separate sheets
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    summary_3pI.to_excel(writer, sheet_name="Tri_Convergence_M1", index=True)
    summary_3pII.to_excel(writer, sheet_name="Tri_Convergence_M2", index=True)
# ------------------------------------------------------------------

# %%
