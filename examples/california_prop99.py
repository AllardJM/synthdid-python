"""
Synthetic Difference-in-Differences: California Proposition 99 Example
=======================================================================

Replicates the R synthdid starter code in Python:

    library(synthdid)
    data('california_prop99')
    setup = panel.matrices(california_prop99)
    tau.hat = synthdid_estimate(setup$Y, setup$N0, setup$T0)
    se = sqrt(vcov(tau.hat, method='placebo'))
    sprintf('point estimate: %1.2f', tau.hat)
    sprintf('95%% CI (%1.2f, %1.2f)', tau.hat - 1.96 * se, tau.hat + 1.96 * se)
    plot(tau.hat)

Dataset: California Prop 99 (tobacco tax, 1989). Outcome is cigarette
consumption (packs per capita). Treatment effect should be approximately
-15 to -16 packs per capita.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving to file
import matplotlib.pyplot as plt

# Add the project root to the path so we can import synthdid
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synthdid import (
    panel_matrices,
    synthdid_estimate,
    vcov,
    synthdid_effect_curve,
    synthdid_controls,
    synthdid_plot,
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "syntheticdid", "synthdid", "data", "california_prop99.csv",
)

print("=" * 60)
print("Synthetic Difference-in-Differences: California Prop 99")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=None, engine="python")  # auto-detect separator
print(f"\nData loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head())

# ---------------------------------------------------------------------------
# 2. Convert to matrix format (equivalent to panel.matrices() in R)
# ---------------------------------------------------------------------------
setup = panel_matrices(
    df,
    unit="State",
    time="Year",
    outcome="PacksPerCapita",
    treatment="treated",
)

Y = setup["Y"]
N0 = setup["N0"]
T0 = setup["T0"]
unit_names = setup["unit_names"]
time_names = [str(t) for t in setup["time_names"]]

print(f"\nMatrix dimensions: {Y.shape[0]} units × {Y.shape[1]} time periods")
print(f"Control units (N0): {N0}")
print(f"Pre-treatment periods (T0): {T0}")
print(f"Treated unit(s): {unit_names[N0:]}")
print(f"Treatment starts: {time_names[T0]}")

# ---------------------------------------------------------------------------
# 3. Estimate treatment effect (equivalent to synthdid_estimate() in R)
# ---------------------------------------------------------------------------
print("\n--- Estimating treatment effect ---")
tau_hat = synthdid_estimate(
    Y, N0, T0,
    unit_names=unit_names,
    time_names=time_names,
)
print(f"Point estimate: {float(tau_hat):.2f}")

# ---------------------------------------------------------------------------
# 4. Compute standard error via placebo method (Algorithm 4)
# ---------------------------------------------------------------------------
print("\n--- Computing SE via placebo method (200 replications) ---")
np.random.seed(42)
variance = vcov(tau_hat, method="placebo", replications=200)
se = np.sqrt(variance)
print(f"SE (placebo): {se:.2f}")

ci_lo = float(tau_hat) - 1.96 * se
ci_hi = float(tau_hat) + 1.96 * se
print(f"95% CI: ({ci_lo:.2f}, {ci_hi:.2f})")

# ---------------------------------------------------------------------------
# 5. Period-by-period treatment effect curve
# ---------------------------------------------------------------------------
print("\n--- Period-by-period effect curve (post-treatment years) ---")
effect_curve = synthdid_effect_curve(tau_hat)
post_years = time_names[T0:]
print(pd.Series(effect_curve, index=post_years, name="tau(t)").round(2).to_string())

# ---------------------------------------------------------------------------
# 6. Top control units by weight
# ---------------------------------------------------------------------------
print("\n--- Top control units by synthetic weight (omega) ---")
controls_df = synthdid_controls(tau_hat, weight_type="omega", mass=0.9)
print(controls_df.round(4).to_string())

print("\n--- Top pre-treatment periods by weight (lambda) ---")
lambda_df = synthdid_controls(tau_hat, weight_type="lambda", mass=0.9)
print(lambda_df.round(4).to_string())

# ---------------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------------
print("\n--- Generating plot ---")
fig = synthdid_plot(tau_hat, se=se, figsize=(12, 7))
plot_path = os.path.join(os.path.dirname(__file__), "output", "synthdid_plot.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {plot_path}")
plt.close(fig)

print("\n" + "=" * 60)
print("Done.")
