"""
Out-of-time validation for synthetic difference-in-differences.

Fits the synthdid model on a pre-period and evaluates its counterfactual
predictions on a separate predict period. The predict period need not be
contiguous with or immediately follow the pre period.

The counterfactual prediction for the treated average at time t is:

    predicted(t) = omega @ (Y - X_beta)[:N0, t]
                 + X_beta_treated_avg(t)
                 + intercept

where the intercept aligns the synthetic control level to the treated level
using the lambda-weighted pre-period:

    intercept = mean_treated_pre(lambda) - omega @ mean_control_pre(lambda)

This is the same formula as synthdid_effect_curve, but expressed as a level
rather than a difference, allowing standard regression-style evaluation metrics.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .estimator import synthdid_estimate, SynthdidEstimate, _contract3
from .inference import vcov


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OOTResult:
    """
    Container for out-of-time validation results.

    Attributes
    ----------
    actual : ndarray, shape (T_predict,)
        Average of treated units in the predict period (raw Y scale).
    predicted : ndarray, shape (T_predict,)
        Synthetic control counterfactual for the treated average.
    residuals : ndarray, shape (T_predict,)
        actual - predicted.
    actual_by_unit : ndarray, shape (N1, T_predict)
        Individual treated unit outcomes in the predict period.
    pre_cols : list of int
        Column indices used for fitting.
    predict_cols : list of int
        Column indices used for evaluation.
    time_names_pre : list
        Time labels for the pre period.
    time_names_predict : list
        Time labels for the predict period.
    metrics : dict
        Regression-style performance metrics.
    estimate : SynthdidEstimate
        The fitted synthdid model (weights, setup, etc.).
    placebo_tau : float
        The synthdid point estimate treating the predict period as post-treatment.
        Should be close to zero in a valid pre-treatment holdout test.
    placebo_se : float
        Standard error of placebo_tau (via placebo method).
    placebo_pvalue : float
        Two-sided p-value for H0: placebo_tau = 0. Should be > 0.05 for a
        well-specified model on a pre-treatment holdout period.
    """
    actual: np.ndarray
    predicted: np.ndarray
    residuals: np.ndarray
    actual_by_unit: np.ndarray
    pre_cols: List[int]
    predict_cols: List[int]
    time_names_pre: list
    time_names_predict: list
    metrics: dict
    estimate: SynthdidEstimate
    placebo_tau: float = 0.0
    placebo_se: float = float("nan")
    placebo_pvalue: float = float("nan")

    def __repr__(self):
        m = self.metrics
        sig = "✓ not significant" if self.placebo_pvalue > 0.05 else "✗ significant!"
        return (
            f"OOTResult(\n"
            f"  pre_periods     = {self.time_names_pre[0]}..{self.time_names_pre[-1]} "
            f"({len(self.pre_cols)} periods)\n"
            f"  predict_periods = {self.time_names_predict[0]}..{self.time_names_predict[-1]} "
            f"({len(self.predict_cols)} periods)\n"
            f"  RMSE={m['RMSE']:.3f}, MAE={m['MAE']:.3f}, "
            f"R²={m['R2']:.3f}, Bias={m['Bias']:.3f}\n"
            f"  Placebo τ={self.placebo_tau:.3f}, SE={self.placebo_se:.3f}, "
            f"p={self.placebo_pvalue:.3f}  {sig}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def synthdid_out_of_time(
    Y,
    N0,
    pre_periods,
    predict_periods,
    X=None,
    unit_names=None,
    time_names=None,
    se_method="placebo",
    se_replications=200,
    **synthdid_kwargs,
):
    """
    Fit a synthdid model on a pre-period and evaluate counterfactual predictions
    on a separate predict period.

    The model is fitted using only the pre-period columns. The predict period
    is then used purely for out-of-sample evaluation — its values play no role
    in estimating the weights.

    Parameters
    ----------
    Y : ndarray, shape (N, T)
        Full outcome matrix (all units, all time periods).
    N0 : int
        Number of control units (first N0 rows).
    pre_periods : array-like of int, or slice
        Column indices of Y to use for fitting (pre-period).
        E.g. range(0, 19), [0,1,...,18], or a list of non-contiguous indices.
    predict_periods : array-like of int, or slice
        Column indices of Y to evaluate predictions on.
        These need not be contiguous with or follow pre_periods.
    X : ndarray, shape (N, T, C) or None
        Optional time-varying covariate array. If supplied, it is subsetted
        to match the pre and predict columns.
    unit_names : list or None
        Unit identifiers (row labels of Y).
    time_names : list or None
        Time identifiers (column labels of Y). Used for axis labels.
    se_method : {'placebo', 'bootstrap', 'jackknife'}
        Method for computing the SE of the placebo treatment effect.
        Default 'placebo'. Use 'jackknife' for speed when N1 > 1.
    se_replications : int
        Number of replications for placebo/bootstrap SE. Default 200.
    **synthdid_kwargs
        Additional keyword arguments passed to synthdid_estimate
        (e.g. eta_omega, omega_intercept, do_sparsify).

    Returns
    -------
    OOTResult
        Contains actual values, predictions, residuals, per-unit actuals,
        period indices, time labels, fitted estimate, performance metrics,
        and placebo significance test (tau, SE, p-value).
    """
    Y = np.asarray(Y, dtype=float)
    N, T = Y.shape
    N1 = N - N0

    # Resolve period specs to sorted lists of column indices
    pre_cols = _resolve_periods(pre_periods, T)
    predict_cols = _resolve_periods(predict_periods, T)

    if len(set(pre_cols) & set(predict_cols)) > 0:
        raise ValueError("pre_periods and predict_periods must not overlap.")
    if len(pre_cols) < 2:
        raise ValueError("pre_periods must contain at least 2 time periods.")
    if len(predict_cols) < 1:
        raise ValueError("predict_periods must contain at least 1 time period.")

    # Build a combined matrix [pre | predict] for synthdid_estimate.
    # T0_fit = len(pre_cols); the predict columns act as the "post-treatment"
    # period so the algorithm can estimate lambda weights (which use the
    # control post-period average as the regression target).
    # Crucially, treated values in the predict period are NOT used in
    # weight estimation — only control units drive lambda/omega fitting.
    all_cols = pre_cols + predict_cols
    Y_fit = Y[:, all_cols]
    T0_fit = len(pre_cols)

    # Subset X if provided
    X_fit = None
    if X is not None:
        X = np.asarray(X)
        if X.ndim == 3:
            X_fit = X[:, all_cols, :]
        # else ignore

    # Time name subsets
    tnames_pre = [time_names[c] for c in pre_cols] if time_names else list(pre_cols)
    tnames_predict = [time_names[c] for c in predict_cols] if time_names else list(predict_cols)

    # Fit the model
    est = synthdid_estimate(
        Y_fit, N0, T0_fit,
        X=X_fit,
        unit_names=unit_names,
        time_names=list(tnames_pre) + list(tnames_predict),
        **synthdid_kwargs,
    )

    # -------------------------------------------------------------------------
    # Compute the counterfactual prediction for the predict period
    #
    # Full formula (handles covariates):
    #   Y_adj = Y - X_beta
    #   intercept = mean_treated_pre(lambda) - omega @ mean_control_pre(lambda)
    #   predicted(t) = omega @ Y_adj[:N0, t] + intercept + X_beta_treated_avg(t)
    # -------------------------------------------------------------------------
    weights = est.weights
    omega = weights["omega"]          # (N0,)
    lambda_ = weights["lambda"]       # (T0_fit,)

    # Covariate-adjusted outcomes (Y_adj = Y - X_beta)
    X_beta_fit = _contract3(est.setup["X"], weights["beta"])  # (N, T_pre+T_predict)
    Y_adj = Y_fit - X_beta_fit

    # Pre-period: lambda-weighted averages
    Y_adj_pre = Y_adj[:, :T0_fit]                              # (N, T0_fit)
    mean_control_pre = omega @ Y_adj_pre[:N0, :] @ lambda_     # scalar
    mean_treated_pre = Y_adj_pre[N0:, :].mean(axis=0) @ lambda_  # scalar

    intercept = mean_treated_pre - mean_control_pre

    # Predict period columns in Y_fit (the last len(predict_cols) columns)
    Y_adj_predict = Y_adj[:, T0_fit:]                       # (N, T_predict)
    X_beta_predict = X_beta_fit[:, T0_fit:]                 # (N, T_predict)

    # Synthetic control: omega @ adjusted control outcomes in predict period
    synth_control = omega @ Y_adj_predict[:N0, :]           # (T_predict,)

    # Add back covariate contribution for treated average
    X_beta_treated_avg = X_beta_predict[N0:, :].mean(axis=0)  # (T_predict,)

    predicted = synth_control + intercept + X_beta_treated_avg  # (T_predict,)

    # Actual treated average (raw Y scale)
    actual = Y[N0:, predict_cols].mean(axis=0)              # (T_predict,)

    # Per-unit actuals (raw Y)
    actual_by_unit = Y[N0:, :][:, predict_cols]             # (N1, T_predict)

    residuals = actual - predicted

    metrics = _compute_metrics(actual, predicted)

    # -------------------------------------------------------------------------
    # Placebo significance test
    #
    # The synthdid point estimate on [pre | predict] treats the predict period
    # as post-treatment. In a valid pre-treatment holdout, this should be close
    # to zero and not statistically significant (p > 0.05).
    # -------------------------------------------------------------------------
    placebo_tau = float(est)
    try:
        placebo_se_val = float(np.sqrt(vcov(est, method=se_method,
                                            replications=se_replications)))
        # Two-sided p-value: P(|Z| > |tau/se|) under H0: tau = 0
        z_stat = placebo_tau / placebo_se_val if placebo_se_val > 0 else np.nan
        from scipy.special import ndtr  # standard normal CDF
        placebo_pvalue = float(2 * (1 - ndtr(abs(z_stat))))
    except Exception:
        placebo_se_val = float("nan")
        placebo_pvalue = float("nan")

    return OOTResult(
        actual=actual,
        predicted=predicted,
        residuals=residuals,
        actual_by_unit=actual_by_unit,
        pre_cols=pre_cols,
        predict_cols=predict_cols,
        time_names_pre=tnames_pre,
        time_names_predict=tnames_predict,
        metrics=metrics,
        estimate=est,
        placebo_tau=placebo_tau,
        placebo_se=placebo_se_val,
        placebo_pvalue=placebo_pvalue,
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def synthdid_oot_plot(result, show_units=True, figsize=(13, 10)):
    """
    Plot out-of-time validation results.

    Layout:
      - Top row    : actual vs predicted trajectory (full width)
      - Bottom-left: residuals bar chart
      - Bottom-right: performance metrics table

    Parameters
    ----------
    result : OOTResult
        Output of synthdid_out_of_time().
    show_units : bool
        If True and there are multiple treated units, plot each unit's
        trajectory as a faint line alongside the average.
    figsize : tuple
        Figure size in inches. Default (13, 10).

    Returns
    -------
    matplotlib.figure.Figure
    """
    actual = result.actual
    predicted = result.predicted
    residuals = result.residuals
    t_labels = [str(t) for t in result.time_names_predict]
    T_predict = len(actual)
    x = np.arange(T_predict)

    fig = plt.figure(figsize=figsize)
    # Two rows: main plot on top, residuals + metrics side-by-side on bottom
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[2.2, 1.4],
        hspace=0.55,
        wspace=0.35,
    )

    ax_main  = fig.add_subplot(gs[0, :])    # spans both columns
    ax_resid = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    # -------------------------------------------------------------------------
    # Main panel: trajectories
    # -------------------------------------------------------------------------
    N1 = result.actual_by_unit.shape[0]
    if show_units and N1 > 1:
        for i in range(N1):
            unit_label = (result.estimate.unit_names[result.estimate.setup["N0"] + i]
                          if result.estimate.unit_names else f"Unit {i+1}")
            ax_main.plot(x, result.actual_by_unit[i], color="black",
                         alpha=0.2, linewidth=1, linestyle="--",
                         label=unit_label if i == 0 else "_")
        ax_main.plot([], [], color="black", alpha=0.3, linewidth=1,
                     linestyle="--", label="Individual treated units")

    ax_main.plot(x, actual, color="black", linewidth=2.5,
                 marker="o", markersize=5, label="Actual (treated avg)")
    ax_main.plot(x, predicted, color="steelblue", linewidth=2.5,
                 marker="s", markersize=5, linestyle="--",
                 label="Predicted (synthetic control)")
    ax_main.fill_between(x, actual, predicted, alpha=0.12, color="steelblue",
                         label="Prediction gap")

    ax_main.set_title("Out-of-Time Validation: Actual vs Predicted",
                      fontsize=13, fontweight="bold", pad=10)
    ax_main.set_ylabel("Outcome", fontsize=10)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(t_labels, rotation=45, ha="right", fontsize=9)
    ax_main.legend(fontsize=9, loc="best", framealpha=0.8)
    ax_main.axhline(np.mean(actual), color="gray", linestyle=":",
                    linewidth=0.8, alpha=0.5)
    ax_main.grid(axis="y", alpha=0.3)

    # Placebo test annotation in top-right corner of main panel
    tau = result.placebo_tau
    se  = result.placebo_se
    pv  = result.placebo_pvalue
    sig_str = "p > 0.05  ✓" if pv > 0.05 else "p ≤ 0.05  ✗"
    color   = "green" if pv > 0.05 else "red"
    annot = (
        f"Synthdid placebo test\n"
        f"τ = {tau:.2f}  (SE = {se:.2f})\n"
        f"p = {pv:.3f}   {sig_str}"
    )
    ax_main.text(
        0.98, 0.05, annot,
        transform=ax_main.transAxes,
        ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=color, linewidth=1.5, alpha=0.9),
        color=color,
    )

    # -------------------------------------------------------------------------
    # Bottom-left: residuals bar chart
    # -------------------------------------------------------------------------
    colors = ["tomato" if r > 0 else "steelblue" for r in residuals]
    ax_resid.bar(x, residuals, color=colors, alpha=0.75, width=0.6)
    ax_resid.axhline(0, color="black", linewidth=1)
    ax_resid.axhline(
        np.mean(residuals), color="gray", linewidth=1.2, linestyle="--",
        label=f"Bias = {np.mean(residuals):.2f}",
    )
    ax_resid.set_title("Residuals (Actual − Predicted)", fontsize=10,
                       fontweight="bold")
    ax_resid.set_ylabel("Residual", fontsize=9)
    ax_resid.set_xticks(x)
    ax_resid.set_xticklabels(t_labels, rotation=45, ha="right", fontsize=8)
    ax_resid.legend(fontsize=8)
    ax_resid.grid(axis="y", alpha=0.3)

    # -------------------------------------------------------------------------
    # Bottom-right: metrics table
    # -------------------------------------------------------------------------
    ax_table.axis("off")
    m = result.metrics
    pv = result.placebo_pvalue
    metrics_data = [
        ["RMSE",          f"{m['RMSE']:.4f}"],
        ["MAE",           f"{m['MAE']:.4f}"],
        ["MAPE",          f"{m['MAPE']:.2f}%"],
        ["R²",            f"{m['R2']:.4f}"],
        ["Bias",          f"{m['Bias']:.4f}"],
        ["Max Abs Error", f"{m['MaxAbsError']:.4f}"],
        ["N periods",     str(m['N'])],
        ["— Placebo test —", ""],
        ["Synthdid τ",    f"{result.placebo_tau:.4f}"],
        ["SE",            f"{result.placebo_se:.4f}"],
        ["p-value",       f"{pv:.3f}  {'✓' if pv > 0.05 else '✗'}"],
    ]
    col_labels = ["Metric", "Value"]

    tbl = ax_table.table(
        cellText=metrics_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0.05, 0.0, 0.9, 1.0],   # [left, bottom, width, height] in axes coords
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row shading; special style for the section divider row
    divider_row = 8  # "— Placebo test —" is at index 7 in metrics_data → row 8 in table
    for i in range(1, len(metrics_data) + 1):
        if i == divider_row:
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor("#D0D8F0")
                tbl[i, j].set_text_props(fontweight="bold", fontstyle="italic")
        else:
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor("#EEF2FF" if i % 2 == 0 else "white")

    # Colour the p-value row green/red
    pv_row = len(metrics_data)  # last row
    pv_color = "#D4EDDA" if result.placebo_pvalue > 0.05 else "#F8D7DA"
    for j in range(len(col_labels)):
        tbl[pv_row, j].set_facecolor(pv_color)

    ax_table.set_title("Performance Metrics", fontsize=10,
                       fontweight="bold", pad=8)

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_periods(periods, T):
    """
    Convert a period spec to a sorted list of integer column indices.

    Accepts: range, list/array of ints, or a slice.
    """
    if isinstance(periods, slice):
        return list(range(*periods.indices(T)))
    periods = list(periods)
    if not all(isinstance(p, (int, np.integer)) for p in periods):
        raise ValueError("Period indices must be integers. "
                         "To use time labels, resolve them to column indices first.")
    periods_sorted = sorted(set(int(p) for p in periods))
    if any(p < 0 or p >= T for p in periods_sorted):
        raise ValueError(f"Period indices must be in [0, {T-1}].")
    return periods_sorted


def _compute_metrics(actual, predicted):
    """
    Compute regression-style performance metrics.

    Returns
    -------
    dict with keys: RMSE, MAE, MAPE, R2, Bias, MaxAbsError, N
    """
    n = len(actual)
    residuals = actual - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    # MAPE: skip periods where actual == 0 to avoid division by zero
    nonzero = actual != 0
    mape = np.mean(np.abs(residuals[nonzero] / actual[nonzero])) * 100 if nonzero.any() else np.nan

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    bias = np.mean(residuals)
    max_abs = np.max(np.abs(residuals))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "Bias": bias,
        "MaxAbsError": max_abs,
        "N": n,
    }
