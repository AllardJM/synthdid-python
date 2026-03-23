"""
Summary and diagnostic utilities for synthetic difference-in-differences.

Ported from R/synthdid.R (synthdid_effect_curve) and R/summary.R
(synthdid_controls) in the synthdid package by Arkhangelsky et al.
Reference: https://arxiv.org/abs/1812.09970
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estimator import SynthdidEstimate, _contract3


@dataclass
class EffectCurveDetail:
    """
    Detailed output of synthdid_effect_curve(detail=True).

    Attributes
    ----------
    tau : ndarray, shape (T1,)
        Period-by-period treatment effect: actual - predicted.
    actual : ndarray, shape (T1,)
        Average of treated units in each post-treatment period (raw Y).
    predicted : ndarray, shape (T1,)
        Synthetic control counterfactual for the treated average:
            predicted(t) = omega @ Y_adj[control, t]  (synth control level)
                         + intercept                   (pre-treatment alignment)
                         + X_beta_treated_avg(t)       (covariate adjustment)
    intercept : float
        Lambda-weighted pre-treatment gap between the treated average and the
        synthetic control. Subtracted from the post-treatment gap to remove
        pre-existing level differences — this is the DID component that
        distinguishes SynthDiD from plain synthetic control.
    time_names : list
        Time labels for the post-treatment periods.
    """
    tau: np.ndarray
    actual: np.ndarray
    predicted: np.ndarray
    intercept: float
    time_names: list

    def __repr__(self):
        return (
            f"EffectCurveDetail(\n"
            f"  periods   = {self.time_names[0]}..{self.time_names[-1]}\n"
            f"  tau       = [{', '.join(f'{v:.2f}' for v in self.tau)}]\n"
            f"  actual    = [{', '.join(f'{v:.2f}' for v in self.actual)}]\n"
            f"  predicted = [{', '.join(f'{v:.2f}' for v in self.predicted)}]\n"
            f"  intercept = {self.intercept:.4f}\n"
            f")"
        )


def synthdid_effect_curve(estimate, detail=False):
    """
    Compute the period-by-period treatment effect curve.

    For each post-treatment period t, returns the difference between the
    observed treated average and its synthetic control counterfactual,
    adjusted for pre-treatment trends via the estimated lambda weights.

    Decomposition:
        Y_adj(t)  = Y(t) - X_beta(t)
                    covariate-adjusted outcome matrix

        gap(t)    = treated_avg(t) - omega @ control(t)   [on Y_adj]
                    cross-sectional gap between treated and synthetic control
                    at each time period t (= [-omega, 1/N1] @ Y_adj(t))

        intercept = lambda @ gap[:T0]
                    lambda-weighted pre-treatment gap; captures the pre-existing
                    level difference between treated and synthetic control

        tau(t)    = gap(T0+t) - intercept
                    treatment effect at post-treatment period t; the DID removes
                    the pre-existing gap so only the causal effect remains

    When detail=True, also returns the levels (not just differences):
        actual(t)    = mean(Y[treated, T0+t])           (raw treated average)
        predicted(t) = omega @ Y_adj[control, T0+t]     (synth control level)
                       + intercept                       (pre-treatment alignment)
                       + X_beta_treated_avg(T0+t)        (covariate part)
        tau(t)       = actual(t) - predicted(t)          (same as above)

    Parameters
    ----------
    estimate : SynthdidEstimate
        Output of synthdid_estimate().
    detail : bool
        If False (default), return a plain ndarray of treatment effects —
        identical to the original behaviour, fully backwards compatible.
        If True, return an EffectCurveDetail object with tau, actual,
        predicted, intercept, and time_names.

    Returns
    -------
    ndarray, shape (T1,)
        When detail=False (default).
    EffectCurveDetail
        When detail=True.
    """
    setup = estimate.setup
    weights = estimate.weights

    Y = setup["Y"]
    N0, T0 = setup["N0"], setup["T0"]
    N1 = Y.shape[0] - N0
    T1 = Y.shape[1] - T0

    # Covariate adjustment
    X_beta = _contract3(setup["X"], weights["beta"])
    Y_adj = Y - X_beta

    # Synthetic control trajectory across all periods: shape (T,)
    w_unit = np.concatenate([-weights["omega"], np.ones(N1) / N1])
    tau_sc = w_unit @ Y_adj

    # Lambda-weighted pre-treatment intercept (the DID level correction)
    intercept = float(tau_sc[:T0] @ weights["lambda"])

    # Period-by-period treatment effect
    tau_curve = tau_sc[T0:] - intercept

    if not detail:
        return tau_curve

    # --- detailed decomposition ---
    # actual: raw treated average (Y scale, not adjusted)
    actual = Y[N0:, T0:].mean(axis=0)

    # predicted: synthetic control counterfactual on the Y scale
    #   = omega @ Y_adj[control, post] + intercept + X_beta_treated_avg
    synth_control = weights["omega"] @ Y_adj[:N0, T0:]       # (T1,)
    X_beta_treated_avg = X_beta[N0:, T0:].mean(axis=0)       # (T1,)
    predicted = synth_control + intercept + X_beta_treated_avg

    time_names = (
        estimate.time_names[T0:] if estimate.time_names
        else list(range(T0, T0 + T1))
    )

    return EffectCurveDetail(
        tau=tau_curve,
        actual=actual,
        predicted=predicted,
        intercept=intercept,
        time_names=time_names,
    )


def synthdid_controls(estimates, sort_by=0, mass=0.9, top_n=None, weight_type="omega"):
    """
    Summarize the most important control units or time periods by weight.

    Returns a table of omega (unit) or lambda (time) weights, sorted in
    descending order and truncated to the smallest set covering at least
    `mass` fraction of total weight.

    Parameters
    ----------
    estimates : SynthdidEstimate or list of SynthdidEstimate
        One or more estimates to compare.
    sort_by : int
        Index of the estimate to sort by. Default 0.
    mass : float
        Minimum cumulative weight to retain. Default 0.9 (top 90% of weight).
        Ignored when top_n is provided.
    top_n : int or None
        If provided, return exactly this many rows (or fewer if there are fewer
        units/periods). Takes precedence over mass.
    weight_type : {'omega', 'lambda'}
        Whether to summarize unit weights ('omega') or time weights ('lambda').

    Returns
    -------
    pd.DataFrame
        Rows = control units or time periods (sorted by descending weight),
        Columns = estimate labels.
        Truncated so that cumulative weight >= mass for each estimate.
    """
    if isinstance(estimates, SynthdidEstimate):
        estimates = [estimates]

    if weight_type not in ("omega", "lambda"):
        raise ValueError("weight_type must be 'omega' or 'lambda'.")

    # Collect weights from each estimate: shape (n_units_or_times, n_estimates)
    all_weights = np.column_stack([
        est.weights[weight_type] for est in estimates
    ])
    if all_weights.ndim == 1:
        all_weights = all_weights[:, np.newaxis]

    # Build row labels from the first estimate's setup
    if weight_type == "omega":
        N0 = estimates[0].setup["N0"]
        unit_names = estimates[0].unit_names
        labels = unit_names[:N0] if unit_names else [f"unit_{i}" for i in range(N0)]
    else:
        # Time labels for pre-treatment periods
        time_names = estimates[0].time_names
        T0 = estimates[0].setup["T0"]
        if time_names:
            labels = time_names[:T0]
        else:
            labels = [f"t_{i}" for i in range(all_weights.shape[0])]

    # Sort by the chosen estimate's weights (descending)
    sort_col = all_weights[:, sort_by]
    order = np.argsort(sort_col)[::-1]
    sorted_weights = all_weights[order]
    sorted_labels = [labels[i] for i in order]

    # Truncate: top_n overrides mass-based truncation
    if top_n is not None:
        tab_len = min(top_n, len(sorted_labels))
    else:
        def _min_prefix(col):
            cumsum = np.cumsum(col)
            idx = np.searchsorted(cumsum, mass, side="left")
            return min(idx + 1, len(col))

        tab_len = max(_min_prefix(sorted_weights[:, j]) for j in range(sorted_weights.shape[1]))

    tab = sorted_weights[:tab_len]
    labels_trunc = sorted_labels[:tab_len]

    # Column names: use estimate index if no names available
    col_names = [f"estimate {i+1}" for i in range(len(estimates))]

    return pd.DataFrame(tab, index=labels_trunc, columns=col_names)
