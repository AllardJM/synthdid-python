"""
Summary and diagnostic utilities for synthetic difference-in-differences.

Ported from R/synthdid.R (synthdid_effect_curve) and R/summary.R
(synthdid_controls) in the synthdid package by Arkhangelsky et al.
Reference: https://arxiv.org/abs/1812.09970
"""

import numpy as np
import pandas as pd

from .estimator import SynthdidEstimate, _contract3


def synthdid_effect_curve(estimate):
    """
    Compute the period-by-period treatment effect curve.

    For each post-treatment period t, this gives the difference between
    the observed treated average and its synthetic control counterfactual,
    adjusted for pre-treatment trends using the estimated lambda weights.

    Mathematically:
        tau_sc(t) = [-omega^T, (1/N1)·1^T] @ Y   for all T periods
        tau_curve(t) = tau_sc(T0 + t) - tau_sc[0:T0] @ lambda
                       for t = 1, ..., T1

    Parameters
    ----------
    estimate : SynthdidEstimate
        Output of synthdid_estimate().

    Returns
    -------
    ndarray, shape (T1,)
        Estimated treatment effect for each post-treatment period.
    """
    setup = estimate.setup
    weights = estimate.weights

    Y = setup["Y"]
    N0, T0 = setup["N0"], setup["T0"]
    N1 = Y.shape[0] - N0
    T1 = Y.shape[1] - T0

    # Subtract covariate contribution if covariates were used
    X_beta = _contract3(setup["X"], weights["beta"])
    Y_adj = Y - X_beta

    # Synthetic control trajectory across all time periods: shape (T,)
    w_unit = np.concatenate([-weights["omega"], np.ones(N1) / N1])
    tau_sc = w_unit @ Y_adj  # (T,)

    # Pre-treatment weighted average (the "trend" to subtract)
    tau_pre = tau_sc[:T0] @ weights["lambda"]  # scalar

    # Post-treatment effect curve
    tau_curve = tau_sc[T0:] - tau_pre  # shape (T1,)
    return tau_curve


def synthdid_controls(estimates, sort_by=0, mass=0.9, weight_type="omega"):
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
    Y = estimates[0].setup["Y"]
    if weight_type == "omega":
        # Unit labels: use unit_names if stored, else row index
        unit_names = estimates[0].unit_names
        labels = unit_names[:Y.shape[0] - (Y.shape[0] - estimates[0].setup["N0"])] \
            if unit_names else [f"unit_{i}" for i in range(all_weights.shape[0])]
        # Only control unit names (first N0)
        N0 = estimates[0].setup["N0"]
        if unit_names:
            labels = unit_names[:N0]
        else:
            labels = [f"unit_{i}" for i in range(N0)]
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

    # Truncate: find smallest prefix with cumulative weight >= mass
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
