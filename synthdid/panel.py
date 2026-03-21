"""
Panel data utilities for synthetic difference-in-differences.

Ported from R/utils.R in the synthdid package by Arkhangelsky et al.
Reference: https://arxiv.org/abs/1812.09970
"""

import numpy as np
import pandas as pd


def panel_matrices(panel, unit=0, time=1, outcome=2, treatment=3, treated_last=True):
    """
    Convert a long-format balanced panel to the wide matrix format required
    by synthdid estimators.

    Parameters
    ----------
    panel : pd.DataFrame
        Long panel data with one row per (unit, time) observation.
    unit : int or str
        Column index or name for the unit identifier. Default 0 (first column).
    time : int or str
        Column index or name for the time identifier. Default 1.
    outcome : int or str
        Column index or name for the outcome variable. Default 2.
    treatment : int or str
        Column index or name for the binary treatment indicator (0/1). Default 3.
    treated_last : bool
        If True (default), sort rows so control units appear before treated units.

    Returns
    -------
    dict with keys:
        Y  : ndarray, shape (N, T) — outcome matrix, rows=units, cols=time
        N0 : int — number of control units
        T0 : int — number of pre-treatment time periods
        W  : ndarray, shape (N, T) — binary treatment indicator matrix
        unit_names : list — unit identifiers (row labels of Y)
        time_names : list — time identifiers (column labels of Y)

    Raises
    ------
    ValueError
        If the panel is unbalanced, treatment is not binary, simultaneous
        adoption is violated, or there is no treatment variation.
    """
    # Resolve column names from integer indices if needed
    def _resolve_col(col):
        if isinstance(col, int):
            return panel.columns[col]
        return col

    unit_col = _resolve_col(unit)
    time_col = _resolve_col(time)
    outcome_col = _resolve_col(outcome)
    treatment_col = _resolve_col(treatment)

    # Keep only the four relevant columns
    panel = panel[[unit_col, time_col, outcome_col, treatment_col]].copy()

    if panel.isnull().any().any():
        raise ValueError("Missing values in panel data.")

    treatment_vals = panel[treatment_col].unique()
    if len(treatment_vals) == 1:
        raise ValueError("There is no variation in treatment status.")
    if not set(treatment_vals).issubset({0, 1}):
        raise ValueError("Treatment status must be binary (0 or 1).")

    # Check balanced panel: every unit observed at every time period
    n_units = panel[unit_col].nunique()
    n_times = panel[time_col].nunique()
    counts = panel.groupby([unit_col, time_col]).size()
    if len(counts) != n_units * n_times or (counts != 1).any():
        raise ValueError(
            "Panel must be balanced: each unit must be observed at every time period."
        )

    # Sort and pivot to wide format
    panel = panel.sort_values([unit_col, time_col])
    units = panel[unit_col].unique().tolist()
    times = sorted(panel[time_col].unique().tolist())

    Y = panel.pivot(index=unit_col, columns=time_col, values=outcome_col)
    W = panel.pivot(index=unit_col, columns=time_col, values=treatment_col)

    # Align column order to sorted time
    Y = Y[times]
    W = W[times]

    # Identify treated units and pre-treatment cutoff
    ever_treated = W.any(axis=1)  # boolean Series: unit -> treated at any time
    N0 = int((~ever_treated).sum())

    # T0 = index of last period before first treatment starts
    treatment_starts = W.any(axis=0)  # bool Series: time -> any unit treated
    first_treated_col = int(np.where(treatment_starts.values)[0][0])
    T0 = first_treated_col  # columns 0..T0-1 are pre-treatment

    # Validate simultaneous adoption:
    #   - controls never treated
    #   - no one treated pre-period T0
    #   - treated units treated every post-period
    W_arr = W.values.astype(int)
    ctrl_mask = ~ever_treated.values  # (N,)
    treat_mask = ever_treated.values

    if not (
        np.all(W_arr[ctrl_mask] == 0)           # controls never treated
        and np.all(W_arr[:, :T0] == 0)           # nobody treated pre-treatment
        and np.all(W_arr[treat_mask, T0:] == 1)  # treated units treated every post-period
    ):
        raise ValueError(
            "Treatment adoption is not simultaneous. All treated units must "
            "begin treatment at the same time and remain treated thereafter."
        )

    # Sort rows: controls first, then treated (alphabetically within each group)
    if treated_last:
        row_order = np.argsort(
            [int(ever_treated[u]) * 2 + 0 for u in Y.index],
            kind="stable",
        )
        # Simpler: sort by (is_treated, unit_name)
        sort_key = [(int(ever_treated[u]), str(u)) for u in Y.index]
        row_order = sorted(range(len(Y.index)), key=lambda i: sort_key[i])
        Y = Y.iloc[row_order]
        W = W.iloc[row_order]

    unit_names = list(Y.index)
    time_names = list(Y.columns)

    return {
        "Y": Y.values.astype(float),
        "N0": N0,
        "T0": T0,
        "W": W.values.astype(int),
        "unit_names": unit_names,
        "time_names": time_names,
    }


def collapsed_form(Y, N0, T0):
    """
    Collapse the N×T outcome matrix to an (N0+1)×(T0+1) matrix.

    The collapsed matrix averages treated units into one synthetic row and
    post-treatment periods into one synthetic column:
        - Top-left  (N0 × T0): control pre-treatment — unchanged
        - Top-right (N0 × 1) : mean of control post-treatment outcomes
        - Bottom-left (1 × T0): mean of treated pre-treatment outcomes
        - Bottom-right (1 × 1): mean of treated post-treatment outcomes

    Parameters
    ----------
    Y : ndarray, shape (N, T)
    N0 : int — number of control rows
    T0 : int — number of pre-treatment columns

    Returns
    -------
    Yc : ndarray, shape (N0+1, T0+1)
    """
    N, T = Y.shape
    Yc = np.zeros((N0 + 1, T0 + 1))

    # Control pre-treatment: keep as-is
    Yc[:N0, :T0] = Y[:N0, :T0]

    # Control post-treatment: average over post-treatment columns
    Yc[:N0, T0] = Y[:N0, T0:].mean(axis=1)

    # Treated pre-treatment: average over treated rows
    Yc[N0, :T0] = Y[N0:, :T0].mean(axis=0)

    # Treated post-treatment: grand mean
    Yc[N0, T0] = Y[N0:, T0:].mean()

    return Yc
