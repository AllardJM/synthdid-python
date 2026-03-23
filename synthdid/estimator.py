"""
Synthetic difference-in-differences estimator.

Implements Algorithm 1 from Arkhangelsky et al. (2021):
  "Synthetic Difference in Differences"
  https://arxiv.org/abs/1812.09970

Ported from R/synthdid.R in the synthdid package.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .panel import collapsed_form
from .solver import sc_weight_fw, sparsify


@dataclass
class SynthdidEstimate:
    """
    Container for a synthetic difference-in-differences estimate.

    Attributes
    ----------
    estimate : float
        The point estimate τ̂ (average treatment effect on the treated).
    weights : dict
        Optimization outputs:
          - 'omega'  : ndarray (N0,) — unit weights for control group
          - 'lambda' : ndarray (T0,) — time weights for pre-treatment periods
          - 'beta'   : ndarray — covariate coefficients (empty if no covariates)
          - 'vals'   : list — objective values per iteration
    setup : dict
        Problem inputs:
          - 'Y'  : ndarray (N, T) — outcome matrix
          - 'X'  : ndarray (N, T, 0) — covariate array (empty by default)
          - 'N0' : int
          - 'T0' : int
    opts : dict
        Optimization hyperparameters used (zeta_omega, zeta_lambda, etc.).
    unit_names : list
        Row labels of Y (unit identifiers).
    time_names : list
        Column labels of Y (time identifiers).
    """

    estimate: float
    weights: dict
    setup: dict
    opts: dict
    unit_names: list = field(default_factory=list)
    time_names: list = field(default_factory=list)

    def __float__(self):
        return float(self.estimate)

    def __repr__(self):
        N, T = self.setup["Y"].shape
        N0, T0 = self.setup["N0"], self.setup["T0"]
        return (
            f"SynthdidEstimate(tau={self.estimate:.4f},"
            f" n_units={N}, n_control={N0}, n_treated={N - N0},"
            f" n_pre={T0}, n_post={T - T0})"
        )

    def __format__(self, spec):
        return format(self.estimate, spec)

    def summary(self, se_method="placebo", replications=200):
        """
        Compute standard error and return a formatted results table.

        Parameters
        ----------
        se_method : {'placebo', 'bootstrap', 'jackknife'}
            Variance estimation method. Default 'placebo'.
        replications : int
            Number of replications for placebo / bootstrap. Default 200.

        Returns
        -------
        SynthDIDResults
            Results object whose str() renders a statsmodels-style table.
        """
        from .inference import vcov
        from .results import SynthDIDResults
        import numpy as np

        variance = vcov(self, method=se_method, replications=replications)
        se = float(np.sqrt(variance))
        N, T = self.setup["Y"].shape
        N0, T0 = self.setup["N0"], self.setup["T0"]
        return SynthDIDResults(
            tau=self.estimate,
            se=se,
            se_method=se_method,
            replications=replications,
            n_units=N,
            n_periods=T,
            n_control=N0,
            n_treated=N - N0,
            n_pre=T0,
            n_post=T - T0,
        )

    def plot(self, se=None, **kwargs):
        """
        Plot the estimate. See synthdid_plot() for full parameter docs.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plot import synthdid_plot
        return synthdid_plot(self, se=se, **kwargs)

    def weights_plot(self, **kwargs):
        """
        Bar charts of top-N control units and time periods by weight.
        See synthdid_weights_plot() for full parameter docs.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plot import synthdid_weights_plot
        return synthdid_weights_plot(self, **kwargs)

    def effect_curve(self, detail=False):
        """
        Period-by-period treatment effect curve.
        See synthdid_effect_curve() for full parameter docs.

        Returns
        -------
        ndarray or EffectCurveDetail
        """
        from .summary import synthdid_effect_curve
        return synthdid_effect_curve(self, detail=detail)

    def top_weights(self, top_n=10, weight_type="omega"):
        """
        Table of the most important control units or time periods by weight.
        See synthdid_controls() for full parameter docs.

        Returns
        -------
        pd.DataFrame
        """
        from .summary import synthdid_controls
        return synthdid_controls(self, top_n=top_n, weight_type=weight_type)


def synthdid_estimate(
    Y,
    N0,
    T0,
    X=None,
    noise_level=None,
    eta_omega=None,
    eta_lambda=1e-6,
    zeta_omega=None,
    zeta_lambda=None,
    omega_intercept=True,
    lambda_intercept=True,
    weights=None,
    update_omega=None,
    update_lambda=None,
    min_decrease=None,
    max_iter=10000,
    do_sparsify=True,
    max_iter_pre_sparsify=100,
    unit_names=None,
    time_names=None,
):
    """
    Estimate the average treatment effect using synthetic difference-in-differences.

    Implements Algorithm 1 of Arkhangelsky et al. (2021). The estimator
    combines a synthetic control weighting over units with a time weighting
    over pre-treatment periods, yielding a doubly-robust DiD estimate.

    Parameters
    ----------
    Y : ndarray, shape (N, T)
        Outcome matrix. Rows 0..N0-1 are control units; rows N0..N-1 are
        treated units. Columns 0..T0-1 are pre-treatment; T0..T-1 are
        post-treatment.
    N0 : int
        Number of control units.
    T0 : int
        Number of pre-treatment time periods.
    X : ndarray, shape (N, T, C) or None
        Optional time-varying covariates array. Not used by default.
    noise_level : float or None
        Estimate of the noise standard deviation σ. Defaults to the standard
        deviation of first differences of control pre-treatment outcomes.
    eta_omega : float or None
        Scales the unit-weight regularization: zeta_omega = eta_omega * noise_level.
        Defaults to (N_treated * T_post)^(1/4).
    eta_lambda : float
        Scales the time-weight regularization: zeta_lambda = eta_lambda * noise_level.
        Default 1e-6 (near-zero, so lambda is mostly unregularized).
    zeta_omega : float or None
        Override for zeta_omega directly.
    zeta_lambda : float or None
        Override for zeta_lambda directly.
    omega_intercept : bool
        Demean columns when estimating unit weights. Default True.
    lambda_intercept : bool
        Demean columns when estimating time weights. Default True.
    weights : dict or None
        Pre-specified weights {'omega': ..., 'lambda': ...}. If provided and
        update_omega/update_lambda are False, these are used as-is.
    update_omega : bool or None
        Whether to optimize omega. Defaults to True if weights['omega'] is None.
    update_lambda : bool or None
        Whether to optimize lambda. Defaults to True if weights['lambda'] is None.
    min_decrease : float or None
        Convergence threshold; stop when objective decrease < min_decrease^2.
        Defaults to 1e-5 * noise_level.
    max_iter : int
        Maximum Frank-Wolfe iterations. Default 10,000.
    do_sparsify : bool
        If True, run a second Frank-Wolfe pass initialized at the sparsified
        weights to encourage sparse solutions. Default True.
    max_iter_pre_sparsify : int
        Max iterations for the first (pre-sparsification) pass. Default 100.
    unit_names : list or None
        Row labels for Y. Stored in the returned object.
    time_names : list or None
        Column labels for Y. Stored in the returned object.

    Returns
    -------
    SynthdidEstimate
        Contains the scalar estimate and all intermediate weights/setup.
    """
    Y = np.asarray(Y, dtype=float)
    N, T = Y.shape
    N1 = N - N0
    T1 = T - T0

    assert N > N0, "Need at least one treated unit (N > N0)."
    assert T > T0, "Need at least one post-treatment period (T > T0)."

    # Default covariate array: N × T × 0 (no covariates)
    if X is None:
        X = np.zeros((N, T, 0))

    # Initialize weights dict
    if weights is None:
        weights = {"omega": None, "lambda": None}
    else:
        weights = dict(weights)  # shallow copy to avoid mutating caller's dict

    # Determine which weights to update
    if update_omega is None:
        update_omega = weights.get("omega") is None
    if update_lambda is None:
        update_lambda = weights.get("lambda") is None

    # Default noise level: SD of first differences of control pre-treatment data
    if noise_level is None:
        diffs = np.diff(Y[:N0, :T0], axis=1)
        noise_level = float(np.std(diffs, ddof=1)) if diffs.size > 0 else 1.0

    # Default regularization strengths
    if eta_omega is None:
        eta_omega = (N1 * T1) ** 0.25
    if zeta_omega is None:
        zeta_omega = eta_omega * noise_level
    if zeta_lambda is None:
        zeta_lambda = eta_lambda * noise_level

    # Default convergence threshold
    if min_decrease is None:
        min_decrease = 1e-5 * noise_level

    # Effective max_iter for first pass
    first_pass_max_iter = max_iter_pre_sparsify if do_sparsify else max_iter

    # -------------------------------------------------------------------------
    # Estimate lambda (time weights) via Frank-Wolfe on collapsed data
    # -------------------------------------------------------------------------
    if update_lambda:
        Yc = collapsed_form(Y, N0, T0)
        # Optimize over the N0 control rows; last column is post-treatment target
        lambda_opt, lambda_vals = sc_weight_fw(
            Yc[:N0],
            zeta=zeta_lambda,
            intercept=lambda_intercept,
            lambda_init=weights.get("lambda"),
            min_decrease=min_decrease,
            max_iter=first_pass_max_iter,
        )
        if do_sparsify:
            # Second pass initialized at sparsified solution
            lambda_opt, lambda_vals = sc_weight_fw(
                Yc[:N0],
                zeta=zeta_lambda,
                intercept=lambda_intercept,
                lambda_init=sparsify(lambda_opt),
                min_decrease=min_decrease,
                max_iter=max_iter,
            )
        weights["lambda"] = lambda_opt
        weights["lambda_vals"] = lambda_vals
        weights["vals"] = lambda_vals

    # -------------------------------------------------------------------------
    # Estimate omega (unit weights) via Frank-Wolfe on transposed collapsed data
    # -------------------------------------------------------------------------
    if update_omega:
        Yc = collapsed_form(Y, N0, T0)
        # Transpose so units become "time" axis for sc_weight_fw;
        # use only pre-treatment columns → shape (T0, N0+1)
        omega_opt, omega_vals = sc_weight_fw(
            Yc[:, :T0].T,
            zeta=zeta_omega,
            intercept=omega_intercept,
            lambda_init=weights.get("omega"),
            min_decrease=min_decrease,
            max_iter=first_pass_max_iter,
        )
        if do_sparsify:
            omega_opt, omega_vals = sc_weight_fw(
                Yc[:, :T0].T,
                zeta=zeta_omega,
                intercept=omega_intercept,
                lambda_init=sparsify(omega_opt),
                min_decrease=min_decrease,
                max_iter=max_iter,
            )
        weights["omega"] = omega_opt
        weights["omega_vals"] = omega_vals
        # Merge convergence traces
        if weights.get("vals") is None:
            weights["vals"] = omega_vals
        else:
            weights["vals"] = _pairwise_sum_decreasing(weights["vals"], omega_vals)

    weights.setdefault("beta", np.array([]))

    # -------------------------------------------------------------------------
    # Compute the final treatment effect estimate (Algorithm 1, step 3)
    #
    # τ̂ is a 2×2 difference-in-differences:
    #
    #   τ̂ = (treated_avg − synth_control) in post periods
    #       − (treated_avg − synth_control) in λ-weighted pre periods
    #
    # The first difference removes the cross-sectional gap (unit fixed effects).
    # The second difference removes the pre-existing time trend (time fixed effects).
    # Together they isolate the causal effect under parallel trends.
    #
    # Written as a single matrix product:
    #   w_unit = [-ω, (1/N1)·1]   contrasts synth control vs treated average
    #   w_time = [-λ, (1/T1)·1]   contrasts λ-weighted pre vs uniform post
    #   τ̂ = w_unit @ (Y − Xβ) @ w_time
    # -------------------------------------------------------------------------
    X_beta = _contract3(X, weights["beta"])

    w_unit = np.concatenate([-weights["omega"], np.ones(N1) / N1])  # (N,)
    w_time = np.concatenate([-weights["lambda"], np.ones(T1) / T1])  # (T,)
    tau_hat = float(w_unit @ (Y - X_beta) @ w_time)

    opts = {
        "zeta_omega": zeta_omega,
        "zeta_lambda": zeta_lambda,
        "omega_intercept": omega_intercept,
        "lambda_intercept": lambda_intercept,
        "update_omega": update_omega,
        "update_lambda": update_lambda,
        "min_decrease": min_decrease,
        "max_iter": max_iter,
    }

    setup = {"Y": Y, "X": X, "N0": N0, "T0": T0}

    return SynthdidEstimate(
        estimate=tau_hat,
        weights=weights,
        setup=setup,
        opts=opts,
        unit_names=unit_names or [],
        time_names=time_names or [],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contract3(X, v):
    """
    Compute the weighted sum of covariate slices: sum_c v[c] * X[:, :, c].

    Parameters
    ----------
    X : ndarray, shape (N, T, C)
    v : ndarray, shape (C,)

    Returns
    -------
    ndarray, shape (N, T)
    """
    out = np.zeros(X.shape[:2])
    for c in range(len(v)):
        out += v[c] * X[:, :, c]
    return out


def _pairwise_sum_decreasing(x, y):
    """
    Component-wise sum of two decreasing sequences, treating trailing NaN
    as the last finite value (used to merge lambda and omega convergence traces).
    """
    x = list(x)
    y = list(y)
    # Pad shorter list with its last value
    max_len = max(len(x), len(y))

    def _pad(lst, length):
        if not lst:
            return [np.nan] * length
        return lst + [lst[-1]] * (length - len(lst))

    x = _pad(x, max_len)
    y = _pad(y, max_len)
    return [xi + yi for xi, yi in zip(x, y)]
