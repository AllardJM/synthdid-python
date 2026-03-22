"""
Scikit-learn / statsmodels style API for synthetic difference-in-differences.

Provides a stateful SynthDID estimator class that wraps the functional API
with fit() / summary() conventions familiar from sklearn and statsmodels.

Example
-------
>>> from synthdid import SynthDID
>>> model = SynthDID(se_method='placebo', replications=200)
>>> model.fit(df, unit='State', time='Year',
...           outcome='PacksPerCapita', treatment='treated')
>>> print(model.summary())
"""

import math

import numpy as np
import pandas as pd

try:
    from scipy.special import ndtr as _normal_cdf
except ImportError:
    def _normal_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

from .panel import panel_matrices
from .estimator import synthdid_estimate
from .inference import vcov

_WIDTH = 60


class SynthDIDResults:
    """
    Results container returned by SynthDID.summary().

    Attributes
    ----------
    tau : float
        Point estimate of the average treatment effect.
    se : float
        Standard error (sqrt of vcov output).
    z : float
        z-statistic (tau / se).
    pvalue : float
        Two-sided p-value from a standard normal.
    conf_int : tuple of float
        95% confidence interval (lower, upper).
    se_method : str
        Variance estimation method used ('placebo', 'bootstrap', 'jackknife').
    replications : int
        Number of replications used for placebo / bootstrap SE.
    n_units : int
    n_periods : int
    n_control : int
    n_treated : int
    n_pre : int
    n_post : int
    """

    def __init__(self, tau, se, se_method, replications,
                 n_units, n_periods, n_control, n_treated, n_pre, n_post):
        self.tau = tau
        self.se = se
        self.se_method = se_method
        self.replications = replications
        self.n_units = n_units
        self.n_periods = n_periods
        self.n_control = n_control
        self.n_treated = n_treated
        self.n_pre = n_pre
        self.n_post = n_post

        self.z = tau / se if se and not math.isnan(se) and se != 0 else float("nan")
        pval_raw = 2.0 * (1.0 - _normal_cdf(abs(self.z)))
        self.pvalue = pval_raw if not math.isnan(self.z) else float("nan")
        self.conf_int = (tau - 1.96 * se, tau + 1.96 * se)

    def __repr__(self):
        return (
            f"SynthDIDResults(tau={self.tau:.4f}, se={self.se:.4f},"
            f" method='{self.se_method}')"
        )

    def __str__(self):
        W = _WIDTH
        sep_eq = "=" * W
        sep_da = "-" * W

        def _meta_row(l1, v1, l2="", v2=""):
            left = f" {l1:<24}{str(v1):>5}"          # 30 chars
            right = f"  {l2:<16}{str(v2):>8} " if l2 else ""
            return (left + right).ljust(W)[:W]

        def _fmt(v):
            return f"{v:.4f}" if not math.isnan(v) else "nan"

        lines = [
            sep_eq,
            f"{'Synthetic Difference-in-Differences':^{W}}",
            sep_eq,
            _meta_row("No. Units:", self.n_units,
                      "SE Method:", self.se_method),
            _meta_row("No. Periods:", self.n_periods,
                      "Replications:", self.replications),
            _meta_row("No. Control Units:", self.n_control,
                      "Conf. Level:", "95%"),
            _meta_row("No. Treated Units:", self.n_treated),
            _meta_row("Pre-treatment Periods:", self.n_pre),
            _meta_row("Post-treatment Periods:", self.n_post),
            sep_da,
            f"{'':>11}{'coef':>10}{'std err':>10}{'z':>10}{'P>|z|':>10}",
            sep_da,
            f" {'tau':<10}{_fmt(self.tau):>10}{_fmt(self.se):>10}"
            f"{_fmt(self.z):>10}{_fmt(self.pvalue):>10}",
            sep_da,
            f" 95% Conf. Interval:  [{self.conf_int[0]:>8.4f}, {self.conf_int[1]:>8.4f}]",
            sep_eq,
        ]
        return "\n".join(lines)


class SynthDID:
    """
    Synthetic Difference-in-Differences estimator.

    Wraps the functional synthdid API with a scikit-learn style fit() method
    and a statsmodels style summary() output.

    Parameters
    ----------
    se_method : {'placebo', 'bootstrap', 'jackknife'}
        Default variance estimation method. Default 'placebo'.
    replications : int
        Default number of replications for placebo / bootstrap SE. Default 200.
    **estimator_kwargs
        Additional keyword arguments forwarded to synthdid_estimate()
        (e.g. eta_omega, do_sparsify, max_iter).

    Fitted attributes (available after fit())
    -----------------------------------------
    tau_ : float
        Point estimate.
    weights_ : dict
        Full weights dict with keys 'omega', 'lambda', 'beta', 'vals'.
    omega_ : ndarray, shape (N0,)
        Control unit weights.
    lambda_ : ndarray, shape (T0,)
        Pre-treatment time weights.
    se_ : float or None
        Standard error — populated after summary() is called.
    n_units_, n_periods_, n_control_, n_treated_, n_pre_, n_post_ : int
        Dataset dimensions.

    Examples
    --------
    DataFrame input:
    >>> model = SynthDID()
    >>> model.fit(df, unit='State', time='Year',
    ...           outcome='PacksPerCapita', treatment='treated')
    >>> print(model.summary())

    Raw matrix input:
    >>> model = SynthDID()
    >>> model.fit(Y=Y_matrix, N0=38, T0=19)
    """

    def __init__(self, se_method="placebo", replications=200, **estimator_kwargs):
        self.se_method = se_method
        self.replications = replications
        self.estimator_kwargs = estimator_kwargs

        # Fitted attributes — None until fit() is called
        self.tau_ = None
        self.weights_ = None
        self.omega_ = None
        self.lambda_ = None
        self.se_ = None
        self.n_units_ = None
        self.n_periods_ = None
        self.n_control_ = None
        self.n_treated_ = None
        self.n_pre_ = None
        self.n_post_ = None
        self._estimate = None
        self._se_method_used = None
        self._se_replications_used = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data=None, unit=0, time=1, outcome=2, treatment=3,
            Y=None, N0=None, T0=None, unit_names=None, time_names=None):
        """
        Fit the SynthDiD estimator.

        Accepts either a long-format DataFrame or raw Y / N0 / T0 matrices.

        Parameters
        ----------
        data : pd.DataFrame or None
            Long-format panel data. If provided, unit / time / outcome /
            treatment identify its columns (by name or position).
        unit, time, outcome, treatment : int or str
            Column identifiers — passed to panel_matrices(). Default 0,1,2,3.
        Y : ndarray, shape (N, T) or None
            Outcome matrix (alternative to DataFrame input).
        N0 : int or None
            Number of control units (required with Y).
        T0 : int or None
            Number of pre-treatment periods (required with Y).
        unit_names : list or None
            Row labels for Y (matrix input only).
        time_names : list or None
            Column labels for Y (matrix input only).

        Returns
        -------
        self
        """
        if isinstance(data, pd.DataFrame):
            setup = panel_matrices(data, unit=unit, time=time,
                                   outcome=outcome, treatment=treatment)
            Y_ = setup["Y"]
            N0_ = setup["N0"]
            T0_ = setup["T0"]
            unit_names_ = setup["unit_names"]
            time_names_ = [str(t) for t in setup["time_names"]]
        elif data is None:
            if Y is None or N0 is None or T0 is None:
                raise ValueError(
                    "Provide either a DataFrame as 'data', or Y, N0, and T0."
                )
            Y_ = np.asarray(Y, dtype=float)
            N0_ = int(N0)
            T0_ = int(T0)
            unit_names_ = unit_names
            time_names_ = time_names
        else:
            # numpy array passed as first positional argument
            Y_ = np.asarray(data, dtype=float)
            if N0 is None or T0 is None:
                raise ValueError(
                    "When passing Y as 'data', N0 and T0 must also be provided."
                )
            N0_ = int(N0)
            T0_ = int(T0)
            unit_names_ = unit_names
            time_names_ = time_names

        est = synthdid_estimate(
            Y_, N0_, T0_,
            unit_names=unit_names_,
            time_names=time_names_,
            **self.estimator_kwargs,
        )

        self._estimate = est
        self.tau_ = est.estimate
        self.weights_ = est.weights
        self.omega_ = est.weights["omega"]
        self.lambda_ = est.weights["lambda"]
        self.n_units_ = Y_.shape[0]
        self.n_periods_ = Y_.shape[1]
        self.n_control_ = N0_
        self.n_treated_ = Y_.shape[0] - N0_
        self.n_pre_ = T0_
        self.n_post_ = Y_.shape[1] - T0_
        self.se_ = None
        self._se_method_used = None
        self._se_replications_used = None

        return self

    def summary(self, se_method=None, replications=None):
        """
        Compute standard error and return a formatted results table.

        Parameters
        ----------
        se_method : {'placebo', 'bootstrap', 'jackknife'} or None
            Variance method. Defaults to the value set at construction.
        replications : int or None
            Number of replications. Defaults to the value set at construction.

        Returns
        -------
        SynthDIDResults
            Results object whose str() renders a statsmodels-style table.
        """
        self._check_fitted()
        method = se_method or self.se_method
        reps = replications or self.replications

        # Recompute only when method or replications change
        if (self.se_ is None
                or method != self._se_method_used
                or reps != self._se_replications_used):
            variance = vcov(self._estimate, method=method, replications=reps)
            self.se_ = float(np.sqrt(variance))
            self._se_method_used = method
            self._se_replications_used = reps

        return SynthDIDResults(
            tau=self.tau_,
            se=self.se_,
            se_method=method,
            replications=reps,
            n_units=self.n_units_,
            n_periods=self.n_periods_,
            n_control=self.n_control_,
            n_treated=self.n_treated_,
            n_pre=self.n_pre_,
            n_post=self.n_post_,
        )

    def plot(self, **kwargs):
        """
        Plot the SynthDiD estimate. Delegates to synthdid_plot().

        If summary() has been called, the SE is passed automatically.
        Override by passing se=... explicitly.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._check_fitted()
        from .plot import synthdid_plot
        se = kwargs.pop("se", self.se_)
        return synthdid_plot(self._estimate, se=se, **kwargs)

    def weights_plot(self, **kwargs):
        """
        Bar charts of top-N control units and time periods by weight.
        Delegates to synthdid_weights_plot().

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._check_fitted()
        from .plot import synthdid_weights_plot
        return synthdid_weights_plot(self._estimate, **kwargs)

    def effect_curve(self, detail=False):
        """
        Period-by-period treatment effect curve. Delegates to synthdid_effect_curve().

        Parameters
        ----------
        detail : bool
            If True, return an EffectCurveDetail object. Default False.

        Returns
        -------
        ndarray or EffectCurveDetail
        """
        self._check_fitted()
        from .summary import synthdid_effect_curve
        return synthdid_effect_curve(self._estimate, detail=detail)

    def top_weights(self, top_n=10, weight_type="omega"):
        """
        Table of the most important control units or time periods by weight.
        Delegates to synthdid_controls().

        Parameters
        ----------
        top_n : int
            Number of rows to return. Default 10.
        weight_type : {'omega', 'lambda'}
            Unit weights or time weights.

        Returns
        -------
        pd.DataFrame
        """
        self._check_fitted()
        from .summary import synthdid_controls
        return synthdid_controls(self._estimate, top_n=top_n, weight_type=weight_type)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self):
        if self._estimate is None:
            return (
                f"SynthDID(se_method='{self.se_method}',"
                f" replications={self.replications}) [not fitted]"
            )
        return (
            f"SynthDID(tau={self.tau_:.4f},"
            f" n_units={self.n_units_},"
            f" n_control={self.n_control_},"
            f" n_treated={self.n_treated_},"
            f" n_pre={self.n_pre_},"
            f" n_post={self.n_post_})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self._estimate is None:
            raise RuntimeError("This SynthDID instance is not fitted yet. Call fit() first.")
