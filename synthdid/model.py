"""
Scikit-learn style estimator class for synthetic difference-in-differences.

SynthDID provides a fit() / summary() interface familiar from sklearn and
statsmodels. It is a thin wrapper around synthdid_estimate() — all results
and methods live on the SynthdidEstimate object stored at self.result_.

Example
-------
>>> from synthdid import SynthDID
>>> model = SynthDID(se_method='placebo', replications=200)
>>> model.fit(df, unit='State', time='Year',
...           outcome='PacksPerCapita', treatment='treated')
>>> print(model.summary())
"""

import numpy as np
import pandas as pd

from .panel import panel_matrices
from .estimator import synthdid_estimate


class SynthDID:
    """
    Scikit-learn style synthetic difference-in-differences estimator.

    Parameters
    ----------
    se_method : {'placebo', 'bootstrap', 'jackknife'}
        Default variance estimation method used by summary(). Default 'placebo'.
    replications : int
        Default number of replications for placebo / bootstrap SE. Default 200.
    **estimator_kwargs
        Additional keyword arguments forwarded to synthdid_estimate()
        (e.g. eta_omega, do_sparsify, max_iter).

    Attributes (available after fit())
    -----------------------------------
    result_ : SynthdidEstimate
        The fitted estimate. All analytical methods are available directly
        on this object, or via the convenience shortcuts on SynthDID.

    Examples
    --------
    DataFrame input:
    >>> model = SynthDID()
    >>> model.fit(df, unit='State', time='Year',
    ...           outcome='PacksPerCapita', treatment='treated')
    >>> print(model.summary())
    >>> model.plot()

    Raw matrix input:
    >>> model = SynthDID()
    >>> model.fit(Y=Y_matrix, N0=38, T0=19)
    """

    def __init__(self, se_method="placebo", replications=200, **estimator_kwargs):
        self.se_method = se_method
        self.replications = replications
        self.estimator_kwargs = estimator_kwargs
        self.result_ = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data=None, unit=0, time=1, outcome=2, treatment=3,
            Y=None, N0=None, T0=None, unit_names=None, time_names=None):
        """
        Fit the SynthDiD estimator.

        Accepts either a long-format DataFrame or raw Y / N0 / T0 matrices.

        Parameters
        ----------
        data : pd.DataFrame or ndarray or None
            Long-format panel data (DataFrame) or outcome matrix (ndarray).
            If None, provide Y, N0, T0 as keyword arguments.
        unit, time, outcome, treatment : int or str
            Column identifiers for DataFrame input. Default 0, 1, 2, 3.
        Y : ndarray, shape (N, T) or None
            Outcome matrix — alternative to DataFrame input.
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
            Y_ = np.asarray(data, dtype=float)
            if N0 is None or T0 is None:
                raise ValueError(
                    "When passing Y as 'data', N0 and T0 must also be provided."
                )
            N0_ = int(N0)
            T0_ = int(T0)
            unit_names_ = unit_names
            time_names_ = time_names

        self.result_ = synthdid_estimate(
            Y_, N0_, T0_,
            unit_names=unit_names_,
            time_names=time_names_,
            **self.estimator_kwargs,
        )
        return self

    # ------------------------------------------------------------------
    # Convenience shortcuts — all delegate to self.result_
    # ------------------------------------------------------------------

    def summary(self, se_method=None, replications=None):
        """Compute SE and return a statsmodels-style results table."""
        self._check_fitted()
        return self.result_.summary(
            se_method=se_method or self.se_method,
            replications=replications or self.replications,
        )

    def plot(self, **kwargs):
        """Plot the estimate. See synthdid_plot() for parameter docs."""
        self._check_fitted()
        return self.result_.plot(**kwargs)

    def weights_plot(self, **kwargs):
        """Bar charts of top-N weights. See synthdid_weights_plot() for docs."""
        self._check_fitted()
        return self.result_.weights_plot(**kwargs)

    def effect_curve(self, detail=False):
        """Period-by-period treatment effect curve."""
        self._check_fitted()
        return self.result_.effect_curve(detail=detail)

    def top_weights(self, top_n=10, weight_type="omega"):
        """Table of top control units or time periods by weight."""
        self._check_fitted()
        return self.result_.top_weights(top_n=top_n, weight_type=weight_type)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self):
        if self.result_ is None:
            return (
                f"SynthDID(se_method='{self.se_method}',"
                f" replications={self.replications}) [not fitted]"
            )
        return repr(self.result_)

    def _check_fitted(self):
        if self.result_ is None:
            raise RuntimeError(
                "This SynthDID instance is not fitted yet. Call fit() first."
            )
