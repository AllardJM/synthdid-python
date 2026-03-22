"""
Results container for synthetic difference-in-differences estimates.

SynthDIDResults is returned by SynthdidEstimate.summary() and renders a
statsmodels-style table when printed.
"""

import math

try:
    from scipy.special import ndtr as _normal_cdf
except ImportError:
    def _normal_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

_WIDTH = 60


class SynthDIDResults:
    """
    Results container returned by SynthdidEstimate.summary().

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
        Variance estimation method ('placebo', 'bootstrap', 'jackknife').
    replications : int
        Number of replications used for placebo / bootstrap SE.
    n_units, n_periods, n_control, n_treated, n_pre, n_post : int
        Dataset dimensions.
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
