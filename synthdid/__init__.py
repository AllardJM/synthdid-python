"""
synthdid — Synthetic Difference-in-Differences for Python.

Python port of the R synthdid package by Arkhangelsky et al. (2021).
Reference: "Synthetic Difference in Differences", https://arxiv.org/abs/1812.09970

Public API
----------
panel_matrices        Convert long panel data to matrix format.
synthdid_estimate     Estimate treatment effect via synthetic DiD (Algorithm 1).
vcov                  Estimate variance via placebo, bootstrap, or jackknife.
synthdid_effect_curve Period-by-period treatment effect curve.
synthdid_controls     Table of control unit/time weights.
synthdid_plot         Visualization of the estimate.
"""

from .panel import panel_matrices, collapsed_form
from .estimator import synthdid_estimate, SynthdidEstimate
from .inference import vcov
from .summary import synthdid_effect_curve, synthdid_controls, EffectCurveDetail
from .plot import synthdid_plot
from .validation import synthdid_out_of_time, synthdid_oot_plot, OOTResult

__all__ = [
    "panel_matrices",
    "collapsed_form",
    "synthdid_estimate",
    "SynthdidEstimate",
    "vcov",
    "synthdid_effect_curve",
    "synthdid_controls",
    "EffectCurveDetail",
    "synthdid_plot",
    "synthdid_out_of_time",
    "synthdid_oot_plot",
    "OOTResult",
]
