"""
synthdid — Synthetic Difference-in-Differences for Python.

Python port of the R synthdid package by Arkhangelsky et al. (2021).
Reference: "Synthetic Difference in Differences", https://arxiv.org/abs/1812.09970

Public API
----------
panel_matrices        Convert long panel data to matrix format.
synthdid_estimate     Estimate treatment effect (returns SynthdidEstimate).
vcov                  Estimate variance via placebo, bootstrap, or jackknife.
synthdid_plot         Visualization of the estimate.
synthdid_weights_plot Bar charts of top-N control units and time periods by weight.

SynthdidEstimate      Rich result object — call .summary(), .plot(), .effect_curve(), etc.
SynthDIDResults       Statsmodels-style results table returned by .summary().
SynthDID              Sklearn-style estimator class with fit() / summary().
"""

from .panel import panel_matrices, collapsed_form
from .estimator import synthdid_estimate, SynthdidEstimate
from .inference import vcov
from .summary import synthdid_effect_curve, synthdid_controls, EffectCurveDetail
from .plot import synthdid_plot, synthdid_weights_plot
from .validation import synthdid_out_of_time, synthdid_oot_plot, OOTResult
from .results import SynthDIDResults
from .model import SynthDID

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
    "synthdid_weights_plot",
    "synthdid_out_of_time",
    "synthdid_oot_plot",
    "OOTResult",
    "SynthDIDResults",
    "SynthDID",
]
