"""
Tests for the synthdid Python package.

All deterministic results are validated against the R synthdid package
on the California Proposition 99 dataset.
"""

import os
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synthdid import (
    panel_matrices,
    synthdid_estimate,
    vcov,
    synthdid_effect_curve,
    synthdid_controls,
    SynthDID,
    SynthDIDResults,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "california_prop99.csv"
)


@pytest.fixture(scope="module")
def prop99_estimate():
    """Compute the Prop 99 estimate once and reuse across tests."""
    df = pd.read_csv(DATA_PATH, sep=None, engine="python")
    setup = panel_matrices(
        df, unit="State", time="Year",
        outcome="PacksPerCapita", treatment="treated"
    )
    est = synthdid_estimate(
        setup["Y"], setup["N0"], setup["T0"],
        unit_names=setup["unit_names"],
        time_names=[str(t) for t in setup["time_names"]],
    )
    return est, setup


# ---------------------------------------------------------------------------
# panel_matrices tests
# ---------------------------------------------------------------------------

class TestPanelMatrices:
    def test_dimensions(self, prop99_estimate):
        _, setup = prop99_estimate
        assert setup["Y"].shape == (39, 31)

    def test_N0_T0(self, prop99_estimate):
        _, setup = prop99_estimate
        assert setup["N0"] == 38
        assert setup["T0"] == 19

    def test_treated_unit_last(self, prop99_estimate):
        _, setup = prop99_estimate
        assert setup["unit_names"][-1] == "California"

    def test_weights_balanced(self, prop99_estimate):
        _, setup = prop99_estimate
        # Every unit observed at every time period
        assert not np.any(np.isnan(setup["Y"]))

    def test_unbalanced_raises(self):
        df = pd.DataFrame({
            "unit": ["A", "A", "B"],
            "time": [1, 2, 1],
            "y": [1.0, 2.0, 3.0],
            "w": [0, 0, 1],
        })
        with pytest.raises(ValueError, match="balanced"):
            panel_matrices(df, unit="unit", time="time", outcome="y", treatment="w")

    def test_no_treatment_variation_raises(self):
        df = pd.DataFrame({
            "unit": ["A", "A", "B", "B"],
            "time": [1, 2, 1, 2],
            "y": [1.0, 2.0, 3.0, 4.0],
            "w": [0, 0, 0, 0],
        })
        with pytest.raises(ValueError, match="variation"):
            panel_matrices(df, unit="unit", time="time", outcome="y", treatment="w")


# ---------------------------------------------------------------------------
# synthdid_estimate tests
# ---------------------------------------------------------------------------

class TestSynthdidEstimate:
    def test_point_estimate(self, prop99_estimate):
        """Point estimate should match R to 2 decimal places."""
        est, _ = prop99_estimate
        assert abs(float(est) - (-15.60)) < 0.01

    def test_weights_sum_to_one(self, prop99_estimate):
        est, _ = prop99_estimate
        assert abs(est.weights["omega"].sum() - 1.0) < 1e-9
        assert abs(est.weights["lambda"].sum() - 1.0) < 1e-9

    def test_weights_non_negative(self, prop99_estimate):
        est, _ = prop99_estimate
        assert np.all(est.weights["omega"] >= -1e-10)
        assert np.all(est.weights["lambda"] >= -1e-10)

    def test_omega_weights_match_r(self, prop99_estimate):
        """Top control weights should match R exactly to 4 decimal places."""
        est, setup = prop99_estimate
        unit_names = setup["unit_names"]
        omega = est.weights["omega"]
        assert abs(omega[unit_names.index("Nevada")] - 0.1245) < 1e-4
        assert abs(omega[unit_names.index("New Hampshire")] - 0.1050) < 1e-4
        assert abs(omega[unit_names.index("Connecticut")] - 0.0783) < 1e-4

    def test_lambda_weights_match_r(self, prop99_estimate):
        """Pre-treatment time weights should match R exactly to 4 decimal places."""
        est, setup = prop99_estimate
        time_names = [str(t) for t in setup["time_names"]]
        lambda_ = est.weights["lambda"]
        assert abs(lambda_[time_names.index("1988")] - 0.4271) < 1e-4
        assert abs(lambda_[time_names.index("1986")] - 0.3665) < 1e-4
        assert abs(lambda_[time_names.index("1987")] - 0.2065) < 1e-4

    def test_float_conversion(self, prop99_estimate):
        est, _ = prop99_estimate
        assert isinstance(float(est), float)

    def test_repr(self, prop99_estimate):
        est, _ = prop99_estimate
        assert "SynthdidEstimate" in repr(est)


# ---------------------------------------------------------------------------
# vcov / SE tests
# ---------------------------------------------------------------------------

class TestVcov:
    def test_placebo_returns_positive_variance(self, prop99_estimate):
        est, _ = prop99_estimate
        np.random.seed(0)
        var = vcov(est, method="placebo", replications=50)
        assert var > 0

    def test_jackknife_returns_nan_single_treated(self, prop99_estimate):
        """Jackknife is undefined with only one treated unit."""
        est, _ = prop99_estimate
        var = vcov(est, method="jackknife")
        assert np.isnan(np.sqrt(var))

    def test_jackknife_works_two_treated(self):
        """Jackknife should return a valid SE when N1 > 1."""
        df = pd.read_csv(DATA_PATH, sep=None, engine="python")
        df.loc[(df["State"] == "Nevada") & (df["Year"] >= 1989), "treated"] = 1
        setup = panel_matrices(
            df, unit="State", time="Year",
            outcome="PacksPerCapita", treatment="treated"
        )
        est = synthdid_estimate(setup["Y"], setup["N0"], setup["T0"])
        var = vcov(est, method="jackknife")
        se = np.sqrt(var)
        assert not np.isnan(se)
        assert abs(se - 3.83) < 0.01  # validated against R

    def test_invalid_method_raises(self, prop99_estimate):
        est, _ = prop99_estimate
        with pytest.raises(ValueError):
            vcov(est, method="invalid")


# ---------------------------------------------------------------------------
# synthdid_effect_curve tests
# ---------------------------------------------------------------------------

class TestEffectCurve:
    def test_length(self, prop99_estimate):
        est, setup = prop99_estimate
        curve = synthdid_effect_curve(est)
        T1 = setup["Y"].shape[1] - setup["T0"]
        assert len(curve) == T1

    def test_values_match_r(self, prop99_estimate):
        """Effect curve should match R to 2 decimal places."""
        est, _ = prop99_estimate
        curve = synthdid_effect_curve(est)
        expected = [-4.84, -4.33, -8.65, -8.42, -12.55, -16.11,
                    -18.91, -19.35, -20.88, -22.78, -25.94, -24.48]
        for got, exp in zip(curve, expected):
            assert abs(got - exp) < 0.01

    def test_average_equals_estimate(self, prop99_estimate):
        """The lambda-weighted average of the curve should equal the point estimate."""
        est, _ = prop99_estimate
        curve = synthdid_effect_curve(est)
        # The point estimate is the uniform average over post-treatment periods
        T1 = len(curve)
        assert abs(np.mean(curve) - float(est)) < 0.01


# ---------------------------------------------------------------------------
# synthdid_controls tests
# ---------------------------------------------------------------------------

class TestControls:
    def test_omega_returns_dataframe(self, prop99_estimate):
        est, _ = prop99_estimate
        df = synthdid_controls(est, weight_type="omega")
        assert isinstance(df, pd.DataFrame)

    def test_omega_sorted_descending(self, prop99_estimate):
        est, _ = prop99_estimate
        df = synthdid_controls(est, weight_type="omega")
        vals = df.iloc[:, 0].values
        assert np.all(vals[:-1] >= vals[1:])

    def test_omega_top_unit_is_nevada(self, prop99_estimate):
        est, _ = prop99_estimate
        df = synthdid_controls(est, weight_type="omega")
        assert df.index[0] == "Nevada"

    def test_lambda_top_period_is_1988(self, prop99_estimate):
        est, _ = prop99_estimate
        df = synthdid_controls(est, weight_type="lambda")
        assert df.index[0] == "1988"

    def test_mass_coverage(self, prop99_estimate):
        """Cumulative weight of returned rows should be >= mass."""
        est, _ = prop99_estimate
        for mass in [0.8, 0.9, 0.95]:
            df = synthdid_controls(est, weight_type="omega", mass=mass)
            assert df.iloc[:, 0].sum() >= mass - 1e-9

    def test_top_n(self, prop99_estimate):
        est, _ = prop99_estimate
        df = synthdid_controls(est, top_n=5, weight_type="omega")
        assert len(df) == 5


# ---------------------------------------------------------------------------
# SynthdidEstimate method API tests
# ---------------------------------------------------------------------------

class TestSynthdidEstimateMethods:
    def test_summary_returns_results(self, prop99_estimate):
        est, _ = prop99_estimate
        results = est.summary(se_method="placebo", replications=50)
        assert isinstance(results, SynthDIDResults)

    def test_summary_tau_matches_estimate(self, prop99_estimate):
        est, _ = prop99_estimate
        results = est.summary(se_method="placebo", replications=50)
        assert abs(results.tau - float(est)) < 1e-9

    def test_summary_se_positive(self, prop99_estimate):
        est, _ = prop99_estimate
        results = est.summary(se_method="placebo", replications=50)
        assert results.se > 0

    def test_summary_conf_int_ordered(self, prop99_estimate):
        est, _ = prop99_estimate
        results = est.summary(se_method="placebo", replications=50)
        assert results.conf_int[0] < results.conf_int[1]

    def test_summary_str_renders(self, prop99_estimate):
        est, _ = prop99_estimate
        results = est.summary(se_method="placebo", replications=50)
        s = str(results)
        assert "tau" in s
        assert "SE Method" in s

    def test_effect_curve_method(self, prop99_estimate):
        est, _ = prop99_estimate
        curve = est.effect_curve()
        assert len(curve) == est.setup["Y"].shape[1] - est.setup["T0"]

    def test_effect_curve_detail_method(self, prop99_estimate):
        est, _ = prop99_estimate
        detail = est.effect_curve(detail=True)
        assert hasattr(detail, "tau")
        assert hasattr(detail, "actual")
        assert hasattr(detail, "predicted")

    def test_top_weights_omega(self, prop99_estimate):
        est, _ = prop99_estimate
        df = est.top_weights(top_n=5, weight_type="omega")
        assert len(df) == 5
        assert df.index[0] == "Nevada"

    def test_top_weights_lambda(self, prop99_estimate):
        est, _ = prop99_estimate
        df = est.top_weights(top_n=3, weight_type="lambda")
        assert len(df) == 3
        assert df.index[0] == "1988"

    def test_plot_returns_figure(self, prop99_estimate):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        est, _ = prop99_estimate
        fig = est.plot()
        assert fig is not None
        plt.close(fig)

    def test_weights_plot_returns_figure(self, prop99_estimate):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        est, _ = prop99_estimate
        fig = est.weights_plot(top_n=5)
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# SynthDID class tests
# ---------------------------------------------------------------------------

class TestSynthDID:
    def test_fit_dataframe(self):
        df = pd.read_csv(DATA_PATH, sep=None, engine="python")
        model = SynthDID()
        result = model.fit(df, unit="State", time="Year",
                           outcome="PacksPerCapita", treatment="treated")
        assert result is model  # returns self
        assert model.result_ is not None

    def test_fit_matrix(self, prop99_estimate):
        est, setup = prop99_estimate
        model = SynthDID()
        model.fit(Y=setup["Y"], N0=setup["N0"], T0=setup["T0"])
        assert abs(float(model.result_) - float(est)) < 1e-9

    def test_fit_missing_args_raises(self):
        model = SynthDID()
        with pytest.raises(ValueError):
            model.fit(data=None)

    def test_unfitted_raises(self):
        model = SynthDID()
        with pytest.raises(RuntimeError):
            model.summary()

    def test_summary_delegates_to_result(self, prop99_estimate):
        _, setup = prop99_estimate
        model = SynthDID()
        model.fit(Y=setup["Y"], N0=setup["N0"], T0=setup["T0"])
        results = model.summary(se_method="placebo", replications=50)
        assert isinstance(results, SynthDIDResults)

    def test_repr_unfitted(self):
        model = SynthDID()
        assert "not fitted" in repr(model)

    def test_repr_fitted(self, prop99_estimate):
        _, setup = prop99_estimate
        model = SynthDID()
        model.fit(Y=setup["Y"], N0=setup["N0"], T0=setup["T0"])
        assert "SynthdidEstimate" in repr(model)
