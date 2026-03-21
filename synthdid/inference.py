"""
Variance estimation for synthetic difference-in-differences estimates.

Implements Algorithms 2 (bootstrap), 3 (jackknife), and 4 (placebo) from
Arkhangelsky et al. (2021): "Synthetic Difference in Differences"
https://arxiv.org/abs/1812.09970

Ported from R/vcov.R in the synthdid package.
"""

import numpy as np

from .estimator import synthdid_estimate


def vcov(estimate, method="placebo", replications=200, seed=None):
    """
    Estimate the variance of a SynthdidEstimate via bootstrap, jackknife, or placebo.

    Parameters
    ----------
    estimate : SynthdidEstimate
        Output of synthdid_estimate().
    method : {'placebo', 'bootstrap', 'jackknife'}
        Variance estimation method:
          - 'placebo'   : Algorithm 4. Randomly assigns placebo treatment to
                          subsets of control units. Recommended; works even
                          with only one treated unit.
          - 'bootstrap' : Algorithm 2. Resamples all units with replacement.
                          Requires more than one treated unit.
          - 'jackknife' : Algorithm 3. Leave-one-unit-out. Fastest, but not
                          recommended for synthetic control estimates.
    replications : int
        Number of replications for bootstrap and placebo methods. Default 200.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    float
        Estimated variance (SE²). Take np.sqrt() to get the standard error.

    Raises
    ------
    ValueError
        If the chosen method is incompatible with the data structure.
    """
    if seed is not None:
        np.random.seed(seed)

    method = method.lower()
    if method == "bootstrap":
        se = _bootstrap_se(estimate, replications)
    elif method == "jackknife":
        se = _jackknife_se(estimate)
    elif method == "placebo":
        se = _placebo_se(estimate, replications)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'placebo', 'bootstrap', or 'jackknife'.")

    return se ** 2  # return variance (SE^2), consistent with R's vcov()


def _sum_normalize(x):
    """
    Normalize x to sum to 1. Falls back to uniform weights if sum is zero.
    """
    x = np.asarray(x, dtype=float)
    s = x.sum()
    if s != 0:
        return x / s
    return np.ones_like(x) / len(x)


def _re_estimate(estimate, Y_sub, N0_sub, idx_control, weights_override=None):
    """
    Re-run synthdid_estimate on a subset of units with fixed lambda weights
    and re-normalized omega weights.

    Parameters
    ----------
    estimate : SynthdidEstimate
        Original estimate (provides opts and lambda weights).
    Y_sub : ndarray
        Subsetted outcome matrix.
    N0_sub : int
        Number of control rows in Y_sub.
    idx_control : array-like
        Original indices (within N0) of the control units included in Y_sub.
    weights_override : dict or None
        If provided, use these weights instead of building from estimate.

    Returns
    -------
    float
        New estimate scalar.
    """
    setup = estimate.setup
    opts = estimate.opts
    weights = estimate.weights

    if weights_override is not None:
        w = weights_override
    else:
        # Re-normalize omega to the sampled control indices
        omega_sub = _sum_normalize(weights["omega"][idx_control])
        w = dict(weights)
        w["omega"] = omega_sub

    # Build X subset (no covariates in the basic case)
    X_sub = setup["X"][: Y_sub.shape[0]] if setup["X"].shape[2] > 0 else None

    result = synthdid_estimate(
        Y=Y_sub,
        N0=N0_sub,
        T0=setup["T0"],
        X=X_sub,
        weights=w,
        update_omega=False,   # fix omega weights
        update_lambda=False,  # fix lambda weights
        **{k: opts[k] for k in ("zeta_omega", "zeta_lambda", "omega_intercept",
                                 "lambda_intercept", "min_decrease", "max_iter")},
    )
    return float(result)


# ---------------------------------------------------------------------------
# Algorithm 4: Placebo SE (recommended)
# ---------------------------------------------------------------------------

def _placebo_se(estimate, replications):
    """
    Placebo standard error (Algorithm 4 of Arkhangelsky et al.).

    Randomly assigns N1 control units as 'placebo treated' and estimates
    a placebo treatment effect. The SD of these placebo estimates gives
    the standard error.

    Requires N0 > N1 (more controls than treated units).
    """
    setup = estimate.setup
    opts = estimate.opts
    weights = estimate.weights

    N = setup["Y"].shape[0]
    N0 = setup["N0"]
    N1 = N - N0

    if N0 <= N1:
        raise ValueError(
            "Placebo SE requires more control units than treated units (N0 > N1)."
        )

    placebo_estimates = np.zeros(replications)
    for r in range(replications):
        # Sample N0 indices from the control pool (without replacement → subset)
        ind = np.random.choice(N0, size=N0, replace=False)
        # First N0-N1 are placebo controls; last N1 are placebo treated
        N0_placebo = N0 - N1
        ctrl_idx = ind[:N0_placebo]
        treat_idx = ind[N0_placebo:]

        # Stack: placebo controls first, placebo treated last
        row_order = np.concatenate([ctrl_idx, treat_idx])
        Y_sub = setup["Y"][row_order]
        X_sub = setup["X"][row_order] if setup["X"].shape[2] > 0 else None

        # Re-normalize omega to the placebo control subset
        omega_sub = _sum_normalize(weights["omega"][ctrl_idx])
        w = dict(weights)
        w["omega"] = omega_sub

        result = synthdid_estimate(
            Y=Y_sub,
            N0=N0_placebo,
            T0=setup["T0"],
            X=X_sub,
            weights=w,
            update_omega=False,
            update_lambda=False,
            **{k: opts[k] for k in ("zeta_omega", "zeta_lambda", "omega_intercept",
                                     "lambda_intercept", "min_decrease", "max_iter")},
        )
        placebo_estimates[r] = float(result)

    # Finite-sample corrected SD (matches R: sqrt((B-1)/B) * sd(...))
    B = replications
    se = np.sqrt((B - 1) / B) * np.std(placebo_estimates, ddof=1)
    return se


# ---------------------------------------------------------------------------
# Algorithm 2: Bootstrap SE
# ---------------------------------------------------------------------------

def _bootstrap_se(estimate, replications):
    """
    Bootstrap standard error (Algorithm 2 of Arkhangelsky et al.).

    Resamples all N units with replacement, re-runs the estimator with fixed
    lambda and re-normalized omega. Requires more than one treated unit.
    """
    setup = estimate.setup
    opts = estimate.opts
    weights = estimate.weights

    N = setup["Y"].shape[0]
    N0 = setup["N0"]
    N1 = N - N0

    if N0 == N - 1:
        raise ValueError(
            "Bootstrap SE is not defined when there is only one treated unit."
        )

    bootstrap_estimates = []
    attempts = 0
    max_attempts = replications * 20  # avoid infinite loop on degenerate samples

    while len(bootstrap_estimates) < replications and attempts < max_attempts:
        attempts += 1
        ind = np.random.choice(N, size=N, replace=True)

        # Check that both control and treated units are present
        has_control = np.any(ind < N0)
        has_treated = np.any(ind >= N0)
        if not (has_control and has_treated):
            continue

        sorted_ind = np.sort(ind)
        ctrl_in_sample = sorted_ind[sorted_ind < N0]
        treat_in_sample = sorted_ind[sorted_ind >= N0]
        N0_boot = len(ctrl_in_sample)

        row_order = np.concatenate([ctrl_in_sample, treat_in_sample])
        Y_boot = setup["Y"][row_order]
        X_boot = setup["X"][row_order] if setup["X"].shape[2] > 0 else None

        # Re-normalize omega to the sampled (possibly repeated) control units
        omega_boot = _sum_normalize(weights["omega"][ctrl_in_sample])
        w = dict(weights)
        w["omega"] = omega_boot

        result = synthdid_estimate(
            Y=Y_boot,
            N0=N0_boot,
            T0=setup["T0"],
            X=X_boot,
            weights=w,
            update_omega=False,
            update_lambda=False,
            **{k: opts[k] for k in ("zeta_omega", "zeta_lambda", "omega_intercept",
                                     "lambda_intercept", "min_decrease", "max_iter")},
        )
        bootstrap_estimates.append(float(result))

    boot_arr = np.array(bootstrap_estimates)
    B = len(boot_arr)
    se = np.sqrt((B - 1) / B) * np.std(boot_arr, ddof=1)
    return se


# ---------------------------------------------------------------------------
# Algorithm 3: Jackknife SE
# ---------------------------------------------------------------------------

def _jackknife_se(estimate):
    """
    Fixed-weights jackknife standard error (Algorithm 3 of Arkhangelsky et al.).

    Leaves one unit out at a time and re-runs the estimator with fixed weights.

    Note: Not recommended for SC estimates (see Section 5 of the paper).
    Returns NaN if there is only one treated unit or one non-zero control weight.
    """
    setup = estimate.setup
    opts = estimate.opts
    weights = estimate.weights

    N = setup["Y"].shape[0]
    N0 = setup["N0"]
    N1 = N - N0

    # Return NaN for degenerate cases
    if N1 == 1:
        return np.nan
    if np.sum(weights["omega"] != 0) == 1:
        return np.nan

    loo_estimates = np.zeros(N)

    for i in range(N):
        ind = np.array([j for j in range(N) if j != i])
        ctrl_idx = ind[ind < N0]
        N0_loo = len(ctrl_idx)

        row_order = np.concatenate([ctrl_idx, ind[ind >= N0]])
        Y_loo = setup["Y"][row_order]
        X_loo = setup["X"][row_order] if setup["X"].shape[2] > 0 else None

        # Re-normalize omega to remaining control units
        omega_loo = _sum_normalize(weights["omega"][ctrl_idx])
        w = dict(weights)
        w["omega"] = omega_loo

        result = synthdid_estimate(
            Y=Y_loo,
            N0=N0_loo,
            T0=setup["T0"],
            X=X_loo,
            weights=w,
            update_omega=False,
            update_lambda=False,
            **{k: opts[k] for k in ("zeta_omega", "zeta_lambda", "omega_intercept",
                                     "lambda_intercept", "min_decrease", "max_iter")},
        )
        loo_estimates[i] = float(result)

    # Jackknife SE formula: sqrt(((n-1)/n) * (n-1) * var(u))
    n = N
    se = np.sqrt(((n - 1) / n) * (n - 1) * np.var(loo_estimates, ddof=1))
    return se
