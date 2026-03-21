"""
Frank-Wolfe optimization routines for synthetic difference-in-differences.

Ported from R/solver.R in the synthdid package by Arkhangelsky et al.
Reference: https://arxiv.org/abs/1812.09970
"""

import numpy as np


def fw_step(A, x, b, eta, alpha=None):
    """
    One Frank-Wolfe step for the problem:
        min ||Ax - b||^2 + eta * ||x||^2   s.t. x in unit simplex

    Parameters
    ----------
    A : ndarray, shape (n, p)
        Design matrix.
    x : ndarray, shape (p,)
        Current iterate (in unit simplex).
    b : ndarray, shape (n,)
        Target vector.
    eta : float
        Ridge regularization strength.
    alpha : float or None
        If provided, use fixed step size alpha. Otherwise use exact line search.

    Returns
    -------
    x_new : ndarray, shape (p,)
        Updated iterate.
    """
    Ax = A @ x
    # Gradient of 0.5 * ||Ax - b||^2 + 0.5 * eta * ||x||^2 w.r.t. x,
    # scaled by 2 (the "half gradient")
    half_grad = (Ax - b) @ A + eta * x  # shape (p,)
    i = np.argmin(half_grad)

    if alpha is not None:
        # Fixed step: move fraction alpha toward vertex e_i
        x = x * (1 - alpha)
        x[i] += alpha
        return x
    else:
        # Exact line search
        d_x = -x.copy()
        d_x[i] += 1.0  # direction toward vertex e_i

        if np.all(d_x == 0):
            return x

        d_err = A[:, i] - Ax  # A @ d_x
        # Optimal step size from quadratic line search
        step = -(half_grad @ d_x) / (np.sum(d_err ** 2) + eta * np.sum(d_x ** 2))
        constrained_step = np.clip(step, 0.0, 1.0)
        return x + constrained_step * d_x


def sc_weight_fw(Y, zeta, intercept=True, lambda_init=None, min_decrease=1e-3, max_iter=1000):
    """
    Frank-Wolfe solver for synthetic control weights.

    Solves:
        min_{lambda in simplex} ||A @ lambda - b||^2 / N0 + zeta^2 * ||lambda||^2

    where A = Y[:, :T0] and b = Y[:, T0] (last column is the target).

    Parameters
    ----------
    Y : ndarray, shape (N0, T0+1)
        Data matrix. Last column is the post-treatment (or post-average) target.
    zeta : float
        Regularization parameter.
    intercept : bool
        If True, demean each column of Y before optimizing (removes unit-level
        fixed effects).
    lambda_init : ndarray, shape (T0,) or None
        Initial weights. Defaults to uniform 1/T0.
    min_decrease : float
        Stopping criterion: stop when objective decrease < min_decrease^2.
    max_iter : int
        Maximum number of Frank-Wolfe iterations.

    Returns
    -------
    lambda_ : ndarray, shape (T0,)
        Optimal weights (sum to 1, non-negative).
    vals : list of float
        Objective values at each iteration.
    """
    N0, T_plus1 = Y.shape
    T0 = T_plus1 - 1

    if lambda_init is None:
        lambda_ = np.ones(T0) / T0
    else:
        lambda_ = lambda_init.copy()

    if intercept:
        # Center each column by its mean (removes intercept from regression)
        Y = Y - Y.mean(axis=0)

    A = Y[:, :T0]   # (N0, T0)
    b = Y[:, T0]    # (N0,)
    eta = N0 * (zeta ** 2)

    vals = []
    for t in range(max_iter):
        lambda_ = fw_step(A, lambda_, b, eta)

        # Evaluate objective: zeta^2 * ||lambda||^2 + ||Y @ [lambda; -1]||^2 / N0
        err = Y @ np.concatenate([lambda_, [-1.0]])
        val = (zeta ** 2) * np.sum(lambda_ ** 2) + np.sum(err ** 2) / N0
        vals.append(val)

        # Check convergence after at least 2 iterations
        if t >= 1 and (vals[-2] - vals[-1]) <= min_decrease ** 2:
            break

    return lambda_, vals


def sparsify(v):
    """
    Map a weight vector to a sparser version by zeroing small entries.

    Sets entries <= max(v)/4 to zero, then renormalizes to sum to 1.

    Parameters
    ----------
    v : ndarray
        Non-negative weight vector.

    Returns
    -------
    ndarray
        Sparse weight vector summing to 1.
    """
    v = v.copy()
    v[v <= np.max(v) / 4] = 0.0
    return v / v.sum()
