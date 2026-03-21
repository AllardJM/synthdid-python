"""
Visualization for synthetic difference-in-differences estimates.

Produces a plot analogous to plot.synthdid_estimate() in the R synthdid package,
using matplotlib instead of ggplot2.

Ported from R/plot.R in the synthdid package by Arkhangelsky et al.
Reference: https://arxiv.org/abs/1812.09970
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def synthdid_plot(estimate, se=None, treated_name="Treated", control_name="Synthetic Control",
                  figsize=(10, 6)):
    """
    Plot a synthetic difference-in-differences estimate.

    Creates a two-panel figure:
      - Main panel: time-series trajectories for the observed treated average
        and the synthetic control, plus a 2×2 DiD diagram.
      - Bottom panel: lambda (time) weight distribution over pre-treatment periods.

    Parameters
    ----------
    estimate : SynthdidEstimate
        Output of synthdid_estimate().
    se : float or None
        Standard error of the estimate (sqrt of vcov output). If provided,
        a 95% confidence interval is shown as an annotation.
    treated_name : str
        Legend label for the treated trajectory. Default 'Treated'.
    control_name : str
        Legend label for the synthetic control trajectory. Default 'Synthetic Control'.
    figsize : tuple
        Figure size in inches. Default (10, 6).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    setup = estimate.setup
    weights = estimate.weights

    Y = setup["Y"]
    N0, T0 = setup["N0"], setup["T0"]
    N, T = Y.shape
    N1 = N - N0
    T1 = T - T0

    # Time axis labels
    time_names = estimate.time_names if estimate.time_names else list(range(T))

    # -------------------------------------------------------------------------
    # Build weight vectors for full trajectory computation
    # -------------------------------------------------------------------------
    omega = weights["omega"]           # (N0,) — control unit weights
    lambda_ = weights["lambda"]        # (T0,) — pre-treatment time weights

    # omega_synth: put omega weights on controls, zero on treated
    omega_synth = np.concatenate([omega, np.zeros(N1)])   # (N,)
    # omega_target: uniform over treated units
    omega_target = np.concatenate([np.zeros(N0), np.ones(N1) / N1])  # (N,)

    # lambda_synth: put lambda weights on pre-treatment, zero on post
    lambda_synth = np.concatenate([lambda_, np.zeros(T1)])   # (T,)
    # lambda_target: uniform over post-treatment periods
    lambda_target = np.concatenate([np.zeros(T0), np.ones(T1) / T1])  # (T,)

    # -------------------------------------------------------------------------
    # Compute trajectories
    # -------------------------------------------------------------------------
    obs_trajectory = omega_target @ Y        # (T,) — observed treated average
    syn_trajectory = omega_synth @ Y         # (T,) — synthetic control

    # -------------------------------------------------------------------------
    # Compute 2×2 DiD diagram corners
    # -------------------------------------------------------------------------
    treated_pre  = float(omega_target @ Y @ lambda_synth)   # treated,  pre-treatment avg
    treated_post = float(omega_target @ Y @ lambda_target)  # treated,  post-treatment avg
    control_pre  = float(omega_synth  @ Y @ lambda_synth)   # control,  pre-treatment avg
    control_post = float(omega_synth  @ Y @ lambda_target)  # control,  post-treatment avg
    sdid_post    = control_post + treated_pre - control_pre  # counterfactual for treated

    # Scalar x-coordinates for the 2×2 diagram
    x_pre  = float(np.arange(T)[: T0] @ lambda_)    # weighted pre-treatment time index
    x_post = float(np.mean(np.arange(T)[T0:]))       # mean post-treatment time index

    # -------------------------------------------------------------------------
    # Set up figure layout: main plot (top) + lambda weights (bottom)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 1, figure=fig, hspace=0.05)
    ax_main = fig.add_subplot(gs[:3, 0])
    ax_lambda = fig.add_subplot(gs[3, 0], sharex=ax_main)

    x = np.arange(T)

    # -------------------------------------------------------------------------
    # Main panel: trajectories
    # -------------------------------------------------------------------------
    ax_main.plot(x, obs_trajectory, color="black", linewidth=2,
                 label=treated_name, zorder=3)
    ax_main.plot(x, syn_trajectory, color="steelblue", linewidth=2,
                 linestyle="--", label=control_name, zorder=3)

    # Vertical line at treatment onset
    ax_main.axvline(x=T0 - 0.5, color="gray", linestyle=":", linewidth=1.2, zorder=1)

    # -------------------------------------------------------------------------
    # 2×2 DiD diagram: four corner points and connecting lines
    # -------------------------------------------------------------------------
    did_color = "#999999"
    # Horizontal lines connecting pre-to-post for treated and counterfactual
    ax_main.plot([x_pre, x_post], [treated_pre, treated_post],
                 color=did_color, linewidth=1.5, linestyle="-", zorder=2)
    ax_main.plot([x_pre, x_post], [control_pre, sdid_post],
                 color=did_color, linewidth=1.5, linestyle="-", zorder=2)
    # Vertical line at pre-treatment point connecting treated and control
    ax_main.plot([x_pre, x_pre], [control_pre, treated_pre],
                 color=did_color, linewidth=1.5, linestyle="-", zorder=2)

    # Corner dots
    for px, py in [(x_pre, treated_pre), (x_pre, control_pre),
                   (x_post, treated_post), (x_post, sdid_post)]:
        ax_main.scatter(px, py, color=did_color, s=40, zorder=4)

    # -------------------------------------------------------------------------
    # Treatment effect arrow (from counterfactual to observed post-treatment)
    # -------------------------------------------------------------------------
    tau_hat = float(estimate)
    ax_main.annotate(
        "",
        xy=(x_post, treated_post),
        xytext=(x_post, sdid_post),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2),
        zorder=5,
    )

    # Tau label with optional CI
    mid_y = (treated_post + sdid_post) / 2
    if se is not None:
        ci_lo = tau_hat - 1.96 * se
        ci_hi = tau_hat + 1.96 * se
        label_text = f"τ̂ = {tau_hat:.2f}\n95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]"
    else:
        label_text = f"τ̂ = {tau_hat:.2f}"

    ax_main.text(
        x_post + 0.3, mid_y, label_text,
        va="center", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    # Shaded post-treatment region
    ax_main.axvspan(T0 - 0.5, T - 0.5, alpha=0.04, color="orange", zorder=0)

    ax_main.set_ylabel("Outcome")
    ax_main.legend(loc="upper left", framealpha=0.8)
    ax_main.set_title("Synthetic Difference-in-Differences")

    # Hide x-tick labels on main panel (shared with lambda panel)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # -------------------------------------------------------------------------
    # Bottom panel: lambda weight distribution (pre-treatment periods)
    # -------------------------------------------------------------------------
    pre_x = np.arange(T0)
    bar_colors = ["steelblue" if w > 0 else "lightgray" for w in lambda_]
    ax_lambda.bar(pre_x, lambda_, color=bar_colors, alpha=0.7, width=0.8)
    ax_lambda.axhline(1 / T0, color="gray", linestyle="--", linewidth=0.8,
                      label=f"Uniform (1/{T0})")
    ax_lambda.set_ylabel("λ weight", fontsize=8)
    ax_lambda.set_ylim(bottom=0)

    # X-axis ticks and labels
    tick_step = max(1, T // 10)  # show ~10 ticks
    tick_positions = np.arange(0, T, tick_step)
    tick_labels = [str(time_names[i]) for i in tick_positions if i < len(time_names)]
    ax_lambda.set_xticks(tick_positions)
    ax_lambda.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax_lambda.set_xlabel("Time")

    # Shade post-treatment in lambda panel too
    ax_lambda.axvspan(T0 - 0.5, T - 0.5, alpha=0.04, color="orange")

    plt.tight_layout()
    return fig
