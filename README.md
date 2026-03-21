# synthdid — Synthetic Difference-in-Differences for Python

[![GitHub](https://img.shields.io/badge/github-AllardJM%2Fsynthdid--python-blue)](https://github.com/AllardJM/synthdid-python)

A Python port of the [R synthdid package](https://github.com/synth-inference/synthdid) by Arkhangelsky et al.

Implements the **Synthetic Difference-in-Differences** estimator from:

> Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
> *Synthetic Difference in Differences*. American Economic Review, 111(12), 4088–4118.
> https://arxiv.org/abs/1812.09970

## Installation

```bash
git clone https://github.com/AllardJM/synthdid-python.git
cd synthdid-python
pip install -r requirements.txt
pip install .
```

## Quick Start

Replicating the canonical California Proposition 99 example:

```python
import numpy as np
import pandas as pd
from synthdid import panel_matrices, synthdid_estimate, vcov, synthdid_plot

# Load data (long panel: unit, time, outcome, treatment)
df = pd.read_csv("data/california_prop99.csv", sep=";")

# Convert to matrix format
setup = panel_matrices(df, unit="State", time="Year",
                       outcome="PacksPerCapita", treatment="treated")

# Estimate treatment effect
tau_hat = synthdid_estimate(setup["Y"], setup["N0"], setup["T0"],
                             unit_names=setup["unit_names"],
                             time_names=setup["time_names"])

# Standard error via placebo method (recommended)
se = np.sqrt(vcov(tau_hat, method="placebo", replications=200))

print(f"Point estimate: {float(tau_hat):.2f}")
print(f"95% CI: ({float(tau_hat) - 1.96*se:.2f}, {float(tau_hat) + 1.96*se:.2f})")
# Point estimate: -15.60
# 95% CI: (-33.80, 2.59)

# Plot
fig = synthdid_plot(tau_hat, se=se)
fig.savefig("synthdid_plot.png", dpi=150, bbox_inches="tight")
```

## API Reference

### `panel_matrices(panel, unit, time, outcome, treatment, treated_last=True)`
Convert a long-format balanced panel DataFrame to the matrix format required
by the estimator. Returns a dict with keys `Y`, `N0`, `T0`, `W`,
`unit_names`, `time_names`.

### `synthdid_estimate(Y, N0, T0, ...)`
Estimate the average treatment effect on the treated using Algorithm 1 of
Arkhangelsky et al. Returns a `SynthdidEstimate` object.

### `vcov(estimate, method='placebo', replications=200)`
Estimate variance (SE²). Three methods:
- **`placebo`** *(recommended)*: works with any number of treated units, requires N0 > N1
- **`bootstrap`**: requires more than one treated unit
- **`jackknife`**: fastest, but not recommended for synthetic control estimates (can underestimate variance); returns NaN with a single treated unit

### `synthdid_effect_curve(estimate)`
Period-by-period treatment effect for each post-treatment time period.
Returns an array of length T1.

### `synthdid_controls(estimate, weight_type='omega', mass=0.9)`
Table of the most influential control units (`weight_type='omega'`) or
pre-treatment periods (`weight_type='lambda'`), sorted by weight and
truncated to cover at least `mass` fraction of total weight.

### `synthdid_plot(estimate, se=None)`
Matplotlib figure showing treated and synthetic control trajectories,
the 2×2 DiD diagram, treatment effect arrow, and lambda weight distribution.

## Validated Results

Results are validated against the R synthdid package on the California Prop 99 dataset:

| Metric | R | Python |
|---|---|---|
| Point estimate | -15.60 | -15.60 |
| Effect curve (1989–2000) | [-4.84, -4.33, ..., -24.48] | identical |
| All omega weights | Nevada: 0.1245, NH: 0.1050, ... | identical |
| All lambda weights | 1988: 0.4271, 1986: 0.3665, ... | identical |

## Dependencies

- Python ≥ 3.9
- numpy
- pandas
- matplotlib

## License

BSD 3-Clause License. See [LICENSE](LICENSE).

This project is a port of the R synthdid package, which is copyright 2019,
Stanford University, also under the BSD 3-Clause License.
