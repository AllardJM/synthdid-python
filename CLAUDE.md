# CLAUDE.md

## Project

Python port of the R `synthdid` package (Arkhangelsky et al. 2021).
Reference: https://arxiv.org/abs/1812.09970

## Running tests

```bash
.venv/bin/pytest tests/
```

The `.venv` virtual environment is local-only (gitignored). To recreate it:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install ipykernel
.venv/bin/python -m ipykernel install --user --name synthdid_python --display-name "Python (synthdid)"
```

## Repository structure

```
synthdid/
  panel.py       — panel_matrices(): long DataFrame → Y matrix, N0, T0
  solver.py      — Frank-Wolfe optimiser for omega and lambda weights
  estimator.py   — synthdid_estimate() + SynthdidEstimate result object
  inference.py   — vcov(): placebo / bootstrap / jackknife variance
  summary.py     — synthdid_effect_curve(), synthdid_controls()
  results.py     — SynthDIDResults: statsmodels-style summary table
  plot.py        — synthdid_plot(), synthdid_weights_plot()
  validation.py  — synthdid_out_of_time(), synthdid_oot_plot()
data/
  california_prop99.csv  — semicolon-delimited (sep=";")
examples/
  california_prop99.ipynb
```

## API design

`synthdid_estimate()` returns a `SynthdidEstimate` object. All analysis
methods live directly on that object — there is no separate wrapper class:

```python
est = synthdid_estimate(Y, N0, T0, unit_names=..., time_names=...)
est.summary()           # SynthDIDResults — statsmodels-style table
est.plot()              # synthdid_plot
est.weights_plot()      # bar charts of omega / lambda weights
est.effect_curve()      # per-period treatment effects (ndarray)
est.effect_curve(detail=True)  # EffectCurveDetail with actual/predicted
est.top_weights(top_n=10, weight_type="omega")  # pd.DataFrame
```

This was a deliberate choice: a sklearn-style wrapper class (`SynthDID`)
was considered and removed because the functional API with methods on the
result object is sufficient and avoids duplicating state.

## Data note

`data/california_prop99.csv` uses semicolons as separators:

```python
pd.read_csv("data/california_prop99.csv", sep=";")
# or let pandas detect it:
pd.read_csv("data/california_prop99.csv", sep=None, engine="python")
```

## Docstrings

NumPy docstring style throughout. Every public function and class needs:

```python
def my_function(x, method="placebo"):
    """
    One-line summary.

    Longer explanation if needed. Describe what the function does,
    not how it does it.

    Parameters
    ----------
    x : ndarray, shape (N, T)
        Description of x.
    method : {'placebo', 'bootstrap', 'jackknife'}
        Description of method. Default 'placebo'.

    Returns
    -------
    float
        Description of return value.

    Raises
    ------
    ValueError
        When and why this is raised.
    """
```

Key conventions:
- Shape annotations on arrays: `ndarray, shape (N, T)` or `ndarray, shape (N0,)`
- Choices shown as `{'option_a', 'option_b'}` in the type field
- Default values stated at the end of the description: `Default 'placebo'.`
- Internal helpers (prefixed `_`) do not need full docstrings but should have
  a one-line comment explaining their purpose

## Code style

- No formatter or linter is configured — keep style consistent with the
  existing code (PEP 8, ~90 char lines)
- Type hints are not used; rely on docstring shape/type annotations instead
- Inline comments explain *why*, not *what* — the code itself shows what
- Mathematical quantities use their paper names: `omega`, `lambda_`, `tau`,
  `zeta_omega`, `eta_omega` etc. — do not rename them for "clarity"
- Module-level docstrings follow the pattern:
  ```python
  """
  One-line description.

  Longer explanation if needed.

  Ported from R/<file>.R in the synthdid package by Arkhangelsky et al.
  Reference: https://arxiv.org/abs/1812.09970
  """
  ```

## Versioning

Version is defined once in `pyproject.toml` and exposed as `synthdid.__version__`.
Bump `version` in `pyproject.toml` when cutting a release.
