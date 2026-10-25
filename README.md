# Boosting GLM — Python port

A portable Python implementation of [`Johan246/Boosting-GLM`](https://github.com/Johan246/Boosting-GLM), an actuarial claim-frequency workflow that distills a nonlinear gradient-boosting model into an interpretable generalized linear model.

## What was translated

The original R scripts train a conventional Poisson GLM, a shallow Poisson GBM, extract marginal effects, auto-calibrate those effects with regression trees, and combine them in a final regularized Poisson GLM. This port preserves that architecture while replacing hard-coded Windows paths and serialized R workspaces with a Python package, command-line interface, reproducible splits, `joblib` models, CSV/JSON reports, tests, and CI.

| Original R component | Python equivalent |
|---|---|
| `main.r` | `boosting_glm.cli` and `boosting_glm.pipeline` |
| `load_packages.r` | `pyproject.toml` dependencies |
| dataset imports in `modeling.r` | `boosting_glm.data` |
| quasipoisson benchmark | scikit-learn `PoissonRegressor` |
| `gbm(..., distribution="poisson")` | `HistGradientBoostingRegressor(loss="poisson")` |
| PDP extraction | empirical marginal-effect curves |
| `rpart` auto-calibration | `DecisionTreeRegressor` per effect |
| final `glm`/`glmnet` | regularized `PoissonRegressor` |
| `vis_and_eval.r` and `summary.r` | `boosting_glm.metrics` and `boosting_glm.plots` |
| `.RData` model artifacts | `joblib`, CSV, JSON, and PNG outputs |

The Python implementation uses exposure-weighted frequency models. Fitting `claim_count / exposure` with `exposure` as sample weight is the standard scikit-learn equivalent of a Poisson count model with `log(exposure)` as an offset.

## Install

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Download the original datasets

The datasets remain in the original public repository instead of being duplicated here:

```bash
boosting-glm download-data --output data/raw
```

The loader supports the original CSV and `.rda` formats through `pandas` and `pyreadr`.

## Train a model

Run one of `freMTPL`, `auspriv`, `beMTPL`, or `norauto`:

```bash
boosting-glm train \
  --dataset norauto \
  --data-dir data/raw \
  --output outputs/norauto
```

For a quick smoke run:

```bash
boosting-glm train --dataset norauto --data-dir data/raw --max-rows 5000 --trees 50
```

The output directory contains:

- `model.joblib`: fitted pipeline;
- `metrics_calibration.csv` and `metrics_test.csv`;
- `report.json`: machine-readable run summary;
- `predictions.png`: observed and modeled frequency by risk ventile;
- `cumulative_claims.png`: cumulative claims by predicted risk;
- `effects/`: marginal GBM effect plots.

## Use from Python

```python
from boosting_glm.config import DATASETS
from boosting_glm.data import load_dataset, split_dataset
from boosting_glm.model import BoostedGLM

spec = DATASETS["norauto"]
data = load_dataset("norauto", "data/raw")
split = split_dataset(data, spec)

model = BoostedGLM(categorical_features=spec.categorical)
model.fit(
    split.train[list(spec.features)],
    split.train["frequency"],
    split.train["exposure"],
)

predictions = model.predict(
    split.test[list(spec.features)],
    split.test["exposure"],
)
```

## Complete notebook example

[`examples/complete_method_example.ipynb`](examples/complete_method_example.ipynb) is an executed,
self-contained walkthrough of the complete method. It creates a synthetic insurance portfolio, fits
all three model stages, inspects the extracted and calibrated effects, evaluates holdout results,
plots predictions, and verifies model serialization.

```bash
python -m pip install -e ".[notebook]"
jupyter lab examples/complete_method_example.ipynb
```

## Deliberate differences from the R code

- Paths are supplied by the caller; there are no personal OneDrive or corporate SMB paths.
- Splitting is randomized but reproducible with seed 42. The proportions remain 56% training, 24% calibration, and 20% test.
- The original `glmnet` L1 step is represented by an L2-regularized Poisson GLM because scikit-learn provides a stable, offset-equivalent exposure-weighted implementation. Effect selection still occurs through shallow GBM structure and auto-calibration trees.
- Interaction candidates are selected from the most important GBM features and added as calibrated effect products. This is a computationally lighter analogue of the original Friedman-H scan.
- Predictions retain the original default cap of two claims; set `max_prediction=None` when constructing `BoostedGLM` to disable it.
- The source repository's private `REAL` dataset workflow is intentionally excluded because neither its SAS data nor its import script is published.

## Tests

```bash
pytest
ruff check .
```

Tests cover data splitting, Poisson metrics, mixed numeric/categorical training, prediction, and model serialization.

## Data and licensing note

This repository contains code only. `download-data` retrieves data from the original repository. Review the original datasets' provenance and applicable terms before redistribution or production use. The original repository does not declare a software license, so this port does not assert one on its behalf.
