"""End-to-end training and reporting orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import DATASETS
from .data import load_dataset, split_dataset
from .metrics import model_metrics
from .model import BoostedGLM
from .plots import save_cumulative_claims_plot, save_effect_plots, save_prediction_plot


def train_dataset(
    name: str,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    max_rows: int | None = None,
    trees: int = 300,
    random_state: int = 42,
) -> dict[str, object]:
    spec = DATASETS[name]
    frame = load_dataset(name, data_dir, max_rows=max_rows)
    splits = split_dataset(frame, spec, random_state=random_state)
    model = BoostedGLM(
        categorical_features=spec.categorical,
        max_iter=trees,
        random_state=random_state,
    )
    train = splits.train
    model.fit(train[list(spec.features)], train["frequency"], train["exposure"])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, object] = {"dataset": name, "rows": len(frame), "splits": {}}
    for split_name, split in (("calibration", splits.calibration), ("test", splits.test)):
        components = model.predict_components(split[list(spec.features)], split["exposure"])
        intercept = np.full(len(split), train["frequency"].sum() / train["exposure"].sum()) * split["exposure"].to_numpy()
        predictions = {"intercept": intercept, **components}
        metrics = model_metrics(split["frequency"].to_numpy(), predictions)
        metrics.to_csv(output / f"metrics_{split_name}.csv", index=False)
        reports["splits"][split_name] = metrics.to_dict(orient="records")
        if split_name == "test":
            save_prediction_plot(split["frequency"].to_numpy(), components, output / "predictions.png")
            save_cumulative_claims_plot(split["frequency"].to_numpy(), components, output / "cumulative_claims.png")

    save_effect_plots(model, output / "effects")
    model.save(output / "model.joblib")
    (output / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports

