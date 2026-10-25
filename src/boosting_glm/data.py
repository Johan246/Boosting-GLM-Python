"""Data download, loading, normalization, and reproducible splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyreadr
from sklearn.model_selection import train_test_split

from .config import DATASETS, RAW_BASE_URL, DatasetSpec


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    spec: DatasetSpec


def download_data(output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Download the four source datasets from the original public repository."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for spec in DATASETS.values():
        destination = output / spec.filename
        if destination.exists() and not overwrite:
            continue
        with urlopen(f"{RAW_BASE_URL}/{spec.filename}", timeout=120) as response:
            destination.write_bytes(response.read())
        downloaded.append(destination)
    return downloaded


def _read_rdata(path: Path, object_name: str | None) -> pd.DataFrame:
    objects = pyreadr.read_r(path)
    if object_name and object_name in objects:
        return objects[object_name]
    frames = [value for value in objects.values() if isinstance(value, pd.DataFrame)]
    if len(frames) != 1:
        names = ", ".join(str(name) for name in objects)
        raise ValueError(f"Expected one data frame in {path}; found: {names}")
    return frames[0]


def load_dataset(name: str, data_dir: str | Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Load a source dataset and standardize target/exposure names."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; choose one of {', '.join(DATASETS)}")
    spec = DATASETS[name]
    path = Path(data_dir) / spec.filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `boosting-glm download-data --output {Path(data_dir)}` first."
        )
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else _read_rdata(path, spec.object_name)
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=42)
    required = set(spec.features) | {spec.target, spec.exposure}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, list(spec.features) + [spec.target, spec.exposure]].copy()
    frame = frame.rename(columns={spec.target: "frequency", spec.exposure: "exposure"})
    frame["frequency"] = pd.to_numeric(frame["frequency"], errors="raise").clip(lower=0)
    frame["exposure"] = pd.to_numeric(frame["exposure"], errors="raise").clip(lower=1e-8)
    for column in spec.categorical:
        frame[column] = frame[column].astype("string").fillna("<missing>")
    return frame.reset_index(drop=True)


def split_dataset(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    *,
    train_fraction: float = 0.8,
    calibration_fraction: float = 0.3,
    random_state: int = 42,
) -> DatasetSplit:
    """Reproduce the R workflow's 56/24/20 train/calibration/test proportions."""
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    build, test = train_test_split(frame, train_size=train_fraction, random_state=random_state)
    train, calibration = train_test_split(
        build, test_size=calibration_fraction, random_state=random_state
    )
    return DatasetSplit(
        train=train.reset_index(drop=True),
        calibration=calibration.reset_index(drop=True),
        test=test.reset_index(drop=True),
        spec=spec,
    )


def synthetic_portfolio(rows: int = 2_000, *, random_state: int = 42) -> tuple[pd.DataFrame, DatasetSpec]:
    """Create a small insurance-frequency portfolio for examples and tests."""
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 86, rows)
    region = rng.choice(["north", "south", "city"], rows, p=[0.25, 0.35, 0.4])
    vehicle = rng.choice(["small", "medium", "large"], rows)
    exposure = rng.uniform(0.05, 1.0, rows)
    rate = np.exp(-3.3 + 0.012 * (age - 45) + 0.35 * (region == "city") + 0.2 * (vehicle == "large"))
    claims = rng.poisson(rate * exposure)
    frame = pd.DataFrame(
        {"age": age, "region": region, "vehicle": vehicle, "frequency": claims, "exposure": exposure}
    )
    spec = DatasetSpec(
        filename="synthetic.csv",
        object_name=None,
        target="frequency",
        exposure="exposure",
        features=("age", "region", "vehicle"),
        categorical=("region", "vehicle"),
    )
    return frame, spec

