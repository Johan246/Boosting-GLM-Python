"""Dataset metadata translated from the formulas in the R implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    object_name: str | None
    target: str
    exposure: str
    features: tuple[str, ...]
    categorical: tuple[str, ...]
    drop: tuple[str, ...] = ()


DATASETS: dict[str, DatasetSpec] = {
    "freMTPL": DatasetSpec(
        filename="freMTPL2freq.csv",
        object_name=None,
        target="ClaimNb",
        exposure="Exposure",
        features=(
            "Area",
            "VehPower",
            "VehAge",
            "DrivAge",
            "VehBrand",
            "VehGas",
            "Density",
            "Region",
            "BonusMalus",
        ),
        categorical=("Area", "VehBrand", "VehGas", "Region"),
        drop=("IDpol",),
    ),
    "auspriv": DatasetSpec(
        filename="ausprivauto0405.rda",
        object_name="ausprivauto0405",
        target="ClaimNb",
        exposure="Exposure",
        features=("VehValue", "VehAge", "VehBody", "Gender", "DrivAge"),
        categorical=("VehAge", "VehBody", "Gender", "DrivAge"),
        drop=("ClaimOcc", "ClaimAmount"),
    ),
    "beMTPL": DatasetSpec(
        filename="mtpl_be.rda",
        object_name="mtpl_be",
        target="nclaims",
        exposure="expo",
        features=("coverage", "ageph", "sex", "bm", "power", "agec", "fuel", "use", "fleet"),
        categorical=("coverage", "sex", "fuel", "use", "fleet"),
        drop=("id", "postcode"),
    ),
    "norauto": DatasetSpec(
        filename="norauto.rda",
        object_name="norauto",
        target="NbClaim",
        exposure="Expo",
        features=("Male", "Young", "DistLimit", "GeoRegion"),
        categorical=("Male", "Young", "DistLimit", "GeoRegion"),
        drop=("ClaimAmount",),
    ),
}

SOURCE_REPOSITORY = "https://github.com/Johan246/Boosting-GLM"
RAW_BASE_URL = f"{SOURCE_REPOSITORY}/raw/main/Data"

