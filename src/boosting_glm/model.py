"""Two-stage gradient-boosting to GLM distillation for claim frequency."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


@dataclass
class EffectCurve:
    feature: str
    grid: np.ndarray
    values: np.ndarray
    categorical: bool


class BoostedGLM:
    """Distill a Poisson GBM into calibrated marginal effects and a final GLM.

    The estimator follows the original R design:
    1. fit a conventional Poisson GLM benchmark;
    2. fit a shallow Poisson gradient-boosting reference model;
    3. extract marginal effects from the reference model;
    4. auto-calibrate those effects with shallow regression trees;
    5. combine calibrated effects in a regularized Poisson GLM.
    """

    def __init__(
        self,
        *,
        categorical_features: tuple[str, ...] = (),
        learning_rate: float = 0.05,
        max_iter: int = 300,
        max_depth: int = 2,
        final_alpha: float = 1e-4,
        max_effect_grid: int = 20,
        max_interactions: int = 3,
        effect_sample: int = 1_000,
        max_prediction: float | None = 2.0,
        random_state: int = 42,
    ) -> None:
        self.categorical_features = tuple(categorical_features)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.final_alpha = final_alpha
        self.max_effect_grid = max_effect_grid
        self.max_interactions = max_interactions
        self.effect_sample = effect_sample
        self.max_prediction = max_prediction
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> BoostedGLM:
        self.features_ = tuple(X.columns)
        self.categorical_ = tuple(c for c in self.categorical_features if c in X.columns)
        self.numeric_ = tuple(c for c in X.columns if c not in self.categorical_)
        y = np.asarray(y, dtype=float)
        exposure = np.clip(np.asarray(exposure, dtype=float), 1e-8, None)
        rate = y / exposure

        self.baseline_ = Pipeline(
            [
                ("preprocess", self._linear_preprocessor()),
                ("model", PoissonRegressor(alpha=1e-8, max_iter=500)),
            ]
        )
        self.baseline_.fit(X, rate, model__sample_weight=exposure)

        self.gbm_ = Pipeline(
            [
                ("preprocess", self._tree_preprocessor()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="poisson",
                        learning_rate=self.learning_rate,
                        max_iter=self.max_iter,
                        max_depth=self.max_depth,
                        min_samples_leaf=10,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.gbm_.fit(X, rate, model__sample_weight=exposure)
        self.balance_factor_ = float(np.average(rate, weights=exposure) / np.average(self.gbm_.predict(X), weights=exposure))

        background = X.sample(min(len(X), self.effect_sample), random_state=self.random_state)
        self.effect_curves_ = self._extract_effect_curves(background)
        effects = self._effect_frame(X)
        self.effect_calibrators_: dict[str, DecisionTreeRegressor] = {}
        calibrated = pd.DataFrame(index=X.index)
        for column in effects:
            tree = DecisionTreeRegressor(
                max_depth=4,
                min_samples_leaf=max(10, min(100, len(X) // 100)),
                ccp_alpha=1e-6,
                random_state=self.random_state,
            )
            tree.fit(effects[[column]], rate, sample_weight=exposure)
            self.effect_calibrators_[column] = tree
            calibrated[column] = tree.predict(effects[[column]])

        self.interaction_pairs_ = self._select_interactions(X, rate, exposure)
        for left, right in self.interaction_pairs_:
            name = f"{left}__x__{right}"
            calibrated[name] = calibrated[left] * calibrated[right]

        self.final_feature_names_ = tuple(calibrated.columns)
        self.final_ = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", PoissonRegressor(alpha=self.final_alpha, max_iter=1_000)),
            ]
        )
        self.final_.fit(calibrated, rate, model__sample_weight=exposure)
        return self

    def predict_components(self, X: pd.DataFrame, exposure: np.ndarray) -> dict[str, np.ndarray]:
        self._check_fitted()
        exposure = np.clip(np.asarray(exposure, dtype=float), 1e-8, None)
        baseline = self.baseline_.predict(X) * exposure
        gbm = self.gbm_.predict(X) * self.balance_factor_ * exposure
        effects = self._calibrated_effect_frame(X)
        final = self.final_.predict(effects) * exposure
        if self.max_prediction is not None:
            baseline = np.minimum(baseline, self.max_prediction)
            gbm = np.minimum(gbm, self.max_prediction)
            final = np.minimum(final, self.max_prediction)
        return {"gbm": gbm, "linear_glm": baseline, "boosted_glm": final}

    def predict(self, X: pd.DataFrame, exposure: np.ndarray) -> np.ndarray:
        return self.predict_components(X, exposure)["boosted_glm"]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> BoostedGLM:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return model

    def _linear_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(
            [
                ("category", OneHotEncoder(handle_unknown="ignore"), list(self.categorical_)),
                ("numeric", StandardScaler(), list(self.numeric_)),
            ]
        )

    def _tree_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(
            [
                (
                    "category",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    list(self.categorical_),
                ),
                ("numeric", "passthrough", list(self.numeric_)),
            ],
            verbose_feature_names_out=False,
        )

    def _grid(self, series: pd.Series, categorical: bool) -> np.ndarray:
        if categorical:
            values = series.astype("string").fillna("<missing>").unique()
            return np.asarray(sorted(values.tolist()), dtype=object)
        numeric = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        unique = np.unique(numeric)
        if len(unique) <= self.max_effect_grid:
            return unique
        return np.unique(np.quantile(numeric, np.linspace(0.02, 0.98, self.max_effect_grid)))

    def _extract_effect_curves(self, background: pd.DataFrame) -> dict[str, EffectCurve]:
        curves: dict[str, EffectCurve] = {}
        for feature in self.features_:
            categorical = feature in self.categorical_
            grid = self._grid(background[feature], categorical)
            values = []
            for value in grid:
                changed = background.copy()
                changed[feature] = value
                values.append(float(np.mean(self.gbm_.predict(changed) * self.balance_factor_)))
            curves[feature] = EffectCurve(feature, grid, np.asarray(values), categorical)
        return curves

    def _effect_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)
        fallback = float(np.mean([curve.values.mean() for curve in self.effect_curves_.values()]))
        for feature, curve in self.effect_curves_.items():
            if curve.categorical:
                mapping = dict(zip(curve.grid.astype(str), curve.values, strict=True))
                result[feature] = X[feature].astype("string").map(mapping).fillna(fallback).astype(float)
            else:
                values = pd.to_numeric(X[feature], errors="coerce").fillna(float(np.median(curve.grid)))
                result[feature] = np.interp(values, curve.grid.astype(float), curve.values)
        return result

    def _calibrated_effect_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = self._effect_frame(X)
        calibrated = pd.DataFrame(index=X.index)
        for column, tree in self.effect_calibrators_.items():
            calibrated[column] = tree.predict(raw[[column]])
        for left, right in self.interaction_pairs_:
            calibrated[f"{left}__x__{right}"] = calibrated[left] * calibrated[right]
        return calibrated.loc[:, self.final_feature_names_]

    def _select_interactions(self, X: pd.DataFrame, rate: np.ndarray, exposure: np.ndarray) -> tuple[tuple[str, str], ...]:
        if self.max_interactions <= 0 or len(self.features_) < 2:
            return ()
        sample_size = min(len(X), self.effect_sample)
        sample = X.sample(sample_size, random_state=self.random_state)
        positions = X.index.get_indexer(sample.index)
        importance = permutation_importance(
            self.gbm_,
            sample,
            rate[positions],
            n_repeats=2,
            random_state=self.random_state,
            sample_weight=exposure[positions],
            scoring="neg_mean_poisson_deviance",
        ).importances_mean
        ranked = [self.features_[i] for i in np.argsort(importance)[::-1][:4]]
        return tuple(combinations(ranked, 2))[: self.max_interactions]

    def _check_fitted(self) -> None:
        if not hasattr(self, "final_"):
            raise RuntimeError("Fit the model before predicting")
