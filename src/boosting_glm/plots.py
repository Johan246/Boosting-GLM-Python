"""Portable matplotlib equivalents of the original ggplot reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_prediction_plot(observed: np.ndarray, predictions: dict[str, np.ndarray], path: str | Path) -> None:
    frame = pd.DataFrame({"observed": observed, **predictions})
    grouped = frame.assign(bin=pd.qcut(frame["gbm"], 20, duplicates="drop")).groupby(
        "bin", observed=False
    ).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for column in predictions:
        ax.plot(grouped.index.astype(str), grouped[column], marker="o", label=column)
    ax.plot(grouped.index.astype(str), grouped["observed"], color="black", linewidth=2, label="observed")
    ax.set_ylabel("Mean claim count")
    ax.set_xlabel("GBM risk ventile")
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_cumulative_claims_plot(observed: np.ndarray, predictions: dict[str, np.ndarray], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, prediction in predictions.items():
        order = np.argsort(prediction)
        cumulative = np.cumsum(np.asarray(observed)[order])
        total = cumulative[-1] if cumulative[-1] else 1
        ax.plot(np.linspace(0, 1, len(order)), cumulative / total, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set(xlabel="Portfolio fraction (low to high predicted risk)", ylabel="Cumulative claims fraction")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_effect_plots(model, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for feature, curve in model.effect_curves_.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        positions = np.arange(len(curve.grid)) if curve.categorical else curve.grid.astype(float)
        ax.plot(positions, curve.values, marker="o")
        if curve.categorical:
            ax.set_xticks(positions, curve.grid.astype(str), rotation=45, ha="right")
        ax.set(xlabel=feature, ylabel="Mean GBM predicted frequency", title=f"Marginal effect: {feature}")
        fig.tight_layout()
        fig.savefig(output / f"effect_{feature}.png", dpi=180)
        plt.close(fig)

