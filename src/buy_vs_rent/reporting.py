"""Console and chart reporting."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Some managed environments expose a read-only home directory.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "buy_vs_rent_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .simulation import SimulationResult


def format_summary(summary: pd.DataFrame) -> str:
    """Render the key outputs as an easy-to-scan text table."""
    display = summary[
        [
            "horizon_years",
            "buy_win_probability",
            "median_net_worth_difference",
            "p05_net_worth_difference",
            "p95_net_worth_difference",
        ]
    ].copy()
    display.columns = ["Years", "P(buy wins)", "Median difference", "5th percentile", "95th percentile"]
    display["P(buy wins)"] = display["P(buy wins)"].map(lambda value: f"{value:.1%}")
    for column in ("Median difference", "5th percentile", "95th percentile"):
        display[column] = display[column].map(lambda value: f"${value:,.0f}")
    return display.to_string(index=False)


def save_distribution_chart(result: SimulationResult, path: str | Path) -> None:
    horizons = sorted(result.net_worth_differences)
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 4.2), squeeze=False)
    for axis, horizon in zip(axes[0], horizons):
        values = result.net_worth_differences[horizon] / 1_000.0
        lo, hi = np.quantile(values, [0.005, 0.995])
        visible = values[(values >= lo) & (values <= hi)]
        axis.hist(visible, bins=70, color="#3274A1", alpha=0.85)
        axis.axvline(0.0, color="#B23A48", linewidth=1.5)
        axis.axvline(np.median(values), color="#1B4332", linestyle="--", linewidth=1.5)
        axis.set_title(f"Year {horizon}")
        axis.set_xlabel("Buyer advantage ($000s)")
        axis.set_ylabel("Paths")
        axis.grid(alpha=0.18)
    fig.suptitle("Buy minus rent net-worth distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_sensitivity_chart(frame: pd.DataFrame, path: str | Path, *, horizon: int) -> None:
    plot = frame.sort_values("median_swing", ascending=True).copy()
    y = np.arange(len(plot))
    low = plot[["low_median_difference", "high_median_difference"]].min(axis=1) / 1_000.0
    high = plot[["low_median_difference", "high_median_difference"]].max(axis=1) / 1_000.0
    base = plot["base_median_difference"].iloc[0] / 1_000.0
    fig, axis = plt.subplots(figsize=(9, 5.2))
    axis.barh(y, high - low, left=low, color="#4C956C", alpha=0.85)
    axis.axvline(base, color="#1D3557", linestyle="--", label="Baseline")
    axis.axvline(0.0, color="#B23A48", linewidth=1.2)
    axis.set_yticks(y, plot["parameter"])
    axis.set_xlabel("Median buyer advantage ($000s)")
    axis.set_title(f"One-way sensitivity at year {horizon}")
    axis.grid(axis="x", alpha=0.18)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    result: SimulationResult,
    output_dir: str | Path,
    sensitivity: pd.DataFrame | None = None,
    *,
    sensitivity_horizon: int = 10,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output / "summary.csv", index=False)
    result.diagnostics.to_csv(output / "annual_diagnostics.csv", index=False)
    result.config.to_json(output / "config_used.json")
    save_distribution_chart(result, output / "net_worth_distributions.png")
    if sensitivity is not None:
        sensitivity.to_csv(output / "sensitivity.csv", index=False)
        save_sensitivity_chart(
            sensitivity,
            output / "sensitivity_tornado.png",
            horizon=sensitivity_horizon,
        )
