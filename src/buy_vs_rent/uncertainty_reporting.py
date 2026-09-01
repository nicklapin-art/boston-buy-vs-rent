"""Saved tables, charts, and report for parameter-uncertainty analysis."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "boston-buy-vs-rent-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .uncertainty import ParameterUncertaintyResult


def save_parameter_uncertainty(
    result: ParameterUncertaintyResult,
    output_dir: str | Path,
) -> Path:
    """Write all reproducible robustness-analysis artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(destination / "robustness_summary.csv", index=False)
    result.outcomes.to_csv(destination / "parameter_set_outcomes.csv", index=False)
    result.influence.to_csv(destination / "parameter_influence.csv", index=False)
    result.config.to_json(destination / "config_used.json")
    ranges = [
        {
            "key": item.key,
            "label": item.label,
            "low": item.low,
            "mode": item.mode,
            "high": item.high,
            "group": item.group,
        }
        for item in result.ranges
    ]
    (destination / "parameter_ranges.json").write_text(
        json.dumps(ranges, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "calibration_metadata.json").write_text(
        json.dumps({"method": result.method, **result.metadata}, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_probability_ranges(result, destination / "robustness_probability_ranges.png")
    _plot_influence(result, destination / "parameter_influence.png")
    report_path = destination / "ROBUSTNESS_REPORT.md"
    report_path.write_text(_report_markdown(result), encoding="utf-8")
    return report_path


def _plot_probability_ranges(result: ParameterUncertaintyResult, path: Path) -> None:
    summary = result.summary
    y = np.arange(len(summary))
    median = summary["median_buy_win_probability"].to_numpy() * 100
    low = summary["p10_buy_win_probability"].to_numpy() * 100
    high = summary["p90_buy_win_probability"].to_numpy() * 100
    integrated = summary["integrated_buy_win_probability"].to_numpy() * 100
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.hlines(y, low, high, color="#1f5265", linewidth=7, alpha=0.34)
    axis.scatter(median, y, color="#1f5265", s=65, label="Median assumption set")
    axis.scatter(integrated, y, color="#cc7a29", marker="D", s=46, label="Integrated probability")
    axis.axvline(50, color="#444444", linewidth=1, linestyle="--")
    axis.set_yticks(y, [f"{int(value)} years" for value in summary["horizon_years"]])
    axis.set_xlim(0, 100)
    axis.set_xlabel("Probability buying wins (%)")
    axis.set_title("Buying probability across plausible assumption sets")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_influence(result: ParameterUncertaintyResult, path: Path) -> None:
    final_horizon = int(result.influence["horizon_years"].max())
    frame = result.influence[
        result.influence["horizon_years"] == final_horizon
    ].sort_values("partial_rank_correlation")
    colors = np.where(frame["partial_rank_correlation"] >= 0, "#287b61", "#cc7a29")
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.barh(frame["label"], frame["partial_rank_correlation"], color=colors)
    axis.axvline(0, color="#444444", linewidth=1)
    axis.set_xlim(-1, 1)
    axis.set_xlabel("Partial rank correlation with buying win probability")
    axis.set_title(f"Which uncertain assumptions matter at {final_horizon} years")
    axis.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report_markdown(result: ParameterUncertaintyResult) -> str:
    if result.method == "historical":
        method_description = (
            f"Uncertainty in stocks, Boston homes, Boston rents, and property taxes is "
            f"calibrated from historical blocks through {result.metadata['data_end_year']} and "
            "centered on the configured scenario. Maintenance, insurance, and selling costs "
            "retain judgment bands."
        )
    else:
        method_description = (
            "Parameter ranges are triangular judgment bands, with the configured scenario "
            "as the most likely value."
        )
    lines = [
        "# Assumption Robustness Report",
        "",
        (
            f"This analysis uses {result.parameter_sets:,} stratified parameter sets and "
            f"{result.runs_per_set:,} economic paths per set. {method_description}"
        ),
        "",
        "## Results",
        "",
        "| Horizon | Integrated buy probability | 10th–90th assumption range | Robust buy share | Median of median advantage |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result.summary.itertuples(index=False):
        lines.append(
            f"| {row.horizon_years} years | {row.integrated_buy_win_probability:.1%} | "
            f"{row.p10_buy_win_probability:.1%}–{row.p90_buy_win_probability:.1%} | "
            f"{row.robust_buy_share:.1%} | ${row.median_of_median_net_worth_difference:,.0f} |"
        )
    final_horizon = int(result.influence["horizon_years"].max())
    important = result.influence[
        result.influence["horizon_years"] == final_horizon
    ].head(5)
    lines.extend(
        [
            "",
            f"## Main drivers at {final_horizon} years",
            "",
            "Positive values favor buying as the assumption rises; negative values favor renting.",
            "",
            "| Assumption | Partial rank correlation |",
            "|---|---:|",
        ]
    )
    for row in important.itertuples(index=False):
        lines.append(f"| {row.label} | {row.partial_rank_correlation:+.2f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The integrated probability averages over the assumed triangular parameter distributions. "
            "The robust buy share is the fraction of parameter sets in which buying wins more than half "
            "of economic paths. Neither is a guarantee. Historical indexes do not represent a specific "
            "property, and the remaining judgment bands should be replaced with property-specific "
            "evidence when available.",
            "",
        ]
    )
    return "\n".join(lines)
