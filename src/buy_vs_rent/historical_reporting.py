"""Reports and charts for historical validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "buy_vs_rent_matplotlib")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .historical import HistoricalBacktestResult  # noqa: E402


def save_cohort_chart(result: HistoricalBacktestResult, path: str | Path) -> None:
    horizons = sorted(result.cohorts["horizon_years"].unique())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(11, 3.4 * len(horizons)), sharex=False)
    if len(horizons) == 1:
        axes = [axes]
    for axis, horizon in zip(axes, horizons):
        group = result.cohorts[result.cohorts["horizon_years"] == horizon].sort_values("start_year")
        years = group["start_year"].to_numpy()
        p05 = group["forecast_p05_pct"].to_numpy() * 100.0
        p95 = group["forecast_p95_pct"].to_numpy() * 100.0
        median = group["forecast_median_pct"].to_numpy() * 100.0
        realized = group["realized_difference_pct"].to_numpy() * 100.0
        axis.fill_between(years, p05, p95, color="#8FB7C5", alpha=0.35, label="Forecast 5th–95th")
        axis.plot(years, median, color="#245A70", linewidth=1.8, label="Forecast median")
        colors = np.where(realized >= 0.0, "#2B7A5F", "#C7792B")
        axis.scatter(years, realized, c=colors, s=28, zorder=3, label="Realized")
        axis.axhline(0.0, color="#333333", linewidth=1.0)
        axis.set_title(f"{horizon}-year cohorts")
        axis.set_ylabel("Buyer advantage\n(% of starting price)")
        axis.grid(alpha=0.18)
        axis.legend(loc="best", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Purchase start year")
    fig.suptitle("Historical Boston cohorts: forecast versus realized outcome", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_calibration_chart(result: HistoricalBacktestResult, path: str | Path) -> None:
    frame = result.calibration.sort_values("horizon_years")
    labels = [f"{year} yr" for year in frame["horizon_years"]]
    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(
        x - width / 2,
        frame["actual_buy_win_rate"] * 100,
        width,
        label="Actual",
        color="#2B7A5F",
    )
    axes[0].bar(
        x + width / 2,
        frame["mean_forecast_buy_probability"] * 100,
        width,
        label="Forecast",
        color="#245A70",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Buy-win rate / probability")
    axes[0].set_title("Probability calibration")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.18)

    coverage = frame["forecast_90_interval_coverage"] * 100
    axes[1].bar(x, coverage, width=0.55, color="#8FB7C5")
    axes[1].axhline(90, color="#A33A34", linestyle="--", label="90% target")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Realized outcomes inside interval")
    axes[1].set_title("Forecast interval coverage")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _percent(value: float) -> str:
    return f"{value:.1%}"


def validation_report_markdown(result: HistoricalBacktestResult) -> str:
    rows = []
    for _, row in result.calibration.iterrows():
        rows.append(
            "| {horizon} | {cohorts} | {actual} | {forecast} | {coverage} | {accuracy} | {brier:.3f} |".format(
                horizon=int(row["horizon_years"]),
                cohorts=int(row["cohorts"]),
                actual=_percent(row["actual_buy_win_rate"]),
                forecast=_percent(row["mean_forecast_buy_probability"]),
                coverage=_percent(row["forecast_90_interval_coverage"]),
                accuracy=_percent(row["classification_accuracy"]),
                brier=row["brier_score"],
            )
        )
    moments = []
    for _, row in result.moments.iterrows():
        moments.append(
            f"| {row['series']} | {_percent(row['observed_mean'])} | {_percent(row['model_mean'])} | "
            f"{_percent(row['observed_volatility'])} | {_percent(row['model_volatility'])} |"
        )
    return f"""# Historical validation report

## Scope

- Boston cohorts beginning in {result.data_start_year}–{result.data_end_year}, depending on horizon.
- {result.forecast_runs:,} Monte Carlo paths per cohort.
- FHFA Boston home prices, BLS Boston rents, Freddie Mac mortgage rates, Yale/Shiller stock returns, and City of Boston residential tax rates.
- Each historical outcome uses the same mortgage, refinancing, cash-flow matching, maintenance, insurance, and transaction-cost accounting as the forward simulator.

## Calibration results

| Horizon | Cohorts | Actual buy-win rate | Mean forecast probability | 90% interval coverage | Classification accuracy | Brier score |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

A well-calibrated 90% interval should contain about 90% of realized outcomes. A lower coverage rate means the model is too confident. Brier scores range from 0 (perfect probability forecasts) to 1 (worst); 0.25 is the score from repeatedly forecasting 50% for balanced binary outcomes.

The model means below are unconditional estimates implied by its normal/recession/crash transition system, not merely the normal-state inputs.

## Assumptions versus observed annual history

| Series | Observed mean | Model mean | Observed volatility | Model volatility |
|---|---:|---:|---:|---:|
{chr(10).join(moments)}

## Interpretation limits

- Cohorts overlap, so they are not statistically independent.
- Index results describe the Boston metropolitan market, not the exact property.
- CPI rent includes continuing leases and can lag asking rents.
- PMMS is a national conforming-loan average, not a historical jumbo quote.
- Historical local insurance and maintenance series are unavailable; configured percentages are applied throughout.
- Tax benefits, capital-gains taxes, residential exemptions, and lifestyle value remain excluded.
- This is retrospective calibration evidence, not proof that future outcomes will follow the same distribution.
- The forecast uses today's fixed model assumptions and revised historical indexes; it is not a vintage forecast that could actually have been produced at each start date.
"""


def save_historical_outputs(result: HistoricalBacktestResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.cohorts.to_csv(output / "historical_cohorts.csv", index=False)
    result.calibration.to_csv(output / "calibration_summary.csv", index=False)
    result.moments.to_csv(output / "historical_moments.csv", index=False)
    result.correlations.to_csv(output / "historical_correlations.csv", index=False)
    result.config.to_json(output / "config_used.json")
    (output / "VALIDATION_REPORT.md").write_text(
        validation_report_markdown(result), encoding="utf-8"
    )
    save_cohort_chart(result, output / "cohort_forecast_vs_actual.png")
    save_calibration_chart(result, output / "calibration_summary.png")
