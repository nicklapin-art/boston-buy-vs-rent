"""Counterfactual and break-even analysis for a DIY sweat-equity project."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .simulation import run_simulation


@dataclass
class SweatEquityAnalysis:
    """Configured-project comparison and deterministic value-uplift curve."""

    config: SimulationConfig
    horizon: int
    runs: int
    summary: pd.DataFrame
    curve: pd.DataFrame
    financial_break_even_value: float | None
    economic_break_even_value: float | None
    tested_max_value: float


def _break_even(frame: pd.DataFrame, column: str) -> float | None:
    """Linearly interpolate the first tested uplift reaching a 50% buy probability."""

    ordered = frame.sort_values("value_added").reset_index(drop=True)
    probabilities = ordered[column].to_numpy(dtype=float)
    values = ordered["value_added"].to_numpy(dtype=float)
    crossing = np.flatnonzero(probabilities >= 0.5)
    if not len(crossing):
        return None
    index = int(crossing[0])
    if index == 0:
        return float(values[0])
    lower_probability, upper_probability = probabilities[index - 1 : index + 1]
    lower_value, upper_value = values[index - 1 : index + 1]
    if upper_probability <= lower_probability:
        return float(upper_value)
    weight = (0.5 - lower_probability) / (upper_probability - lower_probability)
    return float(lower_value + weight * (upper_value - lower_value))


def _analysis_config(config: SimulationConfig, horizon: int, runs: int) -> SimulationConfig:
    scenario = deepcopy(config)
    scenario.runs = runs
    scenario.years = horizon
    scenario.horizons = [horizon]
    scenario.validate()
    return scenario


def run_sweat_equity_analysis(
    config: SimulationConfig,
    *,
    horizon: int | None = None,
    runs: int = 20_000,
    curve_points: int = 11,
) -> SweatEquityAnalysis:
    """Compare the configured project with no project and solve its uplift threshold."""

    config.validate()
    sweat = config.sweat_equity
    if not sweat.enabled:
        raise ValueError("Enable sweat equity before running its analysis")
    selected_horizon = int(horizon or max(config.horizons))
    if not sweat.completion_year <= selected_horizon <= config.years:
        raise ValueError("analysis horizon must be after project completion and within the simulation")
    if not 1_000 <= runs <= 100_000:
        raise ValueError("sweat-equity analysis runs must be between 1,000 and 100,000")
    if not 5 <= curve_points <= 21:
        raise ValueError("curve_points must be between 5 and 21")

    analysis_runs = min(runs, config.runs)
    configured = _analysis_config(config, selected_horizon, analysis_runs)
    without_project = deepcopy(configured)
    without_project.sweat_equity.enabled = False
    base_result = run_simulation(without_project)
    project_result = run_simulation(configured)
    base_difference = base_result.net_worth_differences[selected_horizon]
    project_difference = project_result.net_worth_differences[selected_horizon]
    time_cost = sweat.labor_hours * sweat.hourly_time_value
    economic_difference = project_difference - time_cost
    incremental = project_difference - base_difference

    summary = pd.DataFrame(
        [
            {
                "horizon_years": selected_horizon,
                "buy_probability_without_project": float(np.mean(base_difference > 0.0)),
                "buy_probability_with_project": float(np.mean(project_difference > 0.0)),
                "economic_buy_probability_with_project": float(
                    np.mean(economic_difference > 0.0)
                ),
                "median_incremental_financial_value": float(np.median(incremental)),
                "median_incremental_economic_value": float(
                    np.median(incremental) - time_cost
                ),
                "probability_project_improves_financial_result": float(
                    np.mean(incremental > 0.0)
                ),
                "probability_project_improves_economic_result": float(
                    np.mean(incremental - time_cost > 0.0)
                ),
                "time_cost": float(time_cost),
                "cash_cost": float(sweat.cash_cost),
                "median_value_added_at_completion": float(
                    np.median(
                        np.random.default_rng(config.seed + 610_001).triangular(
                            sweat.value_added_low,
                            sweat.value_added_expected,
                            sweat.value_added_high,
                            size=analysis_runs,
                        )
                    )
                    if sweat.value_added_low != sweat.value_added_high
                    else sweat.value_added_expected
                ),
            }
        ]
    )

    tested_max = max(
        config.housing.purchase_price * 0.50,
        sweat.value_added_high * 1.25,
        sweat.cash_cost * 2.0,
        100_000.0,
    )
    raw_values = np.linspace(0.0, tested_max, curve_points)
    values = np.unique(np.round(raw_values / 5_000.0) * 5_000.0)
    curve_rows: list[dict[str, float]] = []
    for value in values:
        point = deepcopy(configured)
        point.sweat_equity.value_added_low = float(value)
        point.sweat_equity.value_added_expected = float(value)
        point.sweat_equity.value_added_high = float(value)
        result = run_simulation(point)
        difference = result.net_worth_differences[selected_horizon]
        curve_rows.append(
            {
                "value_added": float(value),
                "financial_buy_probability": float(np.mean(difference > 0.0)),
                "economic_buy_probability": float(np.mean(difference - time_cost > 0.0)),
                "median_financial_difference": float(np.median(difference)),
                "median_economic_difference": float(np.median(difference) - time_cost),
            }
        )
    curve = pd.DataFrame(curve_rows)
    return SweatEquityAnalysis(
        config=deepcopy(config),
        horizon=selected_horizon,
        runs=analysis_runs,
        summary=summary,
        curve=curve,
        financial_break_even_value=_break_even(curve, "financial_buy_probability"),
        economic_break_even_value=_break_even(curve, "economic_buy_probability"),
        tested_max_value=float(values.max()),
    )
