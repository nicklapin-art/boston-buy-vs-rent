"""Counterfactual and required-uplift analysis for a DIY remodeling project."""

from __future__ import annotations

from collections.abc import Callable
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
    financial_required_uplift: float | None
    economic_required_uplift: float | None
    tested_max_value: float


def _analysis_config(config: SimulationConfig, horizon: int, runs: int) -> SimulationConfig:
    scenario = deepcopy(config)
    scenario.runs = runs
    scenario.years = horizon
    scenario.horizons = [horizon]
    scenario.validate()
    return scenario


def _solve_threshold(
    evaluate: Callable[[float], dict[str, float]],
    column: str,
    *,
    initial_upper: float,
    maximum: float,
    tolerance: float = 500.0,
) -> float | None:
    """Bracket and bisect the value needed to reach a 50% buying probability."""

    if evaluate(0.0)[column] >= 0.5:
        return 0.0
    lower, upper = 0.0, initial_upper
    while evaluate(upper)[column] < 0.5 and upper < maximum:
        lower = upper
        upper = min(upper * 2.0, maximum)
    if evaluate(upper)[column] < 0.5:
        return None
    while upper - lower > tolerance:
        midpoint = (lower + upper) / 2.0
        if evaluate(midpoint)[column] >= 0.5:
            upper = midpoint
        else:
            lower = midpoint
    return float(np.ceil(upper / 1_000.0) * 1_000.0)


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
    selected_horizon = int(horizon or max(config.horizons))
    if not sweat.completion_year <= selected_horizon <= config.years:
        raise ValueError("analysis horizon must be after project completion and within the simulation")
    if not 1_000 <= runs <= 100_000:
        raise ValueError("sweat-equity analysis runs must be between 1,000 and 100,000")
    if not 5 <= curve_points <= 21:
        raise ValueError("curve_points must be between 5 and 21")

    analysis_runs = min(runs, config.runs)
    configured = deepcopy(config)
    configured.sweat_equity.enabled = True
    configured = _analysis_config(configured, selected_horizon, analysis_runs)
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

    evaluated: dict[float, dict[str, float]] = {}

    def evaluate(value: float) -> dict[str, float]:
        key = float(value)
        if key in evaluated:
            return evaluated[key]
        point = deepcopy(configured)
        point.sweat_equity.value_added_low = key
        point.sweat_equity.value_added_expected = key
        point.sweat_equity.value_added_high = key
        result = run_simulation(point)
        difference = result.net_worth_differences[selected_horizon]
        evaluated[key] = {
            "value_added": key,
            "financial_buy_probability": float(np.mean(difference > 0.0)),
            "economic_buy_probability": float(np.mean(difference - time_cost > 0.0)),
            "median_financial_difference": float(np.median(difference)),
            "median_economic_difference": float(np.median(difference) - time_cost),
        }
        return evaluated[key]

    initial_upper = max(
        config.housing.purchase_price * 0.10,
        sweat.cash_cost * 2.0,
        100_000.0,
    )
    maximum = max(config.housing.purchase_price * 4.0, initial_upper)
    financial_threshold = _solve_threshold(
        evaluate,
        "financial_buy_probability",
        initial_upper=initial_upper,
        maximum=maximum,
    )
    economic_threshold = _solve_threshold(
        evaluate,
        "economic_buy_probability",
        initial_upper=initial_upper,
        maximum=maximum,
    )
    finite_thresholds = [
        value for value in (financial_threshold, economic_threshold) if value is not None
    ]
    if finite_thresholds:
        curve_max = max(initial_upper, max(finite_thresholds) * 1.35)
    else:
        curve_max = maximum
    curve_max = float(np.ceil(curve_max / 5_000.0) * 5_000.0)
    raw_values = np.linspace(0.0, curve_max, curve_points)
    values = np.unique(np.round(raw_values / 5_000.0) * 5_000.0)
    curve = pd.DataFrame([evaluate(float(value)) for value in values])
    return SweatEquityAnalysis(
        config=deepcopy(config),
        horizon=selected_horizon,
        runs=analysis_runs,
        summary=summary,
        curve=curve,
        financial_required_uplift=financial_threshold,
        economic_required_uplift=economic_threshold,
        tested_max_value=maximum,
    )
