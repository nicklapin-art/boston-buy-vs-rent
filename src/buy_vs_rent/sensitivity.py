"""One-way sensitivity analysis with common random numbers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from .config import SimulationConfig
from .simulation import SimulationResult, run_simulation


@dataclass(frozen=True)
class SensitivityParameter:
    label: str
    object_name: str
    attribute_name: str
    low: float
    high: float


DEFAULT_PARAMETERS = (
    SensitivityParameter("Mortgage rate", "mortgage", "initial_rate", 0.055, 0.0775),
    SensitivityParameter("Stock return", "market", "stock_return", 0.050, 0.090),
    SensitivityParameter("Home appreciation", "market", "home_appreciation", 0.020, 0.050),
    SensitivityParameter("Rent growth", "market", "rent_growth", 0.020, 0.045),
    SensitivityParameter("Maintenance", "housing", "maintenance_rate", 0.005, 0.015),
    SensitivityParameter("Property tax", "housing", "property_tax_rate", 0.009, 0.014),
    SensitivityParameter("Sale costs", "housing", "sale_cost_rate", 0.040, 0.070),
)


def run_sensitivity(
    config: SimulationConfig,
    *,
    horizon: int = 10,
    runs: int = 20_000,
    parameters: tuple[SensitivityParameter, ...] = DEFAULT_PARAMETERS,
    base_result: SimulationResult | None = None,
) -> pd.DataFrame:
    """Vary each assumption low/high while holding all other inputs constant."""
    if horizon not in config.horizons:
        raise ValueError("sensitivity horizon must be one of the configured horizons")
    sensitivity_runs = min(runs, config.runs)
    base_config = deepcopy(config)
    base_config.runs = sensitivity_runs
    if base_result is None or base_result.config.runs != sensitivity_runs:
        base_result = run_simulation(base_config)
    base_row = base_result.summary.set_index("horizon_years").loc[horizon]

    rows: list[dict[str, float | str]] = []
    for parameter in parameters:
        output: dict[str, float] = {}
        for case, value in (("low", parameter.low), ("high", parameter.high)):
            varied = deepcopy(base_config)
            setattr(getattr(varied, parameter.object_name), parameter.attribute_name, value)
            varied.validate()
            row = run_simulation(varied).summary.set_index("horizon_years").loc[horizon]
            output[f"{case}_median_difference"] = float(row["median_net_worth_difference"])
            output[f"{case}_buy_probability"] = float(row["buy_win_probability"])
        rows.append(
            {
                "parameter": parameter.label,
                "low_value": parameter.low,
                "high_value": parameter.high,
                "base_median_difference": float(base_row["median_net_worth_difference"]),
                "base_buy_probability": float(base_row["buy_win_probability"]),
                **output,
            }
        )
    frame = pd.DataFrame(rows)
    frame["median_swing"] = (
        frame[["low_median_difference", "high_median_difference"]].max(axis=1)
        - frame[["low_median_difference", "high_median_difference"]].min(axis=1)
    )
    return frame.sort_values("median_swing", ascending=False, ignore_index=True)

