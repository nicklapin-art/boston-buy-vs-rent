"""Second-level uncertainty analysis for assumptions that are not known exactly."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import qmc, rankdata

from .config import SimulationConfig
from .simulation import run_simulation


@dataclass(frozen=True)
class ParameterRange:
    """Bounded triangular distribution with the current assumption as its mode."""

    key: str
    label: str
    low: float
    mode: float
    high: float
    group: str

    def validate(self) -> None:
        if not np.isfinite([self.low, self.mode, self.high]).all():
            raise ValueError(f"{self.key} range must be finite")
        if not self.low <= self.mode <= self.high or self.low == self.high:
            raise ValueError(f"{self.key} must satisfy low <= mode <= high with nonzero width")


@dataclass
class ParameterUncertaintyResult:
    """Parameter draws, set-level outcomes, aggregate robustness, and influence."""

    config: SimulationConfig
    ranges: list[ParameterRange]
    parameter_sets: int
    runs_per_set: int
    summary: pd.DataFrame
    outcomes: pd.DataFrame
    influence: pd.DataFrame


def default_parameter_ranges(config: SimulationConfig) -> list[ParameterRange]:
    """Return transparent, deliberately broad ranges centered on the scenario."""

    h, market = config.housing, config.market

    def centered(
        key: str,
        label: str,
        mode: float,
        half_width: float,
        group: str,
        floor: float,
        ceiling: float,
    ) -> ParameterRange:
        low = max(mode - half_width, floor)
        high = min(mode + half_width, ceiling)
        if mode <= low:
            low = max(floor, mode - max(half_width, 1e-6))
        if mode >= high:
            high = min(ceiling, mode + max(half_width, 1e-6))
        return ParameterRange(key, label, low, mode, high, group)

    ranges = [
        centered(
            "stock_return", "Expected stock return", market.stock_return,
            0.025, "market", -0.10, 0.25,
        ),
        centered(
            "home_appreciation", "Expected home appreciation", market.home_appreciation,
            0.020, "market", -0.10, 0.20,
        ),
        centered(
            "rent_growth", "Expected rent growth", market.rent_growth,
            0.0125, "market", -0.10, 0.20,
        ),
        centered(
            "maintenance_rate", "Annual maintenance", h.maintenance_rate,
            0.0040, "housing", 0.0, 0.05,
        ),
        centered(
            "property_tax_rate", "Property-tax rate", h.property_tax_rate,
            0.0025, "housing", 0.0, 0.05,
        ),
        centered(
            "insurance_rate", "Homeowners insurance", h.insurance_rate,
            0.0015, "housing", 0.0, 0.05,
        ),
        centered(
            "sale_cost_rate", "Selling costs", h.sale_cost_rate,
            0.0150, "housing", 0.0, 0.20,
        ),
    ]
    for item in ranges:
        item.validate()
    return ranges


def _triangular_ppf(uniforms: np.ndarray, item: ParameterRange) -> np.ndarray:
    """Inverse CDF for a triangular distribution, including an off-center mode."""

    low, mode, high = item.low, item.mode, item.high
    mode_probability = (mode - low) / (high - low)
    left = low + np.sqrt(uniforms * (high - low) * (mode - low))
    right = high - np.sqrt((1.0 - uniforms) * (high - low) * (high - mode))
    return np.where(uniforms <= mode_probability, left, right)


def sample_parameter_sets(
    ranges: list[ParameterRange],
    count: int,
    seed: int,
) -> pd.DataFrame:
    """Generate reproducible, stratified draws across all uncertain parameters."""

    if count < 8:
        raise ValueError("parameter_sets must be at least 8")
    if not ranges:
        raise ValueError("at least one parameter range is required")
    for item in ranges:
        item.validate()
    sampler = qmc.LatinHypercube(d=len(ranges), scramble=True, seed=seed)
    uniforms = sampler.random(n=count)
    values = {
        item.key: _triangular_ppf(uniforms[:, index], item)
        for index, item in enumerate(ranges)
    }
    frame = pd.DataFrame(values)
    frame.insert(0, "parameter_set", np.arange(1, count + 1, dtype=int))
    return frame


def _apply_parameter_set(config: SimulationConfig, values: pd.Series) -> None:
    setters = {
        "stock_return": lambda value: setattr(config.market, "stock_return", value),
        "home_appreciation": lambda value: setattr(config.market, "home_appreciation", value),
        "rent_growth": lambda value: setattr(config.market, "rent_growth", value),
        "maintenance_rate": lambda value: setattr(config.housing, "maintenance_rate", value),
        "property_tax_rate": lambda value: setattr(config.housing, "property_tax_rate", value),
        "insurance_rate": lambda value: setattr(config.housing, "insurance_rate", value),
        "sale_cost_rate": lambda value: setattr(config.housing, "sale_cost_rate", value),
    }
    for key in values.index:
        if key == "parameter_set":
            continue
        if key not in setters:
            raise ValueError(f"Unsupported uncertain parameter: {key}")
        setters[key](float(values[key]))


def _partial_rank_correlations(
    frame: pd.DataFrame,
    parameter_keys: list[str],
    outcome_key: str,
) -> dict[str, float]:
    """Control for all other sampled inputs when ranking each parameter's effect."""

    ranked_inputs = np.column_stack(
        [rankdata(frame[key].to_numpy()) for key in parameter_keys]
    )
    ranked_outcome = rankdata(frame[outcome_key].to_numpy())
    coefficients: dict[str, float] = {}
    for index, key in enumerate(parameter_keys):
        controls = np.delete(ranked_inputs, index, axis=1)
        design = np.column_stack([np.ones(len(frame)), controls])
        parameter = ranked_inputs[:, index]
        parameter_residual = parameter - design @ np.linalg.lstsq(
            design, parameter, rcond=None
        )[0]
        outcome_residual = ranked_outcome - design @ np.linalg.lstsq(
            design, ranked_outcome, rcond=None
        )[0]
        denominator = float(
            np.sqrt(np.sum(parameter_residual**2) * np.sum(outcome_residual**2))
        )
        coefficients[key] = (
            float(np.sum(parameter_residual * outcome_residual) / denominator)
            if denominator > 1e-12
            else 0.0
        )
    return coefficients


def run_parameter_uncertainty(
    config: SimulationConfig | None = None,
    *,
    ranges: list[ParameterRange] | None = None,
    parameter_sets: int = 64,
    runs_per_set: int = 5_000,
) -> ParameterUncertaintyResult:
    """Run nested Monte Carlo analysis over plausible long-run assumptions.

    Every outer parameter set uses common random numbers. This makes differences
    between parameter sets reflect the assumptions instead of different random
    economic paths, while each set still contains ``runs_per_set`` paths.
    """

    base = deepcopy(config or SimulationConfig())
    base.validate()
    if not 8 <= parameter_sets <= 512:
        raise ValueError("parameter_sets must be between 8 and 512")
    if not 250 <= runs_per_set <= 100_000:
        raise ValueError("runs_per_set must be between 250 and 100,000")
    if parameter_sets * runs_per_set > 5_000_000:
        raise ValueError("parameter_sets times runs_per_set cannot exceed 5,000,000")

    selected_ranges = list(ranges or default_parameter_ranges(base))
    draws = sample_parameter_sets(selected_ranges, parameter_sets, base.seed + 70_001)
    inner_seed = base.seed + 90_001
    outcome_rows: list[dict[str, float | int]] = []

    for _, parameter_row in draws.iterrows():
        scenario = deepcopy(base)
        _apply_parameter_set(scenario, parameter_row)
        scenario.runs = runs_per_set
        scenario.seed = inner_seed
        scenario.validate()
        simulation = run_simulation(scenario)
        parameter_values = {
            item.key: float(parameter_row[item.key]) for item in selected_ranges
        }
        for result_row in simulation.summary.to_dict(orient="records"):
            outcome_rows.append(
                {
                    "parameter_set": int(parameter_row["parameter_set"]),
                    **parameter_values,
                    **result_row,
                }
            )

    outcomes = pd.DataFrame(outcome_rows)
    summary_rows: list[dict[str, float | int]] = []
    influence_rows: list[dict[str, float | int | str]] = []
    for horizon, group in outcomes.groupby("horizon_years", sort=True):
        probability = group["buy_win_probability"]
        median_difference = group["median_net_worth_difference"]
        summary_rows.append(
            {
                "horizon_years": int(horizon),
                "parameter_sets": len(group),
                "runs_per_set": runs_per_set,
                "integrated_buy_win_probability": float(probability.mean()),
                "p10_buy_win_probability": float(probability.quantile(0.10)),
                "median_buy_win_probability": float(probability.median()),
                "p90_buy_win_probability": float(probability.quantile(0.90)),
                "robust_buy_share": float(np.mean(probability > 0.50)),
                "p10_median_net_worth_difference": float(median_difference.quantile(0.10)),
                "median_of_median_net_worth_difference": float(median_difference.median()),
                "p90_median_net_worth_difference": float(median_difference.quantile(0.90)),
            }
        )
        partial_correlations = _partial_rank_correlations(
            group,
            [item.key for item in selected_ranges],
            "buy_win_probability",
        )
        for item in selected_ranges:
            coefficient = partial_correlations[item.key]
            influence_rows.append(
                {
                    "horizon_years": int(horizon),
                    "parameter": item.key,
                    "label": item.label,
                    "partial_rank_correlation": coefficient,
                    "absolute_correlation": abs(coefficient),
                }
            )

    influence = pd.DataFrame(influence_rows).sort_values(
        ["horizon_years", "absolute_correlation"], ascending=[True, False]
    )
    return ParameterUncertaintyResult(
        config=base,
        ranges=selected_ranges,
        parameter_sets=parameter_sets,
        runs_per_set=runs_per_set,
        summary=pd.DataFrame(summary_rows),
        outcomes=outcomes,
        influence=influence.reset_index(drop=True),
    )
