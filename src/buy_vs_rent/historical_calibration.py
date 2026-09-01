"""Historical calibration for long-run parameter uncertainty.

The annual path simulator handles future market randomness. This module adds a
separate layer for uncertainty about the long-run assumptions themselves. It
uses moving-block bootstrap estimates so stock, Boston home, and Boston rent
assumptions move together in combinations observed in history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .historical import HistoricalDataset, load_historical_dataset
from .uncertainty import (
    ParameterRange,
    ParameterUncertaintyResult,
    default_parameter_ranges,
    run_parameter_uncertainty,
    sample_parameter_sets,
)


RETURN_COLUMNS = {
    "stock_return": "stock_return",
    "home_appreciation": "home_return",
    "rent_growth": "rent_return",
}
HISTORICAL_LABELS = {
    "stock_return": "Expected stock return",
    "home_appreciation": "Expected home appreciation",
    "rent_growth": "Expected rent growth",
    "property_tax_rate": "Property-tax rate",
}
JUDGMENT_KEYS = ("maintenance_rate", "insurance_rate", "sale_cost_rate")


@dataclass
class HistoricalParameterSample:
    """Joint parameter draws plus transparent provenance and diagnostics."""

    draws: pd.DataFrame
    ranges: list[ParameterRange]
    metadata: dict[str, object]


def _moving_block_indices(
    observations: int,
    sample_years: int,
    block_years: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample contiguous blocks without joining the end of history to its start."""

    if observations < 2:
        raise ValueError("Historical calibration needs at least two observations")
    width = min(block_years, observations, sample_years)
    block_count = int(np.ceil(sample_years / width))
    starts = rng.integers(0, observations - width + 1, size=block_count)
    indices = np.concatenate([np.arange(start, start + width) for start in starts])
    return indices[:sample_years]


def _geometric_mean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    if np.any(values <= -1.0):
        raise ValueError("Return history contains a value at or below -100%")
    return np.expm1(np.mean(np.log1p(values), axis=axis))


def _historical_panels(dataset: HistoricalDataset) -> tuple[pd.DataFrame, pd.Series]:
    returns = dataset.annual[
        ["stock_return", "home_return", "rent_return"]
    ].dropna()
    taxes = dataset.annual["property_tax_rate"].dropna()
    if len(returns) < 15 or len(taxes) < 10:
        raise ValueError("Insufficient overlapping history for calibration")
    return returns, taxes


def _range_from_draws(
    key: str,
    label: str,
    mode: float,
    values: pd.Series,
    group: str,
) -> ParameterRange:
    low = min(float(values.quantile(0.10)), mode)
    high = max(float(values.quantile(0.90)), mode)
    if high - low < 1e-8:
        low, high = mode - 1e-6, mode + 1e-6
    result = ParameterRange(key, label, low, mode, high, group)
    result.validate()
    return result


def _walk_forward_diagnostics(
    returns: pd.DataFrame,
    *,
    seed: int,
    block_years: int,
    bootstrap_paths: int = 300,
) -> list[dict[str, float | int | str]]:
    """Check 80% return bands using only observations known at each start year."""

    rng = np.random.default_rng(seed)
    first_year = int(returns.index.min())
    last_year = int(returns.index.max())
    rows: list[dict[str, float | int | str]] = []
    for horizon in (5, 10, 20):
        forecasts: dict[str, list[tuple[bool, float]]] = {
            key: [] for key in RETURN_COLUMNS
        }
        for start_year in range(first_year + 15, last_year - horizon + 2):
            training = returns.loc[returns.index < start_year]
            actual = returns.loc[start_year : start_year + horizon - 1]
            if len(training) < 15 or len(actual) != horizon:
                continue
            paths = np.empty((bootstrap_paths, len(RETURN_COLUMNS)), dtype=float)
            history = training.to_numpy(dtype=float)
            for path in range(bootstrap_paths):
                indices = _moving_block_indices(
                    len(training), horizon, block_years, rng
                )
                paths[path] = _geometric_mean(history[indices], axis=0)
            actual_cagr = _geometric_mean(actual.to_numpy(dtype=float), axis=0)
            for column_index, key in enumerate(RETURN_COLUMNS):
                low, median, high = np.quantile(
                    paths[:, column_index], [0.10, 0.50, 0.90]
                )
                forecasts[key].append(
                    (bool(low <= actual_cagr[column_index] <= high), float(median - actual_cagr[column_index]))
                )
        for key, checks in forecasts.items():
            if not checks:
                continue
            rows.append(
                {
                    "horizon_years": horizon,
                    "parameter": key,
                    "label": HISTORICAL_LABELS[key],
                    "cohorts": len(checks),
                    "forecast_80_interval_coverage": float(np.mean([item[0] for item in checks])),
                    "median_forecast_error": float(np.median([item[1] for item in checks])),
                }
            )
    return rows


def sample_historically_calibrated_parameters(
    config: SimulationConfig,
    *,
    count: int,
    block_years: int = 5,
    data_dir: str | None = None,
) -> HistoricalParameterSample:
    """Estimate joint uncertainty widths from history, centered on user inputs.

    Bootstrap deviations are centered on the scenario's long-run assumptions.
    This lets the user's forward-looking view remain the baseline while history
    determines the spread, skew, and dependence of plausible errors around it.
    Parameters without consistent historical series retain labeled judgment
    distributions.
    """

    if count < 8:
        raise ValueError("parameter_sets must be at least 8")
    if block_years < 1:
        raise ValueError("block_years must be positive")
    dataset = load_historical_dataset(data_dir)
    returns, taxes = _historical_panels(dataset)
    rng = np.random.default_rng(config.seed + 170_001)
    return_values = returns.to_numpy(dtype=float)
    return_center = _geometric_mean(return_values, axis=0)
    tax_values = taxes.to_numpy(dtype=float)
    tax_center = float(tax_values.mean())

    draws = pd.DataFrame({"parameter_set": np.arange(1, count + 1, dtype=int)})
    user_centers = np.array(
        [
            config.market.stock_return,
            config.market.home_appreciation,
            config.market.rent_growth,
        ],
        dtype=float,
    )
    calibrated_returns = np.empty((count, 3), dtype=float)
    calibrated_taxes = np.empty(count, dtype=float)
    for index in range(count):
        return_indices = _moving_block_indices(
            len(returns), len(returns), block_years, rng
        )
        boot_center = _geometric_mean(return_values[return_indices], axis=0)
        calibrated_returns[index] = user_centers + (boot_center - return_center)
        tax_indices = _moving_block_indices(
            len(taxes), len(taxes), block_years, rng
        )
        calibrated_taxes[index] = (
            config.housing.property_tax_rate
            + float(tax_values[tax_indices].mean())
            - tax_center
        )

    calibrated_returns[:, 0] = np.clip(calibrated_returns[:, 0], -0.10, 0.25)
    calibrated_returns[:, 1:] = np.clip(calibrated_returns[:, 1:], -0.10, 0.20)
    for column_index, key in enumerate(RETURN_COLUMNS):
        draws[key] = calibrated_returns[:, column_index]
    draws["property_tax_rate"] = np.clip(calibrated_taxes, 0.0, 0.05)

    judgment_ranges = [
        item for item in default_parameter_ranges(config) if item.key in JUDGMENT_KEYS
    ]
    judgment_draws = sample_parameter_sets(
        judgment_ranges, count, config.seed + 270_001
    )
    for key in JUDGMENT_KEYS:
        draws[key] = judgment_draws[key]

    ordered_keys = [item.key for item in default_parameter_ranges(config)]
    draws = draws[["parameter_set", *ordered_keys]]
    ranges: list[ParameterRange] = []
    modes = {
        "stock_return": config.market.stock_return,
        "home_appreciation": config.market.home_appreciation,
        "rent_growth": config.market.rent_growth,
        "property_tax_rate": config.housing.property_tax_rate,
        "maintenance_rate": config.housing.maintenance_rate,
        "insurance_rate": config.housing.insurance_rate,
        "sale_cost_rate": config.housing.sale_cost_rate,
    }
    labels = {item.key: item.label for item in default_parameter_ranges(config)}
    for key in ordered_keys:
        ranges.append(
            _range_from_draws(
                key,
                labels[key],
                modes[key],
                draws[key],
                "historical" if key in HISTORICAL_LABELS else "judgment",
            )
        )

    historical_keys = list(RETURN_COLUMNS) + ["property_tax_rate"]
    correlations = draws[historical_keys].corr()
    correlation_rows: list[dict[str, float | str]] = []
    for first_index, first in enumerate(historical_keys):
        for second in historical_keys[first_index + 1 :]:
            correlation_rows.append(
                {
                    "first": first,
                    "second": second,
                    "correlation": float(correlations.loc[first, second]),
                }
            )

    metadata: dict[str, object] = {
        "method_label": "Historically calibrated",
        "return_data_start_year": int(returns.index.min()),
        "tax_data_start_year": int(taxes.index.min()),
        "tax_data_end_year": int(taxes.index.max()),
        "data_end_year": min(int(returns.index.max()), int(taxes.index.max())),
        "return_observations": len(returns),
        "tax_observations": len(taxes),
        "block_years": min(block_years, len(returns)),
        "historical_parameters": list(HISTORICAL_LABELS),
        "judgment_parameters": list(JUDGMENT_KEYS),
        "initial_mortgage_rate_fixed": True,
        "draw_correlations": correlation_rows,
        "walk_forward": _walk_forward_diagnostics(
            returns,
            seed=config.seed + 370_001,
            block_years=block_years,
        ),
    }
    return HistoricalParameterSample(draws=draws, ranges=ranges, metadata=metadata)


def run_historically_calibrated_uncertainty(
    config: SimulationConfig | None = None,
    *,
    parameter_sets: int = 64,
    runs_per_set: int = 5_000,
    block_years: int = 5,
    data_dir: str | None = None,
) -> ParameterUncertaintyResult:
    """Run nested uncertainty using historically calibrated joint parameter draws."""

    scenario = config or SimulationConfig()
    sample = sample_historically_calibrated_parameters(
        scenario,
        count=parameter_sets,
        block_years=block_years,
        data_dir=data_dir,
    )
    return run_parameter_uncertainty(
        scenario,
        ranges=sample.ranges,
        parameter_draws=sample.draws,
        parameter_sets=parameter_sets,
        runs_per_set=runs_per_set,
        method="historical",
        metadata=sample.metadata,
    )
