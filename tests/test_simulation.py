from copy import deepcopy

import numpy as np

from buy_vs_rent.config import RegimeAssumptions, SimulationConfig
from buy_vs_rent.simulation import run_simulation
from buy_vs_rent.sweat_equity import run_sweat_equity_analysis


def small_config() -> SimulationConfig:
    config = SimulationConfig(runs=1_000, years=10, horizons=[5, 10], seed=123)
    return config


def test_reproducible_and_has_requested_summary():
    first = run_simulation(small_config())
    second = run_simulation(small_config())
    assert list(first.summary["horizon_years"]) == [5, 10]
    assert np.array_equal(
        first.net_worth_differences[10], second.net_worth_differences[10]
    )
    required = {
        "buy_win_probability",
        "median_net_worth_difference",
        "p05_net_worth_difference",
        "p95_net_worth_difference",
    }
    assert required.issubset(first.summary.columns)


def test_zero_volatility_produces_identical_paths():
    config = small_config()
    config.market.stock_volatility = 0.0
    config.market.home_volatility = 0.0
    config.market.rent_volatility = 0.0
    config.mortgage.annual_rate_volatility = 0.0
    config.mortgage.refinance_enabled = False
    config.regimes = RegimeAssumptions(
        names=["normal"],
        transition=[[1.0]],
        stock_return_shifts=[0.0],
        home_return_shifts=[0.0],
        rent_growth_shifts=[0.0],
        mortgage_rate_shifts=[0.0],
        volatility_multipliers=[1.0],
    )
    result = run_simulation(config)
    assert np.ptp(result.net_worth_differences[10]) == 0.0


def test_refinance_triggers_when_rate_drop_clears_threshold():
    config = small_config()
    config.mortgage.long_run_rate = 0.04
    config.mortgage.rate_mean_reversion = 1.0
    config.mortgage.annual_rate_volatility = 0.0
    config.mortgage.refinance_threshold = 0.005
    config.regimes.mortgage_rate_shifts = [0.0, 0.0, 0.0]
    result = run_simulation(config)
    assert result.diagnostics.loc[0, "refinance_path_share"] == 1.0


def test_input_config_is_not_mutated():
    config = small_config()
    original = deepcopy(config)
    run_simulation(config)
    assert config == original


def deterministic_sweat_config() -> SimulationConfig:
    config = SimulationConfig(runs=1_000, years=5, horizons=[5], seed=8)
    config.market.stock_return = 0.0
    config.market.stock_volatility = 0.0
    config.market.home_appreciation = 0.0
    config.market.home_volatility = 0.0
    config.market.rent_growth = 0.0
    config.market.rent_volatility = 0.0
    config.housing.property_tax_rate = 0.0
    config.housing.insurance_rate = 0.0
    config.housing.maintenance_rate = 0.0
    config.housing.sale_cost_rate = 0.0
    config.mortgage.refinance_enabled = False
    config.regimes = RegimeAssumptions(
        names=["normal"],
        transition=[[1.0]],
        stock_return_shifts=[0.0],
        home_return_shifts=[0.0],
        rent_growth_shifts=[0.0],
        mortgage_rate_shifts=[0.0],
        volatility_multipliers=[1.0],
    )
    return config


def test_sweat_equity_adds_value_and_charges_cash_cost_at_completion():
    base = deterministic_sweat_config()
    without = run_simulation(base)
    with_project = deepcopy(base)
    with_project.sweat_equity.enabled = True
    with_project.sweat_equity.completion_year = 1
    with_project.sweat_equity.cash_cost = 20_000
    with_project.sweat_equity.value_added_low = 100_000
    with_project.sweat_equity.value_added_expected = 100_000
    with_project.sweat_equity.value_added_high = 100_000
    result = run_simulation(with_project)
    incremental = result.net_worth_differences[5] - without.net_worth_differences[5]
    assert np.allclose(incremental, 80_000)


def test_sweat_equity_time_value_is_reported_separately():
    config = deterministic_sweat_config()
    config.sweat_equity.enabled = True
    config.sweat_equity.completion_year = 1
    config.sweat_equity.labor_hours = 1_000
    config.sweat_equity.hourly_time_value = 40
    config.sweat_equity.value_added_low = 100_000
    config.sweat_equity.value_added_expected = 100_000
    config.sweat_equity.value_added_high = 100_000
    result = run_simulation(config)
    row = result.summary.iloc[0]
    assert row.median_economic_net_worth_difference == (
        row.median_net_worth_difference - 40_000
    )


def test_sweat_equity_analysis_curve_is_monotonic():
    config = deterministic_sweat_config()
    config.sweat_equity.enabled = True
    config.sweat_equity.completion_year = 1
    config.sweat_equity.cash_cost = 25_000
    config.sweat_equity.value_added_low = 50_000
    config.sweat_equity.value_added_expected = 100_000
    config.sweat_equity.value_added_high = 150_000
    result = run_sweat_equity_analysis(config, runs=1_000, curve_points=5)
    assert result.curve.financial_buy_probability.is_monotonic_increasing
    assert result.summary.loc[0, "median_incremental_financial_value"] > 0


def test_required_uplift_does_not_depend_on_optional_value_estimate():
    low_estimate = deterministic_sweat_config()
    low_estimate.sweat_equity.enabled = False
    low_estimate.sweat_equity.cash_cost = 25_000
    low_estimate.sweat_equity.value_added_low = 0
    low_estimate.sweat_equity.value_added_expected = 10_000
    low_estimate.sweat_equity.value_added_high = 20_000
    high_estimate = deepcopy(low_estimate)
    high_estimate.sweat_equity.value_added_low = 100_000
    high_estimate.sweat_equity.value_added_expected = 150_000
    high_estimate.sweat_equity.value_added_high = 200_000

    first = run_sweat_equity_analysis(low_estimate, runs=1_000, curve_points=5)
    second = run_sweat_equity_analysis(high_estimate, runs=1_000, curve_points=5)

    assert first.financial_required_uplift == second.financial_required_uplift
    assert first.economic_required_uplift == second.economic_required_uplift
    assert (
        first.summary.loc[0, "median_value_added_at_completion"]
        < second.summary.loc[0, "median_value_added_at_completion"]
    )
