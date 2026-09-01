from copy import deepcopy

import numpy as np

from buy_vs_rent.config import RegimeAssumptions, SimulationConfig
from buy_vs_rent.simulation import run_simulation


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

