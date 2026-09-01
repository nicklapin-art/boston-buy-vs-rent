import numpy as np
import pandas as pd

from buy_vs_rent import SimulationConfig
from buy_vs_rent.historical_calibration import (
    run_historically_calibrated_uncertainty,
    sample_historically_calibrated_parameters,
)
from buy_vs_rent.uncertainty import (
    ParameterRange,
    default_parameter_ranges,
    run_parameter_uncertainty,
    sample_parameter_sets,
)
from buy_vs_rent.web_server import robustness_payload


def small_config() -> SimulationConfig:
    config = SimulationConfig(runs=500, years=5, horizons=[5], seed=17)
    return config


def test_default_ranges_are_centered_on_scenario() -> None:
    config = SimulationConfig()
    ranges = {item.key: item for item in default_parameter_ranges(config)}
    assert ranges["stock_return"].mode == config.market.stock_return
    assert ranges["home_appreciation"].mode == config.market.home_appreciation
    assert ranges["maintenance_rate"].mode == config.housing.maintenance_rate
    assert all(item.low <= item.mode <= item.high for item in ranges.values())


def test_parameter_sampling_is_reproducible_and_bounded() -> None:
    ranges = default_parameter_ranges(SimulationConfig())
    first = sample_parameter_sets(ranges, count=16, seed=9)
    second = sample_parameter_sets(ranges, count=16, seed=9)
    pd.testing.assert_frame_equal(first, second)
    for item in ranges:
        assert first[item.key].between(item.low, item.high).all()


def test_uncertainty_summary_is_ordered_and_reproducible() -> None:
    first = run_parameter_uncertainty(
        small_config(), parameter_sets=8, runs_per_set=250
    )
    second = run_parameter_uncertainty(
        small_config(), parameter_sets=8, runs_per_set=250
    )
    pd.testing.assert_frame_equal(first.summary, second.summary)
    row = first.summary.iloc[0]
    assert 0 <= row.p10_buy_win_probability <= row.median_buy_win_probability
    assert row.median_buy_win_probability <= row.p90_buy_win_probability <= 1
    assert 0 <= row.integrated_buy_win_probability <= 1
    assert 0 <= row.robust_buy_share <= 1
    assert set(first.influence["parameter"]) == {
        "stock_return", "home_appreciation", "rent_growth",
        "maintenance_rate", "property_tax_rate", "insurance_rate",
        "sale_cost_rate",
    }
    assert np.isfinite(first.influence["partial_rank_correlation"]).all()


def test_browser_robustness_payload() -> None:
    payload = robustness_payload(
        {
            "scenario": {"runs": 500, "years": 5, "seed": 3},
            "method": "judgment",
            "parameter_sets": 8,
            "runs_per_set": 250,
        }
    )
    assert payload["total_paths"] == 2_000
    assert len(payload["summary"]) == 1
    assert len(payload["ranges"]) == 7
    assert payload["method"] == "judgment"


def test_constant_outcomes_have_finite_zero_influence() -> None:
    config = small_config()
    config.housing.purchase_price = 5_000_000
    config.housing.down_payment = 250_000
    config.housing.monthly_rent = 0
    result = run_parameter_uncertainty(
        config, parameter_sets=8, runs_per_set=250
    )
    assert np.isfinite(result.influence["partial_rank_correlation"]).all()


def test_custom_subset_of_ranges() -> None:
    ranges = [
        ParameterRange(
            "home_appreciation", "Expected home appreciation",
            0.01, 0.035, 0.06, "market",
        )
    ]
    result = run_parameter_uncertainty(
        small_config(), ranges=ranges, parameter_sets=8, runs_per_set=250
    )
    assert set(result.outcomes.columns).issuperset({"home_appreciation", "buy_win_probability"})
    assert result.influence["parameter"].tolist() == ["home_appreciation"]


def test_historical_parameter_draws_are_reproducible_and_user_centered() -> None:
    config = small_config()
    first = sample_historically_calibrated_parameters(config, count=16)
    second = sample_historically_calibrated_parameters(config, count=16)
    pd.testing.assert_frame_equal(first.draws, second.draws)
    assert first.metadata["return_data_start_year"] == 1978
    assert first.metadata["data_end_year"] == 2023
    assert first.metadata["initial_mortgage_rate_fixed"] is True
    assert "initial_rate" not in first.draws.columns
    assert set(first.draws.columns) == {
        "parameter_set", "stock_return", "home_appreciation", "rent_growth",
        "maintenance_rate", "property_tax_rate", "insurance_rate", "sale_cost_rate",
    }
    ranges = {item.key: item for item in first.ranges}
    assert ranges["stock_return"].mode == config.market.stock_return
    assert ranges["home_appreciation"].mode == config.market.home_appreciation
    assert ranges["rent_growth"].mode == config.market.rent_growth
    assert len(first.metadata["walk_forward"]) == 9


def test_historically_calibrated_uncertainty_runs_with_metadata() -> None:
    result = run_historically_calibrated_uncertainty(
        small_config(), parameter_sets=8, runs_per_set=250
    )
    assert result.method == "historical"
    assert result.metadata["method_label"] == "Historically calibrated"
    assert result.summary.loc[0, "parameter_sets"] == 8
    groups = {item.key: item.group for item in result.ranges}
    assert groups["stock_return"] == "historical"
    assert groups["home_appreciation"] == "historical"
    assert groups["rent_growth"] == "historical"
    assert groups["property_tax_rate"] == "historical"
    assert groups["maintenance_rate"] == "judgment"


def test_browser_historical_robustness_payload() -> None:
    payload = robustness_payload(
        {
            "scenario": {"runs": 500, "years": 5, "seed": 3},
            "method": "historical",
            "parameter_sets": 8,
            "runs_per_set": 250,
        }
    )
    assert payload["method"] == "historical"
    assert payload["calibration"]["return_observations"] == 46
    assert len(payload["calibration"]["walk_forward"]) == 9
