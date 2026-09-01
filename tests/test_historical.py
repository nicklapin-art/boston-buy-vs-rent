from buy_vs_rent import SimulationConfig
from buy_vs_rent.historical import (
    load_historical_dataset,
    replay_historical_cohort,
    run_historical_backtest,
)


def test_historical_sources_align_to_complete_panel():
    dataset = load_historical_dataset()
    assert dataset.latest_data_year == 2023
    assert dataset.annual.loc[1991, "property_tax_rate"] == 0.00893
    assert dataset.annual.loc[2008, "stock_return"] < 0
    assert dataset.annual.loc[2008, "home_return"] < 0


def test_historical_cohort_replay_returns_complete_accounting():
    result = replay_historical_cohort(SimulationConfig(), load_historical_dataset(), 2000, 10)
    assert result["start_year"] == 2000
    assert result["end_year"] == 2009
    assert isinstance(result["realized_buy_win"], bool)
    assert result["starting_purchase_price"] > 0
    assert result["starting_monthly_rent"] > 0


def test_small_historical_backtest_has_calibration_metrics():
    config = SimulationConfig(runs=100)
    result = run_historical_backtest(
        config,
        forecast_runs=100,
        horizons=(5,),
        first_start_year=2018,
    )
    assert len(result.cohorts) == 2
    assert result.calibration.loc[0, "cohorts"] == 2
    assert 0 <= result.calibration.loc[0, "forecast_90_interval_coverage"] <= 1
