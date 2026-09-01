import pytest

from buy_vs_rent.config import SimulationConfig
from buy_vs_rent.web_server import (
    config_from_payload,
    defaults_payload,
    run_payload,
    sweat_equity_payload,
)


def test_json_round_trip(tmp_path):
    original = SimulationConfig(runs=77, seed=9)
    path = tmp_path / "config.json"
    original.to_json(path)
    assert SimulationConfig.from_json(path) == original


def test_invalid_correlation_rejected():
    config = SimulationConfig()
    config.market.correlation[0][1] = 0.9
    with pytest.raises(ValueError, match="symmetric"):
        config.validate()


def test_web_payload_builds_requested_horizons_without_mutation():
    payload = {
        "runs": 250,
        "years": 7,
        "seed": 5,
        "housing": {"monthly_rent": 5000},
        "mortgage": {"initial_rate": 0.06, "refinance_enabled": False},
        "market": {"home_appreciation": 0.04},
    }
    config = config_from_payload(payload)
    assert config.runs == 250
    assert config.horizons == [5, 7]
    assert config.housing.monthly_rent == 5000
    assert payload["mortgage"]["refinance_enabled"] is False


def test_web_payload_runs_simulation():
    output = run_payload({"runs": 250, "years": 5, "seed": 1})
    assert output["runs"] == 250
    assert output["summary"][0]["horizon_years"] == 5


def test_web_payload_rejects_oversized_run():
    with pytest.raises(ValueError, match="runs must be between"):
        config_from_payload({"runs": 100001})


def test_rejects_invalid_sweat_equity_range():
    config = SimulationConfig()
    config.sweat_equity.enabled = True
    config.sweat_equity.value_added_low = 200_000
    config.sweat_equity.value_added_expected = 100_000
    config.sweat_equity.value_added_high = 300_000
    with pytest.raises(ValueError, match="low <= expected <= high"):
        config.validate()


def test_browser_defaults_prefill_opt_in_kitchen_remodel():
    sweat = defaults_payload()["sweat_equity"]
    assert sweat == {
        "enabled": False,
        "completion_year": 2,
        "cash_cost": 20_000.0,
        "labor_hours": 750.0,
        "hourly_time_value": 40.0,
        "value_added_low": 15_000.0,
        "value_added_expected": 30_000.0,
        "value_added_high": 45_000.0,
    }


def test_sweat_equity_browser_payload():
    payload = sweat_equity_payload(
        {
            "scenario": {
                "runs": 1_000,
                "years": 5,
                "seed": 3,
                "sweat_equity": {
                    "enabled": True,
                    "completion_year": 2,
                    "cash_cost": 20_000,
                    "labor_hours": 500,
                    "hourly_time_value": 40,
                    "value_added_low": 50_000,
                    "value_added_expected": 100_000,
                    "value_added_high": 150_000,
                },
            },
            "runs": 1_000,
            "curve_points": 5,
            "horizon": 5,
        }
    )
    assert payload["runs"] == 1_000
    assert len(payload["curve"]) == 5
    assert payload["summary"]["time_cost"] == 20_000
