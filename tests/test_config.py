import pytest

from buy_vs_rent.config import SimulationConfig
from buy_vs_rent.web_server import config_from_payload, run_payload


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
