from copy import deepcopy

import numpy as np
import pytest

from buy_vs_rent.flip_config import FlipConfig
from buy_vs_rent.flip_simulation import run_flip_simulation
from buy_vs_rent.flip_web_server import config_from_payload, defaults_payload, simulation_payload


def deterministic_config() -> FlipConfig:
    config = FlipConfig(runs=1_000, seed=7)
    config.acquisition.purchase_price = 100.0
    config.acquisition.as_is_market_value = 100.0
    config.acquisition.down_payment = 100.0
    config.acquisition.purchase_closing_cost_rate = 0.0
    config.acquisition.annual_property_tax_rate = 0.0
    config.acquisition.annual_insurance_rate = 0.0
    config.acquisition.monthly_utilities = 0.0
    config.acquisition.monthly_other_carry = 0.0
    config.renovation.planned_budget = 10.0
    config.renovation.expected_cost_overrun_rate = 0.0
    config.renovation.cost_volatility = 0.0
    config.renovation.duration_months = 1.0
    config.renovation.duration_volatility_months = 0.0
    config.financing.annual_interest_rate = 0.0
    config.financing.lender_points_rate = 0.0
    config.exit.after_repair_value = 150.0
    config.exit.arv_uncertainty = 0.0
    config.exit.marketing_months = 0.0
    config.exit.selling_cost_rate = 0.10
    config.exit.estimated_tax_rate = 0.0
    config.exit.max_holding_months = 2
    config.market.annual_home_appreciation = 0.0
    config.market.annual_home_volatility = 0.0
    config.market.transition = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    return config


def test_deterministic_flip_accounting() -> None:
    result = run_flip_simulation(deterministic_config())
    assert result.summary["median_sale_price"] == pytest.approx(150.0)
    assert result.summary["median_pretax_profit"] == pytest.approx(25.0)
    assert result.summary["median_after_tax_profit"] == pytest.approx(25.0)
    assert result.summary["median_break_even_sale_price"] == pytest.approx(110.0 / 0.9)
    assert result.summary["median_maximum_cash_required"] == pytest.approx(110.0)
    assert result.summary["probability_of_profit"] == 1.0


def test_higher_arv_improves_profit_on_common_paths() -> None:
    low = FlipConfig(runs=2_000, seed=11)
    high = deepcopy(low)
    high.exit.after_repair_value += 200_000.0
    low_result = run_flip_simulation(low)
    high_result = run_flip_simulation(high)
    assert high_result.summary["median_after_tax_profit"] > low_result.summary["median_after_tax_profit"]
    assert high_result.summary["probability_of_profit"] >= low_result.summary["probability_of_profit"]


def test_flip_simulation_is_reproducible() -> None:
    config = FlipConfig(runs=1_000, seed=123)
    first = run_flip_simulation(config)
    second = run_flip_simulation(config)
    assert first.summary == second.summary
    assert np.array_equal(first.profit, second.profit)


def test_flip_config_rejects_impossible_down_payment() -> None:
    config = FlipConfig()
    config.acquisition.down_payment = config.acquisition.purchase_price + 1.0
    with pytest.raises(ValueError, match="down payment"):
        config.validate()


def test_flip_web_payload_and_defaults() -> None:
    defaults = defaults_payload()
    payload = {
        "runs": 1_000,
        "seed": defaults["seed"],
        "acquisition": defaults["acquisition"],
        "renovation": defaults["renovation"],
        "financing": defaults["financing"],
        "exit": defaults["exit"],
        "market": {
            "annual_home_appreciation": defaults["market"]["annual_home_appreciation"],
            "annual_home_volatility": defaults["market"]["annual_home_volatility"],
        },
    }
    config = config_from_payload(payload)
    assert config.acquisition.purchase_price == defaults["acquisition"]["purchase_price"]
    response = simulation_payload(payload)
    assert response["runs"] == 1_000
    assert len(response["histogram"]) == 31
    assert response["regimes"]
