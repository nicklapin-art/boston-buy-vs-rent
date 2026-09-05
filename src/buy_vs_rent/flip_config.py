"""Configuration for the independent real-estate flip simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class AcquisitionAssumptions:
    purchase_price: float = 1_200_000.0
    as_is_market_value: float = 1_200_000.0
    down_payment: float = 300_000.0
    purchase_closing_cost_rate: float = 0.025
    annual_property_tax_rate: float = 0.0124
    annual_insurance_rate: float = 0.0035
    monthly_utilities: float = 750.0
    monthly_other_carry: float = 250.0


@dataclass
class RenovationAssumptions:
    planned_budget: float = 150_000.0
    expected_cost_overrun_rate: float = 0.10
    cost_volatility: float = 0.15
    duration_months: float = 8.0
    duration_volatility_months: float = 2.0
    cost_duration_correlation: float = 0.60


@dataclass
class FinancingAssumptions:
    annual_interest_rate: float = 0.09
    lender_points_rate: float = 0.02


@dataclass
class ExitAssumptions:
    after_repair_value: float = 1_650_000.0
    arv_uncertainty: float = 0.07
    marketing_months: float = 2.0
    selling_cost_rate: float = 0.055
    estimated_tax_rate: float = 0.30
    annual_hurdle_rate: float = 0.20
    max_holding_months: int = 36


@dataclass
class FlipMarketAssumptions:
    annual_home_appreciation: float = 0.035
    annual_home_volatility: float = 0.07
    names: list[str] = field(default_factory=lambda: ["normal", "slowdown", "crash"])
    transition: list[list[float]] = field(
        default_factory=lambda: [
            [0.965, 0.030, 0.005],
            [0.120, 0.840, 0.040],
            [0.180, 0.420, 0.400],
        ]
    )
    annual_return_shifts: list[float] = field(
        default_factory=lambda: [0.00, -0.08, -0.25]
    )
    volatility_multipliers: list[float] = field(
        default_factory=lambda: [1.00, 1.30, 1.75]
    )
    exit_discount_rates: list[float] = field(
        default_factory=lambda: [0.00, 0.03, 0.10]
    )


@dataclass
class FlipConfig:
    acquisition: AcquisitionAssumptions = field(default_factory=AcquisitionAssumptions)
    renovation: RenovationAssumptions = field(default_factory=RenovationAssumptions)
    financing: FinancingAssumptions = field(default_factory=FinancingAssumptions)
    exit: ExitAssumptions = field(default_factory=ExitAssumptions)
    market: FlipMarketAssumptions = field(default_factory=FlipMarketAssumptions)
    runs: int = 50_000
    seed: int = 42

    def validate(self) -> None:
        a, r, f, e, m = (
            self.acquisition,
            self.renovation,
            self.financing,
            self.exit,
            self.market,
        )
        if not 1 <= self.runs <= 100_000:
            raise ValueError("runs must be between 1 and 100,000")
        if a.purchase_price <= 0 or a.as_is_market_value <= 0:
            raise ValueError("purchase price and as-is market value must be positive")
        if not 0 <= a.down_payment <= a.purchase_price:
            raise ValueError("down payment must be between zero and the purchase price")
        if r.planned_budget < 0 or r.duration_months < 1:
            raise ValueError("renovation budget cannot be negative and duration must be positive")
        if r.duration_volatility_months < 0:
            raise ValueError("duration uncertainty cannot be negative")
        if not -0.99 <= r.cost_duration_correlation <= 0.99:
            raise ValueError("cost-duration correlation must be between -0.99 and 0.99")
        if e.after_repair_value <= 0 or e.marketing_months < 0:
            raise ValueError("after-repair value must be positive and marketing time non-negative")
        if not 1 <= e.max_holding_months <= 60:
            raise ValueError("maximum holding period must be between 1 and 60 months")
        for name, value in {
            "purchase closing cost rate": a.purchase_closing_cost_rate,
            "property tax rate": a.annual_property_tax_rate,
            "insurance rate": a.annual_insurance_rate,
            "expected cost overrun": r.expected_cost_overrun_rate,
            "renovation cost volatility": r.cost_volatility,
            "interest rate": f.annual_interest_rate,
            "lender points": f.lender_points_rate,
            "ARV uncertainty": e.arv_uncertainty,
            "selling cost rate": e.selling_cost_rate,
            "estimated tax rate": e.estimated_tax_rate,
            "annual hurdle rate": e.annual_hurdle_rate,
            "home-price volatility": m.annual_home_volatility,
        }.items():
            if not 0 <= value <= 2:
                raise ValueError(f"{name} must be between 0% and 200%")
        if a.monthly_utilities < 0 or a.monthly_other_carry < 0:
            raise ValueError("monthly carrying costs cannot be negative")
        count = len(m.names)
        transition = np.asarray(m.transition, dtype=float)
        arrays = (m.annual_return_shifts, m.volatility_multipliers, m.exit_discount_rates)
        if transition.shape != (count, count) or any(len(values) != count for values in arrays):
            raise ValueError("market regime arrays must match the regime names")
        if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("each market transition row must be non-negative and sum to one")
        if any(value <= 0 for value in m.volatility_multipliers):
            raise ValueError("volatility multipliers must be positive")
        if any(not 0 <= value < 1 for value in m.exit_discount_rates):
            raise ValueError("exit discounts must be between 0% and 100%")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlipConfig":
        allowed = {"acquisition", "renovation", "financing", "exit", "market", "runs", "seed"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown setting: {sorted(unknown)[0]}")
        config = cls(
            acquisition=AcquisitionAssumptions(**data.get("acquisition", {})),
            renovation=RenovationAssumptions(**data.get("renovation", {})),
            financing=FinancingAssumptions(**data.get("financing", {})),
            exit=ExitAssumptions(**data.get("exit", {})),
            market=FlipMarketAssumptions(**data.get("market", {})),
            runs=int(data.get("runs", cls.runs)),
            seed=int(data.get("seed", cls.seed)),
        )
        config.validate()
        return config
