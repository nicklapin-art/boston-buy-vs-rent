"""Configuration objects and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class HousingAssumptions:
    purchase_price: float = 1_200_000.0
    down_payment: float = 250_000.0
    monthly_rent: float = 4_500.0
    property_tax_rate: float = 0.0124
    insurance_rate: float = 0.0035
    maintenance_rate: float = 0.0100
    annual_hoa: float = 0.0
    purchase_closing_cost_rate: float = 0.025
    sale_cost_rate: float = 0.055


@dataclass
class MortgageAssumptions:
    initial_rate: float = 0.0666
    term_years: int = 30
    long_run_rate: float = 0.055
    rate_mean_reversion: float = 0.25
    annual_rate_volatility: float = 0.0075
    rate_floor: float = 0.025
    rate_cap: float = 0.12
    refinance_enabled: bool = True
    refinance_threshold: float = 0.0100
    refinance_cost_rate: float = 0.015
    refinance_fixed_cost: float = 2_500.0
    refinance_term_years: int = 30
    minimum_refinance_balance: float = 100_000.0


@dataclass
class MarketAssumptions:
    stock_return: float = 0.070
    stock_volatility: float = 0.180
    home_appreciation: float = 0.035
    home_volatility: float = 0.070
    rent_growth: float = 0.030
    rent_volatility: float = 0.030
    general_inflation: float = 0.025
    # Shock order is stocks, home prices, rents, mortgage rates.
    correlation: list[list[float]] = field(
        default_factory=lambda: [
            [1.00, 0.25, 0.15, 0.10],
            [0.25, 1.00, 0.45, -0.15],
            [0.15, 0.45, 1.00, 0.10],
            [0.10, -0.15, 0.10, 1.00],
        ]
    )


@dataclass
class SweatEquityAssumptions:
    """A discrete DIY project completed during the ownership period."""

    enabled: bool = False
    completion_year: int = 2
    cash_cost: float = 20_000.0
    labor_hours: float = 750.0
    hourly_time_value: float = 40.0
    value_added_low: float = 15_000.0
    value_added_expected: float = 30_000.0
    value_added_high: float = 45_000.0


@dataclass
class RegimeAssumptions:
    names: list[str] = field(default_factory=lambda: ["normal", "recession", "crash"])
    transition: list[list[float]] = field(
        default_factory=lambda: [
            [0.915, 0.065, 0.020],
            [0.550, 0.380, 0.070],
            [0.500, 0.350, 0.150],
        ]
    )
    stock_return_shifts: list[float] = field(default_factory=lambda: [0.00, -0.15, -0.42])
    home_return_shifts: list[float] = field(default_factory=lambda: [0.00, -0.07, -0.18])
    rent_growth_shifts: list[float] = field(default_factory=lambda: [0.00, -0.015, -0.040])
    mortgage_rate_shifts: list[float] = field(default_factory=lambda: [0.00, -0.006, -0.015])
    volatility_multipliers: list[float] = field(default_factory=lambda: [1.00, 1.25, 1.60])


@dataclass
class SimulationConfig:
    housing: HousingAssumptions = field(default_factory=HousingAssumptions)
    mortgage: MortgageAssumptions = field(default_factory=MortgageAssumptions)
    market: MarketAssumptions = field(default_factory=MarketAssumptions)
    sweat_equity: SweatEquityAssumptions = field(default_factory=SweatEquityAssumptions)
    regimes: RegimeAssumptions = field(default_factory=RegimeAssumptions)
    runs: int = 100_000
    years: int = 20
    horizons: list[int] = field(default_factory=lambda: [5, 10, 20])
    seed: int = 42

    def validate(self) -> None:
        h, m, market, sweat, regimes = (
            self.housing,
            self.mortgage,
            self.market,
            self.sweat_equity,
            self.regimes,
        )
        if self.runs < 1 or self.years < 1:
            raise ValueError("runs and years must be positive")
        if not self.horizons or min(self.horizons) < 1 or max(self.horizons) > self.years:
            raise ValueError("horizons must fall between 1 and years")
        if h.purchase_price <= 0 or not 0 <= h.down_payment < h.purchase_price:
            raise ValueError("down_payment must be non-negative and below purchase_price")
        if h.monthly_rent < 0:
            raise ValueError("monthly_rent cannot be negative")
        for name in (
            "property_tax_rate", "insurance_rate", "maintenance_rate",
            "purchase_closing_cost_rate", "sale_cost_rate",
        ):
            if getattr(h, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if m.term_years < 1 or m.refinance_term_years < 1:
            raise ValueError("mortgage terms must be positive")
        if not 0 < m.rate_floor <= m.initial_rate <= m.rate_cap:
            raise ValueError("mortgage rates must satisfy floor <= initial <= cap")
        if sweat.completion_year < 1:
            raise ValueError("sweat-equity completion_year must be positive")
        for name in ("cash_cost", "labor_hours", "hourly_time_value"):
            if getattr(sweat, name) < 0:
                raise ValueError(f"sweat-equity {name} cannot be negative")
        if not 0 <= sweat.value_added_low <= sweat.value_added_expected <= sweat.value_added_high:
            raise ValueError(
                "sweat-equity value must satisfy 0 <= low <= expected <= high"
            )
        if sweat.enabled and sweat.completion_year > self.years:
            raise ValueError("sweat-equity completion_year cannot exceed simulation years")

        corr = np.asarray(market.correlation, dtype=float)
        if corr.shape != (4, 4) or not np.allclose(corr, corr.T):
            raise ValueError("correlation must be a symmetric 4x4 matrix")
        if not np.allclose(np.diag(corr), 1.0):
            raise ValueError("correlation diagonal must be one")
        if np.linalg.eigvalsh(corr).min() < -1e-10:
            raise ValueError("correlation matrix must be positive semidefinite")

        count = len(regimes.names)
        arrays = (
            regimes.stock_return_shifts, regimes.home_return_shifts,
            regimes.rent_growth_shifts, regimes.mortgage_rate_shifts,
            regimes.volatility_multipliers,
        )
        transition = np.asarray(regimes.transition, dtype=float)
        if transition.shape != (count, count) or any(len(x) != count for x in arrays):
            raise ValueError("regime arrays and transition matrix must match regime names")
        if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("each regime transition row must be non-negative and sum to one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        config = cls(
            housing=HousingAssumptions(**data.get("housing", {})),
            mortgage=MortgageAssumptions(**data.get("mortgage", {})),
            market=MarketAssumptions(**data.get("market", {})),
            sweat_equity=SweatEquityAssumptions(**data.get("sweat_equity", {})),
            regimes=RegimeAssumptions(**data.get("regimes", {})),
            **{
                k: v
                for k, v in data.items()
                if k not in {"housing", "mortgage", "market", "sweat_equity", "regimes"}
            },
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulationConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
