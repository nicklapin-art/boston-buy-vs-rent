"""Vectorized Monte Carlo engine for the buy-versus-rent decision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import SimulationConfig
from .mortgage import amortize_year


FloatArray = NDArray[np.float64]


@dataclass
class SimulationResult:
    """Summary tables plus horizon-level path results for further analysis."""

    config: SimulationConfig
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    net_worth_differences: dict[int, FloatArray]
    buyer_net_worth: dict[int, FloatArray]
    renter_net_worth: dict[int, FloatArray]


def _correlation_root(matrix: list[list[float]]) -> FloatArray:
    """Return a stable square root that also supports semidefinite matrices."""
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    return vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))


def _next_regime(
    current: NDArray[np.int_],
    transition: FloatArray,
    uniforms: FloatArray,
) -> NDArray[np.int_]:
    cumulative = np.cumsum(transition[current], axis=1)
    return np.sum(uniforms[:, None] > cumulative, axis=1).astype(int)


def _draw_arithmetic_returns(
    target_mean: FloatArray,
    volatility: FloatArray,
    standard_shock: FloatArray,
) -> FloatArray:
    """Draw lognormal arithmetic returns with the requested conditional mean."""
    safe_mean = np.maximum(target_mean, -0.95)
    log_return = np.log1p(safe_mean) - 0.5 * volatility**2 + volatility * standard_shock
    return np.expm1(log_return)


def _draw_sweat_equity_value(config: SimulationConfig, runs: int) -> FloatArray:
    """Draw the completed project's immediate market-value contribution."""

    sweat = config.sweat_equity
    if not sweat.enabled:
        return np.zeros(runs, dtype=float)
    if sweat.value_added_low == sweat.value_added_high:
        return np.full(runs, sweat.value_added_expected, dtype=float)
    rng = np.random.default_rng(config.seed + 610_001)
    return rng.triangular(
        sweat.value_added_low,
        sweat.value_added_expected,
        sweat.value_added_high,
        size=runs,
    )


def run_simulation(config: SimulationConfig | None = None) -> SimulationResult:
    """Run a fully vectorized simulation using annual market and cash-flow steps."""
    config = config or SimulationConfig()
    config.validate()
    h, m, market, sweat, regime_cfg = (
        config.housing,
        config.mortgage,
        config.market,
        config.sweat_equity,
        config.regimes,
    )
    n = config.runs
    rng = np.random.default_rng(config.seed)
    root = _correlation_root(market.correlation)
    transition = np.asarray(regime_cfg.transition, dtype=float)

    mortgage_balance = np.full(n, h.purchase_price - h.down_payment, dtype=float)
    mortgage_rate = np.full(n, m.initial_rate, dtype=float)
    remaining_months = np.full(n, m.term_years * 12, dtype=int)
    market_rate = np.full(n, m.initial_rate, dtype=float)
    home_value = np.full(n, h.purchase_price, dtype=float)
    monthly_rent = np.full(n, h.monthly_rent, dtype=float)
    buyer_portfolio = np.zeros(n, dtype=float)
    renter_portfolio = np.full(
        n,
        h.down_payment + h.purchase_price * h.purchase_closing_cost_rate,
        dtype=float,
    )
    annual_hoa = np.full(n, h.annual_hoa, dtype=float)
    regime = np.zeros(n, dtype=int)
    sweat_value_added = _draw_sweat_equity_value(config, n)

    stock_shifts = np.asarray(regime_cfg.stock_return_shifts)
    home_shifts = np.asarray(regime_cfg.home_return_shifts)
    rent_shifts = np.asarray(regime_cfg.rent_growth_shifts)
    rate_shifts = np.asarray(regime_cfg.mortgage_rate_shifts)
    vol_multipliers = np.asarray(regime_cfg.volatility_multipliers)

    difference_paths: dict[int, FloatArray] = {}
    buyer_paths: dict[int, FloatArray] = {}
    renter_paths: dict[int, FloatArray] = {}
    diagnostics: list[dict[str, float | int]] = []
    horizon_set = set(config.horizons)

    for year in range(1, config.years + 1):
        regime = _next_regime(regime, transition, rng.random(n))
        correlated = rng.standard_normal((n, 4)) @ root.T
        vol_multiplier = vol_multipliers[regime]

        stock_return = _draw_arithmetic_returns(
            market.stock_return + stock_shifts[regime],
            market.stock_volatility * vol_multiplier,
            correlated[:, 0],
        )
        home_return = _draw_arithmetic_returns(
            market.home_appreciation + home_shifts[regime],
            market.home_volatility * vol_multiplier,
            correlated[:, 1],
        )
        rent_return = _draw_arithmetic_returns(
            market.rent_growth + rent_shifts[regime],
            market.rent_volatility * vol_multiplier,
            correlated[:, 2],
        )

        market_rate = np.clip(
            market_rate
            + m.rate_mean_reversion * (m.long_run_rate - market_rate)
            + m.annual_rate_volatility * vol_multiplier * correlated[:, 3]
            + rate_shifts[regime],
            m.rate_floor,
            m.rate_cap,
        )
        refinance = (
            m.refinance_enabled
            & (mortgage_balance >= m.minimum_refinance_balance)
            & (remaining_months > 0)
            & (market_rate <= mortgage_rate - m.refinance_threshold)
        )
        refinance_cost = np.where(
            refinance,
            mortgage_balance * m.refinance_cost_rate + m.refinance_fixed_cost,
            0.0,
        )
        mortgage_rate = np.where(refinance, market_rate, mortgage_rate)
        remaining_months = np.where(
            refinance,
            m.refinance_term_years * 12,
            remaining_months,
        )

        mortgage_payment, interest_paid, mortgage_balance, remaining_months = amortize_year(
            mortgage_balance, mortgage_rate, remaining_months
        )
        project_completes = bool(sweat.enabled and year == sweat.completion_year)
        if project_completes:
            home_value += sweat_value_added
        average_home_value = home_value * (1.0 + 0.5 * home_return)
        property_tax = average_home_value * h.property_tax_rate
        insurance = average_home_value * h.insurance_rate
        maintenance = average_home_value * h.maintenance_rate
        owner_cost = (
            mortgage_payment + property_tax + insurance + maintenance
            + annual_hoa + refinance_cost
        )
        if project_completes:
            owner_cost += sweat.cash_cost
        renter_cost = monthly_rent * 12.0

        # Use a common annual housing budget. The cheaper strategy invests the savings.
        buyer_contribution = np.maximum(renter_cost - owner_cost, 0.0)
        renter_contribution = np.maximum(owner_cost - renter_cost, 0.0)
        buyer_portfolio = buyer_portfolio * (1.0 + stock_return) + buyer_contribution
        renter_portfolio = renter_portfolio * (1.0 + stock_return) + renter_contribution

        home_value *= 1.0 + home_return
        monthly_rent *= 1.0 + rent_return
        annual_hoa *= 1.0 + market.general_inflation

        diagnostics.append(
            {
                "year": year,
                "mean_stock_return": float(stock_return.mean()),
                "mean_home_return": float(home_return.mean()),
                "mean_rent_growth": float(rent_return.mean()),
                "mean_market_mortgage_rate": float(market_rate.mean()),
                "mean_mortgage_balance": float(mortgage_balance.mean()),
                "mean_interest_paid": float(interest_paid.mean()),
                "refinance_path_share": float(refinance.mean()),
                "recession_path_share": float(np.mean(regime == 1)) if len(regime_cfg.names) > 1 else 0.0,
                "crash_path_share": float(np.mean(regime == 2)) if len(regime_cfg.names) > 2 else 0.0,
                "mean_sweat_value_added": (
                    float(sweat_value_added.mean()) if project_completes else 0.0
                ),
            }
        )

        if year in horizon_set:
            buyer_net_worth = (
                home_value * (1.0 - h.sale_cost_rate)
                - mortgage_balance
                + buyer_portfolio
            )
            renter_net_worth = renter_portfolio.copy()
            buyer_paths[year] = buyer_net_worth.copy()
            renter_paths[year] = renter_net_worth
            difference_paths[year] = buyer_net_worth - renter_net_worth

    rows: list[dict[str, float | int]] = []
    for horizon in sorted(config.horizons):
        difference = difference_paths[horizon]
        buyer = buyer_paths[horizon]
        renter = renter_paths[horizon]
        rows.append(
            {
                "horizon_years": horizon,
                "buy_win_probability": float(np.mean(difference > 0.0)),
                "median_net_worth_difference": float(np.median(difference)),
                "p05_net_worth_difference": float(np.quantile(difference, 0.05)),
                "p95_net_worth_difference": float(np.quantile(difference, 0.95)),
                "mean_net_worth_difference": float(difference.mean()),
                "median_buyer_net_worth": float(np.median(buyer)),
                "median_renter_net_worth": float(np.median(renter)),
                "economic_buy_win_probability": float(
                    np.mean(
                        difference
                        - (
                            sweat.labor_hours * sweat.hourly_time_value
                            if sweat.enabled and sweat.completion_year <= horizon
                            else 0.0
                        )
                        > 0.0
                    )
                ),
                "median_economic_net_worth_difference": float(
                    np.median(difference)
                    - (
                        sweat.labor_hours * sweat.hourly_time_value
                        if sweat.enabled and sweat.completion_year <= horizon
                        else 0.0
                    )
                ),
            }
        )

    return SimulationResult(
        config=config,
        summary=pd.DataFrame(rows),
        diagnostics=pd.DataFrame(diagnostics),
        net_worth_differences=difference_paths,
        buyer_net_worth=buyer_paths,
        renter_net_worth=renter_paths,
    )
