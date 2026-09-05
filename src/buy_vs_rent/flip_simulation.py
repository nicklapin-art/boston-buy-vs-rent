"""Monthly vectorized Monte Carlo engine for a leveraged real-estate flip."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .flip_config import FlipConfig


FloatArray = NDArray[np.float64]


@dataclass
class FlipSimulationResult:
    config: FlipConfig
    summary: dict[str, float]
    percentiles: pd.DataFrame
    regimes: pd.DataFrame
    histogram: pd.DataFrame
    cost_breakdown: pd.DataFrame
    profit: FloatArray
    annualized_irr: FloatArray


def _next_regime(current: NDArray[np.int_], transition: FloatArray, draws: FloatArray) -> NDArray[np.int_]:
    cumulative = np.cumsum(transition[current], axis=1)
    return np.sum(draws[:, None] > cumulative, axis=1).astype(int)


def _annualized_irr(cash_flows: FloatArray) -> FloatArray:
    """Solve conventional monthly cash flows for annual IRR by vectorized bisection."""

    n, columns = cash_flows.shape
    years = np.arange(columns, dtype=float)[None, :] / 12.0
    lower = np.full(n, -0.999, dtype=float)
    upper = np.full(n, 10.0, dtype=float)
    valid = (cash_flows[:, 0] < 0.0) & np.any(cash_flows[:, 1:] > 0.0, axis=1)
    for _ in range(34):
        midpoint = (lower + upper) / 2.0
        discount = np.exp(-np.log1p(midpoint)[:, None] * years)
        npv = np.sum(cash_flows * discount, axis=1)
        needs_higher_rate = npv > 0.0
        lower = np.where(needs_higher_rate, midpoint, lower)
        upper = np.where(needs_higher_rate, upper, midpoint)
    result = (lower + upper) / 2.0
    # With no positive equity distribution, the investment is treated as a
    # total loss rather than silently dropping that path from IRR percentiles.
    result[~valid] = -1.0
    return result


def _histogram(values: FloatArray, bins: int = 31) -> pd.DataFrame:
    low, high = np.quantile(values, [0.01, 0.99])
    if np.isclose(low, high):
        low, high = float(values.min()), float(values.max() + 1.0)
    counts, edges = np.histogram(np.clip(values, low, high), bins=bins, range=(low, high))
    return pd.DataFrame(
        {
            "low": edges[:-1],
            "high": edges[1:],
            "count": counts,
        }
    )


def run_flip_simulation(config: FlipConfig | None = None) -> FlipSimulationResult:
    config = config or FlipConfig()
    config.validate()
    a, r, f, e, m = (
        config.acquisition,
        config.renovation,
        config.financing,
        config.exit,
        config.market,
    )
    n = config.runs
    rng = np.random.default_rng(config.seed)

    cost_shock = rng.standard_normal(n)
    independent_duration_shock = rng.standard_normal(n)
    duration_shock = (
        r.cost_duration_correlation * cost_shock
        + np.sqrt(1.0 - r.cost_duration_correlation**2) * independent_duration_shock
    )
    if r.cost_volatility == 0.0:
        rehab_cost = np.full(n, r.planned_budget * (1.0 + r.expected_cost_overrun_rate))
    else:
        expected_factor = 1.0 + r.expected_cost_overrun_rate
        log_mean = np.log(expected_factor) - 0.5 * r.cost_volatility**2
        rehab_cost = r.planned_budget * np.exp(log_mean + r.cost_volatility * cost_shock)
    renovation_months = np.rint(
        r.duration_months + r.duration_volatility_months * duration_shock
    ).astype(int)
    renovation_months = np.clip(renovation_months, 1, e.max_holding_months)
    marketing_months = rng.poisson(e.marketing_months, size=n)
    holding_months = np.clip(
        renovation_months + marketing_months,
        1,
        e.max_holding_months,
    )

    arv_shock = rng.standard_normal(n)
    if e.arv_uncertainty == 0.0:
        property_specific_factor = np.ones(n)
    else:
        property_specific_factor = np.exp(
            -0.5 * e.arv_uncertainty**2 + e.arv_uncertainty * arv_shock
        )

    loan_amount = a.purchase_price - a.down_payment
    purchase_closing_cost = a.purchase_price * a.purchase_closing_cost_rate
    lender_points = loan_amount * f.lender_points_rate
    initial_equity = a.down_payment + purchase_closing_cost + lender_points
    monthly_interest = loan_amount * f.annual_interest_rate / 12.0

    max_months = e.max_holding_months
    cash_flows = np.zeros((n, max_months + 1), dtype=float)
    cash_flows[:, 0] = -initial_equity
    market_factor = np.ones(n, dtype=float)
    regime = np.zeros(n, dtype=int)
    transition = np.asarray(m.transition, dtype=float)
    shifts = np.asarray(m.annual_return_shifts, dtype=float)
    vol_multipliers = np.asarray(m.volatility_multipliers, dtype=float)
    exit_discounts = np.asarray(m.exit_discount_rates, dtype=float)

    sale_price = np.zeros(n, dtype=float)
    as_is_sale_price = np.zeros(n, dtype=float)
    exit_regime = np.zeros(n, dtype=int)
    operating_cost = np.zeros(n, dtype=float)
    interest_cost = np.zeros(n, dtype=float)

    for month in range(1, max_months + 1):
        active = holding_months >= month
        regime = _next_regime(regime, transition, rng.random(n))
        annual_mean = np.maximum(m.annual_home_appreciation + shifts[regime], -0.95)
        monthly_mean = np.power(1.0 + annual_mean, 1.0 / 12.0) - 1.0
        monthly_volatility = m.annual_home_volatility * vol_multipliers[regime] / np.sqrt(12.0)
        market_return = np.expm1(
            np.log1p(monthly_mean)
            - 0.5 * monthly_volatility**2
            + monthly_volatility * rng.standard_normal(n)
        )
        market_factor = np.where(active, market_factor * (1.0 + market_return), market_factor)

        estimated_property_value = a.as_is_market_value * market_factor
        monthly_operating = (
            estimated_property_value
            * (a.annual_property_tax_rate + a.annual_insurance_rate)
            / 12.0
            + a.monthly_utilities
            + a.monthly_other_carry
        )
        monthly_carry = monthly_operating + monthly_interest
        cash_flows[active, month] -= monthly_carry[active]
        operating_cost[active] += monthly_operating[active]
        interest_cost[active] += monthly_interest

        renovating = active & (renovation_months >= month)
        monthly_rehab = rehab_cost / renovation_months
        cash_flows[renovating, month] -= monthly_rehab[renovating]

        exits = holding_months == month
        if np.any(exits):
            exit_factor = 1.0 - exit_discounts[regime[exits]]
            sale_price[exits] = (
                e.after_repair_value
                * market_factor[exits]
                * property_specific_factor[exits]
                * exit_factor
            )
            as_is_sale_price[exits] = (
                a.as_is_market_value * market_factor[exits] * exit_factor
            )
            exit_regime[exits] = regime[exits]
            net_sale_proceeds = sale_price[exits] * (1.0 - e.selling_cost_rate)
            cash_flows[exits, month] += net_sale_proceeds - loan_amount

    pretax_profit = cash_flows.sum(axis=1)
    estimated_tax = np.maximum(pretax_profit, 0.0) * e.estimated_tax_rate
    after_tax_cash_flows = cash_flows.copy()
    after_tax_cash_flows[np.arange(n), holding_months] -= estimated_tax
    after_tax_profit = pretax_profit - estimated_tax
    annualized_irr = _annualized_irr(after_tax_cash_flows)

    # All operating, financing, and renovation outflows occur before sale proceeds.
    # This avoids understating liquidity when the final renovation spend and sale
    # happen within the same monthly accounting period.
    maximum_cash_required = initial_equity + rehab_cost + operating_cost + interest_cost
    selling_cost = sale_price * e.selling_cost_rate
    financing_cost = lender_points + interest_cost
    total_project_cost = (
        a.purchase_price
        + purchase_closing_cost
        + rehab_cost
        + operating_cost
        + financing_cost
    )
    break_even_sale_price = total_project_cost / (1.0 - e.selling_cost_rate)
    renovation_value_created = sale_price - as_is_sale_price

    finite_irr = annualized_irr[np.isfinite(annualized_irr)]
    irr_for_summary = finite_irr if finite_irr.size else np.array([-1.0])
    summary = {
        "probability_of_profit": float(np.mean(after_tax_profit > 0.0)),
        "probability_beats_hurdle": float(np.mean(annualized_irr >= e.annual_hurdle_rate)),
        "median_pretax_profit": float(np.median(pretax_profit)),
        "median_after_tax_profit": float(np.median(after_tax_profit)),
        "p05_after_tax_profit": float(np.quantile(after_tax_profit, 0.05)),
        "p95_after_tax_profit": float(np.quantile(after_tax_profit, 0.95)),
        "median_annualized_irr": float(np.median(irr_for_summary)),
        "p05_annualized_irr": float(np.quantile(irr_for_summary, 0.05)),
        "p95_annualized_irr": float(np.quantile(irr_for_summary, 0.95)),
        "median_sale_price": float(np.median(sale_price)),
        "median_rehab_cost": float(np.median(rehab_cost)),
        "median_holding_months": float(np.median(holding_months)),
        "p95_holding_months": float(np.quantile(holding_months, 0.95)),
        "median_maximum_cash_required": float(np.median(maximum_cash_required)),
        "p95_maximum_cash_required": float(np.quantile(maximum_cash_required, 0.95)),
        "median_break_even_sale_price": float(np.median(break_even_sale_price)),
        "median_renovation_value_created": float(np.median(renovation_value_created)),
    }

    percentile_levels = np.array([5, 10, 25, 50, 75, 90, 95])
    percentiles = pd.DataFrame(
        {
            "percentile": percentile_levels,
            "after_tax_profit": np.percentile(after_tax_profit, percentile_levels),
            "annualized_irr": np.percentile(irr_for_summary, percentile_levels),
            "sale_price": np.percentile(sale_price, percentile_levels),
            "maximum_cash_required": np.percentile(maximum_cash_required, percentile_levels),
        }
    )

    regime_rows: list[dict[str, float | str]] = []
    for index, name in enumerate(m.names):
        selected = exit_regime == index
        if not np.any(selected):
            continue
        regime_rows.append(
            {
                "regime": name,
                "path_share": float(np.mean(selected)),
                "median_after_tax_profit": float(np.median(after_tax_profit[selected])),
                "probability_of_profit": float(np.mean(after_tax_profit[selected] > 0.0)),
                "median_sale_price": float(np.median(sale_price[selected])),
            }
        )

    cost_breakdown = pd.DataFrame(
        [
            {"category": "Purchase", "median": a.purchase_price},
            {"category": "Purchase closing", "median": purchase_closing_cost},
            {"category": "Renovation", "median": float(np.median(rehab_cost))},
            {"category": "Operating carry", "median": float(np.median(operating_cost))},
            {"category": "Financing", "median": float(np.median(financing_cost))},
            {"category": "Selling", "median": float(np.median(selling_cost))},
            {"category": "Estimated tax", "median": float(np.median(estimated_tax))},
        ]
    )

    return FlipSimulationResult(
        config=config,
        summary=summary,
        percentiles=percentiles,
        regimes=pd.DataFrame(regime_rows),
        histogram=_histogram(after_tax_profit),
        cost_breakdown=cost_breakdown,
        profit=after_tax_profit,
        annualized_irr=annualized_irr,
    )
