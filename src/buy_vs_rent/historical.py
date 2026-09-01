"""Historical Boston cohort replay and Monte Carlo calibration checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .mortgage import amortize_year
from .simulation import run_simulation


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "historical"


@dataclass
class HistoricalDataset:
    annual: pd.DataFrame
    latest_home_index: float
    latest_rent_index: float
    latest_data_year: int


@dataclass
class HistoricalBacktestResult:
    config: SimulationConfig
    cohorts: pd.DataFrame
    calibration: pd.DataFrame
    moments: pd.DataFrame
    correlations: pd.DataFrame
    data_start_year: int
    data_end_year: int
    forecast_runs: int


def _read_fred_csv(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = ["date", value_name]
    frame["date"] = pd.to_datetime(frame["date"])
    frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
    return frame.dropna(subset=[value_name]).sort_values("date")


def _annual_stock_returns(path: Path) -> pd.Series:
    raw = pd.read_excel(path, sheet_name="Data", header=7)
    date_number = pd.to_numeric(raw["Date"], errors="coerce")
    price = pd.to_numeric(raw["P"], errors="coerce")
    dividend = pd.to_numeric(raw["D"], errors="coerce").ffill()
    stock = pd.DataFrame({"date_number": date_number, "price": price, "dividend": dividend})
    stock = stock.dropna().sort_values("date_number")
    stock["year"] = np.floor(stock["date_number"]).astype(int)
    previous_price = stock["price"].shift(1)
    stock["monthly_return"] = (stock["price"] + stock["dividend"] / 12.0) / previous_price - 1.0
    stock = stock.dropna(subset=["monthly_return"])
    return stock.groupby("year")["monthly_return"].apply(lambda values: (1.0 + values).prod() - 1.0)


def load_historical_dataset(data_dir: str | Path | None = None) -> HistoricalDataset:
    """Load raw source files and align them to annual calendar-year observations."""
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    raw = root / "raw"
    home = _read_fred_csv(raw / "boston_fhfa_hpi.csv", "home_index")
    rent = _read_fred_csv(raw / "boston_bls_rent_cpi.csv", "rent_index")
    mortgage = _read_fred_csv(raw / "freddie_mac_mortgage30.csv", "mortgage_rate")
    mortgage["mortgage_rate"] /= 100.0
    stock_return = _annual_stock_returns(raw / "shiller_ie_data.xls")

    home["year"] = home["date"].dt.year
    rent["year"] = rent["date"].dt.year
    mortgage["year"] = mortgage["date"].dt.year
    home_annual = home.groupby("year")["home_index"].last()
    rent_annual = rent.groupby("year")["rent_index"].last()
    mortgage_start = mortgage.groupby("year")["mortgage_rate"].first()
    mortgage_min = mortgage.groupby("year")["mortgage_rate"].min()

    annual = pd.concat(
        {
            "home_index": home_annual,
            "rent_index": rent_annual,
            "stock_return": stock_return,
            "mortgage_start_rate": mortgage_start,
            "mortgage_min_rate": mortgage_min,
        },
        axis=1,
    ).sort_index()
    annual["home_return"] = annual["home_index"].pct_change(fill_method=None)
    annual["rent_return"] = annual["rent_index"].pct_change(fill_method=None)
    annual["mortgage_rate_change"] = annual["mortgage_start_rate"].diff()

    taxes = pd.read_csv(root / "boston_residential_tax_rates.csv")
    taxes["property_tax_rate"] = taxes["residential_rate_per_1000"] / 1000.0
    annual = annual.join(taxes.set_index("year")["property_tax_rate"], how="left")
    latest_complete = annual.dropna(
        subset=[
            "home_return", "rent_return", "stock_return", "mortgage_start_rate",
            "mortgage_min_rate", "property_tax_rate",
        ]
    ).index.max()
    if pd.isna(latest_complete):
        raise ValueError("Historical sources have no overlapping complete years")
    return HistoricalDataset(
        annual=annual,
        latest_home_index=float(home["home_index"].iloc[-1]),
        latest_rent_index=float(rent["rent_index"].iloc[-1]),
        latest_data_year=int(latest_complete),
    )


def _cohort_start_values(
    config: SimulationConfig,
    dataset: HistoricalDataset,
    start_year: int,
) -> tuple[float, float, float]:
    prior_year = start_year - 1
    if prior_year not in dataset.annual.index:
        raise ValueError(f"Missing index data before {start_year}")
    prior = dataset.annual.loc[prior_year]
    purchase_price = (
        config.housing.purchase_price
        * float(prior["home_index"])
        / dataset.latest_home_index
    )
    monthly_rent = (
        config.housing.monthly_rent
        * float(prior["rent_index"])
        / dataset.latest_rent_index
    )
    down_fraction = config.housing.down_payment / config.housing.purchase_price
    return purchase_price, purchase_price * down_fraction, monthly_rent


def replay_historical_cohort(
    config: SimulationConfig,
    dataset: HistoricalDataset,
    start_year: int,
    horizon: int,
) -> dict[str, float | int | bool]:
    """Replay one realized buy/rent cohort using the same accounting as the simulator."""
    end_year = start_year + horizon - 1
    years = list(range(start_year, end_year + 1))
    missing = [year for year in years if year not in dataset.annual.index]
    if missing:
        raise ValueError(f"Missing historical year {missing[0]}")
    history = dataset.annual.loc[years]
    required = [
        "home_return", "rent_return", "stock_return", "mortgage_start_rate",
        "mortgage_min_rate", "property_tax_rate",
    ]
    if history[required].isna().any().any():
        raise ValueError(f"Incomplete historical data for {start_year}-{end_year}")

    h, m, sweat = config.housing, config.mortgage, config.sweat_equity
    purchase_price, down_payment, monthly_rent = _cohort_start_values(config, dataset, start_year)
    home_value = purchase_price
    mortgage_balance = np.array([purchase_price - down_payment], dtype=float)
    mortgage_rate = np.array([float(history.iloc[0]["mortgage_start_rate"])], dtype=float)
    remaining_months = np.array([m.term_years * 12], dtype=int)
    buyer_portfolio = 0.0
    renter_portfolio = down_payment + purchase_price * h.purchase_closing_cost_rate
    annual_hoa = h.annual_hoa * purchase_price / h.purchase_price
    scaled_refinance_fixed_cost = m.refinance_fixed_cost * purchase_price / h.purchase_price
    refinance_count = 0

    for project_year, (_, row) in enumerate(history.iterrows(), start=1):
        available_rate = float(row["mortgage_min_rate"])
        refinance = bool(
            m.refinance_enabled
            and mortgage_balance[0] >= m.minimum_refinance_balance * purchase_price / h.purchase_price
            and remaining_months[0] > 0
            and available_rate <= mortgage_rate[0] - m.refinance_threshold
        )
        refinance_cost = 0.0
        if refinance:
            refinance_count += 1
            mortgage_rate[0] = available_rate
            remaining_months[0] = m.refinance_term_years * 12
            refinance_cost = mortgage_balance[0] * m.refinance_cost_rate + scaled_refinance_fixed_cost

        payment, _, mortgage_balance, remaining_months = amortize_year(
            mortgage_balance, mortgage_rate, remaining_months
        )
        home_return = float(row["home_return"])
        rent_return = float(row["rent_return"])
        stock_return = float(row["stock_return"])
        project_completes = bool(sweat.enabled and project_year == sweat.completion_year)
        if project_completes:
            home_value += sweat.value_added_expected * purchase_price / h.purchase_price
        average_home = home_value * (1.0 + 0.5 * home_return)
        owner_cost = (
            payment[0]
            + average_home * float(row["property_tax_rate"])
            + average_home * h.insurance_rate
            + average_home * h.maintenance_rate
            + annual_hoa
            + refinance_cost
        )
        if project_completes:
            owner_cost += sweat.cash_cost * purchase_price / h.purchase_price
        renter_cost = monthly_rent * 12.0
        buyer_savings = max(renter_cost - owner_cost, 0.0)
        renter_savings = max(owner_cost - renter_cost, 0.0)
        buyer_portfolio = buyer_portfolio * (1.0 + stock_return) + buyer_savings
        renter_portfolio = renter_portfolio * (1.0 + stock_return) + renter_savings
        home_value *= 1.0 + home_return
        monthly_rent *= 1.0 + rent_return
        annual_hoa *= 1.0 + config.market.general_inflation

    buyer_net_worth = home_value * (1.0 - h.sale_cost_rate) - mortgage_balance[0] + buyer_portfolio
    difference = buyer_net_worth - renter_portfolio
    return {
        "start_year": start_year,
        "end_year": end_year,
        "horizon_years": horizon,
        "starting_purchase_price": purchase_price,
        "starting_monthly_rent": monthly_rent / np.prod(1.0 + history["rent_return"].to_numpy()),
        "starting_mortgage_rate": float(history.iloc[0]["mortgage_start_rate"]),
        "realized_buyer_net_worth": buyer_net_worth,
        "realized_renter_net_worth": renter_portfolio,
        "realized_difference": difference,
        "realized_difference_pct": difference / purchase_price,
        "realized_buy_win": bool(difference > 0.0),
        "historical_refinances": refinance_count,
    }


def _forecast_cohort(
    config: SimulationConfig,
    dataset: HistoricalDataset,
    start_year: int,
    horizon: int,
    runs: int,
) -> dict[str, float]:
    forecast = deepcopy(config)
    purchase_price, down_payment, monthly_rent = _cohort_start_values(config, dataset, start_year)
    forecast.housing.purchase_price = purchase_price
    forecast.housing.down_payment = down_payment
    forecast.housing.monthly_rent = monthly_rent
    forecast.housing.annual_hoa *= purchase_price / config.housing.purchase_price
    forecast.housing.property_tax_rate = float(dataset.annual.loc[start_year, "property_tax_rate"])
    forecast.mortgage.initial_rate = float(dataset.annual.loc[start_year, "mortgage_start_rate"])
    forecast.mortgage.rate_cap = max(forecast.mortgage.rate_cap, forecast.mortgage.initial_rate)
    forecast.mortgage.minimum_refinance_balance *= purchase_price / config.housing.purchase_price
    forecast.mortgage.refinance_fixed_cost *= purchase_price / config.housing.purchase_price
    forecast.runs = runs
    forecast.years = horizon
    forecast.horizons = [horizon]
    if forecast.sweat_equity.completion_year > horizon:
        forecast.sweat_equity.enabled = False
    forecast.seed = config.seed + start_year * 101 + horizon * 10_007
    forecast.validate()
    summary = run_simulation(forecast).summary.iloc[0]
    scale = purchase_price
    return {
        "forecast_buy_probability": float(summary["buy_win_probability"]),
        "forecast_median_difference": float(summary["median_net_worth_difference"]),
        "forecast_p05_difference": float(summary["p05_net_worth_difference"]),
        "forecast_p95_difference": float(summary["p95_net_worth_difference"]),
        "forecast_median_pct": float(summary["median_net_worth_difference"] / scale),
        "forecast_p05_pct": float(summary["p05_net_worth_difference"] / scale),
        "forecast_p95_pct": float(summary["p95_net_worth_difference"] / scale),
    }


def historical_moments(
    config: SimulationConfig,
    dataset: HistoricalDataset,
    start_year: int = 1991,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = dataset.annual.loc[start_year : dataset.latest_data_year].dropna(
        subset=["stock_return", "home_return", "rent_return", "mortgage_rate_change"]
    )
    transition = np.asarray(config.regimes.transition, dtype=float)
    system = transition.T - np.eye(len(transition))
    system[-1] = 1.0
    target = np.zeros(len(transition))
    target[-1] = 1.0
    stationary = np.linalg.solve(system, target)
    volatility_multipliers = np.asarray(config.regimes.volatility_multipliers)
    definitions = [
        (
            "Stocks", "stock_return", config.market.stock_return,
            config.market.stock_volatility, np.asarray(config.regimes.stock_return_shifts),
        ),
        (
            "Boston homes", "home_return", config.market.home_appreciation,
            config.market.home_volatility, np.asarray(config.regimes.home_return_shifts),
        ),
        (
            "Boston rents", "rent_return", config.market.rent_growth,
            config.market.rent_volatility, np.asarray(config.regimes.rent_growth_shifts),
        ),
    ]
    rows = []
    for label, column, base_mean, base_volatility, shifts in definitions:
        conditional_mean = np.maximum(base_mean + shifts, -0.95)
        conditional_volatility = base_volatility * volatility_multipliers
        conditional_variance = (
            np.exp(conditional_volatility**2) - 1.0
        ) * (1.0 + conditional_mean) ** 2
        model_mean = float(np.sum(stationary * conditional_mean))
        model_second_moment = float(
            np.sum(stationary * (conditional_variance + conditional_mean**2))
        )
        model_volatility = float(np.sqrt(max(model_second_moment - model_mean**2, 0.0)))
        rows.append(
            {
                "series": label,
                "observed_mean": float(sample[column].mean()),
                "model_mean": model_mean,
                "observed_volatility": float(sample[column].std(ddof=1)),
                "model_volatility": model_volatility,
                "observations": len(sample),
            }
        )
    moments = pd.DataFrame(rows)

    columns = ["stock_return", "home_return", "rent_return", "mortgage_rate_change"]
    labels = ["Stocks", "Boston homes", "Boston rents", "Mortgage rates"]
    observed = sample[columns].corr().to_numpy()
    assumed = np.asarray(config.market.correlation, dtype=float)
    correlation_rows = []
    for i, first in enumerate(labels):
        for j in range(i + 1, len(labels)):
            correlation_rows.append(
                {
                    "pair": f"{first} / {labels[j]}",
                    "observed_correlation": float(observed[i, j]),
                    "model_correlation": float(assumed[i, j]),
                }
            )
    return moments, pd.DataFrame(correlation_rows)


def run_historical_backtest(
    config: SimulationConfig | None = None,
    *,
    data_dir: str | Path | None = None,
    forecast_runs: int = 5_000,
    horizons: tuple[int, ...] = (5, 10, 20),
    first_start_year: int = 1991,
) -> HistoricalBacktestResult:
    """Forecast and replay every complete Boston cohort at the requested horizons."""
    config = deepcopy(config or SimulationConfig())
    config.validate()
    if forecast_runs < 100:
        raise ValueError("forecast_runs must be at least 100")
    dataset = load_historical_dataset(data_dir)
    rows: list[dict[str, float | int | bool]] = []
    for horizon in sorted(set(horizons)):
        final_start = dataset.latest_data_year - horizon + 1
        for start_year in range(first_start_year, final_start + 1):
            realized = replay_historical_cohort(config, dataset, start_year, horizon)
            forecast = _forecast_cohort(
                config, dataset, start_year, horizon, min(forecast_runs, config.runs)
            )
            row = {**realized, **forecast}
            row["inside_forecast_90"] = bool(
                row["forecast_p05_difference"]
                <= row["realized_difference"]
                <= row["forecast_p95_difference"]
            )
            row["classification_correct"] = bool(
                (row["forecast_buy_probability"] >= 0.5) == row["realized_buy_win"]
            )
            row["probability_squared_error"] = (
                row["forecast_buy_probability"] - float(row["realized_buy_win"])
            ) ** 2
            row["median_error_pct"] = row["forecast_median_pct"] - row["realized_difference_pct"]
            rows.append(row)
    cohorts = pd.DataFrame(rows)
    if cohorts.empty:
        raise ValueError("No complete historical cohorts for the requested horizons")

    calibration_rows = []
    for horizon, group in cohorts.groupby("horizon_years", sort=True):
        calibration_rows.append(
            {
                "horizon_years": int(horizon),
                "cohorts": len(group),
                "actual_buy_win_rate": float(group["realized_buy_win"].mean()),
                "mean_forecast_buy_probability": float(group["forecast_buy_probability"].mean()),
                "brier_score": float(group["probability_squared_error"].mean()),
                "classification_accuracy": float(group["classification_correct"].mean()),
                "forecast_90_interval_coverage": float(group["inside_forecast_90"].mean()),
                "median_bias_pct_of_purchase": float(group["median_error_pct"].median()),
                "median_absolute_error_pct_of_purchase": float(group["median_error_pct"].abs().median()),
            }
        )
    calibration = pd.DataFrame(calibration_rows)
    moments, correlations = historical_moments(config, dataset, first_start_year)
    return HistoricalBacktestResult(
        config=config,
        cohorts=cohorts,
        calibration=calibration,
        moments=moments,
        correlations=correlations,
        data_start_year=first_start_year,
        data_end_year=dataset.latest_data_year,
        forecast_runs=min(forecast_runs, config.runs),
    )
