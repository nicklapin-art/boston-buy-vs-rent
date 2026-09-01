# Boston Buy vs. Rent Monte Carlo

A configurable, vectorized Python model for comparing a Boston-area home purchase with renting and investing the difference. The baseline is a **$1.2M purchase, $250k down, 6.66% 30-year mortgage, and $4,500 monthly rent** over 100,000 simulated paths.

This is a decision-support model, not financial, tax, lending, or real-estate advice. Its outputs are only as reliable as its assumptions.

## Browser interface

On Windows, double-click **`Launch Buy vs Rent.cmd`**. The first launch prepares a private project runtime; later launches start immediately.

From a terminal, install the project and launch the local GUI:

```powershell
python -m buy_vs_rent.web_server
```

It opens `http://127.0.0.1:8000` automatically. Adjust the home, rent, ownership-cost, market, volatility, refinancing, run-count, and horizon inputs, then select **Run simulation**. The browser shows win probabilities, median outcomes, 5th–95th percentile ranges, and exact values. The calculations run locally through the same Python engine; no scenario data is uploaded.

The interface also runs a historical validation panel. It replays complete Boston purchase cohorts with observed market data and shows forecast-versus-realized win rates, interval coverage, cohort outcomes, and historical return/volatility comparisons. Select **Validate this scenario** after changing assumptions.

To choose another port or avoid opening a browser automatically:

```powershell
python -m buy_vs_rent.web_server --port 8080
python -m buy_vs_rent.web_server --no-browser
```

## Historical validation

Run the reproducible 5,000-path-per-cohort backtest from the terminal:

```powershell
python -m buy_vs_rent.historical_cli --forecast-runs 5000
```

The baseline backtest contains 67 overlapping cohorts beginning in 1991: 29 five-year, 24 ten-year, and 14 twenty-year windows. It uses FHFA Boston home prices, BLS Boston rents, Freddie Mac mortgage rates, Yale/Shiller stock returns, and City of Boston residential tax rates.

Baseline calibration results:

| Horizon | Actual buy-win rate | Mean forecast probability | 90% interval coverage |
|---:|---:|---:|---:|
| 5 years | 37.9% | 37.5% | 65.5% |
| 10 years | 45.8% | 52.5% | 75.0% |
| 20 years | 64.3% | 80.9% | 100.0% |

The five-year average probability is well aligned, but the five- and ten-year intervals are too narrow. The model overstates the historical twenty-year buy-win frequency by 16.6 percentage points, while producing a very wide twenty-year interval. The validation therefore supports treating the simulation as a scenario distribution—not a precise probability forecast.

These are retrospective tests of today's assumptions against revised historical indexes, not forecasts that were genuinely issued at each historical date. Cohorts overlap, local indexes do not represent a specific property, CPI rent can lag asking rent, and complete historical insurance/maintenance series are unavailable. See `data/historical/SOURCES.md` and `results/historical_validation/VALIDATION_REPORT.md` for full sourcing and limitations.

## What the model includes

- Correlated annual shocks to stocks, home prices, rents, and mortgage rates.
- Markov regimes for normal conditions, recessions, and rare 2008-style crashes. Crash years lower conditional stock, home-price, rent-growth, and rate assumptions while increasing volatility; they are scenarios, not literal replays or forecasts.
- Path-by-path 30-year mortgage amortization.
- Automatic refinancing when the available rate is at least 1 percentage point below the current loan rate, including percentage and fixed closing costs. Refinancing resets the term to 30 years by default.
- Property tax, homeowners insurance, maintenance, HOA, purchase closing costs, and sale costs.
- A fair cash-flow comparison: both strategies receive the same annual housing budget, and the cheaper strategy invests that year's difference in the same stock portfolio.
- Buyer net worth at each horizon is after a hypothetical sale: home value less sale costs and mortgage balance, plus invested savings. Renter net worth is the invested upfront cash plus annual savings.
- Summary statistics at years 5, 10, and 20; annual diagnostics; distribution plots; and one-way sensitivity analysis.

## Baseline sources and judgment calls

- The 6.66% initial rate is the [Freddie Mac 30-year fixed average for August 27, 2026](https://www.freddiemac.com/pmms). An individual jumbo-loan quote can differ materially.
- Boston's FY2026 residential tax rate is [$12.40 per $1,000 of assessed value](https://www.boston.gov/departments/assessing/how-we-tax-your-property), modeled as 1.24% without a residential exemption. Change this for the exact municipality and exemption status.
- The 2.5% buyer closing-cost default is within the CFPB's typical [2% to 5% range](https://www.consumerfinance.gov/owning-a-home/prepare/determine-your-down-payment/).
- Market returns, volatility, correlations, insurance, maintenance, selling costs, and regime probabilities are transparent modeling assumptions in `config/baseline_boston.json`, not sourced forecasts.

The baseline is nominal and deliberately excludes mortgage-interest/property-tax deductions, capital-gains taxes, investment taxes, the home-sale exclusion, residential exemptions, rent deposits, utilities, and lifestyle differences. That makes no claim that those items are zero; it avoids silently assuming a tax situation. Add them before using the result for a personal decision.

## Quick start

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m buy_vs_rent --config config/baseline_boston.json --sensitivity
```

Or with `uv`:

```powershell
uv sync --extra dev
uv run python -m buy_vs_rent --config config/baseline_boston.json --sensitivity
```

The default command runs 100,000 paths and writes to `results/`:

- `summary.csv` — probability buying wins and the median, 5th, and 95th percentiles of the buyer's net-worth advantage.
- `annual_diagnostics.csv` — realized mean returns, rates, balances, regime shares, and refinance shares.
- `config_used.json` — exact reproducibility record.
- `net_worth_distributions.png` — horizon distributions.
- `sensitivity.csv` and `sensitivity_tornado.png` — generated with `--sensitivity`.

Useful overrides:

```powershell
python -m buy_vs_rent --runs 10000 --seed 7 --output-dir results/quick
python -m buy_vs_rent --config config/baseline_boston.json --runs 100000
python -m buy_vs_rent --write-default-config config/my_scenario.json
```

Edit the JSON to change any assumption. Decimal rates use `0.0666` for 6.66%. The correlation matrix order is stocks, home prices, rents, and mortgage rates.

## Python API

```python
from buy_vs_rent import SimulationConfig, run_simulation
from buy_vs_rent.sensitivity import run_sensitivity

config = SimulationConfig.from_json("config/baseline_boston.json")
config.housing.monthly_rent = 5_000
config.mortgage.initial_rate = 0.0625

result = run_simulation(config)
print(result.summary)
sensitivity = run_sensitivity(config, horizon=10, runs=20_000)
```

## Accounting sequence

Each simulated year:

1. Transition the economic regime and draw correlated shocks.
2. Update the available mortgage rate and refinance qualifying paths.
3. Amortize twelve mortgage payments.
4. Calculate owner and renter housing costs.
5. Apply the same stock return to both portfolios, then invest the cheaper strategy's annual savings.
6. Update the home value and next year's rent.
7. At requested horizons, calculate after-sale buyer net worth and renter portfolio value.

This annual frequency is fast enough for 100,000 paths while preserving exact monthly fixed-rate amortization inside each year.

## Tests

```powershell
pytest
```

Tests cover mortgage payment/amortization math, deterministic zero-volatility paths, reproducibility, refinancing, validation, and JSON round-tripping.
