# Boston Buy vs. Rent Monte Carlo

A configurable, vectorized Python model for comparing a Boston-area home purchase with renting and investing the difference. The baseline is a **$1.2M purchase, $250k down, 6.66% 30-year mortgage, and $4,500 monthly rent** over 100,000 simulated paths.

This is a decision-support model, not financial, tax, lending, or real-estate advice. Its outputs are only as reliable as its assumptions.

## Browser interface

On Windows, double-click **`Launch Buy vs Rent.cmd`**. The first launch prepares a private project runtime; later launches start immediately.

From a terminal, install the project and launch the local GUI:

```powershell
python -m buy_vs_rent.web_server
```

It opens `http://127.0.0.1:8000` automatically. Adjust the home, rent, ownership-cost, market, volatility, refinancing, run-count, and horizon inputs, then select **Run simulation**. The browser shows win probabilities, median outcomes, 5th–95th percentile ranges, and exact values. Select **Stress-test assumptions** to vary the expected returns and ownership-cost assumptions themselves. Choose either historically calibrated uncertainty or the original predefined judgment bands. The optional sweat-equity panel models a discrete DIY project and calculates the value uplift needed for the entire buy-vs-rent decision to reach a 50% buying probability. The calculations run locally through the same Python engine; no scenario data is uploaded.

The interface also runs a historical validation panel. It replays complete Boston purchase cohorts with observed market data and shows forecast-versus-realized win rates, interval coverage, cohort outcomes, and historical return/volatility comparisons. Select **Validate this scenario** after changing assumptions.

## Assumption robustness

Ordinary Monte Carlo paths vary future returns around one fixed set of expected returns and cost rates. The robustness analysis adds a second level: it samples 64 plausible parameter sets, then runs 2,000 correlated economic paths inside each set by default. The starting mortgage rate, purchase price, down payment, and starting rent always remain the values entered by the user.

The default **historically calibrated** method uses a five-year moving-block bootstrap. Stocks, Boston home appreciation, and Boston rent growth are resampled jointly so their long-run assumption errors retain historically observed dependence. The bootstrap deviations are centered on the user's entered long-run assumptions: history determines the width, skew, and joint movement, while the user's forecast remains the baseline. Boston property-tax uncertainty is bootstrapped from its available history. The interface reports the exact data period, calibrated percentiles, and walk-forward checks that use only data available before each historical forecast start.

Consistent local histories are unavailable for maintenance, homeowners insurance, and selling costs, so those three inputs remain explicitly labeled judgment bands. Historical calibration does not vary the initial mortgage rate. It also does not turn historical averages into a forecast or eliminate structural uncertainty; overlapping long-horizon cohorts are a small, dependent sample.

The alternative **predefined judgment bands** reproduces the original analysis. Its triangular ranges use the configured scenario as the most likely value:

The default triangular ranges use the configured scenario as the most likely value:

| Assumption | Default half-width around scenario |
|---|---:|
| Expected stock return | ±2.50 percentage points |
| Expected home appreciation | ±2.00 points |
| Expected rent growth | ±1.25 points |
| Annual maintenance | ±0.40 points |
| Property-tax rate | ±0.25 points |
| Homeowners insurance | ±0.15 points |
| Selling costs | ±1.50 points |

The output distinguishes the **integrated buy-win probability**—the average across parameter sets—from the **robust buy share**, the fraction of parameter sets in which buying wins more than half of economic paths. It also reports the 10th–90th percentile range of set-level probabilities and partial rank correlations that control for the other sampled assumptions.

For the baseline, the historically calibrated 64 × 2,000 analysis produces a 20-year integrated buy-win probability of 37.4%, a 7.3%–64.5% middle-80% assumption range, and a 32.8% robust buy share. The original judgment-band method produces 48.1%, 31.6%–62.3%, and 51.6%, respectively. The difference is material and is why the interface shows the method and its range provenance beside every result.

Run the saved-report workflow from a terminal:

```powershell
python -m buy_vs_rent.uncertainty_cli --config config/baseline_boston.json --method historical --parameter-sets 64 --runs-per-set 5000
python -m buy_vs_rent.uncertainty_cli --config config/baseline_boston.json --method judgment --parameter-sets 64 --runs-per-set 5000
```

It writes `robustness_summary.csv`, every parameter-set outcome, parameter influence, exact ranges, calibration metadata, two charts, and `ROBUSTNESS_REPORT.md` under `results/parameter_uncertainty/` by default. The Python API accepts custom `ParameterRange` objects for fully user-defined distributions.

## Sweat equity

Sweat equity is modeled as a discrete project completed in a selected ownership year. The user supplies cash materials and permit costs, labor hours, a personal hourly time value, and low/expected/high estimates of the immediate post-project market-value increase. The value increase is drawn once per path from a triangular distribution, recognized at the start of the selected year, and appreciates with the home afterward. Property tax, insurance, maintenance, and eventual selling costs automatically apply to the enlarged home value.

The interface is prefilled with an opt-in Boston kitchen-remodel example: $20,000 of total cash cost, 750 hours of labor valued at $40 per hour, and a $15,000/$30,000/$45,000 low/expected/high immediate value uplift in year 2. The project remains disabled by default so the unchanged baseline is still a pure buy-versus-rent comparison.

The cash cost is charged to the buyer in the completion year; under the model's common-budget accounting, the renter retains and invests the same cash. Personal time is not treated as cash. The interface reports both financial results and an economic result that subtracts `labor hours × hourly time value`.

Select **Analyze sweat equity** to compare the configured project with the identical scenario without a project. The analysis runs common market paths, reports the project's median incremental contribution, and plots buying probability across a deterministic value-uplift curve. It linearly interpolates the immediate value increase required for the entire buy-vs-rent decision to reach a 50% buying probability, both before and after valuing the user's time. This is not the remodel's own break-even point or an appraisal: it is the value that comparable sales or a subject-to-completion appraisal would need to support for buying to win half the simulated paths.

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

These are retrospective tests of today's assumptions against revised historical indexes, not forecasts that were genuinely issued at each historical date. Cohorts overlap, local indexes do not represent a specific property, CPI rent can lag asking rent, and complete historical insurance/maintenance series are unavailable. For sweat-equity scenarios, historical replay uses the expected project uplift rather than drawing from its low/expected/high distribution. See `data/historical/SOURCES.md` and `results/historical_validation/VALIDATION_REPORT.md` for full sourcing and limitations.

## What the model includes

- Correlated annual shocks to stocks, home prices, rents, and mortgage rates.
- Markov regimes for normal conditions, recessions, and rare 2008-style crashes. Crash years lower conditional stock, home-price, rent-growth, and rate assumptions while increasing volatility; they are scenarios, not literal replays or forecasts.
- Path-by-path 30-year mortgage amortization.
- Automatic refinancing when the available rate is at least 1 percentage point below the current loan rate, including percentage and fixed closing costs. Refinancing resets the term to 30 years by default.
- Property tax, homeowners insurance, maintenance, HOA, purchase closing costs, and sale costs.
- Optional sweat equity with completion timing, uncertain immediate value uplift, cash inputs, separately reported time value, and a curve showing the uplift needed for buying to reach a 50% win probability.
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
from buy_vs_rent.historical_calibration import run_historically_calibrated_uncertainty
from buy_vs_rent.uncertainty import run_parameter_uncertainty

config = SimulationConfig.from_json("config/baseline_boston.json")
config.housing.monthly_rent = 5_000
config.mortgage.initial_rate = 0.0625

result = run_simulation(config)
print(result.summary)
sensitivity = run_sensitivity(config, horizon=10, runs=20_000)
robustness = run_historically_calibrated_uncertainty(
    config, parameter_sets=64, runs_per_set=5_000
)
judgment_robustness = run_parameter_uncertainty(
    config, parameter_sets=64, runs_per_set=5_000
)
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
