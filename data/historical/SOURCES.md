# Historical validation data

The raw files are retained so every backtest is reproducible and does not silently change when a provider revises a series.

| File | Series and source | Frequency | Retrieved |
|---|---|---:|---:|
| `raw/boston_fhfa_hpi.csv` | FHFA All-Transactions House Price Index for Boston, MA metropolitan division (`ATNHPIUS14454Q`), retrieved through FRED | Quarterly | 2026-08-31 |
| `raw/boston_bls_rent_cpi.csv` | BLS CPI-U Rent of Primary Residence for Boston-Cambridge-Newton (`CUURA103SEHA`), retrieved through FRED | Monthly/periodic | 2026-08-31 |
| `raw/freddie_mac_mortgage30.csv` | Freddie Mac 30-Year Fixed Rate Mortgage Average (`MORTGAGE30US`), retrieved through FRED | Weekly | 2026-08-31 |
| `raw/shiller_ie_data.xls` | Robert Shiller/Yale U.S. stock-market price and dividend data | Monthly | 2026-08-31 |
| `boston_residential_tax_rates.csv` | City of Boston residential tax-rate history, dollars per $1,000 assessed value | Annual | 2026-08-31 |

Primary documentation:

- FHFA HPI datasets: https://www.fhfa.gov/data/hpi/datasets
- BLS Boston CPI: https://www.bls.gov/regions/northeast/news-release/consumerpriceindex_boston.htm
- Freddie Mac PMMS archive: https://www.freddiemac.com/pmms/pmms_archives
- Yale/Shiller data: http://www.econ.yale.edu/~shiller/data.htm
- Boston tax-rate history: https://www.boston.gov/departments/assessing/how-we-tax-your-property

## Important scope limits

- FHFA's all-transactions index represents the Boston metropolitan division, not a specific property or neighborhood.
- CPI rent measures rent paid by continuing and new tenants; it is not a same-unit asking-rent index. Boston's local series has a smaller sample and is not seasonally adjusted.
- PMMS is a national conforming-loan average. A historical Boston jumbo borrower could have received a different rate.
- Insurance, maintenance, buyer closing costs, and seller costs do not have complete consistent local histories in this project; backtests apply the configured percentage assumptions.
- The residential exemption is excluded, matching the baseline simulation.
