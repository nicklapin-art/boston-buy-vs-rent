"""Small dependency-free local web server for the browser GUI."""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import SimulationConfig
from .historical import run_historical_backtest
from .historical_calibration import run_historically_calibrated_uncertainty
from .simulation import run_simulation
from .sweat_equity import run_sweat_equity_analysis
from .uncertainty import run_parameter_uncertainty


WEB_ROOT = Path(__file__).with_name("web")
MAX_REQUEST_BYTES = 100_000
MAX_RUNS = 100_000
MAX_YEARS = 40


class SingleInstanceHTTPServer(ThreadingHTTPServer):
    """Refuse to share a port with a stale copy of the GUI server."""

    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _set_numeric_fields(target: Any, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown setting: {sorted(unknown)[0]}")
    for name, value in data.items():
        setattr(target, name, _finite_number(value, name))


def config_from_payload(payload: dict[str, Any]) -> SimulationConfig:
    """Create a validated config from the GUI's deliberately limited schema."""
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    allowed_top = {
        "runs", "years", "seed", "housing", "mortgage", "market", "sweat_equity"
    }
    unknown = set(payload) - allowed_top
    if unknown:
        raise ValueError(f"Unknown setting: {sorted(unknown)[0]}")

    config = SimulationConfig()
    config.runs = int(_finite_number(payload.get("runs", config.runs), "runs"))
    config.years = int(_finite_number(payload.get("years", config.years), "years"))
    config.seed = int(_finite_number(payload.get("seed", config.seed), "seed"))
    if not 100 <= config.runs <= MAX_RUNS:
        raise ValueError(f"runs must be between 100 and {MAX_RUNS:,}")
    if not 1 <= config.years <= MAX_YEARS:
        raise ValueError(f"years must be between 1 and {MAX_YEARS}")
    config.horizons = [year for year in (5, 10, 20, 30, 40) if year <= config.years]
    if config.years not in config.horizons:
        config.horizons.append(config.years)
    config.horizons.sort()

    housing = dict(payload.get("housing", {}))
    mortgage = dict(payload.get("mortgage", {}))
    market = dict(payload.get("market", {}))
    sweat_equity = dict(payload.get("sweat_equity", {}))
    if not all(
        isinstance(item, dict) for item in (housing, mortgage, market, sweat_equity)
    ):
        raise ValueError("housing, mortgage, market, and sweat_equity settings must be objects")

    _set_numeric_fields(
        config.housing,
        housing,
        {
            "purchase_price", "down_payment", "monthly_rent", "property_tax_rate",
            "insurance_rate", "maintenance_rate", "annual_hoa",
            "purchase_closing_cost_rate", "sale_cost_rate",
        },
    )
    refinance_enabled = mortgage.pop("refinance_enabled", None)
    _set_numeric_fields(
        config.mortgage,
        mortgage,
        {
            "initial_rate", "long_run_rate", "refinance_threshold",
            "refinance_cost_rate", "refinance_fixed_cost",
        },
    )
    if refinance_enabled is not None:
        if not isinstance(refinance_enabled, bool):
            raise ValueError("refinance_enabled must be true or false")
        config.mortgage.refinance_enabled = refinance_enabled
    _set_numeric_fields(
        config.market,
        market,
        {
            "stock_return", "stock_volatility", "home_appreciation",
            "home_volatility", "rent_growth", "rent_volatility", "general_inflation",
        },
    )
    sweat_enabled = sweat_equity.pop("enabled", None)
    completion_year = sweat_equity.pop("completion_year", None)
    _set_numeric_fields(
        config.sweat_equity,
        sweat_equity,
        {
            "cash_cost", "labor_hours", "hourly_time_value",
            "value_added_low", "value_added_expected", "value_added_high",
        },
    )
    if sweat_enabled is not None:
        if not isinstance(sweat_enabled, bool):
            raise ValueError("sweat-equity enabled must be true or false")
        config.sweat_equity.enabled = sweat_enabled
    if completion_year is not None:
        config.sweat_equity.completion_year = int(
            _finite_number(completion_year, "completion_year")
        )
    config.validate()
    return config


def defaults_payload() -> dict[str, Any]:
    config = SimulationConfig()
    return {
        "runs": config.runs,
        "years": config.years,
        "seed": config.seed,
        "housing": asdict(config.housing),
        "mortgage": asdict(config.mortgage),
        "market": {
            key: value
            for key, value in asdict(config.market).items()
            if key != "correlation"
        },
        "sweat_equity": asdict(config.sweat_equity),
    }


def run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = config_from_payload(payload)
    started = time.perf_counter()
    result = run_simulation(config)
    elapsed = time.perf_counter() - started
    return {
        "runs": config.runs,
        "years": config.years,
        "elapsed_seconds": round(elapsed, 3),
        "summary": result.summary.to_dict(orient="records"),
        "sweat_equity": asdict(config.sweat_equity),
    }


def sweat_equity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Calculate required remodeling uplift and compare the optional estimate."""

    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    scenario = payload.get("scenario", {})
    runs = int(_finite_number(payload.get("runs", 20_000), "runs"))
    curve_points = int(_finite_number(payload.get("curve_points", 11), "curve_points"))
    config = config_from_payload(scenario)
    horizon = int(_finite_number(payload.get("horizon", config.years), "horizon"))
    started = time.perf_counter()
    result = run_sweat_equity_analysis(
        config,
        horizon=horizon,
        runs=runs,
        curve_points=curve_points,
    )
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "runs": result.runs,
        "horizon_years": result.horizon,
        "financial_required_uplift": result.financial_required_uplift,
        "economic_required_uplift": result.economic_required_uplift,
        "tested_max_value": result.tested_max_value,
        "summary": result.summary.to_dict(orient="records")[0],
        "curve": result.curve.to_dict(orient="records"),
    }


def historical_validation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    scenario = payload.get("scenario", {})
    forecast_runs = int(_finite_number(payload.get("forecast_runs", 2_000), "forecast_runs"))
    if not 100 <= forecast_runs <= 10_000:
        raise ValueError("forecast_runs must be between 100 and 10,000")
    config = config_from_payload(scenario)
    started = time.perf_counter()
    result = run_historical_backtest(config, forecast_runs=forecast_runs)

    def records(frame: Any) -> list[dict[str, Any]]:
        return json.loads(frame.to_json(orient="records"))

    cohort_columns = [
        "start_year", "end_year", "horizon_years", "realized_buy_win",
        "realized_difference_pct", "forecast_buy_probability",
        "forecast_median_pct", "inside_forecast_90",
    ]
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "forecast_runs_per_cohort": result.forecast_runs,
        "data_start_year": result.data_start_year,
        "data_end_year": result.data_end_year,
        "calibration": records(result.calibration),
        "moments": records(result.moments),
        "correlations": records(result.correlations),
        "cohorts": records(result.cohorts[cohort_columns]),
    }


def robustness_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Run judgment-band or historically calibrated second-level uncertainty."""

    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    scenario = payload.get("scenario", {})
    method = str(payload.get("method", "historical"))
    if method not in {"historical", "judgment"}:
        raise ValueError("method must be historical or judgment")
    parameter_sets = int(_finite_number(payload.get("parameter_sets", 64), "parameter_sets"))
    runs_per_set = int(_finite_number(payload.get("runs_per_set", 2_000), "runs_per_set"))
    if not 8 <= parameter_sets <= 128:
        raise ValueError("parameter_sets must be between 8 and 128")
    if not 250 <= runs_per_set <= 10_000:
        raise ValueError("runs_per_set must be between 250 and 10,000")
    if parameter_sets * runs_per_set > 1_000_000:
        raise ValueError("parameter_sets times runs_per_set cannot exceed 1,000,000")
    config = config_from_payload(scenario)
    started = time.perf_counter()
    if method == "historical":
        result = run_historically_calibrated_uncertainty(
            config,
            parameter_sets=parameter_sets,
            runs_per_set=runs_per_set,
        )
    else:
        result = run_parameter_uncertainty(
            config,
            parameter_sets=parameter_sets,
            runs_per_set=runs_per_set,
            method="judgment",
            metadata={
                "method_label": "User-centered judgment bands",
                "initial_mortgage_rate_fixed": True,
                "historical_parameters": [],
                "judgment_parameters": [
                    "stock_return", "home_appreciation", "rent_growth",
                    "maintenance_rate", "property_tax_rate", "insurance_rate",
                    "sale_cost_rate",
                ],
            },
        )

    def records(frame: Any) -> list[dict[str, Any]]:
        return json.loads(frame.to_json(orient="records"))

    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "parameter_sets": result.parameter_sets,
        "runs_per_set": result.runs_per_set,
        "total_paths": result.parameter_sets * result.runs_per_set,
        "method": result.method,
        "calibration": result.metadata,
        "ranges": [asdict(item) for item in result.ranges],
        "summary": records(result.summary),
        "influence": records(result.influence),
    }


class SimulationHandler(BaseHTTPRequestHandler):
    server_version = "BuyVsRentGUI/0.5"

    def _json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/defaults":
            self._json(defaults_payload())
            return
        if self.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = (WEB_ROOT / "index.html").read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {
            "/api/simulate", "/api/backtest", "/api/robustness", "/api/sweat-equity"
        }:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_REQUEST_BYTES:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(size))
            if self.path == "/api/backtest":
                self._json(historical_validation_payload(payload))
            elif self.path == "/api/robustness":
                self._json(robustness_payload(payload))
            elif self.path == "/api/sweat-equity":
                self._json(sweat_equity_payload(payload))
            else:
                self._json(run_payload(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            self._json(
                {"error": "The simulation could not be completed."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: object) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the buy-versus-rent browser GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = SingleInstanceHTTPServer((args.host, args.port), SimulationHandler)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Buy vs. Rent GUI: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
