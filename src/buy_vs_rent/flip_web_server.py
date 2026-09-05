"""Dependency-free local web server for the independent flip simulator."""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .flip_config import FlipConfig
from .flip_simulation import run_flip_simulation


WEB_ROOT = Path(__file__).with_name("flip_web")
MAX_REQUEST_BYTES = 100_000


class SingleInstanceHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _section(payload: dict[str, Any], name: str, allowed: set[str]) -> dict[str, float]:
    raw = payload.get(name, {})
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} setting: {sorted(unknown)[0]}")
    return {key: _finite(value, key) for key, value in raw.items()}


def config_from_payload(payload: dict[str, Any]) -> FlipConfig:
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    allowed_top = {"runs", "seed", "acquisition", "renovation", "financing", "exit", "market"}
    unknown = set(payload) - allowed_top
    if unknown:
        raise ValueError(f"Unknown setting: {sorted(unknown)[0]}")

    defaults = FlipConfig()
    config = FlipConfig(
        acquisition=type(defaults.acquisition)(
            **_section(payload, "acquisition", set(defaults.acquisition.__dataclass_fields__))
        ),
        renovation=type(defaults.renovation)(
            **_section(payload, "renovation", set(defaults.renovation.__dataclass_fields__))
        ),
        financing=type(defaults.financing)(
            **_section(payload, "financing", set(defaults.financing.__dataclass_fields__))
        ),
        exit=type(defaults.exit)(
            **_section(payload, "exit", set(defaults.exit.__dataclass_fields__))
        ),
        market=defaults.market,
        runs=int(_finite(payload.get("runs", defaults.runs), "runs")),
        seed=int(_finite(payload.get("seed", defaults.seed), "seed")),
    )
    market_values = _section(
        payload,
        "market",
        {"annual_home_appreciation", "annual_home_volatility"},
    )
    for key, value in market_values.items():
        setattr(config.market, key, value)
    config.exit.max_holding_months = int(config.exit.max_holding_months)
    config.validate()
    return config


def defaults_payload() -> dict[str, Any]:
    return FlipConfig().to_dict()


def simulation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = config_from_payload(payload)
    started = time.perf_counter()
    result = run_flip_simulation(config)
    return {
        "runs": config.runs,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "summary": result.summary,
        "percentiles": result.percentiles.to_dict(orient="records"),
        "regimes": result.regimes.to_dict(orient="records"),
        "histogram": result.histogram.to_dict(orient="records"),
        "cost_breakdown": result.cost_breakdown.to_dict(orient="records"),
        "hurdle_rate": config.exit.annual_hurdle_rate,
        "tax_rate": config.exit.estimated_tax_rate,
    }


class FlipSimulationHandler(BaseHTTPRequestHandler):
    server_version = "FlipSimulatorGUI/0.1"

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
        if self.path != "/api/simulate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_REQUEST_BYTES:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(size))
            self._json(simulation_payload(payload))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            self._json(
                {"error": "The flip simulation could not be completed."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: object) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the real-estate flip simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = SingleInstanceHTTPServer((args.host, args.port), FlipSimulationHandler)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Real-estate flip simulator: {url}")
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
