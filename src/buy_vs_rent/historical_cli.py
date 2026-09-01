"""Command-line entry point for historical validation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import SimulationConfig
from .historical import run_historical_backtest
from .historical_reporting import save_historical_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest the buy-versus-rent model")
    parser.add_argument("--config", type=Path, help="JSON assumptions file")
    parser.add_argument("--forecast-runs", type=int, default=5_000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/historical_validation"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimulationConfig.from_json(args.config) if args.config else SimulationConfig()
    started = time.perf_counter()
    result = run_historical_backtest(config, forecast_runs=args.forecast_runs)
    save_historical_outputs(result, args.output_dir)
    display = result.calibration.copy()
    for column in (
        "actual_buy_win_rate", "mean_forecast_buy_probability",
        "classification_accuracy", "forecast_90_interval_coverage",
    ):
        display[column] = display[column].map(lambda value: f"{value:.1%}")
    print(display.to_string(index=False))
    print(f"\nCompleted in {time.perf_counter() - started:.2f}s")
    print(f"Report: {(args.output_dir / 'VALIDATION_REPORT.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

