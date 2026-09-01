"""Command-line interface."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import SimulationConfig
from .reporting import format_summary, save_outputs
from .sensitivity import run_sensitivity
from .simulation import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo Boston buy-versus-rent analysis")
    parser.add_argument("--config", type=Path, help="JSON assumptions file")
    parser.add_argument("--runs", type=int, help="override number of paths")
    parser.add_argument("--seed", type=int, help="override random seed")
    parser.add_argument("--years", type=int, help="override simulation length")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sensitivity", action="store_true", help="run one-way sensitivity analysis")
    parser.add_argument("--sensitivity-runs", type=int, default=20_000)
    parser.add_argument("--sensitivity-horizon", type=int, default=10)
    parser.add_argument(
        "--write-default-config",
        type=Path,
        metavar="PATH",
        help="write default JSON and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_default_config:
        SimulationConfig().to_json(args.write_default_config)
        print(f"Wrote {args.write_default_config}")
        return 0

    config = SimulationConfig.from_json(args.config) if args.config else SimulationConfig()
    if args.runs is not None:
        config.runs = args.runs
    if args.seed is not None:
        config.seed = args.seed
    if args.years is not None:
        config.years = args.years
        config.horizons = [h for h in config.horizons if h <= args.years]
        if not config.horizons and args.years > 0:
            config.horizons = [args.years]
    config.validate()

    started = time.perf_counter()
    result = run_simulation(config)
    sensitivity = None
    if args.sensitivity:
        sensitivity = run_sensitivity(
            config,
            horizon=args.sensitivity_horizon,
            runs=args.sensitivity_runs,
            base_result=result if config.runs <= args.sensitivity_runs else None,
        )
    save_outputs(
        result,
        args.output_dir,
        sensitivity,
        sensitivity_horizon=args.sensitivity_horizon,
    )
    print(format_summary(result.summary))
    print(f"\nCompleted {config.runs:,} paths in {time.perf_counter() - started:.2f}s")
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
