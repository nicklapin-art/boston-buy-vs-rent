"""Command-line entrypoint for parameter-uncertainty analysis."""

from __future__ import annotations

import argparse

from .config import SimulationConfig
from .historical_calibration import run_historically_calibrated_uncertainty
from .uncertainty import run_parameter_uncertainty
from .uncertainty_reporting import save_parameter_uncertainty


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stress-test buy/rent assumptions")
    parser.add_argument("--config", help="Scenario JSON; defaults to built-in Boston assumptions")
    parser.add_argument("--parameter-sets", type=int, default=64)
    parser.add_argument("--runs-per-set", type=int, default=5_000)
    parser.add_argument(
        "--method", choices=("historical", "judgment"), default="historical",
        help="Calibrate uncertainty from history or use predefined judgment bands",
    )
    parser.add_argument("--output-dir", default="results/parameter_uncertainty")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimulationConfig.from_json(args.config) if args.config else SimulationConfig()
    runner = (
        run_historically_calibrated_uncertainty
        if args.method == "historical"
        else run_parameter_uncertainty
    )
    result = runner(
        config, parameter_sets=args.parameter_sets, runs_per_set=args.runs_per_set
    )
    report = save_parameter_uncertainty(result, args.output_dir)
    print(result.summary.to_string(index=False))
    print(f"\nSaved report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
