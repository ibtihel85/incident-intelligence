"""
Autonomous Incident Intelligence System
Main Orchestrator — entry point for the full pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

from pipeline.orchestrator import IncidentOrchestrator
from utils.logger import get_logger
from utils.config import load_config

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Incident Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/synthetic/cpu_spike.json
  python main.py --input data/synthetic/memory_leak.json --output reports/
  python main.py --eval          # run full evaluation benchmark
        """,
    )
    parser.add_argument(
        "--input", type=str, help="Path to incident data file (JSON)"
    )
    parser.add_argument(
        "--output", type=str, default="reports/", help="Directory to write report"
    )
    parser.add_argument(
        "--config", type=str, default="config/settings.yaml", help="Config file path"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation benchmark on all scenarios"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser.parse_args()


def run_single(args: argparse.Namespace) -> None:
    if not args.input:
        logger.error("--input is required when not running --eval")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    config = load_config(args.config)
    orchestrator = IncidentOrchestrator(config)

    logger.info(f"Loading incident data from: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    result = orchestrator.run(raw_data)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"incident_report_{result['incident_id']}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result["report"])

    logger.info(f"Report written to: {report_path}")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n" + "=" * 70)
    print(result["report"])
    print("=" * 70)


def run_eval(args: argparse.Namespace) -> None:
    from evaluation.benchmark import IncidentBenchmark

    config = load_config(args.config)
    benchmark = IncidentBenchmark(config)
    benchmark.run()


def main() -> None:
    args = parse_args()

    if args.eval:
        run_eval(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
