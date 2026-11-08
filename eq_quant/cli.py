"""Command-line interface for data construction, evaluation, and analysis."""

import argparse
import json

from .config import load_config
from .data.build import build_dataset, write_dataset
from .data.schema import SPLIT_ORDER


def _build_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if "all" in args.split:
        split_names = None
    else:
        split_names = args.split
    dataset = build_dataset(config, split_names=split_names)
    write_dataset(dataset, args.output)
    print(json.dumps(dataset.manifest["counts"], indent=2, sort_keys=True))


def _run_evaluation(args: argparse.Namespace) -> None:
    from .evaluate.runner import run_evaluation

    config = load_config(args.config)
    generation_subset = args.generation_subset
    if generation_subset is None:
        generation_subset = int(config.get("generation", {}).get("subset_size", 0))
    manifest = run_evaluation(
        config=config,
        records_path=args.records,
        output_dir=args.output,
        generation_subset=generation_subset,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "records": manifest["records"],
                "conditions": [
                    condition["condition"] for condition in manifest["conditions"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_analysis(args: argparse.Namespace) -> None:
    from .analyze.stats import analyze_predictions

    config = load_config(args.config)
    analysis_config = config.get("analysis", {})
    n_boot = args.n_boot
    if n_boot is None:
        n_boot = int(analysis_config.get("bootstrap_samples", 1000))
    seed = args.seed
    if seed is None:
        seed = int(analysis_config.get("seed", 4108))
    result = analyze_predictions(
        predictions_path=args.predictions,
        output_dir=args.output,
        n_boot=n_boot,
        seed=seed,
    )
    print(
        json.dumps(
            {
                "conditions": result["conditions"],
                "summary_rows": len(result["summary"]),
                "comparison_rows": len(result["comparisons"]),
                "bootstrap_rows": len(result["bootstrap_intervals"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eq_quant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="build deterministic splits")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_build = data_subparsers.add_parser("build", help="generate dataset JSONL")
    data_build.add_argument("--config", required=True)
    data_build.add_argument("--output", required=True)
    data_build.add_argument(
        "--split",
        nargs="+",
        choices=[*SPLIT_ORDER, "all"],
        default=["pilot"],
    )
    data_build.set_defaults(handler=_build_data)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="score records with FP and RTN weights"
    )
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--records", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument(
        "--generation-subset",
        type=int,
        default=None,
        help="number of greedy raw-generation records per condition",
    )
    evaluate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing evaluation outputs in the output directory",
    )
    evaluate_parser.set_defaults(handler=_run_evaluation)

    analyze_parser = subparsers.add_parser(
        "analyze", help="paired base-cluster bootstrap summaries"
    )
    analyze_parser.add_argument("--config", required=True)
    analyze_parser.add_argument("--predictions", required=True)
    analyze_parser.add_argument("--output", required=True)
    analyze_parser.add_argument("--n-boot", type=int, default=None)
    analyze_parser.add_argument("--seed", type=int, default=None)
    analyze_parser.set_defaults(handler=_run_analysis)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
