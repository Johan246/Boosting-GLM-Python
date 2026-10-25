"""Command-line interface for downloading data and running the workflow."""

from __future__ import annotations

import argparse
import json

from .config import DATASETS
from .data import download_data
from .pipeline import train_dataset


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="boosting-glm")
    commands = root.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download-data", help="Download datasets from the original repository")
    download.add_argument("--output", default="data/raw")
    download.add_argument("--overwrite", action="store_true")

    train = commands.add_parser("train", help="Train and evaluate one dataset")
    train.add_argument("--dataset", choices=DATASETS, required=True)
    train.add_argument("--data-dir", default="data/raw")
    train.add_argument("--output", default=None)
    train.add_argument("--max-rows", type=int)
    train.add_argument("--trees", type=int, default=300)
    train.add_argument("--seed", type=int, default=42)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "download-data":
        files = download_data(args.output, overwrite=args.overwrite)
        print(f"Downloaded {len(files)} file(s) to {args.output}")
        return 0
    output = args.output or f"outputs/{args.dataset}"
    report = train_dataset(
        args.dataset,
        args.data_dir,
        output,
        max_rows=args.max_rows,
        trees=args.trees,
        random_state=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

