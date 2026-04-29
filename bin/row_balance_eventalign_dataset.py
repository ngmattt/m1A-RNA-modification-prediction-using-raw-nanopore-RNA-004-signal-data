#!/usr/bin/env python3

from pathlib import Path
import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Balance a site-balanced eventalign table at the row level."
    )
    parser.add_argument("--input", required=True, help="Input site-balanced TSV.gz")
    parser.add_argument("--output", required=True, help="Output fully balanced TSV.gz")
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for row balancing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t", compression="gzip")

    pos_rows = df[df["label"] == 1]
    neg_rows = df[df["label"] == 0]
    n_rows = min(len(pos_rows), len(neg_rows))

    balanced_df = pd.concat(
        [
            pos_rows.sample(n=n_rows, random_state=args.random_state, replace=False),
            neg_rows.sample(n=n_rows, random_state=args.random_state, replace=False),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    balanced_df.to_csv(
        output_path,
        sep="\t",
        index=False,
        compression="gzip",
    )

    print(f"[INFO] Positive rows: {n_rows}")
    print(f"[INFO] Negative rows: {n_rows}")
    print(f"[INFO] Wrote fully balanced dataset to: {output_path}")


if __name__ == "__main__":
    main()
