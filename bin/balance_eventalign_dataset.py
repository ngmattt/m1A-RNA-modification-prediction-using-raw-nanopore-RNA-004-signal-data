#!/usr/bin/env python3

from pathlib import Path
import argparse

import pandas as pd


def balance_dataset(input_path, output_path, max_reads_per_site, random_state):
    df = pd.read_csv(input_path, sep="\t", compression="gzip")
    df["site"] = df["contig"].astype(str) + ":" + df["position"].astype(str)

    if max_reads_per_site > 0:
        df = df.groupby("site", group_keys=False).apply(
            lambda group: group.sample(
                n=min(len(group), max_reads_per_site),
                random_state=random_state,
            )
        ).reset_index(drop=True)

    site_labels = df.groupby("site")["label"].max().reset_index()
    pos_sites = site_labels.loc[site_labels["label"] == 1, "site"]
    neg_sites = site_labels.loc[site_labels["label"] == 0, "site"]

    n_sites = min(len(pos_sites), len(neg_sites))
    pos_keep = pos_sites.sample(n=n_sites, random_state=random_state, replace=False)
    neg_keep = neg_sites.sample(n=n_sites, random_state=random_state, replace=False)
    keep_sites = set(pd.concat([pos_keep, neg_keep]))
    df = df[df["site"].isin(keep_sites)].copy()

    pos_rows = df[df["label"] == 1]
    neg_rows = df[df["label"] == 0]
    n_rows = min(len(pos_rows), len(neg_rows))

    balanced_df = pd.concat(
        [
            pos_rows.sample(n=n_rows, random_state=random_state, replace=False),
            neg_rows.sample(n=n_rows, random_state=random_state, replace=False),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    balanced_df.drop(columns=["site"]).to_csv(
        output_path,
        sep="\t",
        index=False,
        compression="gzip",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct a site-balanced and row-balanced m1A eventalign dataset."
    )
    parser.add_argument("--input", required=True, help="Input labeled TSV.gz")
    parser.add_argument("--output", required=True, help="Output balanced TSV.gz")
    parser.add_argument(
        "--max-reads-per-site",
        type=int,
        default=50,
        help="Maximum number of event rows to retain per site before balancing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for downsampling and balancing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    balance_dataset(
        input_path=args.input,
        output_path=output_path,
        max_reads_per_site=args.max_reads_per_site,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
