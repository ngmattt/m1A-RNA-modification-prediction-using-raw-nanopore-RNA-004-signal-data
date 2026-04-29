#!/usr/bin/env python3

import argparse
import gzip
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Downsample a labeled eventalign table to balanced positive and negative sites."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input labeled eventalign TSV.gz",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output site-balanced TSV.gz",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for site sampling",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] First pass: collect site labels")
    site_labels = defaultdict(set)

    with gzip.open(input_path, "rt") as handle:
        _header = handle.readline()

        for line in handle:
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            pos = int(fields[1])
            label = fields[-1].strip()
            site_labels[(chrom, pos)].add(label)

    pos_sites = {site for site, labels in site_labels.items() if labels == {"1"}}
    neg_sites = {site for site, labels in site_labels.items() if labels == {"0"}}

    print(f"[INFO] Positive sites: {len(pos_sites)}")
    print(f"[INFO] Negative sites: {len(neg_sites)}")

    n_sites = min(len(pos_sites), len(neg_sites))
    pos_sites_sampled = set(random.sample(list(pos_sites), n_sites))
    neg_sites_sampled = set(random.sample(list(neg_sites), n_sites))
    selected_sites = pos_sites_sampled.union(neg_sites_sampled)

    print(f"[INFO] Selected total sites: {len(selected_sites)}")
    print("[INFO] Second pass: writing rows")

    with gzip.open(input_path, "rt") as fin, gzip.open(output_path, "wt") as fout:
        header = fin.readline()
        fout.write(header)

        for line in fin:
            fields = line.rstrip("\n").split("\t")
            site = (fields[0], int(fields[1]))
            if site in selected_sites:
                fout.write(line)

    print("[INFO] Done")


if __name__ == "__main__":
    main()
