#!/usr/bin/env python3

from pathlib import Path
import argparse
import gzip

import pandas as pd


def normalize_chromosome(chrom):
    chrom = str(chrom)
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom
    if chrom == "chrMT":
        chrom = "chrM"
    return chrom


def build_site_set(bed_path, flank_size):
    bed_df = pd.read_csv(bed_path, sep="\t", header=None)
    sites = set()

    for _, row in bed_df.iterrows():
        chrom = normalize_chromosome(row[0])
        start = int(row[1])
        end = int(row[2])

        for position in range(start + 1, end + 1):
            for offset in range(-flank_size, flank_size + 1):
                sites.add((chrom, position + offset))

    return sites


def label_eventalign(eventalign_path, bed_path, output_path, flank_size):
    positive_sites = build_site_set(bed_path, flank_size)

    with gzip.open(eventalign_path, "rt") as fin, gzip.open(output_path, "wt") as fout:
        header = fin.readline().rstrip("\n")
        fout.write(header + "\tlabel\n")

        for line in fin:
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            position = int(fields[1])
            label = 1 if (chrom, position) in positive_sites else 0
            fout.write(line.rstrip("\n") + f"\t{label}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label an eventalign TSV.gz using a BED file of m1A sites."
    )
    parser.add_argument("--eventalign", required=True, help="Filtered eventalign TSV.gz")
    parser.add_argument("--bed", required=True, help="BED file of labeled m1A sites")
    parser.add_argument("--output", required=True, help="Output labeled TSV.gz")
    parser.add_argument(
        "--flank-size",
        type=int,
        default=2,
        help="Number of bases on each side of each BED position to label as positive",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_eventalign(
        eventalign_path=args.eventalign,
        bed_path=args.bed,
        output_path=output_path,
        flank_size=args.flank_size,
    )


if __name__ == "__main__":
    main()
