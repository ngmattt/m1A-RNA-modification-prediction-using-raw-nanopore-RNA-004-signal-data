#!/usr/bin/env python3

import gzip
import random
from collections import defaultdict

INPUT = "/N/project/NGS-JangaLab/Matthew/rna_seq_data/f5c_output/labeled/Hek293T_filtered_labeled_eventalign1.tsv.gz"
OUTPUT = "/N/project/NGS-JangaLab/Matthew/rna_seq_data/ML_data/m1A_site_balanced1.tsv.gz"

random.seed(42)

print("[INFO] First pass: collect site labels")

# Track ALL labels per site
site_labels = defaultdict(set)

with gzip.open(INPUT, "rt") as f:
    header = f.readline()

    for line in f:
        fields = line.split("\t")

        chrom = fields[0]
        pos = int(fields[1])
        label = fields[-1].strip()

        site_labels[(chrom, pos)].add(label)

# Separate PURE sites
pos_sites = {s for s, labels in site_labels.items() if labels == {"1"}}
neg_sites = {s for s, labels in site_labels.items() if labels == {"0"}}

print(f"[INFO] Positive sites: {len(pos_sites)}")
print(f"[INFO] Negative sites: {len(neg_sites)}")

# Balance safely
n = min(len(pos_sites), len(neg_sites))

pos_sites_sampled = set(random.sample(list(pos_sites), n))
neg_sites_sampled = set(random.sample(list(neg_sites), n))

selected_sites = pos_sites_sampled.union(neg_sites_sampled)

print(f"[INFO] Selected total sites: {len(selected_sites)}")

print("[INFO] Second pass: writing rows")

with gzip.open(INPUT, "rt") as f, gzip.open(OUTPUT, "wt") as out:
    header = f.readline()
    out.write(header)

    for line in f:
        fields = line.split("\t")
        site = (fields[0], int(fields[1]))

        if site in selected_sites:
            out.write(line)

awk -F'\t' 'NR>1 {print $1":"$2":"$NF}' | sort | uniq | wc -l
print("[INFO] Done")
