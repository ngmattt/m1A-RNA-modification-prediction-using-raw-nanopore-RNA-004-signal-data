#!/usr/bin/env python3

import gzip

# ----------------------------
# Filepaths
# ----------------------------
FILTERED = "/N/project/NGS-JangaLab/Matthew/rna_seq_data/f5c_output/filtered/Hek293T_filtered1_eventalign.tsv.gz"
BED = "/N/project/NGS-JangaLab/Matthew/rna_seq_data/raw_data/HEK293T_RNA004/rna_mods/HEK293T_m1A_sites.bed"
OUT = "/N/project/NGS-JangaLab/Matthew/rna_seq_data/f5c_output/labeled/Hek293T_filtered_labeled_eventalign1.tsv.gz"

print("[INFO] Loading BED sites...")

# ----------------------------
# Load BED into FAST lookup set
# ----------------------------
sites = set()

with open(BED, "r") as bed:
    for line in bed:
        if line.startswith("#") or line.strip() == "":
            continue

        fields = line.split("\t")

        chrom = fields[0]
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom
        if chrom == "chrMT":
            chrom = "chrM"

        start = int(fields[1])
        end = int(fields[2])

        # convert BED (0-based) → 1-based positions
        for p in range(start + 1, end + 1):
            for offset in [-2, -1, 0, 1, 2]:
                sites.add((chrom, p + offset))

print(f"[INFO] Loaded {len(sites):,} genomic positions")

# ----------------------------
# Stream eventalign and label
# ----------------------------
print("[INFO] Labeling filtered eventalign file...")

with gzip.open(FILTERED, "rt") as fin, \
     gzip.open(OUT, "wt") as fout:

    header = fin.readline()
    fout.write(header.rstrip() + "\tlabel\n")

    count = 0
    positives = 0

    for line in fin:
        fields = line.split("\t")

        chrom = fields[0]
        pos = int(fields[1])

        if (chrom, pos) in sites:
            label = 1
            positives += 1
        else:
            label = 0

        fout.write(line.rstrip() + f"\t{label}\n")
        count += 1

        if count % 10_000_000 == 0:
            print(f"[INFO] Processed {count:,} rows...")

print("[INFO] Labeling complete.")
print(f"[INFO] Total rows processed: {count:,}")
print(f"[INFO] Positive labels: {positives:,}")

# ----------------------------
# Sanity check
# ----------------------------
if count > 0:
    positive_rate = positives / count
    print(f"[INFO] Positive rate: {positive_rate:.8f}")

    if positive_rate == 0:
        print("[WARNING] No positives detected → check coordinate mismatch!")
    elif positive_rate > 0.05:
        print("[WARNING] Positive rate unusually high → check BED alignment!")