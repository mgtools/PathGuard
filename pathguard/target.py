"""Project a phased target genome onto the cohort's variant records.

The target path is the target's allele index at every cohort record, per
haplotype, with -1 where the target VCF has no usable call. Records are matched
on (POS, REF, ALT). The 1000 Genomes panel splits multi-allelic sites into one
biallelic record per ALT, so a multi-allelic cohort record is recovered by
matching each of its ALTs separately, provided every ALT is present in the
target VCF and no haplotype carries two of them.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def parse_gt(field: str) -> tuple[int, int]:
    gt = field.split(":", 1)[0]
    if "|" in gt:
        a, b = gt.split("|", 1)
    elif "/" in gt:
        a, b = gt.split("/", 1)
    else:
        a, b = gt, None
    x = -1 if a in (".", "") else int(a)
    y = -1 if b is None or b in (".", "") else int(b)
    return x, y


def project(target_vcf, sample: str, pos, ref, alt) -> np.ndarray:
    """(n_records, 2) int16 allele indices of `sample` on the cohort records."""
    n = len(pos)
    exact = {}
    for i in range(n):
        exact.setdefault((int(pos[i]), ref[i], alt[i]), i)

    # Multi-allelic cohort records, and which of their ALTs the target VCF has.
    multi = {}
    for i in range(n):
        if "," in alt[i]:
            multi.setdefault((int(pos[i]), ref[i]), set()).update(alt[i].split(","))
    present = {}
    with _open(target_vcf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 6)
            key = (int(f[1]), f[3])
            if key in multi:
                present.setdefault(key, set()).add(f[4])
    split = {}
    for i in range(n):
        if "," in alt[i]:
            key = (int(pos[i]), ref[i])
            alts = alt[i].split(",")
            if set(alts) <= present.get(key, set()):
                for j, a in enumerate(alts, start=1):
                    split.setdefault((key[0], key[1], a), (i, j))

    path = np.full((n, 2), -1, dtype=np.int16)
    split_calls: dict[int, dict[int, tuple[int, int]]] = {}
    col = None
    with _open(target_vcf) as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                names = line.rstrip("\n").split("\t")[9:]
                if sample not in names:
                    raise ValueError(f"{target_vcf} has no sample {sample!r}")
                col = 9 + names.index(sample)
                continue
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            key = (int(f[1]), f[3], f[4])
            gt = parse_gt(f[col])
            row = exact.get(key)
            if row is not None:
                path[row] = gt
            elif key in split:
                row, j = split[key]
                split_calls.setdefault(row, {})[j] = gt

    # Rebuild each multi-allelic record from its biallelic pieces.
    for row, by_alt in split_calls.items():
        out = [-1, -1]
        for h in (0, 1):
            carried = [j for j, gt in by_alt.items() if gt[h] == 1]
            if len(carried) > 1:
                break                      # contradictory: leave the record uncalled
            if carried:
                out[h] = carried[0]
            elif any(gt[h] == -1 for gt in by_alt.values()):
                out[h] = -1
            else:
                out[h] = 0
        else:
            path[row] = out
    return path


def load_sites(sites_tsv) -> tuple[np.ndarray, list[str], list[str]]:
    """POS, REF and ALT of every cohort record, from the cohort's sites.tsv."""
    pos, ref, alt = [], [], []
    with _open(sites_tsv) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ip, ir, ia = header.index("pos"), header.index("ref"), header.index("alt")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            pos.append(int(f[ip]))
            ref.append(f[ir])
            alt.append(f[ia])
    return np.asarray(pos, dtype=np.int64), ref, alt


def build(target_vcf, sample: str, cohort_dir, out_path) -> np.ndarray:
    pos, ref, alt = load_sites(Path(cohort_dir) / "sites.tsv")
    path = project(target_vcf, sample, pos, ref, alt)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, path)
    return path
