"""AF loss with missing genotypes excluded, and how few variant pairs carry the LD loss.

PanMixer's af_loss treats an uncalled cell as an ordinary allele, so a cell whose
called status changes scores like a changed genotype. The two tools differ in how
much of that they do: PathGuard emits a fixed-support path while PanMixer edits a
copy of the individual's own row. This recomputes af_loss with the missingness
held at the original's, so only jointly called cells can differ, and checks the
recomputation against the shipped evaluator's own output first.

ld_loss averages |delta r-squared| over variant pairs within five position units.
Its p_AB term divides by every haplotype while p_A and p_B divide by the called
ones, which inflates pairs that have uncalled cells. The share carried by the
largest few pairs says how much of the total that is.

    python missing_excluded.py --sources ROOT/sources.json --out results/missing.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from summarize import capacity_key, ceiling_key, read_rows

SHARED: dict = {}


def load_build(starting_data: Path, chrom: int) -> dict:
    d = starting_data / f"chr{chrom}"
    positions = np.load(d / "pangenome_positions.npy")
    snp = np.zeros(len(positions), bool)
    snp[np.load(d / "pangenome_mask.npy")] = True
    pangenome = np.load(d / "pangenome.npy")
    subjects = np.load(d / "pangenome_subjects.npy").tolist()
    snp_haplotypes = pangenome[:, snp].astype(float)
    snp_haplotypes[snp_haplotypes == -1] = np.nan
    snp_positions = positions[snp]
    return {"pangenome": pangenome, "subjects": subjects, "snp": snp,
            "snp_haplotypes": snp_haplotypes, "snp_positions": snp_positions,
            # the last partner of each site, five position units away
            "window_end": np.searchsorted(snp_positions, snp_positions + 5, side="left")}


def allele_frequency_loss(args) -> tuple:
    """The shipped metric and the same metric with the missingness held fixed."""
    subject, source = args
    pangenome, subjects = SHARED["pangenome"], SHARED["subjects"]
    index = subjects.index(subject)
    n_sites = pangenome.shape[1]
    original = pangenome[index]
    release = np.load(source)

    held = release.copy()
    uncalled = (original == -1) | (release == -1)
    held[uncalled] = original[uncalled]

    out = {"release_uncalled": int((release == -1).sum()),
           "original_uncalled": int((original == -1).sum())}
    for name, row in (("shipped", release), ("missing_excluded", held)):
        other = pangenome.copy()
        other[index] = row
        # Allele -1 becomes column 0, so each site's frequency vector is a count
        # over the 88 haplotypes of the cohort.
        a = pangenome.transpose(1, 0, 2).reshape(n_sites, -1) + 1
        b = other.transpose(1, 0, 2).reshape(n_sites, -1) + 1
        alleles = int(max(a.max(), b.max())) + 1
        total, counted = 0.0, 0
        for start in range(0, n_sites, 20000):
            chunk_a, chunk_b = a[start:start + 20000], b[start:start + 20000]
            rows = np.arange(chunk_a.shape[0])[:, None]
            count_a = np.zeros((chunk_a.shape[0], alleles), np.int32)
            count_b = np.zeros((chunk_b.shape[0], alleles), np.int32)
            np.add.at(count_a, (rows, chunk_a), 1)
            np.add.at(count_b, (rows, chunk_b), 1)
            present = (count_a > 0) | (count_b > 0)
            freq_a = count_a / count_a.sum(1, keepdims=True)
            freq_b = count_b / count_b.sum(1, keepdims=True)
            total += float(np.abs(freq_a - freq_b)[present].sum())
            counted += int(present.sum())
        out[name] = total / counted
    return subject, out


def r_squared(x: np.ndarray, ys: np.ndarray) -> np.ndarray:
    p_a = np.nanmean(x)
    p_b = np.nanmean(ys, axis=(0, 2))
    xs = np.broadcast_to(x[:, None], ys.shape)
    p_ab = np.nanmean((xs == 1) & (ys == 1), axis=(0, 2))
    d = p_ab - p_a * p_b
    denominator = p_a * (1 - p_a) * p_b * (1 - p_b)
    return d * d / np.where(denominator == 0, np.nan, denominator)


def linkage_loss(args) -> tuple:
    """ld_loss for one release, plus the share carried by its largest pairs."""
    subject, source, largest = args
    haplotypes, positions = SHARED["snp_haplotypes"], SHARED["snp_positions"]
    window_end, subjects = SHARED["window_end"], SHARED["subjects"]
    release = np.load(source)[SHARED["snp"]].astype(float)
    release[release == -1] = np.nan
    other = haplotypes.copy()
    other[subjects.index(subject), :] = release

    total, pairs, contributions = 0.0, 0, []
    for i in range(len(positions)):
        if window_end[i] <= i + 1:
            continue
        partners = np.arange(i + 1, window_end[i])
        before = r_squared(haplotypes[:, i], haplotypes[:, partners])
        after = r_squared(other[:, i], other[:, partners])
        delta = np.abs(before - after)
        total += float(np.nansum(delta))
        pairs += int(np.sum(~np.isnan(before) & ~np.isnan(after)))
        contributions.extend(v for v in delta[~np.isnan(delta)].tolist() if v > 0)
    contributions.sort(reverse=True)
    return subject, {"ld_loss": total / pairs, "pairs": pairs,
                     "largest_share": sum(contributions[:largest]) / total if total else 0.0}


def releases(source: dict, point: str, chrom: int) -> list[tuple]:
    """(subject, staged release) for every individual at one ceiling or capacity."""
    root = Path(source["root"])
    base = root / "experiments" / f"exp_{source['experiment']}" / "data" / f"chr{chrom}"
    rows = [r for r in read_rows(root, source["experiment"], chrom) if r["point"] == point]
    if not rows:
        raise SystemExit(f"exp_{source['experiment']} has no rows at {point}")
    return [(r["subject"], base / str(r["row"]) / "new_haplotypes.npy") for r in rows]


def shipped_af_loss(source: dict, point: str, chrom: int) -> dict:
    """What the evaluator itself wrote, to check the recomputation against."""
    root = Path(source["root"])
    base = root / "experiments" / f"exp_{source['experiment']}" / "data" / f"chr{chrom}"
    rows = [r for r in read_rows(root, source["experiment"], chrom) if r["point"] == point]
    return {r["subject"]: json.loads((base / str(r["row"]) / "af_loss.json").read_text())["all"]
            for r in rows}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chrom", type=int, default=21)
    p.add_argument("--ceiling", default="0.80")
    p.add_argument("--capacity", default="0.9")
    p.add_argument("--workers", type=int, default=22)
    p.add_argument("--largest-pairs", type=int, default=6)
    p.add_argument("--no-linkage", action="store_true",
                   help="skip the LD pass, which is the slow half")
    a = p.parse_args()

    sources = json.loads(a.sources.read_text())
    point = {"pathguard": ceiling_key(float(a.ceiling)),
             "panmixer": capacity_key(float(a.capacity))}
    work = {tool: releases(sources[f"{tool}_af_ld"], point[tool], a.chrom)
            for tool in ("pathguard", "panmixer")}
    starting_data = Path(sources["pathguard_af_ld"]["root"]) / "starting_data"
    # Loaded once here; the worker processes inherit it rather than each
    # unpickling their own copy of the panel.
    SHARED.update(load_build(starting_data, a.chrom))
    print(f"{starting_data.resolve()}: {SHARED['pangenome'].shape[1]:,} sites, "
          f"{int(SHARED['snp'].sum()):,} SNP sites")
    for tool, rows in work.items():
        print(f"  {tool}: {len(rows)} releases at {point[tool]}")

    out = {"chromosome": a.chrom, "ceiling": point["pathguard"],
           "capacity": point["panmixer"], "largest_pairs": a.largest_pairs,
           "af_loss_missing_excluded": {}, "af_loss_shipped_recomputed": {},
           "uncalled_cells": {}}
    with Pool(a.workers) as pool:
        for tool, rows in work.items():
            per_subject = dict(pool.map(allele_frequency_loss, rows))
            out["af_loss_missing_excluded"][tool] = {
                s: v["missing_excluded"] for s, v in per_subject.items()}
            out["af_loss_shipped_recomputed"][tool] = {
                s: v["shipped"] for s, v in per_subject.items()}
            out["uncalled_cells"][tool] = {
                s: v["release_uncalled"] for s, v in per_subject.items()}
            # The recomputation has to land on the evaluator's own number first.
            shipped = shipped_af_loss(sources[f"{tool}_af_ld"], point[tool], a.chrom)
            worst = max(abs(shipped[s] - per_subject[s]["shipped"]) for s in shipped)
            print(f"  {tool}: recomputed shipped af_loss differs from the evaluator's "
                  f"by at most {worst:.2e}")
            if worst > 1e-9:
                raise SystemExit("the recomputation does not reproduce the shipped "
                                 "metric, so its counterfactual means nothing")
            print(f"  {tool}: median af_loss with missing genotypes excluded "
                  f"{st.median(out['af_loss_missing_excluded'][tool].values()):.6f}")

        if not a.no_linkage:
            out["ld_loss_largest_pairs"] = {"n_pairs": a.largest_pairs}
            out["ld_loss_recomputed"] = {}
            for tool, rows in work.items():
                per_subject = dict(pool.map(
                    linkage_loss, [(s, f, a.largest_pairs) for s, f in rows]))
                out["ld_loss_largest_pairs"][tool] = {
                    s: v["largest_share"] for s, v in per_subject.items()}
                out["ld_loss_recomputed"][tool] = {
                    s: v["ld_loss"] for s, v in per_subject.items()}
                print(f"  {tool}: median share carried by the largest {a.largest_pairs} "
                      f"pairs {st.median(out['ld_loss_largest_pairs'][tool].values()):.3f}")

    ours = out["af_loss_missing_excluded"]["pathguard"]
    theirs = out["af_loss_missing_excluded"]["panmixer"]
    both = sorted(set(ours) & set(theirs))
    print(f"\nmissing genotypes excluded: median {st.median(ours[s] for s in both):.6f} "
          f"(PathGuard) vs {st.median(theirs[s] for s in both):.6f} (PanMixer), "
          f"PathGuard lower for {sum(1 for s in both if ours[s] < theirs[s])} of {len(both)}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
