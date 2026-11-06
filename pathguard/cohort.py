"""Build the cohort arrays for one chromosome: graph -> VCF -> haplotypes -> chain.

Steps
  1. `vg deconstruct` the PGGB graph against the GRCh38 path (one VCF record per
     variant site, one column per sample).
  2. drop the CHM13 column, leaving the 44 HPRC individuals (88 haplotypes).
  3. read the genotypes into a (haplotypes x records) allele-index matrix.
  4. give every record a genetic position: linear interpolation on the genetic
     map, and outside the map's span a declared constant rate (cM/Mb) applied to
     base pairs that are not assembly gaps.
  5. choose the chain: records where at least one donor has a call, minus any
     run of zero genetic distance that no single donor is called across (the
     copying model has no path through such a run).

Everything here is a function of the graph, the cohort and the map only. No
target genome is read.
"""
from __future__ import annotations

import gzip
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .target import parse_gt

# The declared rate: the recombination rate, in cM/Mb, given to the stretches a
# genetic map says nothing about. It is a declared convention, not a measurement
# of those stretches, and it is the mean rate over the map's mapped span.
DECLARED_RATE = 1.2707


# --- steps 1-2: external tools ----------------------------------------------

def deconstruct(gfa, out_vcf, vg="vg", ref_prefix="grch38", threads=16) -> None:
    """Run `vg deconstruct` on a (possibly gzipped) GFA."""
    gfa, out_vcf = Path(gfa), Path(out_vcf)
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    plain = gfa
    if gfa.suffix == ".gz":   # vg cannot read a gzipped GFA
        plain = out_vcf.parent / gfa.with_suffix("").name
        with gzip.open(gfa, "rb") as fi, open(plain, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=1 << 24)
    with open(out_vcf, "wb") as fh:
        subprocess.run([vg, "deconstruct", "-P", ref_prefix, "-a", "-t", str(threads),
                        str(plain)], stdout=fh, check=True)
    if plain != gfa:
        plain.unlink()


def drop_sample(in_vcf, out_vcf, sample="chm13") -> None:
    subprocess.run(["bcftools", "view", "-s", f"^{sample}", "--force-samples",
                    "-Oz", "-o", str(out_vcf), str(in_vcf)], check=True)
    subprocess.run(["bcftools", "index", "-f", str(out_vcf)], check=True)


# --- step 3: genotypes --------------------------------------------------------

def read_vcf(path):
    """Return (samples, pos, ref, alt, haplotypes) from a phased cohort VCF.

    haplotypes is (2 * n_samples, n_records) int16; row 2i is sample i's first
    haplotype, row 2i+1 its second; -1 means no call.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    samples, pos, ref, alt, rows = None, [], [], [], []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            f = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = f[9:]
                continue
            pos.append(int(f[1]))
            ref.append(f[3])
            alt.append(f[4])
            row = []
            for g in f[9:]:
                row.extend(parse_gt(g))
            rows.append(np.array(row, dtype=np.int16))
    haps = np.stack(rows, axis=1)
    return samples, np.asarray(pos, dtype=np.int64), ref, alt, haps


# --- step 4: genetic positions ------------------------------------------------

def read_gmap(path) -> tuple[np.ndarray, np.ndarray]:
    """A `pos chr cM` genetic map (SHAPEIT4 / Beagle b38 format)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    pos, cm = [], []
    with opener(path, "rt") as fh:
        fh.readline()
        for line in fh:
            if line.strip():
                f = line.split()
                pos.append(int(f[0]))
                cm.append(float(f[-1]))
    return np.asarray(pos, dtype=np.int64), np.asarray(cm, dtype=np.float64)


def read_gaps(path, chrom: str) -> np.ndarray:
    """One chromosome's assembly gaps, (n, 2) half-open [start, end), from UCSC gap.txt.gz."""
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 8 and f[1] == chrom:
                out.append((int(f[2]), int(f[3])))
    return np.asarray(sorted(out), dtype=np.int64).reshape(-1, 2)


def _gap_bp_below(x: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """Total gap base pairs below each position x."""
    starts, ends = gaps[:, 0], gaps[:, 1]
    full = np.searchsorted(ends, x, side="right")
    total = np.concatenate([[0], np.cumsum(ends - starts)])[full]
    idx = np.clip(np.searchsorted(starts, x, side="right") - 1, 0, None)
    inside = (x > starts[idx]) & (x < ends[idx])
    return total + np.where(inside, x - starts[idx], 0)


def gap_free_bp(lo: np.ndarray, hi: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    if len(gaps) == 0:
        return hi - lo
    return (hi - lo) - (_gap_bp_below(hi, gaps) - _gap_bp_below(lo, gaps))


def genetic_positions(positions, gmap_path, gaps=None) -> np.ndarray:
    """cM for every record. Interpolated inside the map; the declared rate outside it."""
    rate = DECLARED_RATE
    mp, mc = read_gmap(gmap_path)
    cm = np.interp(positions.astype(np.float64), mp.astype(np.float64), mc,
                   left=np.nan, right=np.nan)
    known = np.flatnonzero(np.isfinite(cm))
    lo, hi = known[0], known[-1]
    gaps = np.zeros((0, 2), dtype=np.int64) if gaps is None else gaps
    below = np.arange(len(cm)) < lo
    above = np.arange(len(cm)) > hi
    if below.any():
        bp = gap_free_bp(positions[below], np.full(below.sum(), positions[lo]), gaps)
        cm[below] = cm[lo] - bp / 1e6 * rate
    if above.any():
        bp = gap_free_bp(np.full(above.sum(), positions[hi]), positions[above], gaps)
        cm[above] = cm[hi] + bp / 1e6 * rate
    return cm


# --- step 5: the chain ----------------------------------------------------------

def zero_distance_runs(cm: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs [a, b] (inclusive) of consecutive positions sharing one cM value."""
    idx = np.flatnonzero(np.diff(cm) == 0)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[brk + 1]]
    ends = np.r_[idx[brk], idx[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def chain_mask(haplotypes: np.ndarray, cm: np.ndarray) -> np.ndarray:
    """Which records are chain positions.

    A record needs at least one called donor. Inside a run of zero genetic
    distance the chain cannot switch donors, so a run that no single donor is
    called across is untraversable and is dropped whole.
    """
    called = haplotypes != -1
    mask = called.any(axis=0)
    sites = np.flatnonzero(mask)
    for a, b in zero_distance_runs(cm[sites]):
        if not called[:, sites[a:b + 1]].all(axis=1).any():
            mask[sites[a:b + 1]] = False
    return mask


def build(cohort_vcf, gmap, outdir, gaps_table=None, chrom=None) -> dict:
    """Write the cohort arrays for one chromosome into `outdir`."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    samples, pos, ref, alt, haps = read_vcf(cohort_vcf)
    gaps = read_gaps(gaps_table, chrom) if gaps_table else None
    cm = genetic_positions(pos, gmap, gaps)
    in_chain = chain_mask(haps, cm)

    np.save(outdir / "haplotypes.npy", haps)
    np.save(outdir / "positions.npy", pos)
    np.save(outdir / "genetic_pos.npy", cm)
    np.save(outdir / "in_chain.npy", in_chain)
    ids = np.array([f"{s}:{h}" for s in samples for h in (0, 1)])
    np.save(outdir / "haplotype_ids.npy", ids)
    with open(outdir / "sites.tsv", "w") as fh:
        fh.write("pos\tref\talt\n")
        for p, r, a in zip(pos, ref, alt):
            fh.write(f"{p}\t{r}\t{a}\n")
    return {"samples": len(samples), "records": len(pos), "chain": int(in_chain.sum())}


# --- loading ----------------------------------------------------------------------

class Cohort:
    """The arrays `build` writes, loaded from a cohort directory."""

    def __init__(self, cohort_dir):
        d = Path(cohort_dir)
        self.dir = d
        self.haplotypes = np.load(d / "haplotypes.npy")
        self.positions = np.load(d / "positions.npy")
        self.genetic_pos = np.load(d / "genetic_pos.npy")
        self.in_chain = np.load(d / "in_chain.npy")
        self.haplotype_ids = np.load(d / "haplotype_ids.npy").astype(str)

    @property
    def samples(self) -> list[str]:
        return [h.rsplit(":", 1)[0] for h in self.haplotype_ids[::2]]

    def leave_out(self, sample: str) -> "Cohort":
        """The cohort without both haplotypes of `sample`, with its chain recomputed."""
        rows = [i for i, h in enumerate(self.haplotype_ids) if h.rsplit(":", 1)[0] == sample]
        if len(rows) != 2:
            raise KeyError(f"{sample} is not a cohort member")
        keep = np.ones(len(self.haplotype_ids), dtype=bool)
        keep[rows] = False
        out = object.__new__(Cohort)
        out.dir = self.dir
        out.haplotypes = self.haplotypes[keep]
        out.positions = self.positions
        out.genetic_pos = self.genetic_pos
        out.haplotype_ids = self.haplotype_ids[keep]
        out.in_chain = chain_mask(out.haplotypes, self.genetic_pos)
        return out
