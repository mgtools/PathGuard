"""Helpers shared by the experiment scripts."""
from __future__ import annotations

import os

# BLAS threads would only fight the process pool, and pinning them keeps the
# utility sums reproducible. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathguard import cohort, model  # noqa: E402

CEILINGS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
AUTOSOMES = tuple(f"chr{i}" for i in range(1, 23))
DEFAULT_PATHS = "data/targets/{sample}/{chrom}/path.npy"


def label(c: float) -> str:
    return f"{c:.2f}"


def read_list(path) -> list[str]:
    return [s.strip() for s in Path(path).read_text().split() if s.strip()]


@lru_cache(maxsize=4)
def load_cohort(cohort_dir: str, leave_out: str | None = None) -> cohort.Cohort:
    c = cohort.Cohort(cohort_dir)
    return c.leave_out(leave_out) if leave_out else c


@lru_cache(maxsize=2)
def load_chain(cohort_dir: str, leave_out: str | None = None) -> tuple[model.Chain, float]:
    """The chain for a cohort directory and its untilted log normaliser."""
    c = load_cohort(cohort_dir, leave_out)
    ch = model.make_chain(c.haplotypes, c.genetic_pos, c.positions, c.in_chain)
    return ch, model.baseline_log_norm(ch)


def load_path(template: str, sample: str, chrom: str, in_chain: np.ndarray) -> np.ndarray:
    """A projected target path, restricted to the chain positions."""
    return np.load(template.format(sample=sample, chrom=chrom))[in_chain]


def target_on_chain(cohort_dir: str, sample: str, chrom: str, template: str,
                    cohort_member: bool = False) -> np.ndarray:
    """The target's two haplotypes over the chain positions, as (T, 2).

    A cohort member's path is its own pair of haplotypes in the graph, read from
    the full cohort array; the chain is the one it is held out of. Any other
    target is read from a projection written by `pathguard target`.
    """
    full = load_cohort(cohort_dir)
    if not cohort_member:
        return load_path(template, sample, chrom, full.in_chain)
    rows = [i for i, h in enumerate(full.haplotype_ids) if h.rsplit(":", 1)[0] == sample]
    if len(rows) != 2:
        raise KeyError(f"{sample} is not a member of the cohort at {cohort_dir}")
    in_chain = load_cohort(cohort_dir, sample).in_chain
    return np.ascontiguousarray(full.haplotypes[rows][:, in_chain].T)


def _lengths_one(job):
    cohort_root, chrom, members = job
    c = cohort.Cohort(Path(cohort_root) / chrom)
    return {m: int((c.leave_out(m) if m else c).in_chain.sum()) for m in members}


def chain_lengths(cohort_root, members=(None,), chroms=AUTOSOMES, workers=1):
    """{member: {chrom: chain length}}; member None is the full cohort."""
    res = pool_map(_lengths_one, [(cohort_root, c, tuple(members)) for c in chroms], workers)
    return {m: {c: r[m] for c, r in zip(chroms, res)} for m in members}


def shares(lengths: dict[str, int]) -> dict[str, float]:
    """Each chromosome's share of one genome-wide utility budget, by chain length."""
    total = sum(lengths.values())
    return {c: n / total for c, n in lengths.items()}


def pool_map(fn, jobs, workers: int):
    """map() over a process pool, results in job order."""
    if workers <= 1:
        return [fn(j) for j in jobs]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(fn, jobs, chunksize=1)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1) + "\n")
