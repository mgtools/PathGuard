"""Solve each target's tilt for every accuracy ceiling.

For each target the KL divergence against the constructed decoy is measured at
the probe tilts, KL ~ c eta^2 is fitted, the tilt for each ceiling is read off,
and then corrected `--refinements` times by re-measuring.

Examples (paper settings):
  # external targets, chr21 treated as the whole release
  python scripts/calibrate.py --targets configs/external_targets.txt \
      --probes 227.1 454.19 --refinements 1 --out results/etas_external_chr21.json
  # external targets, all 22 autosomes as one release
  python scripts/calibrate.py --targets configs/external_targets.txt --chroms all \
      --probes 1654.3 3308.5 --refinements 0 --out results/etas_external_genome.json
  # cohort members, chr21 carrying its share of a genome-wide budget
  python scripts/calibrate.py --targets configs/cohort_members.txt --cohort-members \
      --genome-share --probes 15000 30000 --refinements 1 --out results/etas_cohort_chr21.json
"""
import argparse
import math

import numpy as np

from common import (AUTOSOMES, CEILINGS, DEFAULT_PATHS, chain_lengths, label, load_chain,
                    pool_map, read_list, shares, target_on_chain, write_json)
from pathguard import calibrate, model

ARGS = None


def kl_job(job):
    """KL against the constructed decoy for one (target, chromosome) at several tilts."""
    target, chrom, share, etas = job
    held = target if ARGS.cohort_members else None
    d = f"{ARGS.cohort_root}/{chrom}"
    chain, _ = load_chain(d, held)
    p = target_on_chain(d, target, chrom, ARGS.paths, ARGS.cohort_members)
    q = calibrate.constructed_decoy(p)
    beta = model.uniform_beta(chain.T, share)
    return [calibrate.kl(chain, p, q, beta, e) for e in etas]


def measure(targets, chroms, share_of, etas_of):
    """Genome (or chromosome) KL per target, at each target's own list of tilts."""
    jobs = [(t, c, share_of(t, c), etas_of[t]) for t in targets for c in chroms]
    res = pool_map(kl_job, jobs, ARGS.workers)
    out = {}
    for (t, c, _, _), kls in zip(jobs, res):
        out.setdefault(t, []).append(kls)
    return {t: [math.fsum(col) for col in zip(*rows)] for t, rows in out.items()}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True, help="file listing target sample ids")
    ap.add_argument("--chroms", nargs="+", default=["chr21"], help="chromosomes, or 'all'")
    ap.add_argument("--cohort-root", default="data/cohort")
    ap.add_argument("--paths", default=DEFAULT_PATHS, help="target path template")
    ap.add_argument("--cohort-members", action="store_true",
                    help="targets are cohort members: hold each out of its own donor panel")
    ap.add_argument("--genome-share", action="store_true",
                    help="give the chromosomes their share of a budget spread over all "
                         "22 autosomes (default: the budget covers --chroms only)")
    ap.add_argument("--ceilings", type=float, nargs="+", default=list(CEILINGS))
    ap.add_argument("--probes", type=float, nargs="+", required=True)
    ap.add_argument("--refinements", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ARGS = a = ap.parse_args()

    targets = sorted(read_list(a.targets))
    chroms = list(AUTOSOMES) if a.chroms == ["all"] else a.chroms
    budget_chroms = list(AUTOSOMES) if a.genome_share else chroms
    members = targets if a.cohort_members else [None]
    if len(budget_chroms) > 1:
        lengths = chain_lengths(a.cohort_root, members, budget_chroms, a.workers)
        share_table = {m: shares(lengths[m]) for m in members}
    else:
        share_table = {m: {chroms[0]: 1.0} for m in members}

    def share_of(t, c):
        return share_table[t if a.cohort_members else None][c]

    probes = measure(targets, chroms, share_of, {t: a.probes for t in targets})
    tilted = [c for c in a.ceilings if c > 0.5]
    etas = {t: [calibrate.fit_eta(dict(zip(a.probes, probes[t])), c) for c in tilted]
            for t in targets}
    kls = None
    for step in range(a.refinements + (1 if a.refinements else 0)):
        kls = measure(targets, chroms, share_of, etas)
        if step < a.refinements:
            etas = {t: [calibrate.refine_eta(e, k, c) for e, k, c in zip(etas[t], kls[t], tilted)]
                    for t in targets}

    out = {"chroms": chroms, "probes": a.probes, "refinements": a.refinements,
           "cohort_members": a.cohort_members, "targets": {}}
    for t in targets:
        row = {"probe_kl": dict(zip(map(str, a.probes), probes[t])),
               "share": {c: share_of(t, c) for c in chroms}, "ceilings": {}}
        for c in a.ceilings:
            if c <= 0.5:
                row["ceilings"][label(c)] = {"eta": 0.0, "kl": 0.0, "ceiling": 0.5}
                continue
            i = tilted.index(c)
            cell = {"eta": etas[t][i]}
            if kls is not None:
                cell.update(kl=kls[t][i], ceiling=calibrate.pinsker_ceiling(kls[t][i]))
            row["ceilings"][label(c)] = cell
        out["targets"][t] = row
        print(t, " ".join(f"{k}:{v['eta']:.2f}" for k, v in row["ceilings"].items()))
    write_json(a.out, out)


if __name__ == "__main__":
    main()
