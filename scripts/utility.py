"""Exact utility of the releases, per target and accuracy ceiling (Tables 5 and 7).

Nothing here is sampled. The posterior state marginals of the tilted chain give
the exact expectation of every quantity below over the release distribution:

  altered        expected called target alleles the release does not reproduce
  concordance    1 - altered / called
  nonref_altered the same, counting only positions where the target carries a
                 non-reference allele

Both haplotypes count, so the denominator is called target alleles over the two.

  # genome-wide, ten external targets (Table 7)
  python scripts/utility.py --targets configs/external_targets.txt --chroms all \
      --etas results/etas_external_genome.json --out results/utility_genome.json
  # chr21, the 44 cohort members held out of their own donor panel (Table 5)
  python scripts/utility.py --targets configs/cohort_members.txt --cohort-members \
      --etas results/etas_cohort_chr21.json --out results/utility_cohort_chr21.json
"""
import argparse
import json
import math
import statistics

from common import (AUTOSOMES, CEILINGS, DEFAULT_PATHS, label, load_chain, pool_map,
                    read_list, target_on_chain, write_json)
from pathguard import model

ARGS = None


def expectations(chain, path, beta, eta):
    """Exact expected counts for one diploid target at one tilt."""
    out = dict(altered=0.0, called=0, nonref_matched=0.0, nonref=0)
    for h in (0, 1):
        x = path[:, h]
        psi = model.potentials(chain, x, beta, eta / 2.0)
        alpha, _ = model.forward(chain, psi)
        gamma = model.marginals(chain, alpha, psi)
        called = x >= 0
        nonref = x > 0
        out["altered"] += float((gamma.T * ((chain.haps != x[None, :]) & called[None, :])).sum())
        out["called"] += int(called.sum())
        out["nonref_matched"] += float((gamma.T * ((chain.haps == x[None, :]) & nonref[None, :])).sum())
        out["nonref"] += int(nonref.sum())
    return out


def job(args):
    target, chrom = args
    held = target if ARGS.cohort_members else None
    d = f"{ARGS.cohort_root}/{chrom}"
    chain, _ = load_chain(d, held)
    path = target_on_chain(d, target, chrom, ARGS.paths, ARGS.cohort_members)
    out = {}
    for c in ARGS.ceilings:
        cell = ARGS.etas[target]["ceilings"][label(c)]
        beta = model.uniform_beta(chain.T, ARGS.etas[target]["share"][chrom])
        out[label(c)] = expectations(chain, path, beta, cell["eta"])
    return {"target": target, "chrom": chrom, "T": chain.T, "cells": out}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--etas", required=True, help="output of calibrate.py for these targets")
    ap.add_argument("--chroms", nargs="+", default=["chr21"], help="chromosomes, or 'all'")
    ap.add_argument("--cohort-members", action="store_true")
    ap.add_argument("--ceilings", type=float, nargs="+", default=list(CEILINGS))
    ap.add_argument("--cohort-root", default="data/cohort")
    ap.add_argument("--paths", default=DEFAULT_PATHS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ARGS = a = ap.parse_args()
    a.etas = json.load(open(a.etas))["targets"]
    targets = sorted(read_list(a.targets))
    chroms = list(AUTOSOMES) if a.chroms == ["all"] else a.chroms

    rows = pool_map(job, [(t, c) for t in targets for c in chroms], a.workers)

    # Sum each target's chromosomes, then take the median over targets.
    per_target = {t: {} for t in targets}
    for t in targets:
        mine = [r for r in rows if r["target"] == t]
        for c in a.ceilings:
            cells = [r["cells"][label(c)] for r in mine]
            tot = {k: math.fsum(x[k] for x in cells) for k in cells[0]}
            tot["altered_fraction"] = tot["altered"] / tot["called"]
            tot["concordance"] = 1.0 - tot["altered_fraction"]
            tot["nonref_concordance"] = tot["nonref_matched"] / tot["nonref"]
            tot["eta"] = a.etas[t]["ceilings"][label(c)]["eta"]
            per_target[t][label(c)] = tot

    median = {}
    keys = ("altered", "altered_fraction", "concordance", "nonref_concordance", "eta")
    for c in a.ceilings:
        median[label(c)] = {k: statistics.median(per_target[t][label(c)][k] for t in targets)
                            for k in keys}
        median[label(c)]["nonref_altered"] = statistics.median(
            per_target[t][label(c)]["nonref"] - per_target[t][label(c)]["nonref_matched"]
            for t in targets)
    write_json(a.out, {"chroms": chroms, "cohort_members": a.cohort_members,
                       "T": sum(r["T"] for r in rows) // len(targets),
                       "per_chromosome": rows, "per_target": per_target, "median": median})

    print(f"median over {len(targets)} targets, {len(chroms)} chromosome(s)")
    print("metric                 " + "  ".join(f"{label(c):>12}" for c in a.ceilings))
    for k, fmt in (("altered", "12,.1f"), ("altered_fraction", "12.6f"),
                   ("nonref_altered", "12,.1f"), ("concordance", "12.6f"),
                   ("nonref_concordance", "12.6f"), ("eta", "12,.1f")):
        print(f"{k:<22} " + "  ".join(f"{median[label(c)][k]:{fmt}}" for c in a.ceilings))


if __name__ == "__main__":
    main()
