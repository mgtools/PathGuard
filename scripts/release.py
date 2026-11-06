"""Draw the releases an experiment needs, one per target and accuracy ceiling.

Each release is written as the allele indices of the two released haplotypes.
With `--panmixer-format` the same release is also written over the full record
axis, with -1 at records outside the chain, which is the array PanMixer's
evaluators read.

  # the 44 cohort members, chr21, held out of their own donor panel
  python scripts/release.py --targets configs/cohort_members.txt --cohort-members \
      --etas results/etas_cohort_chr21.json --out data/releases/cohort \
      --panmixer-format --workers 24
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import (CEILINGS, DEFAULT_PATHS, label, load_chain, load_cohort, pool_map,
                    read_list, target_on_chain, write_json)
from pathguard import model, release

SEED_BASE = 20261012
ARGS = None


def release_seed(target: str, ceiling: float) -> int:
    """One stream per (target, ceiling), keyed on both so no two cells can share it."""
    h = int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "big")
    i = [label(c) for c in CEILINGS].index(label(ceiling))
    return int(np.random.SeedSequence([SEED_BASE, h, 2000 + i]).generate_state(1)[0])


def job(target):
    a = ARGS
    held = target if a.cohort_members else None
    d = f"{a.cohort_root}/{a.chrom}"
    chain, base = load_chain(d, held)
    in_chain = load_cohort(d, held).in_chain
    path = target_on_chain(d, target, a.chrom, a.paths, a.cohort_members)
    out = {"target": target, "chrom": a.chrom, "T": chain.T, "cells": {}}
    for c in a.ceilings:
        eta = a.etas[target]["ceilings"][label(c)]["eta"]
        beta = model.uniform_beta(chain.T, a.etas[target]["share"][a.chrom])
        seed = release_seed(target, c)
        y, log_z = release.release(chain, path, beta, eta, seed, key=a.chrom, base=base)
        called = path >= 0
        cell = {"eta": eta, "seed": seed, "log_z": log_z,
                "utility": release.utility(y, path, beta),
                "altered_fraction": float(((y != path) & called).sum() / called.sum())}
        point = f"acc_{c:.2f}".replace(".", "p")
        dest = Path(a.out) / target / point / a.chrom
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "released_path.npy", y)
        if a.panmixer_format:
            full = np.full((len(in_chain), 2), -1, dtype=np.int16)
            full[in_chain] = y
            np.save(dest / "new_haplotypes.npy", full)
        out["cells"][label(c)] = cell
    return out


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--etas", required=True, help="output of calibrate.py for these targets")
    ap.add_argument("--ceilings", type=float, nargs="+", default=list(CEILINGS))
    ap.add_argument("--cohort-members", action="store_true",
                    help="targets are cohort members: hold each out of its own donor panel")
    ap.add_argument("--chrom", default="chr21")
    ap.add_argument("--cohort-root", default="data/cohort")
    ap.add_argument("--paths", default=DEFAULT_PATHS)
    ap.add_argument("--panmixer-format", action="store_true",
                    help="also write the release over the full record axis")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True, help="release directory")
    ap.add_argument("--json", help="where to write the per-release record "
                                   "(default: <out>/releases.json)")
    ARGS = a = ap.parse_args()
    a.etas = json.load(open(a.etas))["targets"]
    targets = sorted(read_list(a.targets))

    rows = pool_map(job, targets, a.workers)
    write_json(a.json or f"{a.out}/releases.json",
               {"chrom": a.chrom, "cohort_members": a.cohort_members, "results": rows})
    for r in rows:
        print(r["target"], " ".join(f"{k}:{v['altered_fraction']:.6f}"
                                    for k, v in r["cells"].items()))


if __name__ == "__main__":
    main()
