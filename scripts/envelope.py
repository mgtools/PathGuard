"""Exact KL of each external target against every decoy condition (Table 9).

At each target's calibrated tilt, KL(Q_p || Q_q) is computed exactly for the
constructed decoy and for the nearest, random and farthest real decoys. The
constructed decoy sets the tilt, so it sits at the ceiling's KL budget; the
question is whether every real decoy stays below it.

  python scripts/envelope.py --etas results/etas_external_chr21.json \
      --decoys results/decoys_targets.json --out results/envelope_chr21.json
"""
import argparse
import json
import statistics

from common import CEILINGS, DEFAULT_PATHS, label, load_chain, load_cohort, load_path, pool_map, read_list, write_json
from pathguard import calibrate, model

CONDITIONS = ("constructed", "nearest", "random", "farthest")
ARGS = None


def job(args):
    t, c = args
    d = f"{ARGS.cohort_root}/{ARGS.chrom}"
    chain, _ = load_chain(d)
    ic = load_cohort(d).in_chain
    p = load_path(ARGS.paths, t, ARGS.chrom, ic)
    beta = model.uniform_beta(chain.T)
    eta = ARGS.etas[t]["ceilings"][label(c)]["eta"]
    out = {}
    for cond in CONDITIONS:
        q = (calibrate.constructed_decoy(p) if cond == "constructed"
             else load_path(ARGS.paths, ARGS.decoys[t][cond], ARGS.chrom, ic))
        out[cond] = calibrate.kl(chain, p, q, beta, eta)
    return {"target": t, "ceiling": label(c), "eta": eta, "kl": out}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="configs/external_targets.txt")
    ap.add_argument("--etas", required=True)
    ap.add_argument("--decoys", required=True)
    ap.add_argument("--chrom", default="chr21")
    ap.add_argument("--cohort-root", default="data/cohort")
    ap.add_argument("--paths", default=DEFAULT_PATHS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ARGS = a = ap.parse_args()
    a.etas = json.load(open(a.etas))["targets"]
    a.decoys = json.load(open(a.decoys))
    targets = sorted(read_list(a.targets))

    rows = pool_map(job, [(t, c) for t in targets for c in CEILINGS if c > 0.5], a.workers)
    summary = {}
    print("ceiling  " + "  ".join(f"KL {c:<11}" for c in CONDITIONS) + "  worst real/constructed")
    for c in CEILINGS[1:]:
        cell = [r for r in rows if r["ceiling"] == label(c)]
        med = {k: statistics.median(r["kl"][k] for r in cell) for k in CONDITIONS}
        ratio = max(max(r["kl"][k] for k in CONDITIONS[1:]) / r["kl"]["constructed"] for r in cell)
        summary[label(c)] = {"median_kl": med, "median_ceiling":
                             {k: calibrate.pinsker_ceiling(v) for k, v in med.items()},
                             "worst_real_over_constructed": ratio,
                             "max_real_ceiling": max(calibrate.pinsker_ceiling(r["kl"][k])
                                                     for r in cell for k in CONDITIONS[1:])}
        print(f"{label(c)}     " + "  ".join(f"{med[k]:<14.6f}" for k in CONDITIONS) + f"  {ratio:.6f}")
    write_json(a.out, {"rows": rows, "summary": summary})


if __name__ == "__main__":
    main()
