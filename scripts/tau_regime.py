"""What the universal total variation bound buys, genome-wide (Appendix Tables 2 and 3).

Theorem 1 bounds the distinguishability of any two input paths by tau, at a tilt
of arctanh(tau). This script measures, exactly, what that tilt does to a
genome-scale release: the expected number of altered alleles, the standard
deviation of that count over draws, and the divergence against the constructed
decoy with the accuracy ceiling it implies.

Across tau's whole numerical range the tilt moves a genome-wide release by a few
alleles out of millions, which is why the accuracy ceiling, not tau, is the
operating parameter the tool exposes.

  python scripts/tau_regime.py --target HG00096 --taus 0.99 0.999 0.9999 0.99999 \
      --out results/tau_regime.json

The published numbers were computed with weights inversely proportional to the
number of called donors; the tool itself uses uniform weights, which concentrate
the tilt slightly less.
"""
import argparse
import json
import math

import numpy as np

from common import AUTOSOMES, DEFAULT_PATHS, chain_lengths, load_chain, pool_map, shares, target_on_chain, write_json
from pathguard import calibrate, model

ARGS = None


def job(args):
    chrom, share = args
    d = f"{ARGS.cohort_root}/{chrom}"
    chain, _ = load_chain(d)
    p = target_on_chain(d, ARGS.target, chrom, ARGS.paths)
    q = calibrate.constructed_decoy(p)
    beta = (model.inv_support_beta(chain, share) if ARGS.weights == "inv_support"
            else model.uniform_beta(chain.T, share))
    out = {"chrom": chrom, "T": chain.T, "taus": {}}
    for tau in ARGS.taus:
        eta = math.atanh(tau) if tau > 0 else 0.0
        altered = variance = 0.0
        for h in (0, 1):
            x = p[:, h]
            psi = model.potentials(chain, x, beta, eta / 2.0)
            alpha, _ = model.forward(chain, psi)
            gamma = model.marginals(chain, alpha, psi)
            mism = ((chain.haps != x[None, :]) & (x[None, :] >= 0)).astype(np.float64)
            altered += float((gamma.T * mism).sum())
            variance += model.path_variance(chain, alpha, mism, gamma)
        out["taus"][str(tau)] = {"altered": altered, "variance": variance,
                                 "kl": calibrate.kl(chain, p, q, beta, eta)}
    return out


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.99, 0.999, 0.9999, 0.99999])
    ap.add_argument("--chroms", nargs="+", default=["all"],
                    help="chromosomes to measure; the budget always covers all 22")
    ap.add_argument("--weights", choices=["inv_support", "uniform"], default="inv_support")
    ap.add_argument("--cohort-root", default="data/cohort")
    ap.add_argument("--paths", default=DEFAULT_PATHS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ARGS = a = ap.parse_args()
    chroms = list(AUTOSOMES) if a.chroms == ["all"] else a.chroms
    # The release is genome-wide, so every chromosome's weight is its share of all
    # 22 autosomes even when only some of them are measured here.
    share = shares(chain_lengths(a.cohort_root, [None], AUTOSOMES, a.workers)[None])

    rows = pool_map(job, [(c, share[c]) for c in chroms], a.workers)
    total = {}
    for tau in a.taus:
        cells = [r["taus"][str(tau)] for r in rows]
        kl = math.fsum(c["kl"] for c in cells)
        total[str(tau)] = {
            "eta": math.atanh(tau) if tau > 0 else 0.0,
            "altered": math.fsum(c["altered"] for c in cells),
            "sd": math.sqrt(math.fsum(c["variance"] for c in cells)),
            "kl": kl, "ceiling": calibrate.pinsker_ceiling(kl),
            "pinsker_tv": min(1.0, math.sqrt(kl / 2.0)),
            "theorem_1_bound": (1.0 + tau) / 2.0}
    write_json(a.out, {"target": a.target, "weights": a.weights, "chroms": chroms,
                       "T": sum(r["T"] for r in rows), "per_chromosome": rows,
                       "genome": total})

    print(f"{a.target}, {sum(r['T'] for r in rows):,} positions, {a.weights} weights")
    print(f"{'tau':>9} {'eta':>9} {'altered':>14} {'sd':>10} {'KL':>12} "
          f"{'ceiling':>10} {'Theorem 1':>10}")
    for tau in a.taus:
        t = total[str(tau)]
        print(f"{tau:>9g} {t['eta']:>9.4f} {t['altered']:>14,.2f} {t['sd']:>10,.1f} "
              f"{t['kl']:>12.4e} {t['ceiling']:>10.6f} {t['theorem_1_bound']:>10.4f}")


if __name__ == "__main__":
    main()
