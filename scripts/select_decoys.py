"""Pick the real-genome decoys for each external target.

Concordance between two diploid paths is the fraction of chain positions, among
those where both genomes are called on both haplotypes, at which the two
genotypes agree on both haplotypes. From the candidate pool each target gets
its most concordant genome (nearest), its least concordant (farthest), and one
drawn at random (redrawn if it equals the nearest).

  # the 3,075-genome external panel (Tables 1 and 4)
  python scripts/select_decoys.py --targets configs/external_targets.txt \
      --pool configs/decoy_pool.txt --out results/decoys_panel.json
  # the other nine targets (Table 9)
  python scripts/select_decoys.py --targets configs/external_targets.txt \
      --pool configs/external_targets.txt --out results/decoys_targets.json
"""
import argparse
import hashlib

import numpy as np

from common import DEFAULT_PATHS, load_path, pool_map, read_list, write_json

DECOY_SEED = 20261010
ARGS = None


def concordance(a: np.ndarray, b: np.ndarray) -> float:
    both = (a >= 0).all(axis=1) & (b >= 0).all(axis=1)
    return float((a[both] == b[both]).all(axis=1).mean())


def decoy_rng(target: str) -> np.random.Generator:
    h = int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "big")
    return np.random.default_rng([DECOY_SEED, h])


def score_candidate(cand):
    path = load_path(ARGS.paths, cand, ARGS.chrom, ARGS.in_chain)
    return [concordance(t, path) for t in ARGS.target_paths]


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--pool", required=True, help="file listing candidate decoy sample ids")
    ap.add_argument("--chrom", default="chr21")
    ap.add_argument("--cohort-root", default="data/cohort")
    ap.add_argument("--paths", default=DEFAULT_PATHS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ARGS = a = ap.parse_args()

    a.in_chain = np.load(f"{a.cohort_root}/{a.chrom}/in_chain.npy")
    targets = sorted(read_list(a.targets))
    pool = sorted(set(read_list(a.pool)))
    a.target_paths = [load_path(a.paths, t, a.chrom, a.in_chain) for t in targets]
    scores = dict(zip(pool, pool_map(score_candidate, pool, a.workers)))

    out = {}
    for i, t in enumerate(targets):
        cands = [c for c in pool if c != t]
        conc = {c: scores[c][i] for c in cands}
        order = sorted(cands, key=conc.get)
        nearest, farthest = order[-1], order[0]
        rng = decoy_rng(t)
        random = cands[rng.integers(len(cands))]
        while random == nearest:
            random = cands[rng.integers(len(cands))]
        out[t] = {"nearest": nearest, "random": random, "farthest": farthest,
                  "concordance": {"nearest": conc[nearest], "random": conc[random],
                                  "farthest": conc[farthest]}}
        print(t, nearest, random, farthest)
    write_json(a.out, out)


if __name__ == "__main__":
    main()
