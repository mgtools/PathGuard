"""Attack experiment on external targets (Tables 1 and 4).

For every target, accuracy ceiling and draw, one release is drawn from the
target's distribution Q_p and, for each decoy condition, one from the decoy's
distribution Q_q at the same tilt. Each attack scores both releases; its
balanced accuracy is pooled over targets and draws.

Decoy conditions: the constructed decoy q* (differs at every called position)
and the nearest, random and farthest real genomes from `select_decoys.py`.

Paper runs (seeds as published):
  python scripts/attack.py --attacks lr --seed 50000 --etas results/etas_external_chr21.json \
      --decoys results/decoys_panel.json --draws 100 --out results/attack_lr.json
  python scripts/attack.py --attacks linkage block --seed 70000 34 --etas ... --out results/attack_linkage_block.json
"""
import argparse
import json

import numpy as np

from common import CEILINGS, DEFAULT_PATHS, label, load_chain, load_path, pool_map, read_list, write_json
from pathguard import attacks, calibrate, model, release

CONDITIONS = ("constructed", "nearest", "random", "farthest")
ARGS = None
_cache = {}


def setup():
    """Per-process state: chain, target and decoy paths, linkage weights, log Z of decoys."""
    if not _cache:
        a = ARGS
        d = f"{a.cohort_root}/{a.chrom}"
        chain, base = load_chain(d)
        in_chain = np.load(f"{d}/in_chain.npy")
        paths = {}
        for t in a.targets:
            paths[t] = load_path(a.paths, t, a.chrom, in_chain)
            for c in CONDITIONS[1:]:
                s = a.decoys[t][c]
                if s not in paths:
                    paths[s] = load_path(a.paths, s, a.chrom, in_chain)
        _cache.update(chain=chain, base=base, paths=paths, beta=model.uniform_beta(chain.T),
                      linkage=attacks.RarityLinkage(chain.haps) if "linkage" in a.attacks else None)
    return _cache


def decoy_path(st, target, cond):
    p = st["paths"][target]
    return calibrate.constructed_decoy(p) if cond == "constructed" else st["paths"][ARGS.decoys[target][cond]]


def log_z(st, path, eta, key):
    if key not in st:
        st[key] = release.log_z(st["chain"], path, st["beta"], eta, st["base"])
    return st[key]


def trial(job):
    ti, ai, i = job
    st = setup()
    t, c = ARGS.targets[ti], CEILINGS[ai]
    eta = ARGS.etas[t]["ceilings"][label(c)]["eta"]
    seed = int(np.random.SeedSequence([*ARGS.seed, ti, 2000 + ai, i]).generate_state(1)[0])
    chain, beta, p = st["chain"], st["beta"], st["paths"][t]
    y, lz_p = release.release(chain, p, beta, eta, seed, base=st["base"])
    out = {"target": t, "ceiling": c, "seed": seed, "conditions": {}}
    for k, cond in enumerate(CONDITIONS):
        q = decoy_path(st, t, cond)
        yq, _ = release.release(chain, q, beta, eta, seed ^ (0x5EED * (1 + k)), base=st["base"])
        cell = {}
        if "lr" in ARGS.attacks:
            lz_q = log_z(st, q, eta, ("logz", t, c, cond))
            cell["lr"] = [attacks.lr_statistic(eta, release.utility(r, p, beta),
                                               release.utility(r, q, beta), lz_p, lz_q)
                          for r in (y, yq)]
        if "linkage" in ARGS.attacks:
            cell["linkage"] = [st["linkage"].statistic(r, p, q) for r in (y, yq)]
        if "block" in ARGS.attacks:
            cell["block"] = [attacks.block_statistics(r, p, q, beta, eta) for r in (y, yq)]
        out["conditions"][cond] = cell
    return out


def summarize(trials):
    acc = {}
    for att in ARGS.attacks:
        acc[att] = {}
        for cond in CONDITIONS:
            acc[att][cond] = {}
            for c in CEILINGS:
                cells = [tr["conditions"][cond][att] for tr in trials if tr["ceiling"] == c]
                if att == "block":
                    acc[att][cond][label(c)] = {
                        str(s): attacks.best_block_accuracy(
                            np.array([x[0][s] for x in cells]), np.array([x[1][s] for x in cells]))
                        for s in attacks.BLOCK_SIZES}
                else:
                    acc[att][cond][label(c)] = attacks.balanced_accuracy(
                        [x[0] for x in cells], [x[1] for x in cells])
    return acc


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="configs/external_targets.txt")
    ap.add_argument("--etas", required=True, help="output of calibrate.py for these targets")
    ap.add_argument("--decoys", required=True, help="output of select_decoys.py")
    ap.add_argument("--attacks", nargs="+", default=["lr", "linkage", "block"],
                    choices=["lr", "linkage", "block"])
    ap.add_argument("--seed", type=int, nargs="+", required=True,
                    help="run seed (one or more integers)")
    ap.add_argument("--draws", type=int, default=100, help="draws per target and ceiling")
    ap.add_argument("--only", nargs="+", help="restrict to these targets (seeds are unchanged)")
    ap.add_argument("--chrom", default="chr21")
    ap.add_argument("--cohort-root", default="data/cohort")
    ap.add_argument("--paths", default=DEFAULT_PATHS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-trials", action="store_true",
                    help="store every trial's likelihood-ratio and linkage statistics")
    ap.add_argument("--out", required=True)
    ARGS = a = ap.parse_args()
    a.targets = sorted(read_list(a.targets))
    a.etas = json.load(open(a.etas))["targets"]
    a.decoys = json.load(open(a.decoys))

    keep = set(a.only or a.targets)
    jobs = [(ti, ai, i) for ai in range(len(CEILINGS)) for ti, t in enumerate(a.targets)
            if t in keep for i in range(a.draws)]
    trials = pool_map(trial, jobs, a.workers)
    acc = summarize(trials)
    out = {"attacks": a.attacks, "seed": a.seed, "draws": a.draws, "targets": sorted(keep),
           "decoys": {t: a.decoys[t] for t in keep}, "accuracy": acc}
    if a.keep_trials:
        for tr in trials:
            for cell in tr["conditions"].values():
                cell.pop("block", None)
        out["trials"] = trials
    write_json(a.out, out)
    for att in a.attacks:
        print(f"\n{att}: balanced accuracy, pooled over {len(keep)} targets x {a.draws} draws")
        print("condition    " + " ".join(label(c) for c in CEILINGS))
        for cond in CONDITIONS:
            vals = [acc[att][cond][label(c)] for c in CEILINGS]
            vals = [v["512"] if att == "block" else v for v in vals]
            print(f"{cond:<12} " + " ".join(f"{v['balanced']:.4f}" for v in vals))


if __name__ == "__main__":
    main()
