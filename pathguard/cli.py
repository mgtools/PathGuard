"""Command-line interface: `pathguard {cohort,target,calibrate,release,genome}`."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import calibrate, cohort, model, release, target

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
CEILINGS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


def _load(cohort_dirs, target_paths, leave_out=None):
    """Chains, target paths on the chain, and beta shares for one or more chromosomes.

    With more than one chromosome the weights are split in proportion to chain
    length, so the whole set is one release under one budget.
    """
    chains, paths = [], []
    for d, t in zip(cohort_dirs, target_paths):
        c = cohort.Cohort(d)
        if leave_out:
            c = c.leave_out(leave_out)
        chains.append(model.make_chain(c.haplotypes, c.genetic_pos, c.positions, c.in_chain))
        paths.append(np.load(t)[c.in_chain])
    total = sum(ch.T for ch in chains)
    shares = [ch.T / total if len(chains) > 1 else 1.0 for ch in chains]
    betas = [model.uniform_beta(ch.T, s) for ch, s in zip(chains, shares)]
    return chains, paths, betas


def _kl(chains, paths, betas, eta):
    return math.fsum(calibrate.kl(ch, p, calibrate.constructed_decoy(p), b, eta)
                     for ch, p, b in zip(chains, paths, betas))


def solve(chains, paths, betas, ceiling, probes=None, refinements=2):
    """Tilt eta at which the target's KL against the constructed decoy meets the
    budget of the accuracy ceiling. Returns (eta, kl, achieved ceiling)."""
    if ceiling <= 0.5:
        return 0.0, 0.0, 0.5
    if not probes:
        probes = [0.5 * math.sqrt(sum(ch.T for ch in chains))]
    eta = calibrate.fit_eta({e: _kl(chains, paths, betas, e) for e in probes}, ceiling)
    kl = _kl(chains, paths, betas, eta)
    for _ in range(refinements):
        eta = calibrate.refine_eta(eta, kl, ceiling)
        kl = _kl(chains, paths, betas, eta)
    return eta, kl, calibrate.pinsker_ceiling(kl)


def expected_altered(chain, path, beta, eta):
    """Exact expected number of called target alleles the release changes, and the
    number of called target alleles (both haplotypes)."""
    out, called = 0.0, 0
    for h in (0, 1):
        x = path[:, h]
        psi = model.potentials(chain, x, beta, eta / 2.0)
        alpha, _ = model.forward(chain, psi)
        gamma = model.marginals(chain, alpha, psi)
        known = x >= 0
        out += float((gamma.T * ((chain.haps != x[None, :]) & known[None, :])).sum())
        called += int(known.sum())
    return out, called


def _chain_sites(cohort_dir, leave_out=None):
    """(pos, ref, alt) at the chain positions, in the order a release carries them."""
    d = Path(cohort_dir)
    keep = (cohort.Cohort(d).leave_out(leave_out).in_chain if leave_out
            else np.load(d / "in_chain.npy"))
    pos, ref, alt = target.load_sites(d / "sites.tsv")
    idx = np.flatnonzero(keep)
    return pos[idx], [ref[i] for i in idx], [alt[i] for i in idx]


def _draw(chain, path, beta, eta, seed, chrom, cohort_dir, out_path, sample, leave_out):
    """Draw one chromosome's release and write it as a phased one-sample VCF.

    The draw is keyed on the chromosome name as well as the seed, so one seed
    gives every chromosome its own stream and a chromosome's release does not
    depend on whether it was drawn alone or as part of a genome.
    """
    y, _ = release.release(chain, path, beta, eta, seed, key=chrom)
    pos, ref, alt = _chain_sites(cohort_dir, leave_out)
    release.write_vcf(out_path, chrom, pos, ref, alt, y, sample)
    called = path >= 0
    return int(((y != path) & called).sum()), int(called.sum())


# --- the four steps -----------------------------------------------------------

def cmd_cohort(a):
    vcf = a.vcf
    if a.gfa:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        cohort.deconstruct(a.gfa, out / "deconstruct.vcf", vg=a.vg, threads=a.gfa_threads)
        vcf = out / "cohort.vcf.gz"
        cohort.drop_sample(out / "deconstruct.vcf", vcf, a.drop_sample)
    print(json.dumps(cohort.build(vcf, a.gmap, a.out, gaps_table=a.gaps, chrom=a.chrom)))


def cmd_target(a):
    path = target.build(a.vcf, a.sample, a.cohort, a.out)
    called = (path >= 0).sum(axis=0)
    print(f"{a.sample}: {len(path)} records, called {called[0]} / {called[1]} per haplotype")


def cmd_calibrate(a):
    chains, paths, betas = _load(a.cohort, a.target, a.leave_out)
    print("ceiling\teta\tkl\tachieved_ceiling\texpected_altered_fraction")
    for c in a.ceiling:
        eta, kl, acc = solve(chains, paths, betas, c)
        altered = called = 0.0
        for ch, p, b in zip(chains, paths, betas):
            e, n = expected_altered(ch, p, b, eta)
            altered += e
            called += n
        print(f"{c:.2f}\t{eta:.6g}\t{kl:.6g}\t{acc:.6f}\t{altered / called:.6f}")


def cmd_release(a):
    chains, paths, betas = _load([a.cohort], [a.target], a.leave_out)
    eta = a.eta
    if eta is None:
        eta, kl, acc = solve(chains, paths, betas, a.ceiling)
        print(f"ceiling {a.ceiling}: eta = {eta:.6g} (KL {kl:.6g}, achieved ceiling {acc:.6f})",
              file=sys.stderr)
    altered, called = _draw(chains[0], paths[0], betas[0], eta, a.seed, a.chrom,
                            a.cohort, a.out, a.sample, a.leave_out)
    print(f"wrote {a.out}: {chains[0].T} positions, {altered / called:.4f} of called "
          f"target alleles altered", file=sys.stderr)


def cmd_genome(a):
    chroms = a.chroms or [c for c in AUTOSOMES
                          if (Path(a.cohort_root) / c / "in_chain.npy").exists()]
    if not chroms:
        raise SystemExit(f"no cohort directories under {a.cohort_root}: expected "
                         f"{a.cohort_root}/chr1 ... /chr22, each one written by "
                         f"`pathguard cohort --out {a.cohort_root}/<chrom>`")
    dirs = [Path(a.cohort_root) / c for c in chroms]
    paths = [Path(a.target_dir) / f"{c}.npy" for c in chroms]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("no target path for: " + ", ".join(missing))

    chains, targets, betas = _load(dirs, paths, a.leave_out)
    total = sum(ch.T for ch in chains)
    eta, kl, acc = solve(chains, targets, betas, a.ceiling)
    print(f"{', '.join(chroms)}: {total} chain positions")
    print(f"ceiling {a.ceiling}: eta = {eta:.6g} (KL {kl:.6g}, achieved ceiling {acc:.6f})")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("chrom\tchain\tshare\taltered_fraction")
    altered = called = 0
    for chrom, d, ch, p, b in zip(chroms, dirs, chains, targets, betas):
        n_alt, n_called = _draw(ch, p, b, eta, a.seed, chrom, d, out / f"{chrom}.vcf.gz",
                                a.sample, a.leave_out)
        altered += n_alt
        called += n_called
        print(f"{chrom}\t{ch.T}\t{ch.T / total:.6f}\t{n_alt / n_called:.6f}")
    print(f"genome\t{total}\t1.000000\t{altered / called:.6f}")


# --- the command line ---------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(prog="pathguard", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("cohort", help="build one chromosome's cohort arrays",
                       description="Build one chromosome's cohort arrays from the graph "
                                   "and the genetic map. Run it once per chromosome.")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--vcf", metavar="FILE",
                   help="phased cohort VCF, one record per variant site")
    g.add_argument("--gfa", metavar="FILE",
                   help="graph to build that VCF from, with `vg deconstruct`")
    s.add_argument("--vg", metavar="PATH", default="vg",
                   help="the vg binary, with --gfa (default: vg on $PATH)")
    s.add_argument("--gfa-threads", metavar="N", type=int, default=16,
                   help="threads for vg deconstruct, with --gfa (default 16)")
    s.add_argument("--drop-sample", metavar="NAME", default="chm13",
                   help="a graph sample to leave out of the cohort (default chm13)")
    s.add_argument("--gmap", metavar="FILE", required=True,
                   help="genetic map for this chromosome, columns pos chr cM")
    s.add_argument("--gaps", metavar="FILE",
                   help="UCSC gap table; assembly gaps get no genetic distance")
    s.add_argument("--chrom", metavar="NAME", required=True,
                   help="chromosome name as the gap table spells it, e.g. chr21")
    s.add_argument("--out", metavar="DIR", required=True,
                   help="directory to write the arrays into")
    s.set_defaults(func=cmd_cohort)

    s = sub.add_parser("target", help="project a phased genome onto the cohort records",
                       description="Re-express a target genome on the cohort's record "
                                   "axis, which is what the mechanism reads.")
    s.add_argument("--vcf", metavar="FILE", required=True,
                   help="phased GRCh38 VCF holding the target, e.g. a 1000 Genomes panel")
    s.add_argument("--sample", metavar="NAME", required=True,
                   help="the sample column in --vcf that is the target")
    s.add_argument("--cohort", metavar="DIR", required=True,
                   help="cohort directory written by `pathguard cohort`")
    s.add_argument("--out", metavar="FILE", required=True,
                   help="output .npy: one allele index per record per haplotype")
    s.set_defaults(func=cmd_target)

    s = sub.add_parser("calibrate", help="solve the tilt for one or more accuracy ceilings",
                       description="Solve the tilt that meets each accuracy ceiling. "
                                   "Several chromosomes are treated as one release and "
                                   "share one budget, split by chain length.")
    s.add_argument("--cohort", metavar="DIR", nargs="+", required=True,
                   help="cohort directory, one per chromosome")
    s.add_argument("--target", metavar="FILE", nargs="+", required=True,
                   help="target .npy, one per chromosome, in the same order as --cohort")
    s.add_argument("--ceiling", metavar="A", type=float, nargs="+", default=CEILINGS,
                   help="accuracy ceilings to solve, each in [0.5, 1) "
                        "(default 0.55 0.60 0.65 0.70 0.75 0.80)")
    s.add_argument("--leave-out", metavar="NAME",
                   help="a cohort member to hold out of the donor panel, when the "
                        "target is itself a member")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("release", help="draw one chromosome's release",
                       description="Draw one release for one chromosome and write it "
                                   "as a phased one-sample VCF.")
    s.add_argument("--cohort", metavar="DIR", required=True,
                   help="cohort directory written by `pathguard cohort`")
    s.add_argument("--target", metavar="FILE", required=True,
                   help="target .npy written by `pathguard target`")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--ceiling", metavar="A", type=float,
                   help="accuracy ceiling in [0.5, 1); the tilt is solved for it")
    g.add_argument("--eta", metavar="ETA", type=float,
                   help="a tilt already solved, from `pathguard calibrate`")
    s.add_argument("--seed", metavar="N", type=int, required=True,
                   help="seed for this draw; the same seed gives the same release")
    s.add_argument("--chrom", metavar="NAME", default="chr21",
                   help="CHROM written into the output VCF, and part of the draw's key")
    s.add_argument("--sample", metavar="NAME", default="RELEASE",
                   help="sample name written into the output VCF")
    s.add_argument("--out", metavar="FILE", required=True,
                   help="output VCF, .vcf or .vcf.gz")
    s.add_argument("--leave-out", metavar="NAME",
                   help="a cohort member to hold out of the donor panel, when the "
                        "target is itself a member")
    s.set_defaults(func=cmd_release)

    s = sub.add_parser("genome", help="calibrate and release a whole genome",
                       description="Release a whole genome under one budget: one tilt "
                                   "solved across every chromosome, each chromosome "
                                   "weighted by its share of the chain, then one VCF "
                                   "drawn per chromosome.")
    s.add_argument("--cohort-root", metavar="DIR", required=True,
                   help="directory holding one cohort directory per chromosome, "
                        "named chr1 ... chr22")
    s.add_argument("--target-dir", metavar="DIR", required=True,
                   help="directory holding one target .npy per chromosome, "
                        "named chr1.npy ... chr22.npy")
    s.add_argument("--chroms", metavar="NAME", nargs="+",
                   help="chromosomes to release (default: every one found under "
                        "--cohort-root)")
    s.add_argument("--ceiling", metavar="A", type=float, required=True,
                   help="accuracy ceiling in [0.5, 1) for the whole genome")
    s.add_argument("--seed", metavar="N", type=int, required=True,
                   help="seed for the draw; each chromosome takes its own stream from it")
    s.add_argument("--sample", metavar="NAME", default="RELEASE",
                   help="sample name written into the output VCFs")
    s.add_argument("--out", metavar="DIR", required=True,
                   help="directory to write chr1.vcf.gz ... chr22.vcf.gz into")
    s.add_argument("--leave-out", metavar="NAME",
                   help="a cohort member to hold out of the donor panel, when the "
                        "target is itself a member")
    s.set_defaults(func=cmd_genome)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
