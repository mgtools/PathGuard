"""Draw a sanitized release of a diploid target path.

A diploid genome is one private input. Its two haplotypes are drawn as two
independent tilted chains, each at half the tilt, so that the diploid utility
u = (u_1 + u_2) / 2 stays in [0, 1] and one tilt covers the whole release.
Every released allele is copied from a called cohort haplotype.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from .model import (Chain, baseline_log_norm, forward, log_norm, potentials,
                    sample_states, seed_streams)


def release(chain: Chain, target: np.ndarray, beta: np.ndarray, eta: float, seed: int,
            key: str = "", base: float | None = None) -> tuple[np.ndarray, float]:
    """One draw from Q_{p,eta}. Returns the released alleles (T, 2) and log Z_p.

    `target` is (T, 2) allele indices on the chain (-1 = no call). `base` is the
    untilted chain's log normaliser; pass it when drawing many releases.
    """
    if base is None:
        base = baseline_log_norm(chain)
    streams = seed_streams(seed, key, 2)
    released = np.empty((chain.T, 2), dtype=np.int16)
    log_z = []
    for h in (0, 1):
        alpha, log_c = forward(chain, potentials(chain, target[:, h], beta, eta / 2.0))
        log_z.append(log_norm(log_c) - base)
        z = sample_states(chain, alpha, np.random.default_rng(streams[h]))
        released[:, h] = chain.haps[z, np.arange(chain.T)]
    return released, log_z[0] + log_z[1]


def log_z(chain: Chain, path: np.ndarray, beta: np.ndarray, eta: float, base: float) -> float:
    """log Z for a diploid input path at tilt eta (each haplotype at eta / 2)."""
    out = []
    for h in (0, 1):
        _, log_c = forward(chain, potentials(chain, path[:, h], beta, eta / 2.0))
        out.append(log_norm(log_c) - base)
    return out[0] + out[1]


def utility(released: np.ndarray, path: np.ndarray, beta: np.ndarray) -> float:
    """u(path, release): beta-weighted share of positions where the release carries
    the path's allele, averaged over the two haplotypes."""
    u = []
    for h in (0, 1):
        y, x = released[:, h], path[:, h]
        match = (y == x) & (x >= 0) & (y >= 0)
        u.append(float((match[None, :].astype(np.float64) @ beta)[0]))
    return 0.5 * (u[0] + u[1])


def write_vcf(out_path, chrom: str, positions, ref, alt, released: np.ndarray,
              sample: str) -> None:
    """Write a release as a one-sample phased VCF over the chain positions."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if out_path.suffix == ".gz" else open
    with opener(out_path, "wt") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n")
        for i in range(len(positions)):
            a, b = released[i]
            fh.write(f"{chrom}\t{positions[i]}\t.\t{ref[i]}\t{alt[i]}\t.\tPASS\t.\tGT\t{a}|{b}\n")
