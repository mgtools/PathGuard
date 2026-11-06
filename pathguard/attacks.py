"""Attacks used to evaluate releases of an external target.

Each attack sees one release y and two candidate genomes, the target p and a
decoy q, and decides which one produced y. A statistic > 0 votes for p, < 0 for
q, and a tie counts as half a vote either way.

- Optimal likelihood-ratio distinguisher: knows the mechanism and the tilt, and
  thresholds log Q_p(y) / Q_q(y) = eta (u(p,y) - u(q,y)) - (log Z_p - log Z_q).
  By Neyman-Pearson no test does better for a given pair.
- Rarity-weighted genotype linkage (PanMixer's gap score): sums -log P(genotype)
  over bi-allelic sites where the release's genotype equals the candidate's.
- Block-restricted likelihood-ratio distinguisher: the likelihood-ratio statistic
  restricted to one block of consecutive chain positions, reporting the best block.
"""
from __future__ import annotations

import math

import numpy as np

BLOCK_SIZES = (512, 4096, 32768)


def lr_statistic(eta: float, u_p: float, u_q: float, log_z_p: float, log_z_q: float) -> float:
    return eta * (u_p - u_q) - (log_z_p - log_z_q)


def balanced_accuracy(stat_p, stat_q) -> dict:
    """Accuracy of the sign rule given the statistic on releases from p and from q."""
    lp, lq = np.asarray(stat_p), np.asarray(stat_q)
    tpr = ((lp > 0).sum() + 0.5 * (lp == 0).sum()) / len(lp)
    tnr = ((lq < 0).sum() + 0.5 * (lq == 0).sum()) / len(lq)
    se = 0.5 * math.sqrt(tpr * (1 - tpr) / len(lp) + tnr * (1 - tnr) / len(lq))
    return {"balanced": float(0.5 * tpr + 0.5 * tnr), "se": se,
            "tpr": float(tpr), "tnr": float(tnr), "n": len(lp)}


class RarityLinkage:
    """PanMixer's rarity-weighted genotype linkage score, with allele frequencies
    taken from the cohort haplotypes at bi-allelic chain positions."""

    def __init__(self, haps: np.ndarray):
        called = haps >= 0
        n_called = called.sum(axis=0)
        lo = np.where(called, haps, np.int16(1 << 14)).min(axis=0)
        hi = np.where(called, haps, np.int16(-1)).max(axis=0)
        only_two = ((~called) | (haps == lo[None, :]) | (haps == hi[None, :])).all(axis=0)
        self.mask = (hi > lo) & only_two & (n_called > 0)
        self.alt = hi[self.mask]
        alt_count = ((haps[:, self.mask] == self.alt[None, :]) & called[:, self.mask]).sum(axis=0)
        p = alt_count / n_called[self.mask]
        q = 1.0 - p
        # -log P(genotype) under Hardy-Weinberg, indexed by the number of ALT alleles
        self.weight = np.stack([-np.log(q * q + 1e-10),
                                -np.log(2.0 * p * q + 1e-10),
                                -np.log(p * p + 1e-10)])

    def _genotype(self, path):
        sub = path[self.mask]
        valid = (sub >= 0).all(axis=1)
        g = (sub[:, 0] == self.alt).astype(np.int8) + (sub[:, 1] == self.alt).astype(np.int8)
        return g, valid

    def score(self, y: np.ndarray, candidate: np.ndarray) -> float:
        gy, vy = self._genotype(y)
        gc, vc = self._genotype(candidate)
        w = self.weight[gc, np.arange(len(gc))]
        return float((w * ((gy == gc) & vy & vc)).sum())

    def statistic(self, y, p, q) -> float:
        return self.score(y, p) - self.score(y, q)


def block_statistics(y, p, q, beta, eta, sizes=BLOCK_SIZES) -> dict[int, np.ndarray]:
    """Per-block likelihood-ratio contributions eta/2 * beta_t * (m_p - m_q), where
    m_x counts the release haplotypes that carry x's allele at t."""
    mp = np.zeros(len(beta))
    mq = np.zeros(len(beta))
    for h in (0, 1):
        yh, ph, qh = y[:, h], p[:, h], q[:, h]
        mp += (yh == ph) & (ph >= 0) & (yh >= 0)
        mq += (yh == qh) & (qh >= 0) & (yh >= 0)
    contrib = (eta / 2.0) * beta * (mp - mq)
    return {s: np.add.reduceat(contrib, np.arange(0, len(beta), s)).astype(np.float32)
            for s in sizes}


def best_block_accuracy(blocks_p: np.ndarray, blocks_q: np.ndarray) -> dict:
    """blocks_x is (n_trials, n_blocks). Accuracy of the best single block."""
    acc = [balanced_accuracy(blocks_p[:, b], blocks_q[:, b]) for b in range(blocks_p.shape[1])]
    best = int(np.argmax([a["balanced"] for a in acc]))
    return {**acc[best], "block": best, "n_blocks": blocks_p.shape[1]}
