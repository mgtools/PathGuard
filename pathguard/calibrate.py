"""Exact divergences between release distributions, and the per-target tilt.

For a fixed tilt eta the log-likelihood ratio of a release y between inputs p and
q is eta (u(p,y) - u(q,y)) - log(Z_p / Z_q) (Eq. 3), so

    KL(Q_p || Q_q) = eta * E_p[u(p,Y) - u(q,Y)] - (log Z_p - log Z_q).

Both terms are exact: E_p[...] comes from the posterior state marginals of the
tilted chain, and log Z_p - log Z_q is the difference of the forward passes'
per-position normalisers (the baseline's normaliser cancels). A diploid release
runs each haplotype as its own chain at eta / 2, so its KL is the sum of the two
haplotype KLs.

The operating point is an accuracy ceiling a. Pinsker's inequality turns a KL
budget of 2 (2a - 1)^2 into an equal-prior accuracy bound of a (Eq. 4), and the
tilt eta_{p,a} is the one at which the target's KL against the constructed decoy
q* meets that budget.
"""
from __future__ import annotations

import math

import numpy as np

from .model import Chain, forward, marginals, potentials


def constructed_decoy(path: np.ndarray) -> np.ndarray:
    """q*: differs from the target at every called position (ref <-> alt; alt k -> ref)."""
    q = path.copy()
    called = q >= 0
    q[called] = np.where(q[called] == 0, 1, 0)
    return q


def kl_budget(ceiling: float) -> float:
    return 2.0 * (2.0 * ceiling - 1.0) ** 2


def pinsker_ceiling(kl: float) -> float:
    """Equal-prior accuracy bound implied by a KL divergence (Eq. 4)."""
    return 0.5 * (1.0 + min(1.0, math.sqrt(max(kl, 0.0) / 2.0)))


def kl_terms(chain: Chain, p: np.ndarray, q: np.ndarray, beta: np.ndarray,
             eta: float) -> tuple[list[np.ndarray], float]:
    """The two pieces of KL(Q_p || Q_q) for a diploid input: per-position log-normaliser
    differences, and E_p[u(p,Y) - u(q,Y)]."""
    h = eta / 2.0
    dlogc, d_eu = [], 0.0
    for col in (0, 1):
        pc, qc = p[:, col], q[:, col]
        psi_p = potentials(chain, pc, beta, h)
        alpha_p, logc_p = forward(chain, psi_p)
        _, logc_q = forward(chain, potentials(chain, qc, beta, h))
        dlogc.append(logc_p - logc_q)
        gamma = marginals(chain, alpha_p, psi_p)
        mp = (chain.haps == pc[None, :]) & (pc[None, :] >= 0)
        mq = (chain.haps == qc[None, :]) & (qc[None, :] >= 0)
        diff = mp.astype(np.int8) - mq.astype(np.int8)
        d_eu += 0.5 * float((gamma * diff.T).sum(axis=1) @ beta)
    return dlogc, d_eu


def kl(chain: Chain, p: np.ndarray, q: np.ndarray, beta: np.ndarray, eta: float) -> float:
    """KL(Q_p || Q_q) for diploid inputs p, q of shape (T, 2), at tilt eta."""
    if eta == 0.0:
        return 0.0
    dlogc, d_eu = kl_terms(chain, p, q, beta, eta)
    d_logz = math.fsum(math.fsum(t.tolist()) for t in dlogc)
    return eta * d_eu - d_logz


def fit_eta(kl_at: dict[float, float], ceiling: float) -> float:
    """Tilt for a ceiling from KL measured at probe tilts, using KL ~ c eta^2.

    KL has no linear term in eta, so c = KL / eta^2 is nearly constant over the
    range used; c is averaged over the probes.
    """
    if ceiling <= 0.5:
        return 0.0
    c_each = [k / (e * e) for e, k in sorted(kl_at.items()) if e > 0]
    c = sum(c_each) / len(c_each)
    tv = 2.0 * ceiling - 1.0
    return math.sqrt(2.0 * tv * tv / c)


def refine_eta(eta: float, kl_value: float, ceiling: float) -> float:
    """One multiplicative correction of eta towards the KL budget."""
    return eta * math.sqrt(kl_budget(ceiling) / kl_value) if kl_value > 0 else eta
