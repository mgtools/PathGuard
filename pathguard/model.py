"""Li-Stephens copying model over the cohort haplotypes, and its tilted version.

The baseline R_D is a hidden Markov chain whose hidden state at each chain
position is the donor haplotype being copied. Between adjacent positions the
chain stays on its donor or resamples one uniformly, with probability set by the
genetic distance (Eq. 1 in the paper):

    d_t   = (cM_{t+1} - cM_t) * Ne * r
    s_t   = 1 - exp(-d_t / K)                      # resample probability
    A_t   = (1 - s_t) I + (s_t / K) J

A donor with no call at a position cannot be occupied there. The release tilts
this chain towards donors that carry the target's allele: the potential of donor
j at position t is exp(eta * beta_t * [h_{t,j} == p_t]). Releases are drawn by
forward filtering and backward sampling (Algorithm 1).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

NE = 10_000.0   # effective population size (PanMixer's published value)
R = 1.26        # scaling factor (PanMixer's published value)


def switch_mass(genetic_pos: np.ndarray, n_states: int) -> np.ndarray:
    """s_t for every step of the chain, from genetic positions in cM."""
    delta = np.diff(np.asarray(genetic_pos, dtype=np.float64))
    if (delta < 0).any():
        raise ValueError("genetic positions must be non-decreasing")
    d = delta * NE * R
    return 1.0 - np.exp(-d / n_states)


@dataclass
class Chain:
    """The cohort HMM restricted to the chain positions.

    haps[j, t] is donor j's allele at chain position t (-1 = no call).
    """
    haps: np.ndarray        # (K, T) int16
    switch: np.ndarray      # (T-1,) float64
    positions: np.ndarray   # (T,) int64, bp

    @property
    def K(self) -> int:
        return self.haps.shape[0]

    @property
    def T(self) -> int:
        return self.haps.shape[1]

    @property
    def allowed(self) -> np.ndarray:
        return self.haps != -1


def make_chain(haplotypes, genetic_pos, positions, in_chain) -> Chain:
    """Restrict the cohort arrays to the chain mask and build the transitions."""
    haps = np.ascontiguousarray(haplotypes[:, in_chain])
    if not (haps != -1).any(axis=0).all():
        raise ValueError("a chain position has no called donor")
    cm = np.asarray(genetic_pos)[in_chain]
    return Chain(haps=haps, switch=switch_mass(cm, haps.shape[0]),
                 positions=np.asarray(positions)[in_chain].astype(np.int64))


def uniform_beta(T: int, share: float = 1.0) -> np.ndarray:
    """beta_t = share / T. With share = 1 the utility lies in [0, 1].

    For a genome-wide release each chromosome gets share = T_c / sum_c T_c so
    the weights still sum to one over the whole genome.
    """
    w = np.ones(T, dtype=np.float64)
    return w / w.sum() * share


def inv_support_beta(chain: Chain, share: float = 1.0) -> np.ndarray:
    """beta_t proportional to 1 / (number of donors called at t).

    Only used to reproduce the tau-regime numbers (Appendix Table 3); the
    released tool uses uniform weights.
    """
    w = 1.0 / chain.allowed.sum(axis=0).astype(np.float64)
    return w / w.sum() * share


def potentials(chain: Chain, target: np.ndarray, beta: np.ndarray, eta: float) -> np.ndarray:
    """(T, K) potentials psi_t(j) = exp(eta * beta_t * [h_tj == p_t]), zero where j has no call.

    `target` is one haplotype, (T,) allele indices with -1 for no call. A
    position where the target has no call gets no tilt.
    """
    match = (chain.haps == target[None, :]) & (target[None, :] >= 0)
    psi = np.exp(eta * (match * beta[None, :]))
    psi *= chain.allowed
    return np.ascontiguousarray(psi.T)


def forward(chain: Chain, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised forward pass. Returns alpha (T, K) and log c_t (T,)."""
    T, K = psi.shape
    alpha = np.empty((T, K), dtype=np.float64)
    log_c = np.empty(T, dtype=np.float64)
    r = np.full(K, 1.0 / K) * psi[0]
    c = r.sum()
    alpha[0] = r / c
    log_c[0] = math.log(c)
    s = chain.switch
    for t in range(1, T):
        prev = alpha[t - 1]
        r = ((1.0 - s[t - 1]) * prev + (s[t - 1] / K) * prev.sum()) * psi[t]
        c = r.sum()
        if c <= 0:
            raise ValueError(f"chain position {t} (bp {chain.positions[t]}) has zero mass")
        alpha[t] = r / c
        log_c[t] = math.log(c)
    return alpha, log_c


def log_norm(log_c: np.ndarray) -> float:
    return math.fsum(log_c)


def baseline_log_norm(chain: Chain) -> float:
    """sum_t log c_t of the untilted chain. log Z_p is the tilted sum minus this."""
    psi = np.ascontiguousarray(chain.allowed.T.astype(np.float64))
    return log_norm(forward(chain, psi)[1])


def _categorical(p: np.ndarray, rng: np.random.Generator) -> int:
    cdf = np.cumsum(p)
    i = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
    return min(i, len(p) - 1)


def sample_states(chain: Chain, alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Backward sampling: one donor path z_1..z_T from the tilted posterior."""
    T, K = alpha.shape
    s = chain.switch
    z = np.empty(T, dtype=np.int32)
    z[T - 1] = _categorical(alpha[T - 1], rng)
    for t in range(T - 2, -1, -1):
        w = alpha[t] * (s[t] / K)
        w[z[t + 1]] += alpha[t][z[t + 1]] * (1.0 - s[t])
        z[t] = _categorical(w / w.sum(), rng)
    return z


def marginals(chain: Chain, alpha: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Posterior state marginals gamma_t(k) = P(Z_t = k), by a backward pass."""
    T, K = alpha.shape
    s = chain.switch
    b = np.empty((T, K), dtype=np.float64)
    b[T - 1] = 1.0 / K
    for t in range(T - 2, -1, -1):
        v = psi[t + 1] * b[t + 1]
        r = (1.0 - s[t]) * v + (s[t] / K) * v.sum()
        b[t] = r / r.sum()
    g = alpha * b
    return g / g.sum(axis=1, keepdims=True)


def path_variance(chain: Chain, alpha: np.ndarray, indicator: np.ndarray,
                  gamma: np.ndarray) -> float:
    """Exact variance of sum_t indicator[z_t, t] over draws from the chain.

    Backward sampling makes the state path a Markov chain in reverse, so the
    first two moments of a sum along it follow from one recursion: carry, for
    every state at position t, the conditional mean and second moment of the
    centred sum over the positions before t.
    """
    T, K = alpha.shape
    centre = (gamma * indicator.T).sum(axis=1)
    mean = np.zeros(K)          # E[centred sum so far | state here]
    second = np.zeros(K)        # E[(centred sum so far)^2 | state here]
    for t in range(1, T):
        s = chain.switch[t - 1]
        prev = alpha[t - 1]
        g = indicator[:, t - 1] - centre[t - 1]
        first_term = g + mean
        second_term = g * g + 2.0 * g * mean + second
        denom = (1.0 - s) * prev + s / K
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(denom > 0, ((1.0 - s) * prev * first_term
                                        + (s / K) * (prev @ first_term)) / denom, 0.0)
            second = np.where(denom > 0, ((1.0 - s) * prev * second_term
                                          + (s / K) * (prev @ second_term)) / denom, 0.0)
    g = indicator[:, T - 1] - centre[T - 1]
    return float(alpha[T - 1] @ (g * g + 2.0 * g * mean + second))


def seed_streams(seed: int, key: str = "", n: int = 2) -> list[np.random.SeedSequence]:
    """n independent random streams from one integer seed and an optional string key."""
    if not key:
        return np.random.SeedSequence(seed).spawn(n)
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    return np.random.SeedSequence([int(seed), h]).spawn(n)
