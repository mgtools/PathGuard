"""The mechanism against exhaustive enumeration on a cohort small enough to enumerate.

Every expectation here is written from the paper's formulas and computed by
brute force over all donor-state sequences, so it does not share code with the
implementation it checks. These are the tests that can falsify Theorem 1.
"""
import itertools
import math

import numpy as np
import pytest

from pathguard import calibrate, model, release

NE, R = 10_000.0, 1.26


def dense_transition(delta_cm, k):
    """The full transition matrix, written straight from Eq. 1."""
    decay = math.exp(-delta_cm * NE * R / k)
    off = (1.0 - decay) / k
    a = np.full((k, k), off)
    np.fill_diagonal(a, decay + off)
    assert np.allclose(a.sum(axis=1), 1.0)
    return a


def enumerate_releases(chain, cm, target, beta, eta):
    """{released alleles: probability} for one haplotype, by brute force."""
    k, t = chain.haps.shape
    trans = [dense_transition(cm[i + 1] - cm[i], k) for i in range(t - 1)]
    out = {}
    for z in itertools.product(range(k), repeat=t):
        if any(chain.haps[z[i], i] < 0 for i in range(t)):
            continue                      # a donor with no call cannot be occupied
        prob = 1.0 / k
        for i in range(1, t):
            prob *= trans[i - 1][z[i - 1], z[i]]
        if prob == 0.0:
            continue        # no genetic distance means no switch, so this path cannot occur
        y = tuple(int(chain.haps[z[i], i]) for i in range(t))
        u = sum(beta[i] for i in range(t) if y[i] == target[i] and target[i] >= 0)
        out[y] = out.get(y, 0.0) + prob * math.exp(eta * u)
    total = sum(out.values())
    return {y: w / total for y, w in out.items()}


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(y, 0.0) - q.get(y, 0.0)) for y in keys)


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9, 0.99])
def test_total_variation_respects_theorem_1(toy, tau):
    """TV(Q_p, Q_q) <= tau for the tilt eta = arctanh(tau), for any two inputs."""
    eta = math.atanh(tau)
    chain, cm = toy["chain"], toy["cm"]
    beta = model.uniform_beta(chain.T)
    qp = enumerate_releases(chain, cm, toy["p"][:, 0], beta, eta)
    qq = enumerate_releases(chain, cm, toy["q"][:, 0], beta, eta)
    assert tv(qp, qq) <= tau + 1e-12


@pytest.mark.parametrize("tau", [0.5, 0.9])
def test_likelihood_ratio_stays_inside_the_pointwise_bound(toy, tau):
    """exp(-2 arctanh(tau)) <= Q_p(y) / Q_q(y) <= exp(2 arctanh(tau))."""
    eta = math.atanh(tau)
    chain, cm = toy["chain"], toy["cm"]
    beta = model.uniform_beta(chain.T)
    qp = enumerate_releases(chain, cm, toy["p"][:, 0], beta, eta)
    qq = enumerate_releases(chain, cm, toy["q"][:, 0], beta, eta)
    bound = math.exp(2 * eta)
    for y in set(qp) | set(qq):
        ratio = qp[y] / qq[y]
        assert 1.0 / bound - 1e-12 <= ratio <= bound + 1e-12


def test_forward_pass_normaliser_matches_enumeration(toy):
    """log Z_p from the forward pass equals the log of the enumerated sum."""
    eta = 2.0
    chain, cm = toy["chain"], toy["cm"]
    beta = model.uniform_beta(chain.T)
    x = toy["p"][:, 0]
    _, log_c = model.forward(chain, model.potentials(chain, x, beta, eta))
    got = model.log_norm(log_c) - model.baseline_log_norm(chain)

    k, t = chain.haps.shape
    trans = [dense_transition(cm[i + 1] - cm[i], k) for i in range(t - 1)]
    tilted = untilted = 0.0
    for z in itertools.product(range(k), repeat=t):
        if any(chain.haps[z[i], i] < 0 for i in range(t)):
            continue
        prob = 1.0 / k
        for i in range(1, t):
            prob *= trans[i - 1][z[i - 1], z[i]]
        u = sum(beta[i] for i in range(t)
                if chain.haps[z[i], i] == x[i] and x[i] >= 0)
        tilted += prob * math.exp(eta * u)
        untilted += prob
    assert got == pytest.approx(math.log(tilted / untilted), abs=1e-12)


def test_sampler_draws_from_the_release_distribution(toy):
    """Backward sampling reproduces the enumerated distribution, and is not an argmax."""
    eta = 3.0
    chain, cm = toy["chain"], toy["cm"]
    beta = model.uniform_beta(chain.T)
    x = toy["p"][:, 0]
    exact = enumerate_releases(chain, cm, x, beta, eta)

    psi = model.potentials(chain, x, beta, eta)
    alpha, _ = model.forward(chain, psi)
    rng = np.random.default_rng(0)
    counts = {}
    n = 20_000
    for _ in range(n):
        z = model.sample_states(chain, alpha, rng)
        y = tuple(int(chain.haps[z[i], i]) for i in range(chain.T))
        counts[y] = counts.get(y, 0) + 1

    assert len(counts) > 1, "the sampler returned a single path, so it is decoding"
    for y, want in exact.items():
        got = counts.get(y, 0) / n
        assert abs(got - want) < 4.0 * math.sqrt(want * (1 - want) / n) + 1e-3


def test_state_marginals_match_enumeration(toy):
    """The posterior state marginals agree with brute force, so the exact
    expectations built on them are exact."""
    eta = 2.5
    chain, cm = toy["chain"], toy["cm"]
    beta = model.uniform_beta(chain.T)
    x = toy["p"][:, 0]
    psi = model.potentials(chain, x, beta, eta)
    alpha, _ = model.forward(chain, psi)
    gamma = model.marginals(chain, alpha, psi)

    k, t = chain.haps.shape
    trans = [dense_transition(cm[i + 1] - cm[i], k) for i in range(t - 1)]
    want = np.zeros((t, k))
    for z in itertools.product(range(k), repeat=t):
        if any(chain.haps[z[i], i] < 0 for i in range(t)):
            continue
        prob = 1.0 / k
        for i in range(1, t):
            prob *= trans[i - 1][z[i - 1], z[i]]
        u = sum(beta[i] for i in range(t)
                if chain.haps[z[i], i] == x[i] and x[i] >= 0)
        w = prob * math.exp(eta * u)
        for i in range(t):
            want[i, z[i]] += w
    want /= want.sum(axis=1, keepdims=True)
    assert np.allclose(gamma, want, atol=1e-12)


def test_path_variance_matches_enumeration(toy):
    """The exact variance of the altered-allele count agrees with brute force."""
    eta = 2.0
    chain, cm = toy["chain"], toy["cm"]
    beta = model.uniform_beta(chain.T)
    x = toy["p"][:, 0]
    psi = model.potentials(chain, x, beta, eta)
    alpha, _ = model.forward(chain, psi)
    gamma = model.marginals(chain, alpha, psi)
    mism = ((chain.haps != x[None, :]) & (x[None, :] >= 0)).astype(float)
    got = model.path_variance(chain, alpha, mism, gamma)

    k, t = chain.haps.shape
    trans = [dense_transition(cm[i + 1] - cm[i], k) for i in range(t - 1)]
    weights, counts = [], []
    for z in itertools.product(range(k), repeat=t):
        if any(chain.haps[z[i], i] < 0 for i in range(t)):
            continue
        prob = 1.0 / k
        for i in range(1, t):
            prob *= trans[i - 1][z[i - 1], z[i]]
        if prob == 0.0:
            continue
        u = sum(beta[i] for i in range(t)
                if chain.haps[z[i], i] == x[i] and x[i] >= 0)
        weights.append(prob * math.exp(eta * u))
        counts.append(sum(mism[z[i], i] for i in range(t)))
    w = np.array(weights) / sum(weights)
    c = np.array(counts)
    want = float(w @ (c ** 2) - (w @ c) ** 2)
    assert got == pytest.approx(want, rel=1e-10)


def test_kl_matches_enumeration(toy):
    """The KL used for calibration agrees with brute force over both haplotypes."""
    eta = 1.5
    chain, cm = toy["chain"], toy["cm"]
    beta = model.uniform_beta(chain.T)
    got = calibrate.kl(chain, toy["p"], toy["q"], beta, eta)
    want = 0.0
    for col in (0, 1):
        qp = enumerate_releases(chain, cm, toy["p"][:, col], beta, eta / 2)
        qq = enumerate_releases(chain, cm, toy["q"][:, col], beta, eta / 2)
        want += sum(qp[y] * math.log(qp[y] / qq[y]) for y in qp)
    assert got == pytest.approx(want, rel=1e-10)


def test_both_haplotypes_are_drawn_at_half_the_tilt(toy):
    """The diploid release spends one budget across the two haplotypes: its log Z
    is the sum of two chains run at eta / 2, which is what keeps u in [0, 1]."""
    eta = 4.0
    chain = toy["chain"]
    beta = model.uniform_beta(chain.T)
    base = model.baseline_log_norm(chain)
    got = release.log_z(chain, toy["p"], beta, eta, base)
    halves = []
    for col in (0, 1):
        _, log_c = model.forward(chain, model.potentials(chain, toy["p"][:, col], beta, eta / 2))
        halves.append(model.log_norm(log_c) - base)
    assert got == pytest.approx(sum(halves), rel=1e-12)
    assert 0.0 <= got <= eta + 1e-9


def test_zero_tilt_is_independent_of_the_target(toy):
    """At eta = 0 the release distribution is the cohort baseline, so two
    different targets induce exactly the same distribution."""
    chain = toy["chain"]
    beta = model.uniform_beta(chain.T)
    base = model.baseline_log_norm(chain)
    y1, lz1 = release.release(chain, toy["p"], beta, 0.0, seed=7, base=base)
    y2, lz2 = release.release(chain, toy["q"], beta, 0.0, seed=7, base=base)
    assert np.array_equal(y1, y2)
    assert lz1 == pytest.approx(0.0, abs=1e-12)
    assert lz2 == pytest.approx(0.0, abs=1e-12)
    assert calibrate.kl(chain, toy["p"], toy["q"], beta, 0.0) == 0.0


def test_switch_probability_matches_the_published_closed_form(toy):
    """s_t decomposed from A = (1-s) I + (s/K) J agrees with p_stay as published."""
    cm = toy["cm"]
    k = 3
    s = model.switch_mass(cm, k)
    for i, got in enumerate(s):
        decay = math.exp(-(cm[i + 1] - cm[i]) * NE * R / k)
        assert 1.0 - got + got / k == pytest.approx(decay + (1.0 - decay) / k, abs=1e-15)
    assert s[-1] == 0.0, "no genetic distance must mean no switch"


def test_constructed_decoy_differs_at_every_called_position(toy):
    p = toy["p"]
    q = calibrate.constructed_decoy(p)
    called = p >= 0
    assert (q[called] != p[called]).all()
    assert (q[~called] == -1).all()


def test_pinsker_ceiling_inverts_the_kl_budget():
    for ceiling in (0.55, 0.65, 0.8):
        assert calibrate.pinsker_ceiling(calibrate.kl_budget(ceiling)) == pytest.approx(ceiling)
