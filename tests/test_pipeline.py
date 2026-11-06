"""The whole tool on the bundled test data: build, project, calibrate, release, attack.

These are the properties the paper claims of a release, checked on real data at
a scale that runs in a couple of minutes.
"""
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT, TEST_DATA
from pathguard import attacks, calibrate, cli, cohort, model, release


def test_cohort_build(built):
    s = built["summary"]
    assert s["samples"] == 44, "the graph cohort is 44 individuals once CHM13 is dropped"
    assert s["records"] == 4071
    assert s["chain"] == s["records"], "every record here has a called donor"

    c = cohort.Cohort(built["dir"])
    assert c.haplotypes.shape == (88, 4071)
    assert len(c.samples) == 44
    assert c.haplotype_ids[0].endswith(":0") and c.haplotype_ids[1].endswith(":1")


def test_genetic_positions_are_monotone_and_finite(built):
    cm = np.load(built["dir"] / "genetic_pos.npy")
    assert np.isfinite(cm).all(), "the declared rate must fill in where the map is silent"
    assert (np.diff(cm) >= 0).all(), "a negative genetic distance would leave [0, 1]"


def test_declared_rate_fills_only_outside_the_map():
    """Inside the map the cM comes from interpolation; outside, from the rate over
    base pairs that are not assembly gaps."""
    pos = np.array([1_000_000, 10_500_000, 20_000_000, 47_000_000])
    gaps = cohort.read_gaps(TEST_DATA / "gap.chr21.txt.gz", "chr21")
    cm = cohort.genetic_positions(pos, TEST_DATA / "chr21.b38.gmap.gz", gaps)
    assert np.isfinite(cm).all()
    assert cm[0] < cm[1] < cm[2] < cm[3]
    # the leading position is below the map's first marker, so its distance back
    # from that marker must skip the gaps
    span = cohort.gap_free_bp(np.array([pos[0]]), np.array([10_326_676]), gaps)[0]
    assert span < 10_326_676 - pos[0], "chr21's p arm is mostly assembly gap"


def test_untraversable_runs_leave_the_chain():
    """Inside a run of zero genetic distance the chain cannot switch donors, so a
    run no single donor is called across has no path through it and is dropped."""
    # positions 1 and 2 are one zero-distance run, and no donor is called at both
    haps = np.array([[0, 0, -1], [1, -1, 1], [-1, 1, -1]], dtype=np.int16)
    mask = cohort.chain_mask(haps, np.array([0.0, 1.0, 1.0]))
    assert mask.tolist() == [True, False, False]
    assert cohort.chain_mask(haps, np.array([0.0, 1.0, 2.0])).all(), \
        "with genetic distance between them the same positions are fine"
    # a run one donor does span is traversable and stays
    spanned = np.array([[0, 0, -1], [1, -1, 1], [-1, 1, 0]], dtype=np.int16)
    assert cohort.chain_mask(spanned, np.array([0.0, 1.0, 1.0])).all()


def test_target_projection(built):
    path = built["paths"]["HG00096"]
    assert path.shape == (4071, 2)
    assert path.dtype == np.int16
    called = (path >= 0).sum(axis=0)
    assert (called > 3000).all(), f"expected most records called, got {called}"

    from pathguard import target as tgt
    _, _, alt = tgt.load_sites(built["dir"] / "sites.tsv")
    n_alt = np.array([len(a.split(",")) for a in alt])
    assert (path <= n_alt[:, None]).all(), "an allele index outside the record's alleles"


def test_projection_recovers_multiallelic_records(built):
    """The panel splits a multi-allelic record into one biallelic record per ALT, so
    those records are recovered by matching each ALT separately."""
    from pathguard import target as tgt
    pos, ref, alt = tgt.load_sites(built["dir"] / "sites.tsv")
    multi = np.array(["," in a for a in alt])
    assert multi.sum() > 100, "the test slice should contain multi-allelic records"
    path = built["paths"]["HG00096"]
    assert (path[multi] >= 0).any(), "no multi-allelic record was recovered"


def test_calibration_reaches_the_requested_ceiling(chain):
    """The solved tilt puts the target's divergence against the constructed decoy at
    the ceiling's budget, so the realised ceiling is the one asked for."""
    ch, p, beta = chain["chain"], chain["path"], chain["beta"]
    q = calibrate.constructed_decoy(p)
    probes = {20.0: calibrate.kl(ch, p, q, beta, 20.0),
              40.0: calibrate.kl(ch, p, q, beta, 40.0)}
    for want in (0.60, 0.80):
        eta = calibrate.fit_eta(probes, want)
        kl = calibrate.kl(ch, p, q, beta, eta)
        eta = calibrate.refine_eta(eta, kl, want)
        got = calibrate.pinsker_ceiling(calibrate.kl(ch, p, q, beta, eta))
        assert got == pytest.approx(want, abs=1e-5)


def test_one_budget_splits_across_chromosomes(chain):
    """A genome-wide release spends one budget over every chromosome: the weights
    sum to one across them, each carries its share of the divergence, and the tilt
    per position falls because the same budget covers more sequence."""
    ch, p = chain["chain"], chain["path"]
    one = model.uniform_beta(ch.T)
    eta_one, kl_one, _ = cli.solve([ch], [p], [one], 0.80)

    # the same chromosome twice stands in for two of equal length
    half = [model.uniform_beta(ch.T, 0.5), model.uniform_beta(ch.T, 0.5)]
    assert sum(b.sum() for b in half) == pytest.approx(1.0)
    eta_two, kl_two, acc = cli.solve([ch, ch], [p, p], half, 0.80)

    assert acc == pytest.approx(0.80, abs=1e-5), "the ceiling is the genome's, not a part's"
    assert kl_two == pytest.approx(kl_one, rel=1e-6), "one budget, however it is divided"
    q = calibrate.constructed_decoy(p)
    assert calibrate.kl(ch, p, q, half[0], eta_two) == pytest.approx(kl_two / 2, rel=1e-9), \
        "each of two equal chromosomes carries half the divergence"
    assert eta_two > eta_one, "the tilt is larger because each weight is smaller"
    assert eta_two * half[0][0] < eta_one * one[0], "but the tilt per position is smaller"


def test_release_is_a_mosaic_of_cohort_haplotypes(chain):
    """Every released allele is one a called cohort donor carries at that position,
    so the release introduces nothing of the target's own."""
    ch, p, beta = chain["chain"], chain["path"], chain["beta"]
    y, _ = release.release(ch, p, beta, 30.0, seed=1, base=chain["base"])
    assert y.shape == p.shape
    assert (y >= 0).all(), "a release must not carry a missing allele"
    for h in (0, 1):
        carried = (ch.haps == y[:, h][None, :]) & (ch.haps >= 0)
        assert carried.any(axis=0).all()


def test_release_is_reproducible_and_random(chain):
    ch, p, beta = chain["chain"], chain["path"], chain["beta"]
    a, lz_a = release.release(ch, p, beta, 30.0, seed=5, base=chain["base"])
    b, lz_b = release.release(ch, p, beta, 30.0, seed=5, base=chain["base"])
    c, _ = release.release(ch, p, beta, 30.0, seed=6, base=chain["base"])
    assert np.array_equal(a, b) and lz_a == lz_b
    assert not np.array_equal(a, c), "two seeds must give two draws"


def test_log_z_stays_inside_its_budget(chain):
    ch, p, beta = chain["chain"], chain["path"], chain["beta"]
    for eta in (0.0, 5.0, 30.0):
        _, lz = release.release(ch, p, beta, eta, seed=2, base=chain["base"])
        assert -1e-9 <= lz <= eta + 1e-9


def test_utility_is_bounded_and_rises_with_the_tilt(chain):
    """A larger tilt keeps more of the target's alleles, which is the whole trade."""
    ch, p, beta = chain["chain"], chain["path"], chain["beta"]
    us = []
    for eta in (0.0, 30.0, 120.0):
        u = np.mean([release.utility(release.release(ch, p, beta, eta, seed=s,
                                                     base=chain["base"])[0], p, beta)
                     for s in range(6)])
        assert 0.0 <= u <= 1.0
        us.append(u)
    assert us[0] < us[1] < us[2]


def test_exact_expectations_match_sampling(chain):
    """The expected altered count from the state marginals is what sampling converges
    to, which is why the utility tables need no Monte Carlo error."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from utility import expectations

    ch, p, beta = chain["chain"], chain["path"], chain["beta"]
    eta = 30.0
    exact = expectations(ch, p, beta, eta)
    called = (p >= 0)
    draws = [((release.release(ch, p, beta, eta, seed=s, base=chain["base"])[0] != p)
              & called).sum() for s in range(40)]
    mean = np.mean(draws)
    se = np.std(draws, ddof=1) / np.sqrt(len(draws))
    assert exact["called"] == int(called.sum())
    assert abs(exact["altered"] - mean) < 4 * se + 1


def test_optimal_distinguisher_is_near_chance_without_a_tilt(chain):
    """At the untilted setting the release carries no information about the target,
    so the optimal attacker cannot beat a coin flip."""
    ch, p, beta, q = chain["chain"], chain["path"], chain["beta"], chain["other"]
    lz_p = release.log_z(ch, p, beta, 0.0, chain["base"])
    lz_q = release.log_z(ch, q, beta, 0.0, chain["base"])
    stats = {0: [], 1: []}
    for s in range(30):
        for arm, x in ((0, p), (1, q)):
            y, _ = release.release(ch, x, beta, 0.0, seed=1000 + s, base=chain["base"])
            stats[arm].append(attacks.lr_statistic(0.0, release.utility(y, p, beta),
                                                   release.utility(y, q, beta), lz_p, lz_q))
    acc = attacks.balanced_accuracy(stats[0], stats[1])
    assert acc["balanced"] == pytest.approx(0.5, abs=1e-9)


def test_optimal_distinguisher_stays_under_the_ceiling(chain):
    """The measured accuracy of the optimal attacker must not exceed the calibrated
    ceiling: if it did, the implementation would contradict the bound."""
    ch, p, beta, q = chain["chain"], chain["path"], chain["beta"], chain["other"]
    decoy = calibrate.constructed_decoy(p)
    probes = {20.0: calibrate.kl(ch, p, decoy, beta, 20.0),
              40.0: calibrate.kl(ch, p, decoy, beta, 40.0)}
    want = 0.80
    eta = calibrate.fit_eta(probes, want)
    eta = calibrate.refine_eta(eta, calibrate.kl(ch, p, decoy, beta, eta), want)

    lz_p = release.log_z(ch, p, beta, eta, chain["base"])
    lz_q = release.log_z(ch, q, beta, eta, chain["base"])
    stats = {0: [], 1: []}
    for s in range(40):
        for arm, x in ((0, p), (1, q)):
            y, _ = release.release(ch, x, beta, eta, seed=2000 + s, base=chain["base"])
            stats[arm].append(attacks.lr_statistic(eta, release.utility(y, p, beta),
                                                   release.utility(y, q, beta), lz_p, lz_q))
    acc = attacks.balanced_accuracy(stats[0], stats[1])
    assert acc["balanced"] <= want + 2 * acc["se"] + 1e-12


def test_attacks_identify_the_target_when_the_release_is_the_target(chain):
    """A positive control: handed the target's own path, every attack must point at
    the target. An attack that cannot succeed here proves nothing when it fails."""
    ch, p, beta, q = chain["chain"], chain["path"], chain["beta"], chain["other"]
    eta = 30.0
    y = p.copy()
    y[y < 0] = 0
    lz_p = release.log_z(ch, p, beta, eta, chain["base"])
    lz_q = release.log_z(ch, q, beta, eta, chain["base"])
    assert attacks.lr_statistic(eta, release.utility(y, p, beta),
                                release.utility(y, q, beta), lz_p, lz_q) > 0
    linkage = attacks.RarityLinkage(ch.haps)
    assert linkage.statistic(y, p, q) > 0
    blocks = attacks.block_statistics(y, p, q, beta, eta)
    assert all((b > 0).mean() > 0.9 for b in blocks.values())


def test_rarity_linkage_weights_are_minus_log_genotype_frequency(chain):
    """The score weights a shared genotype by -log of its Hardy-Weinberg frequency in
    the cohort, so a rarer shared genotype counts for more."""
    haps = chain["chain"].haps
    linkage = attacks.RarityLinkage(haps)
    sub = haps[:, linkage.mask]
    called = sub >= 0
    freq = ((sub == linkage.alt[None, :]) & called).sum(axis=0) / called.sum(axis=0)
    expected = np.stack([-np.log((1 - freq) ** 2 + 1e-10),
                         -np.log(2 * freq * (1 - freq) + 1e-10),
                         -np.log(freq ** 2 + 1e-10)])
    assert np.allclose(linkage.weight, expected)
    # where the alternate allele is the rare one, sharing it counts for more
    rare = freq < 0.1
    assert rare.any()
    assert (linkage.weight[2][rare] > linkage.weight[0][rare]).all()


def test_command_line_runs_end_to_end(tmp_path):
    """The documented commands, as a reviewer would run them."""
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}
    def run(*args):
        r = subprocess.run([sys.executable, "-m", "pathguard.cli", *args],
                           cwd=ROOT, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        return r
    run("cohort", "--vcf", str(TEST_DATA / "cohort.chr21.vcf.gz"),
        "--gmap", str(TEST_DATA / "chr21.b38.gmap.gz"),
        "--gaps", str(TEST_DATA / "gap.chr21.txt.gz"),
        "--chrom", "chr21", "--out", str(tmp_path / "cohort"))
    run("target", "--vcf", str(TEST_DATA / "1000g.chr21.vcf.gz"), "--sample", "HG00096",
        "--cohort", str(tmp_path / "cohort"), "--out", str(tmp_path / "HG00096.npy"))
    out = run("calibrate", "--cohort", str(tmp_path / "cohort"),
              "--target", str(tmp_path / "HG00096.npy"), "--ceiling", "0.8")
    assert "0.80" in out.stdout
    # into a directory that does not exist yet, as the documented examples do
    run("release", "--cohort", str(tmp_path / "cohort"),
        "--target", str(tmp_path / "HG00096.npy"), "--ceiling", "0.8",
        "--seed", "1", "--out", str(tmp_path / "releases" / "release.vcf"))
    lines = (tmp_path / "releases" / "release.vcf").read_text().splitlines()
    body = [x for x in lines if not x.startswith("#")]
    assert len(body) == 4071
    assert all("|" in x.split("\t")[9] for x in body)

    # the genome command, on a root standing in for two chromosomes
    (tmp_path / "targets").mkdir()
    for c in ("chr21", "chr22"):
        shutil.copytree(tmp_path / "cohort", tmp_path / "genome" / c)
        shutil.copy(tmp_path / "HG00096.npy", tmp_path / "targets" / f"{c}.npy")
    out = run("genome", "--cohort-root", str(tmp_path / "genome"),
              "--target-dir", str(tmp_path / "targets"), "--ceiling", "0.8",
              "--seed", "1", "--out", str(tmp_path / "wgs"))
    assert "chr21, chr22" in out.stdout
    rows = [x.split("\t") for x in out.stdout.splitlines()]
    shares = [r[2] for r in rows if r[0] in ("chr21", "chr22")]
    assert shares == ["0.500000", "0.500000"], out.stdout
    for c in ("chr21", "chr22"):
        assert (tmp_path / "wgs" / f"{c}.vcf.gz").exists()
