import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathguard import cohort, model, target  # noqa: E402

TEST_DATA = ROOT / "test_data"


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """The test_data cohort built once, with one target projected onto it."""
    out = tmp_path_factory.mktemp("cohort")
    summary = cohort.build(TEST_DATA / "cohort.chr21.vcf.gz",
                           TEST_DATA / "chr21.b38.gmap.gz", out / "chr21",
                           gaps_table=TEST_DATA / "gap.chr21.txt.gz", chrom="chr21")
    paths = {}
    for sample in ("HG00096", "NA20294"):
        paths[sample] = target.build(TEST_DATA / "1000g.chr21.vcf.gz", sample,
                                     out / "chr21", out / f"{sample}.npy")
    return {"dir": out / "chr21", "summary": summary, "paths": paths}


@pytest.fixture(scope="session")
def chain(built):
    c = cohort.Cohort(built["dir"])
    ch = model.make_chain(c.haplotypes, c.genetic_pos, c.positions, c.in_chain)
    return {"cohort": c, "chain": ch, "base": model.baseline_log_norm(ch),
            "beta": model.uniform_beta(ch.T),
            "path": built["paths"]["HG00096"][c.in_chain],
            "other": built["paths"]["NA20294"][c.in_chain]}


@pytest.fixture
def toy():
    """A cohort small enough to enumerate every output path exactly.

    Three donors, four positions. Donor 2 has no call at position 1, so the
    state mask is exercised, and positions 2 and 3 sit at the same genetic
    position, so no switch is possible between them.
    """
    haps = np.array([[0, 1, 0, 1],
                     [1, 0, 1, 0],
                     [0, -1, 1, 1]], dtype=np.int16)
    cm = np.array([0.0, 0.5, 1.0, 1.0])
    ch = model.Chain(haps=haps, switch=model.switch_mass(cm, 3),
                     positions=np.array([10, 20, 30, 40]))
    return {"chain": ch, "cm": cm,
            "p": np.array([[0, 1], [1, 0], [0, 1], [1, 1]], dtype=np.int16),
            "q": np.array([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int16)}
