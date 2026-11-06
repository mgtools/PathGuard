"""Put both tools' releases into PanMixer's experiment layout so its evaluators run on them.

PanMixer resolves its data and experiment roots from the location of its own
`constants.py`, so the staging root holds a copy of that one file with `tools`,
`downloaded_tools` and `starting_data` symlinked back. Nothing in the PanMixer
checkout is written.

One experiment per evaluator per tool, because `gap_score.py` and
`diploid_gap_score.py` write the same output filenames into the row directory:

  exp_1  Beagle reconstruction, PathGuard      exp_6  Beagle reconstruction, PanMixer
  exp_2  rarity-weighted linkage, PathGuard    exp_7  rarity-weighted linkage, PanMixer
  exp_3  haplotype linkage, PathGuard          exp_8  haplotype linkage, PanMixer
  exp_4  membership inference, PathGuard       exp_9  membership inference, PanMixer
  exp_5  af_loss and ld_loss, PathGuard        exp_10 af_loss and ld_loss, PanMixer

Beagle's reconstruction statistic is restricted to the cells named by `xsol.npy`.
PathGuard has no cell-selection step, so it is staged with an all-ones mask, and
PanMixer is staged the same way to keep one denominator across the comparison.

    python stage.py --ours DIR --theirs DIR --root DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

CEILINGS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
CAPACITIES = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
EVALUATORS = ("beagle", "gap_score", "haplotype_gap", "membership", "af_ld")
EXPERIMENTS = {("pathguard", e): str(i + 1) for i, e in enumerate(EVALUATORS)}
EXPERIMENTS |= {("panmixer", e): str(i + 6) for i, e in enumerate(EVALUATORS)}


def point_dir(value: float, prefix: str) -> str:
    return f"{prefix}{value:g}".replace(".", "p") if prefix == "cap" else \
        f"acc_{value:.2f}".replace(".", "p")


def mirror(root: Path, panmixer: Path, starting_data: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(panmixer / "constants.py", root / "constants.py")
    for name, target in (("tools", panmixer / "tools"),
                         ("downloaded_tools", panmixer / "downloaded_tools"),
                         ("starting_data", starting_data)):
        link = root / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())


class Build:
    """The arrays every staged release is checked against."""

    def __init__(self, starting_data: Path, chrom: int):
        d = starting_data / f"chr{chrom}"
        self.positions = np.load(d / "pangenome_positions.npy")
        self.subjects = np.load(d / "pangenome_subjects.npy").tolist()
        self.num_alleles = np.load(d / "num_alleles.npy")
        self.n_sites = len(self.positions)
        pangenome = np.load(d / "pangenome.npy", mmap_mode="r")
        if pangenome.shape != (len(self.subjects), self.n_sites, 2):
            raise SystemExit(f"pangenome.npy {pangenome.shape} does not match this build")
        # The sites gap_score restricts to, so its own assertion is checked here
        # rather than part way through a fan-out.
        snp_mask = np.load(d / "pangenome_mask.npy")
        freq = np.load(d / "allele_frequencies.npy", mmap_mode="r")
        biallelic = np.load(d / "biallelic_snp_mask.npy")[snp_mask]
        self.snp_mask = snp_mask
        self.biallelic_snp = (np.asarray(freq[snp_mask][:, :2]).sum(axis=1) == 1) & biallelic

    def check(self, array: np.ndarray, who: str) -> None:
        if array.shape != (self.n_sites, 2):
            raise SystemExit(f"{who}: shape {array.shape}, expected ({self.n_sites}, 2) -- "
                             f"this release is on another build's site axis")
        if array.dtype != np.int16:
            raise SystemExit(f"{who}: dtype {array.dtype}, expected int16")
        bad = int(((array >= self.num_alleles[:, None]) & (array >= 0)).sum())
        if bad:
            raise SystemExit(f"{who}: {bad} allele code(s) above num_alleles for the site")
        high = int((array[self.snp_mask][self.biallelic_snp] >= 2).sum())
        if high:
            raise SystemExit(f"{who}: {high} allele code(s) >= 2 at bi-allelic SNP sites, "
                             f"which gap_score.py asserts against")


def rows_for(tool: str, releases: Path, subjects: list[str], points, chrom: int) -> list[dict]:
    out = []
    for subject in subjects:
        for value in points:
            if tool == "pathguard":
                name = point_dir(value, "acc_")
                point = {"ceiling": name, "capacity": "n/a"}
            else:
                name = point_dir(value, "cap")
                point = {"ceiling": "n/a", "capacity": value}
            source = releases / subject / name / f"chr{chrom}" / "new_haplotypes.npy"
            if not source.exists():
                raise SystemExit(f"missing release: {source}")
            out.append({"row": len(out), "subject": subject, "point": name,
                        "source": str(source), **point})
    return out


def stage(root: Path, experiment: str, rows: list[dict], chrom: int, build: Build,
          xsol: np.ndarray | None) -> None:
    base = root / "experiments" / f"exp_{experiment}" / "data" / f"chr{chrom}"
    base.mkdir(parents=True, exist_ok=True)
    # Row order is the only thing that tells an evaluator whose release a row is.
    csv_path = root / "experiments" / f"exp_{experiment}" / f"data_chr{chrom}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "ceiling", "capacity"])
        for r in rows:
            writer.writerow([r["subject"], r["ceiling"], r["capacity"]])
    for r in rows:
        source = Path(r["source"]).resolve()
        build.check(np.load(source), f"row {r['row']} {r['subject']} {r['point']}")
        row_dir = base / str(r["row"])
        row_dir.mkdir(parents=True, exist_ok=True)
        link = row_dir / "new_haplotypes.npy"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source)
        if xsol is not None:
            np.save(row_dir / "xsol.npy", xsol)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="staging root to write")
    p.add_argument("--ours", type=Path, required=True, help="PathGuard releases")
    p.add_argument("--theirs", type=Path, required=True, help="PanMixer releases, seed 123")
    p.add_argument("--starting-data", type=Path, required=True)
    p.add_argument("--panmixer", type=Path,
                   default=Path(os.environ.get("PANMIXER", "external/PanMixer")))
    p.add_argument("--chrom", type=int, default=21)
    p.add_argument("--subjects", nargs="+", default=None,
                   help="default: every subject in the graph cohort")
    p.add_argument("--beagle-subject", default="HG00438",
                   help="Beagle on PanMixer's releases is one subject (2.3 h per 8 rows)")
    p.add_argument("--only", nargs="+", default=None,
                   help=f"stage a subset, as tool:evaluator, e.g. panmixer:af_ld. "
                        f"evaluators: {', '.join(EVALUATORS)}")
    a = p.parse_args()

    build = Build(a.starting_data, a.chrom)
    mirror(a.root, a.panmixer, a.starting_data)
    print(f"staging root  : {a.root}")
    print(f"  starting_data -> {a.starting_data.resolve()} ({build.n_sites:,} sites)")
    print(f"  tools         -> {(a.panmixer / 'tools').resolve()}")

    subjects = a.subjects or sorted(build.subjects)
    unknown = [s for s in subjects if s not in set(build.subjects)]
    if unknown:
        raise SystemExit(f"not in pangenome_subjects.npy: {unknown}")
    for tool, releases in (("PathGuard", a.ours), ("PanMixer", a.theirs)):
        missing = [s for s in subjects if not (releases / s).is_dir()]
        if missing:
            raise SystemExit(f"{tool} has no releases for: {missing}")
    print(f"  {len(subjects)} cohort members, both tools")

    n_blocks = len(json.loads(
        (a.starting_data / f"chr{a.chrom}" / "blocks_dict.json").read_text()))
    ones = np.ones((n_blocks, 2), dtype=np.int64)

    every = [f"{t}:{e}" for t in ("pathguard", "panmixer") for e in EVALUATORS]
    wanted = set(a.only or every)
    unknown = sorted(wanted - set(every))
    if unknown:
        raise SystemExit(f"--only {unknown} names nothing to stage; expected "
                         f"tool:evaluator from {every}")
    sources, manifest_rows = {}, {}
    for tool, releases, points in (("pathguard", a.ours, CEILINGS),
                                   ("panmixer", a.theirs, CAPACITIES)):
        for evaluator in EVALUATORS:
            if f"{tool}:{evaluator}" not in wanted:
                continue
            who = [a.beagle_subject] if (tool == "panmixer" and evaluator == "beagle") \
                else subjects
            rows = rows_for(tool, releases, who, points, a.chrom)
            experiment = EXPERIMENTS[(tool, evaluator)]
            stage(a.root, experiment, rows, a.chrom, build,
                  ones if evaluator == "beagle" else None)
            sources[f"{tool}_{evaluator}"] = {"root": str(a.root), "experiment": experiment,
                                              "tool": tool, "evaluator": evaluator}
            manifest_rows[f"exp_{experiment}"] = rows
            print(f"  exp_{experiment:<2} {tool:9} {evaluator:13} {len(rows)} rows")

    manifest = {"chrom": a.chrom, "n_sites": build.n_sites, "n_blocks": n_blocks,
                "starting_data": str(a.starting_data.resolve()),
                "ours": str(a.ours), "theirs": str(a.theirs),
                "subjects": subjects, "ceilings": list(CEILINGS),
                "capacities": list(CAPACITIES),
                "beagle_subject": a.beagle_subject,
                "xsol": "all ones", "rows": manifest_rows}
    (a.root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (a.root / "sources.json").write_text(json.dumps(sources, indent=1) + "\n")
    print(f"manifest -> {a.root / 'manifest.json'}")
    print(f"sources  -> {a.root / 'sources.json'} (summarize.py reads this)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
