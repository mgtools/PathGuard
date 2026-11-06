"""Collect the cohort-member comparison into one JSON and print it.

Reads the evaluator outputs staged by stage.py (through the sources.json it
wrote), PanMixer's own obfuscation stats, PathGuard's altered fractions, and
optionally the missing-excluded numbers from missing_excluded.py.

    python summarize.py --sources ROOT/sources.json --obf DIR --obf-draws DIR \
        --fidelity results/pathguard_altered_fraction.json \
        --out results/panmixer_comparison.json

Keys in the output, all flat:
  altered_fraction_panmixer_seed123   per capacity: median, min, max, n (Table 6)
  altered_fraction_panmixer_mean5     the same over the five draws (Table 6)
  altered_fraction_panmixer_subject   per subject, per capacity: seed123, mean5
  altered_fraction_pathguard          per ceiling: median, min, max, n (Table 5)
  altered_fraction_pathguard_subject  per subject, per ceiling
  pathguard                           per ceiling, one key per evaluator (Table 8)
  panmixer                            per capacity, one key per evaluator (Table 8)
  matched_capacity                    per subject, plus the paired differences
  text_numbers                        the missing-excluded and linkage-loss figures
A linked individual is one whose gap score reaches zero, the rule in PanMixer's
own plotting code. Membership inference reports the rank of the true individual
among the 44 cohort members, so rank 1 is an identification.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

CAPACITIES = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
CEILINGS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
RESULT_FILE = {"beagle": "accuracy_results.json", "gap_score": "gap_score_all.json",
               "haplotype_gap": "gap_score_all.json", "membership": "MIA_privacy.json",
               "af_ld": "af_loss.json"}


def capacity_key(value: float) -> str:
    return f"{value:g}"


def ceiling_key(value: float) -> str:
    return f"{value:.2f}"


def read_rows(root: Path, experiment: str, chrom: int) -> list[dict]:
    """The staged rows of one experiment, in the order its evaluator read them."""
    path = root / "experiments" / f"exp_{experiment}" / f"data_chr{chrom}.csv"
    with path.open() as f:
        table = list(csv.DictReader(f))
    rows = []
    for i, row in enumerate(table):
        ceiling = (row.get("ceiling") or "").strip()
        capacity = (row.get("capacity") or "").strip()
        if ceiling.startswith("acc_"):
            point = ceiling_key(float(ceiling[4:].replace("p", ".")))
        elif capacity and capacity != "n/a":
            point = capacity_key(float(capacity))
        else:
            raise SystemExit(f"{path} row {i}: neither a ceiling nor a capacity")
        rows.append({"row": i, "subject": row["subject"], "point": point})
    return rows


def read_value(row_dir: Path, evaluator: str):
    """One row's statistic. A missing file stops the run; a partial table is worse."""
    path = row_dir / RESULT_FILE[evaluator]
    if not path.exists():
        raise SystemExit(f"missing {path} -- the evaluator did not finish this row")
    data = json.loads(path.read_text())
    if evaluator == "beagle":
        snp = data["snp_only"]
        return snp["beagle_accuracy"] - snp["major_allele_accuracy"]
    if evaluator == "gap_score":
        return float(data["genotype_g_to_gstar"])
    if evaluator == "haplotype_gap":
        return float(data["haplotype_g_to_gstar"])
    if evaluator == "membership":
        return int(data["hap_original_rank"])
    total, pairs = (row_dir / "ld_loss.txt").read_text().split()
    return {"af_loss": float(data["all"]), "ld_loss": float(total) / int(pairs)}


def collect(source: dict, evaluator: str, chrom: int) -> dict:
    """{point: {subject: value}} for one tool and one evaluator."""
    root = Path(source["root"])
    rows = read_rows(root, source["experiment"], chrom)
    base = root / "experiments" / f"exp_{source['experiment']}" / "data" / f"chr{chrom}"
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        value = read_value(base / str(row["row"]), evaluator)
        out.setdefault(row["point"], {})[row["subject"]] = value
    return out


def summarize_evaluators(sources: dict, tool: str, points, chrom: int) -> dict:
    """Table 8 for one tool: a median per point, with the counts the paper reports."""
    out: dict[str, dict] = {}
    for name, source in sorted(sources.items()):
        if not name.startswith(f"{tool}_"):
            continue
        evaluator = name[len(tool) + 1:]
        per_point = collect(source, evaluator, chrom)
        keys = [k for k in (capacity_key(p) if tool == "panmixer" else ceiling_key(p)
                            for p in points) if k in per_point]
        if evaluator == "af_ld":
            out["af_loss"] = {k: st.median(v["af_loss"] for v in per_point[k].values())
                              for k in keys}
            out["ld_loss_per_pair"] = {
                k: st.median(v["ld_loss"] for v in per_point[k].values()) for k in keys}
            out["n_af_ld"] = {k: len(per_point[k]) for k in keys}
            continue
        values = {k: list(per_point[k].values()) for k in keys}
        out[evaluator] = {k: st.median(values[k]) for k in keys}
        out[f"n_{evaluator}"] = {k: len(values[k]) for k in keys}
        if evaluator in ("gap_score", "haplotype_gap"):
            out[f"{evaluator}_linked"] = {k: sum(1 for v in values[k] if v >= 0)
                                          for k in keys}
        if evaluator == "membership":
            out["membership_at_rank_1"] = {k: sum(1 for v in values[k] if v == 1)
                                           for k in keys}
        if evaluator == "beagle":
            out["beagle_subjects"] = {k: sorted(per_point[k]) for k in keys}
    return out


def read_obfuscation(obf: Path, draws: Path, chrom: int) -> dict:
    """{subject: {capacity: {seed: changed fraction}}} from PanMixer's own stats."""
    out: dict[str, dict[str, dict[int, float]]] = {}
    files = list(obf.glob(f"*/cap*/chr{chrom}/stats.json"))
    if draws:
        files += list(draws.glob(f"*/cap*/s*/chr{chrom}/stats.json"))
    if not files:
        raise SystemExit(f"no stats.json under {obf}")
    for path in files:
        stats = json.loads(path.read_text())
        key = capacity_key(round(float(stats["capacity"]), 2))
        out.setdefault(stats["subject_name"], {}).setdefault(key, {})[
            int(stats["seed"])] = float(stats["changed_frac"])
    return out


def read_fidelity(path: Path) -> dict:
    """PathGuard's altered fraction as {subject: {ceiling: value}}.

    Accepts either that mapping directly or the release script's per-target
    records, where the value is the target_fidelity of each ceiling's cell.
    """
    data = json.loads(path.read_text())
    if "results" in data:
        return {r["target"]: {ceiling_key(float(k)): c["target_fidelity"]
                              for k, c in r["cells"].items()} for r in data["results"]}
    return {s: {ceiling_key(float(k)): float(v) for k, v in cells.items()}
            for s, cells in data.items()}


def spread(values: list[float]) -> dict:
    return {"median": st.median(values), "min": min(values), "max": max(values),
            "n": len(values)}


def matched_capacity(ours: dict, theirs_mean: dict, ceiling: str, plateau: str) -> dict:
    """The capacity at which PanMixer's frontier reaches PathGuard's altered fraction.

    Linear interpolation inside the first bracketing pair of capacities. Above
    the plateau the frontier is flat, so for some individuals PathGuard's
    fraction is never reached and there is no matched capacity.
    """
    per_subject, differences = {}, {}
    for subject in sorted(ours):
        target = ours[subject][ceiling]
        frontier = [(c, theirs_mean[subject][capacity_key(c)]) for c in CAPACITIES]
        found = None
        for (c0, v0), (c1, v1) in zip(frontier, frontier[1:]):
            if (v0 - target) * (v1 - target) <= 0 and v1 != v0:
                found = c0 + (target - v0) * (c1 - c0) / (v1 - v0)
                break
        per_subject[subject] = found
        differences[subject] = target - theirs_mean[subject][plateau]
    values = [v for v in per_subject.values() if v is not None]
    paired = list(differences.values())
    return {"ceiling": ceiling, "plateau_capacity": plateau,
            "per_subject": per_subject,
            "n_matched": len(values), "n": len(per_subject),
            "no_crossing": sorted(s for s, v in per_subject.items() if v is None),
            **{k: v for k, v in spread(values).items() if k != "n"},
            "paired_difference": {
                "per_subject": differences, "median": st.median(paired),
                "mean": st.mean(paired), "n": len(paired),
                "pathguard_lower": sum(1 for d in paired if d < 0)}}


def text_numbers(path: Path | None, pathguard: dict, panmixer: dict,
                 ceiling: str, plateau: str) -> dict:
    """The comparison figures the paper states in prose rather than a table."""
    out = {"ld_loss_pathguard": pathguard["ld_loss_per_pair"][ceiling],
           "ld_loss_panmixer": panmixer["ld_loss_per_pair"][plateau]}
    if path is None:
        return out
    extra = json.loads(path.read_text())
    ours = extra["af_loss_missing_excluded"]["pathguard"]
    theirs = extra["af_loss_missing_excluded"]["panmixer"]
    subjects = sorted(set(ours) & set(theirs))
    out |= {"af_loss_missing_excluded_pathguard": st.median(ours[s] for s in subjects),
            "af_loss_missing_excluded_panmixer": st.median(theirs[s] for s in subjects),
            "af_loss_missing_excluded_pathguard_lower":
                sum(1 for s in subjects if ours[s] < theirs[s]),
            "af_loss_missing_excluded_n": len(subjects)}
    share = extra.get("ld_loss_largest_pairs", {})
    if share:
        out |= {"ld_loss_largest_pairs": share.get("n_pairs"),
                "ld_loss_largest_pairs_share_panmixer":
                    st.median(share["panmixer"][s] for s in sorted(share["panmixer"])),
                "ld_loss_largest_pairs_share_pathguard":
                    st.median(share["pathguard"][s] for s in sorted(share["pathguard"]))}
    return out


def print_report(out: dict) -> None:
    def table(title, columns, rows):
        print(f"\n{title}")
        print("  " + " " * 22 + "".join(f"{c:>12}" for c in columns))
        for label, cells in rows:
            print("  " + f"{label:<22}" + "".join(f"{c:>12}" for c in cells))

    caps = [capacity_key(c) for c in CAPACITIES]
    ceil = [ceiling_key(c) for c in CEILINGS]
    for key, title in (("altered_fraction_panmixer_seed123",
                        "PanMixer fraction of alleles changed, seed 123"),
                       ("altered_fraction_panmixer_mean5",
                        "PanMixer fraction of alleles changed, mean of 5 draws")):
        block = out[key]
        table(title, caps, [(stat, [f"{block[c][stat]:.6f}" for c in caps])
                            for stat in ("median", "min", "max")])
    block = out["altered_fraction_pathguard"]
    table("PathGuard fraction of alleles changed", ceil,
          [(stat, [f"{block[c][stat]:.6f}" for c in ceil])
           for stat in ("median", "min", "max")])

    for tool, columns in (("pathguard", ceil), ("panmixer", caps)):
        block = out[tool]
        rows = []
        for key, fmt in (("beagle", "{:+.4f}"), ("gap_score", "{:,.0f}"),
                         ("gap_score_linked", "{:.0f}"), ("haplotype_gap", "{:,.0f}"),
                         ("haplotype_gap_linked", "{:.0f}"), ("membership", "{:.1f}"),
                         ("membership_at_rank_1", "{:.0f}"), ("af_loss", "{:.2e}"),
                         ("ld_loss_per_pair", "{:.2e}")):
            if key in block:
                rows.append((key, [fmt.format(block[key][c]) if c in block[key] else "-"
                                   for c in columns]))
        name = "PathGuard, by accuracy ceiling" if tool == "pathguard" \
            else "PanMixer, by capacity"
        table(f"{name} (median over individuals)", columns, rows)

    m = out["matched_capacity"]
    print(f"\nMatched capacity at ceiling {m['ceiling']}: {m['n_matched']} of {m['n']} "
          f"individuals, median {m['median']:.4f} (range {m['min']:.4f}-{m['max']:.4f})")
    d = m["paired_difference"]
    print(f"Paired difference, ours at {m['ceiling']} minus PanMixer's 5-draw mean at "
          f"capacity {m['plateau_capacity']}: median {d['median']:+.6f}, "
          f"PathGuard lower for {d['pathguard_lower']} of {d['n']}")
    t = out["text_numbers"]
    if "af_loss_missing_excluded_pathguard" in t:
        print(f"Missing genotypes excluded, median AF loss: "
              f"{t['af_loss_missing_excluded_pathguard']:.6f} (PathGuard) vs "
              f"{t['af_loss_missing_excluded_panmixer']:.6f} (PanMixer), PathGuard lower "
              f"for {t['af_loss_missing_excluded_pathguard_lower']} of "
              f"{t['af_loss_missing_excluded_n']}")
    print(f"Median LD loss: {t['ld_loss_pathguard']:.6f} vs {t['ld_loss_panmixer']:.6f}")
    if "ld_loss_largest_pairs_share_panmixer" in t:
        print(f"The largest {t['ld_loss_largest_pairs']} variant pairs carry "
              f"{t['ld_loss_largest_pairs_share_panmixer']:.0%} of PanMixer's LD loss")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=Path, required=True,
                   help="sources.json written by stage.py")
    p.add_argument("--obf", type=Path, required=True, help="PanMixer releases, seed 123")
    p.add_argument("--obf-draws", type=Path, default=None, help="seeds 124-127")
    p.add_argument("--fidelity", type=Path, required=True,
                   help="PathGuard's altered fraction per individual and ceiling")
    p.add_argument("--missing-excluded", type=Path, default=None,
                   help="JSON from missing_excluded.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chrom", type=int, default=21)
    p.add_argument("--ceiling", default="0.80", help="ceiling the matched capacity uses")
    p.add_argument("--plateau", default="0.9",
                   help="capacity PanMixer's frontier is flat at")
    a = p.parse_args()

    sources = json.loads(a.sources.read_text())
    ours_fraction = read_fidelity(a.fidelity)
    draws = read_obfuscation(a.obf, a.obf_draws, a.chrom)
    subjects = sorted(set(draws) & set(ours_fraction))
    if not subjects:
        raise SystemExit("no individual has both a PathGuard and a PanMixer release")

    seed123 = {s: {c: draws[s][c][123] for c in draws[s]} for s in subjects}
    mean5 = {s: {c: st.mean(draws[s][c].values()) for c in draws[s]} for s in subjects}
    n_draws = sorted({len(v) for s in subjects for v in draws[s].values()})

    out = {
        "chromosome": a.chrom, "regime": "cohort-member", "subjects": subjects,
        "ceilings": [ceiling_key(c) for c in CEILINGS],
        "capacities": [capacity_key(c) for c in CAPACITIES],
        "draws_per_cell": n_draws,
        "altered_fraction_panmixer_seed123": {
            capacity_key(c): spread([seed123[s][capacity_key(c)] for s in subjects])
            for c in CAPACITIES},
        "altered_fraction_panmixer_mean5": {
            capacity_key(c): spread([mean5[s][capacity_key(c)] for s in subjects])
            for c in CAPACITIES},
        "altered_fraction_panmixer_subject": {
            s: {"seed123": seed123[s], "mean5": mean5[s]} for s in subjects},
        "altered_fraction_pathguard": {
            ceiling_key(c): spread([ours_fraction[s][ceiling_key(c)] for s in subjects])
            for c in CEILINGS},
        "altered_fraction_pathguard_subject": ours_fraction,
        "pathguard": summarize_evaluators(sources, "pathguard", CEILINGS, a.chrom),
        "panmixer": summarize_evaluators(sources, "panmixer", CAPACITIES, a.chrom),
    }
    out["matched_capacity"] = matched_capacity(
        {s: ours_fraction[s] for s in subjects}, mean5, a.ceiling, a.plateau)
    out["text_numbers"] = text_numbers(a.missing_excluded, out["pathguard"],
                                       out["panmixer"], a.ceiling, a.plateau)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(subjects)} individuals, {n_draws} draws per PanMixer cell")
    print_report(out)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
