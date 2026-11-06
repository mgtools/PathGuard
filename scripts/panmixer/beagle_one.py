"""One row of the Beagle reconstruction attack, from a staged release to their statistic.

Their converter and their `accuracy_stats.py` do the work; this script only runs
Beagle in between. `beagle_refinement.py` builds a SLURM array, so the java
command is taken from its `get_command`. Beagle appends `.vcf.gz` to the `out=`
prefix, and their reader expects the resulting double extension.

    python beagle_one.py --root DIR --experiment 1 --row 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--chrom", type=int, default=21)
    p.add_argument("--panel", type=Path, required=True,
                   help="1000 Genomes SNP panel for Beagle, bgzipped and indexed")
    p.add_argument("--java", default=os.environ.get("JAVA", "java"))
    p.add_argument("--bcftools-bin", type=Path, default=Path(
        os.environ.get("BCFTOOLS_BIN", "")), help="directory holding bgzip and bcftools")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--xmx", default="12g")
    p.add_argument("--keep-vcf", action="store_true")
    a = p.parse_args()

    sys.path.insert(0, str(a.root))
    import numpy as np
    import pandas as pd
    from tools.beagle.accuracy_stats import accuracy_computer
    from tools.common.convert_2_vcf import produce_vcf_from_np

    exp_dir = a.root / "experiments" / f"exp_{a.experiment}"
    table = pd.read_csv(exp_dir / f"data_chr{a.chrom}.csv")
    subject = str(table.iloc[a.row]["subject"])
    row_dir = exp_dir / "data" / f"chr{a.chrom}" / str(a.row)
    jar = a.root / "downloaded_tools" / "beagle.27Feb25.75f.jar"
    print(f"=== row {a.row} {subject} ===", flush=True)

    env = dict(os.environ)
    env["PATH"] = f"{a.bcftools_bin}:{env['PATH']}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    vcf = row_dir / "new_haplotypes.vcf"
    if not (row_dir / "new_haplotypes.vcf.gz").exists():
        produce_vcf_from_np(
            np.load(row_dir / "new_haplotypes.npy"),
            str(a.root / "starting_data" / f"chr{a.chrom}" / "pangenome.vcf.gz"),
            subject, str(vcf))
        subprocess.run([str(a.bcftools_bin / "bgzip"), "-@2", "-f", str(vcf)],
                       check=True, env=env)
        subprocess.run([str(a.bcftools_bin / "bcftools"), "index", "-f", f"{vcf}.gz"],
                       check=True, env=env)

    start = time.time()
    out_prefix = row_dir / "beagle_refined.vcf.gz"
    done = subprocess.run(
        [a.java, f"-Xmx{a.xmx}", "-jar", str(jar), f"gt={vcf}.gz",
         f"ref={a.panel}", f"out={out_prefix}", f"nthreads={a.threads}"],
        capture_output=True, text=True, env=env)
    (row_dir / "beagle_run.log").write_text(done.stdout + "\n--- stderr ---\n" + done.stderr)
    if done.returncode != 0:
        print(f"  Beagle failed: {done.stderr[-2000:]}")
        return 1
    print(f"  Beagle done ({time.time() - start:.0f}s)", flush=True)

    cwd = Path.cwd()
    os.chdir(row_dir)
    try:
        accuracy_computer(a.experiment, str(a.chrom), a.row)
    finally:
        os.chdir(cwd)
    stats = json.loads((row_dir / "accuracy_results.json").read_text())["snp_only"]
    print(f"  beagle - major_allele = "
          f"{stats['beagle_accuracy'] - stats['major_allele_accuracy']:+.6f}")

    if not a.keep_vcf:
        for name in (f"{vcf}.gz", f"{vcf}.gz.csi"):
            Path(name).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
