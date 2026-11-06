# PanMixer side of the cohort-member comparison

PanMixer (third-party, commit `c182c38`) runs unmodified from its own checkout, and
so do its evaluators. Everything here writes outside that checkout.

## Needs

- the PanMixer checkout and its `panmixer` conda environment (`PANMIXER`, `PYTHON`)
- PLINK 1.9 (`PLINK`), for the LD blocks
- the chr21 inputs: our pangenome cohort VCF, the 1000 Genomes 30x phased panel, the
  PanGenie callset (`PANGENOME_VCF`, `PANEL_VCF`, `PANGENIE_VCF`)
- Beagle `beagle.27Feb25.75f.jar` from the checkout's `downloaded_tools`, a JRE (`JAVA`),
  `bgzip`/`bcftools` (`BCFTOOLS_BIN`), and the SNP-only panel `--panel`
  (`1000g_snps_no_missing_renamed_norm.vcf.gz`)
- PathGuard releases in PanMixer's format, `<subject>/acc_0pXX/chr21/new_haplotypes.npy`
- `WORK` (default `$HOME/panmixer_cohort`): 45 GB for the starting data, 60 GB for releases

## Order

    ./prepare.sh                                    # starting data, 45 min
    ./obfuscate.sh                                  # 352 releases at seed 123, 16 min
    SEEDS="124 125 126 127" ./obfuscate.sh          # draw noise, 48 min
    python stage.py --root $WORK/staging --ours DIR --theirs $WORK/obf \
        --starting-data $WORK/starting_data
    ROOT=$WORK/staging ./evaluate.sh <evaluator> <experiment>
    python missing_excluded.py --sources $WORK/staging/sources.json --out results/missing.json
    python summarize.py --sources $WORK/staging/sources.json --obf $WORK/obf \
        --obf-draws $WORK/obf_draws --fidelity results/pathguard_altered_fraction.json \
        --missing-excluded results/missing.json --out results/panmixer_comparison.json

`stage.py` prints the experiment number for each evaluator and tool; run `evaluate.sh`
once per pair. Wall times for the PathGuard side, 308 rows: Beagle 2.3 h at 8 workers,
`gap_score`, `diploid_gap_score` and `MIA_privacy` 8 h at 8, `af_loss` and `ld_loss`
6 min. The PanMixer side is 352 rows and takes 3.5 h for the three linkage and
membership evaluators; Beagle there is one individual, so 8 rows.
`missing_excluded.py` takes 2 min at 22 workers.

Beagle's statistic is restricted to the cells named by `xsol.npy`, and PathGuard has no
cell-selection step, so both tools are staged with an all-ones mask to keep one
denominator across the comparison.
