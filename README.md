# PathGuard

PathGuard releases a genome's path through a public pangenome graph in sanitized
form, with a bound set in advance on how well anyone can tell the release apart
from a decoy. You choose an **accuracy ceiling** `a` between 0.5 and 1 representing your allowable privacy risk. That accuracy ceiling sets an upper limit on pairwise distinguishability (whether a release came from genome p or q).  At `a = 0.5` the release is a random cohort mosaic and carries no information about the target whatsoever.

See the accompanying paper for the method, proof and the evaluation.

## Install

```bash
conda env create -f environment.yml && conda activate pathguard
pip install -e .
```

NumPy is the only runtime dependency. `bcftools` is needed only to build cohort
arrays from a graph, and `vg` only if you start from a GFA rather than a VCF.

## Quick start

`test_data/` holds a 500 kb slice of chromosome 21 — the 44-individual HPRC
cohort, six 1000 Genomes samples, the genetic map and the assembly gap table —
so the whole tool runs in a few seconds, with nothing to download:

```bash
pathguard cohort --vcf test_data/cohort.chr21.vcf.gz \
    --gmap test_data/chr21.b38.gmap.gz --gaps test_data/gap.chr21.txt.gz \
    --chrom chr21 --out work/cohort
```
```
{"samples": 44, "records": 4071, "chain": 4071}
```

```bash
pathguard target --vcf test_data/1000g.chr21.vcf.gz --sample HG00096 \
    --cohort work/cohort --out work/HG00096.npy
```
```
HG00096: 4071 records, called 3718 / 3718 per haplotype
```

```bash
pathguard calibrate --cohort work/cohort --target work/HG00096.npy
```
```
ceiling	eta	kl	achieved_ceiling	expected_altered_fraction
0.55	5.60748	0.02	0.550000	0.162777
0.60	11.2038	0.08	0.600000	0.160808
0.65	16.7924	0.18	0.650000	0.158832
0.70	22.3768	0.32	0.700000	0.156849
0.75	27.9602	0.5	0.750000	0.154861
0.80	33.5462	0.72	0.800000	0.152869
```

```bash
pathguard release --cohort work/cohort --target work/HG00096.npy \
    --ceiling 0.80 --seed 1 --chrom chr21 --out work/release.vcf.gz
```
```
ceiling 0.8: eta = 33.5462 (KL 0.72, achieved ceiling 0.800000)
wrote work/release.vcf.gz: 4071 positions, 0.1954 of called target alleles altered
```

`achieved_ceiling` is what the solved tilt actually delivers, and it should equal
the ceiling you asked for. `expected_altered_fraction` is exact, from the state
marginals rather than from sampling, so a single draw lands near it but not on
it. The released VCF is phased, one sample, over the chain positions; every
allele in it is one that some cohort haplotype carries there.

Run the tests the same way. They use only `test_data/` and take under a minute:

```bash
pytest tests/
```

## Commands

Four steps, in order: build the cohort arrays, project the target onto them,
solve the tilt, draw the release. `pathguard genome` is those steps over a whole
genome under one budget.

### pathguard cohort

Builds one chromosome's cohort arrays from the graph and the genetic map. It
reads no target genome, so the result is public and is reused for every target.

| flag | takes | description |
| --- | --- | --- |
| `--vcf` | a file | phased cohort VCF, one record per variant site. `vg deconstruct` output. This or `--gfa` is required |
| `--gfa` | a file | a graph to build that VCF from with `vg deconstruct`, instead of `--vcf` |
| `--vg` | a path | the `vg` binary, with `--gfa` only (default: `vg` on `$PATH`) |
| `--gfa-threads` | a number | threads for `vg deconstruct`, with `--gfa` only (default 16) |
| `--drop-sample` | a name | a sample in the graph to leave out of the cohort (default `chm13`) |
| `--gmap` | a file | **required.** Genetic map for this chromosome, columns `pos chr cM`, plain or gzipped |
| `--gaps` | a file | UCSC `gap.txt.gz`; assembly gaps get no genetic distance |
| `--chrom` | a name | **required.** The chromosome, spelled as the gap table spells it, e.g. `chr21` |
| `--out` | a directory | **required.** Where to write the arrays |

Writes `haplotypes.npy`, `positions.npy`, `genetic_pos.npy`, `in_chain.npy`,
`haplotype_ids.npy` and `sites.tsv`. `in_chain` marks the positions the mechanism
can release: those with at least one called donor, less any run of zero genetic
distance that no single donor spans, since the chain cannot switch donors inside
such a run. Where the map says nothing, genetic distance comes from a declared
constant rate of 1.2707 cM/Mb over base pairs that are not assembly gaps. That
rate is a stated convention of the method rather than a parameter, so it is not
a flag.

**Example**  
```bash
pathguard cohort --gfa data/raw/chr21.hprc-v1.0-pggb.gfa.gz \
    --gmap data/raw/chr21.b38.gmap.gz --gaps data/raw/gap.txt.gz \
    --chrom chr21 --out data/cohort/chr21
```

One chromosome per run, because the graphs and the maps come one per
chromosome. For a whole genome, loop:

```bash
for c in $(seq 1 22); do
  pathguard cohort --gfa data/raw/chr${c}.hprc-v1.0-pggb.gfa.gz \
      --gmap data/raw/chr${c}.b38.gmap.gz --gaps data/raw/gap.txt.gz \
      --chrom chr${c} --out data/cohort/chr${c}
done
```

### pathguard target

Re-expresses a target genome on the cohort's record axis — the same positions, in
the same order, with the same allele numbering — which is the form the mechanism
reads.

| flag | takes | description |
| --- | --- | --- |
| `--vcf` | a file | **required.** Phased GRCh38 VCF holding the target, such as one chromosome of a 1000 Genomes panel |
| `--sample` | a name | **required.** The sample column inside `--vcf` that is the target |
| `--cohort` | a directory | **required.** The cohort directory `pathguard cohort` wrote |
| `--out` | a file | **required.** Output `.npy`: one allele index per record per haplotype, `-1` where the VCF has no call |

Records match on position, reference and alternate allele. A panel that splits a
multi-allelic site into one biallelic record per alternate allele is recovered by
matching each alternate separately.

**Example**  
```bash
pathguard target --vcf data/raw/1000g_30x_chr21.vcf.gz --sample HG00096 \
    --cohort data/cohort/chr21 --out data/targets/HG00096/chr21.npy
```

### pathguard calibrate

Solves the tilt that meets each accuracy ceiling, and reports what that tilt
costs in utility. It draws nothing, so several ceilings at once are cheap.

| flag | takes | description |
| --- | --- | --- |
| `--cohort` | one or more directories | **required.** A cohort directory per chromosome |
| `--target` | one or more files | **required.** A target `.npy` per chromosome, in the same order as `--cohort` |
| `--ceiling` | one or more numbers | the ceilings to solve, each in `[0.5, 1)` (default `0.55 … 0.80`) |
| `--leave-out` | a name | a cohort member to hold out of the donor panel, when the target is itself a member |

One chromosome calibrates that chromosome alone. Several are treated as one
release of one person: a single budget is divided among them in proportion to
chain length, because releasing somebody's chromosomes under separate budgets
leaks once per chromosome. `pathguard genome` does this for a whole genome
without listing the pairs.

**Example**  
```bash
pathguard calibrate --cohort data/cohort/chr21 \
    --target data/targets/HG00096/chr21.npy --ceiling 0.60 0.80
```

### pathguard release

Draws one release for one chromosome, as a phased one-sample VCF.

| flag | takes | description |
| --- | --- | --- |
| `--cohort` | a directory | **required.** The cohort directory `pathguard cohort` wrote |
| `--target` | a file | **required.** The target `.npy` `pathguard target` wrote |
| `--ceiling` | a number | the accuracy ceiling to calibrate to, in `[0.5, 1)`. This or `--eta` is required |
| `--eta` | a number | a tilt already solved by `pathguard calibrate`, instead of `--ceiling` |
| `--seed` | a number | **required.** Seed for this draw; the same seed gives the same release |
| `--chrom` | a name | the `CHROM` written into the output VCF, and part of the draw's key (default `chr21`) |
| `--sample` | a name | the sample name written into the output VCF (default `RELEASE`) |
| `--out` | a file | **required.** Output VCF, `.vcf` or `.vcf.gz` |
| `--leave-out` | a name | a cohort member to hold out of the donor panel, when the target is itself a member |

The draw is keyed on the chromosome name as well as the seed, so one seed gives
each chromosome its own stream and a chromosome's release does not depend on
whether it was drawn alone or as part of a genome.

**Example**  
```bash
pathguard release --cohort data/cohort/chr21 \
    --target data/targets/HG00096/chr21.npy \
    --ceiling 0.80 --seed 1 --chrom chr21 --sample HG00096 \
    --out releases/HG00096.chr21.vcf.gz
```

### pathguard genome

Releases a whole genome under one budget: one tilt solved across every
chromosome, each chromosome weighted by its share of the chain, then one VCF
drawn per chromosome.

| flag | takes | description |
| --- | --- | --- |
| `--cohort-root` | a directory | **required.** Holds one cohort directory per chromosome, named `chr1` … `chr22` |
| `--target-dir` | a directory | **required.** Holds one target `.npy` per chromosome, named `chr1.npy` … `chr22.npy` |
| `--chroms` | one or more names | which chromosomes to release (default: every one found under `--cohort-root`) |
| `--ceiling` | a number | **required.** The accuracy ceiling for the whole genome, in `[0.5, 1)` |
| `--seed` | a number | **required.** Seed for the draw; each chromosome takes its own stream from it |
| `--sample` | a name | the sample name written into the output VCFs (default `RELEASE`) |
| `--out` | a directory | **required.** Where to write `chr1.vcf.gz` … `chr22.vcf.gz` |
| `--leave-out` | a name | a cohort member to hold out of the donor panel, when the target is itself a member |

Both directories are what `pathguard cohort --out` and `pathguard target --out`
already write, so a genome takes those two in a loop and then this once:

**Example**  
```bash
for c in $(seq 1 22); do
  pathguard target --vcf data/raw/1000g_30x_chr${c}.vcf.gz --sample HG00096 \
      --cohort data/cohort/chr${c} --out data/targets/HG00096/chr${c}.npy
done

pathguard genome --cohort-root data/cohort --target-dir data/targets/HG00096 \
    --ceiling 0.80 --seed 1 --sample HG00096 --out releases/HG00096
```

It prints the weight each chromosome received and what the release cost it. On
the test slice, with the quick start's cohort directory copied to
`work/genome/chr21` and its target to `work/targets/chr21.npy`:

```bash
pathguard genome --cohort-root work/genome --target-dir work/targets \
    --ceiling 0.80 --seed 1 --sample HG00096 --out work/wgs
```
```
chr21: 4071 chain positions
ceiling 0.8: eta = 33.5462 (KL 0.72, achieved ceiling 0.800000)
chrom	chain	share	altered_fraction
chr21	4071	1.000000	0.195401
genome	4071	1.000000	0.195401
```

One chromosome gets the whole weight, so that is the same release `pathguard
release` drew above. With 22 chromosomes under the root, the shares are their
chain lengths and the last row is the genome.

The ceiling is the genome's, not each chromosome's. One budget spread over every
chromosome means a lower tilt per position, which is why a genome-wide release
alters more of the target than a single-chromosome release at the same ceiling.
It holds every chain in memory at once, roughly 7 GB for all 22 autosomes.

## The data the paper uses

None of this is needed to run PathGuard or its tests — `test_data/` covers that.
These are the four public inputs behind the paper's numbers, 44 GB in total.
Each command writes into `data/raw/`.

The pangenome graphs, HPRC v1.0 PGGB, one GFA per chromosome, 15 GB:

```bash
mkdir -p data/raw && for c in $(seq 1 22); do curl -fL -C - -o data/raw/chr${c}.hprc-v1.0-pggb.gfa.gz "https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/freeze1/pggb/chroms/chr${c}.hprc-v1.0-pggb.gfa.gz"; done
```

The genetic maps, SHAPEIT4's GRCh38 maps, 23 MB, which unpack as
`chr1.b38.gmap.gz` … `chr22.b38.gmap.gz`:

```bash
curl -fL -o data/raw/genetic_maps.b38.tar.gz https://github.com/odelaneau/shapeit4/raw/master/maps/genetic_maps.b38.tar.gz && tar -xzf data/raw/genetic_maps.b38.tar.gz -C data/raw
```

The assembly gap table, UCSC hg38, 12 KB, one file covering every chromosome:

```bash
curl -fL -o data/raw/gap.txt.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/gap.txt.gz
```

The 1000 Genomes 30× GRCh38 phased panel, 28 GB, which supplies the external
targets and the decoy pool:

```bash
mkdir -p data/raw && for c in $(seq 1 22); do curl -fL -C - -o data/raw/1000g_30x_chr${c}.vcf.gz "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr${c}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"; done
```

`curl -C -` resumes a partial file, so an interrupted download restarts with the
same command. The panel is the GRCh38-called one on purpose: the pangenome is
GRCh38, and a GRCh37 panel matches almost none of its records.

## Reproducing the paper

The scripts read the cohort arrays from `data/cohort/<chrom>/` and the target
paths from `data/targets/<sample>/<chrom>/path.npy`, which is the one layout they
expect (`--cohort-root` and `--paths` move it). Note that it nests one directory
deeper than the `pathguard genome` layout above:

```bash
for c in $(seq 1 22); do
  pathguard cohort --gfa data/raw/chr${c}.hprc-v1.0-pggb.gfa.gz \
      --gmap data/raw/chr${c}.b38.gmap.gz --gaps data/raw/gap.txt.gz \
      --chrom chr${c} --out data/cohort/chr${c}
  while read -r s; do
    pathguard target --vcf data/raw/1000g_30x_chr${c}.vcf.gz --sample "$s" \
        --cohort data/cohort/chr${c} --out data/targets/$s/chr${c}/path.npy
  done < configs/external_targets.txt
done
```

`configs/` lists the samples: ten external targets, the 44 cohort members, and
the 3,075 eligible decoy genomes left after removing the targets, the cohort and
anyone with a first-degree relative in the panel. A cohort member needs no
projection — its path is its own two rows of the graph — so only the external
targets and the decoys go through `pathguard target`. Chromosome 21 alone is
enough for every table except 2, 3 and 7.

Every script below takes `--workers N` (default 8), which spreads its independent
jobs over N processes — the trials of an attack, the target-by-ceiling pairs of a
utility sweep. That is the only parallelism: the scripts pin the BLAS libraries
to one thread each, because a multithreaded reduction is not bit-reproducible.

### External targets

```bash
# tilts: chromosome 21 as the whole release, then all 22 autosomes as one release
python scripts/calibrate.py --targets configs/external_targets.txt \
    --probes 227.1 454.19 --refinements 1 --workers 24 --out results/etas_chr21.json
python scripts/calibrate.py --targets configs/external_targets.txt --chroms all \
    --probes 1654.3 3308.5 --refinements 0 --workers 24 --out results/etas_genome.json

# decoys: from the external panel, and from the other nine targets
python scripts/select_decoys.py --targets configs/external_targets.txt \
    --pool configs/decoy_pool.txt --out results/decoys_panel.json
python scripts/select_decoys.py --targets configs/external_targets.txt \
    --pool configs/external_targets.txt --out results/decoys_targets.json

# utility, exact (Table 7)
python scripts/utility.py --targets configs/external_targets.txt --chroms all \
    --etas results/etas_genome.json --workers 24 --out results/utility_genome.json

# divergence against every decoy condition (Table 9)
python scripts/envelope.py --etas results/etas_chr21.json \
    --decoys results/decoys_targets.json --workers 24 --out results/envelope_chr21.json

# what the universal bound of Theorem 1 buys genome-wide (Tables 2 and 3)
python scripts/tau_regime.py --target HG00096 --workers 24 --out results/tau_regime.json

# attacks (Tables 1 and 4)
python scripts/attack.py --etas results/etas_chr21.json \
    --decoys results/decoys_panel.json --attacks lr --seed 50000 \
    --draws 100 --workers 24 --out results/attack_lr.json
python scripts/attack.py --etas results/etas_chr21.json \
    --decoys results/decoys_panel.json --attacks linkage block --seed 70000 34 \
    --draws 100 --workers 24 --out results/attack_linkage_block.json
```

The seeds and probes above are the ones the published runs used.

**What comes out identical, and what does not.** The releases do: every draw is a
function of its seed, and the seeds are derived from the run seed above, so the
released haplotypes are the published arrays bit for bit. So are the decoy
choices. The solved tilts are not, quite: the solve is iterative, and reading a
tilt off a fit from different probe points lands a different last few digits —
`eta` agrees with the published value to about ten significant figures, and the
quantity the paper actually claims, the achieved accuracy ceiling, agrees to
within 1e-7 of the requested ceiling, the same as the published run. That
difference is far too small to move a release: the released path at the published
tilt and at the recomputed tilt are the same array.

### Cohort members, and the comparison with PanMixer

```bash
python scripts/calibrate.py --targets configs/cohort_members.txt --cohort-members \
    --genome-share --probes 15000 30000 --refinements 1 --workers 24 \
    --out results/etas_cohort.json
python scripts/utility.py --targets configs/cohort_members.txt --cohort-members \
    --etas results/etas_cohort.json --workers 24 --out results/utility_cohort.json
python scripts/release.py --targets configs/cohort_members.txt --cohort-members \
    --etas results/etas_cohort.json --out data/releases/cohort \
    --panmixer-format --workers 24
```

`scripts/panmixer/` runs PanMixer and its published evaluators, unmodified, on
both tools' releases; see the README there. It needs PanMixer's own checkout and
environment, PLINK and Beagle.

### Cost

Measured on a 48-core workstation with several of these running at once, so a
quiet machine should be faster:

| step | workers | wall time |
| --- | --- | --- |
| `calibrate.py`, 10 external targets, chr21 | 24 | 4 min |
| `calibrate.py`, 44 cohort members, genome budget share | 20 | 38 min |
| `select_decoys.py`, the 3,075-genome panel | 8 | 1 min |
| `envelope.py`, Table 9 | 8 | 27 min |
| `utility.py`, 44 cohort members, Table 5 | 20 | 4 min |
| `release.py`, one member, 7 ceilings | 7 | 2 min |
| `tau_regime.py`, 22 autosomes, Tables 2 and 3 | 11 | 45 min |

The genome-wide runs are the expensive ones: the external-target tilts over 22
autosomes take a couple of hours and the genome-wide utility sweep several more.
The attack tables are the long pole — 7,000 trials, each drawing five releases of
chromosome 21 — and from a measured 14-trial slice, budget the better part of a
day for each of the two runs. Building the cohort arrays for all 22 autosomes is
an overnight job, most of it `vg deconstruct`. The PanMixer comparison is roughly
a day, most of it Beagle and the linkage evaluators.

## Layout

```
pathguard/        the mechanism: model, release, calibration, attacks, data loading
  model.py        the copying model, the tilt, forward filtering, backward sampling,
                  and the exact mean and variance of a released path
  release.py      drawing a release, its utility and log normaliser
  calibrate.py    exact divergence, the KL budget, solving the tilt
  attacks.py      the three attacks and their balanced accuracy
  cohort.py       graph and genetic map to cohort arrays; the chain
  target.py       projecting a phased genome onto the cohort records
  cli.py          the command line
scripts/          the paper's experiments
scripts/panmixer/ the comparison against PanMixer
tests/            the mechanism against exhaustive enumeration, and the tool on test_data
test_data/        a 500 kb slice of chromosome 21, enough to run everything
configs/          the sample lists the experiments use
```

## Citation

Smith D, Ye Y, Tang H. *Privacy-Preserving Genome Path Release against a Public
Pangenome Graph.*

## License

MIT, in `LICENSE`. The data under `test_data/` is a slice of the public sources
listed above and carries their terms, not this one.
