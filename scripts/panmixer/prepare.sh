#!/bin/bash
# Build PanMixer's starting data for chr21 on our pangenome build.
#
# PanMixer's preprocessing scripts derive their output root from their own
# location, so they are copied into $WORK and run from there. Nothing in the
# PanMixer checkout is written.
#
#   ./prepare.sh
#
# Takes about 45 minutes, most of it converting the 1000 Genomes panel.
set -euo pipefail

PANMIXER=${PANMIXER:-external/PanMixer}
WORK=${WORK:-$HOME/panmixer_cohort}
PYTHON=${PYTHON:-python}
PLINK=${PLINK:-plink}
CHROM=${CHROM:-21}

# The three inputs: our pangenome cohort VCF, the 1000 Genomes phased panel and
# the PanGenie callset PanMixer anchors its HMM on.
PANGENOME_VCF=${PANGENOME_VCF:-data/cohort/chr$CHROM/cohort.vcf.gz}
PANEL_VCF=${PANEL_VCF:-data/raw/1000g_30x_phased.vcf.gz}
PANGENIE_VCF=${PANGENIE_VCF:-data/raw/PG_chr$CHROM.vcf.gz}

SD=$WORK/starting_data
SRC=$SD/scripts/src
mkdir -p "$SRC" "$WORK/tools/common" "$SD/chr$CHROM" "$WORK/logs"

step () { echo; echo "### [$(date +%H:%M:%S)] $*"; }

step "copying PanMixer's preprocessing scripts into $WORK"
for f in paths.py get_possible_alleles.py build_biallelic_snp_mask.py get_mappings.py \
         get_af.py get_blocks_simple.py verify_blocks.py apply_mask.py; do
  cp -f "$PANMIXER/starting_data/scripts/src/$f" "$SRC/$f"
done
cp -f "$PANMIXER/tools/common/VCFtoNP.py" "$WORK/tools/common/VCFtoNP.py"

step "linking inputs"
cd "$SD/chr$CHROM"
ln -sf "$PANGENOME_VCF" pangenome.vcf.gz
ln -sf "$PANGENOME_VCF.csi" pangenome.vcf.gz.csi
ln -sf "$PANEL_VCF" 1000g_phased.vcf.gz
ln -sf "$PANEL_VCF.tbi" 1000g_phased.vcf.gz.tbi
ln -sf "$PANGENIE_VCF" PG.vcf.gz
ln -sf "$PANGENIE_VCF.tbi" PG.vcf.gz.tbi
cd "$SD"

step "1/10 get_possible_alleles.py"
$PYTHON "$SRC/get_possible_alleles.py" "$CHROM"

step "2/10 pangenome VCF to numpy (phased)"
$PYTHON "$WORK/tools/common/VCFtoNP.py" "./chr$CHROM/pangenome.vcf.gz" true

step "3/10 build_biallelic_snp_mask.py -- fails if the VCF and the position array disagree"
$PYTHON "$SRC/build_biallelic_snp_mask.py" "$CHROM"

step "4/10 PanGenie callset to numpy (unphased)"
$PYTHON "$WORK/tools/common/VCFtoNP.py" "./chr$CHROM/PG.vcf.gz" false

step "5/10 1000 Genomes panel to numpy (phased) -- about 30 minutes, 12.8 GB"
$PYTHON "$WORK/tools/common/VCFtoNP.py" "./chr$CHROM/1000g_phased.vcf.gz" true

step "6/10 get_mappings.py"
$PYTHON "$SRC/get_mappings.py" "$CHROM"

step "7/10 get_af.py"
$PYTHON "$SRC/get_af.py" "$CHROM"

step "8/10 PLINK 1.9 bed and LD blocks"
"$PLINK" --version
"$PLINK" --vcf "./chr$CHROM/1000g_phased.vcf.gz" --make-bed \
         --out "./chr$CHROM/1000g_phased" --allow-extra-chr --vcf-half-call missing
"$PLINK" --blocks no-pheno-req --bfile "./chr$CHROM/1000g_phased" \
         --out "./chr$CHROM/blocks" --allow-extra-chr

step "9/10 get_blocks_simple.py, then verify_blocks.py (both counts must match)"
$PYTHON "$SRC/get_blocks_simple.py" "$CHROM"
$PYTHON "$SRC/verify_blocks.py" "$CHROM"

step "10/10 apply_mask.py"
$PYTHON "$SRC/apply_mask.py" "$CHROM"

step "done -- starting data in $SD/chr$CHROM"
ls -la "$SD/chr$CHROM"/*.npy "$SD/chr$CHROM"/*.pickle "$SD/chr$CHROM"/*.json
