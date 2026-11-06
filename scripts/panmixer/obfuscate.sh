#!/bin/bash
# PanMixer releases for the whole graph cohort: every subject at eight capacities,
# seed 123 plus four more draws.
#
# PanMixer runs unmodified from its own checkout. Seed 123 is the draw the
# evaluators are run on; seeds 124-127 exist only to measure draw noise in
# stats.json, so their VCFs (38 MB of each 41 MB run) are deleted.
#
#   ./obfuscate.sh            # seed 123, 352 runs, about 16 minutes at 40 workers
#   SEEDS="124 125 126 127" ./obfuscate.sh   # the extra draws, about 48 minutes
set -euo pipefail

PANMIXER=${PANMIXER:-external/PanMixer}
WORK=${WORK:-$HOME/panmixer_cohort}
PYTHON=${PYTHON:-python}
CHROM=${CHROM:-21}
WORKERS=${WORKERS:-40}
SEEDS=${SEEDS:-123}
CAPACITIES=${CAPACITIES:-"0.01 0.05 0.1 0.2 0.3 0.5 0.7 0.9"}

SD=$WORK/starting_data
LOGS=$WORK/logs
mkdir -p "$LOGS"

# numpy's BLAS spawns ~95 threads per process for a single-threaded run.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SUBJECTS=${SUBJECTS:-$($PYTHON -c "
import numpy as np
print(' '.join(np.load('$SD/chr$CHROM/pangenome_subjects.npy').tolist()))")}

cat > "$LOGS/run_one.sh" <<'RUNNER'
#!/bin/bash
subject=$1; capacity=$2; seed=$3
label=cap$(echo "$capacity" | tr . p)
if [ "$seed" = "123" ]; then out=$WORK/obf/$subject/$label
else out=$WORK/obf_draws/$subject/$label/s$seed; fi
mkdir -p "$out"
$PYTHON "$PANMIXER/tools/panmixer/obfuscate.py" \
    "$WORK/starting_data" "$subject" "$capacity" "$CHROM" "$out" --seed "$seed" \
    > "$out/run.log" 2>&1
status=$?
if [ "$seed" != "123" ]; then
  rm -f "$out/chr$CHROM/obfuscated.vcf.gz" "$out/chr$CHROM/obfuscated.vcf.gz.csi"
fi
echo "$status $subject $capacity $seed" >> "$WORK/logs/obfuscate_results.txt"
exit $status
RUNNER
chmod +x "$LOGS/run_one.sh"
export WORK PYTHON PANMIXER CHROM

jobs_file=$LOGS/obfuscate_jobs.txt
: > "$jobs_file"
for subject in $SUBJECTS; do
  for capacity in $CAPACITIES; do
    for seed in $SEEDS; do
      echo "$subject $capacity $seed" >> "$jobs_file"
    done
  done
done
want=$(wc -l < "$jobs_file")

echo "PanMixer releases: $want runs ($WORKERS workers), seeds: $SEEDS"
: > "$LOGS/obfuscate_results.txt"
xargs -P "$WORKERS" -n 3 -a "$jobs_file" "$LOGS/run_one.sh" || true

# Count the outputs on disk, not the jobs this script thinks it launched.
got=0
for subject in $SUBJECTS; do
  for capacity in $CAPACITIES; do
    label=cap$(echo "$capacity" | tr . p)
    for seed in $SEEDS; do
      if [ "$seed" = "123" ]; then f=$WORK/obf/$subject/$label/chr$CHROM/stats.json
      else f=$WORK/obf_draws/$subject/$label/s$seed/chr$CHROM/stats.json; fi
      [ -s "$f" ] && got=$((got + 1))
    done
  done
done
echo "stats.json on disk: $got of $want"
awk '$1 != 0' "$LOGS/obfuscate_results.txt" || true
[ "$got" -eq "$want" ]
