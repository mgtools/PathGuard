#!/bin/bash
# Run one of PanMixer's evaluators, unmodified, over every row of a staged experiment.
#
#   ./evaluate.sh <evaluator> <experiment>
#
# evaluator is beagle, gap_score, diploid_gap_score, MIA_privacy, af_loss or ld_loss;
# experiment is the number stage.py printed. Each row runs in its own working
# directory because gap_score.py and diploid_gap_score.py both write test_hist.png
# relative to it. Rough wall times on 44 x 7 rows: Beagle 2.3 h at 8 workers,
# the two linkage scores and membership inference 8 h at 8, af_loss and ld_loss 6 min.
#
# The linkage evaluators load the masked 1000 Genomes panel (3.1 GB, and
# diploid_gap_score a second 3.2 GB array), so WORKERS stays low and the run
# stops rather than pushing a shared machine into swap.
set -u

evaluator=${1:?evaluator}
experiment=${2:?experiment number}
ROOT=${ROOT:-$HOME/panmixer_cohort/staging}
PANMIXER=${PANMIXER:-external/PanMixer}
PYTHON=${PYTHON:-python}
CHROM=${CHROM:-21}
WORKERS=${WORKERS:-8}
MEMORY_FLOOR_GB=${MEMORY_FLOOR_GB:-40}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGS=$ROOT/logs/${evaluator}_exp${experiment}
SCRATCH=$ROOT/scratch/${evaluator}_exp${experiment}
mkdir -p "$LOGS" "$SCRATCH" "$ROOT/mplconfig"

# Their __pycache__ holds a 3.12 .pyc and this environment is 3.11, so an import
# would write into the checkout.
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=$ROOT/mplconfig
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

case $evaluator in
  beagle)            result=accuracy_results.json ;;
  gap_score|diploid_gap_score) result=gap_score_all.json ;;
  MIA_privacy)       result=MIA_privacy.json ;;
  af_loss)           result=af_loss.json ;;
  ld_loss)           result=ld_loss.txt ;;
  *) echo "unknown evaluator: $evaluator" >&2; exit 2 ;;
esac

csv=$ROOT/experiments/exp_$experiment/data_chr$CHROM.csv
[ -s "$csv" ] || { echo "no staged experiment at $csv" >&2; exit 1; }
n=$(( $(wc -l < "$csv") - 1 ))
[ "$n" -gt 0 ] || { echo "exp_$experiment has no rows" >&2; exit 1; }

memory_gb () { awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo; }

run_one () {
  local row=$1 cwd=$SCRATCH/row$1
  mkdir -p "$cwd"
  case $evaluator in
    beagle)
      ( cd "$cwd" && "$PYTHON" -B "$HERE/beagle_one.py" --root "$ROOT" \
          --experiment "$experiment" --row "$row" --chrom "$CHROM" \
          > "$LOGS/row$row.log" 2>&1 ) ;;
    gap_score|diploid_gap_score|MIA_privacy)
      ( cd "$cwd" && PYTHONPATH="$ROOT" "$PYTHON" -B \
          "$PANMIXER/tools/downstream/privacy/$evaluator.py" "$experiment" "$CHROM" "$row" \
          > "$LOGS/row$row.log" 2>&1 ) ;;
    af_loss|ld_loss)
      ( cd "$cwd" && PYTHONPATH="$ROOT" "$PYTHON" -B \
          "$PANMIXER/tools/downstream/utility_in/$evaluator.py" \
          "$experiment" False "$CHROM" "$row" > "$LOGS/row$row.log" 2>&1 ) ;;
  esac
  echo $? > "$LOGS/row$row.status"
}

echo "$evaluator on exp_$experiment: $n rows, $WORKERS workers, started $(date -Is)"
if [ "$evaluator" = gap_score ] || [ "$evaluator" = diploid_gap_score ]; then
  "$PYTHON" -B -c "import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.figure()" > "$LOGS/mplwarm.log" 2>&1
fi

for row in $(seq 0 $((n - 1))); do
  while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 3; done
  if [ "$(memory_gb)" -lt "$MEMORY_FLOOR_GB" ]; then
    echo "MemAvailable $(memory_gb) GB is below the ${MEMORY_FLOOR_GB} GB floor; waiting"
    while [ "$(memory_gb)" -lt "$MEMORY_FLOOR_GB" ]; do sleep 30; done
  fi
  run_one "$row" &
done
wait

# Completeness from the files on disk, not from this loop's bookkeeping.
present=$(find "$ROOT/experiments/exp_$experiment/data/chr$CHROM" -name "$result" | wc -l)
echo "$result on disk: $present of $n"
for row in $(seq 0 $((n - 1))); do
  status=$(cat "$LOGS/row$row.status" 2>/dev/null || echo missing)
  [ "$status" = 0 ] || echo "  row $row exited $status"
done
echo "finished $(date -Is)"
[ "$present" -eq "$n" ]
