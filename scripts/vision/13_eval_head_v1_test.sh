#!/bin/bash
# Test evaluation of the current head (head_nowalls_conv.pt; head + whitening fit on train after the head),
# mirroring results/perquery/vision_damage_valid_B and vision_full_valid_B: rooms removed (no walls), crop,
# cover x f 0.25/0.5/0.75 (damage seed 42), then the full plan. Same queries and damages as the frozen test files
# (results/perquery/vision_test_B) and the head v2 test (12_eval_head_v2.sh test).
# Writes only new files: results/vision_head_v1/perquery/test/ (refuses if not empty). Gate: TEST_PREREGISTERED_R2.
#
# Usage:
#   sbatch scripts/vision/13_eval_head_v1_test.sh

#SBATCH --job-name="vx_13_eval_head_v1_test"
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G"

set -uo pipefail
umask 002
source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR" || exit 1
mkdir -p logs
export PYTHONUNBUFFERED=1

SPLIT=test
python -m src.evaluation.multiseed_runs gate --split "$SPLIT" || exit 1
[ -f embeddings/vision/pespatial/gem/head_nowalls_conv.pt ] || { echo "!! manca head_nowalls_conv.pt" >&2; exit 1; }
D="results/vision_head_v1/perquery/$SPLIT"
if [ -n "$(ls -A "$D" 2>/dev/null)" ]; then echo "!! $D non vuota: niente sovrascritture" >&2; exit 1; fi
mkdir -p "$D"
TAG="vision_pespatial_gem_head-nowalls-conv+whiten-train"
COMMON="model.name=pespatial model.variant=$(variant_for gem native) model.kwargs.pooling=gem \
  head.enabled=true head.file=head_nowalls_conv.pt whitening.enabled=true whitening.fit_split=train query_head.enabled=false \
  eval.split=$SPLIT eval.perquery_dir=$D"
echo "=== $(date) | head attuale: valutazione $SPLIT -> $D ==="
nvidia-smi
START_MARK=$(mktemp)
FAILED=0
python -m src.vision.evaluation.evaluate $COMMON partial.enabled=true partial.seed=42 \
  partial.strategies.random.enabled=false partial.strategies.semantic.enabled=false \
  partial.strategies.topology.enabled=false partial.strategies.crop.enabled=true partial.strategies.patch.enabled=true \
  partial.strategies.nowalls_random.enabled=true "partial.strategies.nowalls_random.fractions=[0.25,0.5,0.75]" \
  partial.strategies.nowalls_semantic.enabled=false partial.strategies.nowalls_topology.enabled=false || FAILED=1
python -m src.vision.evaluation.evaluate $COMMON partial.enabled=false || FAILED=1

MISSING=0
for S in nowalls-random crop patch; do for F in 0.25 0.5 0.75; do
  P="$D/${TAG}_partial-${S}-f${F}_${SPLIT}.npz"
  [ -f "$P" ] && [ "$P" -nt "$START_MARK" ] || { echo "!! MANCANTE: $P" >&2; MISSING=$((MISSING + 1)); }
done; done
P="$D/${TAG}_full_${SPLIT}.npz"
[ -f "$P" ] && [ "$P" -nt "$START_MARK" ] || { echo "!! MANCANTE: $P" >&2; MISSING=$((MISSING + 1)); }
rm -f "$START_MARK"
echo "=== completato: $(date) | fallite: $FAILED | file mancanti: $MISSING ==="
[ $FAILED -eq 0 ] && [ $MISSING -eq 0 ] || exit 1
