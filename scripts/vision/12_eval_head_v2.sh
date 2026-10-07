#!/bin/bash
# Head v2 vision evaluation: same 2000 queries, damages and damage seed 42 as results/perquery/vision_damage_valid_B,
# head v2 on the query only: rooms removed (no walls), crop, cover x f 0.25/0.5/0.75, then the full plan.
# Gallery = frozen whiten-train (checkpoint's whitening).
# Writes only new files: results/vision_head_v2/perquery/<split>/ (refuses if not empty).
#
# Usage:
#   sbatch scripts/vision/12_eval_head_v2.sh valid
#   sbatch scripts/vision/12_eval_head_v2.sh test      # only with the TEST_PREREGISTERED_R2 gate

#SBATCH --job-name="vx_12_eval_head_v2"
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
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002
source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR" || exit 1
mkdir -p logs
export PYTHONUNBUFFERED=1

SPLIT="${1:-}"
case "$SPLIT" in valid|test) ;; *) echo "!! uso: 12_eval_head_v2.sh <valid|test>" >&2; exit 1 ;; esac
python -m src.evaluation.multiseed_runs gate --split "$SPLIT" || exit 1
[ -f embeddings/vision/pespatial/gem/head_v2.pt ] || { echo "!! manca head_v2.pt (11_train_head_v2.sh)" >&2; exit 1; }
D="results/vision_head_v2/perquery/$SPLIT"
if [ -n "$(ls -A "$D" 2>/dev/null)" ]; then echo "!! $D non vuota: niente sovrascritture" >&2; exit 1; fi
mkdir -p "$D"
TAG="vision_pespatial_gem_whiten-train+qhead-v2"
COMMON="model.name=pespatial model.variant=$(variant_for gem native) model.kwargs.pooling=gem \
  head.enabled=false whitening.enabled=true whitening.fit_split=train query_head.enabled=true query_head.file=head_v2.pt \
  eval.split=$SPLIT eval.perquery_dir=$D"
echo "=== $(date) | head v2: valutazione $SPLIT -> $D ==="
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
