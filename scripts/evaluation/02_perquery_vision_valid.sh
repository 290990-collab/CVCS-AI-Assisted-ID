#!/bin/bash
# Vision grid on VALID with per-query persistence (`eval.perquery_dir`, needed for paired tests).
# Selection among encoder x pooling x transform is made here; the test is touched once, by
# 03_perquery_vision_test.sh.
#
# Coverage: 5 encoders x valid poolings (14 combos) x 4 transforms {raw, whiten, head, head+whiten}
# = 56 evaluations, one .npz each, tagged <encoder>_<pooling>_<transform>.
#
# Warning: the 56-run grid is the five historical encoders. Three encoders were added in select_models
# (_common.sh): `tipsv2` (6 features: 3 poolings x 2 resolutions, +24), `pecore` (3, fixed 224, +12),
# `pespatial` (3 at 224, +12). Without arguments this script now runs 104 and OVERWRITES existing .npz.
# To evaluate only the new encoder pass it explicitly: `... 02_...sh pespatial`.
#
# Warning: the 28 head rows depend on the head checkpoint (selected on InfoNCE val-loss); if the head is
# retrained with a retrieval probe they must be redone, hence the selectable group (`frozen` first, `head` later).
#
# Usage:
#   sbatch scripts/evaluation/02_perquery_vision_valid.sh              # all models, all 4 transforms
#   sbatch scripts/evaluation/02_perquery_vision_valid.sh dinov3       # one encoder
#   sbatch scripts/evaluation/02_perquery_vision_valid.sh "" frozen    # raw + whiten only (no head)
#   sbatch scripts/evaluation/02_perquery_vision_valid.sh dinov3 head  # head + head+whiten only
#   RES=448 sbatch scripts/evaluation/02_perquery_vision_valid.sh tipsv2  # one resolution only
#
# Output: results/perquery/vision_valid/vision_<enc>_<pool>_<transform>_full_valid.npz

#SBATCH --job-name="ev_02_perquery_vision_valid"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Excludes the Blackwell nodes (sm_120): same whitelist as 05.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR" || exit 1
mkdir -p logs results/perquery/vision_valid

MODEL_ARG="${1:-}"
GROUP="${2:-all}"
# Resolution: empty = all of the model (one for the five historical encoders); RES=448 restricts to one.
RES_ARG="${RES:-}"
# Env overrides: PERQUERY_DIR separates protocol B (shared gallery, whiten-train) from the historical
# vision_valid files (same names for raw/head); EXTRA = dotlist override (e.g. "head.file=head_probe_conv.pt");
# POOLS = poolings (empty = all).
PERQUERY_DIR="${PERQUERY_DIR:-results/perquery/vision_valid}"
EXTRA="${EXTRA:-}"
POOLS="${POOLS:-}"
mkdir -p "$PERQUERY_DIR"

case "$GROUP" in
  all|frozen|head) ;;
  *) echo "!! ERRORE: gruppo '$GROUP' non riconosciuto (attesi: all | frozen | head)" >&2
     exit 1 ;;
esac

echo "=== $(date) | GRIGLIA VISION su VALID (fase B.1) ==="
echo "modelli: $(select_models "$MODEL_ARG") | gruppo: $GROUP | per-query -> $PERQUERY_DIR"
nvidia-smi

FAILED=0

# run_eval <label> <override...>
run_eval() {
  local label="$1"; shift
  echo ""
  echo "########  VALID: $label  ########"
  python -m src.vision.evaluation.evaluate "$@"
  local rc=$?
  [ $rc -ne 0 ] && { echo "!! FALLITA: $label (rc=$rc)" >&2; FAILED=$((FAILED + 1)); }
}

for MODEL in $(select_models "$MODEL_ARG"); do
  for POOL in ${POOLS:-$(poolings_for "$MODEL")}; do
    for RES in $(select_resolutions "$MODEL" "$RES_ARG"); do
      VAR=$(variant_for "$POOL" "$RES")
      LBL=$(label_for "$MODEL" "$POOL" "$RES")
      # same keys as scripts/vision/04 and 05; only split and per-query differ
      COMMON="model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $(res_flags "$RES")"
      COMMON="$COMMON partial.enabled=false eval.split=valid eval.perquery_dir=$PERQUERY_DIR $EXTRA"

      if [ "$GROUP" = "all" ] || [ "$GROUP" = "frozen" ]; then
        run_eval "$LBL/raw"    $COMMON head.enabled=false whitening.enabled=false
        run_eval "$LBL/whiten" $COMMON head.enabled=false whitening.enabled=true
      fi

      if [ "$GROUP" = "all" ] || [ "$GROUP" = "head" ]; then
        run_eval "$LBL/head"        $COMMON head.enabled=true whitening.enabled=false
        run_eval "$LBL/head+whiten" $COMMON head.enabled=true whitening.enabled=true
      fi
    done
  done
done

echo ""
echo "=== completato: $(date) | valutazioni fallite: $FAILED ==="
ls -la "$PERQUERY_DIR" | tail -20
echo ""
echo "PROSSIMO PASSO — scegliere la vincente SUL VALID, poi confronti appaiati:"
echo "  python -m src.evaluation.significance --a $PERQUERY_DIR/<A>.npz --b $PERQUERY_DIR/<B>.npz --k 10"
echo "e SOLO DOPO il test una volta sola:"
echo "  sbatch scripts/evaluation/03_perquery_vision_test.sh <encoder> <pooling> <raw|whiten|head|head+whiten>"

[ $FAILED -eq 0 ] || exit 1
