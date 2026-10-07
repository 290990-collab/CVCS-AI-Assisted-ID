#!/bin/bash
# Vision test for ONE configuration already selected on valid. The test selects nothing: the three
# arguments are mandatory and there is no "all" mode (a grid on test is selection on test).
#
# Protocol B (final numbers):
#   - shared gallery and split come from the YAML; only eval.split=test is forced here;
#   - `partial` mode (4th argument): degradation curve on test with per-query `self_rr`;
#   - default folder `vision_test_B`: old-protocol files in `vision_test/` stay untouched;
#   - EXTRA for the head checkpoint, e.g. EXTRA="head.file=head_probe_conv.pt".
#
# Usage:
#   sbatch scripts/evaluation/03_perquery_vision_test.sh <encoder> <pooling> <raw|whiten|head|head+whiten> [full|partial]
#   EXTRA="head.file=head_probe_conv.pt" \
#     sbatch scripts/evaluation/03_perquery_vision_test.sh dinov3 natural head full
#   EXTRA="head.file=head_probe_conv.pt" \
#     sbatch scripts/evaluation/03_perquery_vision_test.sh dinov3 natural head partial
#   Env: RES (tipsv2: 224|448) · PERQUERY_DIR · EXTRA · STRAT (partial: random | all | crop | patch | damage)
#
# Output: $PERQUERY_DIR/vision_<enc>_<pool>_<tag>_full_test.npz
#         $PERQUERY_DIR/vision_<enc>_<pool>_<tag>_partial-<strat>-f<frac>_test.npz

#SBATCH --job-name="ev_03_perquery_vision_test"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Excludes the Blackwell nodes (sm_120): same whitelist as 05/06 and scripts/vision/*.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR" || exit 1

MODEL="${1:-}"
POOL="${2:-}"
TRANSFORM="${3:-}"
MODE="${4:-full}"
# Resolution: "native" = the YAML preset (all but tipsv2, which has two per pooling); must match the valid choice.
RES="${RES:-native}"
PERQUERY_DIR="${PERQUERY_DIR:-results/perquery/vision_test_B}"
EXTRA="${EXTRA:-}"
STRAT="${STRAT:-random}"
mkdir -p logs "$PERQUERY_DIR"

if [ -z "$MODEL" ] || [ -z "$POOL" ] || [ -z "$TRANSFORM" ]; then
  echo "!! ERRORE: servono encoder, pooling e trasformazione." >&2
  echo "   uso: sbatch $0 <encoder> <pooling> <raw|whiten|head|head+whiten> [full|partial]" >&2
  echo "   La configurazione va scelta PRIMA sul valid." >&2
  exit 1
fi

# transform -> the two keys defining it
case "$TRANSFORM" in
  raw)         HEAD=false; WHITEN=false ;;
  whiten)      HEAD=false; WHITEN=true  ;;
  head)        HEAD=true;  WHITEN=false ;;
  head+whiten) HEAD=true;  WHITEN=true  ;;
  *) echo "!! ERRORE: trasformazione '$TRANSFORM' non riconosciuta" >&2
     echo "   attese: raw | whiten | head | head+whiten" >&2
     exit 1 ;;
esac

# Mode: full (whole plan) or partial (rooms removed, degradation curve); strategies as in 05.
case "$MODE" in
  full)    MODE_FLAGS="partial.enabled=false" ;;
  partial)
    # crop/patch explicitly off in random/all; damage = random + crop + patch (as 05)
    OFF_ROOMS="partial.strategies.semantic.enabled=false partial.strategies.topology.enabled=false"
    case "$STRAT" in
      random) MODE_FLAGS="partial.enabled=true $OFF_ROOMS partial.strategies.crop.enabled=false partial.strategies.patch.enabled=false" ;;
      all)    MODE_FLAGS="partial.enabled=true partial.strategies.crop.enabled=false partial.strategies.patch.enabled=false" ;;
      crop)   MODE_FLAGS="partial.enabled=true partial.strategies.random.enabled=false $OFF_ROOMS partial.strategies.crop.enabled=true partial.strategies.patch.enabled=false" ;;
      patch)  MODE_FLAGS="partial.enabled=true partial.strategies.random.enabled=false $OFF_ROOMS partial.strategies.crop.enabled=false partial.strategies.patch.enabled=true" ;;
      damage) MODE_FLAGS="partial.enabled=true $OFF_ROOMS partial.strategies.crop.enabled=true partial.strategies.patch.enabled=true" ;;
      *) echo "!! ERRORE: STRAT '$STRAT' non riconosciuta (attese: random | all | crop | patch | damage)" >&2
         exit 1 ;;
    esac ;;
  *) echo "!! ERRORE: modalita' '$MODE' non riconosciuta (attese: full | partial)" >&2
     exit 1 ;;
esac

# pooling must be valid for the encoder (i-jepa has no `mean`), else the run measures something else
VALID_POOLS=$(poolings_for "$MODEL")
case " $VALID_POOLS " in
  *" $POOL "*) ;;
  *) echo "!! ERRORE: pooling '$POOL' non valido per '$MODEL' (validi: $VALID_POOLS)" >&2
     exit 1 ;;
esac

echo "=== $(date) | TEST (una volta sola) | $(label_for "$MODEL" "$POOL" "$RES") / $TRANSFORM / $MODE ==="
echo "per-query -> $PERQUERY_DIR | extra: ${EXTRA:-nessuno} | strategie: $([ "$MODE" = partial ] && echo "$STRAT" || echo -)"
echo "⚠️  Se questa non e' la configurazione scelta (e pre-registrata) sul valid,"
echo "    fermati: stai selezionando sul test."
nvidia-smi

# shellcheck disable=SC2086  # MODE_FLAGS and EXTRA must split into separate overrides
python -m src.vision.evaluation.evaluate \
  model.name="$MODEL" model.variant="$(variant_for "$POOL" "$RES")" \
  model.kwargs.pooling="$POOL" $(res_flags "$RES") \
  $MODE_FLAGS eval.split=test eval.perquery_dir="$PERQUERY_DIR" \
  head.enabled=$HEAD whitening.enabled=$WHITEN $EXTRA
RC=$?

echo ""
if [ $RC -eq 0 ]; then
  echo "=== completato: $(date) ==="
  ls -la "$PERQUERY_DIR"
  echo ""
  echo "Confronti appaiati col ramo graph (stessa gallery condivisa, nessun override):"
  echo "  full:    python -m src.evaluation.significance --a <vision>_full_test.npz --b <graph>_full_test.npz --k 10"
  echo "  partial: python -m src.evaluation.robustness_auc compare --split test \\"
  echo "             --a $PERQUERY_DIR/vision_<enc>_<pool>_<tag> --b results/perquery/graph_test_B/graph_<enc>_<var>"
else
  echo "=== FALLITO (rc=$RC): $(date) ===" >&2
fi
exit $RC
