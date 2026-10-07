#!/bin/bash
# Degradation curve under masking on VALID, with per-query persistence.
#
# "Best" = most robust: AUC of the self-recovery MRR curve. scripts/vision/06 and 07 do not set
# `eval.perquery_dir` and discard per-query values, so the paired delta is not computable; this is the
# partial twin of 02_perquery_vision_valid.sh.
#
# Differences from 02:
#   partial.enabled=true  (02 hard-wires false)
#   dedicated output folder: results/perquery/vision_partial_valid
#   masking strategies selected by $3, default `random`
#
# One file per (config x strategy x fraction): `vision_<tag>_partial-<strategy>-f<fraction>_valid.npz`
# (evaluate.py:219-221); with `random` only, 4 files per config (f = 0.0/0.25/0.5/0.75, configs/vision_retrieval.yaml).
#
# Much heavier than full (every query is re-degraded and re-extracted): ONE MODEL PER JOB, as 06 and 07.
#
# The `head` group needs head.pt, which exists only for the 5 historical encoders (dinov2, dinov3, siglip2,
# radio, ijepa); for tipsv2/pecore/pespatial use `frozen`. The heads on disk are selected on InfoNCE
# val-loss: if retrained with a retrieval probe the `head` group must be redone, hence the separate selection.
#
# Usage:
#   sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3
#   sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh pespatial frozen
#   sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 head
#   sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 all all
#   RES=448 sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh tipsv2 frozen
#   TRANSFORMS="raw head" sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 all damage
#     $1 = encoder (empty = all: discouraged, very long)
#     $2 = all | frozen | head        (default: all)
#     $3 = random | all | crop | patch | damage   (default: random, the only one the robustness criterion needs)
#          crop/patch (src/vision/data/vision_damage.py): rectangle on the snapshot / whole ViT patches;
#          damage = random + crop + patch
#     env TRANSFORMS = filter within the group (raw | whiten | head | head+whiten, space or comma separated);
#          empty = all of the group
#
# Output: results/perquery/vision_partial_valid/vision_<enc>_<pool>_<transform>_partial-<strat>-f<frac>_valid.npz
# Analysis (CPU, after): paired comparison at the SAME masking level
#   python -m src.evaluation.significance --metric self_rr --a <A>...f0.75_valid.npz --b <B>...f0.75_valid.npz

#SBATCH --job-name="ev_05_perquery_vision_partial_valid"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Excludes the Blackwell nodes (sm_120), where floorplan-env PyTorch crashes at every forward; same whitelist as scripts/vision/04-07.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR" || exit 1
mkdir -p logs results/perquery/vision_partial_valid

MODEL_ARG="${1:-}"
GROUP="${2:-all}"
STRAT="${3:-random}"
RES_ARG="${RES:-}"
# Env overrides: PERQUERY_DIR separates protocol-B runs (shared gallery, exclude_self, whiten-train) from
# the historical files (same names for raw/head); EXTRA adds dotlist overrides, e.g. EXTRA="whitening.fit_split=all".
PERQUERY_DIR="${PERQUERY_DIR:-results/perquery/vision_partial_valid}"
EXTRA="${EXTRA:-}"
POOLS="${POOLS:-}"          # empty = all poolings of the encoder (e.g. POOLS=natural)
TRANSFORMS="${TRANSFORMS:-}"  # empty = all transforms of the group (e.g. TRANSFORMS="raw head")
TRANSFORMS="${TRANSFORMS//,/ }"
mkdir -p "$PERQUERY_DIR"

case "$GROUP" in
  all)    GROUP_TRANSFORMS="raw whiten head head+whiten" ;;
  frozen) GROUP_TRANSFORMS="raw whiten" ;;
  head)   GROUP_TRANSFORMS="head head+whiten" ;;
  *) echo "!! ERRORE: gruppo '$GROUP' non riconosciuto (attesi: all | frozen | head)" >&2
     exit 1 ;;
esac

for T in $TRANSFORMS; do
  case "$T" in
    raw|whiten|head|head+whiten) ;;
    *) echo "!! ERRORE: trasformazione '$T' in TRANSFORMS non riconosciuta (attese: raw | whiten | head | head+whiten)" >&2
       exit 1 ;;
  esac
done

# want <transform>: true if TRANSFORMS is empty or contains it.
want() {
  [ -z "$TRANSFORMS" ] && return 0
  case " $TRANSFORMS " in *" $1 "*) return 0 ;; esac
  return 1
}

N_SELECTED=0
for T in $GROUP_TRANSFORMS; do want "$T" && N_SELECTED=$((N_SELECTED + 1)); done
if [ "$N_SELECTED" -eq 0 ]; then
  echo "!! ERRORE: 0 run selezionate (gruppo '$GROUP' = $GROUP_TRANSFORMS, TRANSFORMS='$TRANSFORMS')" >&2
  exit 1
fi

# `random` is the only strategy of the robustness criterion; `semantic` and `topology` triple the cost and
# are enabled on request only. crop/patch (non-room damage) are explicitly off in random/all.
OFF_ROOMS="partial.strategies.semantic.enabled=false partial.strategies.topology.enabled=false"
case "$STRAT" in
  random) STRAT_FLAGS="$OFF_ROOMS partial.strategies.crop.enabled=false partial.strategies.patch.enabled=false" ;;
  all)    STRAT_FLAGS="partial.strategies.crop.enabled=false partial.strategies.patch.enabled=false" ;;
  crop)   STRAT_FLAGS="partial.strategies.random.enabled=false $OFF_ROOMS partial.strategies.crop.enabled=true partial.strategies.patch.enabled=false" ;;
  patch)  STRAT_FLAGS="partial.strategies.random.enabled=false $OFF_ROOMS partial.strategies.crop.enabled=false partial.strategies.patch.enabled=true" ;;
  damage) STRAT_FLAGS="$OFF_ROOMS partial.strategies.crop.enabled=true partial.strategies.patch.enabled=true" ;;
  *) echo "!! ERRORE: strategie '$STRAT' non riconosciute (attesi: random | all | crop | patch | damage)" >&2
     exit 1 ;;
esac

echo "=== $(date) | PARTIAL per-query su VALID (criterio A.5, status.md § 23) ==="
echo "modelli: $(select_models "$MODEL_ARG") | gruppo: $GROUP | strategie: $STRAT | trasformazioni: ${TRANSFORMS:-tutte}"
echo "per-query -> $PERQUERY_DIR | extra: ${EXTRA:-nessuno}"
nvidia-smi

FAILED=0

# run_eval <label> <override...>
run_eval() {
  local label="$1"; shift
  echo ""
  echo "########  PARTIAL VALID: $label  ########"
  python -m src.vision.evaluation.evaluate "$@"
  local rc=$?
  [ $rc -ne 0 ] && { echo "!! FALLITA: $label (rc=$rc)" >&2; FAILED=$((FAILED + 1)); }
}

for MODEL in $(select_models "$MODEL_ARG"); do
  for POOL in ${POOLS:-$(poolings_for "$MODEL")}; do
    for RES in $(select_resolutions "$MODEL" "$RES_ARG"); do
      VAR=$(variant_for "$POOL" "$RES")
      LBL=$(label_for "$MODEL" "$POOL" "$RES")
      # same keys as 02; only partial.enabled (and the strategies) change
      COMMON="model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $(res_flags "$RES")"
      COMMON="$COMMON partial.enabled=true eval.split=valid eval.perquery_dir=$PERQUERY_DIR $STRAT_FLAGS $EXTRA"

      if [ "$GROUP" = "all" ] || [ "$GROUP" = "frozen" ]; then
        want raw    && run_eval "$LBL/raw"    $COMMON head.enabled=false whitening.enabled=false
        want whiten && run_eval "$LBL/whiten" $COMMON head.enabled=false whitening.enabled=true
      fi

      if [ "$GROUP" = "all" ] || [ "$GROUP" = "head" ]; then
        want head        && run_eval "$LBL/head"        $COMMON head.enabled=true whitening.enabled=false
        want head+whiten && run_eval "$LBL/head+whiten" $COMMON head.enabled=true whitening.enabled=true
      fi
    done
  done
done

echo ""
echo "=== completato: $(date) | valutazioni fallite: $FAILED ==="
ls -la "$PERQUERY_DIR" | tail -20
echo ""
echo "PROSSIMO PASSO — confronto appaiato allo STESSO livello di masking:"
echo "  python -m src.evaluation.significance --metric self_rr \\"
echo "      --a $PERQUERY_DIR/<A>_partial-random-f0.75_valid.npz \\"
echo "      --b $PERQUERY_DIR/<B>_partial-random-f0.75_valid.npz"
echo "⚠️ f=0.0 NON entra nell'AUC di § 23: li' tutti gli encoder fanno MRR 0.970,"
echo "   che e' un tetto dei DATI (duplicati esatti in RPLAN), non dei modelli."
case "$STRAT" in
  crop|patch|damage)
    echo "Curve dei danni crop/patch: python -m src.evaluation.robustness_auc rank --dir $PERQUERY_DIR --strategy crop  (o patch)" ;;
esac

[ $FAILED -eq 0 ] || exit 1
