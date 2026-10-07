#!/bin/bash
# Qualitative retrieval visualisations (PNG panels with query + top-k).
# Not an eval: runs src.vision.utils.retrieval_visualization on a few queries. Needs stage A (embeddings.npy);
# contributions with the head also need head.pt (stage D). Light job.
#
# Environment variables (defaults in parentheses):
#   MODEL    encoder (dinov2), or $1
#   POOL     poolings, list (natural), e.g. "natural gem"
#   MODE     full | partial | both (both)
#   CONTRIB  contributions, list (whiten): raw|whiten|head|head+whiten
#   NQ       number of queries (5)
#   TOPK     results per query (5)
#   THUMB    thumbnail side in px (512)
#   EXTRA    extra dotlist overrides passed to python (e.g. "eval.split=test")
#
# Output is namespaced by the python script:
#   results/visualizations/<MODEL>_<POOL>_<contribution>_<full|partial>/... (partial: one subfolder per run, e.g. .../random_f0.5/)
#
# Uso:
#   sbatch scripts/vision/08_visualize.sh                       # dinov2/natural, both, whiten
#   sbatch scripts/vision/08_visualize.sh dinov3               # change only the model
#   MODE=partial CONTRIB="whiten head+whiten" sbatch scripts/vision/08_visualize.sh dinov2
#   POOL="natural gem" NQ=10 sbatch scripts/vision/08_visualize.sh

#SBATCH --job-name="vx_08_visualize"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Exclude Blackwell nodes (sm_120): floorplan-env PyTorch crashes on every forward. Same whitelist as 07_eval_head_partial.sh.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR"
mkdir -p logs

# --- parameters (env vars with defaults; $1 = model) ---
MODEL="${1:-${MODEL:-dinov2}}"
POOL="${POOL:-natural}"
# Resolution: "native" = YAML preset (all but tipsv2), e.g. RES=448 POOL=gem sbatch scripts/vision/08_visualize.sh tipsv2
RES="${RES:-native}"
MODE="${MODE:-both}"
CONTRIB="${CONTRIB:-whiten}"
NQ="${NQ:-5}"
TOPK="${TOPK:-5}"
THUMB="${THUMB:-512}"
EXTRA="${EXTRA:-}"

# contribution -> head/whitening flags
contrib_flags() {
  case "$1" in
    raw)         echo "head.enabled=false whitening.enabled=false" ;;
    whiten)      echo "head.enabled=false whitening.enabled=true"  ;;
    head)        echo "head.enabled=true  whitening.enabled=false" ;;
    head+whiten) echo "head.enabled=true  whitening.enabled=true"  ;;
    *) echo "CONTRIB '$1' non valido (raw|whiten|head|head+whiten)" >&2; return 1 ;;
  esac
}

# mode -> partial.enabled flags to run
modes() {
  case "$1" in
    full)    echo "false" ;;
    partial) echo "true" ;;
    both)    echo "false true" ;;
    *) echo "MODE '$1' non valido (full|partial|both)" >&2; return 1 ;;
  esac
}

echo "=== $(date) | VIZ | model=$MODEL pool='$POOL' res=$RES mode=$MODE contrib='$CONTRIB' nq=$NQ topk=$TOPK ==="
nvidia-smi

for P in $POOL; do
  for C in $CONTRIB; do
    CFLAGS=$(contrib_flags "$C") || exit 1
    BASE="model.name=$MODEL model.variant=$(variant_for "$P" "$RES") model.kwargs.pooling=$P $(res_flags "$RES") $CFLAGS $EXTRA"
    for PART in $(modes "$MODE"); do
      echo ""
      echo "########  VIZ: $MODEL / $P / $C / partial=$PART  ########"
      python -m src.vision.utils.retrieval_visualization \
        --n-queries "$NQ" --top-k "$TOPK" --thumb "$THUMB" \
        $BASE "partial.enabled=$PART"
    done
  done
done

echo ""
echo "=== completato: $(date) ==="
