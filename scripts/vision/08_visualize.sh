#!/bin/bash
# scripts/vision/08_visualize.sh
# VISUALIZZAZIONI QUALITATIVE del retrieval (pannelli PNG con query + top-k).
# Non e' un eval: gira src.vision.utils.retrieval_visualization su poche query e salva
# le immagini. Richiede solo lo STAGE A (embeddings.npy) del (modello x pooling);
# se un contributo include la head serve anche head.pt (STAGE D). Job leggero.
#
# Tutto e' parametrizzabile via VARIABILI D'AMBIENTE (con default sensati), cosi'
# si cambiano parametri e contributi senza editare lo script:
#   MODEL    encoder                         (default: dinov2)         # o via $1
#   POOL     pooling/variante, lista          (default: natural)        # es. "natural gem"
#   MODE     full | partial | both            (default: both)
#   CONTRIB  contributi, lista                (default: whiten)         # raw|whiten|head|head+whiten
#   NQ       numero di query                  (default: 5)
#   TOPK     risultati per query              (default: 5)
#   THUMB    lato thumbnail in px             (default: 512)
#   EXTRA    override dotlist extra passati a python (es. "eval.split=test")
#
# L'output e' namespaced dallo script Python stesso, quindi resta coerente:
#   results/visualizations/<MODEL>_<POOL>_<contributo>_<full|partial>/...
# (partial: una sottocartella per run, es. .../random_f0.5/).
#
# Uso:
#   sbatch scripts/vision/08_visualize.sh                       # dinov2/natural, both, whiten
#   sbatch scripts/vision/08_visualize.sh dinov3               # cambia solo il modello
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
# Esclude i nodi Blackwell (sm_120), su cui il PyTorch di floorplan-env crasha
# a ogni forward. Stessa whitelist di 07_eval_head_partial.sh.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR"
mkdir -p logs

# --- parametri (env var con default; $1 = modello, come negli altri script) ---
MODEL="${1:-${MODEL:-dinov2}}"
POOL="${POOL:-natural}"
# Risoluzione: "native" = quella del preset YAML (tutti tranne tipsv2).
# Es. RES=448 POOL=gem sbatch scripts/vision/08_visualize.sh tipsv2
RES="${RES:-native}"
MODE="${MODE:-both}"
CONTRIB="${CONTRIB:-whiten}"
NQ="${NQ:-5}"
TOPK="${TOPK:-5}"
THUMB="${THUMB:-512}"
EXTRA="${EXTRA:-}"

# contributo -> flag head/whitening (il tag del nome cartella lo deriva transform_tag)
contrib_flags() {
  case "$1" in
    raw)         echo "head.enabled=false whitening.enabled=false" ;;
    whiten)      echo "head.enabled=false whitening.enabled=true"  ;;
    head)        echo "head.enabled=true  whitening.enabled=false" ;;
    head+whiten) echo "head.enabled=true  whitening.enabled=true"  ;;
    *) echo "CONTRIB '$1' non valido (raw|whiten|head|head+whiten)" >&2; return 1 ;;
  esac
}

# modalita' -> lista di flag partial.enabled da eseguire
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
