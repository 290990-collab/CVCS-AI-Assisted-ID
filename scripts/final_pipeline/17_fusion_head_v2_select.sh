#!/bin/bash
# Checks and alpha* of the fusion W + head v2, ONE replica (CPU, bash on the login node).
# Same tools and rule as 08_fusion_head_select.sh: `fusion_select check` through `fusion_head_v2 check`, then
# `graph_config_select fusion-check` (C1-C3 PASS, MRR f=0.0 >= 0.90; a C3 FAIL only with c3_waiver_<split>.json
# from src.evaluation.near_ties), then `select`. Every json once, under results/final_pipeline/fusion_head_v2/s<S>/.
#
# Usage:
#   bash scripts/final_pipeline/17_fusion_head_v2_select.sh check  valid 42
#   bash scripts/final_pipeline/17_fusion_head_v2_select.sh select valid 42

set -uo pipefail
umask 002
PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
source ~/floorplan-env/bin/activate
cd "$PROJECT_DIR" || exit 1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4

CMD="${1:-}"; SPLIT="${2:-}"; S="${3:-}"
case "$CMD" in check|select) ;; *) echo "!! uso: 17_fusion_head_v2_select.sh <check|select> <valid|test> <seed>" >&2; exit 1 ;; esac
case "$SPLIT" in valid|test) ;; *) echo "!! split '$SPLIT' (valid | test)" >&2; exit 1 ;; esac
[[ "$S" =~ ^[0-9]+$ ]] || { echo "!! seed non valido: '$S'" >&2; exit 1; }

eval "$(python -m src.evaluation.fusion_head_v2 paths --seed "$S" --split "$SPLIT")" || exit 1
[ -d "$FUSION_DIR" ] || { echo "!! manca $FUSION_DIR (16_fusion_head_v2.sh $SPLIT $S)" >&2; exit 1; }
refuse() { [ ! -e "$1" ] || { echo "!! $1 esiste gia': niente sovrascritture" >&2; exit 1; }; }
ALPHA_ARGS=(); FIXED_ARGS=()
if [ "$SPLIT" = "test" ]; then
  [ -f "$SELECT_VALID_JSON" ] || { echo "!! sul test serve $SELECT_VALID_JSON" >&2; exit 1; }
  ALPHA_STAR=$(python -c 'import json,sys; print(format(float(json.load(open(sys.argv[1]))["alpha_star"]), "g"))' "$SELECT_VALID_JSON") || exit 1
  ALPHA_ARGS=(--alphas 0 "$ALPHA_STAR" 1)
  FIXED_ARGS=(--fixed-alpha-from "$SELECT_VALID_JSON")
fi

echo "=== $(date) | fusione CON HEAD v2 $CMD | s$S | $SPLIT ==="
if [ "$CMD" = "check" ]; then
  refuse "$CHECK_RESET_JSON"
  if [ ! -f "$CHECK_JSON" ]; then
    python -m src.evaluation.fusion_head_v2 check --split "$SPLIT" --fusion-dir "$FUSION_DIR" \
      --vision-qvec "$VISION_QVEC" --graph-qvec "$GRAPH_QVEC" \
      --vision-perquery "$VISION_PQ" --graph-perquery "$GRAPH_PQ" \
      "${ALPHA_ARGS[@]}" --out-json "$CHECK_JSON"
    echo "(rc di fusion_select check: $? — il suo C5 [0.965, 0.980] NON e' la regola di RESET: decide la riga sotto)"
    [ -f "$CHECK_JSON" ] || { echo "!! $CHECK_JSON non scritto" >&2; exit 1; }
  fi
  python -m src.evaluation.graph_config_select fusion-check --check-json "$CHECK_JSON" \
    --waiver "$WAIVER_JSON" --out-json "$CHECK_RESET_JSON" || exit 1
  exit 0
fi
python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' "$CHECK_RESET_JSON" 2>/dev/null \
  || { echo "!! $CHECK_RESET_JSON mancante o FAIL: niente scelta di alpha" >&2; exit 1; }
refuse "$SELECT_JSON"
python -m src.evaluation.fusion_head_v2 select --split "$SPLIT" --fusion-dir "$FUSION_DIR" \
  "${FIXED_ARGS[@]}" --out-json "$SELECT_JSON"
