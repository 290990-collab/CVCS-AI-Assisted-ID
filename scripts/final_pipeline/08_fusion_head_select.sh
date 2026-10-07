#!/bin/bash
# Checks, alpha*_H and the comparison with the main fusion, ONE replica (CPU, seconds/minutes: bash on the
# login node, not sbatch). On the test, check uses alphas {0, alpha*_H, 1} and select keeps alpha*_H fixed from
# the valid; compare then also writes the test verdict (decided by Delta_H).
# Same tools and rule as 04_fusion_frozen_vision_select.sh: `fusion_select check` through `fusion_head check`
# (accepts only the pre-registered head), then `graph_config_select fusion-check` (C1-C3 PASS, MRR f=0.0 >= 0.90
# for every alpha), `select`, and `compare` (measures 2-3 + outcome).
# Every json is written once, under results/final_pipeline/fusion_head/s<S>/.
#
# Usage:
#   bash scripts/final_pipeline/08_fusion_head_select.sh check   valid 42
#   bash scripts/final_pipeline/08_fusion_head_select.sh select  valid 42
#   bash scripts/final_pipeline/08_fusion_head_select.sh compare valid 42   # needs the main fusion's select too
#   bash scripts/final_pipeline/08_fusion_head_select.sh check   test 42    # then select, compare

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
source ~/floorplan-env/bin/activate
cd "$PROJECT_DIR" || exit 1
export CUDA_VISIBLE_DEVICES=""

CMD="${1:-}"; SPLIT="${2:-}"; S="${3:-}"
case "$CMD" in check|select|compare) ;; *) echo "!! uso: 08_fusion_head_select.sh <check|select|compare> <valid|test> <seed>" >&2; exit 1 ;; esac
case "$SPLIT" in valid|test) ;; *) echo "!! split '$SPLIT' (valid | test)" >&2; exit 1 ;; esac
[[ "$S" =~ ^[0-9]+$ ]] || { echo "!! seed non valido: '$S'" >&2; exit 1; }

eval "$(python -m src.evaluation.fusion_head paths --seed "$S" --split "$SPLIT")" || exit 1
[ -d "$FUSION_DIR" ] || { echo "!! manca $FUSION_DIR (07_fusion_head.sh $SPLIT $S)" >&2; exit 1; }
refuse() { [ ! -e "$1" ] || { echo "!! $1 esiste gia': niente sovrascritture" >&2; exit 1; }; }
ALPHA_ARGS=(); FIXED_ARGS=()
if [ "$SPLIT" = "test" ]; then
  [ -f "$SELECT_VALID_JSON" ] || { echo "!! sul test serve $SELECT_VALID_JSON" >&2; exit 1; }
  ALPHA_STAR=$(python -c 'import json,sys; print(format(float(json.load(open(sys.argv[1]))["alpha_star"]), "g"))' "$SELECT_VALID_JSON") \
    || { echo "!! alpha_star illeggibile in $SELECT_VALID_JSON" >&2; exit 1; }
  ALPHA_ARGS=(--alphas 0 "$ALPHA_STAR" 1)
  FIXED_ARGS=(--fixed-alpha-from "$SELECT_VALID_JSON")
fi

echo "=== $(date) | fusione CON HEAD $CMD | W = $W_ENC $W_CFG | s$S | $SPLIT ==="
if [ "$CMD" = "check" ]; then
  refuse "$CHECK_RESET_JSON"
  if [ ! -f "$CHECK_JSON" ]; then
    START_MARK=$(mktemp)
    python -m src.evaluation.fusion_head check --split "$SPLIT" --fusion-dir "$FUSION_DIR" \
      --vision-qvec "$VISION_QVEC" --graph-qvec "$GRAPH_QVEC" \
      --vision-perquery "$VISION_PQ" --graph-perquery "$GRAPH_PQ" \
      "${ALPHA_ARGS[@]}" \
      --out-json "$CHECK_JSON"
    echo "(rc di fusion_select check: $? — il suo C5 [0.965, 0.980] NON e' la regola di RESET: decide la riga sotto)"
    if [ ! -f "$CHECK_JSON" ] || [ ! "$CHECK_JSON" -nt "$START_MARK" ]; then
      rm -f "$START_MARK"; echo "!! $CHECK_JSON non scritto" >&2; exit 1
    fi
    rm -f "$START_MARK"
  else
    echo "(rilettura di $CHECK_JSON, check non rilanciato)"
  fi
  python -m src.evaluation.graph_config_select fusion-check --check-json "$CHECK_JSON" \
    --waiver "$WAIVER_JSON" --out-json "$CHECK_RESET_JSON" || exit 1
  echo "PROSSIMO PASSO: bash scripts/final_pipeline/08_fusion_head_select.sh select $SPLIT $S"
  exit 0
fi

if [ "$CMD" = "select" ]; then
  python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' "$CHECK_RESET_JSON" 2>/dev/null \
    || { echo "!! $CHECK_RESET_JSON mancante o FAIL: niente scelta di alpha" >&2; exit 1; }
  refuse "$SELECT_JSON"
  python -m src.evaluation.fusion_head select --split "$SPLIT" --fusion-dir "$FUSION_DIR" \
    "${FIXED_ARGS[@]}" \
    --out-json "$SELECT_JSON" || exit 1
  echo "PROSSIMO PASSO: bash scripts/final_pipeline/08_fusion_head_select.sh compare $SPLIT $S"
  exit 0
fi

# compare: both selections must exist
[ -f "$SELECT_JSON" ] || { echo "!! manca $SELECT_JSON" >&2; exit 1; }
[ -f "$MAIN_SELECT_JSON" ] || { echo "!! manca $MAIN_SELECT_JSON (fusione principale, 04_fusion_frozen_vision_select.sh select)" >&2; exit 1; }
refuse "$COMPARE_JSON"
python -m src.evaluation.fusion_head compare --seed "$S" --split "$SPLIT" || exit 1
