#!/bin/bash
# Checks and alpha* of the fusion of ONE replica (CPU, seconds: bash on the login node, not sbatch).
# Tools: `fusion_select check|select`; rule on the checks: C1-C3 PASS and MRR at f=0.0 >= 0.90 for every
# alpha (lower bound only). A C3 FAIL is overcome only by
# results/final_pipeline/fusion/s<S>/c3_waiver_<split>.json with "accepted": true, written by hand after
# diagnosing that every differing query is a float tie.
#
# Every json is written once, under results/final_pipeline/fusion/s<S>/ (never results/fusion/).
#
# Usage:
#   bash scripts/final_pipeline/04_fusion_frozen_vision_select.sh check  valid 42
#   bash scripts/final_pipeline/04_fusion_frozen_vision_select.sh select valid 42
#   bash scripts/final_pipeline/04_fusion_frozen_vision_select.sh check  test 42      # alphas {0, alpha*(S), 1}
#   bash scripts/final_pipeline/04_fusion_frozen_vision_select.sh select test 42      # alpha* fixed from the valid

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
source ~/floorplan-env/bin/activate
cd "$PROJECT_DIR" || exit 1

CMD="${1:-}"; SPLIT="${2:-}"; S="${3:-}"
case "$CMD" in check|select) ;; *) echo "!! uso: 04_fusion_frozen_vision_select.sh <check|select> <valid|test> <seed>" >&2; exit 1 ;; esac
case "$SPLIT" in valid|test) ;; *) echo "!! split '$SPLIT' (valid | test)" >&2; exit 1 ;; esac
[[ "$S" =~ ^[0-9]+$ ]] || { echo "!! seed non valido: '$S'" >&2; exit 1; }
SMOKE_FLAG=""
[ "${RG_SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"

eval "$(python -m src.evaluation.graph_config_select fusion-paths --seed "$S" --split "$SPLIT" $SMOKE_FLAG)" || exit 1
[ -d "$FUSION_DIR" ] || { echo "!! manca $FUSION_DIR (03_fusion_frozen_vision.sh $SPLIT $S)" >&2; exit 1; }

ALPHA_STAR=""
if [ "$SPLIT" = "test" ]; then
  [ -f "$SELECT_VALID_JSON" ] || { echo "!! sul test serve $SELECT_VALID_JSON" >&2; exit 1; }
  ALPHA_STAR=$(python -c 'import json,sys; print(format(float(json.load(open(sys.argv[1]))["alpha_star"]), "g"))' "$SELECT_VALID_JSON") \
    || { echo "!! alpha_star illeggibile in $SELECT_VALID_JSON" >&2; exit 1; }
fi
refuse() { [ ! -e "$1" ] || { echo "!! $1 esiste gia': niente sovrascritture" >&2; exit 1; }; }

echo "=== $(date) | RESET GRAPHS fusion_select $CMD | W = $W_ENC $W_CFG | s$S | $SPLIT ==="
if [ "$CMD" = "check" ]; then
  refuse "$CHECK_RESET_JSON"
  if [ -f "$CHECK_JSON" ]; then
    # check already run and failed: re-read it (e.g. after a C3 waiver)
    echo "(rilettura di $CHECK_JSON, fusion_select check non rilanciato)"
    python -m src.evaluation.graph_config_select fusion-check --check-json "$CHECK_JSON" \
      --waiver "$WAIVER_JSON" --out-json "$CHECK_RESET_JSON" || exit 1
    echo "PROSSIMO PASSO: bash scripts/final_pipeline/04_fusion_frozen_vision_select.sh select $SPLIT $S"
    exit 0
  fi
  ALPHA_ARGS=()
  [ -n "$ALPHA_STAR" ] && ALPHA_ARGS=(--alphas 0 "$ALPHA_STAR" 1)
  START_MARK=$(mktemp)
  python -m src.evaluation.fusion_select check --split "$SPLIT" --fusion-dir "$FUSION_DIR" \
    --vision-qvec "$VISION_QVEC" --graph-qvec "$GRAPH_QVEC" \
    --vision-perquery "$VISION_PQ" --graph-perquery "$GRAPH_PQ" \
    "${ALPHA_ARGS[@]}" \
    --out-json "$CHECK_JSON"
  echo "(rc di fusion_select check: $? — il suo C5 [0.965, 0.980] NON e' la regola di RESET: decide la riga sotto)"
  if [ ! -f "$CHECK_JSON" ] || [ ! "$CHECK_JSON" -nt "$START_MARK" ]; then
    rm -f "$START_MARK"; echo "!! $CHECK_JSON non scritto (fusion_select fallito prima della fine)" >&2; exit 1
  fi
  rm -f "$START_MARK"
  python -m src.evaluation.graph_config_select fusion-check --check-json "$CHECK_JSON" \
    --waiver "$WAIVER_JSON" --out-json "$CHECK_RESET_JSON" || exit 1
  echo "PROSSIMO PASSO: bash scripts/final_pipeline/04_fusion_frozen_vision_select.sh select $SPLIT $S"
  exit 0
fi

# select: only after a passed RESET check
python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' "$CHECK_RESET_JSON" 2>/dev/null \
  || { echo "!! $CHECK_RESET_JSON mancante o FAIL: niente scelta di alpha" >&2; exit 1; }
refuse "$SELECT_JSON"
FIXED_ARGS=()
[ "$SPLIT" = "test" ] && FIXED_ARGS=(--fixed-alpha-from "$SELECT_VALID_JSON")
python -m src.evaluation.fusion_select select --split "$SPLIT" --fusion-dir "$FUSION_DIR" \
  "${FIXED_ARGS[@]}" \
  --out-json "$SELECT_JSON" || exit 1
