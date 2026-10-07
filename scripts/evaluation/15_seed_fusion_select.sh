#!/bin/bash
# Multi-seed replica, CPU, seconds: validity checks and choice of alpha for the late fusion of ONE seed
# replica S, i.e. the outputs of
#   08_queryvec_vision.sh <split> S · 09_queryvec_graph.sh <split> S · 10_late_fusion.sh <split> S
# Thin wrapper around `python -m src.evaluation.fusion_select` with the seed paths fixed here and always an
# explicit --out-json under results/fusion/seeds/s<S>/ (without it `select` would overwrite the historical
# results/fusion/select_<split>.json).
#
# Run with bash on the login node (no #SBATCH header: seconds of CPU), not sbatch.
#
# Usage:
#   bash scripts/evaluation/15_seed_fusion_select.sh check  valid 100042
#   bash scripts/evaluation/15_seed_fusion_select.sh select valid 100042
#   bash scripts/evaluation/15_seed_fusion_select.sh check  test  100042   # alphas {0, alpha*_valid(S), 1}
#   bash scripts/evaluation/15_seed_fusion_select.sh select test  100042   # alpha* fixed from select_valid.json of S
#   FORCE=1 bash ...                                                       # overwrite an existing out json
#
# check: C1/C2/C3 hard; C4 not applicable (no historical files to compare a new seed with); C5 lower bound
#   only (MRR f=0.0 >= fusion_select.C5_LOW). On valid fusion_select also applies the upper bound
#   (vision-graph pair) and has no flag to drop it, so the C5 verdict here is recomputed from the check json;
#   fusion_select's own exit code is printed but not used.
#   -> results/fusion/seeds/s<S>/check_<split>.json
# select: requires a passing check json of the same split; then
#   -> results/fusion/seeds/s<S>/select_<split>.json
#   and prints the vision robustness R (`robustness_auc rank --robust`, nowalls-random
#   + crop + patch of 08 in seed mode).

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
source ~/floorplan-env/bin/activate
cd "$PROJECT_DIR" || exit 1

CMD="${1:-}"
SPLIT="${2:-}"
SEED="${3:-}"
case "$CMD" in
  check|select) ;;
  *) echo "!! ERRORE: primo argomento check | select, ricevuto '$CMD'" >&2; exit 1 ;;
esac
case "$SPLIT" in
  valid|test) ;;
  *) echo "!! ERRORE: secondo argomento valid | test, ricevuto '$SPLIT'" >&2; exit 1 ;;
esac
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "!! ERRORE: terzo argomento = seed intero, ricevuto '$SEED'" >&2
  exit 1
fi

# Seed paths (as 08/09/10 in seed mode).
BRANCHES="results/perquery/seeds/s${SEED}/fusion_branches_${SPLIT}"
FUSION_DIR="results/perquery/seeds/s${SEED}/fusion_${SPLIT}"
QVEC="results/queryvec/seeds/s${SEED}/${SPLIT}"
VISION_QVEC="$QVEC/vision_pespatial_gem_whiten-train"
GRAPH_QVEC="$QVEC/graph_gat_asymrob_s${SEED}"
VISION_PQ="$BRANCHES/vision_pespatial_gem_whiten-train"
GRAPH_PQ="$BRANCHES/graph_gat_asymrob_s${SEED}"
OUTDIR="results/fusion/seeds/s${SEED}"
SEL_VALID="$OUTDIR/select_valid.json"
CHECK_JSON="$OUTDIR/check_${SPLIT}.json"
SELECT_JSON="$OUTDIR/select_${SPLIT}.json"
case "$OUTDIR" in
  results/fusion/seeds/s[0-9]*) ;;
  *) echo "!! ERRORE: cartella di output non ammessa: '$OUTDIR'" >&2; exit 1 ;;
esac
[ -d "$FUSION_DIR" ] || { echo "!! ERRORE: manca $FUSION_DIR (10_late_fusion.sh $SPLIT $SEED)" >&2; exit 1; }

# Test: alpha* comes from the valid of the same replica.
ALPHA_STAR=""
if [ "$SPLIT" = "test" ]; then
  [ -f "$SEL_VALID" ] || { echo "!! ERRORE: sul test serve $SEL_VALID (select valid della replica)" >&2; exit 1; }
  ALPHA_STAR=$(python -c 'import json,sys; print(format(float(json.load(open(sys.argv[1]))["alpha_star"]), "g"))' "$SEL_VALID") \
    || { echo "!! ERRORE: alpha_star illeggibile in $SEL_VALID" >&2; exit 1; }
fi

refuse_overwrite() {
  if [ -f "$1" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "!! ERRORE: $1 esiste gia': rilancia con FORCE=1 se voluto" >&2
    exit 1
  fi
}

# check_verdict <check json>: C1-C3 PASS and C5 lower bound on every alpha.
# A C3 FAIL counts only if $OUTDIR/c3_waiver_<split>.json exists with "accepted": true, written by hand after
# diagnosing that every differing query is a float32 tie; it is printed, never silent.
WAIVER_JSON="$OUTDIR/c3_waiver_${SPLIT}.json"
check_verdict() {
  python - "$1" "$WAIVER_JSON" <<'PY'
import json, os, sys
from src.evaluation.fusion_select import C5_LOW
r = json.load(open(sys.argv[1]))
hard = {c: (r.get(c) or {}).get("status") for c in ("C1", "C2", "C3")}
if hard["C3"] != "PASS" and os.path.exists(sys.argv[2]):
    w = json.load(open(sys.argv[2]))
    if w.get("accepted") is True:
        hard["C3"] = "PASS"
        print(f"  C3: FAIL superato da waiver {sys.argv[2]}: {w.get('reason', '')}")
c5 = (r.get("C5") or {}).get("per_alpha") or {}
c5_ok = bool(c5) and all(v["mrr"] >= C5_LOW for v in c5.values())
for c, s in hard.items():
    print(f"  {c}: {s}")
print(f"  C5 (solo limite basso, MRR f=0.0 >= {C5_LOW}): {'PASS' if c5_ok else 'FAIL'} "
      + " ".join(f"a={a}:{v['mrr']:.4f}" for a, v in c5.items()))
ok = all(s == "PASS" for s in hard.values()) and c5_ok
print(f"ESITO REPLICA: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
}

echo "=== $(date) | SEED REPLICA s$SEED | fusion_select $CMD | split $SPLIT ==="
if [ "$CMD" = "check" ]; then
  refuse_overwrite "$CHECK_JSON"
  ALPHA_ARGS=()
  [ -n "$ALPHA_STAR" ] && ALPHA_ARGS=(--alphas 0 "$ALPHA_STAR" 1)
  mkdir -p "$OUTDIR"
  START_MARK=$(mktemp)
  python -m src.evaluation.fusion_select check --split "$SPLIT" --fusion-dir "$FUSION_DIR" \
    --vision-qvec "$VISION_QVEC" --graph-qvec "$GRAPH_QVEC" \
    --vision-perquery "$VISION_PQ" --graph-perquery "$GRAPH_PQ" \
    "${ALPHA_ARGS[@]}" \
    --out-json "$CHECK_JSON"
  echo "(rc di fusion_select check: $? — sul valid include il limite alto di C5, qui non vincolante)"
  if [ ! -f "$CHECK_JSON" ] || [ ! "$CHECK_JSON" -nt "$START_MARK" ]; then
    rm -f "$START_MARK"
    echo "!! ERRORE: $CHECK_JSON non scritto (fusion_select e' fallito prima della fine)" >&2
    exit 1
  fi
  rm -f "$START_MARK"
  echo ""
  check_verdict "$CHECK_JSON" || exit 1
  echo "PROSSIMO PASSO: bash scripts/evaluation/15_seed_fusion_select.sh select $SPLIT $SEED"
  exit 0
fi

# select
[ -f "$CHECK_JSON" ] || { echo "!! ERRORE: manca $CHECK_JSON: prima il check" >&2; exit 1; }
echo "verdetto del check ($CHECK_JSON):"
check_verdict "$CHECK_JSON" || { echo "!! ERRORE: check non superato, niente scelta di alpha" >&2; exit 1; }
refuse_overwrite "$SELECT_JSON"
FIXED_ARGS=()
[ "$SPLIT" = "test" ] && FIXED_ARGS=(--fixed-alpha-from "$SEL_VALID")
python -m src.evaluation.fusion_select select --split "$SPLIT" --fusion-dir "$FUSION_DIR" \
  "${FIXED_ARGS[@]}" \
  --out-json "$SELECT_JSON" || exit 1

echo ""
echo "=== robustezza vision R della replica (nowalls-random + crop + patch) ==="
python -m src.evaluation.robustness_auc rank --split "$SPLIT" --dir "$BRANCHES" --robust || exit 1
echo ""
echo "Riepilogo di tutte le repliche: python -m src.evaluation.seed_summary"
