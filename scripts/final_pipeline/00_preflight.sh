#!/bin/bash
# Pre-launch checks, on the login node (bash, not sbatch):
#   1. python/torch/PyG/faiss/numpy versions of this user's ~/floorplan-env: the first user writes
#      results/final_pipeline/env_reference.txt, the others must match it;
#   2. group write permission on the folders the jobs write into;
#   3. shared gallery sha1 = 0c24cfc05e18 and the graph cache present;
#   4. none of the destinations of the given configurations exists (nothing is overwritten).
# Exit 1 = do not launch.
#
# Usage:
#   bash scripts/final_pipeline/00_preflight.sh gcn                 # all the stage-1 configurations of gcn
#   bash scripts/final_pipeline/00_preflight.sh gat hd1 hd8         # only these
#   bash scripts/final_pipeline/00_preflight.sh graph_sage comb     # stage 2

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
source ~/floorplan-env/bin/activate
cd "$PROJECT_DIR" || exit 1

ENC="${1:-}"
[ -n "$ENC" ] || { echo "!! uso: 00_preflight.sh <gcn|graph_sage|gat> [cfg ...]" >&2; exit 1; }
shift
SMOKE_FLAG=""
[ "${RG_SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"
RC_PY=src.graph.final_graph_configs
CFGS="$*"
[ -n "$CFGS" ] || CFGS=$(python -m $RC_PY configs --encoder "$ENC") || exit 1
SEEDS="42 100042 200042 300042"
[ -n "$SMOKE_FLAG" ] && SEEDS="42"
FAIL=0
ok()  { echo "  ok   $*"; }
bad() { echo "  FAIL $*"; FAIL=1; }

eval "$(python -m $RC_PY paths --encoder "$ENC" --cfg "$(echo $CFGS | cut -d' ' -f1)" --seed 42 --split valid $SMOKE_FLAG)" || exit 1
echo "=== preflight RESET GRAPHS | utente $USER | $ENC | config: $CFGS | root $ROOT ==="

# 1. versions
mkdir -p "$ROOT/jobs" || { bad "non posso creare $ROOT/jobs"; exit 1; }
VERS=$(python -c "import sys,torch,torch_geometric,faiss,numpy;print(f'python {sys.version.split()[0]} torch {torch.__version__} pyg {torch_geometric.__version__} faiss {faiss.__version__} numpy {numpy.__version__}')") \
  || { bad "import di torch/PyG/faiss/numpy fallito nel tuo ~/floorplan-env"; exit 1; }
REF="$ROOT/env_reference.txt"
if [ ! -f "$REF" ]; then
  echo "$VERS" > "$REF" && ok "versioni registrate come riferimento ($USER): $VERS"
elif [ "$(cat "$REF")" = "$VERS" ]; then
  ok "versioni identiche al riferimento: $VERS"
else
  bad "versioni diverse dal riferimento — tue: '$VERS' · riferimento: '$(cat "$REF")'"
fi

# 2. permissions
for D in "embeddings/graph/$KEY" results logs "$ROOT" "$ROOT/jobs"; do
  [ -d "$D" ] && [ -w "$D" ] && ok "scrivibile: $D" || bad "non scrivibile (o assente): $D"
done
[ "$(umask)" = "0002" ] && ok "umask 002" || bad "umask $(umask) (atteso 0002)"

# 3. gallery and cache
python - <<'PY' && ok "gallery condivisa sha1 0c24cfc05e18" || bad "gallery condivisa diversa da 0c24cfc05e18"
from src.evaluation.gallery_join import load_shared_names
from src.evaluation.perquery import gallery_sha1
raise SystemExit(0 if gallery_sha1(load_shared_names("results/shared_gallery.json")).startswith("0c24cfc05e18") else 1)
PY
[ -d embeddings/graph/rplan/processed ] && ok "cache dei grafi presente" || bad "manca embeddings/graph/rplan/processed"

# 4. destinations
N=0
for CFG in $CFGS; do
  for S in $SEEDS; do
    eval "$(python -m $RC_PY paths --encoder "$ENC" --cfg "$CFG" --seed "$S" --split valid $SMOKE_FLAG)" \
      || { bad "configurazione non valida: $ENC $CFG"; continue; }
    for D in "$DEST" "${DEST}_selfull"; do
      [ -e "$D" ] && bad "esiste gia': $D"
    done
    ls "$PQ_DIR/${TAG}"_* "$QV_DIR/${TAG}"_* >/dev/null 2>&1 && bad "per-query/qvec gia' presenti per $TAG"
    N=$((N + 1))
  done
done
ok "$N destinazioni controllate"

echo ""
[ $FAIL -eq 0 ] && echo "PREFLIGHT OK — si può lanciare." || echo "PREFLIGHT FALLITO — NON lanciare."
exit $FAIL
