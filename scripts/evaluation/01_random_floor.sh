#!/bin/bash
# Retrieval floor: what a system that ignores the query scores. Two nulls:
#   random   - random ranking, the absolute floor of the metric.
#   constant - the same plans returned for ALL queries: shows whether the metric discriminates.
# Run on both galleries (vision 67,453, graph 67,405): IDCG depends on the gallery.
#
# CPU only (reads .mat + per-axis similarities); a job because the login node's `ulimit -t` = 600 s
# kills these runs (~540 s CPU) silently.
#
# Usage: sbatch 01_random_floor.sh [all|vision|graph] [test|valid]
#   sbatch scripts/evaluation/01_random_floor.sh              # 4 runs on TEST
#   sbatch scripts/evaluation/01_random_floor.sh all valid    # 4 runs on VALID
#   sbatch scripts/evaluation/01_random_floor.sh vision       # vision gallery only, test
# The split must match the queries the floor normalises.
#
# Output (the two files hold different things):
#   results/random_floor/<null>_<branch>_<split>_seed0.npz   per-query, ONE seed
#   results/random_floor/<null>_<branch>_<split>_mean5.txt   table, MEAN over 5 seeds

#SBATCH --job-name="gp_01_random_floor"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
source ~/floorplan-env/bin/activate
cd "$PROJECT_DIR" || exit 1
mkdir -p logs results/random_floor

OUT_DIR="results/random_floor"
WHICH="${1:-all}"
SPLIT="${2:-test}"

if [[ "$SPLIT" != "test" && "$SPLIT" != "valid" ]]; then
  echo "[01_random_floor] ERRORE: split '$SPLIT' non riconosciuto (attesi: test | valid)" >&2
  exit 1
fi

# same queries and seeds as the evaluation protocol: 2000 queries, seed 42, five ranking seeds
BASE=(--split "$SPLIT" --num-queries 2000 --query-seed 42 --ranking-seeds 0 1 2 3 4)

VISION_GALLERY="embeddings/vision/dinov3/gem/image_paths.json"   # 67,453 rows
GRAPH_GALLERY="embeddings/graph/gcn/tau02/names.json"            # 67,405 rows

FAILED=0

# run_floor <tag> <gallery> [flag extra...]
run_floor() {
  local tag="$1" gallery="$2"; shift 2
  # .npz = per-query of the first ranking seed, .txt = table averaged over 5 (differ ~1 mc_std on the constant null)
  local npz="$OUT_DIR/${tag}_seed0.npz"
  local txt="$OUT_DIR/${tag}_mean5.txt"

  if [[ ! -f "$gallery" ]]; then
    echo "[01_random_floor] ERRORE: gallery mancante: $gallery" >&2
    FAILED=$((FAILED + 1))
    return
  fi

  echo "=============================================================="
  echo "[01_random_floor] $tag — split $SPLIT — gallery $gallery"
  echo "=============================================================="
  # tee: keep the table next to the .npz
  python -m src.evaluation.random_floor \
      --gallery "$gallery" "${BASE[@]}" "$@" \
      --out "$npz" 2>&1 | tee "$txt"

  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "[01_random_floor] $tag FALLITA (rc=$rc)" >&2
    FAILED=$((FAILED + 1))
  fi
}

if [[ "$WHICH" == "all" || "$WHICH" == "vision" ]]; then
  run_floor "random_vision_${SPLIT}"   "$VISION_GALLERY"
  run_floor "constant_vision_${SPLIT}" "$VISION_GALLERY" --constant-ranking
fi

if [[ "$WHICH" == "all" || "$WHICH" == "graph" ]]; then
  run_floor "random_graph_${SPLIT}"   "$GRAPH_GALLERY"
  run_floor "constant_graph_${SPLIT}" "$GRAPH_GALLERY" --constant-ranking
fi

if [[ "$WHICH" != "all" && "$WHICH" != "vision" && "$WHICH" != "graph" ]]; then
  echo "[01_random_floor] ERRORE: argomento '$WHICH' non riconosciuto " \
       "(attesi: all | vision | graph)" >&2
  exit 1
fi

echo "=============================================================="
if [[ $FAILED -eq 0 ]]; then
  echo "[01_random_floor] tutte le run completate. Risultati in $OUT_DIR/"
  ls -la "$OUT_DIR"
else
  echo "[01_random_floor] $FAILED run FALLITE — vedi sopra" >&2
  exit 1
fi
