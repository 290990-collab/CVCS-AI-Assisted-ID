#!/bin/bash
# scripts/evaluation/01_random_floor.sh
# FASE A.2 — il "floor" del retrieval: quanto prende un sistema che non guarda
# la query. Due null, per rispondere a due domande diverse:
#   random   — ranking casuale: il pavimento assoluto della metrica.
#   constant — le stesse piante restituite a TUTTE le query: dice se la metrica
#              DISCRIMINA. Se anche questo prende un punteggio alto, il merito
#              non e' del modello ma della scala compressa dell'asse.
# Va lanciato su ENTRAMBE le gallery (vision 67.453, graph 67.405) perche'
# l'IDCG dipende dalla gallery: il floor non e' un numero universale, e' il
# denominatore di quel ramo.
#
# Nessuna GPU: e' un job puramente CPU (lettura dei .mat + similarita' per-asse).
# Esiste perche' sul login node `ulimit -t` = 600 s di CPU e queste run ne
# consumano ~540: un `Killed` muto, non un bug. Vedi status.md § 6.
#
# Uso:
#   sbatch scripts/evaluation/01_random_floor.sh          # tutte e quattro le run
#   sbatch scripts/evaluation/01_random_floor.sh vision   # solo le due della gallery vision
#   sbatch scripts/evaluation/01_random_floor.sh graph    # solo le due della gallery graph
#
# Output (oltre al log del job):
#   results/random_floor/<null>_<ramo>_test.npz   per-query del primo seed
#   results/random_floor/<null>_<ramo>_test.txt   la tabella asse x k, leggibile

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

# Le query e i seed sono gli stessi del protocollo di valutazione: 2000 query
# del test split campionate con seed 42, cinque seed di ranking.
BASE=(--split test --num-queries 2000 --query-seed 42 --ranking-seeds 0 1 2 3 4)

VISION_GALLERY="embeddings/vision/dinov3/gem/image_paths.json"   # 67.453 righe
GRAPH_GALLERY="embeddings/graph/gcn/tau02/names.json"            # 67.405 righe

FAILED=0

# run_floor <tag> <gallery> [flag extra...]
run_floor() {
  local tag="$1" gallery="$2"; shift 2
  local npz="$OUT_DIR/${tag}.npz"
  local txt="$OUT_DIR/${tag}.txt"

  if [[ ! -f "$gallery" ]]; then
    echo "[01_random_floor] ERRORE: gallery mancante: $gallery" >&2
    FAILED=$((FAILED + 1))
    return
  fi

  echo "=============================================================="
  echo "[01_random_floor] $tag — gallery $gallery"
  echo "=============================================================="
  # tee: la tabella resta anche fuori dal log del job, accanto al .npz.
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
  run_floor "random_vision_test"   "$VISION_GALLERY"
  run_floor "constant_vision_test" "$VISION_GALLERY" --constant-ranking
fi

if [[ "$WHICH" == "all" || "$WHICH" == "graph" ]]; then
  run_floor "random_graph_test"   "$GRAPH_GALLERY"
  run_floor "constant_graph_test" "$GRAPH_GALLERY" --constant-ranking
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
