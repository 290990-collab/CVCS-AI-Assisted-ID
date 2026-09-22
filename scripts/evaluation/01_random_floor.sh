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
# Uso:  sbatch 01_random_floor.sh [all|vision|graph] [test|valid]
#   sbatch scripts/evaluation/01_random_floor.sh              # 4 run sul TEST
#   sbatch scripts/evaluation/01_random_floor.sh all valid    # 4 run sul VALID
#   sbatch scripts/evaluation/01_random_floor.sh vision       # solo gallery vision, test
# Lo split va scelto uguale a quello delle query che il floor deve normalizzare:
# le selezioni del progetto si fanno sul valid, quindi serve il floor del valid
# (fino al 13 ago si usava quello del test come proxy — status.md § 11, § 18.4).
#
# Output (oltre al log del job): due file che NON contengono la stessa cosa, e il
# nome lo dice — confonderli e' costato la rettifica di status.md § 18.4.
#   results/random_floor/<null>_<ramo>_<split>_seed0.npz   per-query, UN SOLO seed
#   results/random_floor/<null>_<ramo>_<split>_mean5.txt   tabella, MEDIA sui 5 seed

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

# Le query e i seed sono gli stessi del protocollo di valutazione: 2000 query
# dello split scelto, campionate con seed 42, cinque seed di ranking. Cambiare
# questi valori scollega il floor dalle run che deve normalizzare.
BASE=(--split "$SPLIT" --num-queries 2000 --query-seed 42 --ranking-seeds 0 1 2 3 4)

VISION_GALLERY="embeddings/vision/dinov3/gem/image_paths.json"   # 67.453 righe
GRAPH_GALLERY="embeddings/graph/gcn/tau02/names.json"            # 67.405 righe

FAILED=0

# run_floor <tag> <gallery> [flag extra...]
run_floor() {
  local tag="$1" gallery="$2"; shift 2
  # Due suffissi diversi perche' il contenuto e' diverso: il .npz e' il per-query
  # del PRIMO ranking-seed (random_floor.py:432), il .txt e' la tabella mediata
  # sui 5. Sul null costante differiscono di ~1 mc_std (status.md § 18.4).
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
