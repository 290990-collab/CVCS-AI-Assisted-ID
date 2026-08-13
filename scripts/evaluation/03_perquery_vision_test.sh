#!/bin/bash
# scripts/evaluation/03_perquery_vision_test.sh
# FASE B.1, secondo tempo — il TEST, per UNA sola configurazione.
#
# Questo script esiste per rendere difficile la cosa sbagliata. Il test non
# sceglie nulla (vincolo DURO 1): si legge una volta, per la configurazione gia'
# scelta sul valid con 02_perquery_vision_valid.sh. Per questo i tre argomenti
# sono OBBLIGATORI e non c'e' nessuna modalita' "tutti": una griglia sul test e'
# esattamente il rilievo A1 che stiamo chiudendo.
#
# Uso:
#   sbatch scripts/evaluation/03_perquery_vision_test.sh dinov3 gem whiten
#   sbatch scripts/evaluation/03_perquery_vision_test.sh <encoder> <pooling> <raw|whiten|head|head+whiten>
#
# Output: results/perquery/vision_test/vision_<enc>_<pool>_<trasf>_full_test.npz

#SBATCH --job-name="ev_03_perquery_vision_test"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026

set -uo pipefail
umask 002

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR" || exit 1
mkdir -p logs results/perquery/vision_test

MODEL="${1:-}"
POOL="${2:-}"
TRANSFORM="${3:-}"
# Risoluzione: "native" = quella del preset YAML (tutti tranne tipsv2, che ha
# due varianti per pooling). Deve essere la STESSA scelta sul valid.
RES="${RES:-native}"
PERQUERY_DIR="results/perquery/vision_test"

if [ -z "$MODEL" ] || [ -z "$POOL" ] || [ -z "$TRANSFORM" ]; then
  echo "!! ERRORE: servono tutti e tre gli argomenti." >&2
  echo "   uso: sbatch $0 <encoder> <pooling> <raw|whiten|head|head+whiten>" >&2
  echo "   La configurazione va scelta PRIMA sul valid (02_perquery_vision_valid.sh)." >&2
  exit 1
fi

# Trasformazione -> le due chiavi che la definiscono. Sono le stesse
# combinazioni di scripts/vision/04 (raw, whiten) e 05 (head, head+whiten).
case "$TRANSFORM" in
  raw)         HEAD=false; WHITEN=false ;;
  whiten)      HEAD=false; WHITEN=true  ;;
  head)        HEAD=true;  WHITEN=false ;;
  head+whiten) HEAD=true;  WHITEN=true  ;;
  *) echo "!! ERRORE: trasformazione '$TRANSFORM' non riconosciuta" >&2
     echo "   attese: raw | whiten | head | head+whiten" >&2
     exit 1 ;;
esac

# Il pooling deve essere valido per quell'encoder (i-jepa non ha `mean`):
# senza questa guardia si otterrebbe una run che gira e misura un'altra cosa.
VALID_POOLS=$(poolings_for "$MODEL")
case " $VALID_POOLS " in
  *" $POOL "*) ;;
  *) echo "!! ERRORE: pooling '$POOL' non valido per '$MODEL' (validi: $VALID_POOLS)" >&2
     exit 1 ;;
esac

echo "=== $(date) | TEST (una volta sola) | $(label_for "$MODEL" "$POOL" "$RES") / $TRANSFORM ==="
echo "⚠️  Se questa non e' la configurazione scelta sul valid, fermati: stai"
echo "    selezionando sul test."
nvidia-smi

python -m src.vision.evaluation.evaluate \
  model.name="$MODEL" model.variant="$(variant_for "$POOL" "$RES")" \
  model.kwargs.pooling="$POOL" $(res_flags "$RES") \
  partial.enabled=false eval.split=test eval.perquery_dir="$PERQUERY_DIR" \
  head.enabled=$HEAD whitening.enabled=$WHITEN
RC=$?

echo ""
if [ $RC -eq 0 ]; then
  echo "=== completato: $(date) ==="
  ls -la "$PERQUERY_DIR"
  echo ""
  echo "Confronto appaiato col ramo graph (gallery di taglia diversa ->"
  echo "serve il flag, e il caveat va nel report):"
  echo "  python -m src.evaluation.significance \\"
  echo "      --a $PERQUERY_DIR/vision_${MODEL}_$(variant_for "$POOL" "$RES")_<trasf>_full_test.npz \\"
  echo "      --b results/perquery/graph_test/<file>.npz \\"
  echo "      --k 10 --allow-gallery-mismatch"
else
  echo "=== FALLITO (rc=$RC): $(date) ===" >&2
fi
exit $RC
