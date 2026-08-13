#!/bin/bash
# scripts/evaluation/02_perquery_vision_valid.sh
# FASE B.1 — griglia vision sul VALID, con persistenza per-query.
#
# Perche' esiste: oggi nessuno script vision valuta sul valid (rilievo A1), quindi
# la scelta fra encoder x pooling x trasformazione e' fatta leggendo il TEST. Qui
# la stessa griglia gira su `eval.split=valid`: si sceglie la vincente QUI, e il
# test si tocca UNA SOLA VOLTA con 03_perquery_vision_test.sh.
# La persistenza per-query (`eval.perquery_dir`) e' cio' che rende possibile il
# test appaiato fra due configurazioni: senza, restano due medie non confrontabili.
#
# Copertura: 5 encoder x pooling validi (14 combinazioni) x 4 trasformazioni
# {raw, whiten, head, head+whiten} = 56 valutazioni. Un file .npz per ciascuna,
# distinto dal tag <encoder>_<pooling>_<trasformazione>.
#
# ⚠️ La griglia pre-registrata da 56 run e' quella dei CINQUE encoder storici.
#    TRE encoder sono entrati DOPO in select_models (_common.sh): `tipsv2` con 6
#    feature (3 pooling x 2 risoluzioni) = +24 valutazioni, `pecore` con 3
#    feature (3 pooling, risoluzione fissa 224) = +12, e `pespatial` con 3
#    feature (3 pooling a 224) = +12. Lanciato SENZA argomenti questo script ora
#    ne fa 104 e RISCRIVE i .npz gia' prodotti. Per valutare solo il nuovo
#    encoder passarlo esplicitamente: `... 02_...sh pespatial`.
#
# ⚠️ Le 28 righe con la head dipendono dal checkpoint della head, che oggi e'
#    selezionato sulla val-loss InfoNCE — il criterio che il ramo graph ha
#    smentito (rilievo A3). Se la head verra' riallenata con una sonda di
#    retrieval, quelle 28 righe vanno RIFATTE. Per questo il gruppo e'
#    selezionabile: si puo' lanciare prima `frozen`, e `head` dopo il
#    riallenamento.
#
# Uso:
#   sbatch scripts/evaluation/02_perquery_vision_valid.sh              # tutti i modelli, tutte e 4 le trasformazioni
#   sbatch scripts/evaluation/02_perquery_vision_valid.sh dinov3       # un solo encoder
#   sbatch scripts/evaluation/02_perquery_vision_valid.sh "" frozen    # solo raw + whiten (niente head)
#   sbatch scripts/evaluation/02_perquery_vision_valid.sh dinov3 head  # solo head + head+whiten
#   RES=448 sbatch scripts/evaluation/02_perquery_vision_valid.sh tipsv2  # una risoluzione sola
#
# Output: results/perquery/vision_valid/vision_<enc>_<pool>_<trasf>_full_valid.npz

#SBATCH --job-name="ev_02_perquery_vision_valid"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=10:00:00
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
mkdir -p logs results/perquery/vision_valid

MODEL_ARG="${1:-}"
GROUP="${2:-all}"
# Risoluzione: vuota = tutte quelle previste per il modello (una sola per i
# cinque encoder storici). RES=448 ne restringe una, per parallelizzare i job.
RES_ARG="${RES:-}"
PERQUERY_DIR="results/perquery/vision_valid"

case "$GROUP" in
  all|frozen|head) ;;
  *) echo "!! ERRORE: gruppo '$GROUP' non riconosciuto (attesi: all | frozen | head)" >&2
     exit 1 ;;
esac

echo "=== $(date) | GRIGLIA VISION su VALID (fase B.1) ==="
echo "modelli: $(select_models "$MODEL_ARG") | gruppo: $GROUP | per-query -> $PERQUERY_DIR"
nvidia-smi

FAILED=0

# run_eval <etichetta> <override...>
run_eval() {
  local label="$1"; shift
  echo ""
  echo "########  VALID: $label  ########"
  python -m src.vision.evaluation.evaluate "$@"
  local rc=$?
  [ $rc -ne 0 ] && { echo "!! FALLITA: $label (rc=$rc)" >&2; FAILED=$((FAILED + 1)); }
}

for MODEL in $(select_models "$MODEL_ARG"); do
  for POOL in $(poolings_for "$MODEL"); do
    for RES in $(select_resolutions "$MODEL" "$RES_ARG"); do
      VAR=$(variant_for "$POOL" "$RES")
      LBL=$(label_for "$MODEL" "$POOL" "$RES")
      # Stesse chiavi di scripts/vision/04 e 05, cambiano solo split e per-query.
      COMMON="model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $(res_flags "$RES")"
      COMMON="$COMMON partial.enabled=false eval.split=valid eval.perquery_dir=$PERQUERY_DIR"

      if [ "$GROUP" = "all" ] || [ "$GROUP" = "frozen" ]; then
        run_eval "$LBL/raw"    $COMMON head.enabled=false whitening.enabled=false
        run_eval "$LBL/whiten" $COMMON head.enabled=false whitening.enabled=true
      fi

      if [ "$GROUP" = "all" ] || [ "$GROUP" = "head" ]; then
        run_eval "$LBL/head"        $COMMON head.enabled=true whitening.enabled=false
        run_eval "$LBL/head+whiten" $COMMON head.enabled=true whitening.enabled=true
      fi
    done
  done
done

echo ""
echo "=== completato: $(date) | valutazioni fallite: $FAILED ==="
ls -la "$PERQUERY_DIR" | tail -20
echo ""
echo "PROSSIMO PASSO — scegliere la vincente SUL VALID, poi confronti appaiati:"
echo "  python -m src.evaluation.significance --a $PERQUERY_DIR/<A>.npz --b $PERQUERY_DIR/<B>.npz --k 10"
echo "e SOLO DOPO il test una volta sola:"
echo "  sbatch scripts/evaluation/03_perquery_vision_test.sh <encoder> <pooling> <raw|whiten|head|head+whiten>"

[ $FAILED -eq 0 ] || exit 1
