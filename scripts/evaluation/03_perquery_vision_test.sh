#!/bin/bash
# scripts/evaluation/03_perquery_vision_test.sh
# Il TEST del ramo vision, per UNA sola configurazione gia' scelta sul valid.
#
# Questo script esiste per rendere difficile la cosa sbagliata. Il test non
# sceglie nulla (vincolo DURO 1): si legge una volta, per la configurazione gia'
# scelta sul valid. Per questo i tre argomenti sono OBBLIGATORI e non c'e'
# nessuna modalita' "tutti": una griglia sul test e' selezione sul test.
#
# 11 set 2026 — protocollo B (quello dei numeri finali, experiments.md):
#   - gallery condivisa e split dal YAML; qui si forza solo eval.split=test;
#   - modalita' `partial` (4° argomento): curva di degrado sul test, con
#     `self_rr` per query (criterio di robustezza, status.md § 23);
#   - cartella di default `vision_test_B`: i file del vecchio protocollo in
#     `vision_test/` (3 letture dell'8-9 ago, § 14) restano intatti;
#   - EXTRA per il checkpoint della head, es. EXTRA="head.file=head_probe_conv.pt".
#
# Uso:
#   sbatch scripts/evaluation/03_perquery_vision_test.sh <encoder> <pooling> <raw|whiten|head|head+whiten> [full|partial]
#   EXTRA="head.file=head_probe_conv.pt" \
#     sbatch scripts/evaluation/03_perquery_vision_test.sh dinov3 natural head full
#   EXTRA="head.file=head_probe_conv.pt" \
#     sbatch scripts/evaluation/03_perquery_vision_test.sh dinov3 natural head partial
#   Env: RES (tipsv2: 224|448) · PERQUERY_DIR · EXTRA · STRAT (partial: random | all | crop | patch | damage)
#
# Output: $PERQUERY_DIR/vision_<enc>_<pool>_<tag>_full_test.npz
#         $PERQUERY_DIR/vision_<enc>_<pool>_<tag>_partial-<strat>-f<frac>_test.npz

#SBATCH --job-name="ev_03_perquery_vision_test"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Esclude i nodi Blackwell (sm_120): stessa whitelist di 05/06 e scripts/vision/*.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR" || exit 1

MODEL="${1:-}"
POOL="${2:-}"
TRANSFORM="${3:-}"
MODE="${4:-full}"
# Risoluzione: "native" = quella del preset YAML (tutti tranne tipsv2, che ha
# due varianti per pooling). Deve essere la STESSA scelta sul valid.
RES="${RES:-native}"
PERQUERY_DIR="${PERQUERY_DIR:-results/perquery/vision_test_B}"
EXTRA="${EXTRA:-}"
STRAT="${STRAT:-random}"
mkdir -p logs "$PERQUERY_DIR"

if [ -z "$MODEL" ] || [ -z "$POOL" ] || [ -z "$TRANSFORM" ]; then
  echo "!! ERRORE: servono encoder, pooling e trasformazione." >&2
  echo "   uso: sbatch $0 <encoder> <pooling> <raw|whiten|head|head+whiten> [full|partial]" >&2
  echo "   La configurazione va scelta PRIMA sul valid." >&2
  exit 1
fi

# Trasformazione -> le due chiavi che la definiscono.
case "$TRANSFORM" in
  raw)         HEAD=false; WHITEN=false ;;
  whiten)      HEAD=false; WHITEN=true  ;;
  head)        HEAD=true;  WHITEN=false ;;
  head+whiten) HEAD=true;  WHITEN=true  ;;
  *) echo "!! ERRORE: trasformazione '$TRANSFORM' non riconosciuta" >&2
     echo "   attese: raw | whiten | head | head+whiten" >&2
     exit 1 ;;
esac

# Modalita': full (pianta intera) o partial (stanze tolte, curva di degrado).
# Le strategie seguono 05: `random` e' l'unica del criterio di robustezza.
case "$MODE" in
  full)    MODE_FLAGS="partial.enabled=false" ;;
  partial)
    # crop/patch (11 set 2026, vision_damage.py) spenti in modo esplicito in
    # random/all; damage = random + crop + patch. Stessi casi di 05.
    OFF_ROOMS="partial.strategies.semantic.enabled=false partial.strategies.topology.enabled=false"
    case "$STRAT" in
      random) MODE_FLAGS="partial.enabled=true $OFF_ROOMS partial.strategies.crop.enabled=false partial.strategies.patch.enabled=false" ;;
      all)    MODE_FLAGS="partial.enabled=true partial.strategies.crop.enabled=false partial.strategies.patch.enabled=false" ;;
      crop)   MODE_FLAGS="partial.enabled=true partial.strategies.random.enabled=false $OFF_ROOMS partial.strategies.crop.enabled=true partial.strategies.patch.enabled=false" ;;
      patch)  MODE_FLAGS="partial.enabled=true partial.strategies.random.enabled=false $OFF_ROOMS partial.strategies.crop.enabled=false partial.strategies.patch.enabled=true" ;;
      damage) MODE_FLAGS="partial.enabled=true $OFF_ROOMS partial.strategies.crop.enabled=true partial.strategies.patch.enabled=true" ;;
      *) echo "!! ERRORE: STRAT '$STRAT' non riconosciuta (attese: random | all | crop | patch | damage)" >&2
         exit 1 ;;
    esac ;;
  *) echo "!! ERRORE: modalita' '$MODE' non riconosciuta (attese: full | partial)" >&2
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

echo "=== $(date) | TEST (una volta sola) | $(label_for "$MODEL" "$POOL" "$RES") / $TRANSFORM / $MODE ==="
echo "per-query -> $PERQUERY_DIR | extra: ${EXTRA:-nessuno} | strategie: $([ "$MODE" = partial ] && echo "$STRAT" || echo -)"
echo "⚠️  Se questa non e' la configurazione scelta (e pre-registrata) sul valid,"
echo "    fermati: stai selezionando sul test."
nvidia-smi

# shellcheck disable=SC2086  # MODE_FLAGS ed EXTRA vanno splittati in override separati
python -m src.vision.evaluation.evaluate \
  model.name="$MODEL" model.variant="$(variant_for "$POOL" "$RES")" \
  model.kwargs.pooling="$POOL" $(res_flags "$RES") \
  $MODE_FLAGS eval.split=test eval.perquery_dir="$PERQUERY_DIR" \
  head.enabled=$HEAD whitening.enabled=$WHITEN $EXTRA
RC=$?

echo ""
if [ $RC -eq 0 ]; then
  echo "=== completato: $(date) ==="
  ls -la "$PERQUERY_DIR"
  echo ""
  echo "Confronti appaiati col ramo graph (stessa gallery condivisa, nessun override):"
  echo "  full:    python -m src.evaluation.significance --a <vision>_full_test.npz --b <graph>_full_test.npz --k 10"
  echo "  partial: python -m src.evaluation.robustness_auc compare --split test \\"
  echo "             --a $PERQUERY_DIR/vision_<enc>_<pool>_<tag> --b results/perquery/graph_test_B/graph_<enc>_<var>"
else
  echo "=== FALLITO (rc=$RC): $(date) ===" >&2
fi
exit $RC
