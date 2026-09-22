#!/bin/bash
# scripts/evaluation/05_perquery_vision_partial_valid.sh
# FASE A.5 — curva di degrado sotto masking, sul VALID, CON persistenza per-query.
#
# Perche' esiste: dal 24 ago «migliore» = PIU' ROBUSTO (status.md § 23), e la
# misura e' l'AUC della curva self-recovery MRR. Gli script che fanno il partial
# (scripts/vision/06 e 07) NON valorizzano `eval.perquery_dir`: stampano la media
# nel log e buttano i valori per query. Senza quelli il delta appaiato — punto 5
# di § 23, quello che fa funzionare la regola di spareggio — non e' calcolabile.
# E' esattamente il motivo per cui a luglio e' nato 02_perquery_vision_valid.sh
# per la modalita' full: questo e' il suo gemello per il partial.
#
# Differenze da 02, tutte qui:
#   partial.enabled=true  (02 lo cabla a false)
#   cartella di output dedicata: results/perquery/vision_partial_valid
#   selezione delle strategie di masking ($3), default `random`
#
# ⚠️ UN FILE PER (config x strategia x frazione), non uno per config: il partial
#    scrive `vision_<tag>_partial-<strategia>-f<frazione>_valid.npz`
#    (evaluate.py:219-221). Con la sola strategia `random` sono 4 file per
#    configurazione (f = 0.0/0.25/0.5/0.75, da configs/vision_retrieval.yaml).
#
# ⚠️ MOLTO piu' pesante della full: ogni query va ri-degradata e ri-estratta.
#    Lanciare UN MODELLO PER JOB, come fanno 06 e 07.
#
# ⚠️ Il gruppo `head` richiede head.pt, che esiste SOLO per i 5 encoder storici
#    (dinov2, dinov3, siglip2, radio, ijepa). Per tipsv2/pecore/pespatial usare
#    `frozen`, altrimenti le run falliscono.
#    E le head oggi a disco sono selezionate sulla val-loss InfoNCE (rilievo A3):
#    se verranno riallenate con una sonda di retrieval, il gruppo `head` va
#    RIFATTO. Per questo e' selezionabile separatamente.
#
# Uso:
#   sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3
#   sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh pespatial frozen
#   sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 head
#   sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 all all
#   RES=448 sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh tipsv2 frozen
#   TRANSFORMS="raw head" sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 all damage
#     $1 = encoder (vuoto = tutti: sconsigliato, e' lunghissimo)
#     $2 = all | frozen | head        (default: all)
#     $3 = random | all | crop | patch | damage   (default: random -- l'unica che serve a § 23)
#          crop/patch (11 set 2026, src/vision/data/vision_damage.py): rettangolo sullo
#          snapshot / patch ViT intere; damage = random + crop + patch
#     env TRANSFORMS = filtro dentro il gruppo (raw | whiten | head | head+whiten,
#          separati da spazio o virgola); vuoto = tutte quelle del gruppo (storico)
#
# Output: results/perquery/vision_partial_valid/vision_<enc>_<pool>_<trasf>_partial-<strat>-f<frac>_valid.npz
# Analisi (CPU, dopo): confronto appaiato allo STESSO livello di masking
#   python -m src.evaluation.significance --metric self_rr --a <A>...f0.75_valid.npz --b <B>...f0.75_valid.npz

#SBATCH --job-name="ev_05_perquery_vision_partial_valid"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Esclude i nodi Blackwell (sm_120), su cui il PyTorch di floorplan-env crasha a
# ogni forward. Stessa whitelist di scripts/vision/04-07. ⚠️ 02/03/04 di questa
# cartella ne sono ancora sprovvisti (vedi .claude/TODO.md).
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR" || exit 1
mkdir -p logs results/perquery/vision_partial_valid

MODEL_ARG="${1:-}"
GROUP="${2:-all}"
STRAT="${3:-random}"
RES_ARG="${RES:-}"
# Override via env (10 set 2026): PERQUERY_DIR separa le run del protocollo B
# (gallery condivisa, exclude_self, whiten-train) dai 320 file di § 24, che
# altrimenti verrebbero sovrascritti (stesso nome per raw/head). EXTRA aggiunge
# override dotlist, es. EXTRA="whitening.fit_split=all" per il confronto B.2.
PERQUERY_DIR="${PERQUERY_DIR:-results/perquery/vision_partial_valid}"
EXTRA="${EXTRA:-}"
POOLS="${POOLS:-}"          # vuoto = tutti i pooling dell'encoder (es. POOLS=natural)
TRANSFORMS="${TRANSFORMS:-}"  # vuoto = tutte le trasformazioni del gruppo (es. TRANSFORMS="raw head")
TRANSFORMS="${TRANSFORMS//,/ }"
mkdir -p "$PERQUERY_DIR"

case "$GROUP" in
  all)    GROUP_TRANSFORMS="raw whiten head head+whiten" ;;
  frozen) GROUP_TRANSFORMS="raw whiten" ;;
  head)   GROUP_TRANSFORMS="head head+whiten" ;;
  *) echo "!! ERRORE: gruppo '$GROUP' non riconosciuto (attesi: all | frozen | head)" >&2
     exit 1 ;;
esac

for T in $TRANSFORMS; do
  case "$T" in
    raw|whiten|head|head+whiten) ;;
    *) echo "!! ERRORE: trasformazione '$T' in TRANSFORMS non riconosciuta (attese: raw | whiten | head | head+whiten)" >&2
       exit 1 ;;
  esac
done

# want <trasformazione>: vero se TRANSFORMS e' vuota (storico) o la contiene.
want() {
  [ -z "$TRANSFORMS" ] && return 0
  case " $TRANSFORMS " in *" $1 "*) return 0 ;; esac
  return 1
}

N_SELECTED=0
for T in $GROUP_TRANSFORMS; do want "$T" && N_SELECTED=$((N_SELECTED + 1)); done
if [ "$N_SELECTED" -eq 0 ]; then
  echo "!! ERRORE: 0 run selezionate (gruppo '$GROUP' = $GROUP_TRANSFORMS, TRANSFORMS='$TRANSFORMS')" >&2
  exit 1
fi

# `random` e' l'unica strategia che entra nel criterio § 23 (la curva di degrado
# a frazione crescente). `semantic` e `topology` sono informative per il report
# ma triplicano il costo: si attivano solo chiedendolo. crop/patch (danni non a
# stanze) sono spenti in modo esplicito in random/all: i comandi restano quelli
# di prima anche se il YAML li accendesse.
OFF_ROOMS="partial.strategies.semantic.enabled=false partial.strategies.topology.enabled=false"
case "$STRAT" in
  random) STRAT_FLAGS="$OFF_ROOMS partial.strategies.crop.enabled=false partial.strategies.patch.enabled=false" ;;
  all)    STRAT_FLAGS="partial.strategies.crop.enabled=false partial.strategies.patch.enabled=false" ;;
  crop)   STRAT_FLAGS="partial.strategies.random.enabled=false $OFF_ROOMS partial.strategies.crop.enabled=true partial.strategies.patch.enabled=false" ;;
  patch)  STRAT_FLAGS="partial.strategies.random.enabled=false $OFF_ROOMS partial.strategies.crop.enabled=false partial.strategies.patch.enabled=true" ;;
  damage) STRAT_FLAGS="$OFF_ROOMS partial.strategies.crop.enabled=true partial.strategies.patch.enabled=true" ;;
  *) echo "!! ERRORE: strategie '$STRAT' non riconosciute (attesi: random | all | crop | patch | damage)" >&2
     exit 1 ;;
esac

echo "=== $(date) | PARTIAL per-query su VALID (criterio A.5, status.md § 23) ==="
echo "modelli: $(select_models "$MODEL_ARG") | gruppo: $GROUP | strategie: $STRAT | trasformazioni: ${TRANSFORMS:-tutte}"
echo "per-query -> $PERQUERY_DIR | extra: ${EXTRA:-nessuno}"
nvidia-smi

FAILED=0

# run_eval <etichetta> <override...>
run_eval() {
  local label="$1"; shift
  echo ""
  echo "########  PARTIAL VALID: $label  ########"
  python -m src.vision.evaluation.evaluate "$@"
  local rc=$?
  [ $rc -ne 0 ] && { echo "!! FALLITA: $label (rc=$rc)" >&2; FAILED=$((FAILED + 1)); }
}

for MODEL in $(select_models "$MODEL_ARG"); do
  for POOL in ${POOLS:-$(poolings_for "$MODEL")}; do
    for RES in $(select_resolutions "$MODEL" "$RES_ARG"); do
      VAR=$(variant_for "$POOL" "$RES")
      LBL=$(label_for "$MODEL" "$POOL" "$RES")
      # Stesse chiavi di 02, cambia SOLO partial.enabled (e le strategie).
      COMMON="model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $(res_flags "$RES")"
      COMMON="$COMMON partial.enabled=true eval.split=valid eval.perquery_dir=$PERQUERY_DIR $STRAT_FLAGS $EXTRA"

      if [ "$GROUP" = "all" ] || [ "$GROUP" = "frozen" ]; then
        want raw    && run_eval "$LBL/raw"    $COMMON head.enabled=false whitening.enabled=false
        want whiten && run_eval "$LBL/whiten" $COMMON head.enabled=false whitening.enabled=true
      fi

      if [ "$GROUP" = "all" ] || [ "$GROUP" = "head" ]; then
        want head        && run_eval "$LBL/head"        $COMMON head.enabled=true whitening.enabled=false
        want head+whiten && run_eval "$LBL/head+whiten" $COMMON head.enabled=true whitening.enabled=true
      fi
    done
  done
done

echo ""
echo "=== completato: $(date) | valutazioni fallite: $FAILED ==="
ls -la "$PERQUERY_DIR" | tail -20
echo ""
echo "PROSSIMO PASSO — confronto appaiato allo STESSO livello di masking:"
echo "  python -m src.evaluation.significance --metric self_rr \\"
echo "      --a $PERQUERY_DIR/<A>_partial-random-f0.75_valid.npz \\"
echo "      --b $PERQUERY_DIR/<B>_partial-random-f0.75_valid.npz"
echo "⚠️ f=0.0 NON entra nell'AUC di § 23: li' tutti gli encoder fanno MRR 0.970,"
echo "   che e' un tetto dei DATI (duplicati esatti in RPLAN), non dei modelli."
case "$STRAT" in
  crop|patch|damage)
    echo "Curve dei danni crop/patch: python -m src.evaluation.robustness_auc rank --dir $PERQUERY_DIR --strategy crop  (o patch)" ;;
esac

[ $FAILED -eq 0 ] || exit 1
