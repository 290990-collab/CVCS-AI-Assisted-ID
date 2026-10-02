#!/bin/bash
# scripts/graph/04_eval_gnn.sh
# STAGE 04 — valutazione retrieval degli encoder di grafo addestrati (+ baseline).
# Per ogni encoder: architettura/adjustment dal SUO YAML di training
# (configs/graph_models/<name>.yaml, cosi' il checkpoint si ricarica per
# costruzione) + parametri di valutazione dal config condiviso
# (configs/graph_retrieval.yaml). Con `baseline_hist: true` valuta anche la
# baseline training-free (type_histogram, nessun checkpoint).
# Richiede la cache dei grafi (STAGE 02) e i checkpoint (STAGE 03).
#
# Uso:
#   sbatch scripts/graph/04_eval_gnn.sh                 # baseline + tutti e tre gli encoder (variante `base`)
#   sbatch scripts/graph/04_eval_gnn.sh gcn             # un solo encoder (niente baseline), variante `base`
#   sbatch scripts/graph/04_eval_gnn.sh gcn tau02       # un encoder, UNA variante di ablation
#   sbatch scripts/graph/04_eval_gnn.sh gcn ablation    # un encoder, TUTTE le varianti allenate su disco
#   sbatch scripts/graph/04_eval_gnn.sh graph_sage      # nota: file graph_sage.yaml -> encoder "sage"
#   sbatch scripts/graph/04_eval_gnn.sh hist            # solo la baseline training-free
#
# ⚠️ Il confronto tra ablation si fa sulla SONDA (valid, `training_summary.json`,
#    riepilogo dello stage 03), non qui: questo script misura sul TEST, che va
#    toccato solo per la variante vincente. La modalita' `ablation` serve quando
#    servono i numeri finali di piu' varianti (es. per il report), non per
#    scegliere.

#SBATCH --job-name="gp_04_eval_gnn"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/graph/_common.sh
cd "$PROJECT_DIR"
mkdir -p logs

RETRIEVAL_CFG="configs/graph_retrieval.yaml"

# Ponte YAML -> flag argparse di graph_evaluate.py, ristretto alle chiavi in $2
# (CSV): dai YAML di training vanno presi SOLO architettura/adjustment, non i
# parametri di ottimizzazione (lr/epochs/...) che graph_evaluate non accetta.
# Liste (k_values) -> flag nargs; booleani speciali come in train_flags_from_yaml.
eval_flags_from_yaml() {
  python - "$1" "$2" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True) or {}
keys = sys.argv[2].split(",")

special = {
    "normalize":       lambda v: [] if v else ["--no-normalize"],
    "drop_self_loops": lambda v: [] if v else ["--keep-self-loops"],
    "raw_skip":        lambda v: ["--raw-skip"] if v else [],
}

out = []
for k in keys:
    if k not in cfg or cfg[k] is None:
        continue
    v = cfg[k]
    if k in special:
        out += special[k](bool(v))
        continue
    flag = f"--{k.replace('_', '-')}"
    if isinstance(v, list):
        out += [flag] + [str(x) for x in v]
    else:
        out += [flag, str(v)]

print(" ".join(out))
PY
}

# Valore singolo di una chiave del YAML (stringa nuda, senza flag).
yaml_get() {
  python - "$1" "$2" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True) or {}
print(cfg.get(sys.argv[2], ""))
PY
}

# ----------------------------------------------------------------------
# Varianti di ablation (i nomi sono quelli di 03_train_gnn.sh).
#
# In VALUTAZIONE conta solo cio' che cambia l'ARCHITETTURA: il checkpoint si
# ricarica per costruzione, quindi una forma diversa = errore di caricamento.
# Le varianti di temperatura, augmentation e criterio di selezione hanno
# cambiato il TRAINING, non la rete: per loro basta `--variant` (che namespacia
# la cartella del checkpoint). Le eccezioni sono due: `noskip`, che toglie la
# concatenazione dell'add-pool grezzo e quindi restringe `proj`, e `asymlost`
# (con `asymlostrob` e la sua ombra `_selfull`), che aggiunge la colonna "vicini
# persi" e quindi porta in_dim da 19 a 20. Le varianti `*rob` (15 set 2026) e le
# loro `*_selfull` (checkpoint della regola storica, stesso training) cambiano
# solo la SELEZIONE dell'epoca, non la rete.
# ----------------------------------------------------------------------
KNOWN_VARIANTS="base noskip nosym nojitter nd01 noaug tau02 tau05 selmean selgeom tau02asym asym asymlost asymrep asymrob asymrob_selfull asymlostrob asymlostrob_selfull asymrobrep asymrobrep_selfull"

variant_eval_flags() {
  case "$1" in
    noskip)                                                    echo "--no-raw-skip" ;;
    asymlost|asymlostrob|asymlostrob_selfull)                  echo "--lost-marker" ;;
    base|nosym|nojitter|nd01|noaug|tau02|tau05|selmean|selgeom|tau02asym|asym|asymrep) echo "" ;;
    asymrob|asymrob_selfull|asymrobrep|asymrobrep_selfull)     echo "" ;;
    # repliche multi-seed di asymrob (1 ott 2026, 03: `asymrob_s<S>`): stessa rete
    asymrob_s[0-9]*)
      if [[ "$1" =~ ^asymrob_s[0-9]+(_selfull)?$ ]]; then echo ""; else echo "__INVALID__"; fi ;;
    *)                                                         echo "__INVALID__" ;;
  esac
}

# Guardia (gemella di quella in 03): ogni nome dichiarato deve avere una voce
# nel case. Senza, una variante elencata ma non gestita verrebbe solo saltata a
# runtime e il job sembrerebbe completato con una valutazione in meno.
for V in $KNOWN_VARIANTS; do
  if [ "$(variant_eval_flags "$V")" = "__INVALID__" ]; then
    echo "!! ERRORE: '$V' e' in KNOWN_VARIANTS ma non ha una voce in variant_eval_flags"
    exit 1
  fi
done

# Varianti effettivamente ALLENATE per un encoder (= cartelle con encoder.pt).
# Le si legge da disco invece di riusare la lista di 03 per due motivi: qui si
# puo' valutare solo cio' che esiste (es. `selgeom` non e' mai stata allenata),
# e una variante nuova aggiunta a 03 viene raccolta senza dover sincronizzare
# due liste.
trained_variants() {
  local out=""
  for d in "embeddings/graph/$1"/*/; do
    [ -f "${d}encoder.pt" ] || continue
    out="$out $(basename "$d")"
  done
  echo $out
}

# Varianti da valutare: nessun argomento = solo `base` (comportamento storico
# dello script); `ablation`/`all` = tutte quelle allenate; altrimenti il nome passato.
select_variants() {
  case "$2" in
    "")           echo "base" ;;
    ablation|all) trained_variants "$1" ;;
    *)            echo "$2" ;;
  esac
}

# true/false: la baseline training-free e' abilitata nel config condiviso?
baseline_enabled() {
  python - "$RETRIEVAL_CFG" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True) or {}
print("true" if cfg.get("baseline_hist") else "false")
PY
}

EVAL_FLAGS=$(eval_flags_from_yaml "$RETRIEVAL_CFG" "num_queries,seed,split,k_values,batch_size,gallery_names")

# Persistenza dei valori per-query (fase A.3), opt-in via variabile d'ambiente:
# senza PERQUERY_OUT la riga di comando resta IDENTICA a prima, quindi questo
# script continua a comportarsi esattamente come sempre. La usa
# scripts/evaluation/04_perquery_graph.sh, cosi' il ponte YAML->flag resta in un
# solo posto invece di essere copiato una terza volta.
PERQUERY_FLAGS=""
[ -n "${PERQUERY_OUT:-}" ] && PERQUERY_FLAGS="--perquery-out $PERQUERY_OUT"
# Partial sul grafo (fase C.0), opt-in via env come PERQUERY_OUT: senza, la
# riga di comando resta identica. La usa scripts/evaluation/06_perquery_graph_partial_valid.sh.
[ -n "${GRAPH_PARTIAL_FLAGS:-}" ] && PERQUERY_FLAGS="$PERQUERY_FLAGS $GRAPH_PARTIAL_FLAGS"
# Override opt-in dei flag di valutazione (11 set 2026), appesi DOPO quelli del
# YAML: argparse fa vincere l'ultimo. Lo usa scripts/evaluation/07_perquery_graph_test.sh
# per `--split test` senza toccare configs/graph_retrieval.yaml. Senza, nulla cambia.
[ -n "${GRAPH_EXTRA_FLAGS:-}" ] && PERQUERY_FLAGS="$PERQUERY_FLAGS $GRAPH_EXTRA_FLAGS"

TARGET="$1"
VARIANT_ARG="$2"

echo "=== $(date) | STAGE 04 eval GNN | target: ${TARGET:-baseline + tutti} | varianti: ${VARIANT_ARG:-base} ==="
echo "eval flags: $EVAL_FLAGS"
nvidia-smi

# Baseline training-free: nella run completa (nessun argomento) o da sola (hist).
if { [ -z "$TARGET" ] || [ "$TARGET" = "hist" ]; } && [ "$(baseline_enabled)" = "true" ]; then
  echo ""
  echo "########  EVAL: baseline hist (training-free)  ########"
  python -m src.graph.evaluation.graph_evaluate --baseline-hist $EVAL_FLAGS $PERQUERY_FLAGS
fi
if [ "$TARGET" = "hist" ]; then
  echo ""
  echo "=== completato: $(date) ==="
  exit 0
fi

for NAME in $(select_encoders "$TARGET"); do
  CFG="configs/graph_models/${NAME}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "!! config mancante: $CFG — salto"
    continue
  fi
  MODEL_FLAGS=$(eval_flags_from_yaml "$CFG" \
    "encoder,variant,hidden_dim,out_dim,num_layers,pooling,dropout,raw_skip,heads,attn_dropout,aggr,normalize,drop_self_loops")
  # Chiave del registry (gcn|gat|sage): e' quella che namespacia
  # embeddings/graph/<key>/<variante>/ e NON coincide col basename del YAML
  # (graph_sage.yaml -> sage).
  ENC_KEY=$(yaml_get "$CFG" encoder)

  for V in $(select_variants "$ENC_KEY" "$VARIANT_ARG"); do
    EXTRA=$(variant_eval_flags "$V")
    if [ "$EXTRA" = "__INVALID__" ]; then
      echo "!! variante sconosciuta: '$V' (note: $KNOWN_VARIANTS) — salto"
      continue
    fi
    CKPT="embeddings/graph/${ENC_KEY}/${V}/encoder.pt"
    if [ ! -f "$CKPT" ]; then
      echo "!! checkpoint mancante: $CKPT — salto (variante non allenata?)"
      continue
    fi
    echo ""
    echo "########  EVAL GNN: $NAME | variante: $V  ########"
    echo "model flags: $MODEL_FLAGS --variant $V $EXTRA"
    # `--variant` (e l'eventuale flag di architettura) DOPO i flag del YAML:
    # argparse fa vincere l'ultimo, come in 03_train_gnn.sh.
    python -m src.graph.evaluation.graph_evaluate $MODEL_FLAGS --variant "$V" $EXTRA $EVAL_FLAGS $PERQUERY_FLAGS
  done
done

echo ""
echo "=== completato: $(date) ==="