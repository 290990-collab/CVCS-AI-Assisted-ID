#!/bin/bash
# scripts/graph/03_train_gnn.sh
# STAGE 03 — training self-supervised (InfoNCE) degli encoder di grafo.
# Per ogni encoder legge la sua ricetta in configs/graph_models/<name>.yaml,
# la traduce in flag argparse (train_flags_from_yaml) e lancia train_gnn.py.
# Salva i pesi migliori (selezione sull'nDCG della sonda, vedi train_gnn.py) in
# embeddings/graph/<encoder>/<variant>/{encoder.pt,geom_stats.npz,training_summary.json}.
# Richiede la cache dei grafi (STAGE 02).
#
# Uso:
#   sbatch scripts/graph/03_train_gnn.sh                  # baseline, tutti e tre gli encoder
#   sbatch scripts/graph/03_train_gnn.sh gcn              # baseline, un solo encoder
#   sbatch scripts/graph/03_train_gnn.sh gcn ablation     # TUTTE le varianti di ablation di gcn
#   sbatch scripts/graph/03_train_gnn.sh gcn tau05        # una singola variante
#   (basename del YAML: graph_sage.yaml -> encoder "sage")
#
# ⚠️ In modalita' `ablation` lancia 10 training di fila: usare UN ENCODER PER JOB
#    (3 job in parallelo), altrimenti si rischia di sforare il time limit.
#
# Confronto delle varianti: NON serve lo stage 04. Ogni run scrive
# `training_summary.json` con il `best_score` della sonda (retrieval sul VALID),
# ed e' quello il numero con cui si confrontano le ablation — il test si tocca
# solo alla fine, per la variante vincente. Il riepilogo in coda al job stampa la
# tabella di tutte le varianti presenti su disco.

#SBATCH --job-name="gp_03_train_gnn"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/graph/_common.sh
cd "$PROJECT_DIR"
mkdir -p logs

# ----------------------------------------------------------------------
# Varianti di ablation (OFAT: una modifica per volta rispetto al YAML).
#
# Il YAML e' il riferimento `base` (tutti i cambiamenti del 28 lug attivi);
# ogni variante lo riporta indietro di UN pezzo, cosi' la differenza di
# best_score e' attribuibile a quel pezzo. I flag della variante vengono
# aggiunti DOPO quelli del YAML: argparse fa vincere l'ultimo della riga.
# ----------------------------------------------------------------------
ALL_VARIANTS="base noskip nosym nojitter nd01 noaug tau02 tau05 selmean selgeom"

variant_flags() {
  case "$1" in
    # --- riferimento: il YAML cosi' com'e' ---
    base)     echo "" ;;

    # --- cambio 2: skip dall'add-pool grezzo (composizione/geometria garantite) ---
    noskip)   echo "--no-raw-skip" ;;

    # --- cambio 1: augmentation (scomposto nei suoi pezzi) ---
    nosym)    echo "--flip-prob 0 --rot-prob 0" ;;                              # niente simmetrie (le invarianze VERE)
    nojitter) echo "--geom-jitter 0" ;;                                         # niente jitter geometrico
    nd01)     echo "--node-drop 0.1" ;;                                         # vecchio node_drop (51% vs 75% di grafi toccati)
    noaug)    echo "--flip-prob 0 --rot-prob 0 --geom-jitter 0 --node-drop 0.1" ;;  # cambio 1 completamente disattivato

    # --- temperatura InfoNCE (8,17% di falsi negativi sulla composizione) ---
    tau02)    echo "--temperature 0.2" ;;                                       # valore precedente
    tau05)    echo "--temperature 0.5" ;;                                       # esplorazione verso l'alto

    # --- criterio di selezione del checkpoint ---
    # NB: i YAML hanno gia' `select_criterion: topology`, quindi `base` E' la
    # variante "topologia": una `seltopo` sarebbe un duplicato esatto. Le due
    # alternative che portano informazione sono la media e la geometria.
    selmean)  echo "--select-criterion mean" ;;                                 # media dei tre assi (criterio precedente)
    selgeom)  echo "--select-criterion geometry" ;;                             # asse "onesto" (l'unico senza circolarita')

    *)        echo "__INVALID__" ;;
  esac
}

# Varianti da eseguire: nessun argomento = solo `base` (comportamento storico
# dello script); `ablation`/`all` = tutte; altrimenti il nome passato.
select_variants() {
  case "$1" in
    "")           echo "base" ;;
    ablation|all) echo "$ALL_VARIANTS" ;;
    *)            echo "$1" ;;
  esac
}

# Guardia: ogni nome in ALL_VARIANTS deve avere una voce in variant_flags.
# Senza, una variante elencata ma non definita verrebbe solo saltata a runtime e
# il job girerebbe 9 varianti su 10 senza che nessuno se ne accorga (successo
# apparente): meglio fermarsi subito e correggere lo script.
for V in $ALL_VARIANTS; do
  if [ "$(variant_flags "$V")" = "__INVALID__" ]; then
    echo "!! ERRORE: '$V' e' in ALL_VARIANTS ma non ha una voce in variant_flags — correggi lo script"
    exit 1
  fi
done

ENCODERS=$(select_encoders "$1")
VARIANTS=$(select_variants "$2")

echo "=== $(date) | STAGE 03 training GNN ==="
echo "encoder:  $ENCODERS"
echo "varianti: $VARIANTS"
nvidia-smi

for NAME in $ENCODERS; do
  CFG="configs/graph_models/${NAME}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "!! config mancante: $CFG — salto"
    continue
  fi
  BASE_FLAGS=$(train_flags_from_yaml "$CFG")

  for V in $VARIANTS; do
    EXTRA=$(variant_flags "$V")
    if [ "$EXTRA" = "__INVALID__" ]; then
      echo "!! variante sconosciuta: '$V' (valide: $ALL_VARIANTS) — salto"
      continue
    fi
    echo ""
    echo "########  TRAIN GNN: $NAME | variante: $V  ########"
    echo "flags: $BASE_FLAGS --variant $V $EXTRA"
    # `--variant` dopo i flag del YAML: sovrascrive `variant: base` e namespacia
    # il save_dir, cosi' le varianti non si sovrascrivono a vicenda.
    python -m src.graph.training.train_gnn $BASE_FLAGS --variant "$V" $EXTRA
  done
done

# ----------------------------------------------------------------------
# Riepilogo: tabella di tutte le varianti presenti su disco, per confrontare
# le ablation senza rileggere i log. Il punteggio e' quello della sonda sul
# VALID (piu' alto = meglio); il test si tocca solo alla fine, con lo stage 04.
# ----------------------------------------------------------------------
echo ""
echo "=== riepilogo varianti (sonda sul valid, non il test) ==="
python - <<'PY'
import glob, json, os

rows = []
for path in glob.glob("embeddings/graph/*/*/training_summary.json"):
    try:
        d = json.load(open(path))
    except Exception:
        continue
    rows.append((
        d.get("encoder", "?"), d.get("variant", os.path.basename(os.path.dirname(path))),
        d.get("selection_criterion", "?"), d.get("best_score", float("nan")),
        d.get("best_epoch", 0), d.get("epochs_max", 0),
        (d.get("probe_scores_at_best") or {}),
    ))

if not rows:
    print("(nessun training_summary.json trovato)")
else:
    hdr = f"{'encoder':<8}{'variante':<10}{'criterio':<22}{'best':>8}{'epoch':>7}{'/max':>6}   {'C':>6}{'T':>6}{'G':>6}"
    print(hdr); print("-" * len(hdr))
    # ordinati per encoder, poi per punteggio decrescente: in cima la variante migliore.
    for enc, var, crit, score, ep, mx, sc in sorted(rows, key=lambda r: (r[0], -r[3])):
        c = sc.get("ndcg_composition"); t = sc.get("ndcg_topology"); g = sc.get("ndcg_geometry")
        fmt = lambda v: f"{v:>6.3f}" if isinstance(v, (int, float)) else f"{'—':>6}"
        print(f"{enc:<8}{var:<10}{crit:<22}{score:>8.4f}{ep:>7}{mx:>6}   {fmt(c)}{fmt(t)}{fmt(g)}")
    print()
    print("⚠️  I punteggi della sonda NON sono confrontabili con il report finale")
    print("    (gallery piu' piccola -> IDCG diverso): servono al confronto RELATIVO")
    print("    tra varianti, che essendo su query fisse e' appaiato e molto sensibile.")
PY

echo ""
echo "=== completato: $(date) ==="
