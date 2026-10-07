#!/bin/bash
# Stage 03: self-supervised (InfoNCE) training of the graph encoders.
# Reads configs/graph_models/<name>.yaml per encoder, translates it to argparse flags
# (train_flags_from_yaml) and runs train_gnn.py. Best weights (selected on probe nDCG) go to
# embeddings/graph/<encoder>/<variant>/{encoder.pt,geom_stats.npz,training_summary.json}.
# Needs the graph cache (stage 02).
#
# Uso:
#   sbatch scripts/graph/03_train_gnn.sh                  # baseline, all three encoders
#   sbatch scripts/graph/03_train_gnn.sh gcn              # baseline, one encoder
#   sbatch scripts/graph/03_train_gnn.sh gcn ablation     # all ablation variants of gcn
#   sbatch scripts/graph/03_train_gnn.sh gcn tau05        # a single variant
#   sbatch scripts/graph/03_train_gnn.sh gat asymrob_s100042   # asymrob replica with --seed 100042
#   (YAML basename: graph_sage.yaml -> encoder "sage")
#
# `ablation` runs 10 trainings in a row: use one encoder per job (3 parallel jobs) to stay within the time limit.
# Variants are compared on `best_score` of training_summary.json (probe on valid), not on stage 04;
# the test split is touched only for the winning variant. The summary at the end of the job tabulates all variants on disk.

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

# --- ablation variants (OFAT: one change at a time w.r.t. the YAML `base`) ---
# Variant flags are appended after the YAML ones; argparse keeps the last.
ALL_VARIANTS="base noskip nosym nojitter nd01 noaug tau02 tau05 selmean selgeom tau02asym asym asymlost asymrep asymrob asymlostrob asymrobrep"

# Robustness selection: epoch with max self-recovery AUC (cap 300, no early stop).
# The historical rule runs as a shadow (patience/epochs from gat.yaml) into <encoder>/<variant>_selfull/ (evaluable with 04).
ROB="--selection-probe partial --epochs 300 --patience 0 --shadow-patience 10 --shadow-epochs 150"

variant_flags() {
  case "$1" in
    # --- reference: YAML as is ---
    base)     echo "" ;;

    # --- skip from raw add-pool ---
    noskip)   echo "--no-raw-skip" ;;

    # --- augmentation, split into parts ---
    nosym)    echo "--flip-prob 0 --rot-prob 0" ;;                              # no symmetries
    nojitter) echo "--geom-jitter 0" ;;                                         # no geometric jitter
    nd01)     echo "--node-drop 0.1" ;;                                         # old node_drop
    noaug)    echo "--flip-prob 0 --rot-prob 0 --geom-jitter 0 --node-drop 0.1" ;;  # augmentation fully off

    # --- InfoNCE temperature ---
    tau02)    echo "--temperature 0.2" ;;                                       # previous value
    tau05)    echo "--temperature 0.5" ;;                                       # upward exploration

    # --- asymmetric pairs (full <-> rooms removed) on tau02 ---
    tau02asym) echo "--temperature 0.2 --pair-mode asym_partial" ;;
    asym)      echo "--pair-mode asym_partial" ;;                               # asymmetric pairs on the reference YAML
    # --- "lost neighbours" marker on asym (changes in_dim: 04 passes the same flag) ---
    asymlost)  echo "--pair-mode asym_partial --lost-marker" ;;
    # replica of `asym` to measure training noise (unseeded init)
    asymrep)   echo "--pair-mode asym_partial" ;;
    # --- same pairs, checkpoint chosen on robustness (+ _selfull shadow) ---
    asymrob|asymrobrep) echo "--pair-mode asym_partial $ROB" ;;
    asymlostrob)        echo "--pair-mode asym_partial --lost-marker $ROB" ;;
    # --- multi-seed replicas of asymrob, `asymrob_s<S>` -> --seed S ---
    # Not in ALL_VARIANTS: passed by name. --seed after the YAML (seed: 42) wins; non-numeric suffix = unknown.
    asymrob_s[0-9]*)
      if [[ "$1" =~ ^asymrob_s([0-9]+)$ ]]; then
        echo "--pair-mode asym_partial $ROB --seed ${BASH_REMATCH[1]}"
      else
        echo "__INVALID__"
      fi ;;

    # --- checkpoint selection criterion ---
    # YAMLs already have `select_criterion: topology`, so `base` is the topology variant.
    selmean)  echo "--select-criterion mean" ;;                                 # mean of the three axes
    selgeom)  echo "--select-criterion geometry" ;;                             # least circular axis

    *)        echo "__INVALID__" ;;
  esac
}

# No argument = `base`; `ablation`/`all` = all variants; otherwise the given name.
select_variants() {
  case "$1" in
    "")           echo "base" ;;
    ablation|all) echo "$ALL_VARIANTS" ;;
    *)            echo "$1" ;;
  esac
}

# Guard: every name in ALL_VARIANTS needs a variant_flags entry, else it would be silently skipped.
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
    # --variant after the YAML flags: overrides `variant: base` and namespaces save_dir.
    python -m src.graph.training.train_gnn $BASE_FLAGS --variant "$V" $EXTRA
  done
done

# --- summary: all variants on disk (probe score on valid, higher is better) ---
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
    # sorted by encoder, then score descending
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
