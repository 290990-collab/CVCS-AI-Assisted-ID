#!/bin/bash
# Stage 04: retrieval evaluation of the trained graph encoders (+ baseline).
# Architecture/adjustment come from each encoder's training YAML (configs/graph_models/<name>.yaml,
# so the checkpoint reloads by construction); evaluation parameters from configs/graph_retrieval.yaml.
# With `baseline_hist: true` also evaluates the training-free type_histogram baseline.
# Needs the graph cache (stage 02) and checkpoints (stage 03).
#
# Uso:
#   sbatch scripts/graph/04_eval_gnn.sh                 # baseline + all three encoders (variant `base`)
#   sbatch scripts/graph/04_eval_gnn.sh gcn             # one encoder (no baseline), variant `base`
#   sbatch scripts/graph/04_eval_gnn.sh gcn tau02       # one encoder, one ablation variant
#   sbatch scripts/graph/04_eval_gnn.sh gcn ablation    # one encoder, all variants trained on disk
#   sbatch scripts/graph/04_eval_gnn.sh graph_sage      # graph_sage.yaml -> encoder "sage"
#   sbatch scripts/graph/04_eval_gnn.sh hist            # training-free baseline only
#
# Ablations are compared on the probe (valid, training_summary.json, stage 03 summary), not here:
# this script measures on TEST, to be touched only for the winning variant. `ablation` mode is for
# final numbers of several variants (e.g. the report).

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

# YAML -> graph_evaluate.py flags, restricted to the keys in $2 (CSV): training YAMLs contribute only
# architecture/adjustment, not optimisation params. Lists -> nargs flags; special booleans as in train_flags_from_yaml.
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

# Single YAML key value (bare string).
yaml_get() {
  python - "$1" "$2" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True) or {}
print(cfg.get(sys.argv[2], ""))
PY
}

# --- ablation variants (names as in 03_train_gnn.sh) ---
# Only architecture changes matter at eval (checkpoint reloads by construction): `noskip` narrows `proj`,
# `asymlost*` (and `_selfull` shadows) add the "lost neighbours" column (in_dim 19 -> 20). Other variants
# (temperature, augmentation, selection, `*rob`, `*_selfull`) only need `--variant`.
KNOWN_VARIANTS="base noskip nosym nojitter nd01 noaug tau02 tau05 selmean selgeom tau02asym asym asymlost asymrep asymrob asymrob_selfull asymlostrob asymlostrob_selfull asymrobrep asymrobrep_selfull"

variant_eval_flags() {
  case "$1" in
    noskip)                                                    echo "--no-raw-skip" ;;
    asymlost|asymlostrob|asymlostrob_selfull)                  echo "--lost-marker" ;;
    base|nosym|nojitter|nd01|noaug|tau02|tau05|selmean|selgeom|tau02asym|asym|asymrep) echo "" ;;
    asymrob|asymrob_selfull|asymrobrep|asymrobrep_selfull)     echo "" ;;
    # multi-seed replicas of asymrob (`asymrob_s<S>`): same network
    asymrob_s[0-9]*)
      if [[ "$1" =~ ^asymrob_s[0-9]+(_selfull)?$ ]]; then echo ""; else echo "__INVALID__"; fi ;;
    *)                                                         echo "__INVALID__" ;;
  esac
}

# Guard (as in 03): every declared name needs a case entry.
for V in $KNOWN_VARIANTS; do
  if [ "$(variant_eval_flags "$V")" = "__INVALID__" ]; then
    echo "!! ERRORE: '$V' e' in KNOWN_VARIANTS ma non ha una voce in variant_eval_flags"
    exit 1
  fi
done

# Variants actually trained for an encoder (dirs with encoder.pt); read from disk to avoid syncing with 03's list.
trained_variants() {
  local out=""
  for d in "embeddings/graph/$1"/*/; do
    [ -f "${d}encoder.pt" ] || continue
    out="$out $(basename "$d")"
  done
  echo $out
}

# No argument = `base`; `ablation`/`all` = all trained; otherwise the given name.
select_variants() {
  case "$2" in
    "")           echo "base" ;;
    ablation|all) trained_variants "$1" ;;
    *)            echo "$2" ;;
  esac
}

# true/false: training-free baseline enabled in the shared config?
baseline_enabled() {
  python - "$RETRIEVAL_CFG" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True) or {}
print("true" if cfg.get("baseline_hist") else "false")
PY
}

EVAL_FLAGS=$(eval_flags_from_yaml "$RETRIEVAL_CFG" "num_queries,seed,split,k_values,batch_size,gallery_names")

# Opt-in per-query output via PERQUERY_OUT (used by scripts/evaluation/04_perquery_graph.sh); unset = unchanged command line.
PERQUERY_FLAGS=""
[ -n "${PERQUERY_OUT:-}" ] && PERQUERY_FLAGS="--perquery-out $PERQUERY_OUT"
# Opt-in graph partial flags (used by scripts/evaluation/06_perquery_graph_partial_valid.sh).
[ -n "${GRAPH_PARTIAL_FLAGS:-}" ] && PERQUERY_FLAGS="$PERQUERY_FLAGS $GRAPH_PARTIAL_FLAGS"
# Opt-in eval flag override, appended after the YAML ones (last wins); used by scripts/evaluation/07_perquery_graph_test.sh for `--split test`.
[ -n "${GRAPH_EXTRA_FLAGS:-}" ] && PERQUERY_FLAGS="$PERQUERY_FLAGS $GRAPH_EXTRA_FLAGS"

TARGET="$1"
VARIANT_ARG="$2"

echo "=== $(date) | STAGE 04 eval GNN | target: ${TARGET:-baseline + tutti} | varianti: ${VARIANT_ARG:-base} ==="
echo "eval flags: $EVAL_FLAGS"
nvidia-smi

# Training-free baseline: full run (no argument) or alone (hist).
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
  # Registry key (gcn|gat|sage): namespaces embeddings/graph/<key>/<variant>/; differs from the YAML basename.
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
    # --variant (and architecture flag) after the YAML flags; last wins.
    python -m src.graph.evaluation.graph_evaluate $MODEL_FLAGS --variant "$V" $EXTRA $EVAL_FLAGS $PERQUERY_FLAGS
  done
done

echo ""
echo "=== completato: $(date) ==="