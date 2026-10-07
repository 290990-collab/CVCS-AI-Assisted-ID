"""
Final graph configurations: table of configurations, common recipe and paths in one place.

The shell scripts in scripts/final_pipeline/ never spell a flag or a path: they ask this module
(`python -m src.graph.final_graph_configs <cmd>`), so training, evaluation and selection
(`src/evaluation/graph_config_select.py`) cannot drift apart.

A configuration is a set of overrides of the encoder YAML (configs/graph_models/<name>.yaml) on top of the common recipe.
Train and eval flags are rendered from the same merged dict, with the special booleans of `scripts/graph/_common.sh`
(`train_flags_from_yaml`), so the checkpoint reloads with its training architecture.

Smoke mode (`--smoke`): prefix `rgsmoke`, 2 epochs, results under results/final_pipeline_smoke/.

Usage (CPU, no torch import):

    python -m src.graph.final_graph_configs configs --encoder gat
    python -m src.graph.final_graph_configs paths --encoder gat --cfg t02 --seed 42 --split valid
    python -m src.graph.final_graph_configs train-flags --encoder gat --cfg t02 --seed 42
    python -m src.graph.final_graph_configs eval-flags --encoder gat --cfg t02 --seed 42 --split valid
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from omegaconf import OmegaConf

# --- fixed choices ---

PREFIX, SMOKE_PREFIX = "rg", "rgsmoke"
ROOT, SMOKE_ROOT = "results/final_pipeline", "results/final_pipeline_smoke"
SEEDS = (42, 100042, 200042, 300042)
EPOCHS, SMOKE_EPOCHS = 600, 2
BINDING_FRACTION = 0.9                     # best epoch > 540 of 600: budget binding
CTRL_SEED = 42                             # damage seed of the replica control
CTRL_REPLICA = 100042                      # replica fused with seed 42 in the control
FRACTIONS = (0.0, 0.25, 0.5, 0.75)

GRAPH_MODELS_DIR = Path("configs/graph_models")
RETRIEVAL_CFG = Path("configs/graph_retrieval.yaml")
EMBEDDINGS_DIR = Path("embeddings/graph")

# YAML basename -> registry key (folder name)
ENCODERS = {"gcn": "gcn", "graph_sage": "sage", "gat": "gat"}

# common recipe (= gat/asymrob, 600-epoch cap)
RECIPE = {
    "pair_mode": "asym_partial",
    "selection_probe": "partial",
    "epochs": EPOCHS,
    "patience": 0,
    "shadow_patience": 10,
    "shadow_epochs": 150,
}

# stage 1: one change per row with respect to `ref`
OVERRIDES = {
    "ref":    {},
    "t01":    {"temperature": 0.1},
    "t02":    {"temperature": 0.2},
    "t05":    {"temperature": 0.5},
    "noskip": {"raw_skip": False},
    "nosym":  {"flip_prob": 0.0, "rot_prob": 0.0},
    "lost":   {"lost_marker": True},
    "l3":     {"num_layers": 3},
    "h256":   {"hidden_dim": 256},
    "pmm":    {"pooling": "mean_max"},
    "amax":   {"aggr": "max"},
    "aadd":   {"aggr": "add"},
    "hd1":    {"heads": 1},
    "hd8":    {"heads": 8},
}
COMMON_CFGS = ("ref", "t01", "t02", "t05", "noskip", "nosym", "lost", "l3", "h256", "pmm")
SPECIFIC_CFGS = {"gcn": (), "sage": ("amax", "aadd"), "gat": ("hd1", "hd8")}

# factor of each change: alternatives of one factor are exclusive in the stage-2 combination (best net gain per factor kept)
FACTOR = {cfg: next(iter(ov)) if ov else None for cfg, ov in OVERRIDES.items()}
FACTOR["nosym"] = "symmetries"
COMB = "comb"

# keys passed to graph_evaluate (architecture + adjustment); the rest is training only
EVAL_MODEL_KEYS = ("encoder", "variant", "hidden_dim", "out_dim", "num_layers", "pooling",
                   "dropout", "raw_skip", "heads", "attn_dropout", "aggr", "normalize",
                   "drop_self_loops", "lost_marker")
EVAL_RETRIEVAL_KEYS = ("num_queries", "seed", "split", "k_values", "batch_size", "gallery_names")
SPLITS = ("valid", "test", "ctrl", "ctrltest")
# ctrl = valid, partial only, damage seed CTRL_SEED (replica control); ctrltest = the same on test
CTRL_SPLITS = ("ctrl", "ctrltest")


# --- pure helpers ---

def encoder_key(encoder: str) -> str:
    """YAML basename (gcn | graph_sage | gat) -> registry key (gcn | sage | gat)."""
    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder {encoder!r} (expected one of {sorted(ENCODERS)})")
    return ENCODERS[encoder]


def stage1_configs(encoder: str) -> tuple[str, ...]:
    """The stage-1 configurations of one encoder, in the order of the table."""
    return COMMON_CFGS + SPECIFIC_CFGS[encoder_key(encoder)]


def prefix(smoke: bool = False) -> str:
    return SMOKE_PREFIX if smoke else PREFIX


def root(smoke: bool = False) -> Path:
    return Path(SMOKE_ROOT if smoke else ROOT)


def epochs(smoke: bool = False) -> int:
    return SMOKE_EPOCHS if smoke else EPOCHS


def variant(cfg: str, seed: int, smoke: bool = False) -> str:
    return f"{prefix(smoke)}_{cfg}_s{int(seed)}"


def stage2_path(smoke: bool = False) -> Path:
    return root(smoke) / "selection" / "stage2.json"


def comb_overrides(encoder: str, smoke: bool = False) -> dict:
    """Stage-2 `comb` overrides of an encoder from stage2.json (FileNotFoundError if not written; ValueError if fewer than 2 net-gain changes)."""
    path = stage2_path(smoke)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m src.evaluation.graph_config_select stage1` first")
    combs = json.loads(path.read_text())["combs"]
    entry = combs.get(encoder_key(encoder))
    if not entry:
        raise ValueError(f"no stage-2 comb for {encoder_key(encoder)} (fewer than 2 netta changes)")
    return dict(entry["overrides"])


def config_overrides(encoder: str, cfg: str, smoke: bool = False) -> dict:
    """Overrides of `cfg` for `encoder`; refuses changes that do not apply to it."""
    if cfg == COMB:
        return comb_overrides(encoder, smoke)
    if cfg not in stage1_configs(encoder):
        raise ValueError(f"{cfg!r} is not a configuration of {encoder} "
                         f"(valid: {' '.join(stage1_configs(encoder))} {COMB})")
    return dict(OVERRIDES[cfg])


def load_yaml(path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True) or {}


def merged_config(encoder: str, cfg: str, seed: int, smoke: bool = False) -> dict:
    """YAML of the encoder + recipe + seed/variant + overrides of the configuration."""
    base = load_yaml(GRAPH_MODELS_DIR / f"{encoder}.yaml")
    if base.get("encoder") != encoder_key(encoder):
        raise ValueError(f"{encoder}.yaml: encoder {base.get('encoder')!r}, expected {encoder_key(encoder)!r}")
    out = {**base, **RECIPE, "epochs": epochs(smoke), "seed": int(seed),
           "variant": variant(cfg, seed, smoke)}
    ov = config_overrides(encoder, cfg, smoke)
    unknown = [k for k in ov if k not in out and k != "lost_marker"]
    if unknown:
        raise ValueError(f"{cfg}: override keys {unknown} not in {encoder}.yaml")
    out.update(ov)
    return out


# booleans with non-standard flags (rules of scripts/graph/_common.sh, plus lost_marker and no-raw-skip, which only configurations carry)
_SPECIAL = {
    "normalize":       lambda v: [] if v else ["--no-normalize"],
    "drop_self_loops": lambda v: [] if v else ["--keep-self-loops"],
    "raw_skip":        lambda v: ["--raw-skip"] if v else ["--no-raw-skip"],
    "wandb":           lambda v: ["--wandb"] if v else [],
    "lost_marker":     lambda v: ["--lost-marker"] if v else [],
}


def render_flags(cfg: dict, keys=None) -> list[str]:
    """dict -> argparse flags (`k_v` -> `--k-v`); `keys` restricts and orders them."""
    out = []
    for k in (keys if keys is not None else cfg.keys()):
        if k not in cfg or cfg[k] is None:
            continue
        v = cfg[k]
        if k in _SPECIAL:
            out += _SPECIAL[k](bool(v))
            continue
        if isinstance(v, bool):
            continue                       # no other boolean expected
        flag = f"--{k.replace('_', '-')}"
        out += [flag, *(str(x) for x in v)] if isinstance(v, list) else [flag, str(v)]
    return out


def train_flags(encoder: str, cfg: str, seed: int, smoke: bool = False) -> list[str]:
    return render_flags(merged_config(encoder, cfg, seed, smoke))


def eval_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"split {split!r} (expected one of {SPLITS})")
    return {"ctrl": "valid", "ctrltest": "test"}.get(split, split)


def eval_flags(encoder: str, cfg: str, seed: int, split: str, smoke: bool = False) -> list[str]:
    """Model flags (from the merged config) + retrieval flags (graph_retrieval.yaml).

    The retrieval `seed` is the QUERY sample seed (42), never the training seed.
    """
    model = merged_config(encoder, cfg, seed, smoke)
    retrieval = {**load_yaml(RETRIEVAL_CFG), "split": eval_split(split)}
    return render_flags(model, EVAL_MODEL_KEYS) + render_flags(retrieval, EVAL_RETRIEVAL_KEYS)


def partial_seed(seed: int, split: str) -> int:
    """Damage seed: the replica's own seed, CTRL_SEED in the control eval."""
    return CTRL_SEED if split in CTRL_SPLITS else int(seed)


def run_paths(encoder: str, cfg: str, seed: int, split: str = "valid", smoke: bool = False) -> dict:
    """Every path of one (configuration, seed, split). Refuses unknown configurations."""
    config_overrides(encoder, cfg, smoke)
    key, var = encoder_key(encoder), variant(cfg, seed, smoke)
    r = root(smoke)
    sub = f"ctrl_p{CTRL_SEED}" if split in CTRL_SPLITS else f"s{int(seed)}"
    return {
        "KEY": key,
        "VARIANT": var,
        "DEST": str(EMBEDDINGS_DIR / key / var),
        "TAG": f"graph_{key}_{var}",
        "EVAL_SPLIT": eval_split(split),
        "PARTIAL_SEED": str(partial_seed(seed, split)),
        "PQ_DIR": str(r / "perquery" / eval_split(split) / sub),
        "QV_DIR": str(r / "queryvec" / eval_split(split) / sub),
        "ROOT": str(r),
        "EPOCHS": str(epochs(smoke)),
    }


def recipe_record(encoder: str, cfg: str, seed: int, smoke: bool = False) -> dict:
    """Record 01_train_graph.sh writes next to the checkpoint (reset_recipe.json)."""
    return {
        "encoder": encoder, "key": encoder_key(encoder), "cfg": cfg, "seed": int(seed),
        "variant": variant(cfg, seed, smoke), "smoke": bool(smoke),
        "overrides": config_overrides(encoder, cfg, smoke),
        "recipe": {**RECIPE, "epochs": epochs(smoke)},
        "train_flags": train_flags(encoder, cfg, seed, smoke),
    }


# --- CLI ---

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RESET GRAPHS: configurations, flags, paths (§7)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("configs", help="stage-1 configurations of an encoder")
    c.add_argument("--encoder", required=True)
    for name in ("paths", "train-flags", "eval-flags", "recipe-json"):
        s = sub.add_parser(name)
        s.add_argument("--encoder", required=True)
        s.add_argument("--cfg", required=True)
        s.add_argument("--seed", required=True, type=int)
        if name in ("paths", "eval-flags"):
            s.add_argument("--split", required=True, choices=SPLITS)
        s.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    try:
        _main(parse_args(argv))
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"!! ERRORE final_graph_configs: {exc}")


def _main(args) -> None:
    if args.cmd == "configs":
        print(" ".join(stage1_configs(args.encoder)))
        return
    if args.seed not in SEEDS and not args.smoke:
        raise SystemExit(f"seed {args.seed} not in the pre-registered seeds {SEEDS}")
    if args.cmd == "paths":
        if args.split in CTRL_SPLITS and args.seed != CTRL_REPLICA:
            raise SystemExit(f"split {args.split} only for the replica seed {CTRL_REPLICA}")
        for k, v in run_paths(args.encoder, args.cfg, args.seed, args.split, args.smoke).items():
            print(f"{k}={shlex.quote(v)}")
    elif args.cmd == "train-flags":
        print(" ".join(shlex.quote(x) for x in train_flags(args.encoder, args.cfg, args.seed, args.smoke)))
    elif args.cmd == "eval-flags":
        print(" ".join(shlex.quote(x) for x in
                       eval_flags(args.encoder, args.cfg, args.seed, args.split, args.smoke)))
    else:
        print(json.dumps(recipe_record(args.encoder, args.cfg, args.seed, args.smoke), indent=2))


if __name__ == "__main__":
    main()
