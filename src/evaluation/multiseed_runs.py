"""Evaluations, controls and paths of the second graph round (writes under `results/final_pipeline/round2/`, new folders only).

- evaluations (GPU, `scripts/final_pipeline/12_eval_graph_multiseed.sh`): already trained graph checkpoints
  evaluated with a given damage seed, partial only, query vectors on. Gallery embeddings go to a new folder
  (`graph_evaluate --gallery-out`): `embeddings/graph/` is never rewritten. Needed because re-evaluating
  rewrites `embeddings.npy` and GPU runs are not bit-identical, so old query vectors no longer match the
  gallery on disk and cannot feed new fusions (`late_fusion` checks the sha1).
- controls (CPU, `scripts/final_pipeline/13_ensemble_controls_multiseed.sh`): graph + graph fusions with a
  second copy of the same config (another training seed, same removed rooms) and the same config of another
  encoder (SAGE), for
    A/B  W on the four replicas (s42 already done: `controls/` valid, `controls_test/` test);
    C    two graphs of strength similar to the vision: gat/t05 (~ frozen vision) and gat/ref (~ vision with head).
- the curve on the other three seeds (D): `fusion_curve --seed S` reads its graph inputs through
  `graph_source` (W from this round, the other configs from their stage-1 evaluation).

Gates: valid jobs only after `results/final_pipeline/ROUND2_PREREGISTERED`, test jobs only after
`results/final_pipeline/TEST_PREREGISTERED_R2`.

Usage (CPU, no torch):
    python -m src.evaluation.multiseed_runs evals --split valid              # the evaluations, one per line
    python -m src.evaluation.multiseed_runs eval-paths --split valid --index 0
    python -m src.evaluation.multiseed_runs controls --split valid
    python -m src.evaluation.multiseed_runs control-paths --split valid --index 0
    python -m src.evaluation.multiseed_runs gate --split valid
    python -m src.evaluation.multiseed_runs preflight --split valid          # inputs of the controls present + sha1
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from src.graph import final_graph_configs as rc

ROOT = rc.root() / "round2"
GATES = {"valid": rc.root() / "ROUND2_PREREGISTERED", "test": rc.root() / "TEST_PREREGISTERED_R2"}
SPLITS = ("valid", "test")
SEEDS = rc.SEEDS
W = ("gat", "comb")
E = ("graph_sage", "comb")
# second copy of W for replica S: the next training seed
COPY_OF = {42: 100042, 100042: 200042, 200042: 300042, 300042: 42}
OTHER_SEEDS = (100042, 200042, 300042)
# C: graphs of strength similar to the vision (gat/t05 ~ frozen, gat/ref ~ head); other encoder = same change
# on SAGE. Seed 42, damage seed 42, copy = s100042.
MID = {"gat_t05": (("gat", "t05"), ("graph_sage", "t05")),
       "gat_ref": (("gat", "ref"), ("graph_sage", "ref"))}
MID_SEED, MID_COPY = 42, 100042
# D on the test: reduced curve configurations; W from A
CURVE_TEST_CFGS = (("gcn", "ref"), ("gat", "t05"), ("gat", "ref"), ("gat", "t01"))
FULL_ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))     # as the controls
CURVE_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)                  # as the curve (C is compared to it)
FRACTIONS = (0.0, 0.25, 0.5, 0.75)


def _e(enc, cfg, seed, damage, why):
    return {"encoder": enc, "cfg": cfg, "seed": int(seed), "damage": int(damage), "why": why}


def evals(split: str) -> list[dict]:
    """Evaluations of the round in fixed order (index = SLURM array task)."""
    if split == "valid":
        out = [_e(*W, s, s, "W rivalutato: controlli A, curva D, head v2 H") for s in SEEDS]
        out += [_e(*W, COPY_OF[s], s, f"A: seconda copia di W per la replica {s}") for s in OTHER_SEEDS]
        out += [_e(*MID_CFG, MID_SEED, MID_SEED, f"C: {name} rivalutato (forza media)")
                for name, (MID_CFG, _) in MID.items()]
        out += [_e(*MID_CFG, MID_COPY, MID_SEED, f"C: seconda copia di {name}") for name, (MID_CFG, _) in MID.items()]
        return out
    if split == "test":
        out = [_e(*W, 100042, 100042, "A: W s100042 rivalutato (la valutazione ctrltest ha riscritto la gallery)")]
        out += [_e(*W, COPY_OF[s], s, f"A: seconda copia di W per la replica {s}") for s in OTHER_SEEDS]
        out += [_e(*E, s, s, f"A/E: sage/comb replica {s}") for s in OTHER_SEEDS]
        out += [_e(*MID_CFG, MID_COPY, MID_SEED, f"C: seconda copia di {name}") for name, (MID_CFG, _) in MID.items()]
        out += [_e(*SAGE_CFG, MID_SEED, MID_SEED, f"C: {SAGE_CFG[0]}/{SAGE_CFG[1]} (altro encoder)")
                for _, (_, SAGE_CFG) in MID.items()]
        out += [_e(enc, cfg, s, s, f"D: curva ridotta, replica {s}") for enc, cfg in CURVE_TEST_CFGS for s in OTHER_SEEDS]
        return out
    raise ValueError(f"split {split!r} (expected one of {SPLITS})")


def _find(split: str, enc: str, cfg: str, seed: int, damage: int) -> dict | None:
    for e in evals(split):
        if (e["encoder"], e["cfg"], e["seed"], e["damage"]) == (enc, cfg, int(seed), int(damage)):
            return e
    return None


def eval_paths(split: str, enc: str, cfg: str, seed: int, damage: int) -> dict:
    """Paths of one evaluation of this round (new folders only)."""
    base = rc.run_paths(enc, cfg, seed, split)
    sub = f"s{int(seed)}_p{int(damage)}"
    return {"ENC": enc, "CFG": cfg, "SEED": str(int(seed)), "PARTIAL_SEED": str(int(damage)),
            "KEY": base["KEY"], "VARIANT": base["VARIANT"], "TAG": base["TAG"], "DEST": base["DEST"],
            "EVAL_SPLIT": split,
            "PQ_DIR": str(ROOT / "perquery" / split / sub),
            "QV_DIR": str(ROOT / "queryvec" / split / sub),
            "GALLERY_DIR": str(ROOT / "gallery" / split / sub / base["KEY"] / base["VARIANT"])}


def graph_source(split: str, enc: str, cfg: str, seed: int, damage: int) -> tuple[str, str]:
    """(qvec prefix, per-query prefix) of a graph branch: this round's evaluation, else the stage-1 / test one
    (damage seed = training seed only)."""
    if _find(split, enc, cfg, seed, damage):
        p = eval_paths(split, enc, cfg, seed, damage)
        return f"{p['QV_DIR']}/{p['TAG']}", f"{p['PQ_DIR']}/{p['TAG']}"
    if int(damage) != int(seed):
        raise ValueError(f"{enc}/{cfg} s{seed} with damage {damage}: no evaluation in this round")
    p = rc.run_paths(enc, cfg, seed, split)
    return f"{p['QV_DIR']}/{p['TAG']}", f"{p['PQ_DIR']}/{p['TAG']}"


def _c(name, group, g1, g2, alphas):
    return {"name": name, "group": group, "g1": g1, "g2": g2, "alphas": alphas}


def controls(split: str) -> list[dict]:
    """graph + graph controls: g1 = graph under test (alpha=1 side), g2 = control model."""
    if split not in SPLITS:
        raise ValueError(f"split {split!r}")
    out = []
    for s in OTHER_SEEDS:
        w = (*W, s, s)
        out.append(_c(f"A_replica_s{s}", "A", w, (*W, COPY_OF[s], s), FULL_ALPHAS))
        out.append(_c(f"A_crossenc_s{s}", "A", w, (*E, s, s), FULL_ALPHAS))
    for name, (cfg, sage) in MID.items():
        g = (*cfg, MID_SEED, MID_SEED)
        out.append(_c(f"C_replica_{name}", "C", g, (*cfg, MID_COPY, MID_SEED), CURVE_ALPHAS))
        out.append(_c(f"C_crossenc_{name}", "C", g, (*sage, MID_SEED, MID_SEED), CURVE_ALPHAS))
    return out


def control_paths(split: str, index: int) -> dict:
    c = controls(split)[index]
    d = ROOT / "controls" / c["name"]
    q1, p1 = graph_source(split, *c["g1"])
    q2, p2 = graph_source(split, *c["g2"])
    return {"NAME": c["name"], "GROUP": c["group"],
            "G1_DESC": "{}/{} s{} danno {}".format(*c["g1"]), "G2_DESC": "{}/{} s{} danno {}".format(*c["g2"]),
            "G1_QVEC": q1, "G1_PQ": p1, "G2_QVEC": q2, "G2_PQ": p2,
            "ALPHAS": " ".join(f"{a:g}" for a in c["alphas"]),
            "DIR": str(d / f"fusion_{split}"), "CHECK": str(d / f"check_{split}.json"),
            "CHECK_RESET": str(d / f"check_{split}_reset.json"), "WAIVER": str(d / f"c3_waiver_{split}.json"),
            "SELECT": str(d / f"select_{split}.json"), "SELECT_VALID": str(d / "select_valid.json")}


def gate(split: str) -> list[str]:
    """Reasons a job of this split may not start (empty = allowed)."""
    g = GATES[split]
    out = [] if g.exists() else [f"manca {g}: pre-registrazione del secondo giro non sbloccata"]
    if split == "test" and not GATES["valid"].exists():
        out.append(f"manca anche {GATES['valid']}")
    return out


def preflight(split: str) -> list[str]:
    """Missing or inconsistent inputs of the controls (gallery sha1 of each evaluation)."""
    from src.evaluation import late_fusion as lf
    from src.evaluation.robustness_auc import load_auc
    shared = lf.load_shared_names("results/shared_gallery.json")
    bad, seen = [], set()
    for i, c in enumerate(controls(split)):
        for key in ("g1", "g2"):
            src = c[key]
            if src in seen:
                continue
            seen.add(src)
            q, p = graph_source(split, *src)
            try:
                gq = lf.load_branch_qvecs(q, split, FRACTIONS, lf.GRAPH_STRATEGY)
                ref = gq[FRACTIONS[0]]
                lf.load_graph_gallery(ref.meta["gallery_vectors"]["path"], shared,
                                      ref.meta["gallery_vectors"]["sha1"], "{}/{} s{} p{}".format(*src))
                load_auc(p, split, strategy=lf.GRAPH_STRATEGY)
            except Exception as e:  # noqa: BLE001 - report every input, then fail
                bad.append(f"{c['name']} {key} {src}: {e}")
    return bad


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="round2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("evals", "controls", "gate", "preflight"):
        s = sub.add_parser(name)
        s.add_argument("--split", required=True, choices=SPLITS)
    for name in ("eval-paths", "control-paths"):
        s = sub.add_parser(name)
        s.add_argument("--split", required=True, choices=SPLITS)
        s.add_argument("--index", required=True, type=int)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.cmd == "evals":
        for i, e in enumerate(evals(a.split)):
            print(i, e["encoder"], e["cfg"], f"s{e['seed']}", f"danno {e['damage']}", "·", e["why"])
    elif a.cmd == "controls":
        for i, c in enumerate(controls(a.split)):
            print(i, c["name"], "g1={}/{} s{} p{}".format(*c["g1"]), "g2={}/{} s{} p{}".format(*c["g2"]))
    elif a.cmd == "eval-paths":
        e = evals(a.split)[a.index]
        for k, v in eval_paths(a.split, e["encoder"], e["cfg"], e["seed"], e["damage"]).items():
            print(f"{k}={shlex.quote(v)}")
    elif a.cmd == "control-paths":
        for k, v in control_paths(a.split, a.index).items():
            print(f"{k}={shlex.quote(v)}")
    elif a.cmd == "gate":
        reasons = gate(a.split)
        for r in reasons:
            print(f"!! {r}", file=sys.stderr)
        raise SystemExit(1 if reasons else 0)
    else:
        bad = preflight(a.split)
        if bad:
            raise SystemExit("!! preflight FALLITO:\n" + "\n".join(bad))
        print(f"[round2] preflight {a.split} OK: tutti gli ingressi dei controlli presenti, sha1 della gallery giusti")


if __name__ == "__main__":
    main()
