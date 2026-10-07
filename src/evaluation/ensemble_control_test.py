"""Test-split check that the damage-trained vision adds information beyond the ensemble effect.

Seed 42, fusion weights fixed from the valid. Three fusions on the same 2000 test queries and removed rooms:
  F_H   = W s42 + vision with head          (fusion_head/s42/fusion_test, alpha 0.3)
  C_rep = W s42 + second copy of W          (W s100042 on the test, split `ctrltest`; beta 0.5 from controls/replica)
  C_enc = W s42 + sage/comb s42             (beta 0.6 from controls/crossenc)
Per query g = AUC(fusion) - AUC(best branch of its pair, higher mean AUC);
D_rep = g(F_H) - g(C_rep), D_enc = g(F_H) - g(C_enc), bootstrap CI 95% (B=10000, seed 0).
Outcome: both CI > 0 confirmed, both CI < 0 refuted, otherwise indistinguishable from the named control(s).
Descriptive: same with the frozen vision (fusion/s42, alpha 0.4).

Commands (CPU, seconds):
    python -m src.evaluation.ensemble_control_test paths            # shell variables for 10_ensemble_controls_test.sh
    python -m src.evaluation.ensemble_control_test decide           # -> controls_test/confirm_C_test.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path

from src.evaluation import fusion_select as fs
from src.evaluation import graph_config_select as rs
from src.graph import final_graph_configs as rc

SPLIT = "test"
SEED = 42
RULE = "status.md §62 C"
# weights from the valid, checked against the select json
ALPHA_HEAD, ALPHA_FROZEN = 0.3, 0.4
BETAS = {"replica": 0.5, "crossenc": 0.6}
ENC_CONTROL = ("graph_sage", "comb")
CONTROLS = ("replica", "crossenc")


def paths(smoke: bool = False) -> dict:
    fin = rs._final(smoke)
    w, e = fin["W"], fin["E"]
    if (e["encoder"], e["cfg"]) != ENC_CONTROL:
        raise ValueError(f"E = {e['encoder']}/{e['cfg']}, pre-registered {ENC_CONTROL} (status.md §62)")
    r = rc.root(smoke)
    w42 = rc.run_paths(w["encoder"], w["cfg"], SEED, SPLIT, smoke)
    rep = rc.run_paths(w["encoder"], w["cfg"], rc.CTRL_REPLICA, "ctrltest", smoke)
    enc = rc.run_paths(e["encoder"], e["cfg"], SEED, SPLIT, smoke)
    out = {"W_ENC": w["encoder"], "W_CFG": w["cfg"], "E_ENC": e["encoder"], "E_CFG": e["cfg"],
           "W42_QVEC": f"{w42['QV_DIR']}/{w42['TAG']}", "W42_PQ": f"{w42['PQ_DIR']}/{w42['TAG']}",
           "TEST_ROOT": str(r / "controls_test"),
           "CONFIRM_JSON": str(r / "controls_test" / f"confirm_C_{SPLIT}.json")}
    for ctrl, p in (("replica", rep), ("crossenc", enc)):
        u = ctrl.upper()
        d = r / "controls_test" / ctrl
        out.update({f"{u}_QVEC": f"{p['QV_DIR']}/{p['TAG']}", f"{u}_PQ": f"{p['PQ_DIR']}/{p['TAG']}",
                    f"{u}_DIR": str(d / f"fusion_{SPLIT}"),
                    f"{u}_CHECK": str(d / f"check_{SPLIT}.json"),
                    f"{u}_CHECK_RESET": str(d / f"check_{SPLIT}_reset.json"),
                    f"{u}_WAIVER": str(d / f"c3_waiver_{SPLIT}.json"),
                    f"{u}_SELECT": str(d / f"select_{SPLIT}.json"),
                    f"{u}_SELECT_VALID": str(r / "controls" / ctrl / "select_valid.json")})
    return out


def outcome(d_rep: dict, d_enc: dict) -> str:
    """Verdict from the two CIs."""
    pos = {k: d["ci_lo"] > 0 for k, d in (("rep", d_rep), ("enc", d_enc))}
    neg = {k: d["ci_hi"] < 0 for k, d in (("rep", d_rep), ("enc", d_enc))}
    if all(pos.values()):
        return "confermato: il vision allenato sul danno porta informazione oltre l'effetto d'insieme"
    if all(neg.values()):
        return "smentito: i controlli graph + graph guadagnano più della fusione con il vision con head"
    names = {"rep": "la seconda copia di W", "enc": "sage/comb"}
    which = [names[k] for k in ("rep", "enc") if not pos[k]]
    return "non distinguibile da almeno un controllo: " + " e ".join(which)


def _load_sel(path, fdir, expected_alpha: float, what: str) -> float:
    sel = json.loads(Path(path).read_text())
    if sel.get("split") != SPLIT or Path(str(sel.get("fusion_dir", ""))).resolve() != Path(fdir).resolve():
        raise ValueError(f"{what}: {path} does not describe {fdir} on the {SPLIT}")
    if sel.get("alpha_star_rule") != "fixed":
        raise ValueError(f"{what}: weight not fixed from the valid ({path})")
    a = float(sel["alpha_star"])
    if abs(a - expected_alpha) > 1e-9:
        raise ValueError(f"{what}: weight {a:g}, pre-registered {expected_alpha:g} (status.md §62)")
    return a


def _endpoints(fdir, star, pair):
    strategy = fs.pair_spec(pair)["fused"]
    return fs._fusion_endpoints(fdir, SPLIT, star, strategy, pair)


def decide(smoke: bool = False) -> dict:
    from src.evaluation import fusion_head as fh
    p = paths(smoke)
    hp = fh.head_paths(SEED, SPLIT, smoke)
    mp = rs.fusion_paths(SEED, SPLIT, smoke)
    fus = {
        "F_H": (hp["SELECT_JSON"], hp["FUSION_DIR"], ALPHA_HEAD, "vision-graph"),
        "F_frozen": (mp["SELECT_JSON"], mp["FUSION_DIR"], ALPHA_FROZEN, "vision-graph"),
        "C_rep": (p["REPLICA_SELECT"], p["REPLICA_DIR"], BETAS["replica"], "graph-graph"),
        "C_enc": (p["CROSSENC_SELECT"], p["CROSSENC_DIR"], BETAS["crossenc"], "graph-graph"),
    }
    loaded = {}
    for k, (sel, fdir, w, pair) in fus.items():
        star = _load_sel(sel, fdir, w, k)
        datas, comp = _endpoints(fdir, star, pair)
        loaded[k] = (datas, star, comp)
    sha = {d.meta["gallery"]["sha1"] for datas, _, _ in loaded.values() for d in datas.values()}
    if len(sha) != 1:
        raise ValueError("the fusions use different galleries")
    names = loaded["F_H"][0][ALPHA_HEAD].names

    def arrs(k):
        datas, star, comp = loaded[k]
        return (fs.align_by_name(names, datas[star].names, datas[star].auc),
                fs.align_by_name(names, datas[comp].names, datas[comp].auc))

    stats = {}
    for true in ("F_H", "F_frozen"):
        for ctrl in ("C_rep", "C_enc"):
            stats[f"{true}_vs_{ctrl}"] = fs.complementarity_stats(*arrs(true), *arrs(ctrl), rule="four_way")
    d_rep, d_enc = stats["F_H_vs_C_rep"]["D"], stats["F_H_vs_C_enc"]["D"]
    res = {
        "rule": RULE, "split": SPLIT, "seed": SEED, "n": int(len(names)), "gallery_sha1": sha.pop(),
        "written": datetime.now().isoformat(timespec="seconds"),
        "fusions": {k: {"dir": fus[k][1], "weight": loaded[k][1], "best_component": loaded[k][2],
                        "auc_means": {f"{a:g}": d.mean for a, d in loaded[k][0].items()}}
                    for k in fus},
        "D_rep": d_rep, "D_enc": d_enc,
        "g": {k: stats[f"F_H_vs_{k}"]["G_C"] for k in ("C_rep", "C_enc")} | {"F_H": stats["F_H_vs_C_rep"]["G_F"]},
        "outcome": outcome(d_rep, d_enc),
        "descriptive_frozen": {"g_F_frozen": stats["F_frozen_vs_C_rep"]["G_F"],
                               "D_rep": stats["F_frozen_vs_C_rep"]["D"],
                               "D_enc": stats["F_frozen_vs_C_enc"]["D"],
                               "note": "descrittivo (sul valid: non distinguibile, status.md §59.2-§59.3)"},
    }
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ensemble_control_test")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("paths", "decide"):
        s = sub.add_parser(name)
        s.add_argument("--smoke", action="store_true")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.cmd == "paths":
        for k, v in paths(a.smoke).items():
            print(f"{k}={shlex.quote(str(v))}")
        return
    out = Path(paths(a.smoke)["CONFIRM_JSON"])
    if out.exists():
        raise SystemExit(f"!! {out} esiste gia': niente sovrascritture")
    res = decide(a.smoke)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True))
    for k in ("D_rep", "D_enc"):
        v = res[k]
        print(f"  {k}: {v['mean']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")
    print(f"ESITO (§62 C): {res['outcome']}")
    print(f"[ensemble_control_test] scritto {out}")


if __name__ == "__main__":
    main()
