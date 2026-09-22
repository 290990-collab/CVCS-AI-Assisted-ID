# src/evaluation/late_fusion.py

"""
Late fusion vision + graph by weighted vector concatenation (CPU only,
pre-registered in status.md §49, 16 Sep 2026).

What it computes
----------------
For every alpha in a grid (default {0, 0.1, ..., 1}):

    fused vector = [ sqrt(alpha) * v ; sqrt(1 - alpha) * g ]

where `v` is the vision vector AFTER the job's whitening (L2-normalized) and `g`
the graph vector (already L2-normalized by the encoder). Both halves have unit
norm, so the fused vector has unit norm and the inner product searched by FAISS
is exactly `alpha * sim_vision + (1 - alpha) * sim_graph`. alpha=1 = vision
only, alpha=0 = graph only (sqrt is exact at 0 and 1).

- Full plan: queries = gallery rows of the queries of the qvec files, self
  excluded like `graph_evaluate.evaluate`.
- Damaged plan: query vectors saved by the two partial evaluations (`qvec/1`,
  `src/evaluation/query_vectors.py`), vision `nowalls_random` and graph
  `random`, which remove the SAME rooms (checked query by query, hard error).
  Two views like `graph_evaluate.evaluate_partial`: self-recovery (`self_rr`)
  and per-axis metrics with the self removed.

Metrics come from the graph branch helpers (`axis_metrics.accumulate_axes`,
imported, not copied) and are written in the `perquery/1` format, so
`robustness_auc.load_auc(strategy="nowalls-random")` and `significance` read
the fused files as-is:

    <out>/fusion_a<alpha:g>_partial-nowalls-random-f<f>_<split>.npz
    <out>/fusion_a<alpha:g>_full_<split>.npz

The whitening of the gallery and of the query RAW vectors goes through
`VisionRetrievalPipeline._apply_whitening` (the same code as the vision job),
with the parameters saved by the vision job.

Usage (CPU; one fused gallery in memory at a time):

    python -m src.evaluation.late_fusion run --split valid \\
        --gallery-names results/shared_gallery.json \\
        --vision-qvec results/queryvec/valid/vision_pespatial_gem_whiten-train \\
        --graph-qvec  results/queryvec/valid/graph_gat_asymrob \\
        --out results/perquery/fusion_valid

Ensemble-effect control (status.md §50, 17 Sep 2026), opt-in `--pair graph-graph`:
two trainings of the same graph, `[sqrt(beta) * g1 ; sqrt(1 - beta) * g2]`
(beta=1 = `--graph-qvec`, beta=0 = `--graph2-qvec`), no whitening, both L2
checked, same removed rooms query by query, both gallery sha1 pinned. Fused files
are labelled `random` (`fusion_a<beta:g>_partial-random-f<f>_<split>.npz`) and
carry `fusion.pair = "graph-graph"`. A folder never mixes pairs (vision + graph
files have no `fusion.pair`). The default pair (vision-graph) is unchanged.

Control «different models, same information» (status.md §51), `--pair vision-vision`:
`[sqrt(gamma) * v1 ; sqrt(1 - gamma) * v2]` (gamma=1 = `--vision-qvec`, gamma=0 =
`--vision2-qvec`), each side whitened with the parameters of its own job (fit on
train, head null), same removed rooms, both gallery sha1 pinned, valid only.
Fused files labelled `nowalls-random`, `fusion.pair = "vision-vision"`.

    python -m src.evaluation.late_fusion run --pair graph-graph --split valid \\
        --graph-qvec  results/queryvec/valid/graph_gat_asymrob \\
        --graph2-qvec results/queryvec/valid/graph_gat_asymrobrep \\
        --out results/perquery/fusion_graphgraph_valid
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from src.data.rplan_metadata import load_metadata
from src.evaluation.gallery_join import load_shared_names, restrict_rows
from src.evaluation.perquery import PerQueryRecorder, gallery_sha1
from src.evaluation.query_vectors import QueryVectorData, array_sha1, load_qvec
from src.evaluation.relevance import DISCRETE_AXES, GalleryAxes
from src.evaluation.robustness_auc import fraction_path
from src.graph.evaluation.axis_metrics import accumulate_axes, new_metrics
from src.vision.models.retrieval_model import VisionRetrievalPipeline

VISION_STRATEGY = "nowalls-random"   # file-name strategy of the vision qvec/per-query files
GRAPH_STRATEGY = "random"            # file-name strategy of the graph qvec/per-query files
FUSED_STRATEGY = "nowalls-random"    # label of the fused files (meta declares graph_strategy)
GRAPHGRAPH_STRATEGY = "random"       # label of the graph + graph fused files (§50)
VISIONVISION_STRATEGY = "nowalls-random"   # label of the vision + vision fused files (§51)
PAIRS = ("vision-graph", "graph-graph", "vision-vision")
FUSION_METHOD = "weighted_concat_sqrt"
DEFAULT_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
DEFAULT_ALPHAS = tuple(i / 10 for i in range(11))
NORM_TOL = 1e-4                      # |‖g‖ - 1| above this = not the encoder output
SPREAD_K = 10                        # top-k of the descriptive similarity-spread diagnostic
ALPHA_NOMINAL_NOTE = ("α è nominale: conta lo spread delle similarità in cima al ranking, "
                      "non la dimensione dei vettori dei due modelli")


# ----------------------------------------------------------------------
# Naming.
# ----------------------------------------------------------------------

def fusion_prefix(out_dir, alpha: float) -> Path:
    """Prefix of the fused files of one alpha (`<out>/fusion_a0.3`)."""
    return Path(out_dir) / f"fusion_a{float(alpha):g}"


def fused_strategy(pair: str = "vision-graph") -> str:
    """File-name strategy of the fused files of a pair."""
    if pair not in PAIRS:
        raise ValueError(f"unknown pair {pair!r} (expected one of {PAIRS})")
    return {"vision-graph": FUSED_STRATEGY, "graph-graph": GRAPHGRAPH_STRATEGY,
            "vision-vision": VISIONVISION_STRATEGY}[pair]


def fused_partial_path(out_dir, alpha: float, frac: float, split: str,
                       strategy: str = FUSED_STRATEGY) -> Path:
    """Fused partial file, same convention as the branches (`robustness_auc`)."""
    return fraction_path(fusion_prefix(out_dir, alpha), frac, split, strategy)


def file_pair(meta: dict) -> str:
    """Pair of a fused file: `fusion.pair`, absent = vision-graph (files of §49)."""
    return (meta.get("fusion") or {}).get("pair", "vision-graph")


def check_output_pair(out_dir, pair: str) -> None:
    """Refuses to write fused files of `pair` into a folder holding the other pair.

    Raises:
        ValueError: naming the first fused file of the other pair.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return
    for path in sorted(out_dir.glob("fusion_a*.npz")):
        with np.load(path, allow_pickle=False) as z:
            found = file_pair(json.loads(str(z["meta"].item())))
        if found != pair:
            raise ValueError(f"{out_dir}: contains {found} fused files (e.g. {path.name}); "
                             f"refusing to write {pair} files there")


def spread_path(out_dir, split: str) -> Path:
    """Sidecar JSON of the similarity-spread diagnostic (alpha=0 and alpha=1 passes)."""
    return Path(out_dir) / f"fusion_spread_{split}.json"


def fused_full_path(out_dir, alpha: float, split: str) -> Path:
    """Fused full-plan file."""
    prefix = fusion_prefix(out_dir, alpha)
    return prefix.parent / f"{prefix.name}_full_{split}.npz"


# ----------------------------------------------------------------------
# Loading and hard checks.
# ----------------------------------------------------------------------

def load_branch_qvecs(prefix, split: str, fractions, strategy: str) -> dict:
    """Reads the qvec files of one branch, one per fraction.

    Returns:
        dict f -> QueryVectorData.

    Raises:
        FileNotFoundError: if a fraction is missing.
    """
    out = {}
    for f in fractions:
        path = fraction_path(prefix, f, split, strategy)
        if not path.exists():
            raise FileNotFoundError(f"qvec file missing for f={f}: {path}")
        out[float(f)] = load_qvec(path)
    return out


def _same(values) -> bool:
    first = json.dumps(values[0], sort_keys=True)
    return all(json.dumps(v, sort_keys=True) == first for v in values[1:])


def check_branch(datas: dict, branch: str, split: str, shared_sha1: str, n_shared: int) -> None:
    """Hard checks inside ONE branch, across its fractions.

    - branch, split, mode and gallery identity (sha1 and n of the shared file);
    - query_seed, partial_seed, k_values, run_tag and gallery vectors identical
      across fractions;
    - vision: head must be null; whitening meta and parameters identical across
      fractions (the fusion whitens with ONE set of parameters).

    Raises:
        ValueError: at the first inconsistency, with the offending field.
    """
    for f, d in datas.items():
        m = d.meta
        if m.get("branch") != branch:
            raise ValueError(f"{branch} f={f}: file of branch {m.get('branch')!r}")
        if m.get("split") != split:
            raise ValueError(f"{branch} f={f}: split {m.get('split')!r}, expected {split!r}")
        if m.get("mode") != "partial":
            raise ValueError(f"{branch} f={f}: mode {m.get('mode')!r}, expected 'partial'")
        g = m.get("gallery") or {}
        if g.get("sha1") != shared_sha1 or g.get("n") != n_shared:
            raise ValueError(
                f"{branch} f={f}: gallery n={g.get('n')} sha1={str(g.get('sha1'))[:12]} "
                f"differs from the shared gallery n={n_shared} sha1={shared_sha1[:12]}"
            )
        if branch == "vision" and m.get("head") is not None:
            raise ValueError(f"vision f={f}: head {m.get('head')!r} is not null "
                             "(the fusion is pre-registered on the frozen encoder)")

    metas = [d.meta for d in datas.values()]
    for key in ("query_seed", "partial_seed", "k_values", "run_tag", "gallery_vectors"):
        if not _same([m.get(key) for m in metas]):
            raise ValueError(f"{branch}: '{key}' differs across fractions")
    if branch == "vision":
        if not _same([m.get("whitening") for m in metas]):
            raise ValueError("vision: whitening meta differs across fractions")
        ref = next(iter(datas.values()))
        for f, d in datas.items():
            for key in ("whiten_mean", "whiten_matrix"):
                a, b = getattr(ref, key), getattr(d, key)
                if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
                    raise ValueError(f"vision f={f}: whitening parameter '{key}' differs "
                                     "across fractions")
        if (ref.meta.get("whitening") or {}).get("enabled") and ref.whiten_mean is None:
            raise ValueError("vision: whitening enabled in meta but parameters missing")
        fit_split = (ref.meta.get("whitening") or {}).get("fit_split")
        if fit_split != "train":
            raise ValueError(f"vision: whitening.fit_split {fit_split!r}, expected 'train' "
                             "(hard constraint 1: statistics from the train split only)")


def check_cross_branch(vdatas: dict, gdatas: dict, labels=("vision", "graph")) -> None:
    """Hard checks between the two branches (same protocol of queries and damage)."""
    vm = next(iter(vdatas.values())).meta
    gm = next(iter(gdatas.values())).meta
    what = f"{labels[0]} vs {labels[1]}"
    for key in ("split", "query_seed", "partial_seed", "k_values"):
        if json.dumps(vm.get(key)) != json.dumps(gm.get(key)):
            raise ValueError(f"{what}: '{key}' differs: "
                             f"{vm.get(key)!r} vs {gm.get(key)!r}")
    if set(vdatas) != set(gdatas):
        raise ValueError(f"{what}: fractions differ {sorted(vdatas)} vs {sorted(gdatas)}")


def check_query_rows(data: QueryVectorData, shared: list[str], what: str) -> None:
    """Every stored query must sit at its own row of the shared gallery."""
    n = len(shared)
    for name, qi in zip(data.names, data.qi):
        qi = int(qi)
        if not 0 <= qi < n or shared[qi] != str(name):
            got = shared[qi] if 0 <= qi < n else "<out of range>"
            raise ValueError(f"{what}: query {str(name)!r} has qi={qi}, "
                             f"but shared[qi] = {got!r}")


def check_unit_norm(vectors: np.ndarray, what: str, tol: float = NORM_TOL) -> None:
    """Raises if some row does not have unit L2 norm within `tol`."""
    if len(vectors) == 0:
        return
    dev = np.abs(np.linalg.norm(vectors.astype(np.float64), axis=1) - 1.0)
    worst = int(np.argmax(dev))
    if dev[worst] > tol:
        raise ValueError(f"{what}: row {worst} has |norm - 1| = {dev[worst]:.2e} > {tol:g} "
                         "(expected the L2-normalized encoder output)")


@dataclass
class JoinedQueries:
    """Queries present in BOTH branches at one fraction, in vision order."""

    names: list[str]
    qi: np.ndarray          # [P]
    v_rows: np.ndarray      # [P] rows in the vision qvec
    g_rows: np.ndarray      # [P] rows in the graph qvec
    n_only_vision: int      # e.g. graph queries emptied by the damage
    n_only_graph: int


def join_queries(vdata: QueryVectorData, gdata: QueryVectorData, frac: float,
                 labels=("vision", "graph")) -> JoinedQueries:
    """Inner join by name; checks qi and removed rooms query by query.

    Raises:
        ValueError: same name with different qi, or different removed rooms
            (message carries the first offending query and the fraction).
    """
    g_index = {str(n): j for j, n in enumerate(gdata.names)}
    names, qi, v_rows, g_rows = [], [], [], []
    for i, name in enumerate(vdata.names):
        name = str(name)
        j = g_index.get(name)
        if j is None:
            continue
        if int(vdata.qi[i]) != int(gdata.qi[j]):
            raise ValueError(f"f={frac}: query {name!r} has qi {int(vdata.qi[i])} ({labels[0]}) "
                             f"vs {int(gdata.qi[j])} ({labels[1]})")
        rv, rg = sorted(vdata.removed(i)), sorted(gdata.removed(j))
        if rv != rg:
            raise ValueError(f"f={frac}: removed rooms differ for query {name!r}: "
                             f"{labels[0]} {rv} vs {labels[1]} {rg}")
        names.append(name)
        qi.append(int(vdata.qi[i]))
        v_rows.append(i)
        g_rows.append(j)
    v_names = {str(n) for n in vdata.names}
    return JoinedQueries(
        names=names, qi=np.asarray(qi, dtype=np.int64),
        v_rows=np.asarray(v_rows, dtype=np.int64), g_rows=np.asarray(g_rows, dtype=np.int64),
        n_only_vision=len(vdata.names) - len(names),
        n_only_graph=sum(1 for n in gdata.names if str(n) not in v_names),
    )


# ----------------------------------------------------------------------
# Gallery vectors.
# ----------------------------------------------------------------------

class _WhiteningStub(VisionRetrievalPipeline):
    """Pipeline without encoder: only `_apply_whitening` is used, with the
    parameters saved by the vision job (same pattern as tests/test_contracts.py)."""

    def __init__(self, whiten_mean=None, whiten_matrix=None):
        self.device = "cpu"
        self.head = None
        self.whiten_mean = whiten_mean
        self.whiten_matrix = whiten_matrix


def whiten(vectors: np.ndarray, whiten_mean, whiten_matrix) -> np.ndarray:
    """RAW vision vectors [M, D] -> whitened + L2 [M, D'] (vision job code)."""
    return _WhiteningStub(whiten_mean, whiten_matrix)._apply_whitening(vectors)


def load_vision_gallery(embeddings_path, shared: list[str], expected_sha1: str,
                        what: str = "vision") -> np.ndarray:
    """RAW vision gallery restricted/reordered to the shared names.

    Reads `embeddings.npy` and the sibling `image_paths.json`, restricts with
    `gallery_join.restrict_rows` on the stems (as `evaluate.restrict_gallery`).

    Returns:
        [N, D] float32 RAW rows in shared order.

    Raises:
        ValueError: sha1 of the restricted array differs from the qvec meta.
    """
    embeddings_path = Path(embeddings_path)
    raw = np.load(embeddings_path)
    paths = json.loads((embeddings_path.parent / "image_paths.json").read_text())
    rows = restrict_rows([Path(p).stem for p in paths], shared)
    raw = raw[np.asarray(rows, dtype=np.int64)]
    sha1 = array_sha1(raw)
    if sha1 != expected_sha1:
        raise ValueError(f"{what} gallery vectors {embeddings_path}: sha1 {sha1[:12]} differs "
                         f"from the qvec meta {str(expected_sha1)[:12]} (file rewritten?)")
    return raw


def load_graph_gallery(embeddings_path, shared: list[str], expected_sha1: str,
                       what: str = "graph") -> np.ndarray:
    """Graph gallery as written by `graph_evaluate.save_gallery` (already restricted).

    Raises:
        ValueError: `names.json` differs from the shared names, sha1 differs from
            the qvec meta (the graph evaluation rewrites this file at every run),
            or rows are not unit norm.
    """
    embeddings_path = Path(embeddings_path)
    emb = np.load(embeddings_path)
    names = json.loads((embeddings_path.parent / "names.json").read_text())
    if list(names) != list(shared):
        raise ValueError(f"{what} gallery {embeddings_path.parent / 'names.json'}: names differ "
                         "from the shared gallery (order included)")
    sha1 = array_sha1(emb)
    if sha1 != expected_sha1:
        raise ValueError(f"{what} gallery vectors {embeddings_path}: sha1 {sha1[:12]} differs "
                         f"from the qvec meta {str(expected_sha1)[:12]} (file rewritten?)")
    check_unit_norm(emb, f"{what} gallery")
    return emb


def fuse(v: np.ndarray, g: np.ndarray, alpha: float) -> np.ndarray:
    """[sqrt(alpha) * v ; sqrt(1 - alpha) * g], float32 C-contiguous.

    v: [M, Dv] unit rows · g: [M, Dg] unit rows -> [M, Dv + Dg] unit rows.
    `math.sqrt` is exact at 0 and 1, so alpha=1/0 reproduce one branch.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    a, b = math.sqrt(alpha), math.sqrt(1.0 - alpha)
    return np.ascontiguousarray(np.hstack([a * v, b * g]), dtype=np.float32)


# ----------------------------------------------------------------------
# Retrieval + metrics (same loops as graph_evaluate).
# ----------------------------------------------------------------------

def similarity_spread(scores: np.ndarray, ranked: np.ndarray, qi, partial: bool,
                      k: int = SPREAD_K) -> dict:
    """Spread of the top-k similarities of one pass (DESCRIPTIVE diagnostic).

    At alpha=1 / alpha=0 the FAISS scores are exactly sim_V / sim_G, so this
    says how much each branch separates its top candidates: alpha weighs the
    similarities, and a branch with a wider spread dominates the sum whatever
    its dimension. Partial: the top-k of the self-recovery ranking (self kept).
    Full: self removed (its similarity is trivially 1).

    Args:
        scores, ranked: [Q, max_k+1] from `index.search`.

    Returns:
        {k, n, mean_std_topk, mean_gap_rank1_rankk}; std over the k scores of a
        query (ddof=0), then mean over the queries with at least k scores.
    """
    stds, gaps = [], []
    for j, q in enumerate(qi):
        keep = ranked[j] != -1
        if not partial:
            keep &= ranked[j] != int(q)
        top = scores[j][keep][:k].astype(np.float64)
        if len(top) < k:
            continue
        stds.append(float(np.std(top)))
        gaps.append(float(top[0] - top[k - 1]))
    return {"k": k, "n": len(stds),
            "mean_std_topk": float(np.mean(stds)) if stds else float("nan"),
            "mean_gap_rank1_rankk": float(np.mean(gaps)) if gaps else float("nan")}


def score_queries(index, queries: np.ndarray, qi, names, axes, k_values,
                  partial: bool, return_search: bool = False):
    """FAISS search + per-axis metrics + recorder for one pass.

    Partial: `self_rows` keep the self (self-recovery), `axis_rows` drop it
    (graph_evaluate.py evaluate_partial). Full: self dropped (graph_evaluate.py
    evaluate). Metrics always with exclude_self=True.

    Args:
        queries: [Q, D] fused query vectors, row j <-> qi[j], names[j].

    Returns:
        (recorder, MRR of the self-recovery; NaN in full mode), plus the raw
        `(scores, ranked)` of FAISS when `return_search`.
    """
    max_k = max(k_values)
    metrics = new_metrics(k_values)
    skipped = {ax: 0 for ax in DISCRETE_AXES}
    recorder = PerQueryRecorder(k_values, max_k, with_self_rr=partial)
    rr_all = []
    scores, ranked = index.search(np.ascontiguousarray(queries, dtype=np.float32), max_k + 1)
    for j, q in enumerate(qi):
        q = int(q)
        self_rr = None
        if partial:
            row_list = [int(r) for r in ranked[j] if r != -1]
            self_rows = row_list[:max_k]
            axis_rows = [r for r in row_list if r != q][:max_k]
            hit = [i for i, r in enumerate(self_rows, start=1) if r == q]
            self_rr = 1.0 / hit[0] if hit else 0.0
            rr_all.append(self_rr)
        else:
            axis_rows = [int(r) for r in ranked[j] if r != q and r != -1][:max_k]
        skipped_before = dict(skipped)
        accumulate_axes(metrics, skipped, axes, q, axis_rows, k_values, exclude_self=True)
        recorder.add(name=names[j], qi=q, ret_rows=axis_rows, metrics=metrics,
                     skipped_before=skipped_before, skipped=skipped, axes=axes,
                     exclude_self=True, self_rr=self_rr)
    mrr = float(np.mean(rr_all)) if rr_all else float("nan")
    if return_search:
        return recorder, mrr, (scores, ranked)
    return recorder, mrr


def build_axes(shared: list[str]) -> GalleryAxes:
    """Per-axis relevance features aligned with the shared gallery (.mat)."""
    print(f"[late_fusion] per-axis features on {len(shared)} plans...")
    axes = GalleryAxes([load_metadata(n) for n in shared])
    print(f"[late_fusion] .mat metadata present: {int(axes.valid.sum())}/{len(shared)}")
    return axes


# ----------------------------------------------------------------------
# Run.
# ----------------------------------------------------------------------

def _alphas(args) -> list[float]:
    """Alpha grid: explicit list, or {0, alpha*, 1} from a select json."""
    if args.alphas_from:
        star = float(json.loads(Path(args.alphas_from).read_text())["alpha_star"])
        return sorted({0.0, star, 1.0})
    return [float(a) for a in args.alphas]


def run(args) -> None:
    """Loads, checks, fuses and writes every (alpha, fraction) and (alpha, full).

    Side effects: writes the fused per-query files under `args.out`.
    """
    if getattr(args, "pair", "vision-graph") == "graph-graph":
        run_graph_graph(args)
        return
    if getattr(args, "pair", "vision-graph") == "vision-vision":
        run_vision_vision(args)
        return
    if not args.vision_qvec or not args.graph_qvec:
        raise ValueError("--vision-qvec and --graph-qvec are required for the vision-graph pair")
    fractions = [float(f) for f in args.fractions]
    alphas = _alphas(args)
    modes = set(args.modes)
    shared = load_shared_names(args.gallery_names)
    shared_sha1 = gallery_sha1(shared)
    check_output_pair(args.out, "vision-graph")

    # --- qvec files and hard checks (before any heavy work) ---
    vq = load_branch_qvecs(args.vision_qvec, args.split, fractions, VISION_STRATEGY)
    gq = load_branch_qvecs(args.graph_qvec, args.split, fractions, GRAPH_STRATEGY)
    check_branch(vq, "vision", args.split, shared_sha1, len(shared))
    check_branch(gq, "graph", args.split, shared_sha1, len(shared))
    check_cross_branch(vq, gq)
    for f in fractions:
        check_query_rows(vq[f], shared, f"vision f={f}")
        check_query_rows(gq[f], shared, f"graph f={f}")
        check_unit_norm(gq[f].vectors, f"graph queries f={f}")
    joins = {f: join_queries(vq[f], gq[f], f) for f in fractions}

    vref, gref = vq[fractions[0]], gq[fractions[0]]
    k_values = tuple(int(k) for k in vref.meta["k_values"])
    wmean, wmatrix = vref.whiten_mean, vref.whiten_matrix
    vpath = args.vision_embeddings or vref.meta["gallery_vectors"]["path"]
    gpath = args.graph_embeddings or gref.meta["gallery_vectors"]["path"]

    # --- gallery vectors (sha1-pinned) ---
    t0 = time.time()
    vg = whiten(load_vision_gallery(vpath, shared, vref.meta["gallery_vectors"]["sha1"]),
                wmean, wmatrix)
    gg = load_graph_gallery(gpath, shared, gref.meta["gallery_vectors"]["sha1"])
    print(f"[late_fusion] gallery: vision {vg.shape} (whitened) · graph {gg.shape} "
          f"· sha1 {shared_sha1[:12]} · {time.time() - t0:.1f}s")

    # --- query vectors, whitened once per fraction ---
    queries = {}
    for f in fractions:
        j = joins[f]
        v = whiten(vq[f].vectors[j.v_rows], wmean, wmatrix)
        if vq[f].vectors_final is not None:
            diff = float(np.abs(v - vq[f].vectors_final[j.v_rows]).max()) if len(v) else 0.0
            print(f"[late_fusion] f={f}: max |whitened RAW - job final| = {diff:.2e} (diagnostic)")
        queries[f] = (v, gq[f].vectors[j.g_rows])
        print(f"[late_fusion] f={f}: {len(j.names)} paired queries "
              f"(only vision: {j.n_only_vision}, only graph: {j.n_only_graph})")

    axes = build_axes(shared)
    full_join = joins[fractions[0]]

    base_meta = {
        "branch": "fusion",
        "split": args.split,
        "exclude_self": True,
        "query_seed": vref.meta["query_seed"],
        "partial_seed": vref.meta["partial_seed"],
        "gallery": {"n": len(shared), "sha1": shared_sha1, "source": str(args.gallery_names)},
    }

    def fusion_block(alpha, vqvec, gqvec):
        return {
            "alpha": alpha,
            "method": FUSION_METHOD,
            "vision": {"run_tag": vref.meta["run_tag"], "qvec": str(vqvec),
                       "gallery_vectors": {"sha1": vref.meta["gallery_vectors"]["sha1"]},
                       "whitening": vref.meta.get("whitening")},
            "graph": {"run_tag": gref.meta["run_tag"], "qvec": str(gqvec),
                      "gallery_vectors": {"sha1": gref.meta["gallery_vectors"]["sha1"]}},
        }

    # Descriptive spread of the single branches (alpha=1 vision, alpha=0 graph).
    spread = {"note": ALPHA_NOMINAL_NOTE, "k": SPREAD_K,
              "partial_ranking": "self-recovery top-k (self kept)",
              "full_ranking": "top-k with the self removed",
              "vision": {}, "graph": {}}
    spread_branch = {1.0: "vision", 0.0: "graph"}

    for alpha in alphas:
        t_alpha = time.time()
        index = faiss.IndexFlatIP(vg.shape[1] + gg.shape[1])
        index.add(fuse(vg, gg, alpha))       # one fused gallery in memory at a time
        tag = fusion_prefix(args.out, alpha).name

        if "partial" in modes:
            for f in fractions:
                t = time.time()
                j = joins[f]
                v, g = queries[f]
                rec, mrr, (sc, rk) = score_queries(index, fuse(v, g, alpha), j.qi, j.names,
                                                   axes, k_values, partial=True,
                                                   return_search=True)
                if alpha in spread_branch:
                    spread[spread_branch[alpha]][f"f{f}"] = similarity_spread(sc, rk, j.qi, True)
                label = f"{FUSED_STRATEGY} f={f}"
                meta = {**base_meta, "run_tag": tag, "mode": "partial", "partial_label": label,
                        "damage": {"strategy": "nowalls_random", "params": {"fraction": f},
                                   "graph_strategy": "random"},
                        "fusion": fusion_block(
                            alpha, fraction_path(args.vision_qvec, f, args.split, VISION_STRATEGY),
                            fraction_path(args.graph_qvec, f, args.split, GRAPH_STRATEGY)),
                        "join": {"n_only_vision": j.n_only_vision, "n_only_graph": j.n_only_graph}}
                path = fused_partial_path(args.out, alpha, f, args.split)
                rec.write(path, meta=meta)
                print(f"[late_fusion] alpha={alpha:g} [{label}] MRR={mrr:.4f} "
                      f"n={len(rec)} · {time.time() - t:.1f}s -> {path}")

        if "full" in modes:
            t = time.time()
            j = full_join
            fused_gallery_rows = fuse(vg[j.qi], gg[j.qi], alpha)
            rec, _, (sc, rk) = score_queries(index, fused_gallery_rows, j.qi, j.names, axes,
                                             k_values, partial=False, return_search=True)
            if alpha in spread_branch:
                spread[spread_branch[alpha]]["full"] = similarity_spread(sc, rk, j.qi, False)
            meta = {**base_meta, "run_tag": tag, "mode": "full", "partial_label": None,
                    "fusion": fusion_block(alpha, args.vision_qvec, args.graph_qvec),
                    "join": {"n_only_vision": j.n_only_vision, "n_only_graph": j.n_only_graph,
                             "queries_from_fraction": fractions[0]}}
            path = fused_full_path(args.out, alpha, args.split)
            rec.write(path, meta=meta)
            print(f"[late_fusion] alpha={alpha:g} [full] n={len(rec)} "
                  f"· {time.time() - t:.1f}s -> {path}")

        del index
        print(f"[late_fusion] alpha={alpha:g} done in {time.time() - t_alpha:.1f}s\n")

    if spread["vision"] or spread["graph"]:
        path = spread_path(args.out, args.split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spread, indent=2, sort_keys=True))
        for branch in ("vision", "graph"):
            for key, v in spread[branch].items():
                print(f"[late_fusion] spread {branch} [{key}]: std top-{SPREAD_K} "
                      f"{v['mean_std_topk']:.4f} · gap 1-{SPREAD_K} {v['mean_gap_rank1_rankk']:.4f}")
        print(f"[late_fusion] {ALPHA_NOMINAL_NOTE} -> {path}")


def run_graph_graph(args) -> None:
    """Ensemble-effect control of §50: graph + graph (see `_run_same_kind`)."""
    _run_same_kind(args, "graph")


def run_vision_vision(args) -> None:
    """Control of §51 «different models, same information»: vision + vision."""
    _run_same_kind(args, "vision")


# What changes between the two same-kind controls. `first` = weight 1 side.
_SAME_KIND = {
    "graph": {
        "pair": "graph-graph", "labels": ("graph1", "graph2"), "weight": "beta",
        "qvec_args": ("graph_qvec", "graph2_qvec"),
        "emb_args": ("graph_embeddings", "graph2_embeddings"),
        "branch_strategy": GRAPH_STRATEGY, "fused_strategy": GRAPHGRAPH_STRATEGY,
        "damage_strategy": "random", "section": "§50",
        "spread_note": "β è nominale: conta lo spread delle similarità in cima al ranking",
    },
    "vision": {
        "pair": "vision-vision", "labels": ("vision1", "vision2"), "weight": "gamma",
        "qvec_args": ("vision_qvec", "vision2_qvec"),
        "emb_args": ("vision_embeddings", "vision2_embeddings"),
        "branch_strategy": VISION_STRATEGY, "fused_strategy": VISIONVISION_STRATEGY,
        "damage_strategy": "nowalls_random", "section": "§51",
        "spread_note": "γ è nominale: conta lo spread delle similarità in cima al ranking",
    },
}


def _run_same_kind(args, kind: str) -> None:
    """Fusion of two runs of the SAME branch kind, same loops as `run`.

    graph (§50): beta=1 = `--graph-qvec`, beta=0 = `--graph2-qvec`; no whitening,
    gallery and query vectors must be unit norm.
    vision (§51): gamma=1 = `--vision-qvec`, gamma=0 = `--vision2-qvec`; each side
    whitened with the parameters of ITS OWN job (fit on train, head null, checked
    by `check_branch`).

    Both: valid only, same removed rooms query by query, both gallery sha1 pinned,
    refuses the same run twice and folders holding another pair.

    Side effects: writes the fused per-query files and the spread sidecar under `args.out`.
    """
    spec = _SAME_KIND[kind]
    pair, strategy, labels = spec["pair"], spec["fused_strategy"], spec["labels"]
    qarg1, qarg2 = (getattr(args, a, None) for a in spec["qvec_args"])
    if not qarg1 or not qarg2:
        raise ValueError(f"--{spec['qvec_args'][0].replace('_', '-')} and "
                         f"--{spec['qvec_args'][1].replace('_', '-')} are required for the {pair} pair")
    if args.split != "valid":
        raise ValueError(f"{pair} control is valid only (status.md {spec['section']})")
    fractions = [float(f) for f in args.fractions]
    weights = _alphas(args)
    modes = set(args.modes)
    shared = load_shared_names(args.gallery_names)
    shared_sha1 = gallery_sha1(shared)
    check_output_pair(args.out, pair)

    # --- qvec files and hard checks (before any heavy work) ---
    q1 = load_branch_qvecs(qarg1, args.split, fractions, spec["branch_strategy"])
    q2 = load_branch_qvecs(qarg2, args.split, fractions, spec["branch_strategy"])
    check_branch(q1, kind, args.split, shared_sha1, len(shared))
    check_branch(q2, kind, args.split, shared_sha1, len(shared))
    check_cross_branch(q1, q2, labels)
    ref1, ref2 = q1[fractions[0]], q2[fractions[0]]
    if ref1.meta.get("run_tag") == ref2.meta.get("run_tag"):
        raise ValueError(f"{labels[0]} and {labels[1]} are the same run ({ref1.meta.get('run_tag')!r}): "
                         "the control needs two different runs")
    for f in fractions:
        for lab, q in zip(labels, (q1, q2)):
            check_query_rows(q[f], shared, f"{lab} f={f}")
            if kind == "graph":
                check_unit_norm(q[f].vectors, f"{lab} queries f={f}")
    joins = {f: join_queries(q1[f], q2[f], f, labels) for f in fractions}
    k_values = tuple(int(k) for k in ref1.meta["k_values"])

    # --- gallery vectors (sha1-pinned, both) ---
    t0 = time.time()
    emb1, emb2 = (getattr(args, a, None) for a in spec["emb_args"])
    galleries = []
    for lab, ref, emb in zip(labels, (ref1, ref2), (emb1, emb2)):
        path = emb or ref.meta["gallery_vectors"]["path"]
        expected = ref.meta["gallery_vectors"]["sha1"]
        if kind == "graph":
            galleries.append(load_graph_gallery(path, shared, expected, lab))
        else:
            galleries.append(whiten(load_vision_gallery(path, shared, expected, lab),
                                    ref.whiten_mean, ref.whiten_matrix))
    g1, g2 = galleries
    print(f"[late_fusion] {pair} gallery: {labels[0]} {g1.shape} ({ref1.meta['run_tag']}) · "
          f"{labels[1]} {g2.shape} ({ref2.meta['run_tag']}) · sha1 {shared_sha1[:12]} "
          f"· {time.time() - t0:.1f}s")

    # --- query vectors (vision: whitened with the job's own parameters) ---
    queries = {}
    for f in fractions:
        j = joins[f]
        sides = []
        for lab, q, rows in zip(labels, (q1[f], q2[f]), (j.v_rows, j.g_rows)):
            vec = q.vectors[rows]
            if kind == "vision":
                vec = whiten(vec, q.whiten_mean, q.whiten_matrix)
                if q.vectors_final is not None and len(vec):
                    diff = float(np.abs(vec - q.vectors_final[rows]).max())
                    print(f"[late_fusion] {lab} f={f}: max |whitened RAW - job final| = "
                          f"{diff:.2e} (diagnostic)")
            sides.append(vec)
        queries[f] = tuple(sides)
        print(f"[late_fusion] f={f}: {len(j.names)} paired queries "
              f"(only {labels[0]}: {j.n_only_vision}, only {labels[1]}: {j.n_only_graph})")

    axes = build_axes(shared)
    full_join = joins[fractions[0]]
    base_meta = {
        "branch": "fusion",
        "split": args.split,
        "exclude_self": True,
        "query_seed": ref1.meta["query_seed"],
        "partial_seed": ref1.meta["partial_seed"],
        "gallery": {"n": len(shared), "sha1": shared_sha1, "source": str(args.gallery_names)},
    }

    def side_block(ref, qvec):
        block = {"run_tag": ref.meta["run_tag"], "qvec": str(qvec),
                 "gallery_vectors": {"sha1": ref.meta["gallery_vectors"]["sha1"]}}
        if kind == "vision":
            block["whitening"] = ref.meta.get("whitening")
        return block

    def fusion_block(weight, qvec1, qvec2):
        return {"alpha": weight, "pair": pair, "method": FUSION_METHOD,
                labels[0]: side_block(ref1, qvec1), labels[1]: side_block(ref2, qvec2)}

    def join_block(j):
        return {f"n_only_{labels[0]}": j.n_only_vision, f"n_only_{labels[1]}": j.n_only_graph}

    spread = {"note": spec["spread_note"],
              "k": SPREAD_K, "partial_ranking": "self-recovery top-k (self kept)",
              "full_ranking": "top-k with the self removed", labels[0]: {}, labels[1]: {}}
    spread_branch = {1.0: labels[0], 0.0: labels[1]}
    wname = spec["weight"]

    for weight in weights:
        t_w = time.time()
        index = faiss.IndexFlatIP(g1.shape[1] + g2.shape[1])
        index.add(fuse(g1, g2, weight))
        tag = fusion_prefix(args.out, weight).name
        if "partial" in modes:
            for f in fractions:
                t = time.time()
                j = joins[f]
                a, b = queries[f]
                rec, mrr, (sc, rk) = score_queries(index, fuse(a, b, weight), j.qi, j.names,
                                                   axes, k_values, partial=True,
                                                   return_search=True)
                if weight in spread_branch:
                    spread[spread_branch[weight]][f"f{f}"] = similarity_spread(sc, rk, j.qi, True)
                label = f"{strategy} f={f}"
                meta = {**base_meta, "run_tag": tag, "mode": "partial", "partial_label": label,
                        "damage": {"strategy": spec["damage_strategy"], "params": {"fraction": f}},
                        "fusion": fusion_block(
                            weight, fraction_path(qarg1, f, args.split, spec["branch_strategy"]),
                            fraction_path(qarg2, f, args.split, spec["branch_strategy"])),
                        "join": join_block(j)}
                path = fused_partial_path(args.out, weight, f, args.split, strategy)
                rec.write(path, meta=meta)
                print(f"[late_fusion] {wname}={weight:g} [{label}] MRR={mrr:.4f} "
                      f"n={len(rec)} · {time.time() - t:.1f}s -> {path}")
        if "full" in modes:
            t = time.time()
            j = full_join
            rec, _, (sc, rk) = score_queries(index, fuse(g1[j.qi], g2[j.qi], weight), j.qi,
                                             j.names, axes, k_values, partial=False,
                                             return_search=True)
            if weight in spread_branch:
                spread[spread_branch[weight]]["full"] = similarity_spread(sc, rk, j.qi, False)
            meta = {**base_meta, "run_tag": tag, "mode": "full", "partial_label": None,
                    "fusion": fusion_block(weight, qarg1, qarg2),
                    "join": {**join_block(j), "queries_from_fraction": fractions[0]}}
            path = fused_full_path(args.out, weight, args.split)
            rec.write(path, meta=meta)
            print(f"[late_fusion] {wname}={weight:g} [full] n={len(rec)} "
                  f"· {time.time() - t:.1f}s -> {path}")
        del index
        print(f"[late_fusion] {wname}={weight:g} done in {time.time() - t_w:.1f}s\n")

    if spread[labels[0]] or spread[labels[1]]:
        path = spread_path(args.out, args.split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spread, indent=2, sort_keys=True))
        print(f"[late_fusion] spread -> {path}")


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Late fusion vision + graph (status.md §49)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fuse the two branches and write per-query files")
    r.add_argument("--split", required=True, choices=["valid", "test"])
    r.add_argument("--gallery-names", default="results/shared_gallery.json", dest="gallery_names")
    r.add_argument("--pair", choices=list(PAIRS), default="vision-graph",
                   help="vision-graph (default, §49) | graph-graph (ensemble control, §50) | "
                        "vision-vision (different models, same information, §51)")
    r.add_argument("--vision-qvec", default=None, dest="vision_qvec",
                   help="prefix of the vision qvec files (…/vision_<tag>); vision-graph and vision-vision")
    r.add_argument("--graph-qvec", default=None, dest="graph_qvec",
                   help="prefix of the graph qvec files (…/graph_<enc>_<variant>)")
    r.add_argument("--vision-embeddings", default=None, dest="vision_embeddings",
                   help="vision RAW embeddings.npy (default: path in the qvec meta)")
    r.add_argument("--graph-embeddings", default=None, dest="graph_embeddings",
                   help="graph embeddings.npy (default: path in the qvec meta)")
    r.add_argument("--vision2-qvec", default=None, dest="vision2_qvec",
                   help="vision-vision only: prefix of the second vision run (gamma=0)")
    r.add_argument("--vision2-embeddings", default=None, dest="vision2_embeddings",
                   help="vision-vision only: RAW embeddings.npy of vision2 (default: qvec meta)")
    r.add_argument("--graph2-qvec", default=None, dest="graph2_qvec",
                   help="graph-graph only: prefix of the second graph run (beta=0)")
    r.add_argument("--graph2-embeddings", default=None, dest="graph2_embeddings",
                   help="graph-graph only: embeddings.npy of graph2 (default: qvec meta)")
    r.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    grid = r.add_mutually_exclusive_group()
    grid.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    grid.add_argument("--alphas-from", default=None, dest="alphas_from",
                      help="select json: fuse only {0, alpha*, 1} (the test)")
    r.add_argument("--modes", nargs="+", choices=["partial", "full"], default=["partial", "full"])
    r.add_argument("--out", required=True, help="output folder of the fused per-query files")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.cmd == "run":
        run(args)


if __name__ == "__main__":
    main()
