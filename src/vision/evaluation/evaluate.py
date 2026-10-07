import argparse
import random
import re
from pathlib import Path

import numpy as np

from src.data.rplan_metadata import get_split, load_metadata, split_row_indices
from src.evaluation.gallery_join import load_shared_names, restrict_rows
from src.vision.data.vision_damage import damaged_query, make_patch_context, resolve_patch_size
from src.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.perquery import PerQueryRecorder, gallery_sha1
from src.evaluation.query_vectors import QueryVectorRecorder, array_sha1
from src.evaluation.relevance import AXES, DISCRETE_AXES, GalleryAxes
from src.vision.models.projection_head import load_head
from src.vision.models.retrieval_model import VisionRetrievalPipeline
from src.vision.models.vision_model_manager import VisionModelManager
from src.vision.utils.config import load_vision_config, transform_tag

DEFAULT_CONFIG = "configs/vision_retrieval.yaml"
DEFAULT_K_VALUES = (1, 5, 10, 100)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="config YAML del retrieval vision (sceglie encoder + parametri)",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="forza la modalità partial (override di partial.enabled; le "
             "strategie restano quelle del config)",
    )
    # unrecognised arguments are dotlist overrides (e.g. model.variant=gem)
    return parser.parse_known_args()


def restrict_gallery(pipeline, config) -> None:
    """Restrict the gallery in place to the inner join between branches.

    Driven by `eval.gallery_names`: null = whole gallery. With a path the gallery is
    restricted and reordered to the canonical order of the shared file (`gallery_sha1`
    depends on the order, so both branches reach the same hash).
    Call before `prepare_index`: whitening and index must see the metrics' gallery.
    """
    path = config.eval.get("gallery_names")
    if not path:
        return

    shared = load_shared_names(path)
    rows = restrict_rows([Path(p).stem for p in pipeline.image_paths], shared)
    before = len(pipeline.image_paths)
    pipeline.raw_embeddings = pipeline.raw_embeddings[np.asarray(rows, dtype=np.int64)]
    pipeline.image_paths = [pipeline.image_paths[i] for i in rows]
    pipeline.embeddings = pipeline.raw_embeddings
    print(f"[evaluate] gallery ristretta all'inner join (B.3): "
          f"{before} -> {len(pipeline.image_paths)} righe, ordine canonico da {path}")


def whitening_fit_rows(config, image_paths: list[str]) -> list[int] | None:
    """Rows used to fit the whitening, per `whitening.fit_split`.

    - `train` (default): train-split plans only; the gallery stays whole, only the fit set shrinks.
    - `all`: transductive, fit on the whole gallery; kept as paired reference.

    Returns:
        Row indices, or None when nothing is restricted (whitening off, or `all`).
    """
    if not config.whitening.enabled:
        return None

    fit_split = config.whitening.get("fit_split") or "all"
    if fit_split == "all":
        print("[evaluate] whitening TRASDUTTIVO: stima su tutta la gallery "
              "(whitening.fit_split=all)")
        return None

    rows = split_row_indices(image_paths, fit_split)
    print(f"[evaluate] whitening stimato sullo split '{fit_split}': "
          f"{len(rows)}/{len(image_paths)} righe della gallery")
    return rows


def load_pipeline(config) -> VisionRetrievalPipeline:
    """Build the encoder, reload RAW embeddings and prepare the index with the active contributions (optional head + whitening), reusing the raw cache."""
    manager = VisionModelManager(config)
    pipeline = VisionRetrievalPipeline(
        encoder=manager.encoder,
        transform=manager.transform,
        device=manager.device,
    )
    pipeline.load(config.retrieval.save_dir)
    restrict_gallery(pipeline, config)   # before head and whitening

    head = None
    hcfg = getattr(config, "head", None)
    if hcfg is not None and hcfg.get("enabled"):
        in_dim = pipeline.raw_embeddings.shape[1]
        head = load_head(config.retrieval.save_dir, in_dim,
                         hcfg.hidden_dim, hcfg.out_dim, manager.device,
                         filename=hcfg.get("file") or "head.pt")

    pipeline.prepare_index(
        head=head,
        whiten=config.whitening.enabled,
        eps=config.whitening.eps,
        whiten_dim=config.whitening.get("dim"),
        fit_rows=whitening_fit_rows(config, pipeline.image_paths),
    )
    qcfg = config.get("query_head")
    if qcfg is not None and qcfg.get("enabled"):
        attach_query_head(pipeline, config, manager.device)
    return pipeline


def attach_query_head(pipeline, config, device) -> dict:
    """Attach the query-only residual head (head v2) on the whitened frozen vector.

    Whitening is imposed from the checkpoint (the one the head trained on): refits on other nodes
    rotate eigenvectors of near-equal eigenvalues and a head sees coordinates. The gallery is
    re-whitened with it and re-indexed, without head. The difference from the refit is printed.
    """
    from src.vision.models.projection_head import load_query_head
    from src.vision.training.train_head_v2 import whitening_sha1
    if config.head.get("enabled"):
        raise ValueError("query_head (head v2) excludes head.enabled (the gallery must stay frozen)")
    if not config.whitening.enabled or (config.whitening.get("fit_split") or "all") != "train":
        raise ValueError("query_head (head v2) runs on the frozen whiten-train vector: "
                         "whitening.enabled=true whitening.fit_split=train")
    path = Path(config.retrieval.save_dir) / (config.query_head.get("file") or "head_v2.pt")
    head, ckpt = load_query_head(path, device)
    mean, matrix = ckpt["whiten_mean"].cpu().numpy(), ckpt["whiten_matrix"].cpu().numpy()
    if whitening_sha1(mean, matrix) != ckpt["whitening_sha1"]:
        raise ValueError(f"{path}: whitening inside the checkpoint does not match its sha1")
    diff = float(np.abs(pipeline.whiten_matrix - matrix).max()) if pipeline.whiten_matrix is not None else None
    print(f"[evaluate] head v2 {path} (epoca {ckpt['best_epoch']}) | whitening del checkpoint "
          f"{ckpt['whitening_sha1'][:12]} imposto (max |refit - checkpoint| = {diff})")
    pipeline.whiten_mean, pipeline.whiten_matrix = mean.astype("float32"), matrix.astype("float32")
    pipeline.embeddings = pipeline._apply_whitening(pipeline._apply_head(pipeline.raw_embeddings))
    pipeline.build_index()
    pipeline.query_head = head
    return {"file": path.name, "sha1": _file_sha1(path), "whitening_sha1": ckpt["whitening_sha1"],
            "best_epoch": int(ckpt["best_epoch"])}


def _file_sha1(path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_gallery_axes(image_paths: list[str]) -> tuple[GalleryAxes, dict[str, int]]:
    """Align .mat metadata to the FAISS gallery order and build per-axis features.

    Returns:
        (axes, stem2row), stem2row mapping PNG stem -> row index.
    """
    print(f"[evaluate] Costruzione feature per-asse su {len(image_paths)} piante...")
    metas = [load_metadata(p) for p in image_paths]
    axes = GalleryAxes(metas)
    stem2row = {Path(p).stem: i for i, p in enumerate(image_paths)}
    n_valid = int(axes.valid.sum())
    print(f"[evaluate] Metadati .mat presenti: {n_valid}/{len(image_paths)}")
    return axes, stem2row


def sample_query_rows(
    image_paths: list[str], num_queries: int, seed: int, split: str | None = None
) -> list[int]:
    """Reproducibly sample query rows; with `split` (valid|test) queries are restricted to that RPLAN split (searched on the whole gallery)."""
    pool = list(range(len(image_paths)))
    if split:
        pool = [i for i in pool if get_split(image_paths[i]) == split]
        print(f"[evaluate] query ristrette allo split '{split}': {len(pool)} candidate")
    n = min(num_queries, len(pool))
    rng = random.Random(seed)
    rows = rng.sample(pool, n)
    print(f"[evaluate] {n} query campionate (gallery = query pool)\n")
    return rows


def _rows_from_results(results, qi, stem2row, max_k, exclude_self) -> list[int]:
    """Map FAISS results to gallery rows, optionally excluding self; up to max_k rows in rank order."""
    rows = []
    for r in results:
        row = stem2row.get(Path(r["path"]).stem)
        if row is None or (exclude_self and row == qi):
            continue
        rows.append(row)
    return rows[:max_k]


def retrieved_rows_for_query(
    pipeline, query_path, qi, stem2row, max_k
) -> list[int]:
    """Retrieval with the full image as query; self-match excluded."""
    results = pipeline.query(pipeline.load_query(query_path), top_k=max_k + 1)
    return _rows_from_results(results, qi, stem2row, max_k, exclude_self=True)


def partial_rows(results, qi, stem2row, max_k) -> tuple[list[int], list[int]]:
    """From the same FAISS results, the two lists the partial needs.

    The degraded query's original is still in the gallery; the two views treat it oppositely.

    Returns:
        (self_rows, axis_rows)
        - `self_rows`: self kept, for self-recovery (finding the original is the task).
        - `axis_rows`: self removed, for per-axis metrics (comparable with full, where self is excluded).

    Both stay up to `max_k` long since FAISS is asked for `max_k + 1` results.
    """
    self_rows = _rows_from_results(results, qi, stem2row, max_k, exclude_self=False)
    axis_rows = _rows_from_results(results, qi, stem2row, max_k, exclude_self=True)
    return self_rows, axis_rows


# --- per-axis metric accumulation (shared by full and partial) ---

def _new_metrics(k_values):
    return {
        ax: {"ndcg": {k: [] for k in k_values},
             "recall": {k: [] for k in k_values},
             "map": {k: [] for k in k_values}}
        for ax in AXES
    }


def _accumulate_axes(metrics, skipped, axes, qi, ret_rows, k_values, exclude_self):
    """Update `metrics`/`skipped` with one query's per-axis metrics.

    exclude_self=True: self excluded from gallery, relevant set and results (used by both full and
    partial, so the curves are comparable). exclude_self=False: self stays in relevant set and IDCG
    (unused by the vision branch; self-recovery has its own endpoint `self_rr`).
    """
    ret_rows = np.asarray(ret_rows, dtype=int)
    not_self = np.ones(len(axes), dtype=bool)
    if exclude_self:
        not_self[qi] = False

    for ax in AXES:
        sims = axes.sim(ax, qi)
        gallery_gains = sims[not_self]
        retrieved_gains = sims[ret_rows] if len(ret_rows) else np.empty(0)
        for k in k_values:
            metrics[ax]["ndcg"][k].append(ndcg_at_k(retrieved_gains, gallery_gains, k))

        if ax in DISCRETE_AXES:
            rel = axes.relevant(ax, qi)
            if exclude_self:
                rel[qi] = False
            num_rel = int(rel.sum())
            if num_rel == 0:
                skipped[ax] += 1
                continue
            retrieved_rel = rel[ret_rows] if len(ret_rows) else np.empty(0, dtype=bool)
            for k in k_values:
                metrics[ax]["recall"][k].append(
                    recall_at_k(retrieved_rel, num_rel, k)
                )
                metrics[ax]["map"][k].append(
                    average_precision_at_k(retrieved_rel, num_rel, k)
                )


# --- per-query value persistence (optional) ---

def perquery_context(config, image_paths) -> dict | None:
    """Context for saving per-query values, or None if off; driven by `eval.perquery_dir` (one per-query matrix per run)."""
    out_dir = config.eval.get("perquery_dir")
    if not out_dir:
        return None
    names = [Path(p).stem for p in image_paths]
    return {
        "dir": Path(out_dir),
        "tag": f"{config.model.name}_{config.model.variant}_{transform_tag(config)}",
        "split": config.eval.get("split") or "all",
        "seed": int(config.eval.seed),
        "gallery": {
            "n": len(names),
            "sha1": gallery_sha1(names),
            "source": str(config.retrieval.data_dir),
        },
    }


def query_vectors_context(config, perquery, raw_embeddings=None) -> dict | None:
    """Context for saving the damaged query vectors (`qvec/1`), or None if off.

    Driven by `eval.query_vectors_dir` (late fusion); the vectors are tied to the per-query
    file of the same run, so `eval.perquery_dir` is required.

    Args:
        perquery: context from `perquery_context`.
        raw_embeddings: [N, D] RAW gallery after `restrict_gallery`; its sha1 pins the gallery.

    Raises:
        ValueError: `eval.query_vectors_dir` set without `eval.perquery_dir`.
    """
    out_dir = config.eval.get("query_vectors_dir")
    if not out_dir:
        return None
    if perquery is None:
        raise ValueError(
            "eval.query_vectors_dir requires eval.perquery_dir: the query vectors "
            "are paired with the per-query file of the same run"
        )
    hcfg = getattr(config, "head", None)
    head = None
    if hcfg is not None and hcfg.get("enabled"):
        head = {"enabled": True, "file": hcfg.get("file") or "head.pt"}
    ctx = {
        "dir": Path(out_dir),
        "whitening": {
            "enabled": bool(config.whitening.enabled),
            "fit_split": config.whitening.get("fit_split") or "all",
            "eps": float(config.whitening.eps),
            "dim": config.whitening.get("dim"),
        },
        "head": head,
        "gallery_vectors": None,
    }
    qcfg = config.get("query_head")
    if qcfg is not None and qcfg.get("enabled"):      # key absent otherwise
        ctx["query_head"] = {"enabled": True, "file": qcfg.get("file") or "head_v2.pt"}
    if raw_embeddings is not None:
        ctx["gallery_vectors"] = {
            "path": str(Path(config.retrieval.save_dir) / "embeddings.npy"),
            "sha1": array_sha1(raw_embeddings),
            "shape": list(raw_embeddings.shape),
        }
    return ctx


def _label_slug(label: str) -> str:
    """Run label to file-name fragment ('random f=0.5' -> 'random-f0.5')."""
    return re.sub(r"[^0-9A-Za-z.]+", "-", label.replace("=", "")).strip("-")


def _write_perquery(recorder, ctx, partial_label: str | None = None,
                    damage: dict | None = None) -> Path:
    """Write the per-query .npz of one run and return its path.

    File name `vision_<tag>_<mode>_<split>.npz`, `tag` = encoder_variant_contribution; in partial
    the mode includes the strategy, else masking-curve runs would overwrite each other.
    `damage` (partial only) goes to `meta["damage"]`.
    """
    mode = "full" if partial_label is None else "partial"
    slug = mode if partial_label is None else f"partial-{_label_slug(partial_label)}"
    path = ctx["dir"] / f"vision_{ctx['tag']}_{slug}_{ctx['split']}.npz"
    meta = {
        "branch": "vision",
        "run_tag": ctx["tag"],
        "mode": mode,
        "partial_label": partial_label,
        "split": ctx["split"],
        # per-axis metrics exclude self in partial too; `significance.py` uses this key to
        # reject comparison with older files (where it was False)
        "exclude_self": True,
        "query_seed": ctx["seed"],
        "gallery": ctx["gallery"],
    }
    if damage is not None:
        meta["damage"] = damage
    recorder.write(path, meta=meta)
    print(f"[evaluate] valori per-query salvati in {path}")
    return path


def _write_query_vectors(qrec, qctx, perquery, perquery_path, pipeline, label, strat,
                         params, partial_seed, k_values) -> None:
    """Write the `qvec/1` file of one partial run, same name as its per-query file, in `eval.query_vectors_dir`.

    Whitening parameters are the pipeline's own (None when off), so fusion whitens with exactly them.
    """
    path = qctx["dir"] / Path(perquery_path).name
    meta = {
        "branch": "vision",
        "run_tag": perquery["tag"],
        "perquery_file": str(perquery_path),
        "mode": "partial",
        "partial_label": label,
        "damage": {"strategy": strat, "params": dict(params)},
        "split": perquery["split"],
        "query_seed": perquery["seed"],
        "partial_seed": int(partial_seed),
        "k_values": list(k_values),
        "gallery": perquery["gallery"],
        "gallery_vectors": qctx["gallery_vectors"],
        "whitening": qctx["whitening"],
        "head": qctx["head"],
    }
    if "query_head" in qctx:
        meta["query_head"] = qctx["query_head"]
    qrec.write(path, meta, whiten_mean=pipeline.whiten_mean,
               whiten_matrix=pipeline.whiten_matrix)
    print(f"[evaluate] vettori delle query salvati in {path}")


# --- full evaluation (query = whole image) ---

def evaluate(pipeline, axes, stem2row, query_rows, image_paths, k_values,
             perquery=None) -> None:
    max_k = max(k_values)
    metrics = _new_metrics(k_values)
    skipped = {ax: 0 for ax in DISCRETE_AXES}
    n_query_valid = 0
    recorder = PerQueryRecorder(k_values, max_k) if perquery else None

    for qi in query_rows:
        if not axes.valid[qi]:
            continue
        n_query_valid += 1
        ret_rows = retrieved_rows_for_query(
            pipeline, image_paths[qi], qi, stem2row, max_k
        )
        skipped_before = dict(skipped)   # delta = axes skipped by this query
        _accumulate_axes(metrics, skipped, axes, qi, ret_rows, k_values, exclude_self=True)
        if recorder is not None:
            recorder.add(
                name=Path(image_paths[qi]).stem, qi=qi, ret_rows=ret_rows,
                metrics=metrics, skipped_before=skipped_before, skipped=skipped,
                axes=axes, exclude_self=True,
            )

    print(f"[evaluate] query valutate: {n_query_valid}")
    for ax in DISCRETE_AXES:
        if skipped[ax]:
            print(
                f"[evaluate] {ax}: {skipped[ax]} query escluse da Recall/mAP "
                "(classe di equivalenza singleton, nessun rilevante)"
            )
    print()
    _print_axis_tables(metrics, k_values)
    if recorder is not None:
        _write_perquery(recorder, perquery)


# --- partial evaluation (query = degraded plan) ---

def partial_runs(pcfg) -> list[tuple[str, str, dict]]:
    """Expand the config into (label, strategy, params) runs; `random` yields one run per fraction.

    Crop and patch (`vision_damage.py`) follow topology, only if enabled, so with their keys
    absent the list and order are unchanged (contract with the graph branch, test_contracts).
    `nowalls_*` come last: same room selection as the historical twin, walls erased too.
    Distinct labels give distinct per-query files."""
    runs = []
    s = pcfg.strategies
    if s.get("random", {}).get("enabled", False):
        for f in s.random.fractions:
            runs.append((f"random f={f}", "random", {"fraction": float(f)}))
    if s.get("semantic", {}).get("enabled", False):
        runs.append(("semantic", "semantic", {"keep_types": list(s.semantic.keep_types)}))
    if s.get("topology", {}).get("enabled", False):
        runs.append(("topology", "topology", {"max_degree": int(s.topology.get("max_degree", 1))}))
    crop = s.get("crop", {})
    if crop.get("enabled", False):
        for f in crop.fractions:
            runs.append((f"crop f={f}", "crop", {
                "fraction": float(f),
                "tolerance": float(crop.get("tolerance", 0.05)),
                "max_tries": int(crop.get("max_tries", 10000)),
            }))
    patch = s.get("patch", {})
    if patch.get("enabled", False):
        for f in patch.fractions:
            runs.append((f"patch f={f}", "patch", {
                "fraction": float(f),
                "tolerance": float(patch.get("tolerance", 0.05)),
            }))
    nw = s.get("nowalls_random", {})
    if nw.get("enabled", False):
        for f in nw.fractions:
            runs.append((f"nowalls-random f={f}", "nowalls_random", {"fraction": float(f)}))
    nw = s.get("nowalls_semantic", {})
    if nw.get("enabled", False):
        runs.append(("nowalls-semantic", "nowalls_semantic",
                     {"keep_types": list(nw.keep_types)}))
    nw = s.get("nowalls_topology", {})
    if nw.get("enabled", False):
        runs.append(("nowalls-topology", "nowalls_topology",
                     {"max_degree": int(nw.get("max_degree", 1))}))
    return runs


def _transform_image_size(transform) -> int | None:
    """Canvas side S of the transform (leading ResizeWithPad), None if unknown."""
    steps = getattr(transform, "transforms", None) or []
    return int(steps[0].target_size) if steps and hasattr(steps[0], "target_size") else None


def evaluate_partial(pipeline, axes, stem2row, query_rows, image_paths, k_values, pcfg,
                     perquery=None, patch_size=None, query_vectors=None) -> None:
    """For each strategy/level: partial query, retrieval, metrics.

    Two views: self-recovery (is the original found?) and per-axis against the whole plan.
    Queries come from `vision_damage.damaged_query`, which also measures `area_removed`.
    `patch_size` is needed only by `patch` runs (resolved by `main`).

    `query_vectors` (from `query_vectors_context`): also save the damaged-query vectors from the
    same forward; only runs reporting `removed` rooms (room and nowalls) are saved.
    """
    if query_vectors is not None and perquery is None:
        raise ValueError("query_vectors requires perquery (same run, same file name)")
    max_k = max(k_values)
    seed = int(pcfg.get("seed", 42))
    open_boundary = bool(pcfg.get("open_boundary", True))
    runs = partial_runs(pcfg)
    patch_ctx = None
    if any(strat == "patch" for _, strat, _ in runs):
        if patch_size is None:
            raise ValueError("run `patch` richieste ma patch_size non risolto")
        patch_ctx = make_patch_context(pipeline.transform, patch_size)

    print(f"[evaluate] PARTIAL | {len(runs)} run | open_boundary={open_boundary}\n")

    for label, strat, params in runs:
        metrics = _new_metrics(k_values)
        skipped = {ax: 0 for ax in DISCRETE_AXES}
        selfrec = {"recall": {k: [] for k in k_values}, "rr": []}
        recorder = (PerQueryRecorder(k_values, max_k, with_self_rr=True, with_area_removed=True)
                    if perquery else None)
        n_valid = 0
        n_empty = 0  # queries with no room removed (degenerate to the full query)
        areas = []   # removed plan fraction, one per evaluated query
        n_fallback = 0
        qrec = QueryVectorRecorder() if query_vectors is not None else None

        for qi in query_rows:
            if not axes.valid[qi]:
                continue
            meta = load_metadata(image_paths[qi])
            if meta is None:
                continue
            n_valid += 1

            rng = random.Random(seed + qi)
            query_tensor, dinfo = damaged_query(
                image_paths[qi], meta, strat, params, rng, open_boundary,
                pipeline.transform, patch_ctx,
            )
            if not dinfo["removed_any"]:
                n_empty += 1
            areas.append(dinfo["area_removed"])
            n_fallback += int(dinfo["fallback"])

            if qrec is not None and "removed" in dinfo:
                results, query_raw, query_final = pipeline.query(
                    query_tensor, top_k=max_k + 1, return_embeddings=True)
                qrec.add(name=Path(image_paths[qi]).stem, qi=qi, vector=query_raw[0],
                         removed=dinfo["removed"], vector_final=query_final[0])
            else:
                results = pipeline.query(query_tensor, top_k=max_k + 1)
            # self stays only in the self-recovery view; per-axis metrics exclude it, as in full
            self_rows, axis_rows = partial_rows(results, qi, stem2row, max_k)

            skipped_before = dict(skipped)   # delta = axes skipped by this query
            _accumulate_axes(metrics, skipped, axes, qi, axis_rows, k_values, exclude_self=True)

            ret_arr = np.asarray(self_rows, dtype=int)
            hit = np.where(ret_arr == qi)[0]
            rank = int(hit[0]) + 1 if len(hit) else None
            selfrec["rr"].append(1.0 / rank if rank else 0.0)
            for k in k_values:
                selfrec["recall"][k].append(1.0 if (rank and rank <= k) else 0.0)

            if recorder is not None:
                # rows and flags as the per-axis accumulation (self excluded); self-recovery goes in `self_rr`
                recorder.add(
                    name=Path(image_paths[qi]).stem, qi=qi, ret_rows=axis_rows,
                    metrics=metrics, skipped_before=skipped_before, skipped=skipped,
                    axes=axes, exclude_self=True, self_rr=selfrec["rr"][-1],
                    area_removed=dinfo["area_removed"],
                )

        _print_partial_report(label, metrics, skipped, selfrec, n_valid, n_empty, k_values,
                              areas=areas, n_fallback=n_fallback)
        if recorder is not None:
            damage = {
                "strategy": strat,
                "params": dict(params),
                "patch_size": patch_ctx["patch_size"] if strat == "patch" else None,
                "image_size": _transform_image_size(pipeline.transform),
                "area_space": "resized" if strat == "patch" else "native",
                "n_fallback": n_fallback,
            }
            perquery_path = _write_perquery(recorder, perquery, partial_label=label,
                                            damage=damage)
            if qrec is not None and len(qrec):
                _write_query_vectors(qrec, query_vectors, perquery, perquery_path, pipeline,
                                     label, strat, params, seed, k_values)
            elif qrec is not None:
                print(f"[evaluate] vettori delle query NON salvati per [{label}]: "
                      "il danno non toglie stanze (niente `removed`)")


# --- report ---

def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _coverage(metrics, ax, k) -> float:
    """Fraction of queries with Recall/mAP defined on this axis (queries with a relevant item / all queries with nDCG)."""
    n_all = len(metrics[ax]["ndcg"][k])
    return len(metrics[ax]["recall"][k]) / n_all if n_all else float("nan")


def _print_axis_tables(metrics, k_values) -> None:
    """Per-axis tables, one per depth k.

    - `Rec/Prec`: `recall_at_k` normalises by `min(k, #relevant)`, so with at least k relevant
      items it is Precision@k (the usual composition regime; see `src/evaluation/metrics.py`).
    - `cop.`: singleton-class queries have no relevant item and leave the mean; below 100% the mean
      covers a ground-truth-chosen subset and is optimistic in absolute value (system comparison
      stays paired).

    Exact breakdown: `python -m src.evaluation.metric_diagnostics` on the per-query files.
    """
    for k in k_values:
        print(f"================  K = {k}  ================")
        header = f"{'asse':<13} {'nDCG':>8} {'Rec/Prec':>9} {'mAP':>8} {'cop.':>7}"
        print(header)
        print("-" * len(header))
        for ax in AXES:
            ndcg = _mean(metrics[ax]["ndcg"][k])
            if ax in DISCRETE_AXES:
                rec = _mean(metrics[ax]["recall"][k])
                mapk = _mean(metrics[ax]["map"][k])
                print(f"{ax:<13} {ndcg:>8.3f} {rec:>9.3f} {mapk:>8.3f} "
                      f"{100*_coverage(metrics, ax, k):>6.1f}%")
            else:
                print(f"{ax:<13} {ndcg:>8.3f} {'—':>9} {'—':>8} {'—':>7}")
        print("Rec/Prec: con #rilevanti >= k è Precision@k · cop.: query su cui "
              "Recall/mAP sono definiti")
        print()


def _print_partial_report(label, metrics, skipped, selfrec, n_valid, n_empty, k_values,
                          areas=None, n_fallback=0) -> None:
    """Report of one partial run: self-recovery + per-axis tables.

    Per-axis tables are comparable with full (self excluded from results, relevant set and IDCG);
    finding the original is measured in self-recovery (MRR/Recall).
    Older per-query files have self inside (`exclude_self=False` in meta) and must not be mixed;
    `significance.py` rejects the comparison.
    """
    print(f"################  PARTIAL [{label}]  ################")
    print(f"query valutate: {n_valid} (senza masking: {n_empty})")
    if areas:
        print(f"area di pianta rimossa: media={np.mean(areas):.3f} "
              f"min={np.min(areas):.3f} max={np.max(areas):.3f} · fallback={n_fallback}")
    for ax in DISCRETE_AXES:
        if skipped[ax]:
            print(f"  {ax}: {skipped[ax]} query escluse da Recall/mAP (classe singleton)")
    mrr = _mean(selfrec["rr"])
    rec_line = "  ".join(f"R@{k}={_mean(selfrec['recall'][k]):.3f}" for k in k_values)
    print(f"self-recovery (pianta originale): MRR={mrr:.3f}  {rec_line}")
    print()
    _print_axis_tables(metrics, k_values)
    print("ℹ️  partial (B.4): self ESCLUSO da risultati/rilevanti/IDCG -> le "
          "tabelle per-asse qui sopra sono")
    print("    confrontabili con quelle della modalità full; il self si misura "
          "nel self-recovery."
          + (" Questa run non rimuove nulla: la query è la pianta intatta."
             if n_valid and n_empty == n_valid else ""))
    print()


def main():
    args, overrides = parse_args()
    config = load_vision_config(args.config, overrides)
    k_values = tuple(getattr(config.eval, "k_values", DEFAULT_K_VALUES))

    pcfg = getattr(config, "partial", None)
    partial_on = args.partial or (pcfg is not None and bool(pcfg.get("enabled", False)))
    if partial_on and pcfg is None:
        raise SystemExit("--partial richiesto ma manca il blocco `partial` nel config")
    mode = "PARTIAL" if partial_on else "FULL"

    print(
        f"[evaluate] encoder = {config.model.name} | variante = {config.model.variant} | "
        f"contributo = {transform_tag(config)} | modalità = {mode} | "
        "rilevanza per-asse dai .mat (composizione/topologia/geometria).\n"
    )

    pipeline = load_pipeline(config)
    image_paths = pipeline.image_paths
    axes, stem2row = build_gallery_axes(image_paths)
    query_rows = sample_query_rows(
        image_paths,
        config.eval.num_queries,
        config.eval.seed,
        split=getattr(config.eval, "split", None),
    )

    perquery = perquery_context(config, image_paths)
    query_vectors = query_vectors_context(config, perquery, pipeline.raw_embeddings)
    if query_vectors is not None and not partial_on:
        print("[evaluate] eval.query_vectors_dir ignorato in modalità full "
              "(i vettori della gallery sono già su disco)")

    if partial_on:
        patch_size = None
        if any(strat == "patch" for _, strat, _ in partial_runs(pcfg)):
            # before the first query: a wrong preset fails fast
            patch_size = resolve_patch_size(
                pipeline.encoder, config.model.kwargs.get("patch_size"),
                _transform_image_size(pipeline.transform),
            )
            print(f"[evaluate] danno patch: patch_size={patch_size}\n")
        evaluate_partial(pipeline, axes, stem2row, query_rows, image_paths, k_values, pcfg,
                         perquery=perquery, patch_size=patch_size,
                         query_vectors=query_vectors)
    else:
        evaluate(pipeline, axes, stem2row, query_rows, image_paths, k_values,
                 perquery=perquery)


if __name__ == "__main__":
    main()
