"""Floor of the per-axis metrics: nDCG/Recall/mAP of a system that knows nothing.

On axes with graded gains and large equivalence classes the theoretical zero is not the right reference,
since even a blind ranking scores something. Two nulls:

- `random` (default): random ranking, different for every query.
- `constant` (`--constant-ranking`): the same rows returned to every query, drawn once per seed. If this
  null is close to the models, the metric on that axis does not discriminate.

Design:
- same metric code as the real evaluation (`accumulate_axes` of the graph branch, imported);
- ranking = `max_k` distinct rows (no replacement) drawn from all rows other than the query, including
  plans without `.mat` (gain 0): excluding them would give the blind system information and inflate the floor;
- same queries as the rest of the project (same reproducible sampling), so the floor is paired with the real runs.

The floor depends on the gallery (IDCG): compute it for each gallery.

The per-axis similarity over the whole gallery depends on the query only, not on the ranking: the query
loop is single (seeds inside) and similarities are computed once per query and reused by all seeds.

Usage (the two nulls are run separately):

    python -m src.evaluation.random_floor \
        --gallery embeddings/<...>/image_paths.json --split test \
        --num-queries 2000 --query-seed 42 --ranking-seeds 0 1 2 3 4
    python -m src.evaluation.random_floor ... --constant-ranking
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from src.data.rplan_metadata import load_metadata
from src.evaluation.perquery import PerQueryRecorder, gallery_sha1
from src.evaluation.relevance import AXES, DISCRETE_AXES, GalleryAxes
from src.graph.evaluation.axis_metrics import accumulate_axes, mean_metric, new_metrics

DEFAULT_K_VALUES = (1, 5, 10, 100)
DEFAULT_MAX_K = 100
DEFAULT_OUT_DIR = "results/random_floor"

# null labels (go into `run_tag` and the per-query file name)
NULL_RANDOM = "random"
NULL_CONSTANT = "constant"

# login node kills processes beyond `ulimit -t` CPU seconds: partial tables and the warning limit lost work
LOGIN_NODE_CPU_LIMIT = 600.0
DEFAULT_CPU_WARN_SECONDS = 480.0
DEFAULT_PROGRESS_EVERY = 250


# --- gallery: names, metadata, per-axis features ---

def load_gallery_entries(gallery_path: str | Path) -> list[str]:
    """Gallery rows from `image_paths.json` (vision) or `names.json` (graph): a list of strings in row order."""
    entries = json.loads(Path(gallery_path).read_text())
    if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
        raise ValueError(
            f"{gallery_path}: atteso un JSON con una lista di stringhe "
            f"(image_paths.json o names.json)"
        )
    return entries


def gallery_names(entries: list[str]) -> list[str]:
    """Canonical names = file stems (join key between branches)."""
    return [Path(e).stem for e in entries]


def build_gallery_axes(entries: list[str]) -> GalleryAxes:
    """Per-axis relevance features from the `.mat`, in gallery order."""
    print(f"[random_floor] costruzione feature per-asse su {len(entries)} piante...")
    axes = GalleryAxes([load_metadata(e) for e in entries])
    print(f"[random_floor] metadati .mat presenti: {int(axes.valid.sum())}/{len(entries)}")
    return axes


def sample_query_rows(entries: list[str], num_queries: int, seed: int, split: str) -> list[int]:
    """Query rows with the same sampling as the real evaluation (`src.vision.evaluation.evaluate.sample_query_rows`).

    Local import: that module pulls in torch/faiss.
    """
    from src.vision.evaluation.evaluate import sample_query_rows as _sample

    return _sample(entries, num_queries, seed, split=split)


# --- the two null rankings ---

def draw_rows(rng: random.Random, n_gallery: int, max_k: int) -> list[int]:
    """`max_k + 1` distinct rows drawn without replacement (one extra neighbour, as the real retrieval compensates the self-match)."""
    return rng.sample(range(n_gallery), min(max_k + 1, n_gallery))


def drop_self(rows, qi: int, max_k: int) -> list[int]:
    """Drop the self from the ranking and truncate to `max_k`."""
    return [r for r in rows if r != qi][:max_k]


def random_ret_rows(rng: random.Random, n_gallery: int, qi: int, max_k: int) -> list[int]:
    """`random` null: rows redrawn for every query."""
    return drop_self(draw_rows(rng, n_gallery, max_k), qi, max_k)


def draw_ranking(rng: random.Random, n_gallery: int, valid_rows, max_k: int,
                 constant_ranking: bool) -> list[list[int]]:
    """All `ret_rows` of one seed in query evaluation order.

    Drawn in advance so each seed's RNG is consumed in the same sequence (one `draw_rows` per query;
    a single draw for the constant null).
    """
    if constant_ranking:
        rows = draw_rows(rng, n_gallery, max_k)
        return [drop_self(rows, qi, max_k) for qi in valid_rows]
    return [random_ret_rows(rng, n_gallery, qi, max_k) for qi in valid_rows]


# --- all seeds in a single pass over the queries ---

class QueryAxesCache:
    """`GalleryAxes` proxy caching sim/relevant of the current query (they do not depend on the ranking).

    `accumulate_axes` is untouched; the returned arrays equal those of `GalleryAxes`, so the numbers are bit-identical.
    """

    def __init__(self, axes: GalleryAxes):
        self._axes = axes
        self._qi = None
        self._sim: dict[str, np.ndarray] = {}
        self._relevant: dict[str, np.ndarray] = {}

    def _switch_to(self, qi: int) -> None:
        """Invalidate the cache when the query changes (one query at a time)."""
        if self._qi != qi:
            self._qi = qi
            self._sim.clear()
            self._relevant.clear()

    def __len__(self) -> int:
        return len(self._axes)

    @property
    def valid(self) -> np.ndarray:
        return self._axes.valid

    def sim(self, axis: str, qi: int) -> np.ndarray:
        self._switch_to(qi)
        if axis not in self._sim:
            self._sim[axis] = self._axes.sim(axis, qi)
        return self._sim[axis]

    def relevant(self, axis: str, qi: int) -> np.ndarray:
        """Copy of the mask: the caller writes into it (`rel[qi] = False`)."""
        self._switch_to(qi)
        if axis not in self._relevant:
            self._relevant[axis] = self._axes.relevant(axis, qi)
        return self._relevant[axis].copy()


def run_seeds(axes: GalleryAxes, names, query_rows, k_values, max_k: int, ranking_seeds,
              constant_ranking: bool = False, recorder: PerQueryRecorder | None = None,
              recorder_seed: int | None = None, progress_every: int = 0,
              on_progress=None):
    """Evaluate all seeds in one pass over the queries, with the accumulation of the real evaluation (one container per seed).

    query_rows: candidate rows (those without `.mat` are skipped); `recorder` (optional) receives the
    per-query values of `recorder_seed` only (default: the first); `on_progress(done, total, metrics_per_seed)`
    is called every `progress_every` queries (0 = never).
    Returns (metrics_per_seed, skipped_per_seed, n_evaluated); the first two are dicts seed -> container.
    """
    seeds = list(ranking_seeds)
    if recorder_seed is None:
        recorder_seed = seeds[0]

    # queries without .mat have no ground truth: skipped as in the real evaluation
    valid_rows = [qi for qi in query_rows if axes.valid[qi]]
    rankings = {
        seed: draw_ranking(random.Random(seed), len(axes), valid_rows, max_k,
                           constant_ranking)
        for seed in seeds
    }

    metrics = {seed: new_metrics(k_values) for seed in seeds}
    skipped = {seed: {ax: 0 for ax in DISCRETE_AXES} for seed in seeds}
    cache = QueryAxesCache(axes)

    for j, qi in enumerate(valid_rows):
        for seed in seeds:
            ret_rows = rankings[seed][j]
            skipped_before = dict(skipped[seed])
            accumulate_axes(metrics[seed], skipped[seed], cache, qi, ret_rows,
                            k_values, exclude_self=True)
            if recorder is not None and seed == recorder_seed:
                recorder.add(name=names[qi], qi=qi, ret_rows=ret_rows,
                             metrics=metrics[seed], skipped_before=skipped_before,
                             skipped=skipped[seed], axes=cache, exclude_self=True)
        if progress_every and on_progress and (j + 1) % progress_every == 0:
            on_progress(j + 1, len(valid_rows), metrics)

    return metrics, skipped, len(valid_rows)


# --- aggregation across seeds and printing ---

def seed_means(metrics, k_values) -> dict:
    """Mean over queries of one seed: (axis, metric, k) -> float."""
    return {
        (ax, met, k): mean_metric(metrics[ax][met][k])
        for ax in AXES for met in ("ndcg", "recall", "map") for k in k_values
    }


def print_floor_table(all_means: list[dict], k_values) -> None:
    """Axis x k table: mean across seeds +/- `mc_std`.

    `mc_std` = std of the per-seed means at fixed queries: Monte-Carlo error of the null draw only,
    not metric uncertainty over queries nor a significance threshold.
    """
    n_seeds = len(all_means)
    for k in k_values:
        print(f"================  K = {k}  ================")
        header = (f"{'asse':<13} {'nDCG mean±mc_std':>20} {'Recall mean±mc_std':>20} "
                  f"{'mAP mean±mc_std':>20}")
        print(header)
        print("-" * len(header))
        for ax in AXES:
            cells = []
            for met in ("ndcg", "recall", "map"):
                if ax not in DISCRETE_AXES and met != "ndcg":
                    cells.append(f"{'—':>20}")
                    continue
                values = np.array([m[(ax, met, k)] for m in all_means], dtype=float)
                cells.append(f"{values.mean():>12.4f}±{values.std():.4f}")
            print(f"{ax:<13} " + " ".join(cells))
        print()
    print(f"mc_std = std delle medie fra i {n_seeds} seed di ranking, a query fisse: "
          "errore Monte-Carlo\ndel sorteggio del null, NON incertezza della metrica "
          "e NON un test di significativita'.\n")


# --- entrypoint ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Floor delle metriche per-asse: null casuale o null costante."
    )
    p.add_argument("--gallery", required=True,
                   help="image_paths.json (vision) o names.json (graph): definisce le righe")
    p.add_argument("--split", default="test", choices=["train", "valid", "test", "all"],
                   help="split ufficiale RPLAN da cui pescare le query (la gallery resta intera)")
    p.add_argument("--num-queries", type=int, default=2000, dest="num_queries")
    p.add_argument("--query-seed", type=int, default=42, dest="query_seed",
                   help="seed del campionamento query: deve combaciare con le run vere")
    p.add_argument("--ranking-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   dest="ranking_seeds",
                   help="seed dei sorteggi (la tabella riporta media +/- mc_std fra seed)")
    p.add_argument("--constant-ranking", action="store_true", dest="constant_ranking",
                   help="null costante: le stesse righe per tutte le query "
                        "(un solo sorteggio per seed) — dice se la metrica discrimina")
    p.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES),
                   dest="k_values")
    p.add_argument("--max-k", type=int, default=DEFAULT_MAX_K, dest="max_k",
                   help="profondita' del ranking casuale (>= max dei K)")
    p.add_argument("--out", default=None,
                   help=f"path del .npz per-query del primo seed (default: {DEFAULT_OUT_DIR}/...)")
    p.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY,
                   dest="progress_every",
                   help="ogni quante query stampare la tabella parziale (0 = mai): "
                        "un kill non butta via il lavoro gia' fatto")
    p.add_argument("--cpu-warn-seconds", type=float, default=DEFAULT_CPU_WARN_SECONDS,
                   dest="cpu_warn_seconds",
                   help="soglia di CPU cumulata oltre la quale avvisare che il login "
                        f"node sta per uccidere il processo (default {DEFAULT_CPU_WARN_SECONDS:.0f}s, "
                        f"ulimit -t tipico {LOGIN_NODE_CPU_LIMIT:.0f}s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_k < max(args.k_values):
        raise ValueError(f"--max-k ({args.max_k}) < max(--k-values) ({max(args.k_values)})")

    null_tag = NULL_CONSTANT if args.constant_ranking else NULL_RANDOM
    entries = load_gallery_entries(args.gallery)
    names = gallery_names(entries)
    sha1 = gallery_sha1(names)
    print(f"[random_floor] null '{null_tag}' — gallery: {len(entries)} righe, "
          f"sha1 {sha1[:12]} ({args.gallery})")

    t0 = time.perf_counter()
    axes = build_gallery_axes(entries)
    t_axes = time.perf_counter() - t0

    split = None if args.split == "all" else args.split
    query_rows = sample_query_rows(entries, args.num_queries, args.query_seed, split)

    # per-query saved for the first seed only (an example, not a mean)
    seed0 = args.ranking_seeds[0]
    recorder = PerQueryRecorder(args.k_values, args.max_k)
    warned = False

    def on_progress(done: int, total: int, metrics_per_seed) -> None:
        """Partial table + cumulative CPU (seeds advance together, so an interrupted pass leaves readable numbers)."""
        nonlocal warned
        cpu = time.process_time()
        print(f"[random_floor] parziale: {done}/{total} query — "
              f"CPU cumulata {cpu:.0f}s")
        print_floor_table([seed_means(metrics_per_seed[s], args.k_values)
                           for s in args.ranking_seeds], args.k_values)
        if cpu > args.cpu_warn_seconds and not warned:
            warned = True
            print(f"[random_floor] ATTENZIONE: {cpu:.0f}s di CPU superano la soglia "
                  f"{args.cpu_warn_seconds:.0f}s. Su un login node con "
                  f"ulimit -t {LOGIN_NODE_CPU_LIMIT:.0f} il processo sta per essere "
                  "UCCISO: rilancia su un nodo di calcolo (srun/sbatch) o riduci "
                  "--ranking-seeds / --num-queries.")

    t1 = time.perf_counter()
    metrics, skipped, n_evaluated = run_seeds(
        axes, names, query_rows, args.k_values, args.max_k, args.ranking_seeds,
        constant_ranking=args.constant_ranking, recorder=recorder, recorder_seed=seed0,
        progress_every=args.progress_every, on_progress=on_progress,
    )
    elapsed = time.perf_counter() - t1

    for seed in args.ranking_seeds:
        skip_txt = ", ".join(f"{ax}: {skipped[seed][ax]}" for ax in DISCRETE_AXES)
        print(f"[random_floor] seed {seed}: query escluse da Recall/mAP "
              f"(singleton) {skip_txt}")

    print(f"\n[random_floor] feature per-asse costruite in {t_axes:.1f}s; "
          f"{n_evaluated} query x {len(args.ranking_seeds)} seed valutate in "
          f"{elapsed:.1f}s (CPU totale {time.process_time():.0f}s)")
    print(f"[random_floor] null '{null_tag}' su {len(args.ranking_seeds)} seed, "
          f"query-seed {args.query_seed}\n")
    all_means = [seed_means(metrics[s], args.k_values) for s in args.ranking_seeds]
    print_floor_table(all_means, args.k_values)

    out_path = Path(args.out) if args.out else Path(DEFAULT_OUT_DIR) / (
        f"{null_tag}_floor_{Path(args.gallery).parent.name}_{args.split}_seed{seed0}.npz"
    )
    recorder.write(out_path, meta={
        "branch": f"null-{null_tag}",
        "run_tag": f"{null_tag}_floor/seed{seed0}",
        "mode": "full",
        "partial_label": None,
        "split": args.split,
        "exclude_self": True,
        "query_seed": args.query_seed,
        "gallery": {"n": len(entries), "sha1": sha1, "source": str(args.gallery)},
    })
    print(f"[random_floor] per-query del seed {seed0} salvato in {out_path} "
          f"({len(recorder)} query)")


if __name__ == "__main__":
    main()
