"""Persistence of the per-query values of the retrieval metrics (shared core, contract `perquery/1`).

Paired comparisons, significance tests and per-axis analyses need the value of each query. These are
read from the accumulation containers of the two branches (`metrics[axis][metric][k]`: lists where each
query appends one value) right after the accumulation call: the tail of the list is that query's
contribution, so the printed means stay bit-identical.

`.npz` format (versioned by SCHEMA_VERSION):

    meta         JSON string in a 0-d array (keys below)
    names        <U16 [Q]        query stem: join key between runs (never `qi`: the two galleries differ)
    qi           int32 [Q]       query row in its own gallery
    ndcg         f32 [A,K,Q]     NaN = query not applicable on that axis
    recall, map  f32 [A,K,Q]     NaN on geometry and on singleton queries
    num_relevant int32 [A,Q]     -1 on continuous axes, 0 = query skipped
    ret_rows     int32 [Q,max_k] retrieved rows in rank order, -1 = padding
    n_ret        int16 [Q]       number of valid rows in ret_rows
    self_rr      f32 [Q]         reciprocal rank of the self (mode="partial" only)
    area_removed f32 [Q]         fraction of floor-plan pixels removed by the damage (optional, partial vision)

`meta` keys: schema_version, branch, run_tag, mode ("full"|"partial"), partial_label, split, exclude_self,
num_queries, query_seed, k_values, axes, geometry_weights, gallery{n, sha1, source}, max_k, timestamp, argv.
Optional (additive): damage{strategy, params, patch_size, image_size, area_space: "native"|"resized",
n_fallback} in partial vision. Files without `area_removed`/`damage` stay valid (same SCHEMA_VERSION).

NaN rule: a query skipped on an axis stays in the array (aligned with `names`) with NaN and
`num_relevant = 0`, so `np.nanmean` on the saved matrices reproduces the printed mean.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.evaluation.relevance import AXES, DISCRETE_AXES

SCHEMA_VERSION = "perquery/1"

# fixed weights of `relevance.geometry_sim` (simple mean); stored in meta since runs with different weights are not comparable
GEOMETRY_WEIGHTS = {"area": 1 / 3, "aspect": 1 / 3, "type_area_distribution": 1 / 3}

NAME_DTYPE = "<U16"     # RPLAN stems are short and numeric
PAD_ROW = -1            # ret_rows padding
NO_RELEVANT_SET = -1    # num_relevant on continuous axes (no equivalence class)


# --- gallery identity ---

def gallery_sha1(names) -> str:
    """SHA-1 of the gallery names in row order (not sorted); same hash = comparable row by row.

    sha1 of "\\n".join(names) in UTF-8; do not change without bumping SCHEMA_VERSION.
    """
    joined = "\n".join(str(n) for n in names)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


# --- reading ---

@dataclass
class PerQueryData:
    """Content of a per-query file; matrices are [A, K, Q], `axis_index`/`k_index` map axis names and K values to indices."""

    meta: dict
    names: np.ndarray
    qi: np.ndarray
    ndcg: np.ndarray
    recall: np.ndarray
    map: np.ndarray
    num_relevant: np.ndarray
    ret_rows: np.ndarray | None = None
    n_ret: np.ndarray | None = None
    self_rr: np.ndarray | None = None
    area_removed: np.ndarray | None = None

    def axis_index(self, name: str) -> int:
        """Index of axis `name` (from meta["axes"])."""
        axes = [str(a) for a in self.meta["axes"]]
        if name not in axes:
            raise KeyError(f"asse '{name}' non presente nel file (assi: {axes})")
        return axes.index(name)

    def k_index(self, k: int) -> int:
        """Index of depth `k` (from meta["k_values"])."""
        k_values = [int(v) for v in self.meta["k_values"]]
        if int(k) not in k_values:
            raise KeyError(f"K={k} non presente nel file (K disponibili: {k_values})")
        return k_values.index(int(k))


def load_perquery(path) -> PerQueryData:
    """Read a file written by `write_npz`; optional arrays (`ret_rows`, `n_ret`, `self_rr`, `area_removed`) are None if absent."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"].item()))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema '{meta.get('schema_version')}' incompatibile "
                f"(atteso '{SCHEMA_VERSION}')"
            )
        optional = {
            key: (z[key] if key in z.files else None)
            for key in ("ret_rows", "n_ret", "self_rr", "area_removed")
        }
        return PerQueryData(
            meta=meta,
            names=z["names"],
            qi=z["qi"],
            ndcg=z["ndcg"],
            recall=z["recall"],
            map=z["map"],
            num_relevant=z["num_relevant"],
            **optional,
        )


# --- writing ---

def write_npz(path, *, meta: dict, names, qi, ndcg, recall, map_, num_relevant,
              ret_rows=None, n_ret=None, self_rr=None, area_removed=None) -> None:
    """Write the per-query file (overwrites `path`), validating shapes and dtypes of the contract.

    meta must contain `axes` and `k_values`; `schema_version` is always rewritten, `timestamp`/`argv` added
    if missing. ndcg, recall, map_: [A,K,Q] (NaN where skipped); num_relevant: [A,Q]; ret_rows: [Q,max_k]
    (-1 = padding); n_ret, self_rr, area_removed: [Q]; the last four are optional.
    """
    path = Path(path)
    meta = dict(meta)
    meta["schema_version"] = SCHEMA_VERSION
    meta.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("argv", list(sys.argv))

    for key in ("axes", "k_values"):
        if key not in meta:
            raise ValueError(f"meta['{key}'] mancante: serve a interpretare le matrici")
    n_axes, n_k = len(meta["axes"]), len(meta["k_values"])

    # check before the cast: numpy would truncate silently and break the join by name
    max_len = int(NAME_DTYPE[2:])
    too_long = [n for n in names if len(str(n)) > max_len]
    if too_long:
        raise ValueError(f"nomi troppo lunghi per {NAME_DTYPE} (es. {too_long[0]!r})")
    names = np.asarray(names, dtype=NAME_DTYPE)
    n_query = len(names)

    arrays = {
        "names": names,
        "qi": np.asarray(qi, dtype=np.int32),
        "ndcg": np.asarray(ndcg, dtype=np.float32),
        "recall": np.asarray(recall, dtype=np.float32),
        "map": np.asarray(map_, dtype=np.float32),
        "num_relevant": np.asarray(num_relevant, dtype=np.int32),
    }
    expected = {
        "qi": (n_query,),
        "ndcg": (n_axes, n_k, n_query),
        "recall": (n_axes, n_k, n_query),
        "map": (n_axes, n_k, n_query),
        "num_relevant": (n_axes, n_query),
    }
    if ret_rows is not None:
        arrays["ret_rows"] = np.asarray(ret_rows, dtype=np.int32)
        expected["ret_rows"] = (n_query, arrays["ret_rows"].shape[-1])
    if n_ret is not None:
        arrays["n_ret"] = np.asarray(n_ret, dtype=np.int16)
        expected["n_ret"] = (n_query,)
    if self_rr is not None:
        arrays["self_rr"] = np.asarray(self_rr, dtype=np.float32)
        expected["self_rr"] = (n_query,)
    if area_removed is not None:
        arrays["area_removed"] = np.asarray(area_removed, dtype=np.float32)
        expected["area_removed"] = (n_query,)

    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(
                f"'{key}': forma {arrays[key].shape}, attesa {shape} "
                f"(Q={n_query}, A={n_axes}, K={n_k})"
            )

    meta.setdefault("num_queries", n_query)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)


# --- collection during evaluation ---

def count_relevant(axes, axis: str, qi: int, exclude_self: bool) -> int:
    """Number of relevant items of query `qi` on discrete axis `axis` (exact equivalence class, self excluded if `exclude_self`).

    Recomputed on purpose: the graph accumulation is shared with the checkpoint-selection probe and is not
    touched. One gallery scan per discrete axis, only when per-query saving is on.
    """
    rel = axes.relevant(axis, qi)
    if exclude_self:
        rel[qi] = False
    return int(rel.sum())


class PerQueryRecorder:
    """Per-query buffer filled next to the mean accumulation (re-reads the last appended value, computes no new metric).

    Usage inside the query loop:

        skipped_before = dict(skipped)
        accumulate_axes(metrics, skipped, axes, qi, ret_rows, k_values, exclude_self)
        recorder.add(name=..., qi=qi, ret_rows=ret_rows, metrics=metrics,
                     skipped_before=skipped_before, skipped=skipped,
                     axes=axes, exclude_self=exclude_self)

    k_values: same depths as the accumulation (order = K axis of the saved matrices); max_k: width of
    `ret_rows`; with_self_rr: partial mode; with_area_removed: partial vision.
    """

    def __init__(self, k_values, max_k: int, with_self_rr: bool = False,
                 with_area_removed: bool = False):
        self.k_values = tuple(k_values)
        self.max_k = int(max_k)
        self.with_self_rr = with_self_rr
        self.with_area_removed = with_area_removed
        self._names: list[str] = []
        self._qi: list[int] = []
        self._ndcg: list[np.ndarray] = []      # each [A,K]
        self._recall: list[np.ndarray] = []
        self._map: list[np.ndarray] = []
        self._num_relevant: list[np.ndarray] = []   # each [A]
        self._ret_rows: list[np.ndarray] = []       # each [max_k]
        self._n_ret: list[int] = []
        self._self_rr: list[float] = []
        self._area_removed: list[float] = []

    def add(self, *, name, qi, ret_rows, metrics, skipped_before, skipped,
            axes, exclude_self: bool, self_rr: float | None = None,
            area_removed: float | None = None) -> None:
        """Record the query just accumulated.

        `metrics` is read, never modified; the delta `skipped` - `skipped_before` (copy taken before the
        accumulation) tells on which discrete axes the query was skipped. `self_rr` / `area_removed` are
        required when the recorder was built with `with_self_rr` / `with_area_removed`.
        """
        if self.with_self_rr and self_rr is None:
            raise ValueError("self_rr richiesto: il recorder è in modalità partial")
        if self.with_area_removed and area_removed is None:
            raise ValueError("area_removed richiesto: il recorder registra l'area rimossa")

        n_axes, n_k = len(AXES), len(self.k_values)
        ndcg = np.full((n_axes, n_k), np.nan, dtype=np.float32)
        recall = np.full((n_axes, n_k), np.nan, dtype=np.float32)
        map_ = np.full((n_axes, n_k), np.nan, dtype=np.float32)
        num_relevant = np.full(n_axes, NO_RELEVANT_SET, dtype=np.int32)

        for a, ax in enumerate(AXES):
            for j, k in enumerate(self.k_values):
                ndcg[a, j] = metrics[ax]["ndcg"][k][-1]
            if ax not in DISCRETE_AXES:
                continue
            if skipped[ax] > skipped_before[ax]:
                # singleton class: Recall/mAP undefined, stay NaN
                num_relevant[a] = 0
                continue
            for j, k in enumerate(self.k_values):
                recall[a, j] = metrics[ax]["recall"][k][-1]
                map_[a, j] = metrics[ax]["map"][k][-1]
            num_relevant[a] = count_relevant(axes, ax, qi, exclude_self)

        rows = np.asarray(ret_rows, dtype=np.int32).ravel()
        if len(rows) > self.max_k:
            raise ValueError(f"ret_rows più lunghi di max_k={self.max_k}: {len(rows)}")
        padded = np.full(self.max_k, PAD_ROW, dtype=np.int32)
        padded[: len(rows)] = rows

        self._names.append(str(name))
        self._qi.append(int(qi))
        self._ndcg.append(ndcg)
        self._recall.append(recall)
        self._map.append(map_)
        self._num_relevant.append(num_relevant)
        self._ret_rows.append(padded)
        self._n_ret.append(len(rows))
        if self.with_self_rr:
            self._self_rr.append(float(self_rr))
        if self.with_area_removed:
            self._area_removed.append(float(area_removed))

    def __len__(self) -> int:
        return len(self._names)

    def write(self, path, meta: dict) -> None:
        """Stack the buffers and write the file (overwrites `path`).

        The recorder sets the structural meta keys (`axes`, `k_values`, `max_k`, `geometry_weights`); the
        caller provides the run keys (branch, run_tag, mode, partial_label, split, exclude_self, query_seed, gallery).
        """
        shape = (len(AXES), len(self.k_values))
        write_npz(
            path,
            meta={
                **meta,
                "axes": list(AXES),
                "k_values": list(self.k_values),
                "max_k": self.max_k,
                "geometry_weights": GEOMETRY_WEIGHTS,
            },
            names=self._names,
            qi=np.asarray(self._qi, dtype=np.int32),
            ndcg=_stack(self._ndcg, shape),
            recall=_stack(self._recall, shape),
            map_=_stack(self._map, shape),
            num_relevant=_stack(self._num_relevant, (len(AXES),)),
            ret_rows=(np.stack(self._ret_rows) if self._ret_rows
                      else np.empty((0, self.max_k), dtype=np.int32)),
            n_ret=np.asarray(self._n_ret, dtype=np.int16),
            self_rr=(np.asarray(self._self_rr, dtype=np.float32)
                     if self.with_self_rr else None),
            area_removed=(np.asarray(self._area_removed, dtype=np.float32)
                          if self.with_area_removed else None),
        )


def _stack(rows: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    """Stack per-query contributions on the last axis -> [..., Q]; `shape` = one contribution (for Q=0)."""
    if not rows:
        return np.empty(shape + (0,), dtype=np.float32)
    return np.stack(rows, axis=-1)
