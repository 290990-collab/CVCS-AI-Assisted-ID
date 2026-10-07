"""Persistence of the damaged query vectors of a partial evaluation (shared core, contract `qvec/1`).

Twin of `perquery.py`. The late fusion (`src/evaluation/late_fusion.py`) needs the damaged query vectors,
which only exist inside the forward of a partial evaluation: they are saved from the same forward that
writes the per-query file, with the rooms removed from each query (so the fusion can check that both
branches damaged the same rooms).

File format (`.npz`, versioned by SCHEMA_VERSION):

    meta          JSON string in a 0-d array (keys below)
    names         <U16 [Q]      query stem: join key between branches
    qi            int32 [Q]     query row in the shared gallery
    vectors       f32 [Q, D]    vision: RAW encoder output (before head/whitening)
                                graph:  the vector searched in FAISS (L2-normalized)
    removed_ptr   int32 [Q+1]   CSR pointers: rooms of query i are
    removed_idx   int32 [R]     removed_idx[removed_ptr[i]:removed_ptr[i+1]]
                                (both may describe empty lists, e.g. at f=0.0)
  vision only (optional):
    vectors_final f32 [Q, D']   the vector the vision job searched (diagnostic)
    whiten_mean   f32 [1, D]    whitening parameters of the job (fit on train)
    whiten_matrix f32 [D, D']

Meta keys: schema_version, branch, run_tag, perquery_file, mode ("partial"),
partial_label, damage{strategy, params}, split, query_seed, partial_seed,
k_values, gallery{n, sha1, source}, gallery_vectors{path, sha1, shape},
timestamp, argv, num_queries; vision: whitening{enabled, fit_split, eps, dim},
head (null for the fusion); graph: model{encoder, variant, raw_skip,
lost_marker, normalize, drop_self_loops}, n_degenerate.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.evaluation.perquery import NAME_DTYPE

SCHEMA_VERSION = "qvec/1"


def array_sha1(arr) -> str:
    """SHA-1 of dtype, shape and C-order bytes; pins the gallery vectors a qvec file was computed against."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha1()
    h.update(f"{a.dtype.str}|{tuple(a.shape)}|".encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()


def _check_names(names) -> np.ndarray:
    """Names -> NAME_DTYPE array, refusing names numpy would truncate (as `perquery.write_npz`)."""
    max_len = int(NAME_DTYPE[2:])
    too_long = [n for n in names if len(str(n)) > max_len]
    if too_long:
        raise ValueError(f"names too long for {NAME_DTYPE} (e.g. {too_long[0]!r})")
    return np.asarray([str(n) for n in names], dtype=NAME_DTYPE)


# --- reading ---

@dataclass
class QueryVectorData:
    """Content of a qvec file."""

    meta: dict
    names: np.ndarray            # [Q]
    qi: np.ndarray               # [Q]
    vectors: np.ndarray          # [Q, D]
    removed_ptr: np.ndarray      # [Q+1]
    removed_idx: np.ndarray      # [R]
    vectors_final: np.ndarray | None = None   # [Q, D'] vision only
    whiten_mean: np.ndarray | None = None     # [1, D]  vision only
    whiten_matrix: np.ndarray | None = None   # [D, D'] vision only

    def removed(self, i: int) -> list[int]:
        """Removed room indices of the i-th query, in the stored order."""
        lo, hi = int(self.removed_ptr[i]), int(self.removed_ptr[i + 1])
        return [int(r) for r in self.removed_idx[lo:hi]]


def load_qvec(path) -> QueryVectorData:
    """Read a file written by `write_qvec`; ValueError if the schema is not `qvec/1`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"].item()))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema '{meta.get('schema_version')}' incompatible "
                f"(expected '{SCHEMA_VERSION}')"
            )
        optional = {
            key: (z[key] if key in z.files else None)
            for key in ("vectors_final", "whiten_mean", "whiten_matrix")
        }
        return QueryVectorData(
            meta=meta, names=z["names"], qi=z["qi"], vectors=z["vectors"],
            removed_ptr=z["removed_ptr"], removed_idx=z["removed_idx"], **optional,
        )


# --- writing ---

def write_qvec(path, *, meta: dict, names, qi, vectors, removed,
               vectors_final=None, whiten_mean=None, whiten_matrix=None) -> None:
    """Write a qvec file (overwrites `path`), validating shapes and dtypes of the contract.

    `schema_version`, `num_queries` are set here, `timestamp`/`argv` if missing. names [Q], qi [Q],
    vectors [Q, D], removed = Q lists of room indices; optional vision-only: vectors_final [Q, D'],
    whiten_mean [1, D] and whiten_matrix [D, D'] (both or neither).
    """
    path = Path(path)
    names = _check_names(names)
    n_query = len(names)

    removed = [list(r) for r in removed]
    ptr = np.zeros(n_query + 1, dtype=np.int32)
    if len(removed) != n_query:
        raise ValueError(f"'removed': {len(removed)} lists, expected {n_query}")
    ptr[1:] = np.cumsum([len(r) for r in removed], dtype=np.int64)
    flat = [int(x) for r in removed for x in r]

    arrays = {
        "names": names,
        "qi": np.asarray(qi, dtype=np.int32),
        "vectors": np.asarray(vectors, dtype=np.float32),
        "removed_ptr": ptr,
        "removed_idx": np.asarray(flat, dtype=np.int32),
    }
    if arrays["qi"].shape != (n_query,):
        raise ValueError(f"'qi': shape {arrays['qi'].shape}, expected ({n_query},)")
    if arrays["vectors"].ndim != 2 or arrays["vectors"].shape[0] != n_query:
        raise ValueError(f"'vectors': shape {arrays['vectors'].shape}, expected ({n_query}, D)")

    if vectors_final is not None:
        arrays["vectors_final"] = np.asarray(vectors_final, dtype=np.float32)
        if arrays["vectors_final"].ndim != 2 or arrays["vectors_final"].shape[0] != n_query:
            raise ValueError(f"'vectors_final': shape {arrays['vectors_final'].shape}, "
                             f"expected ({n_query}, D')")
    if (whiten_mean is None) != (whiten_matrix is None):
        raise ValueError("whiten_mean and whiten_matrix go together (both or neither)")
    if whiten_mean is not None:
        dim = arrays["vectors"].shape[1]
        arrays["whiten_mean"] = np.asarray(whiten_mean, dtype=np.float32)
        arrays["whiten_matrix"] = np.asarray(whiten_matrix, dtype=np.float32)
        if meta.get("head"):
            # with a head the whitening is fit after it: input = head output, not the RAW vector
            dim = arrays["whiten_mean"].shape[-1]
        if arrays["whiten_mean"].shape != (1, dim):
            raise ValueError(f"'whiten_mean': shape {arrays['whiten_mean'].shape}, "
                             f"expected (1, {dim})")
        if arrays["whiten_matrix"].ndim != 2 or arrays["whiten_matrix"].shape[0] != dim:
            raise ValueError(f"'whiten_matrix': shape {arrays['whiten_matrix'].shape}, "
                             f"expected ({dim}, D')")

    meta = dict(meta)
    meta["schema_version"] = SCHEMA_VERSION
    meta["num_queries"] = n_query
    meta.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("argv", list(sys.argv))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)


class QueryVectorRecorder:
    """Buffer of query vectors, filled next to the per-query recorder.

    Usage inside the query loop of a partial evaluation:

        recorder.add(name=..., qi=qi, vector=query_raw[0], removed=dinfo["removed"])
        ...
        recorder.write(path, meta, whiten_mean=..., whiten_matrix=...)
    """

    def __init__(self):
        self._names: list[str] = []
        self._qi: list[int] = []
        self._vectors: list[np.ndarray] = []     # each [D]
        self._final: list[np.ndarray] = []       # each [D'] (optional)
        self._removed: list[list[int]] = []

    def add(self, *, name, qi, vector, removed, vector_final=None) -> None:
        """Record one query; `vector_final` for all queries or none."""
        if self._names and (vector_final is None) != (not self._final):
            raise ValueError("vector_final must be given for every query or for none")
        self._names.append(str(name))
        self._qi.append(int(qi))
        self._vectors.append(np.asarray(vector, dtype=np.float32).ravel())
        if vector_final is not None:
            self._final.append(np.asarray(vector_final, dtype=np.float32).ravel())
        self._removed.append([int(r) for r in removed])

    def __len__(self) -> int:
        return len(self._names)

    def write(self, path, meta: dict, whiten_mean=None, whiten_matrix=None) -> None:
        """Stack the buffers and write the file (see `write_qvec`)."""
        if not self._names:
            raise ValueError("no query recorded: nothing to write")
        write_qvec(
            path, meta=meta, names=self._names, qi=self._qi,
            vectors=np.stack(self._vectors), removed=self._removed,
            vectors_final=np.stack(self._final) if self._final else None,
            whiten_mean=whiten_mean, whiten_matrix=whiten_matrix,
        )
