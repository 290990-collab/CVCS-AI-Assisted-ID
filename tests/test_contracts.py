
"""
CPU smoke test of the critical contracts that would otherwise invalidate numbers silently.

  1. row `i` of `embeddings.npy` <-> element `i` of `image_paths.json` (vision) / `names.json` (graph); late-fusion interface.
  2. architecture shape <-> checkpoint: a flag changing `proj` (`raw_skip`, `pooling="mean_max"`) must make the load fail explicitly.
  3. hand-computed reference values of `ndcg_at_k`, `recall_at_k`, `average_precision_at_k`.
  4. disjoint splits (valid/test queries never overlap) and statistics depending only on the fitted rows;
     4-bis: `prepare_index` fits whitening on train rows only but indexes the whole gallery.

All CPU, deterministic, no real dataset (.mat) or downloaded weights.

Usage: python -m pytest tests/test_contracts.py -v
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omegaconf import OmegaConf

from src.data.rplan_metadata import RoomMeta, split_row_indices
from src.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.graph.evaluation.graph_evaluate import (
    _label_slug as graph_label_slug,
    evaluate_partial as graph_evaluate_partial,
    partial_runs as graph_partial_runs,
    sample_query_rows as graph_sample_query_rows,
    save_gallery,
)
from src.graph.graph_builder import build_graph
from src.graph.graph_dataset import RplanGraphDataset
from src.graph.graph_partial_query import filter_meta, make_partial_graph
from src.graph.models import build_graph_encoder
# canonical synthetic gallery lives in test_perquery: reuse avoids two truths on the same RoomMeta
from tests.test_perquery import synthetic_gallery
from src.evaluation.gallery_join import (
    canonical_names,
    compute_shared_names,
    load_shared_names,
    read_gallery_names,
    restrict_rows,
    write_shared_gallery,
)
from src.evaluation.perquery import count_relevant, gallery_sha1, load_perquery
from src.evaluation.relevance import AXES, DISCRETE_AXES, GalleryAxes
from src.vision.evaluation.evaluate import (
    _accumulate_axes as vision_accumulate_axes,
    _label_slug as vision_label_slug,
    partial_runs as vision_partial_runs,
    _new_metrics as vision_new_metrics,
    partial_rows,
    sample_query_rows,
    whitening_fit_rows,
)
from src.vision.data.vision_partial_query import select_rooms_to_remove
from src.vision.models.retrieval_model import VisionRetrievalPipeline
from src.vision.utils.retrieval_visualization import _metrics_summary
from src.vision.utils.config import transform_tag


# --- helpers: embeddings whose first component is the row index, so any reorder is visible in an assert ---

def labelled_embeddings(n: int, dim: int = 4) -> np.ndarray:
    embs = np.zeros((n, dim), dtype="float32")
    embs[:, 0] = np.arange(n, dtype="float32")
    embs[:, 1] = np.arange(n, dtype="float32") * 0.5
    return embs


def row_label(embs: np.ndarray, row: int) -> int:
    """Index encoded in the row; != row means the alignment broke."""
    return int(round(float(embs[row, 0])))


# --- contract 1: row <-> name, both branches, across save/reload ---

def test_graph_save_gallery_preserves_row_name_alignment(tmp_path):
    """`save_gallery` writes two files: a reorder of either would make the name join pair different plans."""
    n = 12
    embs = labelled_embeddings(n)
    names = [f"plan{i:03d}" for i in range(n)]

    save_gallery(tmp_path, embs, names)

    reloaded_embs = np.load(tmp_path / "embeddings.npy")
    reloaded_names = json.loads((tmp_path / "names.json").read_text())

    assert reloaded_names == names
    assert len(reloaded_names) == len(reloaded_embs)
    for i, name in enumerate(reloaded_names):
        assert name == f"plan{row_label(reloaded_embs, i):03d}"


def test_vision_save_load_preserves_row_path_alignment(tmp_path):
    """Same contract on the vision branch (names are PNG paths, join key is the `stem`)."""
    n = 10
    stub = SimpleNamespace(
        raw_embeddings=labelled_embeddings(n),
        image_paths=[f"/fake/snapshot_train/{i}.png" for i in range(n)],
    )
    VisionRetrievalPipeline._save(stub, str(tmp_path))

    loaded = SimpleNamespace()
    VisionRetrievalPipeline.load(loaded, str(tmp_path))

    assert loaded.image_paths == stub.image_paths
    assert len(loaded.image_paths) == len(loaded.raw_embeddings)
    # stem -> row: map used by evaluate.py to translate FAISS results into names
    stem2row = {Path(p).stem: i for i, p in enumerate(loaded.image_paths)}
    for stem, row in stem2row.items():
        assert int(stem) == row_label(loaded.raw_embeddings, row)


def test_row_name_join_catches_a_shuffled_gallery(tmp_path):
    """Names saved in a different order than rows break the join (shows the asserts above can fail)."""
    n = 8
    embs = labelled_embeddings(n)
    shuffled = [f"plan{i:03d}" for i in reversed(range(n))]

    save_gallery(tmp_path, embs, shuffled)
    reloaded_embs = np.load(tmp_path / "embeddings.npy")
    reloaded_names = json.loads((tmp_path / "names.json").read_text())

    mismatched = [
        i for i, name in enumerate(reloaded_names)
        if name != f"plan{row_label(reloaded_embs, i):03d}"
    ]
    assert len(mismatched) == n


# --- contract 2: architecture shape <-> checkpoint (`raw_skip`, pooling) ---

ENCODER_KWARGS = dict(in_dim=19, hidden_dim=16, out_dim=8, num_layers=2)


def test_raw_skip_changes_the_projection_input_width():
    """`raw_skip` concatenates the raw add-pool: `proj` grows by exactly `in_dim`."""
    plain = build_graph_encoder("gcn", pooling="add", raw_skip=False, **ENCODER_KWARGS)
    skip = build_graph_encoder("gcn", pooling="add", raw_skip=True, **ENCODER_KWARGS)

    assert plain.proj.in_features == ENCODER_KWARGS["hidden_dim"]
    assert skip.proj.in_features == ENCODER_KWARGS["hidden_dim"] + ENCODER_KWARGS["in_dim"]


def test_mean_max_pooling_doubles_the_projection_input_width():
    """Second flag with the same effect, tested together with `raw_skip`."""
    add = build_graph_encoder("gcn", pooling="add", raw_skip=False, **ENCODER_KWARGS)
    mean_max = build_graph_encoder("gcn", pooling="mean_max", raw_skip=False, **ENCODER_KWARGS)

    assert mean_max.proj.in_features == 2 * add.proj.in_features


@pytest.mark.parametrize(
    "kwargs_a, kwargs_b",
    [
        (dict(pooling="add", raw_skip=True), dict(pooling="add", raw_skip=False)),
        (dict(pooling="add", raw_skip=False), dict(pooling="mean_max", raw_skip=False)),
    ],
)
def test_checkpoint_of_a_different_shape_is_rejected(kwargs_a, kwargs_b):
    """A checkpoint trained with one shape must not load into another.

    Rejection currently comes from PyTorch's `RuntimeError`; the test pins the observable behaviour
    (refusal naming `proj` in the message), not the text."""
    trained = build_graph_encoder("gcn", **kwargs_a, **ENCODER_KWARGS)
    other = build_graph_encoder("gcn", **kwargs_b, **ENCODER_KWARGS)

    with pytest.raises(RuntimeError) as err:
        other.load_state_dict(trained.state_dict())
    assert "proj" in str(err.value)


def test_checkpoint_of_the_same_shape_reloads_identically():
    """Same shape -> load accepted and forward bit-identical."""
    torch_geometric_data = pytest.importorskip("torch_geometric.data")

    torch.manual_seed(0)
    trained = build_graph_encoder("gcn", pooling="add", raw_skip=True, **ENCODER_KWARGS)
    fresh = build_graph_encoder("gcn", pooling="add", raw_skip=True, **ENCODER_KWARGS)

    fresh.load_state_dict(trained.state_dict())
    trained.eval()
    fresh.eval()

    x = torch.arange(4 * 19, dtype=torch.float32).reshape(4, 19) / 100.0
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data = torch_geometric_data.Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        z_trained = trained(data)
        z_fresh = fresh(data)

    assert torch.equal(z_trained, z_fresh)
    assert z_trained.shape == (1, ENCODER_KWARGS["out_dim"])
    assert torch.allclose(z_trained.norm(dim=-1), torch.ones(1), atol=1e-5)


# --- contract 3: hand-computed metric reference values ---

def test_ndcg_matches_hand_computation():
    """gains [1, 0, 1] on a gallery with two relevant.

    DCG  = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.5
    IDCG = ideal [1, 1, 0] = 1 + 1/log2(3)   = 1.63093
    nDCG = 1.5 / 1.63093                     = 0.91972
    """
    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3)
    expected = dcg / idcg

    got = ndcg_at_k([1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0], k=3)

    assert got == pytest.approx(expected, abs=1e-12)
    assert got == pytest.approx(0.919721, abs=1e-6)


def test_ndcg_edge_cases():
    """Ideal order -> 1; no relevant (IDCG 0) -> 0, not NaN."""
    assert ndcg_at_k([1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], k=3) == pytest.approx(1.0)
    assert ndcg_at_k([0.0, 0.0], [0.0, 0.0, 0.0], k=2) == 0.0


def test_recall_has_two_regimes_and_a_none():
    """Denominator `min(k, #relevant)` gives two regimes.

    - few relevant (2 < k=5): 2 found of 2 -> recall 1.0
    - many relevant (10 >= k=5): 2 found of k -> Precision@5 = 0.4
    - no relevant -> None (query excluded, not counted 0)
    """
    retrieved = [True, False, True, False, False]

    assert recall_at_k(retrieved, num_relevant_total=2, k=5) == pytest.approx(1.0)
    assert recall_at_k(retrieved, num_relevant_total=10, k=5) == pytest.approx(0.4)
    assert recall_at_k(retrieved, num_relevant_total=0, k=5) is None


def test_average_precision_matches_hand_computation():
    """Relevant at positions 1 and 3, 2 relevant total, k=3.

    AP = (1/1 + 2/3) / min(3, 2) = 0.83333
    """
    got = average_precision_at_k([True, False, True], num_relevant_total=2, k=3)

    assert got == pytest.approx((1.0 + 2.0 / 3.0) / 2.0, abs=1e-12)
    assert got == pytest.approx(0.833333, abs=1e-6)
    assert average_precision_at_k([True], num_relevant_total=0, k=1) is None


def test_average_precision_rewards_early_ranks():
    """AP differs from Recall: same relevant found, different order -> different score."""
    early = average_precision_at_k([True, True, False, False], 2, k=4)
    late = average_precision_at_k([False, False, True, True], 2, k=4)

    assert early == pytest.approx(1.0)
    assert late < early


# --- contract 4: disjoint splits, statistics from the passed rows only ---

FAKE_SPLITS = {
    **{f"/fake/{i}.png": "train" for i in range(0, 20)},
    **{f"/fake/{i}.png": "valid" for i in range(20, 30)},
    **{f"/fake/{i}.png": "test" for i in range(30, 40)},
}
FAKE_PATHS = list(FAKE_SPLITS)


def test_vision_query_pools_of_valid_and_test_never_overlap(monkeypatch):
    """The gallery stays whole; the split restricts the queries only (overlapping pools would break the test-selects-nothing constraint)."""
    monkeypatch.setattr(
        "src.vision.evaluation.evaluate.get_split", lambda p: FAKE_SPLITS[p]
    )

    valid_rows = sample_query_rows(FAKE_PATHS, num_queries=10, seed=42, split="valid")
    test_rows = sample_query_rows(FAKE_PATHS, num_queries=10, seed=42, split="test")

    assert set(valid_rows).isdisjoint(test_rows)
    assert all(FAKE_SPLITS[FAKE_PATHS[i]] == "valid" for i in valid_rows)
    assert all(FAKE_SPLITS[FAKE_PATHS[i]] == "test" for i in test_rows)
    # without `split` the pool is everything: the gallery is never restricted
    assert len(sample_query_rows(FAKE_PATHS, num_queries=40, seed=42)) == len(FAKE_PATHS)


def test_vision_query_sampling_is_reproducible(monkeypatch):
    """Same seed -> same queries in the same order (pairs runs)."""
    monkeypatch.setattr(
        "src.vision.evaluation.evaluate.get_split", lambda p: FAKE_SPLITS[p]
    )

    a = sample_query_rows(FAKE_PATHS, num_queries=8, seed=42, split="valid")
    b = sample_query_rows(FAKE_PATHS, num_queries=8, seed=42, split="valid")
    c = sample_query_rows(FAKE_PATHS, num_queries=8, seed=7, split="valid")

    assert a == b
    assert a != c


def test_graph_split_indices_partition_the_dataset():
    """Same invariant on the graph branch, where split is a column of the collated dataset."""
    stub = SimpleNamespace(
        _data=SimpleNamespace(split=["train"] * 5 + ["valid"] * 3 + ["test"] * 2)
    )

    train = RplanGraphDataset.split_indices(stub, "train")
    valid = RplanGraphDataset.split_indices(stub, "valid")
    test = RplanGraphDataset.split_indices(stub, "test")

    assert set(train).isdisjoint(valid)
    assert set(train).isdisjoint(test)
    assert set(valid).isdisjoint(test)
    assert sorted(train + valid + test) == list(range(10))

    with pytest.raises(ValueError):
        RplanGraphDataset.split_indices(stub, "trainval")


def test_whitening_stats_depend_only_on_the_rows_passed():
    """Fitting on a subset gives the same parameters whether it arrives alone or extracted from a larger matrix."""
    rng = np.random.default_rng(0)
    train = rng.normal(size=(64, 8)).astype("float32")
    heldout = rng.normal(loc=3.0, size=(32, 8)).astype("float32")
    everything = np.vstack([train, heldout])

    alone = SimpleNamespace()
    VisionRetrievalPipeline._fit_whitening(alone, train)

    extracted = SimpleNamespace()
    VisionRetrievalPipeline._fit_whitening(extracted, everything[: len(train)])

    assert np.array_equal(alone.whiten_mean, extracted.whiten_mean)
    assert np.array_equal(alone.whiten_matrix, extracted.whiten_matrix)

    # whole-matrix statistics differ: the subset matters
    full = SimpleNamespace()
    VisionRetrievalPipeline._fit_whitening(full, everything)
    assert not np.allclose(alone.whiten_mean, full.whiten_mean)


def test_train_only_stats_leave_a_nonzero_mean_on_heldout():
    """Leakage signature: stats fitted on train, applied to unseen rows, leave a mean close to but not exactly 0."""
    rng = np.random.default_rng(1)
    train = rng.normal(size=(128, 8)).astype("float32")
    heldout = rng.normal(size=(32, 8)).astype("float32")

    fitted = SimpleNamespace()
    VisionRetrievalPipeline._fit_whitening(fitted, train)

    train_centered_mean = np.abs((train - fitted.whiten_mean).mean(axis=0)).max()
    heldout_centered_mean = np.abs((heldout - fitted.whiten_mean).mean(axis=0)).max()

    assert train_centered_mean < 1e-6
    assert heldout_centered_mean > 1e-6
    assert heldout_centered_mean < 1.0   # close: same distribution, no leakage


# --- contract 4-bis: whitening is estimated on train only; `prepare_index(fit_rows=...)` fits on the given rows, indexes the whole gallery ---

class _StubPipeline(VisionRetrievalPipeline):
    """Pipeline without encoder or transform: raw -> head -> whitening -> L2 -> index."""

    def __init__(self, raw_embeddings, image_paths):
        self.raw_embeddings = raw_embeddings
        self.image_paths = image_paths
        self.device = "cpu"
        self.head = None
        self.embeddings = None
        self.index = None
        self.whiten_mean = None
        self.whiten_matrix = None


def _gallery_with_two_splits(n_train=64, n_other=32, dim=8):
    """Mixed gallery like `snapshot_train/`: train + non-train in one matrix, non-train mean-shifted."""
    rng = np.random.default_rng(3)
    train = rng.normal(size=(n_train, dim)).astype("float32")
    other = rng.normal(loc=4.0, size=(n_other, dim)).astype("float32")
    embs = np.vstack([train, other])
    paths = (
        [f"/fake/train_{i}.png" for i in range(n_train)]
        + [f"/fake/valid_{i}.png" for i in range(n_other)]
    )
    return embs, paths, list(range(n_train))


def test_prepare_index_fits_whitening_only_on_the_given_rows():
    embs, paths, train_rows = _gallery_with_two_splits()

    train_only = _StubPipeline(embs.copy(), paths)
    train_only.prepare_index(whiten=True, fit_rows=train_rows)

    transductive = _StubPipeline(embs.copy(), paths)
    transductive.prepare_index(whiten=True)

    reference = SimpleNamespace()
    VisionRetrievalPipeline._fit_whitening(reference, embs[train_rows])

    # same estimate as fitting on the train rows only...
    assert np.array_equal(train_only.whiten_mean, reference.whiten_mean)
    assert np.array_equal(train_only.whiten_matrix, reference.whiten_matrix)
    # ...and different from transductive
    assert not np.allclose(train_only.whiten_mean, transductive.whiten_mean)


def test_prepare_index_keeps_the_whole_gallery_indexed():
    """Split restricts the estimate, never the search corpus."""
    embs, paths, train_rows = _gallery_with_two_splits()

    pipeline = _StubPipeline(embs.copy(), paths)
    pipeline.prepare_index(whiten=True, fit_rows=train_rows)

    assert len(pipeline.embeddings) == len(embs)
    assert pipeline.index.ntotal == len(embs)
    assert len(train_rows) < len(embs)


def test_prepare_index_rejects_an_empty_fit_set():
    """A `fit_split` selecting nothing (missing metadata) must stop the run."""
    embs, paths, _ = _gallery_with_two_splits()
    pipeline = _StubPipeline(embs, paths)

    with pytest.raises(ValueError, match="fit_rows"):
        pipeline.prepare_index(whiten=True, fit_rows=[])


def test_split_row_indices_selects_rows_not_names(monkeypatch):
    """`split_row_indices` must return gallery row indices, not names or relative positions."""
    monkeypatch.setattr(
        "src.data.rplan_metadata.get_split", lambda p, mat_dir=None: FAKE_SPLITS[p]
    )

    train = split_row_indices(FAKE_PATHS, "train")
    valid = split_row_indices(FAKE_PATHS, "valid")

    assert train == list(range(0, 20))
    assert valid == list(range(20, 30))
    assert set(train).isdisjoint(valid)
    with pytest.raises(ValueError):
        split_row_indices(FAKE_PATHS, "trainval")


def test_whitening_fit_rows_honours_the_config(monkeypatch):
    """Config -> rows: `train` restricts, `all` stays transductive, whitening off computes nothing."""
    monkeypatch.setattr(
        "src.data.rplan_metadata.get_split", lambda p, mat_dir=None: FAKE_SPLITS[p]
    )
    cfg = lambda enabled, fit_split: OmegaConf.create(
        {"whitening": {"enabled": enabled, "fit_split": fit_split}}
    )

    assert whitening_fit_rows(cfg(True, "train"), FAKE_PATHS) == list(range(0, 20))
    assert whitening_fit_rows(cfg(True, "all"), FAKE_PATHS) is None
    assert whitening_fit_rows(cfg(False, "train"), FAKE_PATHS) is None


def test_transform_tag_separates_the_two_whitening_protocols():
    """The tag enters the per-query file names: it must distinguish the protocol or runs would overwrite each other."""
    def cfg(fit_split, dim=None, head=False):
        return OmegaConf.create({
            "whitening": {"enabled": True, "fit_split": fit_split, "dim": dim},
            "head": {"enabled": head},
        })

    assert transform_tag(cfg("all")) == "whiten"
    assert transform_tag(cfg("train")) == "whiten-train"
    assert transform_tag(cfg("train", dim=768)) == "whiten768-train"
    assert transform_tag(cfg("train", head=True)) == "head+whiten-train"
    # whitening off: protocol irrelevant, tag unchanged
    off = OmegaConf.create({"whitening": {"enabled": False, "fit_split": "train"},
                            "head": {"enabled": False}})
    assert transform_tag(off) == "raw"


# --- contract 5: partial keeps the self in self-recovery, drops it from per-axis metrics ---

def _faiss_results(rows, stem2path):
    """Fake FAISS answers: only the `path` field, the one the code reads."""
    return [{"path": stem2path[r]} for r in rows]


def test_partial_rows_splits_self_recovery_from_axis_metrics():
    gallery = [f"/fake/{i}.png" for i in range(20)]
    stem2row = {Path(p).stem: i for i, p in enumerate(gallery)}
    stem2path = {i: p for i, p in enumerate(gallery)}
    qi = 7
    # self comes out second: distinguishes the two views
    ranked = [3, qi, 5, 9, 11, 2]

    self_rows, axis_rows = partial_rows(
        _faiss_results(ranked, stem2path), qi, stem2row, max_k=5
    )

    assert qi in self_rows            # self-recovery: self is the target
    assert self_rows == ranked[:5]
    assert qi not in axis_rows        # per-axis: no self (as in full)
    assert axis_rows == [3, 5, 9, 11, 2]
    # removing the self does not shorten the list: it reads the (k+1)-th answer
    assert len(axis_rows) == len(self_rows) == 5


def test_partial_rows_are_identical_when_the_self_is_not_retrieved():
    """If the self does not appear (heavy masking) the two views coincide."""
    gallery = [f"/fake/{i}.png" for i in range(20)]
    stem2row = {Path(p).stem: i for i, p in enumerate(gallery)}
    stem2path = {i: p for i, p in enumerate(gallery)}
    qi = 7
    ranked = [3, 5, 9, 11, 2, 1]

    self_rows, axis_rows = partial_rows(
        _faiss_results(ranked, stem2path), qi, stem2row, max_k=5
    )

    assert self_rows == axis_rows == ranked[:5]


def test_excluding_the_self_changes_the_axis_metrics():
    """With the self in, the query retrieves itself (max similarity on every axis) and scores inflate."""
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    qi = 1                      # p1 has a twin (p2) on composition and topology
    k_values = (1, 3)

    with_self = vision_new_metrics(k_values)
    vision_accumulate_axes(with_self, {ax: 0 for ax in DISCRETE_AXES}, axes, qi,
                           [qi, 2, 0], k_values, exclude_self=False)
    without_self = vision_new_metrics(k_values)
    vision_accumulate_axes(without_self, {ax: 0 for ax in DISCRETE_AXES}, axes, qi,
                           [2, 0, 3], k_values, exclude_self=True)

    assert with_self["composition"]["ndcg"][1][0] == pytest.approx(1.0)
    assert without_self["composition"]["ndcg"][1][0] <= with_self["composition"]["ndcg"][1][0]
    # and the number of relevant changes: the self was counted
    assert (count_relevant(axes, "composition", qi, exclude_self=False)
            == count_relevant(axes, "composition", qi, exclude_self=True) + 1)


def test_visualization_metric_line_drops_the_self_from_retrieved_rows():
    """The line under a partial figure must follow the table convention (self out of both retrieved and relevant)."""
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    qi = 1
    stem2row = {n: i for i, n in enumerate(names)}
    results = [{"path": f"/fake/{names[r]}.png"} for r in (qi, 2, 0)]
    sims = {ax: np.stack([axes.sim(ax, qi)]).ravel() for ax in AXES}

    with_self = _metrics_summary(sims, results, qi, axes, stem2row, exclude_self=False)
    without_self = _metrics_summary(sims, results, qi, axes, stem2row, exclude_self=True)

    assert with_self != without_self
    assert "n/a" not in without_self.split("|")[0]   # query is not a singleton


# --- contract 6: gallery shared across branches (same gallery_sha1, which depends on row order) ---

SHARED_SPLITS = {f"p{i:02d}": ("valid" if i % 3 == 0 else "train") for i in range(12)}


def _two_branch_galleries():
    """The two galleries as they are: same names in the middle, different tails, orders and formats (PNG path vs bare names)."""
    common = [f"p{i:02d}" for i in range(12)]
    vision_entries = [f"/snap/{n}.png" for n in reversed(common)] + ["/snap/onlyV.png"]
    graph_entries = common + ["onlyG1", "onlyG2"]
    return vision_entries, graph_entries, common


def test_read_gallery_names_accepts_both_formats_and_rejects_duplicates(tmp_path):
    vision = tmp_path / "image_paths.json"
    vision.write_text(json.dumps(["/snap/a.png", "/snap/b.png"]))
    graph = tmp_path / "names.json"
    graph.write_text(json.dumps(["a", "b"]))

    assert read_gallery_names(vision) == read_gallery_names(graph) == ["a", "b"]

    dupes = tmp_path / "dupes.json"
    dupes.write_text(json.dumps(["a", "a", "b"]))
    with pytest.raises(ValueError, match="duplicati"):
        read_gallery_names(dupes)


def test_shared_gallery_gives_the_two_branches_the_same_sha1():
    """After restriction both branches have the same list in the same order, hence the same hash (what `check_compatible` verifies)."""
    vision_entries, graph_entries, _ = _two_branch_galleries()
    vn, gn = canonical_names(vision_entries), canonical_names(graph_entries)

    shared = compute_shared_names(vn, gn)
    vision_rows = restrict_rows(vn, shared)
    graph_rows = restrict_rows(gn, shared)

    vision_after = [vn[i] for i in vision_rows]
    graph_after = [gn[i] for i in graph_rows]

    assert vision_after == graph_after == shared
    assert gallery_sha1(vision_after) == gallery_sha1(graph_after)
    # ...and not before restriction
    assert gallery_sha1(vn) != gallery_sha1(gn)


def test_restrict_rows_refuses_a_gallery_that_misses_a_shared_name():
    """An inner-join file not belonging to this gallery must stop the run."""
    with pytest.raises(KeyError):
        restrict_rows(["a", "b"], ["a", "b", "c"])


def test_compute_shared_names_refuses_an_empty_intersection():
    with pytest.raises(ValueError, match="intersezione vuota"):
        compute_shared_names(["a"], ["b"])


def test_shared_gallery_file_roundtrips_and_detects_tampering(tmp_path):
    _, _, common = _two_branch_galleries()
    out = tmp_path / "shared_gallery.json"
    write_shared_gallery(out, common, sources={"vision": {"n": 13}, "graph": {"n": 14}})

    assert load_shared_names(out) == common

    payload = json.loads(out.read_text())
    payload["names"] = payload["names"][:-1]        # a row cut by hand
    out.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sha1"):
        load_shared_names(out)


class _StubGraphDataset:
    """Fake dataset: sampling needs only length and splits."""

    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def split_indices(self, split):
        # graph-only names exist in the dataset but not in the inner join: they still have a split and the filter must drop them
        return [i for i, n in enumerate(self.names)
                if SHARED_SPLITS.get(n, "train") == split]


def test_both_branches_sample_the_same_queries_after_the_join(monkeypatch):
    """Same canonical gallery, seed and split: both branches sample the same plans."""
    vision_entries, graph_entries, _ = _two_branch_galleries()
    vn, gn = canonical_names(vision_entries), canonical_names(graph_entries)
    shared = compute_shared_names(vn, gn)

    # vision: restriction reorders paths; queries sampled afterwards
    vision_paths = [vision_entries[i] for i in restrict_rows(vn, shared)]
    monkeypatch.setattr(
        "src.vision.evaluation.evaluate.get_split",
        lambda p: SHARED_SPLITS[Path(p).stem],
    )
    vision_rows = sample_query_rows(vision_paths, num_queries=3, seed=42, split="valid")

    # graph: restriction yields a dataset-index -> new-row map
    dataset = _StubGraphDataset(gn)
    row_of = {ds: new for new, ds in enumerate(restrict_rows(gn, shared))}
    graph_rows = graph_sample_query_rows(dataset, num_queries=3, seed=42,
                                         split="valid", row_of=row_of)

    assert vision_rows == graph_rows
    assert [shared[i] for i in vision_rows] == [shared[i] for i in graph_rows]
    assert all(SHARED_SPLITS[shared[i]] == "valid" for i in vision_rows)


# --- contract 7: degraded graph query; the graph partial must remove the same rooms as vision without perturbing the rest ---

def _full_meta():
    """4-room plan, chain 0-1-2-3: topology exposes wrong edge remapping."""
    return RoomMeta(
        name="p42", split="valid",
        room_types=(0, 1, 2, 3),
        edges=((0, 1, 1), (1, 2, 1), (2, 3, 1)),
        boxes=((0, 0, 20, 20), (20, 0, 40, 20), (0, 20, 20, 40), (20, 20, 40, 40)),
        footprint=(0, 0, 40, 40), entrance=None,
    )


def test_filter_meta_remaps_edges_and_drops_the_ones_that_touch_a_removed_room():
    meta = _full_meta()

    reduced = filter_meta(meta, [1])          # chain 0-1-2-3 loses node 1

    assert reduced.room_types == (0, 2, 3)
    # edge 2-3 survives as 1-2; 0-1 and 1-2 vanish with node 1
    assert reduced.edges == ((1, 2, 1),)
    assert reduced.boxes == (meta.boxes[0], meta.boxes[2], meta.boxes[3])
    # invariants: identity and footprint of the full plan
    assert reduced.name == meta.name and reduced.split == meta.split
    assert reduced.footprint == meta.footprint


def test_filter_meta_returns_none_when_nothing_survives():
    assert filter_meta(_full_meta(), [0, 1, 2, 3]) is None


def test_partial_graph_keeps_the_surviving_nodes_bit_identical():
    """Removing a room leaves the other rooms' features unchanged (normalised on the fixed 256 grid)."""
    meta = _full_meta()
    full = build_graph(meta)

    partial, removed = make_partial_graph(
        meta, "random", {"fraction": 0.25}, random.Random(0)
    )

    assert len(removed) == 1
    kept = [i for i in range(meta.num_rooms) if i not in removed]
    assert partial.x.shape == (len(kept), full.x.shape[1])
    for new_row, old_row in enumerate(kept):
        assert torch.equal(partial.x[new_row], full.x[old_row])


def test_partial_graph_removes_the_same_rooms_as_the_vision_branch():
    """For the same plan, seed and strategy both branches remove the same rooms."""
    meta = _full_meta()
    qi, seed = 7, 42

    vision_removed = select_rooms_to_remove(
        meta, "random", {"fraction": 0.5}, random.Random(seed + qi)
    )
    _, graph_removed = make_partial_graph(
        meta, "random", {"fraction": 0.5}, random.Random(seed + qi)
    )

    assert graph_removed == vision_removed
    assert len(graph_removed) == 2


@pytest.mark.parametrize(
    "strategy, params, expected_removed",
    [
        ("semantic", {"keep_types": [0, 1]}, [2, 3]),   # keeps only types 0 and 1
        ("topology", {"max_degree": 1}, [0, 3]),        # removes the two leaves
        ("random", {"fraction": 0.0}, []),              # no removal
    ],
)
def test_partial_graph_supports_every_vision_strategy(strategy, params, expected_removed):
    """The three vision strategies behave identically on the graph."""
    meta = _full_meta()

    graph, removed = make_partial_graph(meta, strategy, params, random.Random(1))

    assert removed == expected_removed
    assert graph.x.shape[0] == meta.num_rooms - len(removed)
    # every surviving edge points to existing nodes
    if graph.edge_index.numel():
        assert int(graph.edge_index.max()) < graph.x.shape[0]


def test_the_two_branches_expand_the_same_partial_runs():
    """Same run labels in both branches, so per-query file names are parallel."""
    graph_args = SimpleNamespace(
        partial_strategies=["random", "semantic", "topology"],
        partial_fractions=[0.0, 0.5], partial_keep_types=[0, 2, 3],
        partial_max_degree=1,
    )
    vision_cfg = OmegaConf.create({
        "strategies": {
            "random": {"enabled": True, "fractions": [0.0, 0.5]},
            "semantic": {"enabled": True, "keep_types": [0, 2, 3]},
            "topology": {"enabled": True, "max_degree": 1},
        }
    })

    assert graph_partial_runs(graph_args) == vision_partial_runs(vision_cfg)
    assert [graph_label_slug(l) for l, _, _ in graph_partial_runs(graph_args)] == \
           [vision_label_slug(l) for l, _, _ in vision_partial_runs(vision_cfg)]


def test_graph_partial_evaluation_writes_self_rr_per_query(tmp_path, monkeypatch):
    """Smoke of the whole loop on a synthetic gallery: degraded graphs -> encoder -> FAISS -> two views -> .npz."""
    faiss = pytest.importorskip("faiss")
    metas = [
        RoomMeta(name=f"g{i}", split="valid",
                 room_types=(0, 1, 2, 3)[: 3 + i % 2],
                 edges=((0, 1, 1), (1, 2, 1)),
                 boxes=tuple((10 * j, 10 * i, 10 * j + 8, 10 * i + 8) for j in range(4))[: 3 + i % 2],
                 footprint=(0, 0, 40, 40), entrance=None)
        for i in range(6)
    ]
    by_name = {m.name: m for m in metas}
    monkeypatch.setattr(
        "src.graph.evaluation.graph_evaluate.load_metadata",
        lambda name, *a, **k: by_name[Path(str(name)).stem],
    )

    torch.manual_seed(0)
    encoder = build_graph_encoder("gcn", pooling="add", raw_skip=False,
                                  in_dim=19, hidden_dim=8, out_dim=4, num_layers=2).eval()
    args = SimpleNamespace(
        partial_seed=42, partial_strategies=["random"], partial_fractions=[0.25],
        partial_keep_types=[0, 2, 3], partial_max_degree=1,
        batch_size=4, baseline_hist=False,
    )
    with torch.no_grad():
        gallery = np.ascontiguousarray(
            np.vstack([encoder(build_graph(m)).numpy() for m in metas]), dtype=np.float32
        )
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)

    names = [m.name for m in metas]
    ctx = {"dir": tmp_path, "tag": "gcn_test", "split": "valid", "seed": 42,
           "gallery": {"n": len(names), "sha1": gallery_sha1(names), "source": "synthetic"}}

    graph_evaluate_partial(index, GalleryAxes(metas), list(range(len(metas))),
                           (1, 3), names, encoder, None, args, "cpu", perquery=ctx)

    written = list(tmp_path.glob("*.npz"))
    assert len(written) == 1
    assert written[0].name == "graph_gcn_test_partial-random-f0.25_valid.npz"
    data = load_perquery(written[0])
    assert data.meta["mode"] == "partial"
    assert data.meta["partial_label"] == "random f=0.25"
    assert data.meta["exclude_self"] is True     # holds here too
    assert data.self_rr is not None and len(data.self_rr) == len(metas)
