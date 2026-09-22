# tests/test_graph_robustness_probe.py

"""
Smoke test CPU della sonda di robustezza graph (`src/graph/evaluation/robustness_probe.py`)
e della sua integrazione in `src/graph/training/train_gnn.py` (selezione sulla
robustezza, 15 set 2026). Dati SINTETICI: nessun `.mat`/`graphs.pt` reale, `load_metadata`
e' monkeypatchato su un piccolo mondo di `RoomMeta` fatti a mano (stile
`tests/test_graph_lost_marker.py`/`tests/test_graph_asym_pairs.py`).

Invarianti coperte:
1. righe sonda disgiunte dalle righe di valutazione, tutte nel valid E dentro la
   gallery ristretta (T1); l'esclusione segue esattamente `sample_query_rows`
   del YAML (T2).
2. il grafo parziale della sonda e' bit-identico a quello della valutazione, per
   ogni frazione, con e senza `lost_marker` (T3).
3. `self_reciprocal_ranks` == riferimento FAISS con «pareggio vince il self» (T4).
4. `auc` = media dei MRR sulle frazioni di `AUC_FRACTIONS` (T5).
5. `score()` deterministico, embeddings via batch pre-costruiti == via DataLoader,
   stato/RNG ripristinati (T6).
6. `_ShadowSelection` replica il loop di selezione storico (T7).
7. Default (`selection_probe="full"`, shadow spenta) e chiavi di `_save_summary`
   senza `extra`; guardie di `parse_args` (T8).
8. Wiring testuale di `scripts/graph/03_train_gnn.sh` / `04_eval_gnn.sh` per le
   varianti `*rob`/`*lostrob` (T9).
9. `RobustnessProbe.build` propaga (non ignora) il proprio `lost_marker` alla
   costruzione dei grafi-query, sui VALORI oltre che sulla forma (T10, trovato
   dal final-reviewer: un controllo di sola shape non basta, vedi T10 sotto).
10. `DEFAULT_PARTIAL_SEED` (copiato a mano in `robustness_probe.py`) resta
    uguale al default di `--partial-seed` di `graph_evaluate` (T11, idem).

Esecuzione: python -m pytest tests/test_graph_robustness_probe.py -v
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.data.rplan_metadata import RoomMeta
from src.evaluation.gallery_join import write_shared_gallery
from src.evaluation.robustness_auc import AUC_FRACTIONS
from src.graph.evaluation import graph_evaluate
from src.graph.evaluation.robustness_probe import (
    DEFAULT_PARTIAL_SEED,
    RobustnessProbe,
    partial_query_graphs,
    self_reciprocal_ranks,
)
from src.graph.evaluation import robustness_probe as robustness_probe_module
from src.graph.graph_builder import NODE_FEATURE_DIM, NUM_RELATION_TYPES, build_graph
from src.graph.graph_partial_query import make_partial_graph
from src.graph.models import build_graph_encoder
from src.graph.training.train_gnn import _ShadowSelection, _save_summary, parse_args
from src.graph.transforms import LOST_MARKER_DIM, build_node_transform

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fixture: mondo sintetico (RoomMeta a mano, niente .mat) + dataset finto con
# l'interfaccia minima usata da RobustnessProbe.build/_prepare.
# ---------------------------------------------------------------------------


def _meta(name, n, rng, split):
    types = [rng.randrange(0, 13) for _ in range(n)]
    edges = [(i, i + 1, rng.randrange(0, 9)) for i in range(n - 1)]
    edges += [(0, n - 1, 3)] if n > 3 else []
    boxes = [(10 * i, 5, 10 * i + 9, 30 + i) for i in range(n)]
    return RoomMeta(name=name, split=split, room_types=tuple(types), edges=tuple(edges),
                    boxes=tuple(boxes), footprint=(0, 0, 10 * n, 40), entrance=None)


def _build_metas(n_total=24, seed=0):
    """Nomi p0..p{n_total-1} in ORDINE NON canonico (l'ordine canonico e' quello
    alfabetico dei nomi, deciso da `gallery_join` sul JSON condiviso): con questi
    nomi ("p10" < "p2" come stringhe) l'ordine canonico differisce da quello di
    costruzione, quindi un test che confondesse i due ordini fallirebbe."""
    rng = random.Random(seed)
    sizes = [4, 5, 6, 7, 8]
    splits = ["train", "valid", "test"]
    return [_meta(f"p{i}", sizes[i % len(sizes)], rng, splits[i % 3]) for i in range(n_total)]


class _FakeDataset:
    """Interfaccia minima usata da `RobustnessProbe.build`/`_prepare`: `_indices`,
    `_data.name`, `split_indices`, `__getitem__`. Niente `.mat`/cache su disco."""

    def __init__(self, metas, transform):
        self.metas = metas
        self.transform = transform
        self._indices = None
        self._data = SimpleNamespace(name=[m.name for m in metas], split=[m.split for m in metas])

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, i):
        g = build_graph(self.metas[i])
        return self.transform(g) if self.transform is not None else g

    def split_indices(self, split):
        return [i for i, m in enumerate(self.metas) if m.split == split]


def _patch_load_metadata(monkeypatch, metas_by_name):
    monkeypatch.setattr(
        "src.graph.evaluation.robustness_probe.load_metadata",
        lambda name, *a, **kw: metas_by_name.get(str(name)),
    )


def _write_gallery_config(tmp_path, ds_names, excluded, num_queries, seed,
                          split="valid", k_values=(1, 5, 10)):
    shared = sorted(n for n in ds_names if n not in excluded)
    gallery_path = tmp_path / "shared_gallery.json"
    write_shared_gallery(gallery_path, shared, sources={})
    cfg_path = tmp_path / "graph_retrieval.yaml"
    OmegaConf.save(OmegaConf.create({
        "num_queries": num_queries, "seed": seed, "split": split,
        "k_values": list(k_values), "gallery_names": str(gallery_path),
    }), cfg_path)
    return cfg_path, gallery_path, shared


def _make_world(tmp_path, monkeypatch, *, excluded=(), lost_marker=False,
                eval_num_queries=3, eval_seed=7, probe_num_queries=3, probe_seed=123,
                n_total=24, meta_seed=0, batch_size=4):
    metas = _build_metas(n_total=n_total, seed=meta_seed)
    metas_by_name = {m.name: m for m in metas}
    _patch_load_metadata(monkeypatch, metas_by_name)
    transform = build_node_transform(normalize=False, drop_self_loops=True, lost_marker=lost_marker)
    dataset = _FakeDataset(metas, transform)
    ds_names = [m.name for m in metas]
    ecfg_path, gallery_path, shared = _write_gallery_config(
        tmp_path, ds_names, set(excluded), eval_num_queries, eval_seed)
    probe = RobustnessProbe.build(
        dataset, transform, str(ecfg_path), num_queries=probe_num_queries, seed=probe_seed,
        lost_marker=lost_marker, batch_size=batch_size,
    )
    return SimpleNamespace(metas=metas, metas_by_name=metas_by_name, dataset=dataset,
                            transform=transform, ecfg_path=ecfg_path, gallery_path=gallery_path,
                            shared=shared, probe=probe, lost_marker=lost_marker)


# ---------------------------------------------------------------------------
# T1 -- righe sonda disgiunte dalla valutazione, dentro il valid ristretto
# ---------------------------------------------------------------------------


def test_probe_rows_disjoint_from_eval_and_inside_restricted_valid(tmp_path, monkeypatch):
    w = _make_world(tmp_path, monkeypatch, excluded={"p1", "p2", "p5"})
    ecfg = OmegaConf.load(w.ecfg_path)

    ds_names = [m.name for m in w.metas]
    placeholder = np.empty((len(ds_names), 0), dtype=np.float32)
    _, names, row_of = graph_evaluate.restrict_gallery(placeholder, ds_names, str(w.gallery_path))
    eval_rows = graph_evaluate.sample_query_rows(
        w.dataset, int(ecfg.num_queries), int(ecfg.seed), str(ecfg.split), row_of)

    assert len(w.probe.query_rows) == 3
    assert set(w.probe.query_rows) & set(eval_rows) == set()
    for qi in w.probe.query_rows:
        assert w.metas_by_name[names[qi]].split == "valid"
    assert "p1" not in names   # esclusa dalla gallery ristretta: non deve poter comparire


# ---------------------------------------------------------------------------
# T2 -- l'esclusione segue esattamente sample_query_rows del YAML
# ---------------------------------------------------------------------------


def test_probe_query_rows_equal_direct_recomputation_from_yaml(tmp_path, monkeypatch):
    w = _make_world(tmp_path, monkeypatch, excluded={"p1", "p2", "p5"},
                    eval_num_queries=3, eval_seed=7, probe_num_queries=3, probe_seed=123)
    ecfg = OmegaConf.load(w.ecfg_path)

    ds_names = [m.name for m in w.metas]
    placeholder = np.empty((len(ds_names), 0), dtype=np.float32)
    _, names, row_of = graph_evaluate.restrict_gallery(placeholder, ds_names, str(w.gallery_path))
    eval_rows = graph_evaluate.sample_query_rows(
        w.dataset, int(ecfg.num_queries), int(ecfg.seed), str(ecfg.split), row_of)

    from src.graph.evaluation.robustness_probe import PROBE_SPLIT, probe_query_rows
    pool = sorted(row_of[i] for i in w.dataset.split_indices(PROBE_SPLIT) if i in row_of)
    expected_rows = probe_query_rows(pool, eval_rows, 3, 123)

    assert w.probe.query_rows == expected_rows   # stesso ordine, non solo stesso insieme


def test_probe_query_rows_deterministic_across_rebuilds(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    w1 = _make_world(tmp_path / "a", monkeypatch, excluded={"p1", "p2", "p5"})
    w2 = _make_world(tmp_path / "b", monkeypatch, excluded={"p1", "p2", "p5"})
    assert w1.probe.query_rows == w2.probe.query_rows


# ---------------------------------------------------------------------------
# T3 -- grafo parziale della sonda == percorso di valutazione, con/senza marker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lost_marker", [False, True])
def test_partial_query_graphs_matches_evaluation_path(lost_marker):
    metas = _build_metas(n_total=6, seed=1)
    transform = build_node_transform(normalize=False, drop_self_loops=True, lost_marker=lost_marker)
    rows = [0, 2, 4]           # qi != indice locale: mette alla prova seed+qi
    chosen = [metas[i] for i in rows]
    seed = 42

    per_f, kept_rows, n_dropped = partial_query_graphs(
        chosen, rows, AUC_FRACTIONS, seed, transform, lost_marker)

    assert n_dropped == 0
    assert kept_rows == rows
    for f in AUC_FRACTIONS:
        for local_i, qi in enumerate(rows):
            graph, _ = make_partial_graph(chosen[local_i], "random", {"fraction": float(f)},
                                          random.Random(seed + qi), lost_marker=lost_marker)
            expected = transform(graph)
            got = per_f[f][local_i]
            assert torch.equal(got.x, expected.x)
            assert torch.equal(got.edge_index, expected.edge_index)
            assert torch.equal(got.edge_attr, expected.edge_attr)


# ---------------------------------------------------------------------------
# T4 -- self_reciprocal_ranks == riferimento FAISS, pareggio vince il self
# ---------------------------------------------------------------------------


def test_self_reciprocal_ranks_matches_faiss_reference_no_ties():
    import faiss

    rng = np.random.default_rng(0)
    n, d, p = 300, 16, 40
    gallery = rng.normal(size=(n, d)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    rows = rng.choice(n, size=p, replace=False)
    q = np.empty((p, d), dtype=np.float32)
    for i, r in enumerate(rows):
        if i % 2 == 0:
            q[i] = gallery[r] + rng.normal(scale=0.02, size=d).astype(np.float32)  # self ~vicino
        else:
            q[i] = rng.normal(size=d).astype(np.float32)                          # self spesso fuori rango
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(d)
    index.add(gallery)
    max_rank = 20
    _, ranked_full = index.search(q, n)

    faiss_rr = []
    for i, r in enumerate(rows):
        pos = int(np.where(ranked_full[i] == r)[0][0])
        rank = pos + 1
        faiss_rr.append(1.0 / rank if rank <= max_rank else 0.0)
    faiss_rr = np.asarray(faiss_rr, dtype=np.float32)
    assert bool((faiss_rr == 0.0).any()) and bool((faiss_rr > 0.0).any())  # copre entrambi i rami

    got = self_reciprocal_ranks(torch.from_numpy(q), torch.from_numpy(gallery),
                                torch.from_numpy(rows.astype(np.int64)), max_rank)
    assert np.allclose(got.numpy(), faiss_rr, atol=1e-6)


def test_self_reciprocal_ranks_tie_breaks_in_favor_of_self():
    gallery = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), dim=1)
    q = gallery[[0]]                        # identica a riga 0 E riga 1: pareggio esatto
    rows = torch.tensor([0])
    rr = self_reciprocal_ranks(q, gallery, rows, max_rank=1)
    assert rr.item() == 1.0                 # rank 1 nonostante il pareggio


# ---------------------------------------------------------------------------
# T5 -- auc = media dei MRR sulle frazioni AUC_FRACTIONS
# ---------------------------------------------------------------------------


def test_score_auc_is_mean_of_mrr_over_auc_fractions(tmp_path, monkeypatch):
    w = _make_world(tmp_path, monkeypatch, n_total=30, eval_num_queries=4, probe_num_queries=4)
    encoder = build_graph_encoder("gcn", in_dim=NODE_FEATURE_DIM, hidden_dim=8, out_dim=6,
                                  num_layers=2, pooling="add", dropout=0.0, raw_skip=True)
    scores = w.probe.score(encoder, "cpu")

    assert w.probe.fractions == AUC_FRACTIONS
    assert set(scores) == {"auc", *[f"mrr_f{f}" for f in AUC_FRACTIONS]}
    expected = float(np.mean([scores[f"mrr_f{f}"] for f in AUC_FRACTIONS]))
    assert scores["auc"] == pytest.approx(expected, abs=1e-7)


# ---------------------------------------------------------------------------
# T6 -- score() deterministico, batch pre-costruiti == DataLoader, stato/RNG
# ---------------------------------------------------------------------------


def _gat(in_dim):
    return build_graph_encoder("gat", in_dim=in_dim, hidden_dim=8, out_dim=6, num_layers=2,
                               pooling="add", dropout=0.0, heads=2, edge_dim=NUM_RELATION_TYPES,
                               attn_dropout=0.0, raw_skip=True)


@pytest.mark.parametrize("lost_marker,in_dim", [
    (False, NODE_FEATURE_DIM),
    (True, NODE_FEATURE_DIM + LOST_MARKER_DIM),
])
def test_score_deterministic_matches_dataloader_and_preserves_state(
    tmp_path, monkeypatch, lost_marker, in_dim,
):
    w = _make_world(tmp_path, monkeypatch, n_total=30, lost_marker=lost_marker,
                    eval_num_queries=4, probe_num_queries=4)
    encoder = _gat(in_dim)
    encoder.train()

    rng_before = torch.get_rng_state()
    s1 = w.probe.score(encoder, "cpu")
    assert torch.equal(rng_before, torch.get_rng_state())   # niente consumo dell'RNG globale
    assert encoder.training is True                          # stato ripristinato dopo score()

    s2 = w.probe.score(encoder, "cpu")
    assert s1 == s2                                           # deterministico, bit-esatto in eval

    from torch_geometric.loader import DataLoader
    graphs_gallery = [w.dataset[i] for i in w.probe.gallery_ds_rows]
    loader = DataLoader(graphs_gallery, batch_size=w.probe.batch_size, shuffle=False)
    encoder.eval()
    with torch.no_grad():
        via_loader = torch.cat([encoder(b) for b in loader])
        via_batches = torch.cat([encoder(b) for b in w.probe._gallery_batches])
    assert torch.allclose(via_loader, via_batches, atol=1e-6)


# ---------------------------------------------------------------------------
# T7 -- _ShadowSelection replica il loop di selezione storico
# ---------------------------------------------------------------------------


def _reference_selection(scores_seq, patience, max_epoch):
    """Riproduce alla lettera train.py:310-317 (== :155-163 di _ShadowSelection.update)."""
    best_score, best_epoch, wait = -float("inf"), 0, 0
    for epoch, score in enumerate(scores_seq, start=1):
        if epoch > max_epoch:
            break
        if score > best_score + 1e-4:
            best_score, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
            if patience and wait >= patience:
                break
    return best_epoch, best_score


def test_shadow_selection_matches_reference_loop_on_five_sequences():
    encoder = torch.nn.Linear(2, 2)
    cases = {
        "monotona": ([0.1, 0.2, 0.3, 0.4, 0.5], 3, 10),
        "rumorosa": ([0.50, 0.52, 0.51, 0.53, 0.40, 0.30, 0.20], 2, 10),
        "picco_precoce": ([0.9, 0.5, 0.4, 0.3, 0.2, 0.1], 2, 10),
        "picco_oltre_tetto": ([float(i) for i in range(1, 13)], 100, 10),
        "pareggio_entro_1e-4": ([0.5, 0.50005, 0.49997, 0.50009, 0.3], 2, 10),
        # confine ESATTO (score == best + 1e-4, calcolato in float, non un letterale
        # arrotondato): distingue `>` da `>=` nel confronto, che i primi 5 casi non
        # toccano mai esattamente.
        "confine_esatto_1e-4": ([0.3, 0.3 + 1e-4], 5, 10),
        # patience=1: un solo passo di margine fra "stop dopo epoch 2" (`>=`) e
        # "stop dopo epoch 3" (`>`, off-by-one) — con un rialzo vero all'epoca 3 il
        # best finale cambia SOLO se lo stop scatta un passo tardi.
        "patience_confine": ([1.0, 0.5, 5.0], 1, 10),
    }
    for name, (scores_seq, patience, max_epoch) in cases.items():
        expected_epoch, expected_score = _reference_selection(scores_seq, patience, max_epoch)
        shadow = _ShadowSelection(patience, max_epoch)
        for epoch, score in enumerate(scores_seq, start=1):
            shadow.update(epoch, score, {"s": score}, encoder)
        assert shadow.best_epoch == expected_epoch, name
        assert shadow.best_score == pytest.approx(expected_score), name


# ---------------------------------------------------------------------------
# T8 -- default full/shadow spenta, chiavi storiche di _save_summary, guardie
# ---------------------------------------------------------------------------


def test_default_selection_probe_is_full_and_shadow_disabled(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_gnn.py"])
    args = parse_args()
    assert args.selection_probe == "full"
    assert args.shadow_patience == 0


def test_save_summary_without_extra_matches_historical_keys(tmp_path):
    cfg = SimpleNamespace(encoder="gcn", variant="v", lr=1e-3, weight_decay=1e-5,
                          temperature=0.2, batch_size=8, hidden_dim=16, out_dim=8,
                          num_layers=2, pooling="add", dropout=0.0, node_drop=0.1,
                          edge_drop=0.1, feat_mask=0.1, epochs=5)
    _save_summary(tmp_path, cfg, "ndcg@10:mean", 0.5, 3, {"ndcg_composition": 0.4})
    summary = json.loads((tmp_path / "training_summary.json").read_text())

    expected_keys = {"encoder", "variant", "selection_criterion", "best_score", "best_epoch",
                     "epochs_max", "probe_scores_at_best", "pair_mode", "lost_marker",
                     "hyperparams"}
    assert set(summary) == expected_keys


def test_parse_args_rejects_inconsistent_shadow_and_selection_probe_flags(monkeypatch):
    invalid = [
        ["--shadow-patience", "5"],
        ["--selection-probe", "partial", "--probe-every", "0"],
        ["--selection-probe", "partial", "--shadow-patience", "5", "--probe-every", "2"],
    ]
    for extra in invalid:
        monkeypatch.setattr(sys, "argv", ["train_gnn.py", *extra])
        with pytest.raises(SystemExit):
            parse_args()

    monkeypatch.setattr(sys, "argv", ["train_gnn.py", "--selection-probe", "partial",
                                      "--shadow-patience", "5", "--probe-every", "1"])
    args = parse_args()
    assert args.selection_probe == "partial" and args.shadow_patience == 5


# ---------------------------------------------------------------------------
# T9 -- wiring testuale di 03_train_gnn.sh / 04_eval_gnn.sh (niente `source`)
# ---------------------------------------------------------------------------


def _echo_for_case_label(text: str, label: str) -> str | None:
    """Contenuto di `echo "..."` della riga `case` il cui insieme di alternative
    (separate da `|`) contiene `label`. Legge il testo, non lo esegue."""
    for line in text.splitlines():
        stripped = line.strip()
        if ")" not in stripped:
            continue
        case_label, _, rest = stripped.partition(")")
        if label in [alt.strip() for alt in case_label.split("|")]:
            m = re.search(r'echo "([^"]*)"', rest)
            if m:
                return m.group(1)
    return None


def test_03_and_04_scripts_wire_robustness_probe_variants_to_gat_yaml():
    txt03 = (REPO_ROOT / "scripts" / "graph" / "03_train_gnn.sh").read_text()
    txt04 = (REPO_ROOT / "scripts" / "graph" / "04_eval_gnn.sh").read_text()
    gat_cfg = OmegaConf.load(REPO_ROOT / "configs" / "graph_models" / "gat.yaml")

    rob_match = re.search(r'ROB="([^"]+)"', txt03)
    assert rob_match, "riga ROB non trovata in 03_train_gnn.sh"
    rob_flags = rob_match.group(1)
    assert "--selection-probe partial" in rob_flags
    assert "--epochs 300" in rob_flags
    assert "--patience 0" in rob_flags
    assert f"--shadow-patience {int(gat_cfg.patience)}" in rob_flags   # == gat.yaml patience (10)
    assert f"--shadow-epochs {int(gat_cfg.epochs)}" in rob_flags      # == gat.yaml epochs (150)

    for variant in ("asymrob", "asymrobrep"):
        echo = _echo_for_case_label(txt03, variant)
        assert echo is not None, f"{variant} assente da variant_flags (03)"
        assert "$ROB" in echo and "--lost-marker" not in echo

    echo_lostrob = _echo_for_case_label(txt03, "asymlostrob")
    assert echo_lostrob is not None and "$ROB" in echo_lostrob and "--lost-marker" in echo_lostrob

    known_04 = ("asymrob", "asymrob_selfull", "asymlostrob", "asymlostrob_selfull",
                "asymrobrep", "asymrobrep_selfull")
    for variant in known_04:
        assert variant in txt04, f"{variant} assente da KNOWN_VARIANTS/case (04)"

    for variant in ("asymlostrob", "asymlostrob_selfull"):
        echo = _echo_for_case_label(txt04, variant)
        assert echo == "--lost-marker"

    for variant in ("asymrob", "asymrob_selfull", "asymrobrep", "asymrobrep_selfull"):
        echo = _echo_for_case_label(txt04, variant)
        assert echo == ""                # nessun flag extra: solo la selezione cambia, non la rete


# ---------------------------------------------------------------------------
# T10 -- build() propaga il proprio lost_marker a partial_query_graphs
# (trovato dal final-reviewer: robustness_probe.py:192 passa `lost_marker`
# a `partial_query_graphs`; una mutazione che lo sostituisse con `False` non
# cambia la FORMA di `x` — `AppendLostMarker` appende comunque una colonna,
# a zero se il grafo non porta `lost_marker` — quindi un controllo di sola
# shape non la cattura. Serve verificare anche i VALORI.
# ---------------------------------------------------------------------------


def test_build_propagates_lost_marker_to_query_graphs_and_detects_mutation(tmp_path, monkeypatch):
    w_on = _make_world(tmp_path / "on", monkeypatch, lost_marker=True, n_total=18,
                       eval_num_queries=3, probe_num_queries=3)
    for f in w_on.probe.fractions:
        for g in w_on.probe.query_graphs[f]:
            assert g.x.shape[1] == NODE_FEATURE_DIM + LOST_MARKER_DIM
    any_nonzero_on = any(
        bool((g.x[:, NODE_FEATURE_DIM] > 0).any())
        for f in w_on.probe.fractions for g in w_on.probe.query_graphs[f]
    )
    assert any_nonzero_on   # il marcatore porta informazione vera, non e' un padding fisso

    w_off = _make_world(tmp_path / "off", monkeypatch, lost_marker=False, n_total=18,
                        eval_num_queries=3, probe_num_queries=3)
    for f in w_off.probe.fractions:
        for g in w_off.probe.query_graphs[f]:
            assert g.x.shape[1] == NODE_FEATURE_DIM   # 19: transform senza marcatore

    # Mutazione (riproduce robustness_probe.py:192 con `lost_marker` -> `False`):
    # forzo l'argomento interno a `partial_query_graphs`, lasciando build()
    # invariato per il resto (transform e lost_marker esterno restano True).
    real_partial_query_graphs = robustness_probe_module.partial_query_graphs

    def _buggy_partial_query_graphs(metas, rows, fractions, seed, transform, lost_marker):
        return real_partial_query_graphs(metas, rows, fractions, seed, transform, False)

    monkeypatch.setattr(robustness_probe_module, "partial_query_graphs", _buggy_partial_query_graphs)
    w_bug = _make_world(tmp_path / "bug", monkeypatch, lost_marker=True, n_total=18,
                        eval_num_queries=3, probe_num_queries=3)

    # La forma sopravvive alla mutazione (conferma che non e' un discriminante sufficiente)...
    for f in w_bug.probe.fractions:
        for g in w_bug.probe.query_graphs[f]:
            assert g.x.shape[1] == NODE_FEATURE_DIM + LOST_MARKER_DIM
    # ...ma i valori no: con la mutazione la colonna del marcatore e' SEMPRE zero,
    # quindi l'assert `any_nonzero_on` di questo stesso test la cattura.
    any_nonzero_bug = any(
        bool((g.x[:, NODE_FEATURE_DIM] > 0).any())
        for f in w_bug.probe.fractions for g in w_bug.probe.query_graphs[f]
    )
    assert not any_nonzero_bug   # dimostra che la mutazione azzera il marcatore -> il test lo avrebbe ucciso


# ---------------------------------------------------------------------------
# T11 -- DEFAULT_PARTIAL_SEED == default di --partial-seed in graph_evaluate
# (trovato dal final-reviewer: copiato a mano in robustness_probe.py:50)
# ---------------------------------------------------------------------------


def test_default_partial_seed_matches_graph_evaluate_partial_seed_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["graph_evaluate.py"])
    eval_args = graph_evaluate.parse_args()
    assert DEFAULT_PARTIAL_SEED == eval_args.partial_seed
