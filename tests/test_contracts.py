# tests/test_contracts.py

"""
Smoke test CPU dei QUATTRO CONTRATTI critici (fase B.5 della roadmap).

Sono i quattro punti che, se si rompono in silenzio, invalidano i numeri senza
far fallire nessuna run:

  1. riga `i` di `embeddings.npy` <-> elemento `i` di `image_paths.json`
     (vision) / `names.json` (graph). E' l'interfaccia della late fusion.
  2. forma dell'architettura <-> checkpoint: un flag che cambia `proj`
     (`raw_skip`, `pooling="mean_max"`) rende il checkpoint non ricaricabile, e
     il rifiuto deve essere esplicito, non un silenzioso caricamento parziale.
  3. valori di riferimento delle metriche (`ndcg_at_k`, `recall_at_k`,
     `average_precision_at_k`) calcolati a mano su casi minuscoli.
  4. split disgiunti (le query di valid e test non si toccano mai) e statistiche
     che dipendono SOLO dalle righe su cui sono fittate.

Tutto CPU, deterministico, senza dataset reale (.mat) ne' pesi scaricati.

Il blocco 4-bis (25 ago, fase **B.2**) copre anche il CHIAMANTE: che
`prepare_index` stimi il whitening sulle sole righe di train e indicizzi
comunque la gallery intera.

Esecuzione: python -m pytest tests/test_contracts.py -v
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
# la gallery sintetica canonica vive gia' in test_perquery: riusarla evita due
# verita' diverse sugli stessi RoomMeta
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


# ----------------------------------------------------------------------
# Helper: embedding "parlanti". La riga i ha come prima componente i, cosi'
# qualunque riordino delle righe e' visibile a occhio nudo in un assert.
# ----------------------------------------------------------------------

def labelled_embeddings(n: int, dim: int = 4) -> np.ndarray:
    embs = np.zeros((n, dim), dtype="float32")
    embs[:, 0] = np.arange(n, dtype="float32")
    embs[:, 1] = np.arange(n, dtype="float32") * 0.5
    return embs


def row_label(embs: np.ndarray, row: int) -> int:
    """Recupera l'indice codificato nella riga: se e' != row, l'allineamento
    e' saltato."""
    return int(round(float(embs[row, 0])))


# ======================================================================
# CONTRATTO 1 — riga <-> nome, sui due rami e attraverso salva/ricarica.
# ======================================================================

def test_graph_save_gallery_preserves_row_name_alignment(tmp_path):
    """`save_gallery` scrive due file separati: se l'ordine di uno dei due
    cambiasse, il join per nome della late fusion accoppierebbe piante diverse."""
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
    """Stesso contratto sul ramo vision, dove i nomi sono path di PNG e la
    chiave del join e' lo `stem`."""
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
    # stem -> riga: e' la mappa che evaluate.py usa per tradurre i risultati
    # FAISS in nomi. Deve puntare alla riga che porta lo stesso indice.
    stem2row = {Path(p).stem: i for i, p in enumerate(loaded.image_paths)}
    for stem, row in stem2row.items():
        assert int(stem) == row_label(loaded.raw_embeddings, row)


def test_row_name_join_catches_a_shuffled_gallery(tmp_path):
    """Contro-prova: se i nomi vengono salvati in un ordine diverso dalle righe,
    il join per nome punta a righe sbagliate. Serve a dimostrare che gli assert
    qui sopra sono in grado di vedere il guasto, non solo di passare."""
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


# ======================================================================
# CONTRATTO 2 — forma dell'architettura <-> checkpoint (`raw_skip`, pooling).
# ======================================================================

ENCODER_KWARGS = dict(in_dim=19, hidden_dim=16, out_dim=8, num_layers=2)


def test_raw_skip_changes_the_projection_input_width():
    """`raw_skip` concatena l'add-pool delle feature grezze: `proj` cresce
    esattamente di `in_dim`. E' il flag che rende due checkpoint incompatibili."""
    plain = build_graph_encoder("gcn", pooling="add", raw_skip=False, **ENCODER_KWARGS)
    skip = build_graph_encoder("gcn", pooling="add", raw_skip=True, **ENCODER_KWARGS)

    assert plain.proj.in_features == ENCODER_KWARGS["hidden_dim"]
    assert skip.proj.in_features == ENCODER_KWARGS["hidden_dim"] + ENCODER_KWARGS["in_dim"]


def test_mean_max_pooling_doubles_the_projection_input_width():
    """Secondo flag con lo stesso effetto: va testato insieme a `raw_skip`,
    altrimenti si protegge un solo modo di rompere il checkpoint."""
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
    """Un checkpoint allenato con una forma non deve caricarsi in un'altra: il
    caricamento parziale e' il modo silenzioso di produrre numeri sbagliati.

    ⚠️ Oggi il rifiuto arriva dal `RuntimeError` di PyTorch (rilievo B5: errore
    criptico, nessun controllo esplicito). Il test fissa il COMPORTAMENTO
    osservabile — rifiuto, con `proj` nominato nel messaggio — non il testo."""
    trained = build_graph_encoder("gcn", **kwargs_a, **ENCODER_KWARGS)
    other = build_graph_encoder("gcn", **kwargs_b, **ENCODER_KWARGS)

    with pytest.raises(RuntimeError) as err:
        other.load_state_dict(trained.state_dict())
    assert "proj" in str(err.value)


def test_checkpoint_of_the_same_shape_reloads_identically():
    """Il verso positivo: stessa forma -> ricarica accettata e forward
    bit-identico. Senza questo, il test sopra passerebbe anche se NESSUN
    checkpoint fosse ricaricabile."""
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


# ======================================================================
# CONTRATTO 3 — valori di riferimento delle metriche, calcolati a mano.
# ======================================================================

def test_ndcg_matches_hand_computation():
    """gain recuperati [1, 0, 1] su una gallery con due rilevanti.

    DCG  = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0.5          = 1.5
    IDCG = ordine ideale [1, 1, 0] = 1 + 1/log2(3) + 0          = 1.63093
    nDCG = 1.5 / 1.63093                                        = 0.91972
    """
    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3)
    expected = dcg / idcg

    got = ndcg_at_k([1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0], k=3)

    assert got == pytest.approx(expected, abs=1e-12)
    assert got == pytest.approx(0.919721, abs=1e-6)


def test_ndcg_edge_cases():
    """Ordine ideale -> 1. Gallery senza rilevanti (IDCG 0) -> 0, non NaN."""
    assert ndcg_at_k([1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], k=3) == pytest.approx(1.0)
    assert ndcg_at_k([0.0, 0.0], [0.0, 0.0, 0.0], k=2) == 0.0


def test_recall_has_two_regimes_and_a_none():
    """Il denominatore `min(k, #rilevanti)` da' due regimi con lo stesso nome.

    - pochi rilevanti (2 < k=5): 2 trovati su 2 -> recall vero = 1.0
    - molti rilevanti (10 >= k=5): 2 trovati su k -> e' Precision@5 = 0.4
    - nessun rilevante -> None (la query va esclusa, non contata 0)
    """
    retrieved = [True, False, True, False, False]

    assert recall_at_k(retrieved, num_relevant_total=2, k=5) == pytest.approx(1.0)
    assert recall_at_k(retrieved, num_relevant_total=10, k=5) == pytest.approx(0.4)
    assert recall_at_k(retrieved, num_relevant_total=0, k=5) is None


def test_average_precision_matches_hand_computation():
    """rilevanti in posizione 1 e 3, 2 rilevanti totali, k=3.

    AP = (1/1 + 2/3) / min(3, 2) = 1.66667 / 2 = 0.83333
    """
    got = average_precision_at_k([True, False, True], num_relevant_total=2, k=3)

    assert got == pytest.approx((1.0 + 2.0 / 3.0) / 2.0, abs=1e-12)
    assert got == pytest.approx(0.833333, abs=1e-6)
    assert average_precision_at_k([True], num_relevant_total=0, k=1) is None


def test_average_precision_rewards_early_ranks():
    """La proprieta' che distingue AP da Recall: stessi rilevanti trovati, ordine
    diverso -> punteggio diverso. Se questa cade, AP e Recall sono la stessa cosa."""
    early = average_precision_at_k([True, True, False, False], 2, k=4)
    late = average_precision_at_k([False, False, True, True], 2, k=4)

    assert early == pytest.approx(1.0)
    assert late < early


# ======================================================================
# CONTRATTO 4 — split disgiunti e statistiche dalle sole righe passate.
# ======================================================================

FAKE_SPLITS = {
    **{f"/fake/{i}.png": "train" for i in range(0, 20)},
    **{f"/fake/{i}.png": "valid" for i in range(20, 30)},
    **{f"/fake/{i}.png": "test" for i in range(30, 40)},
}
FAKE_PATHS = list(FAKE_SPLITS)


def test_vision_query_pools_of_valid_and_test_never_overlap(monkeypatch):
    """La gallery resta intera: e' lo SPLIT che restringe le query. Se i due pool
    si toccassero, il vincolo DURO 1 (il test non sceglie nulla) cadrebbe senza
    che nessuna run se ne accorga."""
    monkeypatch.setattr(
        "src.vision.evaluation.evaluate.get_split", lambda p: FAKE_SPLITS[p]
    )

    valid_rows = sample_query_rows(FAKE_PATHS, num_queries=10, seed=42, split="valid")
    test_rows = sample_query_rows(FAKE_PATHS, num_queries=10, seed=42, split="test")

    assert set(valid_rows).isdisjoint(test_rows)
    assert all(FAKE_SPLITS[FAKE_PATHS[i]] == "valid" for i in valid_rows)
    assert all(FAKE_SPLITS[FAKE_PATHS[i]] == "test" for i in test_rows)
    # senza `split` il pool e' tutto: la gallery non viene mai ristretta
    assert len(sample_query_rows(FAKE_PATHS, num_queries=40, seed=42)) == len(FAKE_PATHS)


def test_vision_query_sampling_is_reproducible(monkeypatch):
    """Stesso seed -> stesse query, nello stesso ordine: e' la condizione che
    rende appaiati i confronti fra due run."""
    monkeypatch.setattr(
        "src.vision.evaluation.evaluate.get_split", lambda p: FAKE_SPLITS[p]
    )

    a = sample_query_rows(FAKE_PATHS, num_queries=8, seed=42, split="valid")
    b = sample_query_rows(FAKE_PATHS, num_queries=8, seed=42, split="valid")
    c = sample_query_rows(FAKE_PATHS, num_queries=8, seed=7, split="valid")

    assert a == b
    assert a != c


def test_graph_split_indices_partition_the_dataset():
    """Stesso invariante sul ramo graph, dove lo split e' una colonna del
    dataset collato invece di un lookup nei `.mat`."""
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
    """L'invariante che rende sensato «statistiche dal solo train»: fittare su un
    sottoinsieme deve dare gli stessi parametri, che quel sottoinsieme arrivi da
    solo o estratto da una matrice piu' grande."""
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

    # e le statistiche di tutta la matrice sono diverse: il sottoinsieme conta
    full = SimpleNamespace()
    VisionRetrievalPipeline._fit_whitening(full, everything)
    assert not np.allclose(alone.whiten_mean, full.whiten_mean)


def test_train_only_stats_leave_a_nonzero_mean_on_heldout():
    """Firma del leakage (testing-guide § 4): statistiche fittate sul train e
    applicate a righe mai viste devono lasciare una media VICINA ma non
    esattamente 0. Una media esattamente 0 significa che il held-out era dentro
    il fit."""
    rng = np.random.default_rng(1)
    train = rng.normal(size=(128, 8)).astype("float32")
    heldout = rng.normal(size=(32, 8)).astype("float32")

    fitted = SimpleNamespace()
    VisionRetrievalPipeline._fit_whitening(fitted, train)

    train_centered_mean = np.abs((train - fitted.whiten_mean).mean(axis=0)).max()
    heldout_centered_mean = np.abs((heldout - fitted.whiten_mean).mean(axis=0)).max()

    assert train_centered_mean < 1e-6
    assert heldout_centered_mean > 1e-6
    assert heldout_centered_mean < 1.0   # vicina: stessa distribuzione, non leakage


# ======================================================================
# CONTRATTO 4-bis — B.2: il whitening si STIMA sul solo train (25 ago).
# Qui si testa il CHIAMANTE, che prima del 25 ago era fuori copertura:
# `prepare_index(fit_rows=...)` deve stimare sulle sole righe indicate e
# indicizzare comunque la gallery INTERA.
# ======================================================================

class _StubPipeline(VisionRetrievalPipeline):
    """Pipeline senza encoder ne' transform: serve solo il percorso
    raw -> head -> whitening -> L2 -> indice. Evita di scaricare pesi."""

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
    """Gallery mista come `snapshot_train/`: train + non-train nella stessa
    matrice, con il non-train spostato di media (se le statistiche lo vedono,
    si nota)."""
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

    # stima identica a quella fatta sulle sole righe di train...
    assert np.array_equal(train_only.whiten_mean, reference.whiten_mean)
    assert np.array_equal(train_only.whiten_matrix, reference.whiten_matrix)
    # ...e diversa dal trasduttivo: il protocollo cambia davvero i numeri
    assert not np.allclose(train_only.whiten_mean, transductive.whiten_mean)


def test_prepare_index_keeps_the_whole_gallery_indexed():
    """Vincolo DURO 2: lo split restringe la stima, MAI il corpus di ricerca."""
    embs, paths, train_rows = _gallery_with_two_splits()

    pipeline = _StubPipeline(embs.copy(), paths)
    pipeline.prepare_index(whiten=True, fit_rows=train_rows)

    assert len(pipeline.embeddings) == len(embs)
    assert pipeline.index.ntotal == len(embs)
    assert len(train_rows) < len(embs)


def test_prepare_index_rejects_an_empty_fit_set():
    """Un `fit_split` che non seleziona nulla (metadati mancanti) deve fermare
    la run, non produrre statistiche su zero righe."""
    embs, paths, _ = _gallery_with_two_splits()
    pipeline = _StubPipeline(embs, paths)

    with pytest.raises(ValueError, match="fit_rows"):
        pipeline.prepare_index(whiten=True, fit_rows=[])


def test_split_row_indices_selects_rows_not_names(monkeypatch):
    """`split_row_indices` deve restituire INDICI DI RIGA della gallery: se
    tornasse nomi o posizioni relative, il whitening stimerebbe sulle piante
    sbagliate senza fallire."""
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
    """Il ponte config -> righe: `train` restringe, `all` resta trasduttivo,
    whitening spento non calcola niente."""
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
    """Il tag finisce nel NOME dei file per-query: se non distinguesse il
    protocollo, una run train-only sovrascriverebbe la run trasduttiva con lo
    stesso encoder — e le 320 run di §24 sparirebbero senza errori."""
    def cfg(fit_split, dim=None, head=False):
        return OmegaConf.create({
            "whitening": {"enabled": True, "fit_split": fit_split, "dim": dim},
            "head": {"enabled": head},
        })

    assert transform_tag(cfg("all")) == "whiten"
    assert transform_tag(cfg("train")) == "whiten-train"
    assert transform_tag(cfg("train", dim=768)) == "whiten768-train"
    assert transform_tag(cfg("train", head=True)) == "head+whiten-train"
    # whitening spento: il protocollo non c'entra, il tag non cambia
    off = OmegaConf.create({"whitening": {"enabled": False, "fit_split": "train"},
                            "head": {"enabled": False}})
    assert transform_tag(off) == "raw"


# ======================================================================
# CONTRATTO 5 — B.4: nel partial, due liste per due domande (25 ago).
# Il self resta nel self-recovery e sparisce dalle metriche per-asse: senza
# questa separazione la curva di degrado non e' confrontabile col full.
# ======================================================================

def _faiss_results(rows, stem2path):
    """Risposte FAISS finte: solo il campo `path`, l'unico che il codice legge."""
    return [{"path": stem2path[r]} for r in rows]


def test_partial_rows_splits_self_recovery_from_axis_metrics():
    gallery = [f"/fake/{i}.png" for i in range(20)]
    stem2row = {Path(p).stem: i for i, p in enumerate(gallery)}
    stem2path = {i: p for i, p in enumerate(gallery)}
    qi = 7
    # il self esce SECONDO: e' il caso che distingue le due viste
    ranked = [3, qi, 5, 9, 11, 2]

    self_rows, axis_rows = partial_rows(
        _faiss_results(ranked, stem2path), qi, stem2row, max_k=5
    )

    assert qi in self_rows            # self-recovery: il self e' il bersaglio
    assert self_rows == ranked[:5]
    assert qi not in axis_rows        # per-asse: il self non c'e' (come nel full)
    assert axis_rows == [3, 5, 9, 11, 2]
    # togliere il self NON accorcia la lista: si scorre nella k+1-esima risposta
    assert len(axis_rows) == len(self_rows) == 5


def test_partial_rows_are_identical_when_the_self_is_not_retrieved():
    """Se il self non compare (masking pesante), le due viste coincidono: il fix
    non deve introdurre differenze dove non ce n'e' ragione."""
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
    """Contro-prova quantitativa: con il self dentro, la query recupera se
    stessa (similarita' massima su ogni asse) e i punteggi si gonfiano. E' il
    motivo per cui il partial non era confrontabile col full."""
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    qi = 1                      # p1: ha un gemello (p2) su composizione e topologia
    k_values = (1, 3)

    with_self = vision_new_metrics(k_values)
    vision_accumulate_axes(with_self, {ax: 0 for ax in DISCRETE_AXES}, axes, qi,
                           [qi, 2, 0], k_values, exclude_self=False)
    without_self = vision_new_metrics(k_values)
    vision_accumulate_axes(without_self, {ax: 0 for ax in DISCRETE_AXES}, axes, qi,
                           [2, 0, 3], k_values, exclude_self=True)

    assert with_self["composition"]["ndcg"][1][0] == pytest.approx(1.0)
    assert without_self["composition"]["ndcg"][1][0] <= with_self["composition"]["ndcg"][1][0]
    # e il numero di rilevanti cambia: il self era contato
    assert (count_relevant(axes, "composition", qi, exclude_self=False)
            == count_relevant(axes, "composition", qi, exclude_self=True) + 1)


def test_visualization_metric_line_drops_the_self_from_retrieved_rows():
    """La riga stampata sotto una figura partial deve usare la convenzione delle
    tabelle: se il self restasse fra i recuperati mentre esce dai rilevanti,
    verrebbe contato come un errore e la figura direbbe il falso."""
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    qi = 1
    stem2row = {n: i for i, n in enumerate(names)}
    results = [{"path": f"/fake/{names[r]}.png"} for r in (qi, 2, 0)]
    sims = {ax: np.stack([axes.sim(ax, qi)]).ravel() for ax in AXES}

    with_self = _metrics_summary(sims, results, qi, axes, stem2row, exclude_self=False)
    without_self = _metrics_summary(sims, results, qi, axes, stem2row, exclude_self=True)

    assert with_self != without_self
    assert "n/a" not in without_self.split("|")[0]   # la query non e' singleton


# ======================================================================
# CONTRATTO 6 — B.3: gallery condivisa fra i rami (25 ago).
# Non basta "quasi la stessa": due run sono appaiate solo se hanno lo STESSO
# gallery_sha1, che dipende dall'ORDINE delle righe.
# ======================================================================

SHARED_SPLITS = {f"p{i:02d}": ("valid" if i % 3 == 0 else "train") for i in range(12)}


def _two_branch_galleries():
    """Le due gallery come sono davvero: stessi nomi al centro, code diverse,
    ordini diversi, formati diversi (path PNG vs nomi nudi)."""
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
    """Il cuore di B.3: dopo la restrizione i due rami hanno la stessa lista
    NELLO STESSO ORDINE, quindi lo stesso hash — la condizione che
    `check_compatible` verifica per accettare un confronto appaiato."""
    vision_entries, graph_entries, _ = _two_branch_galleries()
    vn, gn = canonical_names(vision_entries), canonical_names(graph_entries)

    shared = compute_shared_names(vn, gn)
    vision_rows = restrict_rows(vn, shared)
    graph_rows = restrict_rows(gn, shared)

    vision_after = [vn[i] for i in vision_rows]
    graph_after = [gn[i] for i in graph_rows]

    assert vision_after == graph_after == shared
    assert gallery_sha1(vision_after) == gallery_sha1(graph_after)
    # ...e prima della restrizione NON lo era: il fix serve davvero
    assert gallery_sha1(vn) != gallery_sha1(gn)


def test_restrict_rows_refuses_a_gallery_that_misses_a_shared_name():
    """Se il file dell'inner join non appartiene a questa gallery, meglio
    fermarsi che valutare su un corpus diverso da quello dichiarato."""
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
    payload["names"] = payload["names"][:-1]        # qualcuno taglia una riga a mano
    out.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sha1"):
        load_shared_names(out)


class _StubGraphDataset:
    """Dataset finto: al campionamento servono solo la lunghezza e gli split."""

    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def split_indices(self, split):
        # i nomi solo-graph esistono nel dataset ma non nell'inner join: hanno
        # comunque uno split, ed e' giusto che il filtro li veda e li scarti
        return [i for i, n in enumerate(self.names)
                if SHARED_SPLITS.get(n, "train") == split]


def test_both_branches_sample_the_same_queries_after_the_join(monkeypatch):
    """Conseguenza pratica di B.3, ed e' quella che serve al report: con la
    stessa gallery canonica, stesso seed e stesso split, i due rami campionano
    le STESSE piante. Senza l'ordine canonico questo non accadrebbe."""
    vision_entries, graph_entries, _ = _two_branch_galleries()
    vn, gn = canonical_names(vision_entries), canonical_names(graph_entries)
    shared = compute_shared_names(vn, gn)

    # vision: la restrizione riordina i path; le query si campionano dopo
    vision_paths = [vision_entries[i] for i in restrict_rows(vn, shared)]
    monkeypatch.setattr(
        "src.vision.evaluation.evaluate.get_split",
        lambda p: SHARED_SPLITS[Path(p).stem],
    )
    vision_rows = sample_query_rows(vision_paths, num_queries=3, seed=42, split="valid")

    # graph: la restrizione produce la mappa indice-dataset -> riga nuova
    dataset = _StubGraphDataset(gn)
    row_of = {ds: new for new, ds in enumerate(restrict_rows(gn, shared))}
    graph_rows = graph_sample_query_rows(dataset, num_queries=3, seed=42,
                                         split="valid", row_of=row_of)

    assert vision_rows == graph_rows
    assert [shared[i] for i in vision_rows] == [shared[i] for i in graph_rows]
    assert all(SHARED_SPLITS[shared[i]] == "valid" for i in vision_rows)


# ======================================================================
# CONTRATTO 7 — C.0: la query di grafo degradata (25 ago).
# Il partial del ramo graph deve togliere LE STESSE stanze del vision e non
# perturbare quelle che restano.
# ======================================================================

def _full_meta():
    """Pianta a 4 stanze, catena 0-1-2-3: la topologia rende visibile un
    rimappamento sbagliato degli archi."""
    return RoomMeta(
        name="p42", split="valid",
        room_types=(0, 1, 2, 3),
        edges=((0, 1, 1), (1, 2, 1), (2, 3, 1)),
        boxes=((0, 0, 20, 20), (20, 0, 40, 20), (0, 20, 20, 40), (20, 20, 40, 40)),
        footprint=(0, 0, 40, 40), entrance=None,
    )


def test_filter_meta_remaps_edges_and_drops_the_ones_that_touch_a_removed_room():
    meta = _full_meta()

    reduced = filter_meta(meta, [1])          # la catena 0-1-2-3 perde il nodo 1

    assert reduced.room_types == (0, 2, 3)
    # l'arco 2-3 sopravvive e diventa 1-2; 0-1 e 1-2 spariscono con il nodo 1
    assert reduced.edges == ((1, 2, 1),)
    assert reduced.boxes == (meta.boxes[0], meta.boxes[2], meta.boxes[3])
    # invarianti dichiarati: identita' e footprint della pianta INTERA
    assert reduced.name == meta.name and reduced.split == meta.split
    assert reduced.footprint == meta.footprint


def test_filter_meta_returns_none_when_nothing_survives():
    assert filter_meta(_full_meta(), [0, 1, 2, 3]) is None


def test_partial_graph_keeps_the_surviving_nodes_bit_identical():
    """L'invariante che rende leggibile il degrado: togliere una stanza NON
    cambia le feature delle altre (sono normalizzate sulla griglia 256 fissa).
    Se cambiassero, la curva misurerebbe anche una perturbazione spuria."""
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
    """Il cuore di C.0: a parita' di pianta, seed e strategia, i due rami
    tolgono LE STESSE stanze. Senza questo, il confronto sotto masking
    misurerebbe il masking invece dei modelli."""
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
        ("semantic", {"keep_types": [0, 1]}, [2, 3]),   # tiene solo i tipi 0 e 1
        ("topology", {"max_degree": 1}, [0, 3]),        # toglie le due foglie
        ("random", {"fraction": 0.0}, []),              # nessuna rimozione
    ],
)
def test_partial_graph_supports_every_vision_strategy(strategy, params, expected_removed):
    """Le tre strategie del vision devono funzionare identiche sul grafo,
    altrimenti la curva di degrado non e' confrontabile per strategia."""
    meta = _full_meta()

    graph, removed = make_partial_graph(meta, strategy, params, random.Random(1))

    assert removed == expected_removed
    assert graph.x.shape[0] == meta.num_rooms - len(removed)
    # ogni arco superstite punta a nodi esistenti
    if graph.edge_index.numel():
        assert int(graph.edge_index.max()) < graph.x.shape[0]


def test_the_two_branches_expand_the_same_partial_runs():
    """Stesse etichette di run nei due rami: e' quello che rende paralleli i nomi
    dei file per-query, quindi appaiabili le due curve senza rinominare a mano."""
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
    """Smoke dell'intero giro C.0 su una gallery sintetica: grafi degradati ->
    encoder -> FAISS -> due viste -> .npz. Serve a scoprire subito se il loop si
    rompe, senza aspettare un job da ore."""
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
    assert data.meta["exclude_self"] is True     # B.4 vale anche qui
    assert data.self_rr is not None and len(data.self_rr) == len(metas)
