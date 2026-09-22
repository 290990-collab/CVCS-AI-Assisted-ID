# src/graph/evaluation/graph_evaluate.py

"""
Valutazione del retrieval del ramo graph: encoder GNN -> FAISS -> metriche per-asse.

Il protocollo e' identico al ramo vision (stesse metriche, stesso ambito):
1. forward dell'encoder addestrato su TUTTI i grafi della gallery (una volta
   sola: gli embedding sono deterministici a pesi congelati) -> embeddings.npy;
2. indice FAISS `IndexFlatIP` sugli embedding L2-normalizzati (inner product =
   cosine similarity, come nel vision);
3. query = un sottoinsieme di piante (di default lo split ufficiale `valid`),
   gallery = TUTTO lo snapshot; self-match escluso;
4. metriche per-asse contro l'intera gallery: nDCG@K su tutti gli assi,
   Recall@K/mAP@K sugli assi discreti (composizione/topologia). Le funzioni
   numeriche vengono dal core condiviso `src/evaluation/` (metrics.py,
   relevance.py) e NON sono duplicate ne' modificate qui.

Baseline training-free (`--baseline-hist`)
------------------------------------------
Con `--baseline-hist` l'embedding non e' la GNN ma il `type_histogram`
L2-normalizzato (13 dim, conteggio stanze per tipo, gia' allegato a ogni grafo).
Serve a validare l'idraulica FAISS+metriche prima/indipendentemente dal training
e da riferimento minimo: una GNN utile deve batterla almeno su topologia e
geometria (sulla composizione l'istogramma E' la ground truth, quindi li' la
baseline e' imbattibile per costruzione).

⚠️ Caveat di circolarita' (questione aperta, vedi CLAUDE.md): le label di
rilevanza degli assi composizione/topologia derivano da rType/rEdge, che sono
esattamente l'input della GNN (node features / edge_index). Su quegli assi il
ramo graph va letto come oracle/upper bound, non confrontato alla pari col
vision (che deve inferire tutto dai pixel). L'asse geometria e' il confronto
piu' onesto.

Uso (i flag del modello DEVONO combaciare con quelli usati in training,
altrimenti il checkpoint non si ricarica):

    python -m src.graph.evaluation.graph_evaluate --encoder sage --variant base
    python -m src.graph.evaluation.graph_evaluate --encoder gat --heads 4
    python -m src.graph.evaluation.graph_evaluate --baseline-hist

Con `--perquery-out <dir>` viene salvato anche il valore di OGNI query (un .npz
per run, formato in src/evaluation/perquery.py): serve ai confronti appaiati fra
run e fra rami. Senza il flag non viene scritto nulla e l'output e' identico.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.data.rplan_metadata import load_metadata
from src.evaluation.gallery_join import load_shared_names, restrict_rows
from src.evaluation.perquery import PerQueryRecorder, gallery_sha1
from src.evaluation.query_vectors import QueryVectorRecorder, array_sha1
from src.evaluation.relevance import DISCRETE_AXES, GalleryAxes
from src.graph.evaluation.axis_metrics import (
    accumulate_axes,
    new_metrics,
    print_axis_tables,
)
from src.graph.graph_dataset import RplanGraphDataset, VALID_SPLITS
from src.graph.graph_partial_query import make_partial_graph
from src.graph.models import build_graph_encoder
from src.graph.training.train_gnn import _default_save_dir, _encoder_kwargs
from src.graph.transforms import build_node_transform, load_geometry_stats

DEFAULT_K_VALUES = (1, 5, 10, 100)


# ----------------------------------------------------------------------
# Gallery: embedding di tutti i grafi + indice FAISS.
# ----------------------------------------------------------------------

def load_encoder(args, device):
    """Ricostruisce l'encoder con la STESSA architettura del training e ne
    ricarica i pesi da `<save_dir>/encoder.pt`.

    L'architettura viene rifatta con `_encoder_kwargs` importata dal training
    (unica fonte di verita': se cambia la' cambia anche qui), per questo i flag
    del modello devono combaciare con quelli usati per addestrare.
    """
    ckpt = Path(args.save_dir) / "encoder.pt"
    if not ckpt.exists():
        raise SystemExit(
            f"[graph_evaluate] checkpoint mancante: {ckpt}\n"
            "Lancia prima il training (scripts/graph/03_train_gnn.sh) "
            "oppure usa --baseline-hist per la baseline training-free."
        )
    encoder = build_graph_encoder(args.encoder, **_encoder_kwargs(args)).to(device)
    encoder.load_state_dict(torch.load(ckpt, map_location=device))
    encoder.eval()
    print(f"[graph_evaluate] encoder ricaricato da {ckpt}")
    return encoder


def build_transform(args):
    """Ricrea la transform di adjustment usata in training.

    Le statistiche di normalizzazione NON si ricalcolano: si ricaricano quelle
    salvate dal training (`geom_stats.npz`, calcolate sul solo train), cosi'
    l'eval vede esattamente le feature su cui l'encoder e' stato addestrato.
    """
    stats = None
    if args.normalize:
        stats_path = Path(args.save_dir) / "geom_stats.npz"
        if not stats_path.exists():
            raise SystemExit(
                f"[graph_evaluate] statistiche mancanti: {stats_path}\n"
                "Servono quelle salvate dal training (stesso save_dir); "
                "in alternativa usa --no-normalize se hai addestrato senza."
            )
        stats = load_geometry_stats(stats_path)
    return build_node_transform(
        normalize=args.normalize, drop_self_loops=args.drop_self_loops, stats=stats,
        lost_marker=getattr(args, "lost_marker", False),
    )


@torch.no_grad()
def extract_gallery(dataset, encoder, args, device) -> tuple[np.ndarray, list[str]]:
    """Forward su tutta la gallery -> (embeddings [N, D] float32, names [N]).

    Con `--baseline-hist` l'embedding e' il `type_histogram` L2-normalizzato
    invece dell'uscita della GNN (encoder ignorato). I nomi vengono letti dagli
    stessi batch degli embedding, quindi l'allineamento riga<->pianta e'
    garantito dall'ordine del loader (shuffle=False).
    """
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    chunks: list[np.ndarray] = []
    names: list[str] = []
    for batch in loader:
        if args.baseline_hist:
            emb = F.normalize(batch.type_histogram.float(), p=2, dim=1)
        else:
            emb = encoder(batch.to(device))
        chunks.append(emb.cpu().numpy())
        names.extend(batch.name)

    embeddings = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    print(f"[graph_evaluate] embedding gallery: {embeddings.shape}")
    return embeddings, names


def restrict_gallery(embeddings: np.ndarray, names: list[str],
                     gallery_names_path: str | None):
    """Restringe la gallery all'inner join fra i rami (fase **B.3**).

    Args:
        gallery_names_path: JSON prodotto da `python -m src.evaluation.gallery_join`,
            oppure None (default) = gallery intera, nessun cambiamento.

    Returns:
        (embeddings, names, row_of) — `row_of` mappa indice-dataset -> riga nuova,
        e serve a campionare le query dentro la gallery ristretta. E' None quando
        non si restringe.

    La gallery viene anche **riordinata** sull'ordine canonico del file: e'
    quell'ordine a dare lo stesso `gallery_sha1` nei due rami.
    """
    if not gallery_names_path:
        return embeddings, names, None

    shared = load_shared_names(gallery_names_path)
    rows = restrict_rows(names, shared)
    row_of = {ds_idx: new_row for new_row, ds_idx in enumerate(rows)}
    print(f"[graph_evaluate] gallery ristretta all'inner join (B.3): "
          f"{len(names)} -> {len(rows)} righe, ordine canonico da {gallery_names_path}")
    return embeddings[np.asarray(rows, dtype=np.int64)], [names[i] for i in rows], row_of


def save_gallery(save_dir: Path, embeddings: np.ndarray, names: list[str]) -> None:
    """Serializza embedding e nomi allineati per riga.

    Servono a valle per la late fusion col ramo vision (join per `name`), senza
    dover rifare il forward.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "embeddings.npy", embeddings)
    (save_dir / "names.json").write_text(json.dumps(names))
    print(f"[graph_evaluate] salvati embeddings.npy + names.json in {save_dir}")


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Indice FAISS a inner product esatto (embedding L2-norm -> cosine)."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ----------------------------------------------------------------------
# Ground truth per-asse e campionamento query (speculari al ramo vision;
# qui la riga della gallery e' identificata dal `name` del grafo).
# ----------------------------------------------------------------------

def build_gallery_axes(names: list[str]) -> GalleryAxes:
    """Feature di rilevanza per-asse dai .mat, allineate all'ordine della gallery."""
    print(f"[graph_evaluate] costruzione feature per-asse su {len(names)} piante...")
    metas = [load_metadata(name) for name in names]
    axes = GalleryAxes(metas)
    n_valid = int(axes.valid.sum())
    print(f"[graph_evaluate] metadati .mat presenti: {n_valid}/{len(names)}")
    return axes


def sample_query_rows(dataset, num_queries: int, seed: int, split: str,
                      row_of: dict[int, int] | None = None) -> list[int]:
    """Campiona righe-query in modo riproducibile. La gallery resta intera; con
    `split` != "all" le query sono ristrette a quello split ufficiale RPLAN.

    Args:
        row_of: mappa indice-dataset -> riga della gallery, presente solo quando
            la gallery e' stata ristretta all'inner join (**B.3**). Le righe
            campionate sono quelle NUOVE, e il pool e' ordinato per riga
            crescente: cosi', a parita' di seed e di ordine canonico, il ramo
            vision campiona le STESSE piante — il confronto e' appaiato anche
            sulle query, non solo sulla gallery.
    """
    if split == "all":
        pool = list(range(len(dataset)))
    else:
        pool = dataset.split_indices(split)
        print(f"[graph_evaluate] query ristrette allo split '{split}': {len(pool)} candidate")

    if row_of is not None:
        pool = sorted(row_of[i] for i in pool if i in row_of)
        print(f"[graph_evaluate] query dentro l'inner join (B.3): {len(pool)} candidate")
    n = min(num_queries, len(pool))
    rng = random.Random(seed)
    rows = rng.sample(pool, n)
    print(f"[graph_evaluate] {n} query campionate (gallery = intero snapshot)\n")
    return rows


# ----------------------------------------------------------------------
# Valutazione full (query = grafo completo). L'accumulo e la stampa delle
# metriche stanno in `axis_metrics.py`, condivisi con la sonda usata durante il
# training: cosi' la metrica che seleziona i pesi e quella che li giudica sono
# letteralmente lo stesso codice.
# ----------------------------------------------------------------------

def evaluate(index, embeddings, axes, query_rows, k_values, names=None,
             perquery=None) -> None:
    """Retrieval batched su FAISS + metriche per-asse contro l'intera gallery.

    A differenza del vision non serve rifare il forward della query: il suo
    embedding e' gia' una riga della gallery (stessa estrazione), quindi la
    ricerca si fa in un'unica chiamata FAISS su tutte le query.

    Con `perquery` (contesto da `perquery_context`) i valori della SINGOLA query
    vengono registrati accanto all'accumulo delle medie e salvati in un .npz;
    `names` serve a etichettare le query (chiave di join fra run e fra rami).
    Le medie restano quelle di sempre: l'accumulo non viene toccato.
    """
    if perquery is not None and names is None:
        raise ValueError("perquery richiede `names` (etichette delle query)")

    max_k = max(k_values)
    metrics = new_metrics(k_values)
    skipped = {ax: 0 for ax in DISCRETE_AXES}
    recorder = PerQueryRecorder(k_values, max_k) if perquery else None

    valid_rows = [qi for qi in query_rows if axes.valid[qi]]
    queries = np.ascontiguousarray(embeddings[valid_rows])
    # +1 per compensare il self-match, che viene rimosso sotto.
    _, ranked = index.search(queries, max_k + 1)

    for j, qi in enumerate(valid_rows):
        ret_rows = [int(r) for r in ranked[j] if r != qi and r != -1][:max_k]
        skipped_before = dict(skipped)   # delta = assi su cui questa query e' saltata
        accumulate_axes(metrics, skipped, axes, qi, ret_rows, k_values, exclude_self=True)
        if recorder is not None:
            recorder.add(
                name=names[qi], qi=qi, ret_rows=ret_rows, metrics=metrics,
                skipped_before=skipped_before, skipped=skipped, axes=axes,
                exclude_self=True,
            )

    print(f"[graph_evaluate] query valutate: {len(valid_rows)}")
    for ax in DISCRETE_AXES:
        if skipped[ax]:
            print(
                f"[graph_evaluate] {ax}: {skipped[ax]} query escluse da Recall/mAP "
                "(classe di equivalenza singleton, nessun rilevante)"
            )
    print()
    print_axis_tables(metrics, k_values)
    if recorder is not None:
        _write_perquery(recorder, perquery)


# ----------------------------------------------------------------------
# Valutazione PARTIAL (query = grafo degradato) — fase C.0.
# Gemello di `evaluate_partial` del ramo vision: stesse strategie, stesse
# etichette di run, stessa separazione fra self-recovery e metriche per-asse.
# ----------------------------------------------------------------------

def partial_runs(args) -> list[tuple[str, str, dict]]:
    """Espande i flag in una lista di (label, strategia, params).

    Le etichette sono **identiche** a quelle del ramo vision
    (`evaluate.py:partial_runs`): e' cio' che rende paralleli i nomi dei file
    per-query e appaiabili le due curve senza rinominare niente a mano.
    """
    runs = []
    chosen = set(args.partial_strategies)
    if "random" in chosen:
        for f in args.partial_fractions:
            runs.append((f"random f={f}", "random", {"fraction": float(f)}))
    if "semantic" in chosen:
        runs.append(("semantic", "semantic",
                     {"keep_types": list(args.partial_keep_types)}))
    if "topology" in chosen:
        runs.append(("topology", "topology",
                     {"max_degree": int(args.partial_max_degree)}))
    return runs


@torch.no_grad()
def embed_query_graphs(graphs, encoder, args, device) -> np.ndarray:
    """Grafi degradati -> embedding [Q, D], con lo STESSO percorso della gallery.

    Con `--baseline-hist` l'embedding e' il `type_histogram` L2-normalizzato,
    esattamente come in `extract_gallery`: se il partial usasse un percorso
    diverso, la differenza misurata sarebbe quella del percorso, non del masking.
    """
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    chunks = []
    for batch in loader:
        if args.baseline_hist:
            emb = F.normalize(batch.type_histogram.float(), p=2, dim=1)
        else:
            emb = encoder(batch.to(device))
        chunks.append(emb.cpu().numpy())
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def evaluate_partial(index, axes, query_rows, k_values, names, encoder, transform,
                     args, device, perquery=None, query_vectors=None) -> None:
    """Per ogni strategia/livello: grafo degradato -> retrieval -> metriche.

    Due viste, separate come impone **B.4**:
    - **self-recovery**: il grafo originale RESTA fra i risultati (ritrovarlo e'
      il compito). Endpoint del criterio A.5: `self_rr`.
    - **per-asse**: il self viene tolto, come nel full → curve confrontabili.

    ⚠️ Le query svuotate dalla strategia (nessuna stanza superstite) non sono
    valutabili: vengono contate e saltate.

    `query_vectors` (context from `query_vectors_context`, late fusion, 16 Sep
    2026): also save the damaged query vectors searched in FAISS and the
    removed rooms (`qvec/1`). None = nothing written, behaviour unchanged.
    """
    if query_vectors is not None and perquery is None:
        raise ValueError("query_vectors requires perquery (same run, same file name)")
    max_k = max(k_values)
    seed = int(args.partial_seed)

    for label, strat, params in partial_runs(args):
        metrics = new_metrics(k_values)
        skipped = {ax: 0 for ax in DISCRETE_AXES}
        selfrec = {"recall": {k: [] for k in k_values}, "rr": []}
        recorder = PerQueryRecorder(k_values, max_k, with_self_rr=True) if perquery else None

        graphs, kept_rows = [], []
        kept_removed = []   # removed rooms, parallel to kept_rows (only for query_vectors)
        n_empty = 0        # query in cui la strategia non ha rimosso nulla
        n_degenerate = 0   # query svuotate del tutto: non valutabili
        for qi in query_rows:
            if not axes.valid[qi]:
                continue
            meta = load_metadata(names[qi])
            if meta is None:
                continue
            # Stesso seed per-query del ramo vision (`seed + qi`): con la gallery
            # condivisa di B.3 il `qi` e' lo stesso, quindi si tolgono le STESSE
            # stanze. Senza B.3 i due rami degradano piante diverse.
            graph, removed = make_partial_graph(
                meta, strat, params, random.Random(seed + qi),
                # getattr: namespace costruiti a mano (test) non hanno il campo.
                lost_marker=getattr(args, "lost_marker", False),
            )
            if graph is None:
                n_degenerate += 1
                continue
            if not removed:
                n_empty += 1
            graphs.append(transform(graph) if transform is not None else graph)
            kept_rows.append(qi)
            kept_removed.append(list(removed))

        if not graphs:
            print(f"[graph_evaluate] PARTIAL [{label}]: nessuna query valutabile")
            continue

        queries = embed_query_graphs(graphs, encoder, args, device)
        _, ranked = index.search(queries, max_k + 1)

        for j, qi in enumerate(kept_rows):
            row_list = [int(r) for r in ranked[j] if r != -1]
            self_rows = row_list[:max_k]                       # self-recovery
            axis_rows = [r for r in row_list if r != qi][:max_k]  # per-asse (B.4)

            skipped_before = dict(skipped)
            accumulate_axes(metrics, skipped, axes, qi, axis_rows, k_values,
                            exclude_self=True)

            hit = [i for i, r in enumerate(self_rows, start=1) if r == qi]
            rank = hit[0] if hit else None
            selfrec["rr"].append(1.0 / rank if rank else 0.0)
            for k in k_values:
                selfrec["recall"][k].append(1.0 if (rank and rank <= k) else 0.0)

            if recorder is not None:
                recorder.add(
                    name=names[qi], qi=qi, ret_rows=axis_rows, metrics=metrics,
                    skipped_before=skipped_before, skipped=skipped, axes=axes,
                    exclude_self=True, self_rr=selfrec["rr"][-1],
                )

        print(f"################  PARTIAL [{label}]  ################")
        print(f"query valutate: {len(kept_rows)} (senza masking: {n_empty}; "
              f"svuotate e saltate: {n_degenerate})")
        mrr = float(np.mean(selfrec["rr"])) if selfrec["rr"] else float("nan")
        rec_line = "  ".join(
            f"R@{k}={float(np.mean(selfrec['recall'][k])):.3f}" for k in k_values
        )
        print(f"self-recovery (pianta originale): MRR={mrr:.3f}  {rec_line}")
        for ax in DISCRETE_AXES:
            if skipped[ax]:
                print(f"  {ax}: {skipped[ax]} query escluse da Recall/mAP (singleton)")
        print()
        print_axis_tables(metrics, k_values)
        print("ℹ️  partial (B.4): self ESCLUSO dalle metriche per-asse -> "
              "confrontabili col full; il self si misura nel self-recovery.\n")

        if recorder is not None:
            perquery_path = _write_perquery(recorder, perquery, partial_label=label)
            if query_vectors is not None:
                _write_query_vectors(query_vectors, perquery, perquery_path, names,
                                     kept_rows, kept_removed, queries, label, strat,
                                     params, seed, k_values, n_degenerate)


# ----------------------------------------------------------------------
# Persistenza dei valori per-query (opzionale: solo con --perquery-out).
# ----------------------------------------------------------------------

def perquery_context(args, tag: str, names: list[str], source) -> dict | None:
    """Contesto per il salvataggio per-query, o None se `--perquery-out` manca.

    `gallery.sha1` identifica la gallery (nomi in ordine di riga): serve a
    rifiutare confronti fra run costruite su gallery diverse, come quelle dei
    due rami (il graph perde le piante senza record .mat).
    """
    if not args.perquery_out:
        return None
    return {
        "dir": Path(args.perquery_out),
        "tag": f"{tag}_{args.variant}",
        "split": args.split,
        "seed": args.seed,
        "gallery": {"n": len(names), "sha1": gallery_sha1(names), "source": str(source)},
    }


def _label_slug(label: str) -> str:
    """Etichetta di run -> frammento di nome file ('random f=0.5' -> 'random-f0.5').

    Stessa regola del ramo vision (`evaluate.py`): i due rami devono produrre
    nomi paralleli, altrimenti appaiare le curve diventa un lavoro manuale.
    """
    return re.sub(r"[^0-9A-Za-z.]+", "-", label.replace("=", "")).strip("-")


def _write_perquery(recorder, ctx, partial_label: str | None = None) -> Path:
    """Scrive il .npz per-query: `graph_<encoder_variante>_<modalita>_<split>.npz`
    (ritorna il path scritto).

    Nel partial la modalita' include la strategia e la frazione, altrimenti le
    run della curva di masking si sovrascriverebbero (**C.0**).

    Side effects: crea la cartella e scrive il file.
    """
    mode = "full" if partial_label is None else "partial"
    slug = mode if partial_label is None else f"partial-{_label_slug(partial_label)}"
    path = ctx["dir"] / f"graph_{ctx['tag']}_{slug}_{ctx['split']}.npz"
    recorder.write(
        path,
        meta={
            "branch": "graph",
            "run_tag": ctx["tag"],
            "mode": mode,
            "partial_label": partial_label,
            "split": ctx["split"],
            "exclude_self": True,
            "query_seed": ctx["seed"],
            "gallery": ctx["gallery"],
        },
    )
    print(f"[graph_evaluate] valori per-query salvati in {path}")
    return path


def query_vectors_context(args, embeddings: np.ndarray) -> dict | None:
    """Context for saving the damaged query vectors (`qvec/1`), or None.

    Governed by `--query-vectors-out` (getattr: hand-built namespaces in the
    tests do not have the field). `embeddings` is the gallery AFTER the
    restriction to the shared names, i.e. exactly what `save_gallery` writes:
    its sha1 pins the gallery the query vectors belong to.
    """
    out_dir = getattr(args, "query_vectors_out", None)
    if not out_dir:
        return None
    return {
        "dir": Path(out_dir),
        "gallery_vectors": {
            "path": str(Path(args.save_dir) / "embeddings.npy"),
            "sha1": array_sha1(embeddings),
            "shape": list(embeddings.shape),
        },
        "model": {
            "encoder": "hist-baseline" if args.baseline_hist else args.encoder,
            "variant": args.variant,
            "raw_skip": getattr(args, "raw_skip", None),
            "lost_marker": getattr(args, "lost_marker", False),
            "normalize": getattr(args, "normalize", None),
            "drop_self_loops": getattr(args, "drop_self_loops", None),
        },
    }


def _write_query_vectors(qctx, perquery, perquery_path, names, kept_rows, kept_removed,
                         queries, label, strat, params, partial_seed, k_values,
                         n_degenerate) -> None:
    """Writes the `qvec/1` file of ONE partial run, next to its per-query file.

    `queries` [Q, D] are the vectors passed to `index.search`, row j <-> kept_rows[j].

    Side effects: creates the folder and writes the file.
    """
    recorder = QueryVectorRecorder()
    for j, qi in enumerate(kept_rows):
        recorder.add(name=names[qi], qi=qi, vector=queries[j], removed=kept_removed[j])
    path = qctx["dir"] / Path(perquery_path).name
    recorder.write(path, meta={
        "branch": "graph",
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
        "model": qctx["model"],
        "n_degenerate": int(n_degenerate),
    })
    print(f"[graph_evaluate] vettori delle query salvati in {path}")


# ----------------------------------------------------------------------
# Entrypoint.
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Valutazione retrieval del ramo graph (metriche per-asse)."
    )
    # modello: DEVONO combaciare col training per ricaricare il checkpoint.
    p.add_argument("--encoder", default="sage", choices=["gcn", "gat", "sage"])
    p.add_argument("--variant", default="base", help="etichetta per namespacing salvataggi")
    p.add_argument("--hidden-dim", type=int, default=128, dest="hidden_dim")
    p.add_argument("--out-dim", type=int, default=128, dest="out_dim")
    p.add_argument("--num-layers", type=int, default=2, dest="num_layers")
    p.add_argument("--pooling", default="add", choices=["add", "mean", "max", "mean_max"])
    p.add_argument("--dropout", type=float, default=0.0)
    # --- partial (fase C.0): query = grafo degradato ---
    p.add_argument("--partial", action="store_true",
                   help="valuta con query DEGRADATE invece che complete (C.0). "
                        "Stesse strategie e stesse etichette di run del ramo vision.")
    p.add_argument("--partial-seed", type=int, default=42, dest="partial_seed",
                   help="seed del masking; per query si usa `seed + qi`, come il "
                        "vision. ⚠️ stesse stanze rimosse nei due rami SOLO con "
                        "la gallery condivisa di B.3 (--gallery-names).")
    p.add_argument("--partial-strategies", nargs="+", default=["random"],
                   dest="partial_strategies", choices=["random", "semantic", "topology"])
    p.add_argument("--partial-fractions", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75], dest="partial_fractions",
                   help="frazioni di stanze rimosse per la strategia `random`")
    p.add_argument("--partial-keep-types", nargs="+", type=int, default=[0, 2, 3],
                   dest="partial_keep_types", help="tipi tenuti da `semantic`")
    p.add_argument("--partial-max-degree", type=int, default=1,
                   dest="partial_max_degree", help="grado massimo tolto da `topology`")
    p.add_argument("--gallery-names", default=None, dest="gallery_names",
                   help="JSON dell'inner join fra i rami (fase B.3), da "
                        "python -m src.evaluation.gallery_join. Default: gallery intera.")
    p.add_argument("--raw-skip", action="store_true", dest="raw_skip",
                   help="deve combaciare col training: cambia la forma di proj")
    p.add_argument("--no-raw-skip", action="store_false", dest="raw_skip",
                   help="per valutare le varianti allenate senza skip (es. --variant noskip)")
    p.add_argument("--lost-marker", action="store_true", dest="lost_marker",
                   help="marcatore 'vicini persi' (es. --variant asymlost); "
                        "cambia in_dim: deve combaciare col training")
    p.add_argument("--heads", type=int, default=4, help="solo GAT")
    p.add_argument("--attn-dropout", type=float, default=0.0, dest="attn_dropout", help="solo GAT")
    p.add_argument("--aggr", default="mean", help="solo SAGE (mean|max|add|lstm)")
    # adjustment (come in training)
    p.add_argument("--no-normalize", action="store_false", dest="normalize")
    p.add_argument("--keep-self-loops", action="store_false", dest="drop_self_loops")
    # baseline training-free
    p.add_argument(
        "--baseline-hist", action="store_true", dest="baseline_hist",
        help="embedding = type_histogram L2-norm (nessun checkpoint richiesto)",
    )
    # valutazione
    p.add_argument("--num-queries", type=int, default=2000, dest="num_queries")
    p.add_argument("--seed", type=int, default=42, help="seed del campionamento query")
    p.add_argument("--split", default="valid", choices=list(VALID_SPLITS) + ["all"],
                   help="split ufficiale RPLAN da cui campionare le query")
    p.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES),
                   dest="k_values")
    p.add_argument("--perquery-out", default=None, dest="perquery_out",
                   help="cartella dove salvare il valore di OGNI query "
                        "(graph_<encoder_variante>_full_<split>.npz); "
                        "default: nessuna scrittura, solo le medie a schermo")
    p.add_argument("--query-vectors-out", default=None, dest="query_vectors_out",
                   help="late fusion: folder for the damaged query vectors (qvec/1, "
                        "same file name as the per-query file). Partial only; "
                        "requires --perquery-out. Default: nothing written")
    # infra
    p.add_argument("--batch-size", type=int, default=512, dest="batch_size")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", default=None, dest="save_dir",
                   help="cartella del checkpoint/output (default: namespaced encoder/variante)")
    args = p.parse_args()
    if args.query_vectors_out and not args.perquery_out:
        p.error("--query-vectors-out richiede --perquery-out (stessa run, stesso nome file)")
    if args.lost_marker and args.baseline_hist:
        p.error("--lost-marker non si combina con --baseline-hist (niente GNN, niente transform)")
    if args.save_dir is None:
        # La baseline non ha checkpoint: namespace proprio, non quello della GNN.
        name = "hist" if args.baseline_hist else args.encoder
        args.save_dir = _default_save_dir(name, args.variant)
    return args


def main():
    args = parse_args()
    tag = "hist-baseline" if args.baseline_hist else args.encoder
    print(
        f"[graph_evaluate] encoder = {tag} | variante = {args.variant} | "
        f"pooling = {args.pooling} | lost_marker = {args.lost_marker} | "
        f"split query = {args.split} | "
        "rilevanza per-asse dai .mat (composizione/topologia/geometria).\n"
        "⚠️  composizione/topologia derivano dagli stessi rType/rEdge in input "
        "al grafo: su quegli assi il ramo graph va letto come upper bound.\n"
    )

    # Gallery = TUTTI i grafi (nessun filtro split), come nel ramo vision.
    if args.baseline_hist:
        encoder, transform = None, None      # l'istogramma non passa dalla GNN
    else:
        transform = build_transform(args)
        encoder = load_encoder(args, args.device)
    dataset = RplanGraphDataset(transform=transform)
    print(f"[graph_evaluate] gallery: {dataset}")

    embeddings, names = extract_gallery(dataset, encoder, args, args.device)
    embeddings, names, row_of = restrict_gallery(embeddings, names, args.gallery_names)
    save_gallery(Path(args.save_dir), embeddings, names)
    index = build_index(embeddings)

    axes = build_gallery_axes(names)
    query_rows = sample_query_rows(dataset, args.num_queries, args.seed, args.split,
                                   row_of=row_of)

    perquery = perquery_context(args, tag, names, dataset.snapshot_dir)

    if args.partial:
        query_vectors = query_vectors_context(args, embeddings)
        evaluate_partial(index, axes, query_rows, tuple(args.k_values), names,
                         encoder, transform, args, args.device, perquery=perquery,
                         query_vectors=query_vectors)
    else:
        if args.query_vectors_out:
            print("[graph_evaluate] --query-vectors-out ignorato in modalita' full")
        evaluate(index, embeddings, axes, query_rows, tuple(args.k_values),
                 names=names, perquery=perquery)


if __name__ == "__main__":
    main()