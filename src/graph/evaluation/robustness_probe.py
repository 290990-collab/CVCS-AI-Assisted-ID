# src/graph/evaluation/robustness_probe.py

"""
Sonda di ROBUSTEZZA sul valid: seleziona il checkpoint graph sul metro con cui
lo si giudica (self-recovery sotto masking), non sull'nDCG del full.

Perche' esiste
--------------
Dal 14 set «migliore» = piu' robusto (AUC del self-recovery). La sonda storica
(`retrieval_probe.py`) sceglie l'epoca sull'nDCG@10 di topologia della query
COMPLETA, cioe' su un metro diverso da quello di valutazione: l'epoca scelta non
e' quella piu' robusta per costruzione. Questa sonda misura, a ogni epoca, la
stessa grandezza della valutazione finale in piccolo.

Protocollo (identico a `graph_evaluate --partial`, ristretto a 2000 query):
- gallery = la gallery CONDIVISA intera (`gallery_names` di configs/graph_retrieval.yaml),
  nello stesso ordine canonico -> stesso `qi`, quindi stesso seed per-query;
- query   = righe dello split **valid** DISGIUNTE dalle query della valutazione
  (stesso campionamento di `graph_evaluate.sample_query_rows`, importato): la
  selezione non guarda le query su cui il checkpoint verra' misurato;
- degrado = `make_partial_graph(meta, "random", {"fraction": f}, Random(partial_seed + qi))`,
  la stessa chiamata della valutazione, per f in `AUC_FRACTIONS`;
- punteggio = RR del self (1/rank, 0 oltre max(k_values)), MRR per f, AUC = media su f.

Differenze dichiarate rispetto a `graph_evaluate`:
- una query svuotata in QUALUNQUE f viene tolta da TUTTE le f (la valutazione la
  toglie solo da quella f): cosi' l'AUC media MRR sulle stesse query;
- ranking = prodotto scalare esatto con «a parita' vince il self» (rank = 1 +
  righe strettamente piu' simili), come la probe del ramo vision; FAISS non
  garantisce l'ordine a parita', quindi sui pareggi esatti i due possono differire.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from src.data.rplan_metadata import load_metadata
from src.evaluation.perquery import gallery_sha1
from src.evaluation.robustness_auc import AUC_FRACTIONS
from src.graph.graph_partial_query import make_partial_graph

# Le query della sonda vengono SEMPRE dal valid: il test non sceglie nulla.
PROBE_SPLIT = "valid"
# = default di `--partial-seed` in graph_evaluate: stesse stanze tolte della valutazione.
DEFAULT_PARTIAL_SEED = 42


def probe_query_rows(valid_rows, eval_rows, n: int, seed: int) -> list[int]:
    """Righe-query della sonda: `n` righe di `valid_rows` NON in `eval_rows`.

    Il pool e' ordinato prima del campionamento: a parita' di seed e di righe
    il risultato non dipende dall'ordine in cui arrivano gli input.
    """
    pool = sorted(set(valid_rows) - set(eval_rows))
    return random.Random(seed).sample(pool, n)


def partial_query_graphs(metas, rows, fractions, seed: int, transform, lost_marker: bool):
    """Grafi degradati per ogni frazione, sulle STESSE query in tutte le f.

    Args:
        metas:     `RoomMeta` (o None) allineati a `rows`.
        rows:      righe delle query nella gallery condivisa (`qi`).
        fractions: frazioni di stanze tolte (strategia `random`).
        seed:      `partial_seed` della valutazione; per query si usa `seed + qi`.
        transform: la transform dell'encoder (None = grafi grezzi).
        lost_marker: come in `graph_evaluate.evaluate_partial`.

    Returns:
        (graphs, kept_rows, n_dropped) — `graphs[f]` e' la lista allineata a
        `kept_rows`; `n_dropped` conta le query tolte perche' svuotate (grafo
        None) in almeno una f, o senza metadati.
    """
    per_f = {f: [] for f in fractions}
    kept_rows, n_dropped = [], 0
    for meta, qi in zip(metas, rows):
        if meta is None:
            n_dropped += 1
            continue
        built = {}
        for f in fractions:
            graph, _ = make_partial_graph(
                meta, "random", {"fraction": float(f)}, random.Random(seed + qi),
                lost_marker=lost_marker,
            )
            if graph is None:
                break
            built[f] = transform(graph) if transform is not None else graph
        if len(built) != len(fractions):
            n_dropped += 1
            continue
        for f in fractions:
            per_f[f].append(built[f])
        kept_rows.append(qi)
    return per_f, kept_rows, n_dropped


@torch.no_grad()
def self_reciprocal_ranks(q: torch.Tensor, gallery: torch.Tensor, rows: torch.Tensor,
                          max_rank: int, chunk: int = 256) -> torch.Tensor:
    """RR del self per ogni query; 0 oltre `max_rank`.

    Stessa semantica di `src/vision/training/retrieval_probe.self_reciprocal_ranks`
    (duplicata: i rami restano autonomi). `q` [P, d] e `gallery` [N, d] gia'
    L2-norm; rank = 1 + righe con punteggio STRETTAMENTE maggiore del self, quindi
    a parita' di similarita' vince il self.
    """
    out = []
    for i in range(0, q.shape[0], chunk):
        s = q[i:i + chunk] @ gallery.T                                # [c, N]
        own = s.gather(1, rows[i:i + chunk, None])                    # [c, 1]
        rank = 1 + (s > own).sum(dim=1)
        out.append(torch.where(rank <= max_rank, 1.0 / rank.float(),
                               torch.zeros_like(rank, dtype=torch.float)))
    return torch.cat(out)


class RobustnessProbe:
    """Self-recovery sotto masking random su gallery condivisa fissa.

    Costruisci con `RobustnessProbe.build(...)`; `score(encoder, device)` ritorna
    `{"auc", "mrr_f0.25", "mrr_f0.5", "mrr_f0.75"}`.
    """

    def __init__(self, dataset_all, gallery_ds_rows, names, query_graphs, query_rows,
                 fractions, max_rank: int, batch_size: int, info: dict):
        self.dataset_all = dataset_all
        self.gallery_ds_rows = list(gallery_ds_rows)   # riga gallery -> indice dataset
        self.names = names
        self.query_graphs = query_graphs               # {f: [Data]} allineati a query_rows
        self.query_rows = list(query_rows)
        self.fractions = tuple(fractions)
        self.max_rank = int(max_rank)
        self.batch_size = int(batch_size)
        self.info = info
        self._gallery_batches = None                   # preparati alla prima score()
        self._query_batches = None
        self._rows_t = None

    @classmethod
    def build(cls, dataset_all, transform, eval_config_path, num_queries: int, seed: int,
              lost_marker: bool, batch_size: int) -> "RobustnessProbe":
        """Ricostruisce gallery e query della valutazione e sceglie le query sonda.

        Args:
            dataset_all: `RplanGraphDataset` INTERO (nessuno split) con `transform`.
            transform:   la stessa transform del dataset (per i grafi degradati).
            eval_config_path: configs/graph_retrieval.yaml (num_queries, seed,
                         split, k_values, gallery_names).
            num_queries: query della sonda.
            seed:        `partial_seed` della valutazione (default 42) E seed del
                         campionamento delle righe sonda.
        """
        # Import locale: graph_evaluate importa train_gnn, che importa questo modulo.
        from src.graph.evaluation import graph_evaluate

        ecfg = OmegaConf.load(eval_config_path)
        if not ecfg.get("gallery_names"):
            raise ValueError(
                f"{eval_config_path}: gallery_names e' null — la sonda di robustezza "
                "richiede la gallery condivisa (stessi qi, stessi seed per-query della valutazione)"
            )
        if dataset_all._indices is not None:
            raise ValueError("RobustnessProbe.build richiede il dataset INTERO (split=None)")

        # Stessa gallery ristretta di graph_evaluate.main: nomi in ordine di
        # dataset (quello di extract_gallery, shuffle=False), poi restrict_gallery.
        # Gli embedding non servono qui: placeholder a larghezza 0.
        ds_names = list(dataset_all._data.name)
        placeholder = np.empty((len(ds_names), 0), dtype=np.float32)
        _, names, row_of = graph_evaluate.restrict_gallery(
            placeholder, ds_names, str(ecfg.gallery_names))
        gallery_ds_rows = [0] * len(names)
        for ds_idx, row in row_of.items():
            gallery_ds_rows[row] = ds_idx

        eval_rows = graph_evaluate.sample_query_rows(
            dataset_all, int(ecfg.num_queries), int(ecfg.seed), str(ecfg.split), row_of)
        pool = sorted(row_of[i] for i in dataset_all.split_indices(PROBE_SPLIT) if i in row_of)
        rows = probe_query_rows(pool, eval_rows, num_queries, seed)
        overlap = len(set(rows) & set(eval_rows))
        assert overlap == 0, f"sonda e valutazione condividono {overlap} query"

        metas = [load_metadata(names[qi]) for qi in rows]
        fractions = tuple(float(f) for f in AUC_FRACTIONS)
        query_graphs, kept_rows, n_dropped = partial_query_graphs(
            metas, rows, fractions, seed, transform, lost_marker)

        info = {
            "gallery_n": len(names),
            "gallery_sha1": gallery_sha1(names),
            "eval_excluded": len(set(pool) & set(eval_rows)),
            "overlap": overlap,
            "n_queries": len(kept_rows),
            "n_dropped": n_dropped,
            "max_rank": int(max(ecfg.k_values)),
            "fractions": list(fractions),
            "seed": int(seed),
        }
        return cls(dataset_all, gallery_ds_rows, names, query_graphs, kept_rows,
                    fractions, info["max_rank"], batch_size, info)

    def _batches(self, graphs, device):
        return [Batch.from_data_list(graphs[i:i + self.batch_size]).to(device)
                for i in range(0, len(graphs), self.batch_size)]

    def _prepare(self, device) -> None:
        graphs = [self.dataset_all[i] for i in self.gallery_ds_rows]
        mism = sum(g.name != n for g, n in zip(graphs, self.names))
        if mism:
            raise RuntimeError(f"RobustnessProbe: {mism} grafi della gallery non allineati ai nomi")
        self._gallery_batches = self._batches(graphs, device)
        self._query_batches = {f: self._batches(self.query_graphs[f], device)
                               for f in self.fractions}
        self._rows_t = torch.as_tensor(self.query_rows, dtype=torch.long, device=device)

    @torch.no_grad()
    def score(self, encoder, device) -> dict[str, float]:
        """AUC del self-recovery (media su f del MRR) per l'encoder corrente.

        Gli embedding NON si ri-normalizzano: l'encoder emette gia' L2-norm, come in
        valutazione. Niente DataLoader: nessun consumo dell'RNG globale di torch.
        """
        was_training = encoder.training
        encoder.eval()
        if self._gallery_batches is None:
            self._prepare(device)

        gallery = torch.cat([encoder(b) for b in self._gallery_batches])
        out = {}
        for f in self.fractions:
            q = torch.cat([encoder(b) for b in self._query_batches[f]])
            rr = self_reciprocal_ranks(q, gallery, self._rows_t, self.max_rank)
            out[f"mrr_f{f}"] = float(rr.mean())

        if was_training:
            encoder.train()
        return {"auc": float(np.mean([out[f"mrr_f{f}"] for f in self.fractions])), **out}
