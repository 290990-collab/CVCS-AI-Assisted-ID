# src/graph/evaluation/retrieval_probe.py

"""
Sonda di retrieval sul valid: misura durante il training la metrica che ci
interessa davvero (nDCG per-asse), non la loss.

Perche' esiste
--------------
Il training e' self-supervised con InfoNCE, che chiede di distinguere OGNI
pianta da tutte le altre del batch. Ma la valutazione chiede l'opposto: mettere
vicine le piante ARCHITETTONICAMENTE simili. Le due cose divergono, e non di
poco: nel benchmark del 17 lug 2026 la classifica per val-loss era l'INVERSO
della classifica per retrieval (GAT aveva la loss migliore, 2.08, ed era ultimo
per nDCG; GCN la peggiore, 2.73, ed era primo). Il motivo e' che in un batch di
256 piante ce ne sono molte con la stessa composizione/topologia della query:
InfoNCE le tratta da negativi e le allontana, cioe' ottimizzare la loss puo'
peggiorare il retrieval.

Conseguenza: scegliere il checkpoint (e fermare il training) sulla val-loss
significa usare il criterio sbagliato. Questa sonda sostituisce quel criterio
con un mini-retrieval fatto con lo STESSO protocollo della valutazione finale.

Come e' fatta (e perche' costa poco)
------------------------------------
La parte cara della valutazione non e' il forward della GNN ma costruire la
ground truth per-asse dai .mat. Qui gallery e ground truth sono FISSE: si
calcolano UNA volta alla costruzione della sonda e si riusano a ogni epoca.
Per epoca resta solo forward + prodotto scalare + accumulo metriche.

Protocollo (identico a `graph_evaluate`, in piccolo):
- gallery = sottoinsieme campionato dallo split **valid** (mai il test: il test
  si tocca solo alla fine, altrimenti la selezione degli iperparametri ci
  guarderebbe dentro);
- query   = righe della gallery stessa, con self-match escluso;
- ranking = prodotto scalare esatto su embedding L2-norm (equivalente a FAISS
  `IndexFlatIP`, che e' anch'esso brute-force esatto: a questa scala non serve
  costruire un indice);
- metriche = `accumulate_axes`, lo stesso codice della valutazione finale.

⚠️ I numeri della sonda NON sono confrontabili con quelli del report finale:
gallery piu' piccola = IDCG diverso e meno quasi-duplicati, quindi i valori
assoluti differiscono. Servono a confrontare epoche e configurazioni TRA LORO,
che e' esattamente cio' che serve per selezionare pesi e iperparametri.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.rplan_metadata import load_metadata
from src.evaluation.relevance import AXES, DISCRETE_AXES, GalleryAxes
from src.graph.evaluation.axis_metrics import accumulate_axes, mean_metric, new_metrics

# Criteri di selezione ammessi: un singolo asse, oppure la media dei tre.
SELECTION_CRITERIA = ("mean",) + AXES


class RetrievalProbe:
    """Mini-retrieval su gallery fissa: `score(encoder)` -> nDCG per-asse.

    Costruisci con `RetrievalProbe.build(...)`; l'oggetto tiene in memoria i
    grafi della gallery e la loro ground truth per-asse, entrambi immutabili.
    """

    def __init__(self, graphs, names, query_rows, k: int, batch_size: int):
        self.graphs = graphs
        self.names = names
        self.query_rows = list(query_rows)
        self.k = k
        self.batch_size = batch_size

        # Ground truth per-asse: il pezzo caro, calcolato UNA volta sola.
        self.axes = GalleryAxes([load_metadata(n) for n in names])

        # Query con metadati .mat mancanti non sono valutabili (in pratica 0:
        # i grafi senza record .mat non entrano nemmeno nel dataset).
        self.query_rows = [qi for qi in self.query_rows if self.axes.valid[qi]]

    @classmethod
    def build(
        cls,
        dataset,
        num_queries: int = 500,
        gallery_size: int = 5000,
        k: int = 10,
        seed: int = 42,
        batch_size: int = 512,
    ) -> "RetrievalProbe":
        """Campiona gallery e query dal dataset (di norma lo split valid).

        Args:
            dataset:      `RplanGraphDataset` gia' costruito, con la stessa
                          `transform` usata in training (le feature devono
                          essere quelle che l'encoder si aspetta).
            num_queries:  quante query valutare (sono righe della gallery).
            gallery_size: quanti grafi nella gallery (0 = tutto il dataset).
            k:            profondita' delle metriche (nDCG@k).
            seed:         campionamento riproducibile -> la sonda misura sempre
                          sulle stesse piante, quindi le epoche sono confrontabili.
            batch_size:   batch del forward.
        """
        pool = list(range(len(dataset)))
        rng = random.Random(seed)

        size = len(pool) if gallery_size <= 0 else min(gallery_size, len(pool))
        gallery_rows = rng.sample(pool, size)

        graphs = [dataset[i] for i in gallery_rows]
        names = [g.name for g in graphs]

        # Le query sono le prime righe della gallery: gia' in ordine casuale
        # (gallery_rows e' un campione), quindi non serve ricampionare.
        query_rows = range(min(num_queries, len(graphs)))

        return cls(graphs, names, query_rows, k=k, batch_size=batch_size)

    def __len__(self) -> int:
        return len(self.graphs)

    @property
    def num_queries(self) -> int:
        return len(self.query_rows)

    @torch.no_grad()
    def score(self, encoder, device) -> dict[str, float]:
        """Esegue il mini-retrieval e ritorna le metriche.

        Output: dict con `ndcg_<asse>` per i tre assi, `recall_<asse>`/`map_<asse>`
        per i due assi discreti, e `mean` = media dei tre nDCG (il criterio di
        selezione di default). Tutti "piu' alto = meglio".
        """
        was_training = encoder.training
        encoder.eval()

        # --- embedding della gallery (gia' L2-norm dal forward dell'encoder) ---
        loader = DataLoader(self.graphs, batch_size=self.batch_size, shuffle=False)
        emb = torch.cat([encoder(batch.to(device)) for batch in loader])

        # --- ranking: inner product esatto = cosine (embedding L2-norm) ---
        q_idx = torch.as_tensor(self.query_rows, dtype=torch.long, device=emb.device)
        sims = emb[q_idx] @ emb.t()
        # +1 perche' il self-match occupa sempre la prima posizione e va tolto.
        top = torch.topk(sims, min(self.k + 1, sims.size(1)), dim=1).indices.cpu().numpy()

        if was_training:
            encoder.train()

        # --- metriche per-asse, stesso codice della valutazione finale ---
        k_values = (self.k,)
        metrics = new_metrics(k_values)
        skipped = {ax: 0 for ax in DISCRETE_AXES}
        for j, qi in enumerate(self.query_rows):
            ret_rows = [int(r) for r in top[j] if r != qi][: self.k]
            accumulate_axes(metrics, skipped, self.axes, qi, ret_rows, k_values,
                            exclude_self=True)

        out: dict[str, float] = {}
        for ax in AXES:
            out[f"ndcg_{ax}"] = mean_metric(metrics[ax]["ndcg"][self.k])
            if ax in DISCRETE_AXES:
                out[f"recall_{ax}"] = mean_metric(metrics[ax]["recall"][self.k])
                out[f"map_{ax}"] = mean_metric(metrics[ax]["map"][self.k])

        out["mean"] = float(np.mean([out[f"ndcg_{ax}"] for ax in AXES]))
        return out

    @staticmethod
    def selection_value(scores: dict[str, float], criterion: str) -> float:
        """Estrae dallo score il singolo numero su cui selezionare i pesi.

        `criterion` = "mean" (media dei tre assi) oppure il nome di un asse.
        """
        if criterion not in SELECTION_CRITERIA:
            raise ValueError(
                f"criterio '{criterion}' non valido (scegli tra {SELECTION_CRITERIA})"
            )
        return scores["mean"] if criterion == "mean" else scores[f"ndcg_{criterion}"]

    @staticmethod
    def format_scores(scores: dict[str, float]) -> str:
        """Riga compatta per il log: nDCG dei tre assi + media."""
        return (
            f"C {scores['ndcg_composition']:.3f} "
            f"T {scores['ndcg_topology']:.3f} "
            f"G {scores['ndcg_geometry']:.3f} "
            f"| mean {scores['mean']:.4f}"
        )
