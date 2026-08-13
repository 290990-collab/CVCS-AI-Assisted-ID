# src/evaluation/metrics.py

"""
Metriche di retrieval *per-asse*, valutate contro l'INTERA gallery.

Il ground truth è definito su tutta la gallery, quindi le metriche
possono davvero fallire. Funzioni numeriche pure, una valutazione per
query; la media sulle query la fa `evaluate.py`.

Due famiglie:

- **graduata** (`ndcg_at_k`): usa la similarità d'asse [0,1] come *gain*. 
  L'IDCG (ordinamento ideale) è preso dai migliori K candidati sull'INTERA gallery.

- **a insieme binario** (`recall_at_k`, `average_precision_at_k`): hanno bisogno
  di un insieme di rilevanti (la classe di equivalenza esatta dell'asse).
  Normalizzano per `min(k, #rilevanti_totali)` --> mancare dei rilevanti
  costa. Ritornano `None` se la query non ha alcun rilevante (classe singleton):
  quelle query vanno escluse dalla media, non contate come zero.

⚠️ **Come si chiama davvero il numero.** Quel `min(k, ·)` ha una conseguenza da
dichiarare quando si riportano i risultati: se la query ha almeno `k` rilevanti,
il denominatore È `k` e `recall_at_k` calcola `|top-k ∩ rilevanti| / k`, cioè
**Precision@k**, non recall. Su RPLAN non è un caso limite: le classi di
equivalenza della composizione contano migliaia di piante, quindi la colonna
"Recall" di quell'asse è quasi sempre una precision. Il nome della funzione e la
chiave `"recall"` del contratto `perquery/1` restano invariati (li usano
entrambi i rami e i file già scritti), ma **tabelle e report devono dirlo**.
Quanto morde, per asse e per k, lo misura `src/evaluation/metric_diagnostics.py`
sui file per-query già su disco.
"""

import numpy as np


def _dcg(gains: np.ndarray) -> float:
    """Discounted Cumulative Gain: somma dei gain scontati per posizione
    (1/log2(rank+1)). I gain sono già nell'ordine in cui compaiono."""
    gains = np.asarray(gains, dtype=float)
    ranks = np.arange(1, len(gains) + 1)
    return float((gains / np.log2(ranks + 1)).sum())


def ndcg_at_k(retrieved_gains, gallery_gains, k: int) -> float:
    """nDCG@k con rilevanza graduata.

    Args:
        retrieved_gains: gain (similarità d'asse) dei risultati nell'ORDINE in
            cui il retriever li ha restituiti (self già escluso).
        gallery_gains:   gain di TUTTA la gallery per questa query (self escluso);
            serve a costruire l'ordinamento ideale (IDCG).
        k:               profondità.

    Returns:
        DCG(top-k recuperati) / IDCG(migliori k della gallery), in [0,1].
    """
    rg = np.asarray(retrieved_gains, dtype=float)[:k]
    dcg = _dcg(rg)

    ideal = np.sort(np.asarray(gallery_gains, dtype=float))[::-1][:k]
    idcg = _dcg(ideal)

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_relevant, num_relevant_total: int, k: int) -> float | None:
    """Recall@k **troncato** — frazione dei rilevanti trovata nel top-k.

    `|top-k ∩ rilevanti| / min(k, #rilevanti_totali)`. Il `min(k, ·)` evita che
    classi di rilevanti enormi rendano la metrica strutturalmente bassa.

    ⚠️ Due regimi, un solo nome: con `#rilevanti_totali < k` è recall vero; con
    `#rilevanti_totali >= k` il denominatore è `k` e questa è **Precision@k**.
    Sull'asse composizione siamo quasi sempre nel secondo regime (vedi il
    docstring del modulo): riportarlo come "Recall" senza dirlo è scorretto.

    Args:
        retrieved_relevant: array booleano, True dove il risultato i-esimo
            (in ordine di retrieval) è rilevante.
        num_relevant_total: quanti rilevanti esistono nella gallery (self escluso).
        k:                  profondità.

    Returns:
        Recall@k in [0,1], oppure None se non esistono rilevanti (query da escludere).
    """
    if num_relevant_total <= 0:
        return None
    rr = np.asarray(retrieved_relevant, dtype=bool)[:k]
    return float(rr.sum()) / min(k, num_relevant_total)


def average_precision_at_k(
    retrieved_relevant, num_relevant_total: int, k: int
) -> float | None:
    """AP@k — media delle precision@rank nelle posizioni dei rilevanti.

    Premia non solo il TROVARE rilevanti ma il metterli in ALTO. Normalizzata
    per `min(k, #rilevanti_totali)` (coerente con recall_at_k). Ritorna None se
    non ci sono rilevanti.

    Lo stesso `min(k, ·)` di `recall_at_k`: con classi di equivalenza più grandi
    di `k` il denominatore satura a `k` e il valore massimo raggiungibile è 1
    solo se TUTTE le prime k posizioni sono rilevanti.
    """
    if num_relevant_total <= 0:
        return None
    rr = np.asarray(retrieved_relevant, dtype=bool)[:k]

    hits = 0
    precision_sum = 0.0
    for rank, is_rel in enumerate(rr, start=1):
        if is_rel:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(k, num_relevant_total)
