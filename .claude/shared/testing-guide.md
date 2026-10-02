# Guida ai test — smoke test CPU e invarianti

Qui i test **non** misurano quanto è bravo il modello (lo dicono le metriche,
dopo un job sbatch): verificano che la **pipeline sia corretta e riproducibile**.
Un test da 10 secondi su CPU che cattura un disallineamento indice↔nome vale più
di una run da 3 ore.

## Struttura ed esecuzione

- I test stanno in `tests/`, si lanciano con `python -m tests.<nome>` dopo
  `source ~/floorplan-env/bin/activate`.
- ⚠️ **`tests/test_vision_retrieval.py` non è un test**: è l'entrypoint storico
  di **indicizzazione** del ramo vision (gira su GPU sul cluster). Non
  trasformarlo in un test e non lanciarlo per "provare".
- Un test valido qui: **CPU**, pochi campioni, seed fisso, nessun download di
  pesi, nessuna scrittura in `embeddings/` reali (usare una directory
  temporanea).
- **Suite di regressione rapida** (10 set: **96 passed**, ~27 s), l'unica che
  gira senza GPU né dataset intero:

  ```bash
  python -m pytest tests/test_perquery.py tests/test_metric_diagnostics.py \
                   tests/test_loader.py tests/test_contracts.py \
                   tests/test_robustness_auc.py tests/test_head_probe.py \
                   tests/test_graph_asym_pairs.py tests/test_graph_lost_marker.py -q
  ```

  `test_graph_lost_marker.py` (14 set, 18 test): marcatore «vicini persi» di `asymlost` — conteggio a
  mano, fedeltà training↔valutazione, zero sulla pianta intera, default a 19 colonne, guardie, forma
  del checkpoint (19 vs 20).

  `test_head_damage.py` (15 set, 11 test): `training.damage` della head — `random` identico al render
  storico e alla chiave di cache già su disco, `nowalls_random` toglie le stesse stanze con l'rng allineato,
  probe che registra il danno, training che rifiuta coppie/probe/config discordi. Suite completa senza
  `test_vision_retrieval.py`/`test_vision_encoder.py`: **196 passed** (15 set).

  `test_query_vectors.py` + `test_late_fusion.py` (16 set): contratto `qvec/1`, default bit-identici delle
  due valutazioni, α=1/α=0 ≡ rami, rifiuti duri (stanze, sha1, whitening non train), regole di `select`
  di §49, controllo C3 «stesso job», determinismo. Suite completa senza `test_vision_retrieval.py`:
  **260 passed** (16 set, `python -m pytest`).
  `test_fusion_graphgraph.py` (17 set, §50): β=1/β=0 ≡ i due graph, rifiuti (stanze, sha1, stessa run, cartelle miste), C5 solo limite basso, `complementarity` (join per nome, tre esiti, componente migliore, determinismo). Suite completa: **276 passed** (17 set).
  `test_figures.py` (18 set, §54, 19 test): figure del report — curva appaiata sulle sole query comuni, AUC
  che esclude f=0.0, guardia «il json di `fusion_select` e i per-query devono dire lo stesso numero»,
  provenienza salvata accanto a ogni figura, larghezze a due colonne, classifica del teaser ricostruita
  reinserendo la query al proprio rango, regola di scelta della query, appaiamento fra danni diversi,
  guardia «due sistemi devono contare le stesse classi», raggruppamento delle ablation per encoder.
  Suite completa: **316 passed** (18 set).
  `test_fusion_visionvision.py` (17 set, §51): γ=1/γ=0 ≡ i due vision col proprio whitening, rifiuti (stanze, sha1, fit_split, head, stessa run, cartelle miste), C5 solo basso, regola a quattro esiti (confine ±0.04), Spearman, G_C letto dal json di §50. Suite completa: **297 passed** (17 set).

  Gli ultimi tre (10 set) coprono l'AUC di A.5 (`robustness_auc`), la probe
  partial della head (B.6) e le coppie asimmetriche del graph (`pair_mode`,
  default bit-identico allo storico).

  `tests/test_contracts.py` (25 ago, **fase B.5**, 18 test) copre i **quattro
  contratti critici**: riga↔nome su entrambi i rami · forma dell'architettura↔
  checkpoint (`raw_skip`, `pooling="mean_max"`) · valori di riferimento di
  nDCG/Recall/AP calcolati a mano · split disgiunti e statistiche che dipendono
  **solo** dalle righe passate. ⚠️ Non copre che il *chiamante* del whitening
  passi le sole righe di train: quello è B.2.

  ⚠️ `test_vision_encoder.py` **non** è incluso: scarica i pesi dei backbone.
  I 4 test più recenti (24 ago) coprono `self_rr`: presenza del campo solo in
  `mode="partial"`, delta appaiato verificato **a mano** su valori noti, e
  rifiuto esplicito quando un file non è partial.

## Cosa si testa (in ordine di valore)

1. **Invarianti numeriche** — embedding L2-normalizzati (norma ≈ 1); shape
   `[B, D]` / `[num_grafi, out_dim]`; gradienti che fluiscono dove devono
   (e **non** dove il modello è frozen); singolo grafo `batch=None` → `[1, D]`.
2. **Contratti su disco** — riga `i` di `embeddings.npy` ↔ riga `i` di
   `names.json`; `shuffle=False` rispettato; salva → ricarica → stessi valori;
   un checkpoint si ricarica **solo** con la stessa forma di architettura e
   viene **rifiutato** con una forma diversa (è il test che protegge dal flag
   dimenticato in un ponte YAML→flag).
3. **Determinismo e cache** — stesso seed → stesso output; una cache restituisce
   bit-identico ciò che ricalcolerebbe (chiave della cache = tutti i parametri
   che influenzano il contenuto).
4. **Assenza di leakage** — statistiche calcolate sul train e applicate al test
   danno media **vicina ma non esattamente** 0; una media esattamente 0 è la
   firma di un leakage.
5. **Invarianze dichiarate dalle augmentation** — se un'augmentation dichiara
   un'invarianza, il test la verifica *esattamente*: rotazione e riflessione non
   cambiano area/aspect né one-hot né archi; 4 rotazioni = identità;
   la decisione è per-grafo, non per-nodo.
6. **Trasformazioni della pipeline** — whitening/head applicati **una volta
   sola** e in ordine; `prepare_index` su raw/whiten/head/head+whiten produce
   forme coerenti; PCA con `dim` ridotta restituisce la larghezza attesa.
7. **Casi limite del dominio** — PNG senza record `.mat` (dev'essere saltato,
   non far crashare); `mask_fraction=0` → immagine originale intatta; grafo con
   self-loop spurio; classe di equivalenza singleton (query esclusa **e
   contata**); stanza rimossa che svuota il grafo.

## NON testabile in automatico

Valori reali delle metriche, qualità visiva dei rendering, tempi e memoria su
GPU, comportamento sul dataset intero, ricarica di pesi pretrained che richiede
rete. Vanno **elencati nel report come verifica manuale o da sbatch**, con il
comando esatto.

## Qualità dei test

- Un test mai visto fallire non dimostra nulla: per un bug, prima il test rosso,
  poi il fix, poi il verde.
- Comportamento osservabile, non dettagli interni: il test sopravvive a un
  refactoring.
- Test indipendenti dall'ordine, senza stato condiviso mutabile.
- Niente `sleep`: sincronizzazione esplicita.
- Dati parlanti: una pianta reale con la sua vera composizione dice più di un
  tensore di zeri — ma il test deve restare veloce.
- Mai indebolire un'asserzione per farla passare: rosso = bug o test sbagliato,
  si decide, non si maschera.
