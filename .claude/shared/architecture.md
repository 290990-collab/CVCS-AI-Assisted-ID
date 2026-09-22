# Architettura — flusso, confini, contratti

## Le due fasi

1. **Fase 1 — Retrieval** (in corso): data una pianta (anche **parziale**),
   restituire le top-k più simili da RPLAN. Due pipeline complementari che
   condividono ground truth e metriche, poi **late fusion**.
2. **Fase 2 — Generazione** (non iniziata, **fuori da questo report**): generazione di layout
   constraint-aware condizionata su input parziali (numero/tipi di stanze,
   adiacenze) con loop di raffinamento iterativo.

## Flusso dei due rami

```
VISION   PNG → encoder frozen → pooling → [head] → [PCA whitening] → L2 → FAISS(IP) → top-K
GRAPH    .mat → grafo PyG → [transform] → GNN allenata (InfoNCE) → L2 → FAISS(IP) → top-K
FUSIONE  concatenazione pesata [√α·v ; √(1−α)·g] su embedding L2-norm  (fatta: α*=0.6 sul valid, status.md §49-§51.1)
```

Entrambi i rami producono **embedding L2-normalizzati** indicizzati in
`IndexFlatIP` (coseno = prodotto interno) e le **stesse tabelle per-asse**.

**Perché il whitening.** Un encoder frozen su colormap RPLAN (sintetiche, fuori
distribuzione rispetto alle immagini naturali su cui è pre-allenato) produce
embedding collassati in una piccola regione dell'ipersfera: gli score finiscono
tutti in `[0.97, 0.99]` e il ranking diventa indiscriminato. Il PCA whitening li
spalma e rende il ranking discriminativo. Va applicato **anche alle query**.

**Gallery condivisa** (fase B.3, 25 ago 2026). I due rami possono valutare
sull'**inner join** dei nomi (`src/evaluation/gallery_join.py` → JSON congelato
con sha1; `eval.gallery_names` / `--gallery-names`). La gallery viene ristretta
**e riordinata** sull'ordine canonico: stesso `gallery_sha1`, stesse query
campionate, confronto cross-ramo appaiato senza override. **Attiva dal 10 set
in entrambi i YAML** (`results/shared_gallery.json`, `status.md §29`); `null`
riporterebbe al comportamento storico — gallery intera, confronto zoppo.

**Su quali righe si stima** (decisione dell'utente, 25 ago 2026 — fase B.2).
Le statistiche del whitening (media + matrice PCA) si stimano sul **solo split
train**: `whitening.fit_split: train`, il default. La gallery indicizzata resta
**intera** — lo split restringe l'insieme di *stima*, mai il corpus di ricerca
(vincoli DURI 1 e 2 insieme). Il protocollo **trasduttivo** di prima
(`fit_split: all`, stima su tutta la gallery) resta disponibile come termine di
confronto appaiato, non come default: è quello di **tutte** le run fino al
24 ago 2026, §24 compresa. ⚠️ I due protocolli producono **tag diversi**
(`whiten` vs `whiten-train`) proprio per non sovrascrivere i file dell'altro.
Il ramo graph **non usa whitening**: la decisione lo tocca solo di riflesso,
sui confronti cross-ramo. **Misurato il 10 set** (`status.md §31.1`): il
train-only vale quanto il trasduttivo (delta minimo, a favore del train-only) →
il guadagno del whitening **non era leakage**.

**Perché l'embedding RAW su disco.** Whitening e head sono trasformazioni
lineari/leggere applicabili al volo: salvando il RAW, una sola estrazione GPU per
(encoder, pooling) alimenta tutte e quattro le combinazioni a costo ~0.
Simmetricamente, nel ramo graph si cachano i **grafi grezzi** e non gli
embedding — che cambierebbero a ogni epoca, dato che lì la rete si allena.

## Confini e responsabilità

- **`src/data/` + `src/evaluation/` = core condiviso.** Sono l'unico punto in cui
  si legge la ground truth RPLAN e l'unico in cui vivono le funzioni di metrica.
  Entrambi i rami li usano **as-is**. Motivo: se la definizione di rilevanza
  divergesse fra i rami, i loro numeri non sarebbero più confrontabili.
- **`src/vision/` e `src/graph/` sono autonomi.** Nessun refactor cross-ramo: gli
  helper di *accumulo* delle metriche sono duplicati **di proposito**. Si
  condividono le funzioni numeriche, non l'orchestrazione.
- **Config e codice separati**: gli iperparametri vivono nei YAML; il codice non
  contiene default nascosti che li contraddicono.
- **Gli script `.sh` sono parte del sistema**, non un dettaglio operativo: i
  ponti YAML→flag rendono i config vivi, quindi una chiave YAML è a tutti gli
  effetti un'API.

## I contratti (cambiarli è una decisione architetturale)

1. **Artefatti su disco.** `embeddings.npy` (RAW, vision) + `image_paths.json` ·
   `embeddings.npy` + `names.json` allineati per riga (graph, **interfaccia
   della late fusion**) · `graphs.pt` (grafi grezzi) · `pairs.npz` (con campo
   `splits`) · `head.pt` · `encoder.pt` + `geom_stats.npz` +
   `training_summary.json` · render-cache. Cambiare un formato **obbliga a
   rigenerare** e costa ore di GPU: va dichiarato.
2. **Forma dell'architettura ↔ checkpoint.** Ogni flag che cambia la forma di un
   layer (`raw_skip`, `hidden_dim`, `out_dim`, `pooling`, `heads`) dev'essere
   identico fra training ed eval, e quindi presente in **tutti e quattro** i
   livelli: codice, argparse, YAML, ponte YAML→flag. Presente in tre su quattro
   = ignorato in silenzio.
   **Eccezione voluta — `--lost-marker` (graph, 14 set, `status.md §41`)**: aggiunge una colonna a `x`
   (`in_dim` 19→20, `transforms.LOST_MARKER_COL`) ma **non** ha chiave YAML: vive solo come variante
   (`asymlost`) nei `case` di `03_train_gnn.sh` **e** `04_eval_gnn.sh`. Una chiave nel YAML `gat` a
   `true` renderebbe a 20 anche `base`/`asym` e ne romperebbe la ricarica. Chi costruisce query
   parziali per l'encoder graph deve passare il flag a `make_partial_graph` e `build_node_transform`.
3. **Ordine indice ↔ nome.** `shuffle=False` nei loader di indicizzazione:
   la riga `i` dell'embedding *è* la pianta `i` della lista di nomi. Rompere
   questo invalida silenziosamente ogni metrica.
4. **Namespacing degli output.** `save_dir` è composto da
   `model.name`/`model.variant` (vision) e `encoder`/`variant` (graph): due run
   con lo stesso `variant` **si sovrascrivono**. È il meccanismo di tracciabilità
   degli esperimenti e insieme la trappola più insidiosa nei confronti.
5. **Contratti dei registry**: `BaseVisionEncoder` (`forward → [B,D]` L2-norm,
   `build_transform`, `embedding_dim`) e `BaseGraphEncoder` (`build_conv` è
   l'unico punto di variazione). Un encoder nuovo si aggiunge implementando il
   contratto e registrandolo, senza toccare la pipeline.
6. **`perquery/1` — i valori per-query** (`src/evaluation/perquery.py`, 31 lug
   2026). Un `.npz` per run di valutazione: `names` (stem, **chiave di join fra
   run: mai `qi`**, le due gallery hanno righe diverse), `qi`, `ndcg/recall/map`
   `[assi,K,query]` con **NaN = query saltata** su quell'asse, `num_relevant`
   (0 = skip singleton), `ret_rows` (top-K recuperati) e un `meta` JSON con
   `gallery{n,sha1}`, `split`, `exclude_self`, `query_seed`, `geometry_weights`.
   Opt-in: `eval.perquery_dir` (vision, default `null`), `--perquery-out`
   (graph), `PERQUERY_OUT` per `scripts/graph/04_eval_gnn.sh`. Senza, il
   comportamento è identico a prima.
   **A cosa serve**: `significance.py` rifiuta di appaiare due run se `meta`
   discorda (gallery, split, seed…) — è la guardia che impedisce confronti non
   appaiati; `geometry_variants.py` ricalcola l'asse geometria da `ret_rows`
   **senza rilanciare il retrieval** (i pesi cambiano il gain, non il ranking).
   ⚠️ I `ret_rows` valgono solo per la gallery con quel `sha1`.
7. **`qvec/1` — i vettori delle query danneggiate** (`src/evaluation/query_vectors.py`,
   16 set 2026, late fusion di `status.md §49`). Un `.npz` per run partial, **stesso
   nome file** del per-query della stessa run, in una cartella propria: `names`,
   `qi` (riga nella gallery condivisa), `vectors [Q,D]` (vision: **RAW** dell'encoder;
   graph: il vettore cercato in FAISS, L2), stanze tolte in CSR `removed_ptr [Q+1]` +
   `removed_idx [R]`; solo vision `vectors_final` (diagnostica), `whiten_mean [1,D]`,
   `whiten_matrix [D,D']` **del job**. Meta: `perquery_file`, `damage`, `split`,
   `query_seed`, `partial_seed`, `k_values`, `gallery{n,sha1}`,
   `gallery_vectors{path,sha1,shape}` (sha1 dei vettori gallery dopo la restrizione
   alla gallery condivisa: il graph riscrive `embeddings.npy` a ogni valutazione),
   vision `whitening` + `head` (deve essere `null`), graph `model` + `n_degenerate`.
   Opt-in: `eval.query_vectors_dir` (vision) / `--query-vectors-out` (graph), entrambi
   **richiedono** il per-query; solo run con stanze tolte (crop/patch saltati).
   **File fusi** (`src/evaluation/late_fusion.py`): formato `perquery/1`, nome
   `fusion_a<α:g>_partial-nowalls-random-f<f>_<split>.npz` e `fusion_a<α:g>_full_<split>.npz`,
   meta `branch: fusion`, `damage{strategy: nowalls_random, graph_strategy: random}`,
   `fusion{alpha, method: weighted_concat_sqrt, vision{run_tag,qvec,gallery_vectors.sha1,
   whitening}, graph{…}}`, `join{n_only_vision, n_only_graph}`. Letti as-is da
   `robustness_auc.load_auc(strategy="nowalls-random")` e `significance`.
   Controlli e scelta di α: `src/evaluation/fusion_select.py` (`check`, `select`).

## Decisioni vincolanti già prese

- **Gallery = l'intero corpus che entrambi i rami vedono**, mai splittata: lo
  split restringe le **query**. Dal 10 set è l'inner join `snapshot_train/` ∩
  `.mat` (`results/shared_gallery.json`): si tolgono solo le 48 PNG senza
  `.mat`, che il graph non può rappresentare — da dichiarare in ogni tabella.
- **Selezione sul valid, test solo per il numero finale.**
- **Il checkpoint del ramo graph si sceglie sulla sonda di retrieval**, non sulla
  val-loss InfoNCE (che ne inverte la classifica — vedi `.claude/shared/status.md`).
- **Capacità fissa ViT-B** per il benchmark vision (I-JEPA è l'eccezione
  obbligata, ViT-H: divario da dichiarare). Le varianti di taglia non si
  confrontano.
- **Encoder vision frozen**; l'unica parte allenabile lì è la projection head.
- **Nessuna dipendenza di rete per il ramo graph**: GNN da zero, dati locali,
  wandb spento di default (e comunque `offline`-capace).
- **Scope del report: solo RPLAN.** ResPlan e CubiCasa5K (obiettivo O3) sono
  in **stand-by** (decisione utente 11 set, da discutere coi dottorandi);
  i rischi noti (nessuna label, tassonomia non allineabile senza perdite) vanno
  nel testo come motivo. Maticad rimandato.
- **Head vision**: selezione `probe_partial` (B.6) con budget di epoche ampio;
  la val-loss resta come confronto nella stessa run (`status.md §30.2-§30.5`).
  ⚠️ Config vision **definitiva frozen** `pespatial/gem/whiten` (`status.md §44`): la head rifatta
  sui danni nuovi (coppie `nowalls`, 16 set) perde su crop e patch e non è adottata (§48).

## Valutare una proposta di design

1. Quali contratti tocca? (nessuno = rischio molto più basso)
2. Obbliga a rigenerare artefatti? Quante ore di GPU costa all'utente?
3. Può cambiare numeri già riportati? Se sì, il confronto resta appaiato?
4. Introduce una via per cui il test influenza una scelta? (allora è da rifare)
5. Il guadagno atteso è misurabile e superiore al rumore noto?
6. Qual è l'alternativa più semplice che risolve il 90% del problema? (KISS)

## Questione aperta — circolarità della valutazione

Le label di rilevanza degli assi **composizione** e **topologia** derivano da
`rType`/`rEdge`, che sono **esattamente** l'input del ramo graph (node features
ed `edge_index`). Il grafo riceve in ingresso ciò che la ground truth misura in
uscita.

⚠️ **Aggiornato il 24 ago (`status.md §21`), due misure che spostano il quadro.**
La vecchia formulazione — «l'unico asse equo è la geometria», «il vision deve
inferire la struttura dai pixel» — **non regge su nessuno dei due fronti**:

1. **Nemmeno la geometria è equa.** `gtBox[-1] == unione(gtBoxNew)[[1,0,3,2]] -
   [0,0,1,1]` al **100%** su 24.218 piante → `footprint_area` e
   `footprint_aspect` sono ricostruibili esattamente dalle node feature, come già
   lo era `type_area_distribution`. Tutti e tre i componenti di `geometry_sim`
   sono funzioni dell'input del grafo (rilievo **A5 pieno**).
2. **Anche il vision legge le etichette.** Il colore delle PNG è una funzione
   **deterministica** di `rType` (purezza 100% su 11 tipi su 12), ma **non
   iniettiva**: 13 tipi → **6 colori**. Il vision non infera la semantica, la
   **legge** — a grana grossa.

**Il quadro nuovo, in una riga:** non c'è un asse pulito e non c'è un ramo
pulito; c'è una **differenza di risoluzione dell'informazione**. Sul lato
composizione il graph vede 13 classi e il vision 6; sul lato geometria il graph
ha i box esatti e il vision i pixel. Il confronto non è «simbolico vs visivo»,
è **fine vs grosso sullo stesso input**.

Uscite possibili, riviste:

1. dichiarare la circolarità **per entrambi i rami** e presentare i numeri come
   *skyline su RPLAN*, non come capacità di generalizzare;
2. misurare il **tetto a 6 classi** (baseline `hist` sulle 6 classi visive): dà
   il massimo ottenibile dal vision leggendo il colore, e rende leggibile il gap
   0.83 vs 0.97 sulla composizione — costo basso, fase B;
3. l'unico terreno davvero neutro resta ciò che il grafo **non può
   rappresentare**: forma non rettangolare, muri, aperture. Nessun asse attuale
   lo misura;
4. estrarre il grafo **dall'immagine** (stile Graph2Plan) → confronto onesto,
   fuori scope per la Fase 1.

**Decisione da prendere** (domanda aperta n.1 della roadmap): come si presenta il
confronto, ora che nessun asse è esente. Finché non è presa, ogni tabella che
confronta i due rami porta il caveat — su **tutti e tre** gli assi, non solo su
composizione e topologia.
