# Struttura del progetto — mappa dei moduli

Ogni modulo è un entrypoint `python -m`. Non c'è una CLI centrale.

## Core condiviso (fuori dai due rami — usato **as-is**, mai rifattorizzato)

- **`src/data/rplan_metadata.py`** — unico lettore dei `.mat` RPLAN.
  `RoomMeta` (`room_types`, `edges` tipizzati 0..9, `boxes` da `gtBoxNew`,
  `footprint` da `gtBox`, `entrance`, `split`) + `load_metadata(png)` +
  `get_split(png)`. Join via `name` = stem del PNG.
- **`src/evaluation/relevance.py`** — `GalleryAxes`: unica fonte di verità della
  rilevanza. Feature per-asse vettorializzate, similarità per-asse, classi di
  equivalenza esatta su tutta la gallery.
- **`src/evaluation/metrics.py`** — funzioni pure: `ndcg_at_k` (primaria),
  `recall_at_k`, `average_precision_at_k` (solo assi discreti).
- **`src/evaluation/perquery.py`** — schema `perquery/1`: scrive e rilegge il
  valore di **ogni** query (`PerQueryRecorder` / `load_perquery`). È il formato
  che rende possibili tutti i confronti appaiati. Nel **partial** aggiunge
  `self_rr` `[Q]`, il reciprocal rank del self.
- **`src/evaluation/significance.py`** — delta appaiato + CI 95% bootstrap +
  Wilcoxon. `--allow-gallery-mismatch` solo per i file **pre-B.3** (con la
  gallery condivisa non serve più); `--metric self_rr`
  è l'endpoint del criterio A.5 (forma `[Q]`, senza asse né K → gestito a parte,
  **non** dentro `METRICS`).
- ⚠️ *(14 set)* `robustness_auc --robust` = **metro in vigore** (`status.md §38`): media per
  query delle AUC su `nowalls-random`, `crop`, `patch`.
- **`src/evaluation/robustness_auc.py`** *(10 set)* — AUC del criterio A.5 per
  query (`rank` = classifica delle config trovate in una o più cartelle,
  `compare` = delta appaiato A − B con CI). Riproduce §24 esatto. È lo strumento
  per i confronti vision↔graph sulla robustezza (tollera le query saltate dal
  graph, `significance --metric self_rr` no).
- **Late fusion** *(16 set, `status.md §49`)* — `src/evaluation/query_vectors.py` (contratto `qvec/1`:
  vettori delle query danneggiate + stanze tolte, scritti dalle due valutazioni con un flag opt-in) ·
  `late_fusion.py` (CPU: concatenazione `[√α·v ; √(1−α)·g]`, file fusi in `perquery/1`) ·
  `fusion_select.py` (`check` C1-C5, `select` = α*, verdetto, oracolo). Script `evaluation/08-10`.
  Controllo effetto d'insieme *(17 set, §50)*: `late_fusion --pair graph-graph` (asymrob + replica), `fusion_select check/select --pair graph-graph` e `complementarity`. Script `evaluation/11-12`.
  Controllo «modelli diversi, stessa informazione» *(17 set, §51)*: `--pair vision-vision` (pespatial + radio), `complementarity --control-pair vision-vision` (regola a quattro esiti). Script `evaluation/13-14`.
- **`src/evaluation/gallery_join.py`** *(25 ago, B.3)* — inner join dei nomi fra
  i due rami: CLI che congela l'intersezione in un JSON con provenienza e sha1,
  più `restrict_rows` che restringe **e riordina** una gallery sull'ordine
  canonico. È ciò che dà ai due rami lo stesso `gallery_sha1` (quindi le stesse
  query) e rende il confronto cross-ramo appaiato senza override.
- **`src/evaluation/random_floor.py`** — i due null (`random`, `constant`): il
  pavimento contro cui si normalizzano tutte le percentuali.
- **`src/evaluation/geometry_variants.py`** — sensibilità dell'asse geometria ai
  pesi delle sue tre componenti (post-hoc sui `ret_rows`, niente GPU).
  `--gallery-b` per confrontare due rami con gallery diverse.
- **`src/evaluation/metric_diagnostics.py`** — diagnostica di saturazione/scala
  delle metriche.

## Figure del report (`src/figures/`) *(18 set, `status.md §54`)*

Solo lettura di risultati già su disco, CPU, nessun job. Un modulo per figura,
entrypoint `python -m src.figures.<nome>`, output in `figures/` (PDF per il report,
PNG per guardarla, `*.sources.txt` con file letti/comando/numeri).

- **`style.py`** — larghezze del report a **due colonne** (3.25″ / 6.875″), font,
  palette Okabe-Ito con tratteggi e marker per il bianco e nero, `new_figure`,
  `mean_ci` (stesso bootstrap di `significance`), `save_figure`.
- **`damage_test.py`** — F7: self-recovery vs stanze tolte sul test, quattro sistemi
  (baseline `hist` · vision · graph · fusione), letti dai file fusi (α=1 / α=0).
- **`alpha_valid.py`** — F8: AUC vs α con le tre coppie fuse; ricalcola le medie dai
  per-query e **rifiuta** di disegnare se non coincidono col json di `fusion_select`.
- **`teaser_valid.py`** — F1: query danneggiata + primi 5 risultati di vision/graph/fusione.
  Ricostruisce la classifica da `ret_rows` + `self_rr` e ridisegna il danno dalle stanze del
  `qvec` con `vision_damage.render_wiped_image`; nessun job, nessuna GPU.
- **`damage_kinds_valid.py`** — F3: due pannelli, i tre danni sulla config di record e
  l'artefatto dei muri (`random` vs `nowalls-random`, stesse stanze).
- **`pipeline.py`** — F2: schema dei due rami + fusione + misura condivisa; le dimensioni,
  α e la gallery sono **lette dagli artefatti** a ogni disegno, non scritte a mano.
- **`classes_valid.py`** — F5: dimensione delle classi di equivalenza da `num_relevant`;
  verifica che due sistemi diano lo stesso conteggio.
- **`ablation_valid.py`** — F6: R di tutte le config per encoder (box + punti), con cache
  in `figures/f_ablation_valid.data.json` (`--refresh`).

## Ramo vision (`src/vision/`)

**Encoder intercambiabili** — `models/vision_encoders/`, contratto
`BaseVisionEncoder` (`forward → [B,D]` L2-norm, `build_transform`,
`embedding_dim`), tutti **frozen**. Helper condivisi `gem_pool` /
`pool_global` (`POOLINGS = natural|mean|gem`).

| File | Encoder | Pooling "naturale" | Note |
|---|---|---|---|
| `dinov2.py` | DINOv2 | token [CLS] | norm. ImageNet; `extraction_layer` via `output_hidden_states` |
| `dinov3.py` | DINOv3 | token [CLS] | sottoclasse di DINOv2; checkpoint **gated** |
| `siglip2.py` | SigLIP2 | attention pooling | norm. **[0.5]×3, non ImageNet**; `extraction_layer` non applicabile |
| `radio.py` | RADIO | summary token | `trust_remote_code`; normalizza **internamente** (transform a [0,1]); `embedding_dim` dedotta con un forward di prova |
| `ijepa.py` | I-JEPA | mean-pool patch | **nessun [CLS]**; solo taglie grandi (ViT-H) |
| `tipsv2.py` | TIPSv2 B/14 | token [CLS] | `trust_remote_code` + `sentencepiece`; transform a **[0,1]** come RADIO; **register token scartati**; nativa **448** → la risoluzione entra nella variante |
| `pe_core.py` | PE-Core B/16-224 | attention pool + proiez. CLIP | da **timm**, non `transformers`; norm. **0.5×3**; `natural` esce a **1024d**, `mean`/`gem` a 768d; risoluzione **bloccata** (RoPE) |
| `pe_spatial.py` | PE-Spatial B/16-512 | token [CLS] | da timm; coppia controllata di `pe_core` (stesso ViT-B/16); `head=Identity` e `global_pool="avg"` → `model(x)` è la **media**, quindi `natural` legge il CLS; 768d su tutti i pooling |

- `models/vision_encoders/__init__.py` — `ENCODER_REGISTRY` + `build_encoder`.
- `models/vision_model_manager.py` — `VisionModelManager(config)`: costruisce
  l'encoder dal registry, device + eval, espone `.encoder` e `.transform`.
  **Niente optimizer/scheduler** (il ramo vision non allena il backbone).
- `models/retrieval_model.py` — `VisionRetrievalPipeline`, agnostico
  all'encoder. ⚠️ **Su disco si salva l'embedding RAW** (`embeddings.npy`);
  `prepare_index(head, whiten, whiten_dim)` applica al volo head → whitening →
  L2 e costruisce FAISS (`IndexFlatIP`), con le stesse trasformazioni sulla
  query. Una sola estrazione GPU per (encoder, pooling) alimenta tutte le
  combinazioni raw/whiten/head/head+whiten.
- `models/projection_head.py` — `ProjectionHead` (MLP `Linear→GELU→Linear`,
  uscita L2-norm) + `info_nce(za, zp, temperature)` + `load_head`.
- `data/preprocess.py` — `ResizeWithPad` (condiviso, no distorsione),
  `compose_transform`, `load_image`. `data/loader.py` — `RPLANDataset` +
  `get_dataloader()` (`shuffle=False`: l'ordine indice↔path è un contratto).
- `data/vision_partial_query.py` — query parziali:
  `select_rooms_to_remove(meta, strategy, params, rng)` (random / semantic /
  topology) + `render_partial_image(png, meta, removed_idx, open_boundary)`
  (affine griglia-256→px, flood-fill, cancellazione dei muri) +
  `make_partial_query`.
- `data/vision_damage.py` *(11-14 set)* — entry point unico del danno alla query
  (`damaged_query`, anche `return_image=True` per le figure): stanze storiche, crop,
  patch ViT e `nowalls_*` (stanze tolte **con** i muri, `status.md §36`). Lo usano
  `evaluate.py` e `retrieval_visualization.py`: figura e numeri vedono lo stesso danno.
- `data/projection_pairs.py` — cache delle coppie positive self-supervised
  (`pairs.npz`: `anchors`, `positives`, `splits`). Render-cache condivisa su
  disco (`embeddings/vision/_render_cache/<hash>/`), scrittura **atomica**,
  seeding deterministico **per pianta** (non per indice) → il rendering si paga
  una volta invece di 14, bit-identico. *(15 set)* `training.damage` = `random` (storico) |
  `nowalls_random` (`vision_damage.room_damage_image`): entra nell'hash della cache solo se ≠
  `random`, si salva in `pairs.npz` (`damage`); la probe lo scrive in `meta.strategy` e
  `train_projection` rifiuta coppie, probe e config con danni diversi (`status.md §46`).
- `training/train_projection.py` — allena la head sui vettori cachati (backbone
  frozen) con InfoNCE; train su `train`. Selezione dell'epoca con
  `training.selection`: `val_loss` (storico → `head.pt`) o `probe_partial`
  (B.6 → `head.file`, es. `head_probe_conv.pt`; salva anche la scelta della
  val-loss nella stessa run). Init della head seedato (10 set). wandb opzionale.
- `training/retrieval_probe.py` *(10 set, B.6)* — probe partial della head:
  degrada 1000 query **valid disgiunte** da quelle di eval, ne salva le feature
  RAW (`probe_partial.npz`) e calcola l'AUC di self-recovery per epoca.
- `evaluation/evaluate.py` — orchestratore: modalità **full** o **partial**
  (blocco `partial` del config), accumulo per-asse (`_accumulate_axes`, flag
  `exclude_self`), tabella asse × metrica.
- `utils/config.py` — `load_vision_config(path, overrides)` (innesta il preset
  `configs/vision_models/<name>.yaml` e applica gli **override dotlist** per
  ultimi) + `transform_tag(cfg)` (`raw|whiten|whiten768|head|head+whiten`).
- ⚠️ *(14 set)* il partial delle visualizzazioni passa da `vision_damage.damaged_query`.
- `utils/retrieval_visualization.py` — CLI: pannelli query vs top-k in
  `results/visualizations/<name>_<variant>_<full|partial>/`, thumbnail 512px,
  rilevanza per-asse in header, marcatore `exact:C/T`, `[orig]` per il
  self-recovery.

## Ramo graph (`src/graph/`)

- **`graph_builder.py`** — adapter `RoomMeta` → PyG `Data`. **Nessun I/O sui
  `.mat`**: chiama `load_metadata`. `NODE_FEATURE_DIM = 19` = one-hot tipo (13)
  + geometria (6: `cx, cy, w, h, area, aspect`, normalizzata su griglia 256).
  `edge_index` simmetrizzato (`to_undirected`), `edge_attr` = one-hot relazione
  0..9 (10-dim). Attributi graph-level: `name`, `split`, `num_rooms`,
  `footprint_area/aspect`, `type_histogram` (base della baseline training-free).
  ⚠️ Geometria **solo da `boxes`/`gtBoxNew`**, mai da `footprint`.
- **`graph_dataset.py`** — `RplanGraphDataset(InMemoryDataset)`: costruisce i
  grafi una volta e li serializza in `embeddings/graph/rplan/processed/graphs.pt`
  (117 MB; build 2m47s → reload ~15s; il costo evitato è la lettura dei `.mat`,
  ~35s per processo). **67.405 grafi** (48 PNG senza record `.mat`);
  split 47.126 / 10.152 / 10.127. Grafi in cache **GREZZI** → una sola cache
  serve tutte le varianti; `split=` filtra **in memoria**. Gli attributi stringa
  sopravvivono al collate.
- **`transforms.py`** — adjustment come `transform` PyG (accendibile in
  ablation, **fuori** dal builder). `NormalizeNodeGeometry` (clip di `aspect` al
  p99 **prima** dello z-score: la coda arriva a 37 contro one-hot ≤ 1, e `area`
  media 0.047 sarebbe inerte), `RemoveSpuriousSelfLoops` (GCN/GAT aggiungono i
  self-loop da sé → altrimenti peso doppio). `compute_geometry_stats(train)` +
  `save_/load_geometry_stats` (`.npz`) + `build_node_transform(...)`.
  Verificato: p99 `aspect` = 4.58; no leakage; **non** serve ricentrare `cx/cy`
  (RPLAN centra ogni pianta: centro footprint 0.500 ± 0.001).
- **`models/`** — encoder intercambiabili, **allenabili** (a differenza del
  vision). `base.py`: `BaseGraphEncoder` con **un solo punto di variazione**
  (`build_conv`); stack, pooling, proiezione e L2-norm stanno nella base.
  Default che codificano il dominio: `pooling="add"` (preserva il conteggio
  stanze, che è un asse di rilevanza; `mean` lo normalizzerebbe via) e
  `num_layers=2` (grafi 4-8 nodi, diametro 2-3 → più profondità = over-smoothing).
  **`raw_skip`**: concatena l'add-pool delle feature **grezze** all'embedding
  poolato prima della proiezione (`proj` 128→147) → composizione e geometria
  arrivano all'uscita per costruzione. ⚠️ È un flag di **architettura**: cambia
  la forma di `proj`, quindi dev'essere identico fra training ed eval.
  `gcn.py` (isotropo, baseline), `gat.py` (`GATv2Conv`, **unico** a usare
  `edge_attr`, `heads` mediate con `concat=False`), `graph_sage.py` (`SAGEConv`,
  separa nodo/vicini → preserva l'identità del tipo).
  `GRAPH_ENCODER_REGISTRY` = `gcn|gat|sage`.
- **`training/augment.py`** — augmentation di grafo, parametri in
  `AugmentParams` (`node_drop`, `edge_drop`, `feat_mask`, `geom_jitter`,
  `flip_prob`, `rot_prob`). Principio: *ogni augmentation dichiara
  un'invarianza, da confrontare con la ground truth* — flip/rot sono invarianze
  **vere** (i tre assi sono esattamente invarianti), `node_drop` è **falsa** sul
  full (cambia la composizione) ma vera sul partial. ⚠️ Le simmetrie si applicano
  in coordinate **grezze** (le stats non sono isotrope: `std(cy)/std(cx)=1.24`)
  → de-normalizza, trasforma, ri-normalizza. `feat_mask` è **per-cella**
  (nodo, colonna). Le viste mantengono numero nodi e vettore `batch` → righe
  InfoNCE allineate. **`pair_mode`** *(10 set, opzione D di §30.6)*:
  `symmetric` (default, storico bit per bit) | `asym_partial` (vista A = grafo
  intero, vista B = stanze **rimosse** davvero, f~U[0.25, 0.75], come il partial
  di valutazione); flag `--pair-mode`, chiave in `gcn.yaml`.
- **`graph_partial_query.py`** *(25 ago, C.0)* — query di grafo degradata:
  `filter_meta` (toglie stanze e **rimappa** gli archi) + `make_partial_graph`.
  Importa `select_rooms_to_remove` dal ramo vision di proposito: l'oggetto
  condiviso è il **protocollo**, e una copia che diverge romperebbe
  l'appaiamento. Query svuotate: contate e saltate.
- **`training/train_gnn.py`** — `info_nce`, `_train_epoch` (due viste per
  batch), `_val_loss` (augmentation deterministica, **solo diagnostica**),
  `train(cfg)` con early stopping + `encoder.pt` + `geom_stats.npz` +
  `training_summary.json`. ⚠️ **Il criterio di selezione è l'nDCG della
  `RetrievalProbe`, non la val-loss** (`--probe-every`, `--probe-queries`,
  `--probe-gallery`, `--probe-k`, `--select-criterion`); `probe_every: 0` =
  fallback alla val-loss negata. La `patience` conta **sonde**, non epoche.
- **`evaluation/retrieval_probe.py`** — `RetrievalProbe`: mini-retrieval sul
  **valid** che guida best-checkpoint ed early stopping. `build(...)` campiona
  una gallery fissa e calcola la `GalleryAxes` **una volta sola**; `score(...)`
  fa forward → prodotto scalare su vettori L2-norm → metriche per-asse. Costo
  ~2.6 s/epoca (≈20%), build 30 s. ⚠️ I valori **non** sono confrontabili col
  report finale (gallery più piccola → IDCG diverso): servono a confronti
  relativi, appaiati perché le query sono fisse.
- **`evaluation/axis_metrics.py`** — helper di accumulo/stampa per-asse
  (`new_metrics`, `accumulate_axes`, `mean_metric`, `print_axis_tables`),
  condivisi fra sonda e valutazione finale: la metrica che **seleziona** i pesi
  e quella che li **giudica** sono lo stesso codice.
- **`evaluation/graph_evaluate.py`** — valutazione retrieval: ricarica
  `encoder.pt` + `geom_stats.npz` → forward **una volta** su tutta la gallery →
  salva `embeddings.npy` + `names.json` allineati (interfaccia della late
  fusion) → FAISS `IndexFlatIP` → query dallo split → metriche per-asse.
  Ricerca **batched** (l'embedding query è già una riga della gallery).
  **`--baseline-hist`** = baseline training-free (`type_histogram` L2-norm,
  nessun checkpoint). Stampa il caveat di circolarità.
- **`draw_graph.py`** — visualizzazione + CLI (`python -m src.graph.draw_graph
  10002 --overlay`). Layout `spatial` (nodi ai centroidi reali → sovrapponibile
  al PNG: è il **test di correttezza del builder**) o `spring`. ⚠️ Da CLI ogni
  invocazione ricarica i `.mat` (~35s): per molte piante usare `--grid`.

## Config

- `configs/vision_retrieval.yaml` — principale: `model.name` (selettore),
  `model.variant` (etichetta che namespacizza), `retrieval.*`, `whitening.*`
  (`enabled`/`eps`/`dim`), `head.*`, `training.*`, `eval.*`
  (`num_queries`/`seed`/`split`/`k_values`), blocco `partial`.
  `save_dir = embeddings/vision/${model.name}/${model.variant}`.
- `configs/vision_models/<name>.yaml` — kwargs del costruttore per encoder
  (`hf_name`, `image_size`, `pooling`, `gem_p`, `extraction_layer`).
- `configs/graph_models/{gcn,gat,graph_sage}.yaml` — **ricette auto-contenute**,
  chiavi 1:1 con gli argomenti di `train_gnn.py` (il ramo graph non ha ancora un
  config-loader: "un YAML = un job"). ⚠️ file `graph_sage`, chiave
  `encoder: sage`. Specifici: `gat` → `heads`/`attn_dropout`; `graph_sage` →
  `aggr`.
- `configs/graph_retrieval.yaml` — parametri **di valutazione** condivisi
  (`num_queries`, `seed`, `split`, `k_values`, `batch_size`, `baseline_hist`).
  I parametri del modello **non** stanno qui: vengono dal YAML di training.

## Script HPC (`scripts/`) — non toccare se non richiesto

- `vision/_common.sh`, `graph/_common.sh` — `PROJECT_DIR`, `select_encoders`, e
  i **ponti YAML→flag** (leggono i YAML con OmegaConf e li traducono in flag
  argparse: underscore→trattino, booleani con mapping esplicito). I config sono
  quindi **vivi**, non documentazione morta.
- `vision/01_extract_raw → 02_build_pairs → 03_train_head → 04/05_eval_*_full →
  06/07_eval_*_partial → 08_visualize`; `09_train_head_probe` (B.6).
- `graph/01_graph_builder → 02_graph_dataset → 03_train_gnn → 04_eval_gnn`.
- `evaluation/` (protocollo di misura, core condiviso): `01_random_floor` ·
  `02/03_perquery_vision_{valid,test}` · `04_perquery_graph` ·
  `05_perquery_vision_partial_valid` · `06_perquery_graph_partial_valid`.
  Sono gli **unici** che salvano i per-query.
- Convenzione: **nessun argomento = tutti i modelli**, un argomento = uno solo
  (parallelizza). Secondo argomento (graph) = variante di ablation.
- Ordine, dipendenze e comandi pronti: **`COMANDI.md`** (root).

## Documenti

`COMANDI.md` (pipeline sbatch) · `PAPER.md` (indice bibliografico vivo) ·
`spiegazione_metriche_e_risultati_grafi.md` (spiegazione discorsiva di metriche,
risultati graph e fusione — fonte per il report) · `vision_pipline.xlsx` (tutti
i numeri del benchmark vision).
