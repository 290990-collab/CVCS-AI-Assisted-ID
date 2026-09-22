# PAPER.md — Indice dei contenuti per il paper/report

> **Documento vivo.** Non è il paper: è l'*indice* di tutto ciò che deve finirci,
> con le motivazioni e i numeri da non dimenticare. Va aggiornato a ogni
> cambiamento della pipeline.

## Come usare questo file (convenzione)

- **Mai dare per scontato ciò che è scritto qui.** Ogni affermazione numerica o
  di design va **ri-verificata contro il codice/log finali** prima di entrare
  nel paper. Per questo ogni punto "duro" è taggato con il **file sorgente**.
- Marker di stato:
  - `[DONE]` implementato e verificato sui dati;
  - `[TODO]` da fare;
  - `[VERIFY]` scritto qui ma **da riconfermare contro il codice** prima del paper;
  - `[DECISION]` scelta di design da motivare nel paper.
- Quando la pipeline cambia: aggiorna il punto **e** sposta/aggiorna il marker.
- Scope attuale: **solo RPLAN, Fase 1 (Retrieval)**. Lingua del paper: probabilmente EN.
- Tipo: mid-paper / report per i prof.
- **Fonti vive dei numeri** (questo file NON li duplica): `.claude/shared/status.md`
  (risultati e ipotesi smentite), `vision_pipline.xlsx` (benchmark vision),
  `spiegazione_metriche_e_risultati_grafi.md` (spiegazione discorsiva per il report).

---

## 0. Checklist di verifica pre-paper (da rifare a ridosso della stesura)

Spuntare **eseguendo/leggendo il codice**, non a memoria:

- [ ] Numeri dataset (quante piante, copertura `.mat`) → da `src/data/rplan_metadata.py` + log indicizzazione.
- [ ] Dettagli encoder (dim embedding, pooling, input size) → `src/vision/models/vision_encoders/` + log.
- [ ] Parametri pipeline (whitening, tipo indice FAISS) → `src/vision/models/retrieval_model.py`.
- [ ] Definizione esatta delle 3 similarità d'asse → `src/evaluation/relevance.py` (`GalleryAxes`).
- [ ] Definizione esatta delle metriche + normalizzazioni → `src/evaluation/metrics.py`.
- [ ] Valori di K, num query, seed → `configs/vision_retrieval.yaml` (`eval`).
- [ ] Tabella risultati rigenerata con l'ultimo codice (non copiare vecchi log).
- [ ] Ogni claim "X motiva Y" regge ancora dopo eventuali modifiche.

---

## 1. Abstract & Contributi

Bullet dei **contributi** (da limare):
> ⚠️ **10 set — fase di chiusura.** Il piano definitivo di tabelle/figure e le run
> che le riempiono è `.claude/shared/roadmap.md §1`; i numeri finali sono sul
> **protocollo B** (`experiments.md`). Le sezioni sotto con numeri di giugno-agosto
> vanno **rigenerate**, non copiate.

- Due pipeline di retrieval complementari per piante RPLAN: **vision** (encoder frozen → [head] → whitening → FAISS) e **graph** (GNN allenata con InfoNCE), sulla **stessa** gallery e le stesse query.
- **Ground truth di rilevanza architettonica scomposta per asse** (composizione/topologia/geometria) derivata dai metadati `.mat`, come *proxy* trasparente in assenza di label umane.
- **Protocollo di valutazione de-saturato**: diagnosi della saturazione delle metriche ingenue + redesign con ambito sull'intera gallery e metriche per-asse, con **floor casuale** e test appaiati.
- **Robustezza a query incomplete come criterio di selezione**: AUC del self-recovery mediata su **tre danni che tolgono davvero informazione** (stanze tolte coi muri, crop, patch), fissata prima della griglia; sceglie la config vision su 53 (`pespatial/gem/whiten`) con un costo piccolo e dichiarato sulla pianta intera (§6.7). ~~Ribalta la classifica fatta sulla pianta intera (§6.6)~~ — era misurato con un danno difettoso, vedi sotto.
- **Un artefatto di rendering trovato e misurato** (§6.7, Caveat 8): il masking a stanze storico lasciava i muri interni della stanza tolta; valeva **quasi tutta** la robustezza misurata del vision e tutta la vittoria della projection head, che su crop e patch **perde** contro il frozen.
- Risultati negativi riportati come tali: il self-recovery del graph **crolla** sotto masking (`status.md §30`); la head vision **non si trasferisce** a danni diversi da quello di training, né col render difettoso né rifatta col render corretto su un altro encoder (`status.md §37, §48`); la circolarità non risparmia nessun asse né ramo (§5.5).

---

## 2. Introduzione & Motivazione

- Workflow architettonico reale: si parte da layout **parziali/incompleti** → retrieval deve completare semanticamente, non solo somigliare. `[DECISION]`
- **La similarità visiva ≠ rilevanza architettonica.** Un encoder su raster cattura forma/colore; per un architetto contano funzione e **topologia** (connessioni, accessi, circolazione). Due piante possono *sembrare* simili ed essere funzionalmente diverse (e viceversa). → fonte: `.claude/shared/retrieval.md`. `[DECISION]`
- Perché RPLAN per primo: fornisce direttamente raster + metadati strutturali `.mat` (composizione/topologia/geometria) → consente sia il retrieval visivo sia la ground truth strutturale.

---

## 3. Related Work / Background

Da citare (dettaglio bibliografico in memoria `reference-relevance-literature`; **verificare le citazioni esatte**):
- **Rappresentazioni graph-based di piante**: Graph2Plan; HouseGAN++; modelli di layout transformer/diffusion (LayoutDM). `[VERIFY]`
- **Retrieval di piante (CBIR)**: DANIEL/ROBIN (Sharma et al., ICDAR 2017) — primo CBIR deep per piante. `[VERIFY]`
- **Rilevanza/similarità tra piante**: **SSIG** (Vidanapathirana et al., 2023, arXiv 2309.04357) — relevance = **IoU + GED**, validata con user study; il giudizio umano correla con la **Graph Edit Distance** sul grafo delle stanze, *non* con la pura forma. `[VERIFY]`
- **Teoria architettonica**: **Space Syntax / Justified Plan Graph** (Hillier & Hanson 1984; Lee, Ostwald & Gu 2018) — misure configurazionali (depth, integration, connectivity) validate sull'uso umano → fondamento teorico dell'asse topologico. `[VERIFY]`
- **Vision encoders** usati nel benchmark: DINOv2, DINOv3, SigLIP2, RADIO, I-JEPA (vedi §6.3).
- **GED** (Graph Edit Distance): numero minimo di modifiche (nodi/archi) per trasformare un grafo nell'altro; mappa diretta sui campi RPLAN (`rType`=nodi, `rEdge`=archi).

> Tesi di fondo del paper (da rendere esplicita): **la rilevanza ha assi separabili**
> (geometrico vs topologico) e gli encoder li catturano in modo diverso.

---

## 4. Dataset (RPLAN)

`[VERIFY]` tutti i numeri contro `src/data/rplan_metadata.py` + log:
- ~**67.453** PNG in `snapshot_train/` (gallery unica = anche pool di query).
- Metadati `.mat`: **67.405/67.453 (99,9%)** delle piante hanno una struct → join via `name` = stem del PNG. ~0,1% senza metadati (esclusi dalla rilevanza).
- I `.mat` sono **3 file aggregati** (`data_{train,valid,test}.mat` = **56.511 / 12.108 / 12.110**, disgiunti, union 80.729), non uno per pianta.
- **13 tipi di stanza** (`ROOM_TYPES` in `rplan_metadata.py`): LivingRoom, MasterRoom, Kitchen, Bathroom, DiningRoom, ChildRoom, StudyRoom, SecondRoom, GuestRoom, Balcony, Entrance, Storage, Wall-in.
- Campi `.mat` usati: `rType` (tipi), `rEdge` (archi i,j,relazione 0..9), `gtBoxNew` (bbox per stanza), `gtBox` (+footprint globale), `boundary` (contorno + ingresso). Coordinate su griglia 256.
- **`[DECISION]` Split ufficiali (verificato giugno 2026):** ⚠️ `snapshot_train/` **NON è il solo training** nonostante il nome: i 67.453 PNG **mescolano** i 3 split (47.126 train + 10.152 valid + 10.127 test + 48 senza `.mat`). La gallery di retrieval resta l'**intero** snapshot (non si splitta il corpus di ricerca); la valutazione **restringe le query** allo split val/test (`eval.split`, lookup `get_split`/`RoomMeta.split`) per evitare il *fishing* nelle ablation. Per il training di Fase 3 il pool sarà lo split `train` (disgiunto da valid/test). Usare lo split **ufficiale** dà riproducibilità e confrontabilità con la letteratura RPLAN.

---

## 5. Metodo

### 5.1 Pipeline vision retrieval `[DONE]`
Fonte: `src/vision/models/retrieval_model.py`, `src/vision/models/vision_model_manager.py`.
- Encoder **frozen** → embedding (pooling configurabile `natural|mean|gem`) → **PCA whitening** opzionale (centering + decorrelazione + scaling a varianza unitaria + L2, con riduzione facoltativa a `whitening.dim` componenti) → **FAISS `IndexFlatIP`** (inner product = cosine su vettori L2-normalizzati).
- Motivazione whitening: su colormap RPLAN gli score DINOv2 collassano in `[0.97, 0.99]`; il whitening li "spalma" e rende il ranking discriminativo. Le **stesse** trasformazioni si applicano alla query. `[VERIFY]` (numeri std: log mostra 0.0080 → 0.0361).
- `save_dir` namespaced per encoder **e variante** (`embeddings/vision/${model.name}/${model.variant}`) → indici separati per ogni punto di sweep.

### 5.2 Ground truth di rilevanza per-asse `[DONE]` `[DECISION]`
Fonte: `src/evaluation/relevance.py` (`GalleryAxes`), `src/data/rplan_metadata.py`.
- **Perché un proxy**: per RPLAN **non esistono label di rilevanza umane** → qualsiasi definizione è un proxy. La forza sta nell'essere **trasparente e scomponibile**, non in una formula.
- **Tre assi indipendenti** (ognuno in [0,1]):
  - **Composizione**: Weighted Jaccard sull'istogramma dei 13 tipi (`composition_vector`).
  - **Topologia**: Weighted Jaccard sull'adiacenza **per tipo** di stanza, **invariante alla permutazione**; vettore denso di **91 coppie** (`topology_vector`).
  - **Geometria**: media di area-footprint, aspect ratio, distribuzione di area per tipo.
- **`[DECISION]` Niente score fuso / niente pesi / niente soglia assoluta.** Motivazione: i pesi (vecchi 0.5/0.3/0.2) sono arbitrari e indifendibili; la decomposizione per-asse è più onesta e più informativa, ed elimina il problema dei pesi.
- **Classi di equivalenza esatta** (per gli assi discreti): rilevante = *stesso istogramma* (composizione) / *stessa adiacenza* (topologia). La dimensione del set rilevante **emerge dai dati**, nessun M da scegliere. (`composition_relevant`, `topology_relevant`).
- **Da dire nel paper**: perché Weighted Jaccard (∈[0,1], gestisce i conteggi — 2 bagni ≠ 1 bagno — penalizza sia il mancante sia l'eccesso).
- **`[TODO]` rifinitura**: validare il proxy contro un *piccolo* set di giudizi umani (correlazione), **senza allenarci sopra** (allenare i pesi sarebbe circolare; per 3 scalari basterebbe comunque una logistic/learning-to-rank, non una rete).

### 5.3 Metriche `[DONE]` `[DECISION]`
Fonte: `src/evaluation/metrics.py`, orchestrazione `src/evaluation/evaluate.py`.
- **Ambito = intera gallery** (non solo i top-k recuperati). È la correzione chiave: senza un insieme di rilevanti definito sui 67k, la metrica è circolare e non può fallire.
- **nDCG@K = primaria**: rilevanza **graduata** come gain, **IDCG sui migliori K della gallery**. Niente M, niente soglia → copre **tutti e 3** gli assi (geometria continua compresa).
- **Recall@K + mAP@K = secondarie**, solo assi **discreti** (composizione, topologia): rilevanti = classe di equivalenza esatta, normalizzazione **`min(K, |R|)`**; query **singleton escluse e contate** a parte.
- **MRR rimossa** `[DECISION]`: satura sugli assi densi (con classi grandi il primo hit è quasi sempre al rango 1) → niente segnale.
  - ⚠️ **Da non confondere nel testo**: la MRR è rimossa **come metrica per-asse**. Il **reciprocal rank del self-recovery** (§5.4) è un'altra domanda — una sola risposta giusta, la pianta originale — ed è l'endpoint della robustezza. nDCG@10 resta la metrica della pianta intera, del costo dichiarato e dello spareggio; Recall e mAP non entrano nella scelta.
- **K ∈ {1, 5, 10, 100}** (`configs/vision_retrieval.yaml`, `eval.k_values`); num_queries=2000, seed=42. `[VERIFY]`
- **⚠️ Caveat metodologico da scrivere esplicitamente** (vedi §7).

### 5.4 Partial layout retrieval `[DONE]` (vision) · `[DONE]` (graph, C.0)
Fonte: `.claude/shared/retrieval.md § Partial retrieval`.
- Si rimuovono **stanze** (strategia `random`, frazione f), con bordo aperto nel vision; il graph toglie **le stesse stanze** (stessa funzione di selezione, gallery condivisa).
  - ⚠️ **Render storico difettoso** (`status.md §35`): cancellava solo i muri scuri (soglia ≤ 120 → il perimetro, grigio 79); le **pareti interne** (grigio 128) restavano → la stanza tolta diventava una cella bianca contornata (50% delle stanze tolte). Misurava «stanze **svuotate**» (tipo perso, forma visibile), non «stanze tolte».
  - `[DECISION]` Nessuna correzione sul posto (i numeri storici restano riproducibili): modalità nuova **`nowalls`** (`src/vision/data/vision_damage.py`, `status.md §36`). **Stesse stanze tolte** (rng per query) e stesso riempimento; il muro diventa «grigio neutro non-sfondo» → celle chiuse **50% → 5%**. Limite dichiarato, non aggirabile sull'immagine: resta la **forma del buco** (i colori delle vicine finiscono dove finiva la stanza); toglierla vorrebbe dire ridisegnare un'altra pianta dal `.mat`.
- Ground truth doppia: **self-recovery** (reciprocal rank della pianta originale, self dentro, 0 oltre il 100° posto) + **per-asse** contro la pianta completa (self fuori, come nel full — B.4).
- **Tre danni sul vision** a livelli f ∈ {0.25, 0.5, 0.75} (`roadmap.md §2` passo 2): **stanze tolte coi muri** (`nowalls-random`, f = frazione di stanze; area media tolta 0.29/0.52/0.71), **crop** = un rettangolo bianco a lati e posizione casuali, accettato se copre la frazione f dei **pixel della pianta** ±5%, **patch** = patch ViT intere sbiancate dopo il resize, stessa frazione di area. Il graph non ha pixel: solo stanze.
- **Metro di robustezza** (`status.md §38`, fissato prima della griglia): per ogni danno AUC di self-recovery = media del reciprocal rank sui tre livelli (pesi uguali); **R** = media per query delle tre AUC, poi media sulle query. Delta appaiato con CI 95% bootstrap; spareggio (CI che contiene 0) → nDCG@10 full di topologia → config più economica. f=0.0 escluso (tetto dei dati: duplicati RPLAN). Strumento: `python -m src.evaluation.robustness_auc rank|compare --robust`. ~~Sintesi sul solo `random` storico~~ (§23-§24): superata.

### 5.5 Ramo graph `[DONE]` `[VERIFY]`
- Grafi PyG dai `.mat` → **GCN / GAT / GraphSAGE** addestrate con **InfoNCE contrastive** → FAISS. Implementate, allenate e valutate (memoria `project-graph-pipeline`, `project-gnn-requirement`).
- Requisito prof soddisfatto: l'encoder finale è una **GNN addestrata**; il descrittore training-free (istogramma dei tipi) resta come **baseline** — ⚠️ sulla composizione è un oracolo per costruzione.
- Da riportare nel paper: la selezione del checkpoint usa una **sonda di retrieval sul valid**, non la val-loss InfoNCE (che ne inverte la classifica), e il denominatore onesto è la **GNN a pesi casuali**, non solo la baseline.
- **Fusione** `[DONE]` (§6.8): late fusion per concatenazione pesata dei vettori L2 `[√α·v ; √(1−α)·g]`, α scelto sul valid sulla robustezza (self-recovery, stanze tolte) con regole pre-registrate (`status.md §49`); vision **`pespatial/gem/whiten`**, graph **`gat/asymrob`** (§47), **α* = 0.6** — la fusione aiuta e supera anche l'oracolo per query. Le fusioni intermedie (testa congiunta, O5 b-d) restano future work (§9).
- **Graph sotto masking** (`status.md §30`): il self-recovery crolla già a f=0.25 — un grafo parziale è il grafo completo plausibile di altre piante; il retrieval per-asse resta sopra il floor. Tentativo dichiarato: coppie di training asimmetriche (opzione D, §30.6) → **adottato** (§30.8): la robustezza sale di molto ma resta lontana dal vision, e si paga sul full soprattutto in **topologia** — l'asse su cui il graph era forte (previsione smentita, da scrivere così).
- ⚠️ **Circolarità** da dichiarare — **nessun asse è esente e nessun ramo è pulito** (misurato 24 ago, `status.md §21`): (a) su composizione/topologia il grafo riceve in input ciò che la GT misura; (b) **anche la GT geometrica** lo è (`gtBox[-1]` = unione esatta dei `gtBoxNew`, 100% su 24.218 piante); (c) **anche il vision legge `rType`**, dal colore delle PNG, ma compresso 13 tipi → **6 colori**. Il confronto va presentato come **fine vs grosso sullo stesso input**, non come «simbolico vs visivo».
- Numeri: `.claude/shared/status.md`; versione discorsiva: `spiegazione_metriche_e_risultati_grafi.md`.

---

## 6. Esperimenti & Risultati

### 6.1 Setup `[VERIFY]`
- **Tutti e 5 gli encoder verificati end-to-end** su GPU (run 29 giu 2026), input **224px**, frozen → whitening → FAISS. Dim embedding di output: DINOv2/DINOv3/SigLIP2 **768-d**, I-JEPA **1280-d** (ViT-H), RADIO **2304-d** (summary multi-teacher). DINOv3 gated (auth HF, poi in cache).
- ⚠️ (giugno) **2000 query** campionate dallo split **test** (seed 42). **Setup finale = protocollo B**: gallery condivisa (inner join), query dal **valid** per scegliere, test una volta sola; whitening stimato sul solo train (`experiments.md § Protocollo B`). VRAM: **16 GB bastano per tutti** (I-JEPA ViT-H incluso, batch 256).

### 6.2 Diagnosi della saturazione (contributo metodologico) `[DONE]`
**Da raccontare come motivazione del redesign** — è un risultato, non solo un bugfix:
- Metriche ingenue (score fuso + soglia 0.5, valutate solo sui top-k): Recall@5 = 1.0, mAP = 0.99, nDCG = 0.97 → **sature**.
- **Prova quantitativa**: su 2000 coppie **casuali** (indipendenti dal retriever) il **96% supera la soglia 0.5** → "rilevante". Quindi anche un retriever casuale avrebbe Recall@5 ≈ 1. (script di misura: scratchpad, **rigenerabile**).
- Cause: (1) **ambito circolare**, (2) **etichetta troppo permissiva** (geometria ~0.85 quasi costante e già vista dall'encoder; composizione ~0.74 perché tutte le piante condividono il set base di stanze; topologia ~0.46 = unico asse discriminante).

### 6.3 Risultati DINOv2 per-asse `[DONE]` — **rigenerare prima del paper**
Run del **29 giu 2026** (DINOv2, split test, **n=2000**). **NON copiare ciecamente: rilanciare e riverificare.**

| K | asse | nDCG | Recall | mAP |
|---|---|---|---|---|
| 1 | composition | 0.862 | 0.458 | 0.458 |
| 1 | topology | 0.724 | 0.374 | 0.374 |
| 1 | geometry | 0.949 | — | — |
| 5 | composition | 0.836 | 0.328 | 0.273 |
| 5 | topology | 0.676 | 0.251 | 0.217 |
| 5 | geometry | 0.939 | — | — |
| 10 | composition | 0.826 | 0.287 | 0.211 |
| 10 | topology | 0.659 | 0.216 | 0.173 |
| 10 | geometry | 0.935 | — | — |
| 100 | composition | 0.800 | 0.201 | 0.095 |
| 100 | topology | 0.623 | 0.173 | 0.106 |
| 100 | geometry | 0.925 | — | — |

- Topologia: **313/2000 query escluse** (classe singleton, ~15,6%).
- **Lettura**: ordinamento nDCG **geometria > composizione > topologia** su tutti i K → DINOv2 cattura forma bene, topologia peggio. **Sanity anti-saturazione**: random composizione ≈ 6%, DINOv2 ≈ 29% @K=10 (~5×).

### 6.4 Benchmark multi-encoder `[DONE]` — **rigenerare prima del paper**
5 encoder **scelti per coprire assi diversi** (proposto dal team, confermato dai prof):

| Encoder | Famiglia | Ruolo | Pooling |
|---|---|---|---|
| DINOv2 | self-supervised | baseline self-sup | [CLS] |
| DINOv3 | self-supervised | generazione successiva | [CLS] |
| SigLIP2 | vision-language | semantica guidata dal linguaggio | attention pooling |
| RADIO | agglomerativo | distilla DINOv2+CLIP+SAM | summary token |
| I-JEPA | self-sup predittivo | vicino al partial retrieval | mean-pool patch |

**Run del 29 giu 2026** (full mode, query = split **test**, **n=2000**, seed 42; eval-only, indici riusati). nDCG@10 per asse (+ Recall@10/mAP@10 per gli assi discreti):

| Encoder | comp nDCG | comp R@10 | comp mAP@10 | topo nDCG | topo R@10 | topo mAP@10 | geo nDCG |
|---|---|---|---|---|---|---|---|
| DINOv2 | 0.826 | 0.287 | 0.211 | 0.659 | 0.216 | 0.173 | 0.935 |
| DINOv3 | 0.824 | 0.285 | 0.207 | **0.662** | 0.218 | 0.174 | 0.934 |
| SigLIP2 | 0.822 | 0.273 | 0.197 | 0.637 | 0.195 | 0.156 | 0.928 |
| RADIO | 0.812 | 0.252 | 0.177 | 0.630 | 0.181 | 0.145 | 0.935 |
| I-JEPA* | 0.826 | 0.290 | 0.209 | 0.654 | 0.199 | 0.158 | **0.943** |

(*I-JEPA = ViT-H, vedi nota capacità.) **Stabilità**: i numeri a n=2000 coincidono con quelli del primo giro a n=200 entro **≤0.008** → stime affidabili, ranking non rumore.

**Letture (tutte stabili a n=200 e n=2000):**
- Pattern per-asse **geometria ≫ composizione > topologia** identico per tutti e 5 → robusto, **motiva il ramo graph** (la topologia è l'asse dove la visione pura è più debole).
- **Terzetto di testa DINOv2 ≈ DINOv3 ≈ I-JEPA**, appaiati su composizione/topologia (entro ~0.003–0.008): su questi assi **non si incorona un vincitore**.
- **I-JEPA nettamente migliore sulla geometria** (0.943 vs ~0.934), distacco identico a n=200 → **reale**, coerente col pretraining predittivo/spaziale (+ ViT-H).
- **SigLIP2** sotto il terzetto (soprattutto topologia); **RADIO** il più debole su comp/topo (pareggia solo la geometria).
- **Conclusioni forti**: (1) **DINOv3 non batte DINOv2** → pretraining più nuovo non si trasferisce al dominio planimetrico; (2) **I-JEPA ViT-H (~7× i parametri di un ViT-B) guadagna solo sulla geometria** → collo di bottiglia = **domain gap**, non la taglia; (3) né il vision-language (SigLIP2) né l'agglomerativo (RADIO) aiutano qui. Encoder di **default pragmatico**: DINOv2 ViT-B.
- **Ipotesi confermata**: la **topologia separa gli encoder più della geometria** (spread nDCG@10 ~0.032 vs ~0.015).

- **`[DECISION]` Equità — due "taglie" da non confondere.**
  1. **Capacità del backbone** (n. parametri / classe ViT): DINOv2, DINOv3, SigLIP2 e **RADIO** (`C-RADIOv2-B`) sono **tutti ViT-Base** → capacità *appaiata*. **I-JEPA è l'unica eccezione** (nessun checkpoint ViT-B ufficiale → `ijepa_vith14_1k` ViT-H): divario di capacità **da dichiarare**.
  2. **Dimensione dell'embedding di output** (larghezza del vettore): cosa **diversa** dalla capacità — i 768-d di DINOv2/v3/SigLIP2 convivono coi **2304-d di RADIO** (la summary multi-teacher è più larga, *non* è più capacità del backbone) e i **1280-d di I-JEPA**. **Non confonde le conclusioni**: RADIO ha l'embedding più largo ma è il **peggiore** su comp/topo → la larghezza non dà vantaggio. Per azzerare anche questo confondente: riduzione PCA `whitening.dim=768` (ablation già pronta, §6.5).
  - Le **varianti di taglia** (small/large/ViT-L/H) **non** si confrontano (sporcherebbero gli assi veri).

### 6.5 Ablation / Sensitivity — frozen `[DONE infra]`, da lanciare
Tutte **config-driven**, **un-asse-alla-volta** rispetto a una baseline per encoder, riportate **per-asse** (C/T/G); sweep via **override CLI** (`model.variant=... model.kwargs.<knob>=...`), output namespaced per `model.variant`. `[DECISION]`
- **Pooling**: `natural | mean | gem` (`pool_global`/`gem_pool`, `gem_p`). CLS↔mean↔gem pieno solo DINOv2/v3; gli altri naturale-vs-gem.
- **PCA whitening**: `whitening.enabled` + `whitening.dim` (top-D componenti; neutralizza anche le differenze di dim nativa di **output** — RADIO 2304, I-JEPA 1280 vs 768 — riducendo tutti a una larghezza comune).
- **Risoluzione**: `model.kwargs.image_size` (SigLIP2: via cambio `hf_name` -224/-256/-384, non `image_size`).
- **Extraction layer**: `extraction_layer` (-1 ultimo, -2 penultimo…) su DINOv2/v3/I-JEPA; non applicabile a SigLIP2/RADIO.
- **Sensitivity su K** (e su eventuale rilassamento topologico) → mostrare che il *ranking degli encoder* è robusto (sostituisce la "scelta del peso giusto").
- [Opz.] **GED topologica** stile SSIG come alternativa/affianco al Jaccard d'adiacenza.
- **`[DONE infra Fase 3]`** **projection-head** allenabile (default per tutti i 5, backbone frozen, embedding cachati). **Nodo coppie positive risolto con A+ self-supervised** (non circolare): positivo = stessa pianta **degradata** (masking) + augmentation valide (flip/rot90); loss **InfoNCE** (τ, V configurabili). Scartate le alternative circolari (positivi dalla rilevanza per-asse = ciò che valutiamo). Pipeline **disaccoppiata**: embedding RAW salvati una volta, whitening/head applicati al volo da `prepare_index` → si valutano `raw/whiten/whiten768/head/head+whiten` riusando lo stesso RAW. Training su **train+valid** (test held-out). Moduli: `projection_head.py`, `projection_pairs.py`, `train_projection.py`; ordine sbatch in `COMANDI.md`. **LoRA** sul solo encoder estremo **ancora rimandata** (decisione dopo i numeri della head). Da lanciare via sbatch.
  - ⚠️ **15 set**: le viste degradate delle coppie e la probe di selezione usano il **render storico** coi muri interni rimasti (`projection_pairs.py:92`, `retrieval_probe.py:62`): la head ha imparato anche quell'artefatto (Caveat 8). Col metro a tre danni è 17ª su 53 (§6.7).

### 6.6 Robustezza a query incomplete — **il criterio di selezione del progetto** `[DONE]`

> ⛔ **SUPERATA dal 15 set → §6.7.** Tutta questa sezione è misurata col masking a stanze
> **storico**, che lasciava i muri interni (§5.4). Il punto 2 sotto (la head vale +0.284 e
> generalizza) **non regge**; i numeri del punto 1 vanno rifatti (l'idea «full ≠ incompleto»
> resta, §6.7 punto 4); il punto 3 (fra i frozen vince `pespatial`) regge. Resta utile come
> storia del criterio e come «prima» del confronto prima/dopo il bug.

Numeri: `status.md §23` (criterio) e **`§24`** (griglia, 24 ago). 320 run
per-query sul valid, 80 configurazioni × 4 frazioni di masking, appaiamento
verificato su tutte.

⚠️ **§24 è il protocollo vecchio** (gallery 67.453, self nelle metriche
per-asse, whitening trasduttivo). Sul protocollo B il verdetto **regge** (§31:
`dinov3/natural/head` resta primo, delta sul secondo con CI che esclude 0), ma i
numeri cambiano e la head sarà sostituita da quella di B.6: la tabella qui sotto
è la classifica a 80 config, **non** la tabella finale.

**Definizione operativa, fissata prima delle run**: «migliore» = **più robusto**
= media di MRR di self-recovery su f ∈ {0.25, 0.5, 0.75}, strategia `random`,
delta appaiato con CI 95% bootstrap sui valori per query. `f=0.0` **escluso**:
tutte le 80 config stanno fra 0.9705 e 0.9712 (tetto dei **dati** — duplicati
esatti in RPLAN — non dei modelli).

**Risultato — la tabella da mettere nel paper:**

| configurazione | AUC robustezza | f=.75 | full C/T/G |
|---|---|---|---|
| **`dinov3/natural/head`** (scelta) | **0.8157** | 0.6665 | 0.8284/0.6416/0.9349 |
| `dinov3/mean/head` | 0.7557 | 0.5883 | 0.8215/0.6177/0.9389 |
| `pespatial/natural/whiten` (miglior **frozen**) | 0.6360 | 0.3035 | 0.8072/0.6113/0.9430 |
| `tipsv2/gem448/whiten` (miglior sul **full**) | 0.2831 | 0.0622 | 0.8433/0.6896/0.9347 |

Tutti i delta appaiati hanno CI che escludono lo zero.

**I tre punti che il testo deve fare:**
1. **Il criterio ribalta la classifica.** Il migliore a immagine intera è **49° su
   80** per robustezza (2.9× di distanza). «Bravo sul completo» e «bravo
   sull'incompleto» sono qualità diverse: va detto, con il **costo dichiarato**
   della scelta (topologia −0.048 sul full).
2. **La head vale +0.284** a encoder fisso, ed è il fattore dominante — più
   dell'encoder. Generalizza **fuori distribuzione**: allenata con masking
   ≤ 0.5, il suo vantaggio è massimo a **f=0.75**.
3. **Fra i soli frozen vince `pespatial`** (+0.104 su dinov3): è lì che il
   confronto *fra encoder* è equo.

⚠️ **Questa sezione sostituisce la conclusione «sul full la head non aiuta»**
(§6.4, luglio) come criterio di scelta. Le due non sono in contraddizione: sul
full la head è neutra, sul partial è decisiva. Cambia il **metro**, non i numeri.

### 6.7 Robustezza con danni che tolgono davvero informazione — **la tabella finale del valid** `[DONE]`

Numeri: `status.md §37` (ondata 1), **`§38`** (metro, pre-registrato), **`§43-§44`** (griglia
completa, 15 set). Valid, n=2000, gallery condivisa, protocollo B. **53 config** = tutte le 52
frozen (8 encoder × pooling × risoluzione × raw/whiten) + la head congelata prima. Nessuna
estrapolazione da un encoder agli altri. Prima del paper: rigenerare con `rank --robust`.

| # | configurazione | **R** | stanze tolte | crop | patch |
|---|---|---|---|---|---|
| 1 | **`pespatial/gem/whiten`** (scelta) | **0.5245** | 0.3920 | 0.6605 | 0.5209 |
| 2 | `radio/natural/whiten` | 0.5161 | 0.3821 | 0.6032 | 0.5631 |
| 3 | `pespatial/natural/whiten` | 0.5148 | 0.3707 | 0.6855 | 0.4881 |
| 17 | `dinov3/natural/head-probe-conv` (la head scelta prima) | 0.2915 | 0.3942 | 0.2017 | 0.2785 |

#1 − #2 = +0.0084 [+0.0030, +0.0138]; tutti i delta del #1 hanno CI che esclude 0 → nessuno
spareggio. Il migliore per singolo danno è ogni volta diverso (stanze → head, crop →
`pespatial/natural`, patch → `radio/natural`): è il motivo della media sui tre.

**Costo sulla pianta intera** (nDCG@10, `pespatial/gem/whiten` = 0.8130 / 0.6306 / 0.9377):

| rispetto a | composizione | topologia | geometria |
|---|---|---|---|
| head scelta prima | −0.0163 [−0.0191, −0.0135] | −0.0164 [−0.0203, −0.0126] | −0.0020 [−0.0030, −0.0010] |
| `dinov3/natural/whiten-train` | −0.0120 [−0.0145, −0.0095] | −0.0321 [−0.0357, −0.0287] | +0.0030 [+0.0020, +0.0039] |

**I punti che il testo deve fare:**
1. **I muri rimasti valevano quasi tutta la robustezza** (§37). Stesse query, stesse stanze
   tolte, unica differenza i muri: il self-recovery della head a f=0.75 passa da **0.840 a
   0.036**. «Il vision è robusto al masking» era un artefatto del rendering (Caveat 8).
2. **La head non si trasferisce** (H1, pre-registrata in §34, **falsificata**). Head − frozen
   dello stesso encoder: **−0.183** su crop, **−0.164** su patch (CI < 0); resta avanti solo
   sulle stanze tolte (+0.090), il danno più vicino al suo training. Da 1ª a **17ª**.
3. **Cambia la trasformazione, non l'encoder**: `pespatial` era già il miglior frozen col metro
   vecchio (§6.6 punto 3). I vincitori per encoder non cambiano, tranne tipsv2 (224 invece di
   448: +0.0111 [+0.0061, +0.0161]). **Whitening > raw in 26/26** coppie encoder × pooling.
4. **Il prezzo è piccolo e dichiarato**: ~1.6 punti di composizione e topologia rispetto alla
   head, geometria praticamente pari. «Bravo sul completo» e «bravo sull'incompleto» restano
   qualità diverse: il migliore sul full del protocollo vecchio (tipsv2) resta fuori dalle
   prime 14 su 53 (i numeri del punto 1 di §6.6 vanno rifatti).
5. **H2 — a parità di area conta l'informazione o l'artefatto?** (pre-registrata, §34): a f=0.75
   la patch fa **peggio** del crop in **7 encoder su 8** (1 pari) → «artefatto»: la griglia di
   tasselli bianchi disturba l'embedding più di quanto aiutino i pezzi di stanza rimasti.
   ⚠️ A f=0.75 i valori sono vicini al pavimento, e a danno moderato il segno era opposto
   (dinov3, §37): il verdetto vale per il livello pre-registrato.
6. **Vision vs graph a informazione più vicina** (§37): AUC sulle stanze tolte, vision ~0.39
   (head 0.3942, `pespatial/gem/whiten` 0.3920) vs graph `gat/asym` **0.2490**: il divario
   scende da +0.67 a circa +0.14. Residuo di protocollo: il vision lascia la forma del buco, il
   graph perde nodo e archi. Delta appaiato con la config nuova: da calcolare in T3.
7. **Anche col render corretto la head non si trasferisce** (`status.md §46` pre-registrata, **§48**).
   Head rifatta da zero su `pespatial/gem` con coppie a stanze tolte coi muri, epoca scelta solo su
   quel danno; candidato migliore head+whitening (R 0.4912). Rispetto al frozen: stanze tolte
   **+0.147** (in distribuzione), crop **−0.142**, patch **−0.106**, R −0.033 (tutti CI esclusi 0)
   → **non adottata**, config definitiva `pespatial/gem/whiten`. Sul full: composizione pari,
   topologia −0.018, geometria −0.009. Da scrivere: la head impara il danno che vede, non una
   robustezza generale; il difetto dei muri gonfiava il vantaggio ma non spiega la mancata
   trasferibilità (stesso segno con dinov3 e render vecchio, §37).

---

### 6.8 Late fusion e i due controlli (valid) `[DONE]`

Numeri: `status.md §49-§51.1`. **Metodo**: concatenazione pesata dei vettori L2-normalizzati dei due rami
(vision dopo il whitening fittato sul solo train); le query danneggiate sono ricalcolate in entrambi i rami
**con le stesse stanze tolte**. α scelto sul valid con regole fissate **prima** dei job (§49): criterio = AUC di
self-recovery sulle stanze tolte, α* = valore centrale dell'insieme dei pari (CI del delta col migliore che
contiene 0).

**Fusione vision + graph** (vision `pespatial/gem/whiten`, graph `gat/asymrob`):

| α | 0 (graph) | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | **0.6** | 0.7 | 0.8 | 0.9 | 1 (vision) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC stanze tolte | 0.4562 | 0.5029 | 0.5436 | 0.5797 | 0.6049 | 0.6213 | **0.6301** | 0.6258 | 0.5922 | 0.5159 | 0.3920 |

- α* = 0.6, unico pari. Verdetto **«la fusione aiuta»**: α* − graph (ramo migliore) **+0.1740 [+0.1642, +0.1838]**;
  α* − vision +0.2381 [+0.2285, +0.2476]. ~4× la soglia di rumore fra training del graph (0.04, §47).
- **MRR per livello** (graph / vision / fusione): f=0.25 0.8547 / 0.8181 / **0.9553** · f=0.5 0.4258 / 0.2971 /
  **0.7201** · f=0.75 0.0880 / 0.0609 / **0.2151**.
- **Oracolo per query** (descrittivo, **non** un tetto): 0.5557; la fusione lo supera di **+0.0744**, su
  **1255/2000** query. Meccanismo verificato (§49.3): di norma **un** ramo tiene la pianta sorgente vicino alla
  cima e l'altro abbassa i concorrenti che la precedevano.
- **Pianta intera** (nDCG@10 C/T/G, **descrittivo e circolare**, §5.5): α*=0.6 0.8501/0.6841/0.9536 · graph
  0.8603/0.6780/0.9498 · vision 0.8130/0.6306/0.9377. α* − graph: composizione **−0.0102** [−0.0125, −0.0078] ·
  topologia **+0.0061** [+0.0028, +0.0094] · geometria **+0.0039** [+0.0032, +0.0045].
- **Previsioni pre-registrate**: fusione > entrambi i rami sotto danno ✅ · composizione sotto il graph ✅ ·
  topologia sotto il graph ❌ **smentita** (è sopra, +0.006) · geometria pari o sopra ✅.

**Controllo 1 — graph + graph** (`gat/asymrob` + la replica `asymrobrep`, §50): stessa informazione, stesso
modello, cambia solo il seme del training. β* = 0.4 (pari {0.3, 0.4, 0.5}), guadagno sul componente migliore
G_C = **+0.0336 [+0.0286, +0.0387]**. D = G_F − G_C = **+0.1403 [+0.1287, +0.1519]**; quota G_C/G_F = 0.193.
⚠️ Prima dei job il controllo era stato descritto come «severo»: è il contrario, due training identici hanno
errori **più correlati**, quindi è il controllo **minimo** (§50.1). Pianta intera: previsione «entro ±0.01» ❌
smentita di poco sulla topologia (+0.0116).

**Controllo 2 — vision + vision** (`pespatial/gem/whiten` + `radio/natural/whiten`, §51): **stessa
informazione** (stessa immagine), modelli diversi. γ* = 0.5, G_V = **+0.0982 [+0.0916, +0.1047]**.
D₂ = G_F − G_V = **+0.0758 [+0.0639, +0.0876]**, fuori dal margine di equivalenza pre-registrato ±0.04 →
esito **complementarità**; ma quota G_V/G_F = **0.564** (previsione «G_V < 0.5·G_F» ❌ smentita).

**Scala dei tre guadagni** (sul rispettivo componente migliore): stesso modello ripetuto **+0.0336** · modelli
diversi, stessa informazione **+0.0982** · informazione diversa **+0.1740**. **Correlazione di Spearman fra i
componenti** (per query): vision/graph **0.092** · pespatial/radio 0.622 · graph/replica 0.649. **A danno zero**
(fuori dal metro): vision + vision non rompe **nessun** pareggio (MRR 0.9707 identico sugli 11 γ: i duplicati
RPLAN hanno immagini identiche), mentre vision + graph sale fino a 0.982 (§49.1).

**Sintesi da riportare**: il guadagno della fusione è in parte effetto d'insieme fra modelli diversi (circa il
56%) e in parte informazione complementare (circa il 44%): non è né solo l'uno né solo l'altro. Limite da
dichiarare: due encoder vision restano meno diversi fra loro di quanto lo siano un encoder di immagini e una
rete su grafo, quindi informazione e modello non si separano del tutto.

### 6.9 TEST finale (letto una volta, pre-registrato) `[DONE]`

Pre-registrazione: `status.md §52` (approvata il 18 set, prima dei job); risultati e controlli: `§53`. Sul test
non si è scelto nulla: config congelate e **α = 0.6 fisso dal valid**.

| stanze tolte (AUC self-recovery) | `hist` (training-free) | vision | graph | **fusione α=0.6** |
|---|---|---|---|---|
| media f = 0.25/0.5/0.75 | 0.0013 | 0.3965 | 0.4700 | **0.6424** |
| f=0.25 | 0.0026 | 0.8216 | 0.8620 | **0.9580** |
| f=0.5 | 0.0011 | 0.3037 | 0.4471 | **0.7363** |
| f=0.75 | 0.0003 | 0.0642 | 0.1008 | **0.2331** |

- **Fusione − ramo migliore (graph): +0.1725 [+0.1625, +0.1826]** — sul valid era +0.1740, quindi l'ottimismo
  della scelta di α è risultato trascurabile. Oracolo per query 0.5693: la fusione lo supera di **+0.0731**, su
  **1218/2000** query.
- **Vision col metro a tre danni**: R **0.5210** (nowalls 0.3965 · crop 0.6543 · patch 0.5123); valid 0.5245.
- **Pianta intera** (nDCG@10 C/T/G, descrittivo e circolare): `hist` 0.9998\*/0.6428/0.8936 (\*oracolo per
  costruzione) · graph 0.8585/0.6748/0.9494 · vision 0.8144/0.6328/0.9372 · **fusione 0.8512/0.6856/0.9541**.
  Fusione − graph: composizione **−0.0073** [−0.0096, −0.0049] · topologia **+0.0107** [+0.0074, +0.0141] ·
  geometria **+0.0047** [+0.0040, +0.0053].
- **Controlli di validità**: tutti PASS — stesse stanze e stesse query, α=1 e α=0 riproducono i due rami
  **esattamente** (0/2000 self_rr diversi a ogni livello), MRR senza danno 0.970-0.979. Controllo aggiuntivo sul
  doppio rendering del danno vision: 0/1/0 query diverse su 2000 (pareggi).
- **Previsioni di §52**: rami entro ±0.01 dal valid → vision ✅ e metro a tre danni ✅, **graph ❌** (+0.0138, fuori
  di poco e in meglio) · Δ fusione > +0.12 ✅ · ordine fusione > graph > vision ✅ · pianta intera come previsto ✅.
- **Sotto danno la baseline training-free non ritrova nulla** (0.0013): graph e fusione sono due-tre ordini di
  grandezza sopra.

⚠️ I numeri dei due controlli di complementarità (§6.8) restano del **valid**: non sono stati ripetuti sul test.

---

## 7. Discussione & Caveat (da scrivere esplicitamente — onestà metodologica)

- **Storia principale**: geometria (vista) ≫ topologia (struttura) per un encoder visivo → **motiva il ramo graph**. ⚠️ Il «≫» è in nDCG **grezzo**: i floor casuali sono molto diversi per asse (`retrieval.md § Il floor`), quindi il confronto fra assi va fatto sullo **spazio utile** normalizzato, dove il divario si riduce. Rifare la frase coi numeri del protocollo B.
- **`[DECISION]` Caveat 1 — Recall@K ≈ Precision@K per la composizione.** Con `min(K,|R|)` e classi enormi (|R|≈4000 ≫ K), `Recall@K = hits/K` = precision@K, che **cala** con K (per questo Recall@1 > Recall@100). Non è un bug: per gli assi a classe grande va letto in chiave **precision**. La topologia (classi piccole, |R|<K) resta Recall classica. → Possibile rifinitura: **rinominare** la set-metric della composizione in *Precision@K*.
- **`[DECISION]` Caveat 2 — geometria a basso range dinamico.** La similarità geometrica sta quasi sempre in [0.85, 1.0] → nDCG geometrico alto ma **poco discriminante tra encoder**. Da dichiarare: la geometria è l'asse "facile".
- **Caveat 3 — topologia sparsa.** Match esatto → 15% di piante singleton (gallery-wide); per Recall/mAP si escludono, ma **nDCG copre tutte le query**. Rilassamento (Jaccard alto / GED≤1) come opzione futura.
- **Caveat 4 — il proxy non è ground truth umana.** Dichiarare il limite + (se fatto) la validazione su un piccolo set umano.
- **`[DECISION]` Caveat 5 — il test è stato letto tre volte, e va dichiarato.** Il protocollo (vincolo DURO 1) prevede una sola lettura del test, per il numero finale. Il ramo vision ne ha prodotte **tre**: il primario `siglip2/mean/whiten` (8 ago), **pre-registrato in `status.md §13.1` prima di vedere il test**, e due varianti aggiuntive `dinov2/natural/whiten` e `dinov3/natural/whiten` (9 ago). Il report **riporta tutte e tre** e dice esplicitamente quale era scelta prima; riportarne una sola sarebbe selezione sul test mascherata.
  - nDCG@10 test (comp/topo/geom): siglip2 **0.8259 / 0.6537 / 0.9356** · dinov2 0.8262 / 0.6590 / 0.9350 · dinov3 0.8243 / 0.6618 / 0.9342.
  - **Perché la dichiarazione non è una formalità**: il confronto **si ribalta** fra valid e test (il vantaggio di siglip2 sulla composizione sparisce, il pareggio sulla topologia diventa sconfitta significativa). Le tre varianti sono **equivalenti entro il rumore**: il margine con cui il primario era stato scelto (≈0.004) è dello stesso ordine del suo drift valid→test. Il primario resta tale **perché pre-registrato, non perché vince** (`status.md §14.1`).
  - Conseguenza da scrivere: nessun claim del tipo «il miglior encoder vision è X» sopravvive a questi dati; il claim difendibile è «le varianti di testa sono indistinguibili, e la selezione su un solo split di validazione non si replica».
- **`[DECISION]` Caveat 6 — il claim geometrico aggregato non sopravvive alla sua stessa formula** (A.4, 24 ago, `status.md §22`). `geometry_sim` è la media a pesi `(1,1,1)` di area, aspect e `type_area_distribution`, pesi scelti a priori e hardcoded (`relevance.py:137-146`). Rifacendo il gain con 6 pesature: **2 su 6 invertono il vincitore** vision/graph con CI 95% che esclude lo zero, e l'inversione (−0.033) è **più grande** del vantaggio di baseline (+0.002). Verdetto **NON REGGE**, criterio fissato *prima* dei dati.
  - **Nel paper NON si sceglie la pesatura che vince** (sarebbe selezione sul risultato): si dichiara `(1,1,1)` come convenzione e si pubblica **accanto** la tabella di sensibilità.
  - **Il claim da fare è quello decomposto**, che è più forte e più informativo: sulla **forma del footprint** (area+aspect) il vision batte il graph di **+0.019 / +0.027** — ~10× il delta aggregato — mentre perde su `type_area_distribution`.
  - **Meccanismo da scrivere**: `type_area_distribution` è una **somma** sulle feature dei nodi (esattamente ciò che l'add-pool del graph calcola gratis); area e aspect richiedono un **min/max**, che l'add-pool non fornisce. Il confine non è «visione vs simboli» ma **quale aggregazione l'architettura implementa nativamente**.
  - ⚠️ Vale anche **dentro** il ramo vision: «ijepa è il miglior encoder in geometria» inverte anch'esso su `(0,0,1)` (1 pesatura su 6, quindi più robusto ma non unanime).
- **`[DECISION]` Caveat 7 — la vittoria della head NON è un confronto fra encoder** (`status.md §24.3`). ⛔ **15 set: superato in parte da Caveat 8** — col render corretto la head non vince più (§6.7); resta valido e da scrivere il punto «nessuna fuga di dati nei pesi». La head è allenata con InfoNCE il cui positivo è *la vista mascherata della stessa pianta*: è **esattamente** il task del self-recovery su cui poi la giudichiamo. Il claim corretto è «addestrare una testa per l'incompletezza funziona, e molto», **non** «dinov3 è l'encoder migliore». Il confronto equo fra encoder è quello **dentro** il gruppo frozen.
  - **Nessuna fuga di dati nei pesi**, verificato: le coppie vengono da `train+valid` (`projection_pairs.py:123-132`) ma il gradiente fitta **solo** le righe `train` (`train_projection.py:72-76`); il valid serve alla sola val-loss di early stopping. Resta l'asimmetria normale — il checkpoint della head è *scelto* sul valid, le config frozen non hanno nulla da scegliere — e specularmente il **whitening è fittato su tutta la gallery** (rilievo A2), che valid e test li contiene.
  - ~~Il numero è un limite inferiore perché la head è selezionata sulla val-loss~~ — **superato il 10 set** (`status.md §30.3-§30.5`): con una probe partial, val-loss e probe scelgono epoche vicine; il limite era il **budget di ottimizzazione** (tetto di 60 epoche, ~12 step/epoca), e la curva della probe sale fino a ~1600 epoche. Da scrivere così: *la head era sotto-allenata, non scelta male*. ⚠️ Più training = più specializzazione sul self-recovery → il **costo sul full** della head B.6 va riportato accanto.
- **`[DECISION]` Caveat 8 — il masking a stanze storico lasciava i muri interni, e la head ci ha costruito sopra** (`status.md §35-§37`). Da scrivere come **scoperta**, col meccanismo: RPLAN disegna il perimetro in grigio 79 e le pareti interne in grigio 128; il render cancellava solo i muri ≤ 120, quindi la stanza tolta restava una cella bianca contornata (forma e posizione visibili, tipo perso).
  - **Non è barare**: nessun dato di valutazione nei pesi, InfoNCE fittata sul solo train (Caveat 7). È **shortcut learning** su un difetto condiviso da training e misura: coppie, probe di selezione e valutazione usavano lo stesso render, e il self-recovery premiava proprio la sagoma rimasta.
  - **Come è stato trovato e chiuso**: pre-registrazione di crop e patch (H1: «la head si trasferisce») → falsificata; modalità `nowalls` appaiata (stesse stanze) → i muri valevano quasi tutta la robustezza; nuovo metro fissato **prima** della griglia completa (§5.4, §6.7).
  - **Lezione metodologica da mettere nel testo**: un criterio di robustezza va validato su **più tipi di danno**; con un solo danno si misurano anche i suoi artefatti. Il claim caduto va scritto come tale: «il vision è robusto al masking» e «la head vale +0.284» valgono solo per le stanze svuotate.
- **`[DECISION]` Caveat 9 — i limiti del metro finale** (`status.md §38`), da dichiarare:
  - **Fissato dopo aver visto 2 config** (ondata 1, entrambe `dinov3/natural`: frozen 0.3771 vs head 0.2915). Nessuna delle altre 51 era nota.
  - **Il livello f non è la stessa quantità nei tre danni**: superficie per crop e patch, frazione di stanze per `nowalls` (area media ottenuta 0.29/0.52/0.71) → i pesi uguali sono sui livelli **nominali**.
  - **La griglia delle patch cambia con l'encoder** (P14 vs P16, tipsv2 a 448): il danno patch non è identico fra encoder.
  - **Il controllo pre-registrato di bit-identità non è passato alla lettera**: rieseguendo la stessa valutazione, 6/2000 query scambiano rango 1↔2 (ipotesi: non determinismo GPU fra piante duplicate). Accettato come rumore di fondo, 2-3 ordini di grandezza sotto gli effetti riportati.
- **`[DECISION]` Caveat 10 — i limiti della fusione** (`status.md §49-§51.1`), da dichiarare:
  - **Numero ottimista sul valid**: α* scelto fra 11 valori sulle stesse 2000 query su cui si misura il guadagno (e anche i due rami erano stati scelti su queste query). **Verificato sul test** (§6.9): con α* fisso il guadagno resta +0.1725, quindi l'ottimismo era trascurabile. I due **controlli** di §6.8 restano però solo sul valid.
  - **Asimmetria del danno**: a parità di stanze tolte il vision conserva la sagoma del buco, il graph perde nodo e archi.
  - **Legato a questo checkpoint del graph**: differenze sotto 0.04 non si distinguono da un riallenamento (§47); il guadagno osservato è molto sopra, ma il verdetto vale per `gat/asymrob`.
  - **Le differenze per asse sulla pianta intera sono descrittive e circolari** (§5.5): non decidono nulla.
  - **I due controlli non sono equi allo stesso modo**: in graph + graph i componenti sono più forti della fusione vera (0.4963 e 0.4562) — per questo si confrontano i **guadagni**, non i valori assoluti; in vision + vision il componente migliore è più debole (0.3920 contro 0.4562), quindi ha più margine di guadagno: il controllo parte favorito.
  - **Limite strutturale**: informazione e modello non si separano del tutto; la scala dei tre guadagni (§6.8) è indicativa, non una misura pura della complementarità.
  - **Pareggi e riproducibilità**: rilanciare una fusione non riproduce i file bit per bit (FAISS ordina i pareggi esatti in modo diverso col numero di thread, §51.0); i numeri riportati sono quelli dei job.

---

## 8. Limitazioni
- Solo RPLAN. ResPlan/CubiCasa5K/Maticad fuori scope (O3 → future work): nessuna label di rilevanza, out-of-domain, tassonomia non allineabile senza perdite.
- Lato vision **backbone mai aggiornato**: encoder **frozen**, si allena solo una **projection-head**; LoRA fuori.
- Proxy di rilevanza non validato con umani; **circolarità** su tutti e tre gli assi (§5.5).
- Un solo split di validazione: la selezione fra varianti di testa non si replica sul test (Caveat 5).
- Gallery = inner join dei due rami: 48 PNG senza `.mat` escluse.
- **Danni sintetici**, non piante parziali reali; la stanza tolta lascia comunque la forma del buco (§5.4). Crop e patch esistono solo per il vision: il confronto fra rami sotto danno è sulle sole stanze.
- **Late fusion**: verdetto confermato sul test (§6.9) ma specifico al checkpoint graph `gat/asymrob`; i due controlli di complementarità sono solo sul valid; i due controlli di complementarità non sono equi allo stesso modo (Caveat 10, §7) e informazione e modello non si separano del tutto.
- **Head vision**: provata con un solo tipo di danno in training (stanze tolte, prima coi muri rimasti poi coi muri tolti, `status.md §37, §48`); una head allenata su **più** danni insieme non è stata provata, e andrebbe giudicata su un danno ancora diverso.

---

## 9. Conclusioni & Future Work
- Riassunto: due rami sulla stessa gallery + protocollo per-asse de-saturato con floor e test appaiati + criterio di robustezza sotto masking + **late fusion che aiuta** (α*=0.6, +0.174 sul ramo migliore, sopra anche l'oracolo per query) **verificata con due controlli**: il guadagno è in parte effetto d'insieme fra modelli diversi (~56%) e in parte informazione complementare (~44%, §6.8). Conferma sul test con α* fisso da fare.
- **Future work** (tagliato in chiusura, `roadmap.md §3`): multi-dataset (O3, in stand-by); fusione intermedia — testa congiunta, distillazione, cross-attention (O5 b-d, in stand-by); GED topologica e metriche diagnostiche top-K; validazione umana del proxy; grafo estratto dall'immagine (confronto senza circolarità); Fase 2 generativa; head vision allenata sui danni onesti e giudicata su danni fuori distribuzione (se non fatta in chiusura); danno a stanze senza la forma del buco (richiede ridisegnare dal `.mat`).
- **Ramo graph — perché il marcatore «vicini persi» non aggiunge robustezza** `[TODO]` (`status.md §41-§42, §47`). A parità di regola di selezione (checkpoint scelto sulla robustezza) `asymlostrob` **pareggia** `asymrob` (Δ AUC −0.004, CI con lo zero), pur essendo migliore sulla pianta intera (topologia +0.057). *Ipotesi, non verificata*: la geometria delle stanze rimaste (centro e dimensioni sulla griglia 256) rivela già dove manca qualcosa, quindi il marcatore è ridondante; in più la loss asimmetrica (vista incompleta avvicinata alla sua versione completa) obbliga già la rete a inferire ciò che manca. Test possibili: (a) marcatore con la geometria oscurata — se aiuta solo lì, ridondanza confermata; (b) più repliche (rumore fra training ~0.04 AUC) e budget oltre 300 epoche (vincolante in §47). **Scartata** la variante «due viste entrambe degradate»: la gallery è sempre completa, e senza la coppia incompleta↔completa il training non allena il compito del retrieval (a viste entrambe perturbate `gat/base` crollava sotto masking, §30); inoltre in gallery il marcatore vale sempre 0.

---

## 10. Struttura, figure e tabelle

### 10.a Struttura di riferimento — da `papers/examples/pippo_paper.pdf` (CVPR 2025)

Struttura estratta dal paper di esempio (26 pagine: **12 di corpo** + 5 di
reference + 6 di appendice; 8 figure e 8 tabelle nel corpo):

| § | Sezione | Pagine |
|---|---|---|
| 1 | Introduction — chiude con «we make the following contributions:» in elenco | 1-3 |
| 2 | Related Work | 3-4 |
| 3 | Method — 3.1 base · 3.2 estensione · 3.3-3.4 le due analisi · **3.5 nuova metrica** | 4-9 |
| 4 | Experiments — 4.1 Data · 4.2 Evaluation Setup and Metrics · 4.3 Results · 4.4 Ablations | 9-12 |
| 5 | Conclusion | 12 |
| — | References · Appendix A, B, … | 16-26 |

**Le quattro lezioni da imitare:**
1. **Figura 1 = teaser del risultato**, in prima pagina, non l'architettura.
   L'architettura è **una sola** figura d'insieme (Fig. 2, «Pipeline overview»).
2. **I numeri stanno nelle tabelle, non nei grafici.** 8 tabelle contro 2 soli
   grafici quantitativi: le figure servono per il qualitativo e per **una**
   analisi concettuale (Fig. 4, entropia vs γ). ⚠️ È il contrario di quello che
   fanno oggi i notebook (~40 grafici a barre).
3. **Le didascalie sono autosufficienti**: un paragrafo che rimanda alla sezione
   («…(Sec. 3.3). We overfit our multi-view model for 10K iterations…») e
   **dichiara la conclusione**, non un'etichetta.
4. Una **nuova metrica** merita una sottosezione del Metodo (§3.5): la rilevanza
   per-asse di questo progetto va lì, non in Experiments.

**Budget nostro:** 10 pagine + 2 di reference → più stretto di Pippo. Realistico:
**5-7 figure + 4-5 tabelle**, appendice per il resto.

### 10.b Figure e tabelle da produrre

⚠️ Nessuna figura si finalizza prima che i numeri del **protocollo B** siano
letti. La mappa definitiva output → run → stato è in `roadmap.md §1`; qui resta
la descrizione di ciascun output.

**Figure (obiettivo: 6 nel corpo; 7 prodotte il 18 set — F1-F8 tranne F4)**
- [x] **F1 · teaser** — `figures/f_teaser_valid.pdf` (`src/figures/teaser_valid.py`, 18 set).
      ⚠️ **Cambiata rispetto al piano**: non più «una query con i top-5 per ciascun asse»
      (che illustrava l'impostazione per-asse, ora dichiarata circolare, §21) ma **una query
      danneggiata e i primi 5 risultati dei tre sistemi**, vision · graph · fusione, con
      l'originale incorniciato dove rientra. Illustra il risultato del report, non il
      protocollo. Ricostruita dai file già su disco, **senza job**: `ret_rows` dà la
      classifica senza la query e `self_rr` dice a che posto stava, così la classifica vera
      si ricompone esattamente; il danno è ridisegnato dalle stanze salvate nel `qvec`.
      Split **valid** di proposito (una figura qualitativa non ha bisogno del test).
      Query scelta con regola dichiarata: la prima in ordine alfabetico fra le 423 in cui la
      fusione mette l'originale al 1° posto e nessuno dei due rami lo fa (query `10340`,
      f=0.5: vision oltre il 100° · graph 3° · fusione 1°).
      L'alternativa a job (`scripts/vision/08_visualize.sh` sulla config finale) resta
      possibile se si vuole il pannello per-asse del ramo vision.
- [x] **F2 · pipeline overview** — `figures/f_pipeline.pdf` (`src/figures/pipeline.py`, 18 set):
      i due rami affiancati (immagine → encoder congelato → whitening → vettore 768-d;
      `.mat` → grafo → GNN GAT allenata → vettore 128-d), la fusione pesata (α=0.6 → 896-d)
      e la barra condivisa (stessa gallery di 67.405 piante, stesse query, stessa misura).
      I quattro numeri scritti nel diagramma sono **letti dagli artefatti** a ogni disegno,
      così non possono divergere dai file. La head non compare: non è nella config finale.
- [x] **F3 · curva di masking coi tre danni** — `figures/f_damage_kinds_valid.pdf`
      (`src/figures/damage_kinds_valid.py`, 18 set), due pannelli. **Sinistra**: la config di
      record `pespatial/gem/whiten` sotto i tre danni onesti — stanze tolte AUC 0.392 · ritaglio
      0.661 · toppe 0.521, media R 0.524: lo stesso encoder non è «robusto» in generale, dipende
      da cosa gli si toglie. **Destra**: l'artefatto dei muri su `dinov3/natural/whiten`, le stesse
      stanze tolte con i muri lasciati (AUC 0.536) o cancellati (0.304), **+0.336 a f=0.5** →
      quanto valeva il bug del render. ⚠️ Non più «una linea per encoder»: il confronto fra
      encoder è F6, qui il confronto è fra **tipi di danno**.
- [ ] **F4 · la loss non predice il retrieval**: scatter loss finale (asse x) vs
      nDCG@10 (asse y), un punto per run, con la correlazione. Trasforma
      un'ipotesi smentita in evidenza. ⚠️ I dati esistono già
      (`notebooks/vision/vision_encoders_results.csv` + xlsx) ma **dinov2 manca
      del tutto** dal csv e `ijepa_mean` non c'è: 11 serie su 15 attese.
- [x] **F5 · distribuzione delle classi di equivalenza** — `figures/f_classes_valid.pdf`
      (`src/figures/classes_valid.py`, 18 set): quante piante sono «rilevanti» per una query, in
      scala logaritmica, composizione vs topologia. Mediana **5007** contro **19** su 67.405; le
      query singleton (nessun rilevante oltre sé stessa, saltate da Recall/mAP) sono **6/2000** in
      composizione e **311/2000** in topologia. I conteggi sono `num_relevant` dei file per-query,
      cioè gli stessi usati dalle metriche; la figura verifica che due sistemi diversi diano gli
      stessi numeri e si ferma se non è così.
- [x] **F6 · robustezza alla scelta dell'ablation** — `figures/f_ablation_valid.pdf`
      (`src/figures/ablation_valid.py`, 18 set): un box per encoder con dentro tutte le sue
      configurazioni (52 frozen + 3 head), y = **R** del metro §38. Ordine delle mediane
      pecore 0.149 · siglip2 0.162 · tipsv2 0.210 · dinov2 0.259 · ijepa 0.266 · dinov3 0.276 ·
      radio 0.335 · **pespatial 0.386**; la config scelta (stella, R 0.5245) e la linea
      tratteggiata al suo valore mostrano che `radio/natural/whiten` (0.5161) le arriva
      vicinissimo: **il primo posto non è speciale**, la scelta dell'encoder conta più della
      combinazione. Valori in cache in `figures/f_ablation_valid.data.json` (`--refresh` per
      ricalcolarli). Sostituisce le classifiche top-12.
- [x] **F7 · curva del danno sul TEST** — `figures/f_damage_test.pdf`
      (`src/figures/damage_test.py`, 18 set): MRR di auto-ritrovamento vs frazione di
      stanze tolte, quattro sistemi (baseline `hist` · vision · graph · fusione α=0.6),
      bande bootstrap e AUC in legenda. Numeri identici a §6.9 (`status.md §53`), riletti
      dai per-query. f=0.0 è disegnato ma marcato «tetto dei dati»: per questo è fuori
      dall'AUC. **Candidata al posto di F3 nel corpo**: è il risultato principale.
- [x] **F8 · scala dei guadagni (valid)** — `figures/f_alpha_valid.pdf`
      (`src/figures/alpha_valid.py`, 18 set): AUC di robustezza in funzione di α con le
      tre coppie fuse sovrapposte, il guadagno di ciascuna sul proprio miglior estremo
      (+0.174 vision+graph · +0.098 vision+vision · +0.034 graph+graph) e la linea
      dell'oracolo di vision+graph. È §6.8 in una figura sola.

**Tabelle (obiettivo: 4)**
- [ ] **T1 · dataset e setup**: split, gallery, numero di query, esclusioni.
- [ ] **T2 · benchmark encoder × asse** (nDCG/Recall/mAP@10), con **CI** e il
      **floor** del ranking casuale come prima riga. Rigenerata in fase B.
- [x] **T3 · confronto fra i rami e fusione — numeri completi (valid + TEST)**:
      `hist` · vision · graph · late fusion (α=0.6) · oracolo, per asse e sotto
      danno; pointer `status.md §49-§53`. Figure **prodotte** il 18 set: **F8**
      (AUC vs α, tre coppie, valid) e **F7** (curva del danno sul test, quattro
      sistemi).
- [ ] **T4 · ablation**: pooling × trasformazione (vision) e OFAT (graph). La
      griglia a heatmap dei notebook (`§ 3`) è già compatta: va bene come figura
      d'appendice, ma nel corpo la versione tabellare costa meno spazio.
- [ ] **T5 · tipi di danno** (vision): R + AUC su stanze tolte / crop / patch per le prime
      config e la head di prima come riga di confronto, con il costo sul full accanto. Numeri
      esistenti in §6.7 (`status.md §43-§44`).

⚠️ **15 set — cosa cambia per le figure**: F1 va rigenerata sulla config finale
(`pespatial/gem/whiten`; `08_visualize` ora disegna stanze tolte, crop e patch, `status.md §39`);
F3 si fa coi tre danni onesti, e il `random` storico compare solo come confronto «con muri»
(la curva prima/dopo il bug è essa stessa una figura candidata: stesse query, stesse stanze).

### 10.c Stato dei notebook — sintesi

**Inventario completo, correzioni con priorità e 21 grafici da aggiungere:
`notebooks/CHARTS.md`.** Strategia decisa: **produrne il più possibile**, la
selezione per il report avviene qui in § 10.b. Sotto solo il riassunto.

Struttura attuale: `results.ipynb` (training loss della head) +
`results_full_retrieval` / `results_partial_per_asse` /
`results_partial_self_recovery`, ognuno con le stesse 5 sezioni (top-12 ×4
metriche · andamento ×3 · heatmap ablation · confronto per-asse · boxplot
robustezza). Sorgenti: `vision_pipline.xlsx` (5 fogli) e
`vision_encoders_results.csv`.

**Da correggere prima di portarne uno nel report** (dettaglio del perché:
`current_state.md`):
1. ⚠️ **Barre con base troncata su scala `logit`** (`barh(valore − lo_lim,
   left=lo_lim)` + `set_xscale('logit')`): la lunghezza della barra non ha
   interpretazione, e differenze da 0.002 diventano visivamente enormi — proprio
   quelle che il rilievo A4 dichiara indistinguibili dal rumore. Le barre partono
   da 0; per valori vicini a 1 con differenze minime si usa un **dot plot con
   intervalli di confidenza**.
2. ⚠️ **"Score medio" = media di nDCG + Recall + mAP su tutti gli assi**: mescola
   metriche di significato diverso e media gli assi, contro la regola
   «sempre per-asse» (composizione e topologia si muovono in senso opposto). Ed è
   il criterio di **ordinamento predefinito** di tutte le classifiche.
3. ⚠️ **Le classifiche "top 12" sono leaderboard sul test** → visualizzano il
   winner's curse del rilievo A1. In fase B diventano classifiche sul valid.
4. ⚠️ **Nessuna barra d'errore** da nessuna parte (rilievo A4).
5. ⚠️ `results.ipynb` ordina gli encoder per **train loss finale**, che questo
   progetto ha dimostrato non predittiva del retrieval: come classifica è
   fuorviante, come **F4** è un risultato.
6. **Ridondanza**: 4 grafici quasi identici per sezione × 3 notebook. Nei
   notebook si **tengono** (per strategia); nel report ognuno di quei gruppi
   diventa **una** tabella.
7. ⚠️ **0 `savefig()` in 4 notebook**: 31 grafici esistono solo inline e non
   esiste `figures/`. Bloccante per la strategia "produrne molti e scegliere
   dopo" → `notebooks/CHARTS.md § 1`.

**Da tenere così com'è:** etichette diritte in fondo alle linee invece della
legenda, colore per encoder coerente fra i notebook, spine rimosse, heatmap
5-pannelli. Il gusto grafico è buono: il problema è l'inquadramento statistico.

---

## 11. Riferimenti (da completare con citazioni esatte) `[VERIFY]`
- SSIG — Vidanapathirana et al. 2023 (arXiv 2309.04357).
- DANIEL/ROBIN — Sharma et al., ICDAR 2017.
- Space Syntax — Hillier & Hanson 1984; Lee, Ostwald & Gu 2018.
- Graph2Plan; HouseGAN++; LayoutDM. `[VERIFY]`
- Encoder: DINOv2, DINOv3, SigLIP2, RADIO, I-JEPA (checkpoint esatti in `configs/vision_models/`).

---

### Log di aggiornamento
- 2026-06-27: creazione. Stato pipeline: vision retrieval + valutazione per-asse `[DONE]` (DINOv2 valutato); partial retrieval, multi-encoder, ramo graph `[TODO]`.
- 2026-06-28: Fasi 0/1 (ablation frozen) implementate + verificate (test CPU leggeri). Scoperta split: `snapshot_train/` mescola i 3 split RPLAN → `eval.split` per le query, gallery intera (§4). Decisione capacità fissa ViT-B + I-JEPA eccezione (§6.4). Ablation config-driven: pooling/whitening-dim/risoluzione/extraction_layer via override CLI + `model.variant` (§6.5). Training (projection-head/LoRA) rimandato a Fase 3 col nodo coppie-positive. Resta da indicizzare/valutare via sbatch.
- 2026-06-29: **Fase 3 (projection head) implementata.** Pipeline disaccoppiata (RAW salvato una volta; whitening/head al volo via `prepare_index`). Head A+ self-supervised (positivo = pianta degradata + flip/rot90, InfoNCE, τ/V configurabili), training su train+valid (test held-out). Moduli `projection_head/projection_pairs/train_projection`; ordine sbatch in `COMANDI.md`; job di stage `index/pairs/trainhead/eval_head_all`. LoRA ancora fuori. Smoke test CPU verdi. ⚠️ formato `embeddings/` cambiato (RAW) → rigenerare dallo Stage A. Da lanciare via sbatch.
- 2026-06-29: **Benchmark multi-encoder eseguito** (5 encoder, full, split test, n=2000): tabella in §6.4 `[DONE]`. Pattern geo≫comp>topo per tutti; terzetto DINOv2≈DINOv3≈I-JEPA appaiato su comp/topo, I-JEPA best su geometria, SigLIP2/RADIO sotto. Chiarita §6.4 la distinzione **capacità backbone** (tutti ViT-B tranne I-JEPA ViT-H) vs **dim embedding output** (RADIO 2304 / I-JEPA 1280 / altri 768). num_queries 200→2000 (stabile ≤0.008). Partial retrieval (`partial_all_job.sh`) ancora da analizzare.
- 2026-08-24: **Fase A.1 — residuo A1 n.3 chiuso lato report**: aggiunto §7 Caveat 5 (tre letture del test + pre-registrazione del primario). Numeri da `status.md §14/§14.1`, non ricalcolati.
- 2026-08-24: **A.4 eseguita** (prima volta): sensibilità ai pesi della geometria → il claim aggregato vision>graph **NON REGGE**; aggiunto §7 Caveat 6. Numeri in `status.md §22`, output in `results/geometry_variants/`.
- 2026-08-24: **A.5 applicata** (320 run partial per-query): config vision congelata **`dinov3/natural/head`**; aggiunte §6.6 e §7 Caveat 7. Numeri in `status.md §24`.
- 2026-09-10: **allineamento di chiusura**: contributi (§1), partial fatto su entrambi i rami (§5.4), fusione e crollo del graph (§5.5), setup = protocollo B (§6.1), §6.6 marcata come protocollo vecchio, Caveat 7 corretto (head sotto-allenata, §30), limitazioni e future work (§8-§9). Piano di chiusura: `roadmap.md`.
- 2026-09-15: **allineamento ai risultati 11-15 set** (`status.md §35-§44`, numeri non ricalcolati): §1 contributi (metro a tre danni, artefatto dei muri), §5.3 (MRR per-asse ≠ reciprocal rank del self-recovery), §5.4 (bug dei muri interni, `nowalls`, crop, patch, metro §38), §5.5 (config della fusione), §6.5 (head col render vecchio), §6.6 marcata **superata**, **§6.7 nuova** (53 config, `pespatial/gem/whiten`, H1 falsificata, H2 «artefatto», costo full), Caveat 7 annotato, **Caveat 8-9**, §8-§9, T5 e note figure in §10.b.
- 2026-09-16: §9 future work: perché il marcatore «vicini persi» pareggia a selezione equa (ipotesi di ridondanza con la geometria, test possibili, variante «due viste degradate» scartata). Numeri da `status.md §47`. ⚠️ Ancora da allineare a §47: riga graph di §6.7 e confronto vision↔graph (config graph ora `gat/asymrob`).
- 2026-09-16: **head nuova chiusa** (`status.md §46` pre-registrata, **§48**): §1 risultati negativi, §6.7 punto 7 (head `nowalls` su `pespatial/gem` non si trasferisce: crop −0.142, patch −0.106, R −0.033 → non adottata), §8 limitazione sulla head. Config vision **definitiva** `pespatial/gem/whiten`.
- 2026-09-17: **late fusion e i due controlli chiusi sul valid** (`status.md §49-§51.1`, numeri non ricalcolati): **§6.8 nuova** (metodo, griglia α, verdetto «la fusione aiuta» con α*=0.6, oracolo superato, controlli graph+graph e vision+vision, scala dei tre guadagni, sintesi 56%/44%); §5.5 Fusione → `[DONE]` con `gat/asymrob`; §7 **Caveat 10**; §8 limitazione sulla fusione; §9 riassunto; §10.b T3 spuntata (valid) + figura candidata.
- 2026-09-18: **TEST letto una volta** (`status.md §52` pre-registrazione, **§53** risultati): **§6.9 nuova** (tabella dei quattro sistemi sotto danno, +0.1725 sul ramo migliore, oracolo superato, pianta intera con i delta appaiati, controlli tutti PASS, esito delle previsioni 3/4); §7 Caveat 10 aggiornato (ottimismo verificato sul test, controlli solo sul valid); §8 limitazione; §9 riassunto; §10.b T3 completa + seconda figura candidata.
- 2026-09-18b: **prime due figure prodotte** (`src/figures/`, stile a **due colonne** deciso
  dall'utente): **F7** `figures/f_damage_test.pdf` (curva del danno sul test, quattro sistemi) e
  **F8** `figures/f_alpha_valid.pdf` (AUC vs α, le tre coppie fuse). Numeri riletti dai per-query,
  non ricalcolati; `alpha_valid` rifiuta di disegnare se una media non coincide con quella del json
  di `fusion_select`. Ogni figura salva PDF + PNG + `*.sources.txt` (file letti, comando, numeri).
  Restano F1-F6 (§10.b).
- 2026-09-18c: **teaser F1 prodotto senza job** — `figures/f_teaser_valid.pdf`: una query danneggiata
  (metà stanze, valid) e i primi 5 risultati di vision, graph e fusione. Contenuto **cambiato** rispetto al
  piano di §10.b (non più i top-5 per asse: quell'inquadratura illustrava l'impostazione per-asse, che è
  dichiarata circolare). Classifiche ricostruite da `ret_rows` + `self_rr`, danno ridisegnato dalle stanze
  del `qvec`: nessun ricalcolo, nessuna GPU.
- 2026-09-18d: **F2, F3, F5, F6 prodotte** (`src/figures/`, tutte CPU, nessun job): pipeline
  overview coi numeri letti dagli artefatti · i tre danni + l'artefatto dei muri (due pannelli) ·
  distribuzione delle classi di equivalenza (mediana 5007 vs 19, singleton 6 vs 311 su 2000) ·
  boxplot delle 52 config frozen per encoder, dove la config scelta (R 0.5245) e la migliore di
  `radio` (0.5161) quasi coincidono. ⚠️ F3 cambia inquadratura rispetto al piano (danni, non
  encoder: il confronto fra encoder è F6). Resta **solo F4** (scatter loss↔retrieval), che dipende
  da un csv incompleto (11 serie su 15).
