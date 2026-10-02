# PAPER.md — tutto ciò che serve per scrivere il paper

> **Documento di riferimento per la stesura** (paper stile CVPR, in inglese). Contiene la tesi, il
> metodo, tutti i numeri finali, le scelte con il loro perché, i caveat, la storia del progetto e il
> piano di figure e tabelle. Si può scrivere il paper **leggendo solo questo file**; i rimandi
> (`status.md §N`, `file.py:riga`) servono a verificare o approfondire sul progetto in locale.
>
> **Allineato allo stato del progetto al 2 ott 2026** (ultimo lavoro: verifica SAGE vs GAT a parità di
> regola di selezione, `status.md §58`; prima: repliche multi-seed, §57). Dove la storia e lo stato attuale divergono, **vale lo stato attuale**; le versioni
> superate compaiono solo nel capitolo §10 (storia), marcate come tali.

## Come usare questo file

- **Fonti** (path dalla root del repo):
  - `.claude/shared/status.md` — diario numerato (§1-§58) di ogni misura, decisione e ipotesi. È la
    fonte primaria: ogni numero qui ha accanto la sezione da cui viene.
  - `.claude/shared/{retrieval,dataset,architecture,experiments,roadmap}.md` — metriche, dati,
    contratti, protocollo, piano di chiusura.
  - `COMPETITORS.md` — related work e baseline (letti sui PDF in `papers/comp/`).
  - `results/csv/*.csv` + `results/csv/_sources.txt` — numeri finali riaggregati dai per-query
    (`status.md §55-§56`); `results/fusion/seeds/summary.json` — repliche.
  - `figures/{english,italian}/` — figure finali, ognuna con `*.sources.txt` (file letti, comando,
    numeri disegnati).
  - `spiegazione_metriche_e_risultati_grafi.md` — spiegazione discorsiva delle metriche.
- **Marcatori**: `[DONE]` misurato e verificato · `[TODO]` da fare · `[VERIFY]` da ricontrollare
  (soprattutto citazioni) · `[DECISION]` scelta da motivare nel testo · `[CLAIM]` frase che il paper
  può sostenere così com'è · `[NO-CLAIM]` frase che il paper **non** può fare.
- **Terminologia** (italiano qui → inglese nel paper): stanze tolte → *room removal*; ritaglio →
  *crop*; toppe → *patch masking*; auto-ritrovamento → *self-recovery*; pianta intera → *full plan*;
  spazio utile → *normalized gain over the random floor*; ramo → *branch*.
- **Quale metro**: ogni numero dice su quale split (valid/test), quale protocollo (A = storico, B =
  finale, §5.1) e quale misura. Un numero senza queste tre cose non entra nel paper.

---

## 0. Checklist pre-stesura

Già fatto (`status.md §55`, 28 set): **tutti i numeri finali ricalcolati dai per-query** e coincidenti
alla quarta cifra con quelli riportati qui; export in `results/csv/`. Restano:

- [ ] Citazioni esatte (§3, §12): tutte `[VERIFY]` tranne quelle lette in `COMPETITORS.md`.
- [ ] Decidere come presentare la **circolarità** (§7.1): è l'unica domanda aperta del progetto
      (`roadmap.md §6` n.1). La proposta è in §7.1.
- [ ] Decidere se includere la **storia del bug dei muri** come contributo (§7.3; vedi nota in §10).
- [ ] Decidere F4 (§11.2) e se lanciare la baseline LayoutGKN (§3.2, §9).
- [ ] Impaginare T1-T5 (§11.3): i numeri esistono tutti, manca solo la tabella.
- [ ] Se si cita un numero non presente qui, leggerlo da `status.md` o dai CSV, mai a memoria.

---

## 1. Tesi, contributi, abstract

### 1.1 La tesi in una frase `[CLAIM]`

Per ritrovare planimetrie a partire da una pianta **incompleta**, un encoder di immagini congelato e
una GNN allenata sul grafo delle stanze **sbagliano su query diverse**; fonderli con una semplice
somma pesata delle similarità ritrova la pianta originale molto meglio di ciascuno dei due
(test: AUC 0.640 ± 0.003 contro 0.470 del graph e 0.392 del vision, 4 repliche), e il guadagno è in
parte informazione complementare e in parte effetto d'insieme fra modelli diversi.

### 1.2 Contributi (da limare)

1. **Due rami di retrieval sulla stessa gallery** di RPLAN — vision (encoder congelato → pooling →
   PCA whitening → FAISS) e graph (GNN GAT allenata con InfoNCE su coppie asimmetriche intero ↔
   parziale) — con **late fusion** per concatenazione pesata. Sul test la fusione batte il ramo
   migliore di **+0.1725 [+0.1625, +0.1826]** in AUC di self-recovery sotto rimozione di stanze, sta
   sopra l'oracolo per query (0.5693) e regge su 4 repliche (§6.9-§6.10).
2. **Un controllo a tre gradini della complementarità**: guadagno della fusione di due training dello
   stesso graph (+0.034), di due encoder vision diversi sulla stessa immagine (+0.098), di vision +
   graph (+0.174). La fusione vision + graph non è solo un effetto d'insieme, ma nemmeno solo
   informazione nuova: circa 56% / 44% (§6.8).
3. **Un protocollo di valutazione per il retrieval di piante incomplete**: robustezza come AUC del
   self-recovery (media di 1/rango sui livelli di danno, schema di m@B di *Sketch Less for More*) su
   **tre danni che tolgono davvero informazione** (stanze tolte coi muri, crop, patch), con
   confronti appaiati, CI bootstrap e pre-registrazione delle ipotesi e del test (§5.4, §5.8).
4. **Ground truth di rilevanza scomposta per asse** (composizione, topologia, geometria) dai metadati
   RPLAN, misurata **sull'intera gallery** e normalizzata sul **floor del ranking casuale**, con la
   diagnosi della saturazione delle metriche ingenue (§5.2-§5.3, §6.1-§6.2).
5. **Risultati negativi e analisi di validità riportati come tali**:
   - una projection head allenata su un danno **non si trasferisce** a danni diversi (falsificato due
     volte, su due encoder e due render, §6.5);
   - un **artefatto di rendering** del masking valeva quasi tutta la robustezza misurata del vision
     (*shortcut learning* su un difetto condiviso da training e misura, §7.3);
   - il graph addestrato su coppie simmetriche **crolla** sotto rimozione di stanze (§6.6);
   - la **circolarità** della ground truth non risparmia nessun asse né ramo (§7.1);
   - la selezione fra varianti di testa fatta sul valid **non si replica** sul test (§7.6).

### 1.3 Abstract — punti da coprire

Problema (retrieval da piante incomplete, workflow reale dell'architetto) → mancanza in letteratura
(nessun lavoro fa retrieval di piante con query incomplete né fonde immagine e grafo, §3.4) → metodo
(due rami + late fusion) → protocollo (tre danni, AUC di self-recovery, per-asse con floor) → numero
principale (0.640 vs 0.470 / 0.392 sul test, 4 repliche) → complementarità (scala dei tre guadagni) →
lezioni (la head non si trasferisce, artefatto di rendering, circolarità).

---

## 2. Introduzione & motivazione

- **Il task.** Un architetto parte quasi sempre da un layout **parziale** (qualche stanza, un contorno,
  dei vincoli) e cerca precedenti simili. Il retrieval deve funzionare quando la query è incompleta,
  non solo quando è una pianta intera. Graph2Plan motiva lo stesso scenario dal lato generativo
  (contorno + vincoli → retrieval di layout graph, §3). `[DECISION]`
- **Similarità visiva ≠ rilevanza architettonica.** Un encoder su raster cattura forma e colore; per
  un architetto contano funzione e **topologia** (quali stanze, come connesse). Il giudizio umano di
  similarità fra piante correla con la Graph Edit Distance sul grafo delle stanze più che con la
  sovrapposizione di forma (SSIG; user study con architetti in LayoutGKN, §3). Da qui due rami: uno
  sui pixel, uno sul grafo.
- **Perché RPLAN.** Fornisce per ogni pianta sia un raster sia i metadati strutturali `.mat` (tipi
  di stanza, adiacenze tipizzate, box): servono a costruire il grafo **e** una ground truth di
  rilevanza senza annotazione umana. Scope del report: **solo RPLAN** (§8).
- **Cosa misuriamo come «migliore».** `[DECISION]` (`status.md §23`, poi §38) «Migliore» = **più
  robusto**: chi ritrova meglio la pianta giusta quando la query è danneggiata. La prestazione sulla
  pianta intera si riporta sempre accanto, come costo della scelta. Motivo: i due criteri danno
  classifiche diverse (§6.4: il migliore a pianta intera non entra fra i primi 14 su 53 per
  robustezza), e la domanda pratica è «quale sistema uso quando la pianta è incompleta?».

---

## 3. Related work

Dettaglio, numeri degli autori e livello di lettura di ciascun lavoro in **`COMPETITORS.md`**. I
numeri degli altri lavori vengono dai loro protocolli e **non** sono confrontabili coi nostri.

### 3.1 Retrieval e similarità di piante
- **LayoutGKN** (van Engelenburg, van Gemert, Khademi, BMVC 2025) `[LETTO]`: similarità fra piante RPLAN
  come graph kernel su embedding per stanza; nodi quasi identici ai nostri (tipo + cx, cy, w, h, area),
  archi porta/adiacenza; P@5/P@10 sui top-50 del modello, triplet da MIoU + sGED. Solo piante intere.
  È **l'unica baseline di componente eseguibile** nel nostro protocollo (al posto del ramo graph e
  dentro la fusione) — non ancora lanciata (§9).
- **LayoutGMN** (Patil et al., CVPR 2021) `[PARZIALE]`: graph matching network, IoU come supervisione;
  baseline di riferimento del settore, ma embedding dipendenti dalla coppia → non scala a 67K come
  retrieval.
- **URE-Net** (Zhang et al., arXiv 2025) `[SEZIONE]`: codifica a «unit region», triplet accuracy su
  RPLAN nel setting di LayoutGMN. Il più vicino al nostro ramo vision, ma niente ranking sulla gallery
  né query parziali.
- **SSIG** (van Engelenburg, Khademi, van Gemert, ICCVW 2023, arXiv 2309.04357) `[ABSTRACT]`: metrica
  IoU + GED senza learning, segnala i quasi-duplicati di RPLAN. ⚠️ La versione precedente di questo
  file lo attribuiva a «Vidanapathirana et al.»: **sbagliato**, autori verificati sul PDF
  (`COMPETITORS.md §3.1`). Serve a motivare una ground truth strutturale.
- **DANIEL/ROBIN** (Sharma et al., ICDAR 2017): primo CBIR deep per piante. `[VERIFY]`

### 3.2 Query incomplete
- **Sketch Less for More** (Bhunia et al., CVPR 2020) `[SEZIONE]`: retrieval della foto esatta da uno
  schizzo incompleto; metrica **m@B** = area sotto la curva di 1/rango al variare della completezza.
  **È il precedente della nostra AUC di self-recovery**: stessa struttura (una sola risposta giusta,
  1/rango mediato sui livelli). Differenze: lì la query cresce ed è in un'altra modalità, qui perde
  stanze/pixel ed è nella stessa modalità della gallery.
- **PlanCraft / SketchPlan** (Zeng et al., arXiv 2026) `[SEZIONE]`: piante parziali su RPLAN a livelli
  25/50/75/100% per la *generazione* progressiva; toglie muri/porte/finestre in ordine di disegno (un
  quarto tipo di danno, plausibile, non usato da noi).
- **Graph2Plan** (Hu et al., SIGGRAPH 2020) `[ABSTRACT]`: retrieval di layout graph RPLAN da contorno +
  vincoli, con regole; origine del formato `.mat` che usiamo.

### 3.3 Fusione multimodale e ramo vision
- **CrossOver** (Sarkar et al., CVPR 2025) `[SEZIONE]`: allinea RGB, point cloud, CAD, floorplan e testo;
  fusione **appresa**, gestisce modalità mancanti. La nostra è una late fusion di similarità con un
  solo peso scelto sul valid.
- **MMFE** (Anadón et al., ECCVW 2026) `[SEZIONE]`: DINOv3 frozen + testa contrastiva su piante
  multimodali; osserva che le feature DINO sono dominate dall'aspetto, non dal layout — coerente col
  nostro divario vision↔graph sulla topologia.
- **Encoder del benchmark**: DINOv2, DINOv3, SigLIP2, C-RADIOv2, I-JEPA, TIPSv2, PE-Core, PE-Spatial
  (checkpoint in §5.5). GNN: GCN, GraphSAGE, GATv2. `[VERIFY]` citazioni.
- **Teoria architettonica**: Space Syntax / Justified Plan Graph (Hillier & Hanson 1984; Lee, Ostwald
  & Gu 2018) come fondamento dell'asse topologico. `[VERIFY]`

### 3.4 Il vuoto in letteratura `[CLAIM]` (ricerca del 1 ott 2026, non esaustiva)
- Nessun lavoro trovato fa **retrieval di piante con query incomplete** (le piante parziali su RPLAN
  compaiono solo nella generazione).
- Nessun lavoro trovato fa **late fusion fra un encoder di immagini e una GNN** per il retrieval di
  piante.
- Nessuno usa una **rilevanza scomposta per asse**; i lavori letti misurano triplet accuracy o P@k sui
  top-50 del modello, **non** il ranking sull'intera gallery né il confronto con un floor casuale.

Scartati e perché: `COMPETITORS.md §5`.

---

## 4. Dataset (RPLAN)

Fonte: `.claude/shared/dataset.md`, `src/data/rplan_metadata.py`, `status.md §12, §21`.

- **Raster**: 67.453 PNG 1167×875 in `snapshot_train/`, piante residenziali con colori per tipo di
  stanza, perimetro grigio 79, muri interni grigio 128.
- **Metadati `.mat`**: 3 file aggregati, split ufficiali **disgiunti** 56.511 / 12.108 / 12.110 (train /
  valid / test, unione 80.729). Campi usati: `rType` (13 tipi), `rEdge` (archi `i, j, relazione 0..9`),
  `gtBoxNew` (box per stanza), `gtBox` (ultima riga = footprint), `boundary`.
- **13 tipi di stanza**: LivingRoom, MasterRoom, Kitchen, Bathroom, DiningRoom, ChildRoom, StudyRoom,
  SecondRoom, GuestRoom, Balcony, Entrance, Storage, Wall-in.
- ⚠️ `[DECISION]` **`snapshot_train/` non è il solo train**: i 67.453 PNG mescolano i tre split (47.126
  train + 10.152 valid + 10.127 test + **48 senza `.mat`**). Per questo:
  - **gallery = intero corpus** (non si splitta il corpus di ricerca); lo split restringe le **query**;
  - **gallery condivisa** fra i due rami = inner join PNG ∩ `.mat` = **67.405 piante**
    (`results/shared_gallery.json`, sha1 `0c24cfc05e18`, `status.md §27, §29`); si escludono solo le
    48 PNG che il graph non può rappresentare;
  - pool di **training** = split `train`; **selezione** sul `valid`; **test** solo per il numero finale.
- **Grafi**: 67.405, piccoli (4-8 nodi, media ≈ 6,8; ~14-28 archi non orientati). 397 piante hanno un
  self-loop in `rEdge` (0,5% dei grafi perdono 1-3 archi simmetrizzando); nessuna coppia di archi
  ripetuta (`status.md §21.3`).
- **Duplicati**: il **3%** dei PNG ha un gemello pixel-identico (970 gruppi, 2.020 file) più
  quasi-duplicati. Conseguenza: a query intatta il self-recovery si ferma a MRR ≈ 0.970 per **tutti**
  gli encoder — è un tetto dei **dati**, non dei modelli (per questo f=0 è escluso dall'AUC, §5.4).
- **Il colore codifica il tipo, ma compresso** (`status.md §21.2`, 300 piante): il colore è funzione
  deterministica di `rType` (purezza 100% su 11 tipi su 12) ma **13 tipi → 6 colori** (un solo giallo
  per MasterRoom, SecondRoom, StudyRoom, ChildRoom, GuestRoom). Il vision quindi *legge* le etichette,
  a grana grossa (rilevante per la circolarità, §7.1).
- **Footprint** (`status.md §21.1`): `gtBox[-1]` = unione esatta dei `gtBoxNew` (con assi scambiati e
  massimi inclusivi) nel **100%** di 24.218 piante → il footprint è ricostruibile dai box delle stanze.

---

## 5. Metodo

### 5.1 Problema e protocollo `[DONE]` `[DECISION]`

**Task.** Data una query (pianta intera o danneggiata), ordinare l'intera gallery di 67.405 piante.

**Protocollo B** — quello di **tutti** i numeri finali (`experiments.md § Protocollo B`):
- gallery condivisa (sopra), identica nei due rami **riga per riga** (stesso `gallery_sha1`) → stesse
  query campionate, confronti vision↔graph appaiati senza eccezioni;
- **2000 query** (seed 42) dal `valid` per scegliere, dal `test` una volta per il numero finale;
- self escluso dalle metriche per-asse (anche sotto danno); incluso nel self-recovery, dove è il
  bersaglio;
- whitening stimato sul **solo train** (`whitening.fit_split: train`), applicato a tutta la gallery.
  Verifica (`status.md §31.1`): train-only vs stima su tutta la gallery, Δ AUC +0.0003…+0.0020 → il
  guadagno del whitening **non era leakage**.

**Vincoli che ogni numero rispetta** (`CLAUDE.md`): il test non sceglie nulla; statistiche di
normalizzazione dal solo train; confronti appaiati (stesse query, gallery, esclusioni).

### 5.2 Ground truth di rilevanza per asse `[DONE]` `[DECISION]`

Fonte: `src/evaluation/relevance.py` (`GalleryAxes`).

- **Perché un proxy**: per RPLAN **non esistono label umane** di rilevanza. Il proxy si difende
  perché è **trasparente e scomponibile**, non perché sia «giusto».
- **Tre assi**, ognuno in [0,1]:
  - **Composizione** — Weighted Jaccard sull'istogramma dei 13 tipi (gestisce i conteggi: 2 bagni ≠ 1
    bagno; penalizza sia il mancante sia l'eccesso).
  - **Topologia** — Weighted Jaccard sull'adiacenza **per coppia di tipi** (91 coppie), invariante alla
    permutazione delle stanze.
  - **Geometria** — media di similarità di area del footprint, aspect ratio del footprint e
    distribuzione dell'area per tipo, pesi `(1,1,1)` fissati a priori (`relevance.py:137-146`).
- `[DECISION]` **Niente score fuso, niente pesi fra assi, niente soglia**: i pesi sarebbero arbitrari;
  la scomposizione è più onesta e più informativa (gli assi si muovono spesso in senso opposto).
- **Rilevanti esatti** per gli assi discreti = **classe di equivalenza** (stesso istogramma / stessa
  adiacenza): la dimensione del set rilevante emerge dai dati. Sul valid (2000 query, self escluso,
  F5): composizione mediana **5007** piante rilevanti (max 15.251, 6 query senza altri rilevanti);
  topologia mediana **19** (max 2.633, **311** query singleton).
- ⚠️ Gli assi **non sono indipendenti** (Pearson fra i gain: comp↔topo 0.69, comp↔geom 0.30,
  topo↔geom 0.27, `status.md §12`): «vince su 2 assi su 3» non è evidenza indipendente.

### 5.3 Metriche e floor `[DONE]` `[DECISION]`

Fonte: `src/evaluation/metrics.py`, `random_floor.py`, `status.md §10, §20, §55`.

- **Ambito = intera gallery**: è ciò che permette alla metrica di fallire (vedi §6.1 per cosa succede
  altrimenti).
- **nDCG@K** (K ∈ {1, 5, 10, 100}, si riporta K=10): gain graduato = similarità d'asse, IDCG sui
  migliori K **dell'intera gallery**. Copre tutti e tre gli assi.
- **Recall@K, mAP@K** solo sugli assi discreti, normalizzate per `min(K, |R|)`; query singleton escluse
  e contate. ⚠️ Con classi enormi (composizione, |R| ≫ K) `Recall@K` **è di fatto Precision@K**
  (cala con K): nel paper chiamarla così su quell'asse.
- **MRR per-asse rimossa**: satura sugli assi densi. Da non confondere con il **reciprocal rank del
  self-recovery** (§5.4), che è un'altra domanda (una sola risposta giusta) ed è l'endpoint della
  robustezza.
- **Floor del ranking casuale** (`[DECISION]` denominatore, `status.md §20`): nDCG@10 sul **test**,
  gallery condivisa = **0.7403 / 0.4663 / 0.8700** (C/T/G, §55); sul valid 0.7406 / 0.4655 / 0.8699.
  Recall/mAP dello stesso ranking casuale ≈ 0.085 / 0.035.
  - **Spazio utile** = `(score − floor)/(1 − floor)`. Si riporta **sempre** accanto all'nDCG: con un
    floor di 0.87 la geometria ha solo 0.13 di spazio, quindi un nDCG «alto» può valere poco.
  - Il null «costante» (stesse 10 piante a tutte le query) non va al denominatore: dipende troppo dal
    sorteggio (±8 punti percentuali); serve solo a dire se un asse discrimina.
- **Lettura chiave** `[CLAIM]`: l'nDCG per-asse è in gran parte saturo **per costruzione** (gain
  continui su piante simili fra loro); la **topologia** è l'asse che discrimina (floor più basso).

### 5.4 Danni alla query e misura di robustezza `[DONE]` `[DECISION]`

Fonte: `src/vision/data/vision_damage.py`, `src/graph/graph_partial_query.py`,
`src/evaluation/robustness_auc.py`, `status.md §23, §34-§38`.

**Tre danni sul vision**, livelli f ∈ {0.25, 0.5, 0.75}:
1. **Stanze tolte** (`nowalls-random`): si sceglie a caso una frazione f delle **stanze**, si sbianca
   l'interno e si cancellano anche i muri che le bordavano (sia il perimetro grigio 79 sia i muri
   interni grigio 128, regola «grigio neutro non-sfondo»). Area media effettivamente tolta
   0.29 / 0.52 / 0.71. Nessun muro nuovo: l'incompletezza resta visibile.
   - Limite dichiarato, non aggirabile su immagine: resta la **forma del buco** (le stanze vicine
     finiscono dove finiva quella tolta). Toglierla vorrebbe dire ridisegnare un'altra pianta.
2. **Crop**: un rettangolo bianco di lati e posizione casuali, accettato se copre la frazione f dei
   **pixel della pianta** ±5%.
3. **Patch**: patch ViT intere sbiancate dopo il resize, stessa frazione di area. ⚠️ La griglia
   dipende dall'encoder (P14 vs P16, TIPSv2 a 448): il danno non è identico fra encoder.

**Sul graph** (non ha pixel) c'è solo il danno a stanze: si tolgono **le stesse stanze** del vision
(la funzione di selezione è condivisa, rng `seed + qi` per query; il pairing regge solo con la gallery
condivisa), con nodi e archi e rimappatura. Le stanze superstiti restano nella posizione assoluta.

**Asimmetria del danno da dichiarare**: a parità di stanze tolte il vision conserva la sagoma del
buco, il graph perde nodo e archi senza traccia.

**Self-recovery**: reciprocal rank della pianta originale nella gallery (self incluso; 0 oltre il 100°
posto). **AUC** di una configurazione su un danno = media del reciprocal rank su f ∈ {0.25, 0.5, 0.75}
(pesi uguali), per query, poi media sulle query. f=0 è **escluso** (tetto dei dati, §4).

**Metro di robustezza R** del vision (`status.md §38`, fissato **prima** della griglia completa):
`R(q)` = media delle tre AUC (stanze tolte, crop, patch) per query; `R` = media sulle query.
Per il graph e per la fusione il metro è l'AUC sulle sole stanze tolte (l'unico danno che entrambi i
rami possono ricevere).
- **Assoluto, non relativo**: vince chi arriva più in alto sotto danno, non chi cala di meno.
- **Spareggio** (CI del delta che contiene 0): nDCG@10 full di topologia → poi la config più
  economica.
- **Costo sulla pianta intera** sempre riportato accanto.
- Perché tre danni: con un solo danno si misurano anche i suoi artefatti (§7.3). Una config non può
  vincere sfruttando un danno solo.

**Per-asse sotto danno**: metriche contro la pianta **completa**, self escluso come nel full
(descrittive).

### 5.5 Ramo vision `[DONE]`

Fonte: `src/vision/models/retrieval_model.py`, `src/vision/models/vision_encoders/`,
`configs/vision_models/`.

**Pipeline**: PNG → encoder **congelato** → pooling (`natural` | `mean` | `gem`) → PCA whitening
(centering, decorrelazione, varianza unitaria, L2) → FAISS `IndexFlatIP` (coseno). Embedding RAW
salvati una volta; whitening (e head) applicati al volo.

**Perché il whitening** `[CLAIM]`: sulle colormap RPLAN (fuori distribuzione per encoder pre-allenati
su immagini naturali) gli embedding collassano in una piccola regione (similarità tutte in
[0.97, 0.99]). Il whitening rende il ranking discriminativo: **whitening > raw in 26/26** coppie
encoder × pooling sul metro finale (§6.4): senza whitening sotto danno gli encoder collassano.

**8 encoder** — scelti per coprire famiglie di pre-training diverse, non come lista di SOTA:

| Encoder | Checkpoint | Famiglia | Pooling `natural` | Dim |
|---|---|---|---|---|
| DINOv2 | `facebook/dinov2-base` | self-supervised | [CLS] | 768 |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` (gated) | self-supervised | [CLS] | 768 |
| SigLIP2 | `google/siglip2-base-patch16-224` | vision-language | attention pool | 768 |
| RADIO | `nvidia/C-RADIOv2-B` | agglomerativo (DINOv2+CLIP+SAM) | summary token | 2304 |
| I-JEPA | `facebook/ijepa_vith14_1k` (**ViT-H**) | predittivo | mean dei patch | 1280 |
| TIPSv2 | `google/tipsv2-b14` (nativa 448, anche 224) | immagine-testo spaziale | [CLS] | 768 |
| PE-Core | timm `vit_pe_core_base_patch16_224.fb` | contrastivo CLIP-like | attention pool | 1024 (768 mean/gem) |
| **PE-Spatial** | timm `vit_pe_spatial_base_patch16_512.fb`, usato a **224** | allineamento denso | [CLS] | 768 |

- `[DECISION]` **Capacità appaiata ViT-B** per tutti tranne I-JEPA (nessun ViT-B ufficiale → ViT-H):
  divario da dichiarare. La **larghezza dell'embedding** è un'altra cosa (RADIO 2304 non è più capacità
  del backbone) e non dà vantaggio (nel primo benchmark RADIO era il più debole su composizione e
  topologia). Varianti di taglia non confrontate.
- **PE-Core e PE-Spatial sono una coppia controllata**: stesso ViT-B/16, stessa normalizzazione,
  stessa risoluzione; cambia solo l'obiettivo di training (contrastivo vs denso).
- **Input 224** per tutti salvo TIPSv2 (224 e 448). PE-Spatial a 224: timm ricampiona il
  positional embedding; il CLS a 224 correla 0.956 con quello a 512.
- **Griglia**: 8 encoder × pooling × risoluzione = 26 estrazioni RAW × {raw, whiten} = **52
  configurazioni frozen**.
- ⚠️ **Il pooling `gem` del progetto non è un GeM standard** (`base.py:24`, ancora così nel codice):
  `tokens.clamp(min=1e-6).pow(p).mean().pow(1/p)` su token post-LayerNorm azzera le componenti
  negative (50,7% misurato, `status.md §12`). È una media generalizzata della **sola parte positiva**.
  La config finale si chiama `pespatial/gem/whiten`: nel paper va descritta come «GeM sulla parte
  positiva (p=3)», o rinominata, non come GeM.

**Config vision definitiva: `pespatial/gem/whiten`**, frozen, 768-d (`status.md §44, §48`).

**Projection head** (provata, **non adottata**): MLP `Linear(→1024)→GELU→Linear(→256)` + L2 sopra
l'encoder congelato, InfoNCE (τ 0.07, batch 4096, lr 1e-3, 3 viste positive per pianta), positivo =
la stessa pianta danneggiata + flip/rot90 (self-supervised, **nessuna label** di valutazione nella
loss); gradiente solo sulle righe `train`; epoca scelta con una sonda di AUC su 1000 query valid
**disgiunte** da quelle di valutazione; budget max 5000 epoche, patience 100. Esito in §6.5.

### 5.6 Ramo graph `[DONE]`

Fonte: `src/graph/`, `configs/graph_models/gat.yaml`, `scripts/graph/03_train_gnn.sh`.

**Grafo** (PyG, dai `.mat`, non dal raster):
- nodi `x [N, 19]` = one-hot del tipo (13) + `cx, cy, w, h, area, aspect` da `gtBoxNew` su griglia 256
  (aspect clippato al p99, z-score);
- archi da `rEdge`, non orientati; `edge_attr` = one-hot della relazione spaziale (10 tipi), usato
  solo da GAT.

**Encoder finale `gat/asymrob`**:
- **GATv2**, 2 layer (diametro dei grafi 2-3: più hop = over-smoothing), hidden 128, 4 teste mediate,
  pooling `add`, **`raw_skip`** (add-pool delle feature grezze concatenato prima della proiezione) →
  embedding **128-d** L2.
- **InfoNCE** τ 0.3, batch 256, lr 1e-3, weight decay 1e-5.
- **Coppie asimmetriche** (`pair_mode=asym_partial`): vista A = grafo intero (solo flip/rotazioni),
  vista B = stesso grafo con una frazione f ~ U[0.25, 0.75] di stanze tolte davvero. Il training
  insegna esattamente il compito: avvicinare una pianta incompleta alla sua versione completa.
- **Selezione del checkpoint sulla robustezza**: ogni epoca, AUC di self-recovery su 2000 query valid
  **disgiunte** dalle 2000 di valutazione, gallery intera; tetto **300 epoche**, patience 0.
  ⚠️ Il tetto è vincolante (epoca scelta 294 per la storica, 300 / 293 / 300 nelle repliche): la
  robustezza del graph è un **limite inferiore**.
- ⚠️ Iperparametri: la config finale usa il YAML `gat` di riferimento (τ 0.3, node-drop 0.2, flip/rot
  0.5), cioè valori che l'ablation OFAT aveva trovato peggiori **sulla pianta intera con coppie
  simmetriche** (§6.3); non sono stati ri-ottimizzati per le coppie asimmetriche (in chiusura niente
  training aggiuntivi, `roadmap.md §3`). Da dichiarare.
- `[DECISION]` Perché non la val-loss: per il graph la val-loss InfoNCE **inverte** la classifica del
  retrieval (§6.3, §10).
- Denominatori onesti: la baseline training-free **`hist`** (istogramma L2 dei tipi; è un oracolo per
  costruzione sulla composizione) e la **GNN a pesi casuali** (§6.3).
- ⚠️ **GAT non è l'encoder più robusto della ricetta** (verifica del 2 ott, §6.6 punto 7): l'ordine
  GAT ≫ SAGE/GCN era stato stabilito con la regola di selezione precedente; rifatto con la regola nuova,
  **SAGE batte GAT** (+0.026 di AUC sul valid, 4 repliche). GAT resta nella fusione per scelta
  dichiarata prima di leggere i numeri (analisi aggiuntiva, non sostituzione). Da dichiarare.

### 5.7 Late fusion `[DONE]`

Fonte: `src/evaluation/{query_vectors,late_fusion,fusion_select}.py`, `status.md §49`.

- **Concatenazione pesata** dei vettori L2 dei due rami: `[√α·v ; √(1−α)·g]` (vision dopo il whitening
  fittato su train) ⇒ prodotto scalare = `α·sim_V + (1−α)·sim_G`. Un solo indice FAISS (896-d).
- Sotto danno i vettori delle query danneggiate sono ricalcolati **in entrambi i rami con le stesse
  stanze tolte** per query.
- **Scelta di α** sul valid, griglia {0, 0.1, …, 1}, criterio = AUC di self-recovery sulle stanze tolte
  (non la topologia full, che sarebbe circolare a favore del graph). Insieme dei «pari» = α con CI del
  delta col migliore che contiene 0; si prende il **centrale**. Risultato: **α* = 0.6** (unico pari).
- Regola del verdetto (pre-registrata): «la fusione aiuta» se α* ∉ {0,1} e il guadagno sul **ramo con
  AUC media più alta** ha CI 95% > 0. L'oracolo per query (miglior ramo query per query) è riportato
  ma **non è un tetto**: sommando le similarità la fusione può superare entrambi i rami sulla stessa
  query.
- Controlli di validità (tutti passati): α=1 e α=0 riproducono **esattamente** i due rami (0/2000
  self_rr diversi); stesse stanze nei due rami; sha1 gallery; MRR a f=0 ≥ 0.965.
- Nessuna fusione per rango (RRF) nel metodo finale: provata solo in esplorazione (§10).

### 5.8 Statistica e pre-registrazione `[DONE]` `[DECISION]`

- **Confronti appaiati** per query, join sui nomi; CI 95% **bootstrap** (B=10.000, seed 0); Wilcoxon
  + Holm sui 3 assi per le metriche per-asse (`src/evaluation/significance.py`). Lo strumento rifiuta
  da solo confronti non appaiati (gallery, split, seed, `exclude_self` diversi).
- **Pre-registrazione**: ogni decisione importante ha ipotesi, previsione e regola scritte **prima**
  dei job (`status.md §23, §34, §38, §40-§41, §45-§46, §49-§52, §57`); gli esiti sono riportati anche
  quando smentiscono la previsione (tabella in §7.8).
- **Livelli di rumore misurati** (servono a leggere ogni delta):
  - fra **training** identici del graph: ~0.03-0.04 di AUC su una singola replica (`status.md §42,
    §47`); nelle 4 repliche finali il graph sul test sta in 0.466-0.474 (sd 0.0035, §6.10);
  - fra **repliche** della pipeline (training graph + stanze tolte): sd ≈ 0.003 sulla fusione (§6.10);
  - di **valutazione** (stessa config, stesso checkpoint, due esecuzioni): Δ AUC +0.00039, dovuto a
    pareggi di ranking (`status.md §55`);
  - non determinismo GPU/FAISS: 6/2000 query che scambiano rango 1↔2 fra piante duplicate
    (`status.md §37`); quasi-pareggi entro 1 ulp float32 (§57.2).
- **Precisione dei per-query**: i conteggi «A > B» vanno fatti sui float non arrotondati (con 6 cifre
  i pareggi diventavano vittorie: 1394 invece di 1218, `status.md §56`).

---

## 6. Esperimenti & risultati

Tutti su protocollo B salvo dove detto. «C/T/G» = composizione / topologia / geometria (nDCG@10).

### 6.1 Diagnosi della saturazione — perché il protocollo è fatto così `[DONE]`

Da raccontare come **motivazione** del protocollo (giugno 2026, `retrieval.md`):
- Prima formulazione: score fuso (pesi 0.5/0.3/0.2) + soglia 0.5, valutato **solo sui top-k**:
  Recall@5 = 1.0, mAP 0.99, nDCG 0.97 → **saturo**.
- Prova: su 2000 coppie **casuali** il **96%** supera la soglia → anche un retriever casuale avrebbe
  Recall@5 ≈ 1.
- Cause: ambito circolare (nessun insieme di rilevanti definito sulla gallery) ed etichetta troppo
  permissiva. → redesign per asse, sull'intera gallery, con floor (§5.2-§5.3).

### 6.2 Floor, classi e cosa discrimina `[DONE]`

- Floor casuale test 0.7403 / 0.4663 / 0.8700 (§5.3). Spazio disponibile: comp 0.26, topo 0.53,
  **geom 0.13**.
- Classi di equivalenza (F5): composizione mediana 5007 (saturazione), topologia mediana 19 con 311
  query singleton su 2000.
- La fusione sul test (§6.9) copre **0.427 / 0.411 / 0.647** dello spazio utile: la pianta intera è
  molto meno «quasi risolta» di quanto suggeriscano 0.85 / 0.69 / 0.95 (`status.md §55`).

### 6.3 Ramo graph sulla pianta intera — cosa impara la rete `[DONE]`

Valid, gallery condivisa (le run graph del 10 ago usano già la gallery `0c24cfc05e18`), coppie
**simmetriche** (training storico, non la config finale). `status.md §2, §16, §21.5`.

| run | C | T | G |
|---|---|---|---|
| `hist` (training-free) | 0.9999\* | 0.6428 | 0.8927 |
| `gat/base` | 0.9281 | 0.6997 | 0.9381 |
| `gat/nosym` | 0.9085 | 0.7441 | **0.9453** |
| `sage/nd01` | 0.9421 | 0.7622 | 0.9342 |
| **`gcn/tau02`** (miglior full) | **0.9741** | **0.8298** | 0.9408 |

\* oracolo per costruzione.

Tre risultati da riportare:
1. **La val-loss InfoNCE non predice il retrieval** (graph) `[CLAIM]`: classifica per val-loss =
   inverso di quella per retrieval (GAT loss migliore 2.08 e nDCG peggiore; GCN loss peggiore 2.73 e
   nDCG migliore). GAT tocca il massimo di retrieval all'epoca 3 e poi peggiora per 20 epoche mentre
   la val-loss migliora (2.952 → 2.235). **Meccanismo**: InfoNCE è instance discrimination; con classi
   di composizione enormi, in un batch da 256 circa 21 «negativi» sono in realtà massimamente
   rilevanti e la loss li allontana. Fix: selezione con una sonda di retrieval sul valid.
2. **Quasi tutto viene dall'architettura** `[CLAIM]` (sonda: gallery 5000, 500 query): add-pool delle
   feature grezze **senza rete** = 0.884 / 0.625 / 0.906 (media 0.805); GNN a pesi casuali 0.799-0.809;
   GCN allenata 0.8615. Il training vale C +0.056, **T +0.100**, G +0.012: la rete impara soprattutto
   a rendere confrontabile la **topologia**; composizione e geometria escono dall'add-pool (che *è*
   l'istogramma dei tipi).
3. **Ablation OFAT** (appaiata sul valid, topologia): τ 0.2 vs 0.3 **+0.0358**, senza flip/rot su GAT
   **+0.0444**, node-drop 0.1 vs 0.2 **+0.0318** (tutte CI > 0). Le tre modifiche di tuning del 28
   lug erano peggiorative sulla pianta intera (⚠️ ma restano nel YAML `gat` usato dalla config
   finale, §5.6).

Contro `hist` la GNN migliore vale sulla topologia +0.187 di nDCG ma soprattutto **mAP@10 0.351 vs
0.015 (24×)**, Recall@10 0.412 vs 0.046 (sul 84,5% di query non singleton).

### 6.4 Ramo vision: pianta intera vs robustezza `[DONE]`

**(a) Pianta intera — benchmark degli 8 encoder** (valid, 2000 query; ⚠️ protocollo **A**: gallery
67.453 e whitening stimato su tutta la gallery — differenze col protocollo B misurate trascurabili,
§5.1; `status.md §13, §18, §20.1`):
- Migliori per asse: composizione `tipsv2/mean448/raw` **0.8509**; topologia `tipsv2/gem448/whiten`
  **0.6896**; geometria `ijepa/gem/whiten` **0.9473**. `tipsv2/gem448/whiten` = 0.8433 / 0.6896 / 0.9347
  (spazio utile 39.9% / 42.0% / 50.2%).
- **Non esiste un vincitore**: 12 configurazioni su 74 sulla frontiera di Pareto, scambio geometria ↔
  (composizione + topologia).
- **Il whitening compra topologia** in 14/14 coppie dei 5 encoder storici (+0.0305 medio); su PE-Core
  +0.0759 (spazio contrastivo molto anisotropo, interpretazione non verificata).
- TIPSv2: **448 > 224** in 6/6 coppie su comp/topo; geometria piatta.
- **Più recente / più grande non vince** `[CLAIM]`: DINOv2 ≈ DINOv3 ≈ SigLIP2 entro il rumore
  (§7.6); I-JEPA ViT-H vince solo la geometria; RADIO (embedding 2304-d) fra i peggiori. Lettura: il
  collo di bottiglia è il **domain gap** (colormap sintetiche), non la taglia.
- Pattern costante: **geometria > composizione > topologia** in nDCG grezzo per tutti gli encoder.
  ⚠️ I floor per asse sono molto diversi: il confronto fra assi va fatto sullo spazio utile, dove il
  divario si riduce.

**(b) Robustezza — la griglia completa, la tabella finale del valid** (protocollo B, metro R,
53 config = 52 frozen + la head dinov3 di riferimento; `status.md §43-§44`):

| # | configurazione | **R** | stanze tolte | crop | patch |
|---|---|---|---|---|---|
| 1 | **`pespatial/gem/whiten`** (scelta) | **0.5245** | 0.3920 | 0.6605 | 0.5209 |
| 2 | `radio/natural/whiten` | 0.5161 | 0.3821 | 0.6032 | 0.5631 |
| 3 | `pespatial/natural/whiten` | 0.5148 | 0.3707 | 0.6855 | 0.4881 |
| 4 | `radio/gem/whiten` | 0.5077 | 0.3809 | 0.6168 | 0.5254 |
| 5 | `radio/mean/whiten` | 0.4961 | 0.3609 | 0.6056 | 0.5218 |
| 6 | `pespatial/mean/whiten` | 0.4799 | 0.3382 | 0.6216 | 0.4799 |
| 7 | `ijepa/natural/whiten` | 0.4259 | 0.2234 | 0.5108 | 0.5435 |
| 8 | `ijepa/gem/whiten` | 0.4067 | 0.2277 | 0.4907 | 0.5016 |
| 9 | `dinov3/mean/whiten` | 0.3855 | 0.2798 | 0.4638 | 0.4129 |
| 10 | `dinov2/natural/whiten` | 0.3810 | 0.2983 | 0.4545 | 0.3900 |
| 17 | `dinov3/natural/head` (head di riferimento) | 0.2915 | 0.3942 | 0.2017 | 0.2785 |

- #1 − #2 = **+0.0084 [+0.0030, +0.0138]**; tutti i delta del #1 hanno CI > 0 → nessuno spareggio.
- **Il migliore per singolo danno cambia ogni volta** (stanze → head; crop → `pespatial/natural`;
  patch → `radio/natural`): è il motivo per mediare sui tre.
- Mediane di R per encoder (F6): pecore 0.149 · siglip2 0.162 · tipsv2 0.210 · dinov2 0.259 · ijepa
  0.266 · dinov3 0.276 · radio 0.335 · **pespatial 0.386**. La scelta dell'encoder conta più della
  combinazione; il primo posto non è speciale (radio a 0.008).
- **Whitening > raw in 26/26** coppie encoder × pooling.
- TIPSv2, migliore sulla pianta intera, è fuori dalle prime 14 (migliore R 0.3168, e a 224 non a 448)
  `[CLAIM]`: **«bravo sul completo» e «bravo sull'incompleto» sono qualità diverse**.
- **Costo sulla pianta intera** di `pespatial/gem/whiten` (0.8130 / 0.6306 / 0.9377):

| rispetto a | composizione | topologia | geometria |
|---|---|---|---|
| head dinov3 di riferimento | −0.0163 [−0.0191, −0.0135] | −0.0164 [−0.0203, −0.0126] | −0.0020 [−0.0030, −0.0010] |
| `dinov3/natural/whiten-train` | −0.0120 [−0.0145, −0.0095] | −0.0321 [−0.0357, −0.0287] | +0.0030 [+0.0020, +0.0039] |

- **H2 — a parità di area conta l'informazione o l'artefatto?** (pre-registrata, `status.md §34, §44`):
  a f=0.75 la patch fa **peggio** del crop in **7 encoder su 8** (1 pari) → «artefatto»: la griglia di
  tasselli bianchi disturba l'embedding più di quanto aiutino i pezzi di stanza rimasti. ⚠️ A f=0.75 i
  valori sono vicini al pavimento, e a danno moderato il segno era opposto: il verdetto vale per il
  livello pre-registrato.

### 6.5 La projection head non si trasferisce `[DONE]` `[CLAIM]`

**H3** (pre-registrata, `status.md §46, §48`): una head allenata su un danno che toglie davvero
informazione (stanze tolte coi muri) migliora anche danni **mai visti** (crop, patch)? Encoder
`pespatial/gem`, epoca scelta **solo** sulle stanze tolte (crop e patch non toccano né pesi né epoca).
Candidato migliore = head + whitening (R 0.4912).

| danno | head+whiten | frozen `pespatial/gem/whiten` | Δ |
|---|---|---|---|
| stanze tolte (in distribuzione) | 0.5392 | 0.3920 | **+0.1472** [+0.1371, +0.1572] |
| crop | 0.5191 | 0.6605 | **−0.1415** [−0.1496, −0.1332] |
| patch | 0.4154 | 0.5209 | **−0.1055** [−0.1122, −0.0988] |
| **R** | 0.4912 | 0.5245 | **−0.0333** [−0.0384, −0.0281] |

→ **H3 falsificata**, head non adottata. Costo sul full: composizione pari, topologia −0.0177,
geometria −0.0094. Stesso segno trovato prima su un altro encoder (dinov3) e un altro render
(crop −0.183, patch −0.164, stanze +0.090, `status.md §37`).
**Da scrivere**: la head impara **il danno che vede**, non una robustezza generale. È anche una
lezione di protocollo: valutare la robustezza su un solo danno (quello di training) premia la
specializzazione.

### 6.6 Ramo graph sotto danno `[DONE]`

`status.md §30, §32, §42, §45-§47`. Valid, AUC di self-recovery sulle stanze tolte.

1. **Con coppie simmetriche il graph crolla** `[CLAIM]`: AUC 0.004-0.024 per tutte le GNN (`gcn/tau02`
   0.0058; il self esce dal top-100 nel 91% delle query già a f=0.25), pur ritrovando il self a f=0
   (MRR 0.964-0.977). Le metriche per-asse restano sopra il floor: ritrova piante *simili*, non la
   propria. **Meccanismo** (interpretazione): un grafo con il 25% di stanze in meno è il grafo
   **completo** plausibile di molte piante più piccole; nel vision invece i pixel delle stanze
   superstiti sono identici all'originale.
2. **Coppie asimmetriche intero ↔ parziale** (opzione D, regola d'uscita pre-registrata): AUC
   gcn 0.0058 → 0.0968, sage 0.0089 → 0.1105, **gat 0.0071 → 0.2490** (tutte CI > 0).
3. **Scegliere il checkpoint sulla robustezza, non sulla pianta intera** (`status.md §47`): sulla
   stessa traiettoria di training la regola vecchia (sonda full di topologia) sceglie l'epoca 6, quella
   nuova la 294 → AUC **0.1071 → 0.4568** (Δ +0.3498 [+0.3394, +0.3602]). La regola vecchia è
   **erratica** fra training identici (epoche 6 / 31 / 33 → AUC 0.107 / 0.249 / 0.267).
   Allenare a lungo e scegliere sulla robustezza **non vende la topologia** (stesso training: C
   −0.0336, T +0.0076, G +0.0145).
4. **Config graph `gat/asymrob`**: AUC valid 0.4568 (f = 0.25 / 0.5 / 0.75: 0.8541 / 0.4279 / 0.0886).
   Full (valid) 0.8603 / 0.6780 / 0.9498. Costo descrittivo rispetto al miglior graph a pianta intera
   (`gcn/tau02`, §6.3, stesse query): composizione −0.114, topologia −0.152, geometria +0.009 — la
   robustezza del graph si paga cara sulla pianta intera (confronto non calcolato appaiato).
5. **Marcatore «vicini persi»** (una feature per stanza = quante vicine sono state tolte): a selezione
   equa **pareggia** (Δ AUC −0.0039 [−0.0137, +0.0059]) pur essendo migliore sulla pianta intera
   (topologia +0.057). Non adottato (regola pre-registrata). Ipotesi non verificata in §9.
6. **La robustezza si trasferisce a un danno diverso** (`semantic`: tiene soggiorno, cucina, bagno e
   toglie il resto, ~54% delle stanze): Δ self_rr `asymrob − base` **+0.3658 [+0.3484, +0.3831]**,
   rapporto 0.86 rispetto a `random f=0.5`. Al contrario della head vision (§6.5), qui c'è
   trasferimento — ma il danno resta di tipo «stanze», non è un cambio di modalità.
7. **SAGE vs GAT a parità di regola** (`status.md §58-§58.1`, `VERIFICA_GRAFI.md`; analisi aggiuntiva
   pre-registrata il 1 ott). Il punto 2 aveva scelto GAT con la regola di selezione vecchia (epoche
   precoci ed erratiche, SAGE all'epoca 3). Riallenato `sage/asymrob` con **la stessa ricetta di
   `gat/asymrob`** (coppie asimmetriche, selezione sulla robustezza, tetto 300; YAML SAGE invariato:
   τ 0.3, `aggr` mean) in 4 repliche, appaiate a quelle di GAT (stesse 2000 query, gallery, stanze
   tolte). Valid, AUC self-recovery, Δ = SAGE − GAT, CI 95% bootstrap:

   | replica | SAGE | GAT | Δ [CI] |
   |---|---|---|---|
   | 0 (quella della fusione) | 0.4779 | 0.4568 | +0.0210 [+0.0123, +0.0298] |
   | s100042 | 0.4979 | 0.4691 | +0.0288 [+0.0201, +0.0371] |
   | s200042 | 0.4882 | 0.4719 | +0.0163 [+0.0078, +0.0250] |
   | s300042 | 0.5030 | 0.4670 | +0.0360 [+0.0275, +0.0445] |
   | media | 0.4918 | 0.4662 | **+0.0255** |

   - **Verdetto** (regola pre-registrata): **SAGE più robusto** (CI replica 0 > 0 e media dei 4 Δ > 0).
     Previsione «GAT più robusto» ❌.
   - **Compromesso**: sulla pianta intera SAGE ritrova il self un po' meno (MRR f=0 0.963-0.967 vs
     0.977-0.979 di GAT, stesse query). **Deviazione dichiarata**: il controllo di validità «MRR f=0 ∈
     [0.965, 0.980]», calibrato su GAT per scovare guasti di pipeline, fallisce in 2 repliche su 4
     (0.9644, 0.9628); accettato dopo la diagnosi (gallery, query, split e danno identici; nessun guasto).
   - Tetto vincolante anche per SAGE (epoca scelta 295 / 299 / 297 / 299 su 300): stima per difetto.
   - **Cosa cambia**: la fusione resta con GAT (scelta dichiarata prima dei numeri; SAGE non va sul test).
     Il guadagno della fusione è misurato con un ramo graph che non è il più robusto disponibile:
     se mai è **conservativo**, la tesi non ne esce indebolita. Il vantaggio di GAT del punto 2
     (0.2490 vs 0.1105) era un effetto della regola di selezione, non dell'architettura.
   - Perimetro: vale per questa ricetta; τ e `aggr` non ottimizzati, GCN non riprovato, solo valid.

### 6.7 Vision vs graph sotto danno `[DONE]`

A parità di stanze tolte (valid): vision `pespatial/gem/whiten` **0.3920**, graph `gat/asymrob`
**0.4562** (dai file della fusione, α=1 / α=0). Per livello (graph / vision): f=0.25 0.8547 / 0.8181 ·
f=0.5 0.4258 / 0.2971 · f=0.75 0.0880 / 0.0609. Correlazione per query fra le AUC dei due rami
(Spearman) **0.092**: sbagliano su query diverse — è la premessa della fusione.
⚠️ Confronto descrittivo: asimmetria del danno (§5.4) e circolarità (§7.1).

### 6.8 Late fusion e i due controlli (valid) `[DONE]`

`status.md §49-§51.1`.

| α | 0 (graph) | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | **0.6** | 0.7 | 0.8 | 0.9 | 1 (vision) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC stanze tolte | 0.4562 | 0.5029 | 0.5436 | 0.5797 | 0.6049 | 0.6213 | **0.6301** | 0.6258 | 0.5922 | 0.5159 | 0.3920 |

- α* = 0.6 (unico pari; curva unimodale). Fusione − graph **+0.1740 [+0.1642, +0.1838]**; − vision
  +0.2381 [+0.2285, +0.2476]. ~4× il rumore fra training del graph.
- **Per livello** (graph / vision / fusione): f=0.25 0.8547 / 0.8181 / **0.9553** · f=0.5 0.4258 /
  0.2971 / **0.7201** · f=0.75 0.0880 / 0.0609 / **0.2151**.
- **Oracolo per query** 0.5557: la fusione lo supera di **+0.0744**, su 1255/2000 query.
  **Meccanismo verificato** (`status.md §49.3`): nelle query in cui solo la fusione mette l'originale
  al 1° posto, di norma **un** ramo lo tiene vicino alla cima e l'altro abbassa i concorrenti che lo
  precedevano.
- **Spread delle similarità** top-10 sotto danno dello stesso ordine nei due rami → α non è
  schiacciato verso un estremo (α è un peso nominale, non «0.5 = equilibrato»).
- **Pianta intera** (descrittivo, circolare): fusione 0.8501 / 0.6841 / 0.9536 · graph 0.8603 / 0.6780 /
  0.9498 · vision 0.8130 / 0.6306 / 0.9377. Fusione − graph: C **−0.0102** · T **+0.0061** · G
  **+0.0039** (CI > 0 o < 0 su tutti).

**Controllo 1 — graph + graph** (due training identici, cambia solo l'inizializzazione): β* = 0.4,
guadagno sul componente migliore **+0.0336 [+0.0286, +0.0387]**; differenza dei guadagni D = G_F − G_C
= **+0.1403 [+0.1287, +0.1519]**. ⚠️ È il controllo **minimo** (due training identici sono l'insieme
meno diverso), non «severo» come scritto prima dei job.

**Controllo 2 — vision + vision** (`pespatial/gem/whiten` + `radio/natural/whiten`: stessa immagine,
modelli diversi): γ* = 0.5, guadagno **+0.0982 [+0.0916, +0.1047]**; D₂ = **+0.0758 [+0.0639,
+0.0876]**, fuori dal margine di equivalenza pre-registrato ±0.04 → esito **complementarità**; ma
quota G_V / G_F = **0.564** (previsione «< 0.5» smentita).

**Scala dei tre guadagni** `[CLAIM]`: stesso modello ripetuto **+0.034** · modelli diversi, stessa
informazione **+0.098** · informazione diversa **+0.174**. Correlazione di Spearman per query fra i
componenti: vision/graph **0.092** · pespatial/radio 0.622 · graph/replica 0.649.
A danno zero (fuori dal metro) vision + vision non rompe nessun pareggio fra duplicati (MRR 0.9707
per tutti i γ), vision + graph sale a 0.98: indizio in più di informazione diversa.

**Sintesi**: il guadagno della fusione è in parte effetto d'insieme fra modelli diversi (~56%) e in
parte informazione complementare (~44%). Limite strutturale: due ViT sono meno diversi fra loro di un
ViT e una GNN, quindi informazione e modello non si separano del tutto; il controllo vision + vision
parte inoltre favorito (componente migliore più debole, 0.392 vs 0.456).

### 6.9 Test finale (letto una volta, pre-registrato) `[DONE]`

`status.md §52` (pre-registrazione approvata il 18 set, prima dei job), §53 (risultati). Nessuna
scelta sul test: config congelate, **α = 0.6 fisso dal valid**.

| stanze tolte (AUC self-recovery) | `hist` | vision | graph | **fusione α=0.6** |
|---|---|---|---|---|
| media f = 0.25/0.5/0.75 | 0.0013 | 0.3965 | 0.4700 | **0.6424** |
| f=0.25 | 0.0026 | 0.8216 | 0.8620 | **0.9580** |
| f=0.5 | 0.0011 | 0.3037 | 0.4471 | **0.7363** |
| f=0.75 | 0.0003 | 0.0642 | 0.1008 | **0.2331** |

- **Fusione − graph +0.1725 [+0.1625, +0.1826]** (valid +0.1740: l'ottimismo della scelta di α è
  trascurabile). Oracolo per query 0.5693: la fusione lo supera di **+0.0731**, su **1218/2000** query.
- **Vision col metro a tre danni**: R **0.5210** (stanze 0.3965 · crop 0.6543 · patch 0.5123); valid
  0.5245.
- **Pianta intera** (descrittivo, circolare): `hist` 0.9998\* / 0.6428 / 0.8936 · graph 0.8585 /
  0.6748 / 0.9494 · vision 0.8144 / 0.6328 / 0.9372 · **fusione 0.8512 / 0.6856 / 0.9541**.
  Fusione − graph: C −0.0073 [−0.0096, −0.0049] · T **+0.0107** [+0.0074, +0.0141] · G +0.0047
  [+0.0040, +0.0053]. Oracolo per asse 0.8702 / 0.6963 / 0.9537.
- **Controlli**: tutti passati (stesse stanze e query; α=1 e α=0 riproducono i rami esattamente; MRR
  senza danno 0.970-0.979; doppio rendering del danno vision 0/1/0 query diverse).
- **Previsioni**: rami entro ±0.01 dal valid → vision ✅, metro a tre danni ✅, **graph ❌** (+0.0138,
  fuori di poco e in meglio) · Δ fusione > +0.12 ✅ · ordine fusione > graph > vision ✅ · pianta intera
  come previsto ✅.
- La baseline training-free **sotto danno non ritrova nulla** (0.0013): graph e fusione sono 2-3 ordini
  di grandezza sopra. (Anche senza danno `hist` ritrova la pianta esatta solo nel 2,6% dei casi: il suo
  0.9998 di composizione è tautologia.)
- ⚠️ I controlli di complementarità (§6.8) restano del **valid**.

### 6.10 Repliche multi-seed `[DONE]`

`status.md §57` (pre-registrazione del 1 ott, prima dei job), §57.3, `results/fusion/seeds/summary.json`.
Fra una replica e l'altra cambiano **training del graph** (seed degli augment/coppie; init e ordine dei
batch non seedati) e **stanze tolte** (stesso seed nei due rami); stesse 2000 query e stessa gallery.
α scelto sul valid per ogni replica con la regola di §5.7; test con quell'α.

| test, AUC stanze tolte | fusione | graph | vision | R vision |
|---|---|---|---|---|
| replica 0 (storica, §6.9) | 0.6424 | 0.4700 | 0.3965 | 0.5210 |
| s100042 / s200042 / s300042 | 0.6426 / 0.6392 / 0.6360 | 0.4659 / 0.4742 / 0.4687 | 0.3914 / 0.3928 / 0.3856 | 0.5225 / 0.5208 / 0.5217 |
| **media ± sd (n=4)** | **0.6401 ± 0.0031** | **0.4697 ± 0.0035** | **0.3916 ± 0.0045** | **0.5215 ± 0.0008** |

- Valid n=4: fusione 0.6397 ± 0.0066 · graph 0.4661 ± 0.0069 · vision 0.3907 ± 0.0015 · R 0.5246 ±
  0.0013 (la replica 0 è la più bassa su fusione e graph: nessun ottimismo da selezione visibile).
- Guadagno della fusione sul test per replica: +0.1767 / +0.1650 / +0.1674 (storico +0.1725), tutti CI
  > 0. **α* = 0.6 in tutte e tre**.
- Previsioni: P2 sd fusione < sd graph ✅ · P3 la fusione aiuta in 3/3 ✅ · P4 α* ∈ {0.5, 0.6, 0.7} ✅ ·
  P5 ordine fusione > graph > vision in ogni replica ✅ · **P1 ❌ sul test** (sd vision 0.0045 > sd graph
  0.0035: il campione di stanze tolte pesa sul vision quanto il training sul graph).
- **Lettura** `[CLAIM]`: la sd fra repliche (~0.003) è ~50× più piccola del guadagno della fusione
  (~0.17).
- Da dichiarare: best epoch del graph al tetto (300 / 293 / 300); una replica (s300042) ha richiesto
  di trattare come pareggi 11 query a f=0.75 del valid che differivano entro 1 ulp float32 (regola
  estesa dall'utente prima di select/test, `status.md §57.2`); sul test tutti i controlli esatti.
- ⚠️ Il rumore fra training ~0.04 misurato in §47 su una sola replica **non** si ripresenta qui (graph
  test 0.466-0.474); probabile che §47 confrontasse condizioni diverse — non verificato.

---

## 7. Discussione & caveat (onestà metodologica, da scrivere esplicitamente)

### 7.1 Circolarità — nessun asse è esente e nessun ramo è pulito `[DECISION]` ⚠️ aperta

Misurato il 24 ago (`status.md §21`, `architecture.md § Questione aperta`):
- su **composizione e topologia** il graph riceve in input (`rType`, `rEdge`) ciò che la GT misura;
- **anche la GT geometrica** è funzione dell'input del graph (footprint = unione esatta dei box, §4);
- **anche il vision legge `rType`**, dal colore, ma compresso 13 → 6 classi.

Il confronto vision↔graph per asse non è «simbolico vs visivo» ma **fine vs grosso sullo stesso
input**: il benchmark misura quanto bene si comprime in uno spazio metrico un'informazione già
posseduta, non la capacità di inferire la struttura da un input grezzo.

**Proposta di presentazione** (`roadmap.md §2` punto 5, da confermare):
- il **criterio principale** (self-recovery) **non usa label** → immune alla circolarità;
- i confronti **dentro** un ramo sono equi;
- solo i confronti vision↔graph **per asse** ne sono toccati: si presentano come descrittivi
  («informazione esatta vs grossolana»).
- Regime reale: su planimetrie non colorate (CAD, scansioni) il segnale del colore sparisce → i numeri
  per-asse del vision su RPLAN sono ottimistici.

### 7.2 La geometria aggregata dipende dai pesi della formula `[DECISION]`

`status.md §22` (criterio fissato prima dei dati). Rifacendo il gain geometrico con 6 pesature, 2 su 6
**invertono** il vincitore vision vs graph con CI che esclude lo zero, e l'inversione (−0.033) è più
grande del vantaggio di baseline (+0.002).
- **Non si sceglie la pesatura che vince**: si dichiara `(1,1,1)` come convenzione e si pubblica la
  tabella di sensibilità.
- **Claim decomposto** (più forte): sulla forma del footprint (area + aspect) il vision batte il graph
  di +0.019 / +0.027 (~10× l'aggregato); perde sulla distribuzione di area per tipo.
- **Meccanismo**: la distribuzione di area per tipo è una **somma** sulle feature dei nodi (esattamente
  ciò che l'add-pool del graph calcola gratis); area e aspect del footprint richiedono un **min/max**,
  che l'add-pool non fornisce. Il confine è quale aggregazione l'architettura implementa nativamente.
- Vale anche dentro il vision: «I-JEPA è il migliore in geometria» inverte su 1 pesatura su 6.
- ⚠️ Questi numeri sono del protocollo A (agosto), con le config di allora; non rifatti sulle config
  finali.

### 7.3 Un artefatto di rendering e lo *shortcut learning* `[DECISION]` (vedi nota in §10)

`status.md §35-§37`, figura F3 (pannello destro).
- Il primo masking a stanze cancellava solo i muri scuri (perimetro, grigio 79) e lasciava quelli
  interni (grigio 128): la stanza tolta restava una **cella bianca contornata** (forma e posizione
  visibili, tipo perso). Misurava «stanze svuotate», non «stanze tolte».
- A parità di query e di stanze tolte, i muri rimasti valevano **quasi tutta la robustezza** misurata
  del vision: `dinov3/natural/whiten` AUC 0.536 con i muri vs 0.304 senza (+0.336 a f=0.5); per la head
  allenata su quel render il self-recovery a f=0.75 passava da 0.840 a 0.036.
- **Non è una fuga di dati** (nessun dato di valutazione nei pesi, gradiente solo sul train): è
  **shortcut learning** su un difetto condiviso da coppie di training, sonda di selezione e misura.
- È stato trovato **perché** si era pre-registrata la verifica su danni diversi (crop, patch). Lezione
  `[CLAIM]`: **un criterio di robustezza va validato su più tipi di danno**; con un solo danno si
  misurano anche i suoi artefatti. Il metro finale (§5.4) nasce da qui.
- Claim caduti, da non usare mai senza qualifica `[NO-CLAIM]`: «il vision è robusto al masking», «la
  head vale +0.284 sotto masking».

### 7.4 Limiti del metro finale `[DECISION]`

`status.md §38`.
- Fissato dopo aver visto **2** configurazioni (entrambe dinov3/natural); nessuna delle altre 51 era
  nota.
- Il livello f non è la stessa quantità nei tre danni (superficie per crop e patch, frazione di stanze
  per le stanze tolte): i pesi uguali sono sui livelli **nominali**. A f=0.5 le aree davvero tolte sono
  comunque vicine (crop 0.498, patch 0.500, stanze 0.522).
- Griglia delle patch dipendente dall'encoder.
- Controllo pre-registrato di bit-identità non passato alla lettera (6/2000 scambi di rango 1↔2 fra
  duplicati): accettato come rumore, 2-3 ordini di grandezza sotto gli effetti.

### 7.5 Limiti della fusione `[DECISION]`

`status.md §49-§51.1`.
- α* scelto fra 11 valori sulle stesse query del valid → ottimista; **verificato sul test** (+0.1725 vs
  +0.1740) e sulle repliche.
- Asimmetria del danno (il vision tiene la sagoma del buco).
- Verdetto legato al checkpoint graph; le repliche (§6.10) lo estendono a 4 training.
- Il ramo graph della fusione (GAT) **non è il più robusto** della ricetta: SAGE fa +0.026 sul valid
  (§6.6 punto 7). Fusione non rifatta con SAGE (scelta pre-registrata): il guadagno è conservativo.
- Le differenze per asse sulla pianta intera sono descrittive e circolari.
- I due controlli non sono equi allo stesso modo: in graph + graph i componenti sono più forti della
  fusione vera (si confrontano i guadagni, non i valori); in vision + vision il componente migliore è
  più debole (il controllo parte favorito).
- Pareggi esatti: rilanciare una fusione non riproduce i file bit per bit (FAISS ordina i pareggi in
  modo diverso col numero di thread); i numeri riportati sono quelli dei job.

### 7.6 Selezione su un solo split di validazione `[DECISION]`

Il test è stato letto in più momenti; tutti vanno dichiarati:
1. **Luglio** — benchmark vision e graph valutati sul test (selezione sul test): numeri **non citabili**,
   sostituiti da quelli del valid (§6.4a).
2. **8-9 ago** — tre varianti vision sul test, pipeline a pianta intera: `siglip2/mean/whiten`
   (pre-registrata prima di vedere il test), `dinov2/natural/whiten`, `dinov3/natural/whiten`. nDCG@10:
   0.8259 / 0.6537 / 0.9356 · 0.8262 / 0.6590 / 0.9350 · 0.8243 / 0.6618 / 0.9342. **Il confronto si
   ribalta** fra valid e test (il vantaggio di SigLIP2 sulla composizione sparisce, il pareggio in
   topologia diventa sconfitta significativa): il margine di scelta (~0.004) era dell'ordine del drift
   valid → test. `[CLAIM]` «Le varianti di testa sono indistinguibili; la selezione su un solo split di
   validazione non si replica». `[NO-CLAIM]` «il miglior encoder vision è X» sulla pianta intera.
3. **18 set** — test finale pre-registrato (§6.9): **i numeri del paper**.
4. **1 ott** — repliche pre-registrate (§6.10), nessuna scelta.

### 7.7 Il proxy di rilevanza non è giudizio umano

Nessuna validazione umana fatta (future work). Il self-recovery, il metro principale, non dipende dal
proxy.

### 7.8 Previsioni pre-registrate smentite (riportarle così)

| Ipotesi (scritta prima) | Esito | Dove |
|---|---|---|
| La val-loss InfoNCE indica la qualità del retrieval (graph) | ❌ invertita | §6.3 |
| τ più alto smorza i falsi negativi e aiuta | ❌ τ 0.3 costa 0.038 di topologia, τ 0.5 il peggiore | §6.3 |
| Allenare più a lungo il graph (pianta intera) migliora | ❌ +0.003 GCN, SAGE peggiora | §10 |
| Encoder più recenti / più grandi vincono | ❌ domain gap | §6.4 |
| La head si trasferisce a danni non visti (H1, H3) | ❌ due volte | §6.5 |
| A parità di area, la patch conserva più informazione del crop (H2) | ❌ «artefatto» 7/8 | §6.4 |
| Coppie asimmetriche: topologia full ≈ invariata | ❌ (poi spiegata dalla selezione precoce) | §6.6 |
| Marcatore «vicini persi» più robusto | ❌ pareggio | §6.6 |
| Fusione: topologia sotto il graph sulla pianta intera | ❌ è sopra (+0.006 valid, +0.011 test) | §6.8-§6.9 |
| Controllo graph+graph: pianta intera entro ±0.01 | ❌ di poco (topologia +0.0116) | §6.8 |
| Controllo vision+vision: guadagno < metà di quello vision+graph | ❌ 0.564 | §6.8 |
| Graph sul test entro ±0.01 dal valid | ❌ +0.0138, in meglio | §6.9 |
| Repliche: sd vision < sd graph (P1) | ❌ sul test | §6.10 |
| A parità di regola di selezione GAT resta più robusto di SAGE | ❌ SAGE +0.026 (4/4 repliche) | §6.6 |

### 7.9 Claim che il paper NON può fare `[NO-CLAIM]`

«Il vision è robusto al masking» · «la head aiuta sotto masking» (solo sul danno di training) · «il
vision batte il graph in geometria» (dipende dai pesi) · «un asse o un ramo è pulito» · «sul full la
head non aiuta» (dipende da metro e head) · «la val-loss non predice il retrieval» e «allenare di più
non serve» **in generale** (valgono per il graph, non per la head) · «il miglior encoder vision è X a
pianta intera» · qualunque cosa fuori da RPLAN · «GAT è l'encoder graph più robusto» (a parità di regola
vince SAGE, §6.6) · «SAGE è il miglior encoder graph» (una sola ricetta, GCN non riprovato, solo valid,
mai sul test) · «la fusione con SAGE sarebbe migliore» (non misurata).

---

## 8. Limitazioni

- **Solo RPLAN.** ResPlan e CubiCasa5K in stand-by (nessuna label di rilevanza, fuori dominio,
  tassonomia non allineabile senza perdite); Maticad rimandato.
- **Danni sintetici**, non piante parziali reali; la stanza tolta lascia la forma del buco. Crop e
  patch esistono solo per il vision: vision↔graph e fusione sono confrontati solo sulle stanze tolte.
- **Graph dai `.mat` esatti** (regime con l'informazione strutturale perfetta); il grafo estratto
  dall'immagine non è stato provato.
- **Backbone vision mai aggiornato** (frozen; head provata e scartata; niente LoRA).
- **Circolarità** su tutti e tre gli assi (§7.1); proxy non validato con umani.
- **Un solo split di validazione** (§7.6).
- **Budget di training graph vincolante** (300 epoche, best epoch al tetto): robustezza graph
  sottostimata.
- **Encoder graph della fusione non ottimale**: a parità di regola SAGE è più robusto di GAT (+0.026
  sul valid, §6.6 punto 7); la fusione resta con GAT. Iperparametri graph (τ, `aggr`, augment) non
  ri-ottimizzati per la selezione sulla robustezza; GCN non riprovato con la regola nuova.
- **Gallery = inner join**: 48 PNG senza `.mat` escluse.
- **Head vision** provata con un solo tipo di danno in training per volta; una head allenata su più
  danni insieme e giudicata su un danno ancora diverso non è stata provata.
- **Pooling `gem`** non standard (§5.5).
- Late fusion con **un solo peso** globale; fusioni apprese non provate.

---

## 9. Conclusioni & future work

**Conclusioni**: due rami sulla stessa gallery, protocollo per asse con floor e confronti appaiati,
criterio di robustezza su tre danni; late fusion che sul test porta l'AUC di self-recovery da 0.470
(graph) / 0.392 (vision) a **0.640 ± 0.003** (4 repliche), sopra l'oracolo per query; guadagno
scomposto in effetto d'insieme (~56%) e informazione complementare (~44%). Lezioni di metodo: la head
impara il danno che vede; un criterio di robustezza va validato su più danni; scegliere il checkpoint
sulla stessa misura su cui si giudica.

**Future work** (tagliati in chiusura, `roadmap.md §3`):
- **Baseline LayoutGKN** nel nostro protocollo (al posto del ramo graph e dentro la fusione):
  in corso di valutazione (`COMPETITORS.md §2.1`, `.claude/TODO.md`).
- Fusione intermedia: testa congiunta su [vision ; graph] (sarebbe una seconda tornata pre-registrata,
  test letto in un secondo momento, dichiarato), distillazione, cross-attention patch ↔ nodi.
- Grafo estratto dall'immagine (confronto senza circolarità).
- Multi-dataset (ResPlan, CubiCasa5K, MSD).
- Quarto danno «da disegno» (togliere muri/porte in ordine di disegno, come SketchPlan); danno a stanze
  senza la forma del buco (richiede di ridisegnare dal `.mat`).
- Head vision allenata su più danni e giudicata su un danno fuori distribuzione.
- GED topologica (stile SSIG), metriche diagnostiche top-K, validazione umana del proxy.
- Ramo graph: fusione con SAGE (seconda tornata pre-registrata, test letto a parte e dichiarato);
  GCN con la selezione sulla robustezza; τ e `aggr` ri-ottimizzati con la regola nuova; budget oltre
  300 epoche.
- **Marcatore «vicini persi» che pareggia** (`status.md §41-§42, §47`): *ipotesi*, non verificata — la
  geometria delle stanze rimaste (posizione e dimensioni sulla griglia 256) rivela già dove manca
  qualcosa, e la loss asimmetrica obbliga già la rete a inferire ciò che manca. Test possibili:
  marcatore con la geometria oscurata; più repliche e budget oltre 300 epoche. Scartata la variante
  «due viste entrambe degradate» (la gallery è sempre completa: senza la coppia incompleta ↔ completa il
  training non allena il compito).
- Fase 2 generativa (altro progetto).

---

## 10. Storia del progetto — cosa abbiamo provato, cosa non ha funzionato e perché

In ordine cronologico, con il **perché** di ogni svolta. **Esclusi** gli inciampi dovuti a codice
buggato o a impostazioni concettualmente sbagliate (lista in fondo). Utile per la narrativa del
paper, per la sezione di ablation e per le domande dei revisori.

**Giugno — il benchmark degli encoder vision.**
- Pipeline vision con 5 encoder (DINOv2, DINOv3, SigLIP2, RADIO, I-JEPA), poi protocollo per asse
  (§6.1). Primo risultato robusto su tutte le misure successive: **più nuovo o più grande non vince**;
  il collo di bottiglia è il domain gap fra immagini naturali e colormap sintetiche.
- Il **whitening** si rivela necessario: senza, gli embedding sono collassati.

**Luglio — il ramo graph e la scoperta sulla loss.**
- GNN (GCN, SAGE, GAT) con InfoNCE su coppie aumentate. Allenare 3× più a lungo **non serve** (GCN
  +0.003, SAGE peggiora).
- **La val-loss inverte la classifica del retrieval** (§6.3): InfoNCE allontana i rilevanti che
  finiscono nello stesso batch come negativi. Si passa a una **sonda di retrieval** sul valid per
  scegliere il checkpoint; i guadagni seguono il danno previsto (GAT +0.016, il più danneggiato).
- **Ablation a pesi casuali**: un add-pool delle feature senza rete è già al ~93% del sistema
  allenato; il training compra soprattutto topologia. Il denominatore onesto è la GNN casuale.
- Augmentation riscritte con il principio «ogni augmentation dichiara un'invarianza»: flip/rot sono
  invarianze vere per i tre assi (ma su GAT toglierle aiuta); `raw_skip` garantisce composizione e
  geometria (+0.026 su GCN, −0.023 su GAT). Un secondo giro di tuning (τ 0.3, node-drop 0.2)
  **peggiora** la topologia e l'ablation OFAT lo misura (§6.3).
- **Late fusion sulla pianta intera, esplorazione** (`status.md §4, §15`): i due rami recuperano piante
  quasi disgiunte (Jaccard top-10 0.036; nel 67,8% delle query nessun risultato in comune), ma la
  diversità è sfruttabile solo sulla geometria; RRF dà +0.003. **Conclusione**: sulla pianta intera la
  fusione sposta poco (il graph domina composizione e topologia, che sono circolari a suo favore); il
  valore va cercato **sotto danno**, dove i rami falliscono in modo diverso. È la decisione che porta
  al risultato principale.

**Agosto — floor, nuovi encoder, circolarità, criterio di robustezza.**
- **Floor del ranking casuale** (§5.3): l'nDCG per asse è in gran parte saturo; la topologia è l'asse
  che discrimina. Da qui lo spazio utile come lettura obbligata.
- **Griglia sul valid** e **tre varianti sul test** (§7.6): la scelta fra varianti di testa fatta sul
  valid non si replica sul test → le varianti sono equivalenti; nessun vincitore a pianta intera.
- **Tre encoder nuovi** (TIPSv2, PE-Core, PE-Spatial): TIPSv2 è il primo salto reale sulla pianta
  intera (+0.03 di topologia), PE-Core misura bene quanto conta il whitening (+0.076 di topologia).
- **Circolarità piena** (§7.1): anche la geometria e anche il vision. Il confronto per asse fra i rami
  diventa descrittivo.
- **Pesi della geometria** (§7.2): il claim aggregato «vision > graph in geometria» non regge; regge
  quello decomposto, con un meccanismo (somma vs min/max).
- **«Migliore» = più robusto** (24 ago, decisione): la classifica cambia completamente rispetto alla
  pianta intera; fra i frozen emerge PE-Spatial.

**10-11 settembre — il graph sotto danno e la head vision.**
- Con coppie simmetriche il **graph crolla** sotto rimozione di stanze (§6.6, punto 1). Si cambia
  **una** variabile: coppie asimmetriche intero ↔ parziale → robustezza ×12-35 secondo l'encoder (AUC da 0.006-0.009 a
  0.10-0.25); GAT il più robusto *con la regola di selezione di allora* (ribaltato il 2 ott, sotto).
- Head vision: il limite era il **budget di ottimizzazione** (~12 step per epoca), non il criterio di
  scelta dell'epoca: val-loss e sonda scelgono epoche vicine; plateau dopo ~1600 epoche. Da qui il
  budget 5000 / patience 100 usato poi per la head finale. (La val-loss non predice il retrieval per il
  graph, ma per la head sì: le due affermazioni vanno tenute separate.)

**11-15 settembre — tre danni, il metro finale, la scelta del vision.**
- La verifica pre-registrata «la head si trasferisce a crop e patch?» la **falsifica** e porta alla
  luce un difetto di rendering del masking a stanze (§7.3). Da qui: danno `nowalls`, il metro a **tre
  danni** fissato prima della griglia, la griglia completa di 52 config frozen senza estrapolazioni.
- Vince **`pespatial/gem/whiten`** (R 0.5245), con un costo piccolo e dichiarato sulla pianta intera.
- **H2**: a parità di area la griglia di patch disturba più di quanto informi («artefatto», 7/8).

**15-16 settembre — selezione del graph sulla robustezza, head rifatta.**
- La sonda della pianta intera sceglieva per il graph epoche **precoci ed erratiche** (6-33): scegliere
  sulla robustezza, sulla stessa traiettoria, porta l'AUC da 0.107 a 0.457. Lezione: si sceglie il
  checkpoint sulla misura su cui si giudica. Config graph **`gat/asymrob`**.
- Marcatore «vicini persi»: a selezione equa pareggia; non adottato.
- Il rumore fra training identici del graph è misurato (~0.03-0.04).
- **Head rifatta** sul vincitore frozen con il danno corretto: migliora il danno di training (+0.147),
  perde su crop e patch → **H3 falsificata**, non adottata. Config vision **definitiva**.

**16-18 settembre — fusione, controlli, test.**
- Late fusion: α* = 0.6, +0.174 sul ramo migliore, sopra l'oracolo.
- Controllo graph + graph (+0.034) e vision + vision (+0.098): complementarità, ma il modello pesa ~56%.
- Test letto una volta: +0.1725, il valid generalizza.

**28 set - 1 ott — verifica e repliche.**
- Ricalcolo di tutti i numeri finali dai per-query (coincidono), export CSV, notebook di analisi.
- **Repliche multi-seed**: fusione 0.6401 ± 0.0031 sul test, α* = 0.6 in tutte; il risultato è stabile.
- Ricerca dei competitor (`COMPETITORS.md`): nessun lavoro fa il task intero; LayoutGKN è l'unica
  baseline di componente eseguibile.

**1-2 ottobre — SAGE vs GAT a parità di regola.**
- La scelta di GAT (10 set) era avvenuta con la regola di selezione vecchia. Riallenato SAGE con la
  stessa ricetta di `gat/asymrob`, 4 repliche: **SAGE più robusto** (+0.026 sul valid), un po' peggio
  sulla pianta intera. Deciso prima dei numeri di trattarlo come analisi aggiuntiva: la fusione resta
  con GAT. Lezione: un confronto fra architetture va rifatto quando cambia la regola di selezione.

**Esclusi da questa storia** (bug o impostazioni concettualmente sbagliate; restano documentati in
`status.md`, `current_state.md` e `debugging-playbook.md`):
- metriche iniziali sature (score fuso + soglia, valutate sui top-k) — restano in §6.1 **solo** come
  motivazione del protocollo;
- selezione delle configurazioni vision sul test a luglio (rilievo A1) e numeri «migliori per asse»
  massimizzati sul test;
- whitening stimato su tutta la gallery (poi train-only; differenza misurata trascurabile);
- self incluso nelle metriche per-asse sotto danno (rilievo B3);
- confronti con gallery diverse fra i rami prima della gallery condivisa;
- confusione fra due file del floor costante (seed 0 vs media di 5 seed);
- il **render storico coi muri interni rimasti** e tutti i numeri della head misurati con esso (head
  «+0.284», `dinov3/natural/head` prima in classifica, «il vision è robusto al masking»). ⚠️ Fa
  eccezione la sua **scoperta** (§7.3, figura F3): è dichiarata fra i contributi; se il gruppo
  preferisce trattarla come semplice bug, togliere §7.3, il contributo 5 «artefatto di rendering» e il
  pannello destro di F3;
- l'esito negativo del marcatore «vicini persi» (−0.134) e il «graph vende la topologia» (§30.8):
  erano effetti della selezione precoce, superati dalla selezione sulla robustezza;
- head selezionate con un tetto di 60 epoche (sotto-allenate);
- incidenti operativi (GPU non supportata, argomenti sbagliati, cartelle sovrascritte, varianti non
  lanciate, wandb senza chiave);
- pooling `gem` con clamp: **non** escluso dal paper, è un caveat attivo (§5.5).

---

## 11. Struttura, figure e tabelle

### 11.1 Struttura di riferimento (da `papers/examples/pippo_paper.pdf`, CVPR 2025)

| § | Sezione | Contenuto nostro |
|---|---|---|
| 1 | Introduction (chiude con «we make the following contributions:») | §1-§2 |
| 2 | Related Work | §3 |
| 3 | Method — base · estensione · analisi · **nuova metrica** in una sottosezione | §5 (§5.2-§5.4 = la metrica) |
| 4 | Experiments — Data · Setup & Metrics · Results · Ablations | §4, §6 |
| 5 | Conclusion | §9 |
| — | References · Appendix | §7.2, §6.3, §6.4a, storia |

Lezioni da imitare: **Fig. 1 = teaser del risultato** (non l'architettura); **una sola** figura di
pipeline; **i numeri nelle tabelle**, le figure per il qualitativo e per un'analisi; **didascalie
autosufficienti** che dichiarano la conclusione. Budget: 10 pagine + 2 di reference → 5-7 figure +
4-5 tabelle nel corpo, il resto in appendice. Stile a **due colonne** (deciso).

### 11.2 Figure

Tutte in `figures/english/` (per il paper) e `figures/italian/`, PDF + PNG + `*.sources.txt`; codice in
`src/figures/` (`python -m src.figures.<nome> --lang en --out-dir figures/english`). Nessuna richiede
GPU; i numeri disegnati sono letti dagli artefatti e verificati contro i json.

- [x] **F1 · teaser** — `f_teaser_valid`: una query danneggiata (metà delle stanze tolte, valid) e i
      primi 5 risultati di vision, graph e fusione, l'originale incorniciato. Query scelta con regola
      dichiarata: la prima in ordine alfabetico fra le 423 in cui la fusione mette l'originale al 1°
      posto e nessuno dei due rami lo fa → `10340` (vision oltre il 100° · graph 3° · fusione 1°).
- [x] **F2 · pipeline** — `f_pipeline`: i due rami (immagine → encoder congelato → whitening → 768-d;
      `.mat` → grafo → GAT → 128-d), fusione α=0.6 → 896-d, gallery condivisa di 67.405 piante.
- [x] **F3 · tre danni + artefatto** — `f_damage_kinds_valid`: sinistra, `pespatial/gem/whiten` sui tre
      danni (stanze 0.392 · crop 0.661 · patch 0.521 → R 0.524); destra, l'artefatto dei muri su
      `dinov3/natural/whiten` (0.536 con i muri vs 0.304 senza, +0.336 a f=0.5).
- [ ] **F4 · la loss non predice il retrieval** (scatter) — dati incompleti (11 serie su 15); il claim
      vale **fra varianti del graph**, non per la head. Decisione degli utenti: farla dichiarando il
      buco o lasciarla fuori.
- [x] **F5 · classi di equivalenza** — `f_classes_valid`: composizione mediana 5007 vs topologia 19;
      singleton 6 vs 311 su 2000.
- [x] **F6 · robustezza per encoder** — `f_ablation_valid`: boxplot di R per encoder (52 frozen + 3
      head), la config scelta (0.5245) e radio (0.5161) quasi coincidono.
- [x] **F7 · curva del danno sul test** — `f_damage_test`: self-recovery vs frazione di stanze tolte,
      quattro sistemi (`hist`, vision, graph, fusione), bande CI 95%; f=0 marcato «tetto dei dati». **È
      il risultato principale**: candidata per il corpo.
- [x] **F8 · scala dei guadagni** — `f_alpha_valid`: AUC vs α per le tre coppie fuse (+0.174 · +0.098 ·
      +0.034) e la linea dell'oracolo.
- [x] **F9 · curva del danno sul test con le repliche** — `f_damage_test_seeds`: come F7 ma media ± 1 sd
      fra le 4 repliche (fusione 0.6401 ± 0.0031, graph 0.4697 ± 0.0035, vision 0.3916 ± 0.0045).
      Alternativa a F7 nel corpo.

Proposta per il corpo: F1, F2, F7 o F9, F8, F3; in appendice F5, F6.

### 11.3 Tabelle (numeri tutti esistenti; da impaginare)

- [ ] **T1 · dataset e setup**: split, gallery condivisa (67.405, 48 escluse), 2000 query, esclusioni
      singleton (6 / 311), floor (§4, §5.1, §5.3).
- [ ] **T2 · vision**: le prime config col metro R + tre AUC + costo full (§6.4b); in appendice il
      benchmark a pianta intera (§6.4a, protocollo A dichiarato).
- [ ] **T3 · rami + fusione + oracolo**, valid e test, sotto danno e a pianta intera (§6.8-§6.9), con
      le repliche (§6.10). CSV: `results/csv/{test,valid}_summary.csv`, `test_oracle.csv`,
      `test_paired_deltas.csv`.
- [ ] **T4 · ablation**: graph (pesi casuali, OFAT, coppie asimmetriche, selezione, marcatore,
      SAGE vs GAT a parità di regola: §6.3, §6.6) · vision (whitening, pooling, head: §6.4-§6.5) · pesi della geometria (§7.2, appendice).
- [ ] **T5 · tipi di danno**: R + tre AUC per le prime config e la head (§6.4b, §6.5);
      `results/csv/test_vision_damage_kinds.csv` per il test.

⚠️ Nel paper niente barre su scala troncata, niente «score medio» che mescola metriche e assi, sempre
intervalli di confidenza (lezioni dai notebook, `current_state.md`).

---

## 12. Riferimenti (da completare con citazioni esatte)

- LayoutGKN — van Engelenburg, van Gemert, Khademi, BMVC 2025, arXiv 2509.03737.
- LayoutGMN — Patil, Li, Fisher, Savva, Zhang, CVPR 2021, arXiv 2012.06547.
- URE-Net — Zhang, Wang, Li, Li, Wu, arXiv 2501.11097.
- SSIG — van Engelenburg, Khademi, van Gemert, ICCVW 2023, arXiv 2309.04357.
- Sketch Less for More — Bhunia et al., CVPR 2020, arXiv 2002.10310.
- PlanCraft / SketchPlan — Zeng et al., arXiv 2607.23491.
- Graph2Plan — Hu et al., SIGGRAPH 2020, arXiv 2004.13204.
- CrossOver — Sarkar et al., CVPR 2025, arXiv 2502.15011.
- MMFE — Anadón, Pautrat, Wang, ECCVW 2026, arXiv 2609.12723.
- DANIEL/ROBIN — Sharma et al., ICDAR 2017. `[VERIFY]`
- Space Syntax — Hillier & Hanson 1984; Lee, Ostwald & Gu 2018. `[VERIFY]`
- HouseGAN++, LayoutDM (contesto generativo, opzionali). `[VERIFY]`
- Encoder: DINOv2, DINOv3, SigLIP2, C-RADIOv2, I-JEPA, TIPSv2, Perception Encoder (Core, Spatial);
  GNN: GCN, GraphSAGE (`papers/GraphSAGE.pdf`), GATv2; FAISS; InfoNCE. `[VERIFY]`
- Dataset: RPLAN (Wu et al. 2019). `[VERIFY]`

---

### Log di aggiornamento
- 2026-06-27 → 2026-09-18: versioni precedenti (indice con rimandi; ultima copia salvata fuori dal repo
  prima della riscrittura).
- 2026-10-01: **riscrittura completa** per renderlo autosufficiente per la stesura del paper CVPR.
  Aggiunti: encoder nuovi e loro checkpoint, dettagli del ramo graph, floor e spazio utile sul test,
  repliche multi-seed (§6.10, F9), competitor (§3), storia del progetto (§10), tabella delle previsioni
  smentite (§7.8), claim vietati (§7.9), caveat sul pooling `gem`. Corretti: autori di SSIG; §9 diceva
  il test con α fisso «da fare» (fatto); tabelle di giugno sul test sostituite dai numeri del valid;
  vecchia §6.6 (robustezza col render difettoso) rimossa, la sostanza è in §7.3 e §10; stato di
  T1-T5 e figure allineato. Nessun numero nuovo: tutti da `status.md` e `figures/*/sources.txt`.
- 2026-10-02: verifica **SAGE vs GAT a parità di regola di selezione** (`status.md §58-§58.1`): SAGE più
  robusto (+0.026 valid, 4 repliche), deviazione sul controllo MRR f=0 dichiarata; fusione invariata con
  GAT (analisi aggiuntiva). Aggiornati §5.6, §6.6 (punto 7), §7.5, §7.8, §7.9, §8, §9, §10, §11.3.
