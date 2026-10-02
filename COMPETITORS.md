# COMPETITORS.md — Baseline e related works

Lavori della letteratura rilevanti per il nostro progetto: retrieval di piante RPLAN con query anche
incomplete, ramo vision + ramo graph e late fusion. I PDF sono in `papers/comp/`.

## Premessa: nessun competitor per il sistema intero

Da quanto trovato (ricerca web del 1 ott 2026, non esaustiva) **nessun lavoro fa il nostro task
intero**: retrieval di piante con query incomplete, con un ramo vision e un ramo graph fusi. I lavori
qui sotto toccano **una parte** del sistema. Per questo li dividiamo in due ruoli, che nel paper vanno
in sezioni diverse:

| Ruolo | Cosa significa | Dove va nel paper |
|---|---|---|
| **Baseline di componente** | Fa lo **stesso sotto-problema sugli stessi dati** (data una pianta RPLAN, ordinare le altre). Si può **eseguire nel nostro protocollo** e mettere in tabella accanto al componente corrispondente | **Experiments**, come riga di confronto. Si cita anche nei related works |
| **Related work** | Affronta un problema vicino o una sua parte: altro task, altro dominio o altri dati. **Si cita e si discute**, non si esegue | **Related works** |

Il confronto quindi si fa **per componente** (per esempio GAT vs LayoutGKN, vision+GAT vs
vision+LayoutGKN). Il resto del valore si mostra con le nostre ablation e con il **vuoto** in
letteratura descritto nella §4.

**Livello di lettura**: `[LETTO]` paper letto per intero · `[SEZIONE]` lette nel PDF le sezioni
citate · `[ABSTRACT]` solo abstract o pagina web · `[PARZIALE]` noto solo da citazioni in altri paper.
I numeri riportati vengono dai **protocolli degli autori** e **non** sono confrontabili con i nostri.

---

## 1. Mappa

| Lavoro | Venue | Ruolo | Componente o aspetto del nostro sistema | Lettura | PDF (`papers/comp/`) |
|---|---|---|---|---|---|
| **LayoutGKN** | BMVC 2025 | **Baseline di componente** | ramo graph (e quindi fusione) | `[LETTO]` | `LayoutGKN.pdf` |
| **LayoutGMN** | CVPR 2021 | Baseline possibile (costosa) · related work | ramo graph | `[PARZIALE]` | `LayoutGMN.pdf` |
| **URE-Net** | arXiv 2025 | Related work | retrieval RPLAN da immagine (ramo vision) | `[SEZIONE]` | `URE-Net.pdf` |
| **SSIG** | ICCVW 2023 | Related work | definizione di rilevanza (ground truth) | `[ABSTRACT]` | `SSIG.pdf` |
| **Sketch Less for More** | CVPR 2020 | Related work | metrica per le query incomplete | `[SEZIONE]` | `SketchLessForMore.pdf` |
| **PlanCraft / SketchPlan** | arXiv 2026 | Related work | piante parziali su RPLAN (danno alla query) | `[SEZIONE]` | `PlanCraft.pdf` |
| **Graph2Plan** | SIGGRAPH 2020 | Related work | retrieval su RPLAN da informazione parziale; fonte dei `.mat` | `[ABSTRACT]` | `Graph2Plan.pdf` |
| **CrossOver** | CVPR 2025 | Related work | fusione multimodale con le piante come modalità | `[SEZIONE]` | `CrossOver.pdf` |
| **MMFE** | ECCVW 2026 | Related work | ramo vision (encoder frozen + testa contrastiva su piante) | `[SEZIONE]` | `MMFE.pdf` |

---

## 2. Baseline di componente

### 2.1 LayoutGKN — `[LETTO]`

van Engelenburg, van Gemert, Khademi (TU Delft). *LayoutGKN: Graph Similarity Learning of Floor Plans*.
BMVC 2025. [arXiv 2509.03737](https://arxiv.org/pdf/2509.03737) ·
[BMVC](https://bmvc2025.bmva.org/proceedings/184/) ·
[codice](https://github.com/caspervanengelenburg/LayoutGKN).

**Perché è una baseline e non un competitor del sistema.** Fa retrieval di piante RPLAN **da grafo**,
sulla **pianta intera**. Si confronta quindi solo con il nostro **ramo graph**, e attraverso questo con
la fusione. Non ha immagini, non ha query incomplete, non fonde nulla.

#### Che cosa fa

- **Problema** (§3): similarità tra due piante, formulata come confronto tra grafi.
- **Grafo** (§3.1): nodo = stanza, arco = permeabilità.
  - Nodo: one-hot del tipo, **8 categorie** (living, bedroom, kitchen, bathroom, dining, store,
    balcony, corridor) + forma `s = [cx, cy, w, h, √a, p/4]` + **istogramma dei cammini minimi**
    `M(u)` con δ = 4.
  - Arco: `[1;0]` = **porta**, `[0;1]` = **solo adiacenza** (muro senza porta).
  - I grafi sono estratti dalle **immagini semantiche** RPLAN (Fig. 2).
- **Modello** (§4.1):
  - MLP per tipo, forma e arco → message passing a L layer (aggregazione media, update con GRU) →
    **un embedding per ogni stanza**;
  - similarità = **graph kernel GraphHopper**:
    `k_G = Σ_u Σ_v ⟨M(u), M(v)⟩ · exp(−µ‖h_u − h_v‖²)`, normalizzato con
    `√(k_G(Hi,Hi)·k_G(Hj,Hj))`;
  - le interazioni fra i nodi dei due grafi avvengono solo nel kernel, quindi gli embedding si
    precalcolano. LayoutGMN invece non lo permette.
- **Training** (§4.2):
  - triplet loss `[m + log(s_an/s_ap)]+`;
  - mining: top-50 per **MIoU**, riordinate per **sGED** = `exp(−GED/(|Gi|+|Gj|))`;
  - positivo con `0.6 < sGED < 0.9`, negativo con rapporto `0.7–0.9` rispetto al positivo.
- **Perché la sGED** (supplementare): user study con architetti. Il giudizio umano correla con la
  sGED, quasi per niente con l'IoU.
- **Dati e setup** (§5.1): RPLAN ripulito da quasi-duplicati e piante non connesse → **46K+** piante,
  split proprio **8:2**, 4-fold CV. Zero-shot su MSD (16K+). AdamW, lr 1e-4, batch 64, early stopping.
  Analisi con d = 64, L = 5.
- **Metriche**: triplet accuracy · **P@5/P@10**, calcolate riordinando per sGED i **top-50 del
  modello** (non su tutta la gallery) · tempo per 10K coppie.
- **Risultati** (Tab. 1, RPLAN):

  | Metodo | Acc. | P@5 | P@10 | t / 10K coppie | P@5 MSD | P@10 MSD |
  |---|---|---|---|---|---|---|
  | GK (kernel senza learning) | 65.63 | 0.389 | 0.439 | 1.2 s | na | na |
  | GEN | 96.24 | 0.603 | 0.665 | 0.7 s | 0.595 | 0.605 |
  | LayoutGMN* | 97.74 | 0.616 | 0.675 | 35.6 s | 0.585 | 0.596 |
  | **LayoutGKN** | 97.78 | 0.623 | 0.683 | 1.8 s | 0.674 | 0.697 |

  \* re-implementato con i loro grafi e il loro mining.

#### Confronto con il nostro ramo graph

| Aspetto | Noi | LayoutGKN | Conseguenza |
|---|---|---|---|
| Feature dei nodi | tipo (13) + cx, cy, w, h, area, aspect | tipo (8) + cx, cy, w, h, √a, p/4 | **Quasi identiche** |
| Archi | 10 relazioni spaziali (`rEdge`) | porta / solo muro | Le porte non le abbiamo: si usa `rEdge`, va dichiarato |
| Cammini minimi | assenti | richiesti dal kernel | Calcolabili dal nostro grafo. Sono cammini di adiacenza, non di percorrenza |
| Rappresentazione | un vettore per pianta → FAISS | un insieme di vettori + kernel | Niente FAISS: serve uno scoring dedicato |
| Query | anche incomplete (metrica principale) | solo piante intere | Le stanze tolte sono un terreno nuovo per loro |
| Metrica | nDCG per asse su tutta la gallery + self-recovery | P@k entro i top-50 + sGED | Si usa il nostro metro, il loro come extra |
| Dati e split | gallery 67K, split ufficiali | 46K ripuliti, split proprio | Il loro checkpoint contamina valid/test: **va riallenato** |
| Circolarità | grafo da `rType`/`rEdge`, gli stessi della GT | idem | Stesso avvertimento |

#### Come si esegue nel nostro protocollo

1. **Ramo graph alternativo**: grafi dai nostri `.mat`, la loro ricetta di training allenata sul
   nostro split train, valutazione con le nostre metriche (pianta intera e stanze tolte). Si ottiene
   **GAT vs GKN** a parità di dati, query e gallery.
2. **Nella fusione**: la nostra concatenazione equivale a `α·cos_vision + (1−α)·cos_graph`. Si
   sostituisce il coseno del graph con `s_GKN` (già in [0,1]) e si sceglie α sul valid. Si ottiene
   **vision+GAT vs vision+GKN**.
3. *Ipotesi:* confrontando stanza per stanza, GKN potrebbe reggere le stanze tolte meglio della GAT,
   che riassume la pianta in un solo vettore.
4. *Extra:* i nostri modelli valutati sul loro metro (sGED, P@5/P@10 sui top-50).

#### Costi e rischi

- **Scoring**: ~67K confronti a query. Al loro ritmo (1–2 s per 10K coppie) sono circa 10 s a query,
  cioè ore per 2000 query se fatto in modo ingenuo. *Ipotesi:* a blocchi su GPU scende molto.
  Riordinare una short-list FAISS cambierebbe il metodo.
- **Mining**: MIoU + GED su molte coppie del train. Probabilmente è il costo maggiore.
- **Categorie** 13 vs 8, e **regola del checkpoint** (la loro è l'early stopping): scelte da dichiarare.

### 2.2 LayoutGMN — `[PARZIALE]` — baseline possibile, ma costosa

Patil, Li, Fisher, Savva, Zhang. *LayoutGMN: Neural Graph Matching for Structural Layout Similarity*.
CVPR 2021. [arXiv 2012.06547](https://arxiv.org/pdf/2012.06547).

- **Graph matching network**: i nodi dei due grafi si scambiano informazione durante il message
  passing. Gli embedding dipendono dalla coppia e non si precalcolano (circa 20× più lento di
  LayoutGKN, Tab. 1 di LayoutGKN).
- Allenato e valutato con **IoU** (triplet scelti per MIoU). Grafi completamente connessi. Retrieval
  misurato con Precision@k.
- È **la baseline di riferimento del settore**: LayoutGKN la re-implementa, URE-Net usa il suo
  protocollo.
- **Ruolo per noi.** Stesso componente di LayoutGKN (il ramo graph), ma con una gallery da 67K serve un
  passaggio di rete per ogni coppia. Come baseline è realistica solo come riordino di una short-list,
  e cambierebbe il metodo. Se non si esegue, resta un **related work**. La disponibilità del codice
  non è verificata.

---

## 3. Related works

### 3.1 Retrieval e similarità di piante

#### URE-Net (Unit Region Encoding) — `[SEZIONE]` (Tab. 5)

Zhang, Wang, Li, Li, Wu. *Unit Region Encoding: A Unified and Compact Geometry-aware Representation for
Floorplan Applications*. [arXiv 2501.11097](https://arxiv.org/abs/2501.11097) (gen 2025, venue non
indicata). Manyi Li è coautrice anche di LayoutGMN.

- La pianta viene divisa in «unit region» adattate al contorno, partendo da una density map più la
  semantica. Non usa né pixel né grafi di stanze. Una sola codifica per space planning, metric
  learning e generazione.
- **Retrieval**: RPLAN con **lo stesso setting di training di LayoutGMN**. I numeri delle baseline sono
  presi dal paper di LayoutGMN. Metrica: triplet accuracy, con ground truth per IoU e per annotazioni
  utente.
- **Risultati** (Tab. 5, IoU-based / user-based):

  | Metodo | IoU-based | User-based |
  |---|---|---|
  | Graph Kernel | 92.07 | 95.60 |
  | U-Net Triplet | 93.01 | 91.00 |
  | GCN-CNN Triplet | 92.50 | 91.80 |
  | LayoutGMN | 97.54 | 97.60 |
  | URE-Net senza density | 99.3 | 98.5 |
  | **URE-Net con density** | **99.48** | **98.62** |

- **Relazione con noi.** È la cosa più vicina a un nostro **ramo vision** in letteratura: retrieval
  RPLAN da una rappresentazione immagine. Però misura la triplet accuracy (non un ranking su tutta la
  gallery), solo su piante intere. Codice non verificato.

#### SSIG — `[ABSTRACT]`

van Engelenburg, Khademi, van Gemert (TU Delft, lo stesso gruppo di LayoutGKN). *SSIG: A
Visually-Guided Graph Edit Distance for Floor Plan Similarity*. ICCVW 2023.
[arXiv 2309.04357](https://arxiv.org/pdf/2309.04357).

- **Metrica** di similarità strutturale **senza learning**, che combina IoU e GED. Segnala i
  quasi-duplicati di RPLAN.
- ⚠️ Autori verificati sulla prima pagina del PDF. `PAPER.md §3` lo attribuisce erroneamente a
  «Vidanapathirana et al.».
- **Relazione con noi.** Non è un modello ma una **definizione di rilevanza** alternativa alla nostra
  per asse (composizione, topologia, geometria). Serve a motivare la nostra scelta della ground truth.

### 3.2 Query incomplete

#### Sketch Less for More — `[SEZIONE]` (§4, metriche)

Bhunia et al. *Sketch Less for More: On-the-Fly Fine-Grained Sketch-Based Image Retrieval*. CVPR 2020
(Oral). [arXiv 2002.10310](https://arxiv.org/pdf/2002.10310) ·
[codice](https://github.com/AyanKumarBhunia/on-the-fly-FGSBIR).

- **Task**: ritrovare **la foto esatta** a partire da uno schizzo **incompleto**, cominciando il
  retrieval mentre l'utente disegna. Dataset QMUL Shoe-V2 e Chair-V2.
- **Metrica** (§4): Acc.@q, più due aree sotto la curva al variare della percentuale di schizzo:
  **m@A** (ranking percentile) e **m@B** (**1/rank**). Gli autori dichiarano che non esisteva lavoro
  precedente sull'early retrieval negli SBIR.
- **Metodo**: reinforcement learning che ottimizza il rango della foto giusta lungo tutto l'episodio di
  disegno.
- **Relazione con noi.**
  - Il nostro **AUC di self-recovery** (media di 1/rank sui livelli di danno, con la pianta originale
    come unica risposta giusta) è **lo stesso schema di m@B**. È il precedente da citare quando
    presentiamo la metrica.
  - Differenze: da loro la query **cresce** tratto per tratto ed è in un'altra modalità (schizzo vs
    foto), da noi la query **perde** stanze o pixel ed è nella stessa modalità della gallery.

#### PlanCraft / SketchPlan — `[SEZIONE]` (dataset SketchPlan)

Zeng et al. *PlanCraft: Sketch, Refine, and Furnish for Architect-Inspired Progressive 3D Residential
Scene Generation*. [arXiv 2607.23491](https://arxiv.org/pdf/2607.23491) (lug 2026).

- **SketchPlan**, costruito da **RPLAN** (circa 80K piante):
  - ogni pianta raster diventa un insieme di poligoni etichettati (stanze, muri, porte, finestre);
  - **Step-by-step Design Progressions**: genera una sequenza `V(0) ⊂ V(1) ⊂ … ⊂ V(K) = V` che
    simula il disegno progressivo. Un elemento entra solo se i muri da cui dipende sono già presenti;
    muri scelti con probabilità λ, porte e finestre con 1 − λ;
  - completezza `τ = |V(k)| / |V|`; la figura del metodo mostra i livelli **25 / 50 / 75 / 100%**.
- **Task**: completamento progressivo (diffusione) e arredo 3D. Il retrieval c'è solo per gli asset
  3D (CLIP), **non** per le piante. Nel PDF non c'è un link a codice o dati.
- **Relazione con noi.**
  - Stessa fonte (RPLAN) e livelli quasi uguali ai nostri (f = 0.25 / 0.5 / 0.75): conferma che
    lavorare con piante parziali su RPLAN è una pratica riconosciuta.
  - Il loro «parziale» toglie **muri, porte e finestre in ordine di disegno**, non stanze, crop o
    patch. È un **quarto tipo di danno**, plausibile per un architetto, che potremmo riprodurre (la
    procedura è descritta) per mostrare che la robustezza non dipende dal nostro modo di rovinare la
    query.

#### Graph2Plan — `[ABSTRACT]`

Hu et al. *Graph2Plan: Learning Floorplan Generation from Layout Graphs*. SIGGRAPH 2020.
[arXiv 2004.13204](https://arxiv.org/pdf/2004.13204) · [codice](https://github.com/HanHan55/Graph2plan).

- L'utente dà il **contorno** e dei **vincoli** (numero di stanze per tipo, posizioni, adiacenze). Il
  sistema **filtra** i layout graph di RPLAN per vincoli, li **ordina** per somiglianza del contorno,
  poi genera la pianta.
- È l'origine del formato `.mat` che usiamo (`rType`, `rEdge`, `gtBoxNew`).
- **Relazione con noi.** È retrieval su RPLAN **da informazione parziale** (contorno + vincoli), ma con
  regole e non con embedding, e senza una valutazione del retrieval. Motiva il task: un architetto
  parte da informazioni incomplete.

### 3.3 Fusione multimodale e ramo vision

#### CrossOver — `[SEZIONE]` (Tab. 2)

Sarkar et al. *CrossOver: 3D Scene Cross-Modal Alignment*. CVPR 2025 (Highlight).
[arXiv 2502.15011](https://arxiv.org/html/2502.15011v1) ·
[codice](https://github.com/GradientSpaces/CrossOver).

- Allinea **RGB, point cloud, CAD, floorplan e testo** in un unico spazio a livello di scena.
- La **floorplan** è un'immagine dall'alto del layout 3D, codificata con **DINOv2** (pesi condivisi con
  RGB).
- **Fusione**: somma pesata delle feature delle modalità, con pesi **appresi** (attenzione fra
  modalità con softmax).
- **Modalità mancanti**: le loss delle coppie assenti vengono mascherate. Il retrieval funziona anche
  con modalità mancanti.
- **Retrieval di scene cross-modale** (Tab. 2, ScanNet, top-1/5/10): immagine → floorplan
  **58.01 / 81.09 / 89.10**, point cloud → floorplan **55.77 / 78.53 / 86.54**.
- **Relazione con noi.**
  - È la fusione multimodale più vicina che coinvolge le piante. Però la loro è **appresa** e
    **cross-modale** (stessa scena in un'altra modalità), la nostra è una **late fusion** di score con
    α scelto sul valid.
  - La «modalità mancante» è un parente della query danneggiata, ma a livello di modalità intera, non
    di una parte della pianta.

#### MMFE — `[SEZIONE]` (abstract, Tab. 7)

Anadón, Pautrat, Wang. *Multimodal Floorplan Encoding: Learning Dense Modality-Invariant
Representations*. ECCV 2026 TwinWorld Workshop. [arXiv 2609.12723](https://arxiv.org/abs/2609.12723).

- **DINOv3 frozen** + testa DPT allenabile, con InfoNCE per cella e consistenza geometrica (warping
  della griglia di feature). Modalità: CAD vettoriale, raster, density map, LiDAR 2D.
- **Retrieval cross-modale** su Structured3D (valid, fuori dominio), descrittore con NetVLAD o SALAD
  allenati con una loss contrastiva.
- **Risultati** (Tab. 7, Top-1/5/10): DINOv3 + SALAD 43.72 / 64.77 / 72.06 → **MMFE + SALAD 55.06 /
  74.49 / 79.35**.
- Osservazione degli autori: le feature DINO sono dominate dall'**aspetto**, non dal layout.
- **Relazione con noi.** È l'impianto più simile al nostro ramo vision (encoder fondazionale frozen +
  testa leggera + loss contrastiva sulle piante). Però il task è cross-modale, niente RPLAN, niente
  query parziali, niente grafo.

---

## 4. Il vuoto in letteratura (materiale per i related works)

- **Query incomplete.** Nessun lavoro trovato fa retrieval di piante con query incomplete. Su RPLAN le
  piante parziali compaiono solo nella **generazione** (PlanCraft, Graph2Plan). Per la metrica esiste
  un precedente in un altro dominio: m@B di Sketch Less for More.
- **Fusione.** Nessun lavoro trovato fa late fusion fra un encoder di immagini e una GNN per il
  retrieval di piante. La fusione più vicina è **appresa** e cross-modale (CrossOver), non una somma
  pesata di score.
- **Ground truth.** Gli altri usano IoU (LayoutGMN, URE-Net) oppure GED (LayoutGKN), a volte combinate
  (SSIG). Nessuno usa una rilevanza scomposta per asse.
- **Protocollo.** Fra i lavori sulle piante di cui abbiamo letto la valutazione (LayoutGKN, URE-Net) si
  misura la triplet accuracy oppure P@k sui top-50 del modello. Nessuno dei due misura il ranking su
  tutta la gallery né confronta con un floor casuale. LayoutGMN, secondo LayoutGKN, usa P@k.

---

## 5. Scartati

| Lavoro | Perché scartato |
|---|---|
| SHRAG (3DV 2022) | PDF non ad accesso aperto (IEEE). Era un rivale del ramo graph su RPLAN |
| Uda & Ozaki (BESC 2025) | PDF non ad accesso aperto (Springer). Usava il grafo come supervisione di un encoder di immagini |
| Partial Person ReID (ICCV 2015) | Query parziale ma su persone. Il framing è già coperto da Sketch Less for More, che ha anche la metrica |
| NeuroMatch (2020) | Subgraph matching generico: idea utile per il ramo graph sotto masking, ma non è un retrieval di piante |
| FloorplanMAE / FP-MAE (2025) | Completamento di piante parziali su un dataset proprio da 8K piante. PlanCraft copre lo stesso punto su RPLAN |
| SceneGraphLoc (ECCV 2024) | Localizzazione in scene 3D. Per la fusione CrossOver è più vicino, perché usa le piante |
| Park et al. (2023) | Query da requisiti con approssimazione della GED, ma dataset e numeri non verificabili |
| GCN-CNN (ECCV 2020) | Layout di interfacce utente. Compare già come baseline in URE-Net |
| Shih & Peng, Azizi et al., Roominoes, a.SCatch, Takada et al., VINS, WAFFLE, MSD, ResPlan, Survey 3ST | Solo contesto generico |
