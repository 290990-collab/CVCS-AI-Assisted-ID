# Stato — risultati, decisioni, ipotesi smentite

Cronologia **compatta** di ciò che è stato misurato e deciso. Fonte dei numeri:
`vision_pipline.xlsx` (vision), log e `training_summary.json` dei job (graph),
`spiegazione_metriche_e_risultati_grafi.md` (versione discorsiva per il report).

> **Come si aggiorna** (economia dei token): si **aggiunge** una voce quando un
> risultato è letto o un'ipotesi è risolta — non si riscrive il file. Ogni voce:
> *data · configurazione · numeri · conclusione (confermata/smentita) · perché*.
> Quando una sezione supera ~40 righe, si comprimono le voci vecchie in una riga
> di sintesi, tenendo per intero solo l'ultima misura di ogni cosa. Lo storico
> integrale pre-29 lug 2026 è in `.claude/archive/CLAUDE-2026-07-29-full.md`.

**Setup di riferimento** (comune a tutti i numeri qui sotto, salvo diversa
indicazione): gallery = intero `snapshot_train/`; query = **2000 campionate dal
test split**; metriche per-asse contro l'intera gallery; nDCG@10 riportato come
`composizione / topologia / geometria`.

---

## 1. Ramo vision

**Benchmark completo (lug 2026)** — 5 encoder × pooling (natural/gem/mean) ×
trasformazione (raw/whiten/head/head+whiten) = 28 combinazioni per contributo,
full e partial.

- **Full retrieval: la head NON aiuta.** Frozen `whiten` (larghezza nativa) ≥
  head e head+whiten (256d) su ogni asse e modello, per ~0.01 di nDCG. Atteso:
  la head comprime e il suo obiettivo è il masking, non il full.
- **Partial self-recovery: la head è un guadagno grande e crescente col
  masking** — che è esattamente il suo scopo. dinov3/natural, MRR: f=0.5
  0.535 → 0.835; f=0.75 **0.205 (frozen-whiten) → 0.668 (head)**, ~3×.
  dinov2/natural: f=0.5 0.594 → 0.742; f=0.75 0.258 → 0.490. Con raw puro
  siglip2/radio/ijepa collassano già a f=0.25; il solo whitening li recupera in
  gran parte.
- **Head pura vs head+whiten dipende dal modello**: per dinov3/siglip2 la head
  pura è pari o migliore già a masking medio; per radio/ijepa il whiten-dopo-head
  aiuta; **a f=0.75 la head pura vince quasi ovunque** → default head+whiten,
  head pura per incompletezza estrema.
- **Ranking partial con head (f=0.5, natural):** dinov3 > dinov2 > radio >
  siglip2 > ijepa. La strategia `topology` (togli foglie) è quasi satura per
  tutti; i casi duri sono `random` pesante e `semantic`.
- **Migliori numeri full per asse** (fra tutte le combinazioni): **C 0.830 ·
  T 0.662 · G 0.948** (geometria: ijepa/gem/whiten; topologia: dinov3/natural/
  whiten). Media migliore: ijepa/natural/whiten 0.8077.
  ⚠️ **NON citabile (24 ago, A1 residuo 3)**: è un *oracolo per-asse* — tre
  configurazioni diverse, ciascuna massimizzata **sul test** su ~60 config
  (viola il vincolo DURO 1). I numeri citabili sono quelli del **valid**:
  geom `ijepa/gem/whiten` **0.9473**, comp/topo `tipsv2/gem448/whiten`
  **0.8433 / 0.6896** (§18).
- ⚠️ **Tetto a f=0.0: MRR 0.970 identico per tutti gli encoder** → limite dei
  **dati**, non dei modelli: a masking nullo la query è il PNG originale intatto
  e RPLAN contiene duplicati esatti (3% dei file) e quasi-duplicati.
- Primo giro (giu 2026, n=200): geometria > composizione > topologia per
  **tutti** gli encoder; cluster di testa DINOv2 ≈ DINOv3 ≈ I-JEPA entro il
  rumore, SigLIP2 sotto, RADIO ultimo → *più recente/più grande non vince*:
  domina il **domain gap** (colormap sintetiche vs immagini naturali). 16 GB di
  VRAM bastano per tutti.

**Da analizzare ancora:** confronto sistematico pooling gem/mean; metriche
per-asse-vs-pianta-completa sotto masking.

---

## 2. Ramo graph — la scoperta che ha riscritto il metodo

**(a) Primi giri, selezione su val-loss (17-18 lug).** 50 epoche: GCN
0.931/0.781/0.939, SAGE 0.912/0.759/0.942, GAT 0.892/0.726/0.942 vs baseline
`hist` 1.000\*/0.643/0.894 (\*oracolo per costruzione). Nessun early stopping →
`epochs` alzato a 150-200. **Allenare 3× più a lungo non ha dato nulla**: GCN
+0.003, GAT invariato, **SAGE peggiorato** (0.912→0.904, 0.759→0.746).

**(b) Diagnosi (27 lug).** La classifica per val-loss è l'**inverso** di quella
per retrieval: GAT ha la loss migliore (2.08) e l'nDCG peggiore; GCN la loss
peggiore (2.73) e l'nDCG migliore. **Causa:** InfoNCE è *instance
discrimination* — in un batch da 256 molte piante condividono composizione e
topologia con la query (classi enormi: solo 0,4% di singleton sulla
composizione, mediana 3980 piante) e la loss le tratta da **negativi**,
allontanando attivamente proprio ciò che la valutazione considera rilevante.
Ottimizzare meglio la loss può quindi **peggiorare** il retrieval.

**(c) Fix — `RetrievalProbe`.** Mini-retrieval sul **valid** (gallery fissa
campionata, self escluso) che gira ogni `probe_every` epoche e guida
best-checkpoint ed early stopping al posto della val-loss (che resta come sola
diagnostica). Protocollo **identico** alla valutazione finale: gli helper sono
condivisi (`axis_metrics.py`), così la metrica che *seleziona* e quella che
*giudica* sono lo stesso codice. Costo 2,6 s/epoca (~20%).

**(d) Conferma empirica (job 74096).** **GAT raggiunge il massimo di retrieval
all'epoca 3** (mean nDCG@10 sonda 0.8446) e poi **peggiora in modo monotono per
20 epoche** (0.8284) **mentre la val-loss continua a migliorare** (2.952 →
2.235). Per asse: composizione 0.890→0.858 e topologia 0.714→0.695 **calano**,
geometria 0.927→0.932 **sale** — esattamente la previsione, perché
composizione/topologia hanno grandi classi di equivalenza e la geometria no.
SAGE stesso profilo, più mite (picco a epoch 41).

**(e) Risultati con selezione su nDCG (27 lug, 74121 + 74184).** Picchi: GCN
epoch 12, SAGE 40, GAT 3; lo stop cade sempre a picco+20 e **non è prematuro**
(0/20 epoche successive superano il picco). Test: **GCN 0.934/0.789/0.938
(media 0.8870)**, SAGE 0.916/0.766/0.943 (0.8750), GAT 0.915/0.761/0.938
(0.8713). Il guadagno segue il danno previsto: GCN +0.001 (era su un plateau),
SAGE +0.010, **GAT +0.016** (il più danneggiato). Effetto maggiore sulle
metriche di **ordine**: GAT composizione mAP@10 0.401→0.525 (+31% rel.),
Recall@1 0.643→0.722. ⚠️ Trade-off reale: la **geometria peggiora** un filo
(GAT −0.008) — è l'asse continuo senza classi di equivalenza, quindi non soffre
di falsi negativi e continuava a migliorare allenando.

**(f) Ablation pesi casuali (28 lug) — quasi tutto viene dall'ARCHITETTURA.**
Misurato con la sonda (valid, gallery 5000, 500 query, nDCG@10):
**add-pool delle feature grezze, senza alcuna rete = 0.884/0.625/0.906 (media
0.8050)**; GNN a **pesi casuali** 0.799-0.809; GNN allenate GCN 0.8615, SAGE
0.8425, GAT 0.8411. Cioè **un sistema mai allenato è già al ~93% di quello
allenato**, e tutto il training vale **+0.056**. Causa: con l'one-hot dei tipi,
l'`add`-pool **è** l'istogramma delle stanze → la composizione arriva gratis e
la geometria quasi. Scomposto per asse, il training compra **essenzialmente la
topologia**: GCN da casuale ad allenato fa C +0.056, **T +0.100**, G +0.012.
→ Le poche epoche **non sono un errore**: su grafi 4-8 nodi con 2 layer,
imparare a propagare a 1-2 hop richiede poche centinaia di passi.
→ Il **denominatore onesto** non è solo la baseline `hist`, è la GNN a pesi
casuali: va riportato nel report.
→ ⚠️ **Riletta il 24 ago**: questa ablation conteneva già l'impronta empirica di
A5, mai interpretata → **§21.5**.

**(g) Fix 1+2 (28 lug).** *(1)* Augmentation riscritte sul principio "*ogni
augmentation dichiara un'invarianza, da confrontare con la ground truth*":
flip/rot sono invarianze **vere** (i tre assi sono esattamente invarianti) →
guadagno puro; `node_drop` è **falsa** sul full ma vera sul partial → si dosa
sapendolo. `node_drop` 0.1→0.2 (grafi effettivamente perturbati 51%→75%),
`feat_mask` reso **per-cella** invece che per-colonna-globale (che spostava lo
spazio uniformemente invece di creare difficoltà). *(2)* **`raw_skip`**:
add-pool delle feature grezze concatenato prima della proiezione → composizione
e geometria garantite per costruzione, capacità della rete libera per la
topologia. **Test (74714 vs 74184):** media **GCN 0.8870→0.9130 (+0.026)**,
SAGE +0.005, **GAT −0.023**. La composizione sale per tutti (GCN 0.934→0.971,
mAP@10 0.632→0.847, Recall@1 0.793→0.922) ma sulla topologia i tre divergono:
GCN +0.039, SAGE invariata, **GAT −0.087**. ⚠️ **Meccanismo:** con `raw_skip` la
composizione diventa alta e gratis dalla prima epoca, quindi il criterio `mean`
è dominato dalla componente regalata e il picco arriva **prestissimo** (GAT e
SAGE a epoch 5, GCN a 35) — cioè *prima* che la topologia si sviluppi. Skip +
criterio `mean` si combinano male → giustifica `select_criterion: topology` e
`probe_every: 1`.

---

## 3. Ablation OFAT (28 lug, job 74839 / 74840 / 74856)

9 varianti su 10 per encoder (`selgeom` mai lanciata: lista desincronizzata, poi
sanata con una guardia). ⚠️ **Si leggono sulla colonna topologia della sonda**,
non su `best_score`: `selmean` è selezionata su un criterio diverso, quindi il
suo punteggio è su un'altra scala.

| Encoder | Migliore | Topologia (sonda) | vs `base` |
|---|---|---:|---:|
| gcn | **`tau02`** (`nd01` pari) | 0.800 | +0.030 |
| sage | **`nd01`** | 0.747 | +0.031 |
| gat | **`nosym`** | 0.703 | +0.012 (al limite del rumore) |

σ della sonda ≈ 0.002-0.004.

**Verifica sul TEST delle vincitrici (job 74885-74887, 74931-74933):**
gcn `base` 0.965/0.788/0.939 → **`tau02` 0.972/0.826/0.940**; sage `base`
0.935/0.726/0.933 → `nd01` 0.940/0.759/0.934; gat `base` 0.927/0.697/0.938 →
`nosym` 0.907/0.743/0.946.

- ✅ **La sonda predice il test**: Δ topologia sonda→test = gcn +0.030→+0.038,
  sage +0.031→+0.033, gat +0.012→+0.046 — segno sempre giusto, magnitudo quasi
  identica su gcn/sage. Il protocollo "ablation sul valid, test solo per le
  finaliste" è **validato**; non serve ingrandire `probe_gallery`.
- **Classifica test (media nDCG@10):** `gcn/tau02` **0.913** > `gcn/base` 0.897 >
  `sage/nd01` 0.878 > `sage/base` 0.865 ≈ `gat/nosym` 0.865 > `gat/base` 0.854 >
  baseline `hist` 0.846. Ordine encoder invariato **GCN > SAGE > GAT**.
- Valore vero rispetto alla baseline: la **topologia** — nDCG 0.826 vs 0.643 ma
  soprattutto **Recall@1 0.558 vs 0.027 (20×)**.
- ⚠️ `gat/nosym` ha la **geometria più alta di tutto il ramo graph (0.946)**,
  contro 0.948 del miglior encoder vision: sull'unico asse **senza circolarità**
  i due rami sono ormai pari.
- **Bilancio onesto della giornata del 28 lug su gcn ≈ zero** (topologia 0.826 vs
  0.828 del mattino): il guadagno vero erano i fix 1+2; il secondo giro di
  tuning è stato una regressione che l'ablation ha individuato e annullato.
- **Da fare:** riportare `temperature: 0.2` in `gcn.yaml`/`graph_sage.yaml` (gat
  resta 0.3 con flip/rot a 0) e provare la **combinazione** τ0.2 + `node_drop
  0.1`, che l'OFAT per costruzione non copre.

---

## 4. Confronto appaiato vision vs graph e attese sulla fusione (28 lug)

Stesse 2000 query, stessa gallery, stesse metriche — verificato: entrambi i log
riportano le identiche esclusioni singleton (10 composizione, 313 topologia).

**nDCG@10 — vision best per asse C 0.830 / T 0.662 / G 0.948** vs **graph GCN
0.934 / 0.789 / 0.938** (SAGE geometria 0.943, gat/nosym 0.946).

1. **Sulla geometria vince il vision (0.948 vs 0.946)** ed è **l'unico asse dove
   il confronto è alla pari**: è il dato più onesto della tabella.
   ⚠️ **SMENTITO il 24 ago (§21.1)**: anche la GT geometrica è funzione esatta
   dell'input del grafo. Nessun asse è alla pari. Il numero resta, la lettura no.
2. Sulla topologia il miglior encoder visivo (0.662) supera di pochissimo la
   baseline `hist` (0.643), che le adiacenze **non le guarda affatto** →
   leggere la connettività dai pixel è genuinamente difficile (muro pieno vs
   passaggio = pochi pixel).
3. Il grafo rappresenta le stanze come **bounding box**: una stanza a L e una
   rettangolare con lo stesso box sono per lui identiche. Muri, aperture e forma
   del contorno sono informazione che **solo il vision** ha — e che **nessuno
   dei tre assi attuali misura**.

⚠️ **Aspettativa realistica per la fusione.** Prendendo il meglio per asse fra
tutti i sistemi (C 1.000 `hist`, T 0.789 GCN, G 0.948 vision) si arriva a media
**0.912 contro 0.887 del solo GCN**: **+0.025, quasi tutto dalla composizione
della baseline**; il contributo specifico del vision vale **+0.005÷0.010**. Sulle
metriche attuali la fusione **sposta poco**, e questo va detto nel report invece
di promettere un salto di nDCG. Il valore va cercato nel **partial** (i rami
falliscono in modo diverso: il vision degrada coi pixel rimasti, il grafo è
nativamente robusto al node-drop) e nell'applicabilità reale.

**Implementazione consigliata:** *concatenazione pesata* `[√α·v ; √(1−α)·g]` su
embedding L2-norm → il prodotto scalare **è già** `α·simV + (1−α)·simG`, quindi
basta **un solo indice FAISS e una sola ricerca**. Prima normalizzare la scala
degli score per ramo; confrontare con **RRF** (fusione per rango, scale-free) come
baseline. ⚠️ Prima di costruire: misurare la complementarità dagli embedding già
salvati (sovrapposizione top-10, correlazione nDCG per-query, upper bound
oracolo), con **inner join sui nomi** (67.405 vs 67.453).

**Due regimi da dichiarare:** (A) su RPLAN coi `.mat` la fusione è **parzialmente
auto-avverante** su C/T; (B) col grafo estratto dall'immagine la circolarità
sparisce e la complementarità è piena — fuori scope Fase 1, ma è la direzione in
cui la fusione ha senso pieno.

---

## 5. Ipotesi smentite dalla misura (da riportare così come sono)

| Ipotesi | Esito | Perché |
|---|---|---|
| "La val-loss InfoNCE indica la qualità del retrieval" | ❌ **Invertita** (graph) | InfoNCE allontana i rilevanti finiti nello stesso batch. ⚠️ 10 set: nel training della head vision val-loss e probe partial scelgono epoche vicine (§30.3) |
| "Allenare più a lungo migliora" | ❌ **sul graph** | 50→150 epoche: +0.003 su GCN (rumore), SAGE **peggiora**. ⚠️ 10 set: **non** si estende alla head vision, che era sotto-allenata (§30.3-§30.5) |
| "τ più alto (0.3-0.5) smorza i falsi negativi e aiuta" | ❌ | τ 0.2→0.3 costa **0.038 di topologia** su gcn; τ=0.5 è la peggiore su tutti e tre. La stima dell'8,17% misurava la composizione, ma la topologia ha classi mediane di 16 → ammorbidire la pressione distrugge il segnale che serve |
| "Encoder più recenti/grandi vincono" | ❌ | Cluster di testa entro il rumore, RADIO ultimo: domina il domain gap |
| "La projection head migliora anche il full retrieval" | ❌ | Sul full frozen+whiten ≥ head; la head serve al **partial** (dove vale fino a 3×). ⚠️ misurato con head a 60 epoche: il costo sul full della head B.6 va rimisurato (§30.4) |
| "Il tetto di MRR 0.970 è un limite dei modelli" | ❌ | È un limite dei **dati** (duplicati esatti nel 3% dei PNG) |
| "Il guadagno delle ablation svanisce sul test" | ❌ (era un artefatto) | Il riferimento era una cartella **sovrascritta**, non la configurazione ipotizzata |

Confermate invece: la sonda di retrieval come criterio di selezione; `raw_skip`
sulla composizione; le simmetrie flip/rot su gcn/sage (ma **non** su gat).

---

## 6. Incidenti operativi

- **GPU Blackwell sm_120** non supportata dal PyTorch dell'env: job vx07
  65653-65657 morti con `no kernel image` al forward della head; rilancio
  riuscito su GPU compatibile (67424, 67448, 67451-67453). Da allora tutti gli
  script sensibili portano la constraint con allowlist.
- **Job 74848 finito in 0 s**: passato `sage` invece di `graph_sage` (gli script
  vogliono il basename del YAML) → encoder saltato con avviso, successo
  apparente.
- **Job 74096 scartato**: partito mentre il codice veniva modificato → giro misto
  GCN-vecchio / GAT-SAGE-nuovo. Rifatto da 74121.
- **`selgeom` mai lanciata**: `ALL_VARIANTS` e la tabella dei flag erano
  desincronizzate → 9 varianti su 10, con successo apparente. Sanata con una
  guardia che ora fallisce.
- **31 lug — le 4 run del floor uccise dal login node**: `ulimit -t` = **600 s
  di CPU** su `ailb-login-02`; `random_floor` ne consuma ~180 s per seed
  (misurati 196.0 / 178.0 / 161.2 s) e muore durante il 4°, con un `Killed`
  muto. Non è memoria (110 GB liberi). Regola: **ogni job CPU su scala dataset
  va su nodo di calcolo**, anche se non usa la GPU.

---

## 7. Prossimi passi

1. **Late fusion vision+graph** — miglior rapporto valore/sforzo: gli
   `embeddings.npy` + `names.json` allineati dei 4 sistemi sono **già salvati**.
   Prima misurare la complementarità, poi implementare.
2. **Modalità partial sul ramo graph** (gemella del vision), dove la robustezza
   nativa al node-drop dovrebbe pagare.
3. **Cambiare l'obiettivo, non allenare di più**: testa di proiezione stile
   SimCLR (loss su una proiezione usa-e-getta, retrieval sulla rappresentazione a
   monte) — attacca i falsi negativi alla fonte. Poi **GIN** (bias induttivo
   giusto per la topologia), dropout su GAT, scheduler lr, batch più grande.
4. **Combinazione τ0.2 + `node_drop` 0.1** (l'OFAT non la copre).
5. **`GraphModelManager`** (config-loader vero: a quel punto i ponti YAML→flag
   negli script diventano superflui).
6. **Decidere la questione circolarità** (oracle dichiarato / solo geometria /
   grafo estratto dall'immagine) — vedi `.claude/shared/architecture.md`.
7. **Report per i prof**: metriche introdotte, studio del livello di masking,
   tabella comparativa multi-encoder, confronto fra i rami coi caveat.

---

## 8. Audit metodologico della codebase (30 lug) — nessuna run, solo lettura

Diagnosi completa in **`current_state.md`** (root). Qui solo ciò che chiude o
apre un'ipotesi; i dettagli non si duplicano.

**Chiuso in positivo (verificato, non più da sospettare):**
- Le **2000 query sono davvero le stesse** nei due rami: verificato per
  costruzione (stesso `sorted()`, stesso predicato di filtro, stesso
  `Random(42).sample`) *e* sui due json (`names.json` ⊂ `image_paths.json`,
  solo-vision = 48, solo-graph = 0). Il claim di §4 regge.
- **IDCG e self coerenti** in entrambi i rami e in entrambe le modalità: nessuna
  deflazione sistematica dell'nDCG.
- **Esclusioni singleton identiche per costruzione** (le 48 piante extra del
  vision hanno `valid=False`): il 10/313 dei log non era una coincidenza.
- Nessun leakage nel training della head (coppie da `train`+`valid`).

**Aperto o contraddetto (da decidere prima del report):**
- ⚠️ **«La geometria è l'unico asse alla pari» non è supportata dal codice**: la
  GT geometrica è funzione di `rType`/`gtBoxNew`, che il grafo riceve in input.
  Cambia la § circolarità di `CLAUDE.md` e l'argomento sulla fusione.
- ⚠️ **Selezione sul test** nel ramo vision (nessuno script valuta su `valid`):
  i massimi per-asse, **0.948 compreso**, sono max su ~60 config lette sul test.
- ⚠️ **Whitening vision fittato su tutta la gallery** (contiene valid+test),
  mentre il graph normalizza col solo train: protocolli diversi fra i rami.
- ⚠️ **La head vision è ancora selezionata sulla val-loss InfoNCE**, il criterio
  smentito in §2: la conclusione «sul full la head non aiuta» non è difendibile
  finché non si riallena con selezione su probe.
- ⚠️ **Nessuna stima di incertezza e per-query mai salvati**: ogni Δ ≤ 0.005 —
  incluso 0.948 vs 0.946 — è indistinguibile dal rumore finché non c'è un test
  appaiato. Il **floor** (ranking casuale) non è mai stato misurato.

## 9. Fase A.1 — riconferma dei rilievi, uno per uno (30 lug, nessuna GPU)

Metodo: grep + letture a range + **esecuzione** per i punti verificabili solo
così. Esito: **11 confermati, 1 ridimensionato, 1 non verificato**.

**Confermati** (evidenza letta o eseguita in sessione):
- **A1** nessuno script vision valuta su `valid`; `eval.split=test` in
  `scripts/vision/{04,05,06,07}_eval_*.sh:33-34`. Il consumatore gestisce `valid`
  correttamente: manca l'invocazione, non il meccanismo.
- **A2** whitening su tutta la gallery, **nessun flag** train-only esiste.
- **A3** `train_projection.py:108-109` monitora solo `vl < best_val`; grep di
  `probe|ndcg|monitor` in codice e config → **zero hit**: nessun criterio
  alternativo dietro un flag.
- **A4** unico hit di `std` in `src/` è `graph/transforms.py:142` (z-score sulle
  feature, **non** incertezza); nessun per-query persistito.
- **A6** `sim = (area_sim + aspect_sim + dist_sim) / 3.0`, pesi hardcoded,
  `GalleryAxes.__init__` senza parametro di pesi → **nessun punto di iniezione**.
- **A7** `grep -c assert tests/*.py` → **0, 0, 0** (115 righe totali).
- **B1** su disco oggi: `temperature: 0.3`, `node_drop: 0.2`,
  `flip_prob/rot_prob: 0.5` — nessuno dei tre è la variante vincente misurata.
- **B2** 67.405 nomi graph, 67.453 vision, |vision∖graph| = **48**,
  |graph∖vision| = **0**, e l'ordine della lista vision filtrata coincide
  esattamente con quello graph (inner join pulito, non un disallineamento).
- **B3** partial `exclude_self=False` anche sulle metriche per-asse, full
  `exclude_self=True`.
- **B4** riprodotto: YAML con `amp: true`/`foo_flag: false` in più → output del
  ponte **byte-identico**, exit code 0 in entrambi i casi.
- **B5** riprodotto: `proj.weight` **(128,147)** con `raw_skip=True`,
  **(128,128)** senza; in eval `load_state_dict` a nudo, errore generico.

**Ridimensionato — A5 (parziale).** `type_area_distribution` **è** ricostruibile
dalle node feature (somma aree per tipo / totale). Ma `footprint_area` e
`footprint_aspect` derivano da `rec.gtBox`, che **non** è nelle feature dei nodi:
il grafo ha i box delle singole stanze, non il bbox globale. Quindi la GT
geometrica non è interamente calcolabile dall'input del grafo. ⚠️ Resta da
verificare se `gtBox` coincide con l'unione dei `gtBoxNew`: se sì, A5 torna
pieno. La formulazione «di grado, non di natura» va comunque corretta.

**Non verificato:** **B6** (clamp degli id di relazione e `reduce="mean"` su
`edge_attr`) richiede di leggere `graphs.pt`; e la domanda «il colore delle
stanze nelle PNG codifica `rType`?» richiede una PNG + il `.mat` corrispondente.
Entrambi sono artefatti pesanti: servono l'ok dell'utente.

### 9.1 — A1 ri-verificato il 13 ago: **risolto in pratica, non nel codice**

⚠️ **Il criterio della tabella non discrimina più.** «`grep -rn 'eval.split'
scripts/` → esiste almeno uno script con `valid`?» oggi passa
(`scripts/evaluation/02_perquery_vision_valid.sh:96`), ma quello script è **nato
per chiudere A1** (lo dichiara alle righe 5-8): il rivelatore trova la propria
toppa. La riconferma va fatta sulla sostanza.

**Risolto** — la selezione vision **non** avviene più sul test: 80 `.npz` in
`results/perquery/vision_valid/`, la vincente è scelta lì (§13, §18) e il test è
letto dopo (§14). Meccanismo consumatore intatto:
`src/vision/evaluation/evaluate.py:90-98` e `:468`. Il graph non era toccato da
A1 (probe sul valid) e ora ha anche i per-query del valid (§16).

**Residui aperti (4):**
1. Le quattro griglie storiche hardcodano ancora `eval.split=test` e **non hanno
   passthrough** di override: `scripts/vision/{04,05,06,07}_eval_*.sh:40-42`.
   Chi le rilancia torna a selezionare sul test senza accorgersene.
2. Il test è stato letto **3 volte, non una** (§14.1) — la condizione di chiusura
   di A1 («leggere il test una volta sola») è violata; mitigata solo dalla
   pre-registrazione del primario, che va dichiarata nel report.
3. **La metà "oracolo per-asse" di A1 è ancora viva nei documenti**: §1:43-45 e
   `CLAUDE.md § Stato attuale` citano **C 0.830 · T 0.662 · G 0.948**, tre
   configurazioni diverse ciascuna massimizzata **sul test** (luglio). Sul valid
   lo stesso `ijepa/gem/whiten` fa **0.9473**: è quello il numero citabile.
4. `configs/graph_retrieval.yaml:20` ha `split: test` come default e
   `configs/vision_retrieval.yaml:55` ha `split: null` (= gallery intera): il
   default di entrambi i rami non è `valid`.

Nessun residuo richiede una run: 1 e 4 sono edit di script/config, 2 e 3 sono
scrittura del report.

## 10. Il floor misurato (A.2) — 31 lug, `results/random_floor/`

2000 query del test, 5 seed di ranking, `exclude_self=True`. **nDCG@10:**

| null | gallery | comp | topo | geom |
|---|---|---|---|---|
| casuale | vision (67.453) | 0.7388 | 0.4658 | 0.8689 |
| costante | vision | 0.7690 | 0.4960 | 0.8729 |
| casuale | graph (67.405) | 0.7399 | 0.4664 | 0.8696 |
| costante | graph | 0.7372 | 0.4716 | 0.8700 |

**Spazio utile** `(score − floor)/(1 − floor)`, floor casuale della propria
gallery: geometria **vision 0.948 → 60%**, graph 0.940 → 54%, `hist` 0.894 →
**19%**; topologia graph 0.826 → **67%**, `hist` 0.643 → 33%; composizione graph
0.972 → 89%.

**Conclusioni.**
1. **L'nDCG per-asse è la metrica meno informativa che il progetto abbia**:
   il floor casuale è 0.74 (comp) e 0.87 (geom). Recall/mAP dello stesso ranking
   casuale stanno a 0.085 e 0.035: lì lo spazio utile è quasi intero.
2. **La geometria ha solo 0.131 di spazio**: il margine 0.002 fra i due rami è
   **1.5%** di quell'intervallo. Il claim resta indecidibile senza test appaiato.
3. **La topologia è l'asse che discrimina** (floor 0.466, il più basso) ed è
   quello dove il training compra di più: 67% contro il 33% di `hist`.
4. **Il null costante non inganna la metrica**: nessun vantaggio sistematico sul
   casuale (graph: 0.7372 vs 0.7399 su comp). ⚠️ Ma la sua `mc_std` è 10-100×
   più grande (0.025 vs 0.0007), perché è un solo sorteggio per seed: alcune
   scelte fisse vanno molto meglio di altre.
5. **Le due gallery danno lo stesso floor** (Δ ≤ 0.0011): le 48 piante in più
   non spostano il pavimento — ma quel Δ è dello stesso ordine del margine 0.002.
6. ⚠️ **`recall_at_k` normalizza per `min(k, R)`** (`metrics.py:73`): quando la
   classe di rilevanti ha ≥ k elementi **è Precision@k**, non Recall. Il nome
   nelle tabelle è fuorviante e va dichiarato nel report. A k=1 coincide con mAP
   per costruzione (`:96` stessa normalizzazione): riportarle entrambe a k=1 non
   aggiunge nulla.

## 11. Griglia vision sul VALID (B.1) — 31 lug, job 76919, analisi PARZIALE

> ⚠️ **SUPERATA dalla §13** (8 ago): la griglia è **completa, 56/56, dinov3
> incluso** — i file a disco sono del 6 ago, quindi il job è stato rilanciato con
> successo. Restano validi gli effetti misurati (confermati su 14 coppie invece
> di 5); **cambia il vincitore** e cade la premessa «dinov3 manca».

19 run su 56 a disco (`results/perquery/vision_valid/`), job ancora in corso.
**dinov2** completo (3 pooling × 4), **siglip2** quasi; **dinov3 fallito in
blocco** (`GatedRepoError` 401, repo gated + nessun token HF) → il job chiuderà a
44/56 e il miglior encoder del progetto **manca dalla selezione**.

**Appaiamento verificato**: tutte le run hanno gallery `22d91dcff041` (67.453),
`split=valid`, `query_seed=42`, 2000 query, e **la stessa firma di skip**
(comp 6, topo 311, geom 0 — geometria è continua: `num_relevant=-1`, quindi
Recall/mAP non esistono su quell'asse). Confronto appaiato lecito senza riserve.

**Tre effetti misurati** (nDCG@10, delta appaiati, CI 95% bootstrap, Wilcoxon):

1. **Il whitening compra topologia, su tutti i modelli**: +0.033/+0.040/+0.035
   (dinov2 gem/mean/natural), +0.015/+0.024 (siglip2), CI sempre lontano da 0.
   Su composizione l'effetto **cambia segno per encoder**: +0.005…+0.009 su
   dinov2, **−0.004…−0.005 su siglip2** (p<0.002). Su geometria è ~0.
2. **La head aiuta la composizione e paga in topologia/geometria.** Contro
   `raw`: comp +0.007…+0.008 (ns su gem), topo **−0.016** (dinov2/gem) o ns,
   geom −0.005…0. Unica eccezione **siglip2/natural**, dove migliora tutto.
3. **Sopra il whitening la head è un danno netto**: `head+whiten` < `whiten` in
   **tutte e 4** le coppie disponibili, su **tutti e tre** gli assi
   (topo −0.004…−0.009, geom −0.002…−0.004, p ≤ 2e-3). Coerente con A3: quella
   head è selezionata sulla val-loss InfoNCE.

**Chi vince (sul valid, parziale).** Per asse: comp `siglip2/natural/head`
0.8353 · topo `dinov2/natural/whiten` 0.6600 · geom `dinov2/gem/whiten` 0.9393.
In frazione di spazio utile (floor **del test**, proxy: 0.7388/0.4658/0.8689) →
**37% / 36% / 54%**: la geometria resta l'asse dove il vision copre di più.
Media dei tre assi: `dinov2/natural/whiten` 40.0% ≈ `siglip2/gem/whiten` 39.8%.

⚠️ **Due cautele.** (a) Il floor è misurato sulle query del **test**: per
normalizzare il valid serve `01_random_floor.sh --split valid` (oggi `--split
test` è cablato alla riga 50). (b) `dinov2/natural/whiten` vs
`siglip2/gem/whiten` è **indistinguibile** su composizione e geometria (CI
include 0 su 5 confronti su 7): vince solo su nDCG topo (+0.0073), e su
Recall/mAP topo l'ordine si **inverte** (siglip2 avanti, ma ns). Senza la
decisione A.5 su cosa significa "migliore", la griglia non elegge un vincitore.

---

## 12. Audit del ramo vision (6 ago 2026) — sola lettura + misure CPU

Misure fatte in sessione sui `.mat` reali (80.729 record) e su
`embeddings/vision/*/image_paths.json` (nessun `.npy` letto, nessuna GPU).

**Struttura della gallery (verificata).** 67.453 righe = test 10.127 · train
47.126 · valid 10.152; 48 righe (0,07%) senza `.mat`; 0 stem duplicati. Le **14**
gallery vision hanno **ordine identico** → l'appaiamento dentro il ramo è
garantito. ⚠️ `snapshot_train/` copre l'84% dei `.mat` (67.453/80.729): il test
ufficiale ha 12.110 piante, in gallery ce ne sono 10.127.

**Gli assi NON sono indipendenti** (359.940 coppie, 60 query × 6.000 righe).
Pearson fra i gain: comp↔topo **0.69**, comp↔geom **0.30**, topo↔geom 0.27, e
comp↔`dist_sim` **0.53**. Smentisce `relevance.py:6` («TRE segnali
indipendenti»): "vince su 2 assi su 3" non è evidenza indipendente.

**Perché la geometria ha il floor a 0.869 — causa meccanica.** `footprint_area`
∈ [0.193, 0.684] (std 0.060) → `area_sim = 1−|Δa|` ha media **0.933**, min
0.569: un terzo del gain è quasi costante. `geometry_sim` totale: media 0.849,
std 0.060, p1–p99 = [0.707, 0.955]. Il floor non è un caso, è la definizione del
gain. Inoltre area/aspect misurano il **bounding box**, non la pianta.

**"Recall" sulla composizione è Precision@k.** Classe di equivalenza mediana
≈3.990 righe (p90 15.140): nel **98,5%** delle query num_rel ≥ 10, quindi
`min(k,num_rel)=k` e la formula degenera in |top-k ∩ rel|/k. A k=100 vale ancora
per il 90%. Topologia: 59% a k=10, e **27,3% di query singleton escluse** →
Recall/mAP topologia si misurano solo sulle topologie comuni (bias ottimistico,
ma uguale per tutti i modelli → confronto appaiato salvo).

**Due difetti negli encoder (verificati nel sorgente).**
1. `base.py:24` — `gem_pool` fa `clamp(min=1e-6)` su token post-LayerNorm:
   misurato, **50,7%** delle componenti azzerate. Il "GeM" del progetto è la
   media della sola parte positiva. ⚠️ Tocca il numero di punta del ramo:
   geometria 0.948 = `ijepa/gem/whiten`.
2. `dinov2.py:72` (ereditato da `DINOv3Encoder`) — `patches = hidden[:, 1:, :]`.
   In transformers 5.8.1 `modeling_dinov3_vit.py:90` concatena
   `[CLS, register, patch]` e usa `num_prefix = 1 + num_register_tokens`
   (riga 596): per DINOv3 i register entrano in `mean`/`gem`. Non tocca
   `dinov3/natural` (topo 0.662). Resta da leggere `num_register_tokens` del
   checkpoint (gated, non letto).

**Whitening fittato sulla gallery intera** (`retrieval_model.py:137`), quindi
anche su test e valid → contraddice il vincolo «statistiche dal solo train».
Effetto atteso piccolo (1 query su 67k) ma il claim va corretto o il fit
ristretto alle 47.126 righe train. Con `dim: null` + `eps=1e-6` l'amplificazione
delle direzioni a varianza nulla arriva a **1000×**: lo spettro reale non è
stato letto (serve l'ok su un `.npy`).

### 12.1 — M1/M2/M4a-b applicati (6 ago). Numeri invariati.

Correzioni di **lettura**, non di calcolo: nessuna media cambia, cambia cosa
c'è scritto sopra la colonna. Verificato sui dati sintetici che le medie
restino identiche (`_accumulate_axes` non toccato).

- **nuovo** `src/evaluation/metric_diagnostics.py` — post-hoc dai `.npz`
  `perquery/1`: regime precision (`#rel >= k`), copertura della media
  (`num_relevant == 0`), correlazione fra gli assi. Nessuna GPU.
- `metrics.py` — docstring: il `min(k,·)` fa degenerare `recall_at_k` in
  Precision@k. Nome della funzione e chiave `"recall"` del contratto
  **invariati** di proposito (li usa il ramo graph e i 51 file già scritti).
- `relevance.py` — «TRE segnali indipendenti» → «distinti», con le correlazioni
  misurate e la saturazione della geometria dichiarate nel docstring.
- `vision/evaluation/evaluate.py` — colonna `Recall` → `Rec/Prec`, **nuova
  colonna `cop.`** (copertura, derivata dalle lunghezze delle liste di accumulo:
  nessun parametro nuovo), avviso nel report partial che quelle tabelle non sono
  confrontabili col full + caso «run che non rimuove nulla».
- `tests/test_metric_diagnostics.py` — 11 test, fra cui l'invariante che conta:
  copertura calcolata **in linea** e **post-hoc** devono coincidere. 11/11 verdi,
  `test_perquery` 16/16 (nessuna regressione).

**Non verificato**: il modulo non ha ancora girato sui `.npz` reali (serve l'ok
sugli artefatti). Comando pronto:

    python -m src.evaluation.metric_diagnostics \
        --perquery results/perquery/vision_valid/*.npz --k 10
    python -m src.evaluation.metric_diagnostics \
        --gallery embeddings/vision/dinov2/natural/image_paths.json

**Decisione aperta**: `src/graph/evaluation/axis_metrics.py:100` stampa ancora
`Recall` senza copertura — stessa metrica, stesso difetto. Non toccato per la
regola «niente refactor cross-ramo»: l'allineamento (6 righe, identiche) va
deciso, altrimenti le tabelle dei due rami divergono nel report.

---

## 13. Griglia vision sul VALID — COMPLETA (56/56), analisi dell'8 ago

I `.npz` in `results/perquery/vision_valid/` sono datati **6 ago 18:26-21:52**:
il job è stato rilanciato e chiuso, **dinov3 compreso**. Supera la §11 e la voce
del TODO che dava dinov3 per fallito.

**Appaiamento: una sola firma su 56 run** — gallery `22d91dcff041` (67.453),
`split=valid`, seed 42, 2000 query identiche per nome, skip (comp 6, topo 311).
Confronto appaiato lecito senza riserve.

**Floor usato**: il null **`constant`** (0.7942/0.5144/0.8703), non il `random`:
sulla composizione il constant è **+0.055** sopra il random, quindi è lui il
riferimento onesto. ⚠️ È misurato sul **test** (proxy per il valid, come da §11).

> ⚠️ **Rettifica 13 ago (§18.4)**: quei tre numeri sono il **solo ranking-seed 0**
> (oggi `constant_vision_test_seed0.npz`, `run_tag=constant_floor/seed0`), non la media sui
> 5 seed, che è **0.7690/0.4960/0.8729** (§10 e il `.txt`). Sul null costante il
> seed sposta il floor di ±0.025 sulla composizione, quindi **tutte le
> percentuali di spazio utile qui sotto sono da leggere come una realizzazione,
> non come una stima**: comp 19.9% → **28.7%** con la media dei 5 seed, **36.9%**
> col floor `random`. L'ordine delle run non cambia (il denominatore è comune),
> cambia solo la scala. Convenzione **decisa il 24 ago (§20)**: denominatore = floor `random`,
> quindi **36.9%**.

**Non esiste un vincitore: 14 run su 56 sono sulla frontiera di Pareto.** La
frontiera è una curva di scambio **geometria ↔ composizione**, da
`ijepa/gem/whiten` (geom 0.9473, comp 0.8229) a `siglip2/natural/head`
(comp 0.8353, geom 0.9315). Migliori per asse: comp `siglip2/natural/head`
0.8353 (19.9% dello spazio utile) · topo `dinov3/natural/whiten` 0.6623 (30.5%)
· geom `ijepa/gem/whiten` 0.9473 (59.4%).

⚠️ **La media dello spazio utile è fuorviante**: `ijepa/gem/whiten` è primo
(33.4%) ma nei test appaiati **perde su composizione e topologia** contro tutti
e tre i suoi inseguitori, con CI che escludono 0 (vs `siglip2/mean/whiten`:
comp −0.0064, topo −0.0149, geom +0.0114; p ≤ 3e-7). Vince la media solo grazie
all'asse più saturo.

**Tre effetti ora misurati su tutte e 14 le coppie (encoder × pooling):**

1. **Il whitening compra topologia, 14/14 senza eccezioni**: +0.0305 medio
   (comp +0.0016, geom +0.0034). È l'effetto più forte e sistematico del
   benchmark: `whiten` è il contributo migliore sulla topologia in **14 casi
   su 14**.
2. **La head da sola aiuta solo la composizione** (+0.0018) e paga topologia
   (−0.0090) e geometria (−0.0015).
3. **Sopra il whitening la head è un danno netto su tutti e tre gli assi**
   (comp −0.0013, topo −0.0092, geom −0.0040), 14 coppie su 14. Conferma A3:
   quella head è selezionata sulla val-loss InfoNCE.

**Le metriche concordano in grande e divergono in testa.** Spearman fra i
ranking delle 56 run: nDCG~Recall +0.96, nDCG~mAP +0.96 (entrambi gli assi
discreti). Ma il **vincitore cambia**: su topologia nDCG dice
`dinov3/natural/whiten`, Recall e mAP dicono `siglip2/mean/whiten`; su
composizione nDCG dice `siglip2/natural/head`, Recall/mAP dicono
`siglip2/mean/head`. Sulla profondità invece il verdetto tiene
(Spearman @10~@100 = +0.93; `dinov3/natural/whiten` primo a k=1, 10 e 100).

**Diagnostica delle metriche sui dati veri** (`metric_diagnostics`, prima
esecuzione reale del modulo): composizione copertura **99.7%**, regime precision
**97.8%**, classe di equivalenza mediana **5.007** (p90 15.251) → "Recall@10"
su quell'asse è Precision@10 quasi ovunque. Topologia: copertura **84.5%**
(311 query singleton escluse), regime precision 72.5%, classe mediana 38.
⚠️ Corregge le stime della §12, che venivano da una sotto-gallery di 20.000
righe (davano 27.3% di query escluse e mediana ~3.990).

### 13.1 — La variante per il test: `siglip2/mean/whiten` (8 ago)

Confronti appaiati fra i tre candidati di testa sulla topologia (nDCG@10, CI 95%
bootstrap, Wilcoxon, Holm sui 3 assi). **A = `siglip2/mean/whiten`:**

| vs | composition | topology | geometry |
|---|---|---|---|
| `dinov3/natural/whiten` | **+0.0048** (p 5.7e-5) | −0.0025 (ns, p .135) | **+0.0016** (p 5.8e-4) |
| `dinov2/natural/whiten` | **+0.0038** (p 2.4e-3) | −0.0001 (ns, p .988) | **+0.0008** (p 4.2e-2) |

**Verdetto**: siglip2/mean/whiten vince composizione e geometria contro entrambi
i rivali di testa ed è **indistinguibile** da loro sulla topologia. Regge anche
su Recall/mAP topo (vs dinov3: −0.0005 / −0.0015, entrambi ns).

⚠️ **Smentita una conclusione intermedia dell'8 ago**: «il candidato è
`dinov3/natural/whiten` perché primo sulla topologia». Quel primato è
**+0.0024 su nDCG, non significativo** (p .135–.22). dinov3 batte davvero
*dinov2* su Recall/mAP topo (+0.0097 / +0.0078, p ≤ 1.3e-3) ma **non** siglip2.
Ordinare per la stima puntuale di un asse senza il test appaiato porta alla
variante sbagliata: è il caso da citare nel report.

**Due proprietà che la rendono la scelta difendibile**: (a) non usa la head →
immune al rilievo A3 (head selezionata sulla val-loss InfoNCE) e non va rifatta
se la head si riallena; (b) usa `whiten`, il contributo migliore sulla topologia
in 14/14 combinazioni.

⚠️ Se invece A.5 decide «massimo su un asse», la risposta cambia:
`ijepa/gem/whiten` per la geometria (0.9473, +0.0114 su siglip2, p 4.7e-124).

**Comando (una volta sola, il test non sceglie nulla):**

    sbatch scripts/evaluation/03_perquery_vision_test.sh siglip2 mean whiten

Artefatti presenti: `embeddings/vision/siglip2/mean/{embeddings.npy,
image_paths.json}`. Output atteso:
`results/perquery/vision_test/vision_siglip2_mean_whiten_full_test.npz`.

---

## 14. TEST del ramo vision — `siglip2/mean/whiten` (8 ago, una volta sola)

`results/perquery/vision_test/vision_siglip2_mean_whiten_full_test.npz`, scritto
l'8 ago 12:17. Integrità verificata: `split=test`, `exclude_self=True`, seed 42,
2000 query, gallery `22d91dcff041` (67.453), schema `perquery/1`. **Un solo file
nella cartella**: il test non ha visto altre varianti.

**nDCG@10 = 0.8259 / 0.6537 / 0.9356** (comp / topo / geom).
Recall@10 0.2875 / 0.2148 · mAP@10 0.2126 / 0.1705.

**Generalizza**: rispetto al valid della stessa variante (query diverse, non
appaiato) comp −0.0034, topo −0.0061, geom −0.0004. La selezione sul valid non
ha prodotto overfitting.

**Contro i null, appaiato** (le query di test coincidono con quelle dei due
floor — verificato per nome). Contro il null più forte (`constant`):
comp +0.0317, topo **+0.1393**, geom +0.0653, tutti con p ≤ 6e-63. In frazione
di spazio utile: **15.4% / 28.7% / 50.3%** (valid: 17.0/29.9/50.6 → coerente).

⚠️ **L'nDCG nasconde il risultato vero.** Sulle metriche d'ordine il margine sul
null `constant` è di un altro ordine di grandezza: composizione Recall@10 2.0×,
mAP@10 2.3×; **topologia Recall@10 54.9×, mAP@10 92.5×** (0.1705 contro 0.0018).
Da riportare così nel report: il +0.14 di nDCG topologia e il ×92 di mAP sono lo
stesso fatto, ma solo il secondo si legge.

**Contro il ramo graph** (⚠️ gallery diverse, 67.453 vs 67.405, `--allow-gallery-
mismatch`: confronto imperfetto fino a B.3):

| B | composition | topology | geometry |
|---|---|---|---|
| `gcn/tau02` | −0.1463 | −0.1721 | **−0.0049** (p 2e-18) |
| `hist` (training-free) | −0.1739 (oracolo) | **+0.0109** (p 4e-3) | **+0.0419** |

Il vision batte la baseline training-free del graph su topologia e geometria, e
perde contro la GNN allenata su tutti e tre gli assi. Su composizione e topologia
il confronto è **viziato per costruzione** (la GT deriva da `rType`/`rEdge`, che
sono l'input del graph): l'unico asse quasi alla pari è la geometria, e lì il
graph vince di 0.0049.

⚠️ **Conseguenza della decisione A.5, da scrivere nel report.** La variante
scelta è la più *robusta*, non la più forte sulla geometria: `ijepa/gem/whiten`
faceva 0.948 sul test (§1, benchmark di luglio), cioè **sopra** lo 0.9405 del
graph. Scegliere la robustezza è costato al vision l'unico asse dove poteva
battere il graph. **Questo NON autorizza a cambiare variante adesso**: sceglierla
guardando il test è esattamente il rilievo A1. Se si vuole il confronto per-query
anche su ijepa, va dichiarato come analisi secondaria e riportato accanto, non al
posto, di questo numero.

### 14.1 — Due varianti in più sul test (9 ago): la selezione NON si replica

Lanciate `dinov2/natural/whiten` e `dinov3/natural/whiten` sul test (file del
9 ago 15:25). **File integri**: 2000 query, `n_ret` = 100 ovunque, 0 query
degenerate, gallery `22d91dcff041`, `split=test`, seed 42, `argv` coerenti.
Nessun errore tecnico. (La rapidità è attesa: l'eval ricarica `embeddings.npy`
dalla cache e non ri-estrae.) dinov3 gira: il `GatedRepoError` è chiuso.

**nDCG@10 sul test**: dinov2 0.8262/0.6590/0.9350 · dinov3 0.8243/**0.6618**/
0.9342 · siglip2 0.8259/0.6537/**0.9356**.

**Il confronto si ribalta fra valid e test** (A = siglip2, delta appaiati, Holm):

| vs | asse | VALID | TEST |
|---|---|---|---|
| dinov2 | comp | **+0.0038** (p 2.4e-3) | −0.0003 (**ns**) |
| dinov2 | topo | −0.0001 (ns) | **−0.0053** (p 4.4e-3) |
| dinov3 | comp | **+0.0048** (p 5.7e-5) | +0.0015 (**ns**) |
| dinov3 | topo | −0.0025 (ns) | **−0.0081** (p 1.9e-5) |
| dinov3 | geom | **+0.0016** (p 5.8e-4) | +0.0014 (p 1.1e-2) |

Il vantaggio di siglip2 sulla **composizione sparisce** e il pareggio sulla
**topologia diventa sconfitta significativa**. Regge solo la geometria vs dinov3.
`dinov3` vs `dinov2` sul test: **tutto ns dopo Holm**.

**Causa: drift valid→test asimmetrico.** siglip2 comp −0.0034 / topo −0.0061;
dinov2 +0.0007/−0.0010; dinov3 −0.0002/−0.0005. Il margine con cui siglip2 era
stato scelto (≈0.004) è **dello stesso ordine del suo drift fra split**: era in
buona parte rumore dello split di validazione, non un'proprietà del modello.

**Conclusione (smentisce §13.1 nella parte narrativa, non nel numero):** le tre
varianti di testa sono **equivalenti entro il rumore**; nessuna è "la migliore".
Il numero primario resta `siglip2/mean/whiten` **perché pre-registrato in §13.1
prima di vedere il test**, non perché vinca.

⚠️ **Debito metodologico**: ora sul test ci sono 3 varianti. Non invalida il
primario (scelta documentata e datata prima), ma **il report deve dichiararle
tutte e tre** e dire che il primario era pre-registrato. Riportarne una sola
sarebbe cherry-picking. Da qui in avanti ogni ulteriore variante sul test
peggiora il bilancio: fermarsi a queste tre.

---

## 15. Complementarità vision↔graph e baseline D.0 (9 ago, dai `ret_rows`)

Analisi **post-hoc dai file per-query**, nessun job lanciato. ⚠️ Calcolata sui
file di **test**: è **esplorativa**, non un risultato pre-registrato. La scelta
della coppia va rifatta sul valid (per il graph i per-query del valid **non
esistono**: serve `04_perquery_graph.sh` con `eval.split=valid`).

**I due rami recuperano piante quasi disgiunte.** Top-10 per query, join per
nome: **Jaccard 0.036**, in media **0.60/10** piante in comune, e nel **67.8%**
delle query **zero** risultati condivisi. La diversità c'è, ed è enorme.

**Ma la diversità è utile su UN solo asse.** Oracolo per-query (scegliere per
ogni query il ranking migliore dei due) su `siglip2/mean/whiten` + `gcn/tau02`:

| asse | vision | graph | oracolo | guadagno | query in cui vision > graph |
|---|---|---|---|---|---|
| composition | 0.8259 | 0.9722 | 0.9728 | +0.0006 | **2.0%** |
| topology | 0.6537 | 0.8258 | 0.8288 | +0.0030 | **6.0%** |
| geometry | 0.9356 | 0.9405 | 0.9485 | **+0.0080** | **40.5%** |

⚠️ **La premessa «il vision è forte dove il graph è debole» non regge**: il graph
non è debole da nessuna parte. Domina composizione e topologia — dove è un upper
bound **per costruzione**, essendo la GT derivata da `rType`/`rEdge` che sono il
suo input — ed è alla pari sulla geometria. **L'unico asse con complementarità
sfruttabile è la geometria.**

**La fusione funziona (baseline D.0).** RRF (k=60) sui nomi, ranking fusi e
rivalutati sulla gallery vision, asse geometria, 2000 query:

    vision 0.9356 · graph 0.9405 · **RRF 0.9437** · oracolo 0.9485

**+0.0032 sopra il miglior singolo**, batte entrambi nel **35.8%** delle query, e
resta **−0.0048 sotto l'oracolo** → una fusione pesata/supervisionata ha ancora
margine. È il primo numero reale di late fusion del progetto.

**Conseguenza per la scelta della coppia**: il vision da fondere **non** è il
migliore in composizione (inutile: il graph fa 0.9722 e `hist` 0.9998) ma il
migliore in **geometria** → `ijepa/gem/whiten` (0.9473 sul valid), non
`siglip2/mean/whiten` (0.9356). Lato graph: `gcn/tau02` per il sistema completo,
ma il miglior graph in geometria è **`gat_nosym` (0.9464)**, non gcn.

---

## 16. VALID per-query del ramo graph (10 ago) — 7 run, gallery intera

`results/perquery/graph_test/*_full_valid.npz`, scritti il 10 ago 10:35-10:48.
Colma il buco dichiarato in §15 (per il graph i per-query del valid non
esistevano). **Appaiamento verificato**, non assunto: gallery `0c24cfc05e18`
(67.405) e `names` **identici** su tutti e 7 i file, `split=valid`, `mode=full`,
`exclude_self=True`, seed 42, 2000 query, `geometry_weights` uguali, e soprattutto
le query saltate **coincidono riga per riga** (6 sulla composizione, 311 sulla
topologia: dipendono dalla GT, non dal modello).

**nDCG@10 e frazione di spazio utile** (⚠️ floor = ranking casuale misurato sul
**test** §10, usato come proxy: il floor sul valid non è mai stato misurato):

| run | comp | util | topo | util | geom | util |
|---|---:|---:|---:|---:|---:|---:|
| `hist` (training-free) | 0.9999 | 100%\* | 0.6428 | 33% | 0.8927 | 18% |
| `gat/base` | 0.9281 | 72% | 0.6997 | 44% | 0.9381 | 53% |
| `gat/nosym` | 0.9085 | 65% | 0.7441 | 52% | **0.9453** | **58%** |
| `sage/base` | 0.9372 | 76% | 0.7304 | 50% | 0.9336 | 49% |
| `sage/nd01` | 0.9421 | 78% | 0.7622 | 55% | 0.9342 | 50% |
| `gcn/base` | 0.9677 | 88% | 0.7940 | 61% | 0.9391 | 53% |
| **`gcn/tau02`** | **0.9741** | **90%** | **0.8298** | **68%** | 0.9408 | 55% |

\*oracolo per costruzione.

**Le tre ablation OFAT sono confermate con test appaiato** (Wilcoxon + bootstrap
10k, Holm sui 3 assi). Tutte e tre comprano topologia, con delta simili:
`tau02−base` **+0.0358** [+0.0335,+0.0382], `nosym−base` **+0.0444**
[+0.0403,+0.0485], `nd01−base` **+0.0318** [+0.0294,+0.0344]; p Holm ≤ 8e-82.
⚠️ Le tre varianti sono **ritorni allo stato precedente** (τ 0.2, `node_drop`
0.1, flip/rot spente): le tre modifiche del 28 lug restano quindi smentite anche
a gallery intera. Sulla sonda i delta erano +0.030/+0.012/+0.031 → **la sonda
sottostimava gat** (+0.012 vs +0.044 reale), gli altri due sono quasi identici.

**`nosym` è l'unico trade-off vero**: paga −0.0196 di composizione (−7.5% dello
spazio) per +0.0444 di topologia e +0.0073 di geometria — e resta il **miglior
geometria del ramo**, battendo `gcn/tau02` di 0.0045 (p 6.7e-24). Conferma la
scelta della coppia per D.0 indicata in §15, ora **sul valid** come richiesto.

**`gcn/tau02` domina tutto il resto**: vs `sage/nd01` +0.0320/+0.0676/+0.0066, vs
`gat/nosym` +0.0656/+0.0857/−0.0045, tutti p ≤ 2e-24. **Nessuna ambiguità di
scelta sul ramo graph** (a differenza del vision, §13/§14.1): l'unico asse che
perde è la geometria, dove il divario vale 3.5% dello spazio utile.

**Contro `hist`**: comp −0.0259 (l'oracolo), topo **+0.1870** (+35% dello
spazio), geom **+0.0481** (+37%). Sulle metriche d'ordine il divario topologico è
di un altro ordine: **mAP@10 0.3508 vs 0.0148 (24×)**, Recall@10 0.4117 vs
0.0464 (8.9×). ⚠️ `significance` marca questi due come non affidabili perché
coprono 1689/2000 query (84.5% < 90%): l'avviso riguarda la **copertura**, non
l'appaiamento — le 311 query escluse sono le stesse in tutti i file (singleton
topologici). Il claim vale per l'84,5% delle query, e le escluse sono
plausibilmente le più difficili.

**Bias di selezione stimato.** Confronto cella per cella col test già pubblicato
(§3, job 74885-74887/74931-74933; il test **non** è stato riaperto): il valid è
sopra di **+0.001…+0.006** su comp/topo e di ±0.001 sulla geometria, su tutte e 6
le run. È l'entità dell'overfitting da selezione con la probe: piccola, di segno
costante, e da dichiarare nel report.

**Aperto:** il floor sul **valid** non esiste (le % di spazio utile sopra sono
approssimate); e la geometria separa i modelli di appena 0.0117 fra il peggiore e
il migliore (9% dello spazio) → su quell'asse la classifica interna al ramo è
quasi priva di potere discriminante.

---

## 17. Tre encoder vision nuovi (11-13 ago) — implementati, **zero numeri**

Estensione del ramo vision da 5 a **8 encoder registrati**. Nessuno dei tre ha
ancora un embedding su disco: qui c'è solo ciò che è stato **misurato in smoke
test CPU** e le decisioni prese. Finché non gira `01_extract_raw.sh`, ogni
confronto con i 5 storici è vuoto.

| Nome | Checkpoint | D | Provenienza | Risoluzione |
|---|---|---|---|---|
| `tipsv2` | `google/tipsv2-b14` | 768 | `transformers` + `trust_remote_code` | nativa **448**, 224 legale |
| `pecore` | `vit_pe_core_base_patch16_224.fb` | 768 (1024 su `natural`) | **timm** | **224 bloccata** |
| `pespatial` | `vit_pe_spatial_base_patch16_512.fb` | 768 | **timm** | nativa 512, default **224** |

Tutti e tre apache-2.0 e **non gated** (a differenza di DINOv3).

**Decisioni.**
1. **TIPSv2 gira a due risoluzioni** (448 nativa + 224 come gli altri): la
   risoluzione entra nel nome della variante (`tipsv2/natural448`), altrimenti le
   due estrazioni si sovrascrivono. Meccanismo generale in `_common.sh`
   (`resolutions_for`), i 5 encoder storici restano a variante = pooling e i loro
   comandi sono **invariati** (verificato: 98/98 identici).
2. **PE-Core e PE-Spatial si caricano da timm**, non da `transformers`: i repo
   `facebook/PE-*` pubblicano un `.pt` grezzo utilizzabile solo col pacchetto
   `perception_models` di Meta.
3. **PE-Spatial: scelta la B16-512, non la G14-448.** Misurato: la gigantic ha
   **0 prefix token** → nessun CLS, `natural` collasserebbe su `mean` e la
   griglia perderebbe una colonna. La B16 è anche la **coppia controllata** di
   `pecore` (stesso ViT-B/16, stessa norm. 0.5×3): l'unica variabile che cambia è
   l'obiettivo di training, contrastivo vs denso.

**Due claim del pretrained_cfg smentiti dalla misura.**
- PE-Spatial `fixed_input_size=True` **non** blocca la risoluzione: timm
  ricampiona il `pos_embed` (1025 → 197 token da 512 a 224) e il CLS a 224
  correla 0.956 con quello a 512 → default 224, allineato alla griglia.
- PE-Spatial `model(images)` **non** è il pooling naturale: con
  `global_pool="avg"` è la media dei patch (cos 1.0000 con la media esplicita).
  Usarlo per `natural` avrebbe reso `natural ≡ mean` **senza errori**, cioè una
  colonna della griglia sprecata in silenzio.

**Effetto sulla griglia (attenzione).** `select_models` in `_common.sh` elenca
ora **8** nomi (`pespatial` incluso dal 13 ago, §19): le feature RAW passano da
14 a **26** e `02_perquery_vision_valid.sh` lanciato **senza argomenti** da 56 a
**104 run**, che **riscrivono** i .npz già prodotti (§13). Va sempre lanciato
con l'encoder esplicito.

**Aperto:** l'ablation 448 vs 224 di TIPSv2 e il layer intermedio di PE-Core
(il paper sostiene che le feature migliori non sono all'ultimo layer) si
decidono **sul valid**, non sul test. → 448 vs 224 **chiuso** in §18.

---

## 18. VALID per-query di `tipsv2` e `pecore` (13 ago) — 18 run nuove

`results/perquery/vision_valid/`, file del 13 ago 10:34-11:24. **tipsv2**: 12 run
(3 pooling × {224, 448} × {raw, whiten}); **pecore**: 6 (3 pooling × {raw,
whiten}); **`pespatial`: zero, non ancora estratto**. Nessuna variante `head`:
quelle esistono solo per i 5 encoder storici. La cartella passa a **74 run**.

**Appaiamento verificato**: gallery `22d91dcf` (67.453), `split=valid`, seed 42,
2000 query — **firma identica alle 56 della §13**, quindi i confronti con lo
storico sono appaiati senza riserve e i vecchi `.npz` non sono stati riscritti.

### 18.1 TIPSv2 è il primo salto reale della griglia vision

Migliore assoluto **`tipsv2/gem448/whiten` = 0.8433 / 0.6896 / 0.9347**. Batte
tutti sui due assi non saturi; la geometria resta a `ijepa/gem/whiten` (0.9473).

| A = `tipsv2/gem448/whiten` vs | comp | topo | geom |
|---|---|---|---|
| `siglip2/mean/whiten` (variante del test) | **+0.0140** | **+0.0297** | −0.0013 |
| `ijepa/gem/whiten` (miglior geometria) | **+0.0204** | **+0.0446** | −0.0126 |
| `tipsv2/gem224/whiten` (risoluzione) | **+0.0095** | **+0.0171** | −0.0013 |

Tutti con CI 95% che esclude 0 e p Holm ≤ 2e-5. Il +0.0297 di topologia è
**dieci volte** il margine che in §13.1 separava i tre candidati di testa
(+0.0024, non significativo): è il primo delta vision che non è rumore.
Massimi per asse aggiornati: comp `tipsv2/mean448/raw` **0.8509** (era
`siglip2/natural/head` 0.8353) · topo `tipsv2/gem448/whiten` **0.6896** (era
`dinov3/natural/whiten` 0.6623) · geom `ijepa/gem/whiten` 0.9473 **invariato**.

**Tre effetti misurati dentro tipsv2.**
1. **448 > 224 sistematico** su comp (+0.0065…+0.0138) e topo
   (+0.0115…+0.0200) in **6 coppie su 6**, geometria piatta (±0.004). La
   risoluzione nativa del checkpoint vale più di qualunque scelta di pooling
   → l'ablation della §17 è chiusa: **si usa 448**.
2. **Il whitening qui è uno scambio, non un regalo**: +0.0193 topo ma −0.0069
   comp e −0.0046 geom. Diverge dal pattern §13 (whiten migliore su tutto in
   14/14): il massimo di composizione è una run **`raw`**.
3. **`gem` > `mean` > `natural`(CLS)** a 448+whiten su tutti e tre gli assi
   (+0.0021…+0.0049, CI esclude 0). Piccolo ma coerente.

### 18.2 PE-Core sta sotto la media, ma misura il whitening

Migliore `pecore/mean/whiten` = 0.8264 / 0.6517 / 0.9342: **perde su tutti e tre
gli assi** contro `siglip2/mean/whiten` (−0.0029 / −0.0082 / −0.0018, p Holm
≤ 2.1e-5). Le sue tre run `raw` sono le **peggiori delle 74** (`mean/raw`
0.8058 / 0.5758 / 0.9108).

Da qui l'unico risultato interessante: su `pecore` il whitening vale
**+0.0205 / +0.0759 / +0.0234** — l'effetto più grande del benchmark (media §13
sulle 14 coppie: +0.0305 sulla topologia). Interpretazione, **non verificata**:
lo spazio contrastivo CLIP-like di PE-Core è fortemente anisotropo e senza
decorrelazione poche direzioni dominano il coseno.

### 18.3 Frontiera di Pareto e ricadute

**12 run non dominate su 74** (erano 14 su 56): **4 sono tipsv2**
(`gem448/whiten`, `gem224/whiten`, `gem448/raw`, `mean448/raw`), le altre 8 sono
`ijepa` (6) e `dinov3` (2). **siglip2 e dinov2 escono dalla frontiera**; pecore
non ci entra mai. La curva di scambio resta la stessa della §13 — geometria ↔
(composizione+topologia) — ma con un estremo nuovo e più alto.

Spazio utile sopra il floor **costante** (0.7690/0.4960/0.8729, §10, misurato sul
test come proxy): `tipsv2/gem448/whiten` = 32.2% / **38.4%** / 48.6% contro il
33.0% di topologia del miglior storico.

⚠️ Sul floor citato dalla §13 vedi la rettifica in **§18.4**.

**Complementarità per D.0** (per-query, stesse 2000 query): `tipsv2/gem448/whiten`
vs `ijepa/gem/whiten` correlano **r = 0.654 sulla geometria** (la più bassa
misurata) e ijepa vince su **74.9%** delle query di quell'asse pur essendo dietro
sugli altri due. È la coppia più complementare disponibile nel ramo vision.
⚠️ L'oracolo per-query (0.9503 geom, +0.0030) è il tetto di un **selettore**
per-query, non della fusione a punteggi: non è un bound su RRF.

### 18.4 Il floor costante: due numeri veri, due grandezze diverse (13 ago)

**Causa trovata, nessun numero inventato.** `random_floor.py` produce due
artefatti che *non contengono la stessa cosa*:

- il **`.txt`** (`print_floor_table`, `:430`) stampa la **media sui 5 ranking
  seed** → costante vision @10 = **0.7690/0.4960/0.8729** (= §10);
- il **`.npz`** (`recorder.write`, `:434`) salva le per-query del **solo primo
  seed**, `run_tag=constant_floor/seed0` → **0.7942/0.5144/0.8703**, cioè
  esattamente i valori della §13. Verificato ricalcolando dal file.

La §10 ha letto il `.txt`, la §13 il `.npz`. Entrambe corrette, mai confrontate.

**Perché il difetto è invisibile sul null `random` e feroce sul `constant`**: nel
random ogni query ha un sorteggio suo, i 2000 si mediano e il seed sparisce
(seed0 0.7396 vs media 0.7388, mc_std 0.0007). Nel costante il sorteggio è **uno
solo, condiviso da tutte le query**: la mc_std sale a 0.025 sulla composizione
(già notato in §10.4) e il seed 0 è una realizzazione a **+1 mc_std** esatto.

**Effetto sul denominatore** (asse composizione, best 0.8353): floor seed0 →
20.0% di spazio utile; media 5 seed → 28.7%; media −1 mc_std → 35.7%; floor
`random` → 36.9%. **±1 sorteggio = ±8 punti percentuali.**

**Meccanismo dell'ambiguità sul disco**: `random_floor.py:433` mette il seed nel
nome di default (`..._seed0.npz`), ma `01_random_floor.sh:75` passa `--out
results/random_floor/<null>_<ramo>_test.npz` e **lo toglie** (corretto il 24 ago,
§20). Il file su disco allora non
dichiara di essere un seed solo (lo dicono solo il `run_tag` nel meta e il
commento `:23` dello script).

**Decisione PRESA il 24 ago → §20** (era A.5-adiacente e bloccava le
percentuali del report). Raccomandato allora, scelto poi: **denominatore = floor `random`** (mc_std 0.0007 → percentuali
stabili e riproducibili), e null `constant` riportato **sempre come media ±
mc_std sui 5 seed**, nel suo ruolo proprio — dire se un asse discrimina — mai
come singolo numero al denominatore. In alternativa si tiene il costante come
denominatore, ma allora va scritta accanto la banda ±1 mc_std, perché una
percentuale con ±8 punti di rumore non è riportabile da sola.
⚠️ Il seed 0 non è una scelta conservativa: alza il floor su comp/topo e lo
**abbassa** sulla geometria (0.8703 < 0.8729). Sbaglia in due direzioni.

**Chiuso il 24 ago (§20): entrambe le cose fatte insieme, come previsto qui.**
Testo originale — i floor esistono **solo sul test** e i
confronti sul valid li usano come proxy (§11, §13, §18). Se si decide di
misurarli anche sul valid, è quel momento — e solo quello — che conviene
rimettere il seed nel nome del `.npz` in `01_random_floor.sh` (una riga, ma tocca
`scripts/**` e i path già citati in `COMANDI.md`: serve mandato esplicito).

---

## 19. `pespatial` in griglia ed estratto (13 ago) — **zero numeri letti**

Chiude la parte implementativa aperta in §17. Due decisioni e due job, nessuna
metrica: i `.npz` esistono ma **non sono stati aperti** (artefatti pesanti,
serve l'ok dell'utente). Tutto ciò che segue è metadato di filesystem.

**Decisione 1 — `pespatial` entra in `select_models`** (`_common.sh:14`), a
**224 soltanto**: la variante 512 resta fuori per scelta esplicita, quindi
`resolutions_for` e `poolings_for` lo lasciano al caso di default (`native`, 3
pooling — il CLS c'è, a differenza della G14 scartata in §17). Effetto
verificato simulando i loop: feature RAW **23 → 26**, griglia valid senza
argomenti **92 → 104** run.

**Decisione 2 — primo giro solo `frozen`.** Il gruppo `head` caricherebbe un
`head.pt` che per `pespatial` non esiste: quelle 6 run fallirebbero.

**Artefatti su disco.**

| Cosa | Evidenza |
|---|---|
| RAW, 3 varianti | `embeddings/vision/pespatial/{natural,gem,mean}/`, 13 ago 11:39-11:58 |
| forma | 207.215.744 B = header 128 + **67.453 × 768 × 4** → gallery intera, 768d su tutti e tre i pooling (come atteso da §17) |
| valid per-query | 6 `.npz` `vision_pespatial_<pool>_{raw,whiten}_full_valid.npz` → la cartella passa da 74 (§18) a **80** |

**Non verificato**: i valori delle metriche, e l'appaiamento (firma gallery /
seed / n. query) che in §18 era stato controllato per `tipsv2` e `pecore`.
Vanno letti prima di qualunque confronto con la griglia storica.

**Ipotesi da falsificare quando si leggeranno** (scritta *prima*, per non
adattarla al risultato): `pespatial` è la coppia controllata di `pecore` —
stesso ViT-B/16, stessa norm., stessa risoluzione 224, cambia **solo**
l'obiettivo di training (contrastivo → denso). Se l'allineamento denso serve al
retrieval di planimetrie, `pespatial` deve battere `pecore` **sulla geometria**,
che è l'unico asse non circolare (`CLAUDE.md`, vincolo 5). Un vantaggio solo su
composizione/topologia non sosterrebbe la tesi.
⚠️ **Premessa caduta il 24 ago (§21.1)**: la geometria **non** è l'asse non
circolare — non ne esiste uno. L'ipotesi su `pespatial` resta valida come
confronto controllato fra i due encoder, ma non può più appoggiarsi a
«l'unico asse pulito».

---

## 20. Convenzione sul floor — **decisa** il 24 ago (chiude §18.4)

**Decisione dell'utente, due punti.**

**(a) Denominatore dello "spazio utile" = floor `random`, sempre.** `mc_std`
0.0007 → percentuali riproducibili. Il null `constant` **resta**, ma solo nel
ruolo per cui è nato — dire se un asse *discrimina* — e si riporta **sempre come
media ± mc_std sui 5 seed**, mai come numero singolo al denominatore.

Due motivi, il secondo più forte del primo:

1. L'argomento che aveva scelto il constant in §13 («è +0.055 sopra il random
   sulla composizione, quindi è il riferimento onesto») **usava il solo seed 0**:
   con la media dei 5 seed il gap è **+0.0302**, cioè *metà* di quel vantaggio
   era fortuna del sorteggio.
2. Il null `constant` **non è** il baro che voleva imitare. Le 10 piante fisse
   sono **sorteggiate** (`random_floor.py:168`), non scelte: è un baro *pigro*,
   e la sua `mc_std` 0.025 dice proprio che il baro vero prenderebbe un mazzo
   migliore. Non misura né il "non sapere niente" (quello è il `random`) né il
   tetto del baro — che sarebbe un altro esperimento, e sfiorerebbe l'oracolo.

**Ricalcolo** (floor `random` della propria gallery; derivato in sessione dai
numeri di §10/§13/§14/§16, nessuna run nuova). ⚠️ per il valid il floor è ancora
quello del **test**, proxy finché non gira il job (b):

| run | split | comp | topo | geom |
|---|---|---|---|---|
| `siglip2/natural/head` | valid | **36.9%** (era 19.9%) | — | — |
| `dinov3/natural/whiten` | valid | — | **36.8%** (era 30.5%) | — |
| `ijepa/gem/whiten` | valid | — | — | **59.8%** (era 59.4%) |
| `tipsv2/gem448/whiten` | valid | 40.0% | 41.9% | 50.2% |
| `siglip2/mean/whiten` | test | 33.3% | 35.2% | 50.9% |
| `graph gcn/tau02` | test | 89.2% | 67.4% | 54.0% |
| `graph hist` | test | 100.0%\* | 33.1% | 18.7% |

\* oracolo per costruzione. **Il denominatore è comune a tutte le run: nessun
confronto cambia di segno, cambia solo la scala.** §10, §11, §14, §16 e
`CLAUDE.md` usavano già il `random` → invariati; l'unica fuori linea era **§13**
(e §18, che eredita).

**(b) Floor anche sul VALID + nomi dei file disambiguati.**
`01_random_floor.sh` accetta ora `[all|vision|graph] [test|valid]` e scrive due
nomi **diversi**, perché il contenuto è diverso — era la causa meccanica di
§18.4:

- `<null>_<ramo>_<split>_seed0.npz` → per-query, **un solo** ranking-seed;
- `<null>_<ramo>_<split>_mean5.txt` → tabella, **media sui 5**.

Gli 8 file del test già su disco sono stati rinominati con lo schema nuovo
(`mv`, nessun rilancio). Nessun modulo Python legge quei path (verificato con
grep: solo `.md`), quindi la rinomina non rompe codice.

### 20.1 Il floor del VALID è arrivato (24 ago 15:06-15:14) — **A.2 chiusa**

Le 4 run CPU sono girate: 8 file nuovi in `results/random_floor/`
(`*_valid_{seed0.npz,mean5.txt}`). Verificato in sessione, non assunto.

**(a) Il proxy era legittimo: valid e test danno lo stesso floor.** nDCG@10,
media sui 5 ranking-seed, Δ = valid − test:

| null | gallery | comp | topo | geom | Δ max |
|---|---|---|---|---|---|
| random | vision | 0.7392 | 0.4649 | 0.8689 | **0.0009** |
| random | graph | 0.7406 | 0.4655 | 0.8699 | **0.0009** |
| constant | vision | 0.7697 | 0.4957 | 0.8730 | 0.0007 |
| constant | graph | 0.7376 | 0.4716 | 0.8699 | 0.0004 |

Nessuna differenza supera 2×mc_std. **Perché era prevedibile**: il floor dipende
dall'IDCG, cioè dalla **gallery** — che è la stessa (intera) nei due split; lo
split cambia solo *da dove* si pescano le query, e le due pool sono
statisticamente simili. Le percentuali della tabella qui sopra **non vanno
riscritte**: ricalcolate col floor vero si spostano di **≤ 0,2 punti**
percentuali (massimo osservato 0,19, su `hist` geometria).

**(b) Appaiamento verificato, non assunto** — è il vero guadagno di questa run.
`random_vision_valid_seed0.npz` vs `vision_ijepa_gem_whiten_full_valid.npz` e
`random_graph_valid_seed0.npz` vs `graph_gcn_tau02_full_valid.npz`: gallery sha1
identico (`22d91dcff041` / `0c24cfc05e18`), `query_seed` 42, `exclude_self`,
`geometry_weights` e `schema perquery/1` uguali, e le **2000 query coincidono
nello stesso ordine**. ⇒ il floor non è più solo un denominatore: si può usare
come **sistema B in un test appaiato** (`significance.py`) anche sul valid.

**(c) Spazio utile sul VALID, ricalcolato dai `.npz`** (floor `random` della
propria gallery, K=10; le righe graph sono **nuove**, §20 aveva solo il test):

| run | nDCG@10 C/T/G | spazio C/T/G |
|---|---|---|
| `siglip2/natural/head` | 0.8353 / 0.6323 / 0.9315 | 36.8% / 31.3% / 47.8% |
| `dinov3/natural/whiten` | 0.8245 / 0.6623 / 0.9344 | 32.7% / **36.9%** / 49.9% |
| `ijepa/gem/whiten` | 0.8229 / 0.6450 / **0.9473** | 32.1% / 33.7% / **59.8%** |
| `tipsv2/gem448/whiten` | **0.8433** / **0.6896** / 0.9347 | **39.9%** / **42.0%** / 50.2% |
| `graph gcn/tau02` | 0.9741 / 0.8298 / 0.9408 | 90.0% / 68.2% / 54.5% |
| `graph gat/nosym` | 0.9085 / 0.7441 / 0.9453 | 64.7% / 52.1% / **58.0%** |
| `graph hist` | 0.9999\* / 0.6428 / 0.8927 | 100.0%\* / 33.2% / **17.5%** |

\* oracolo per costruzione. **Letture:** (i) sul valid il miglior vision in
geometria (`ijepa` 0.9473) resta sopra il miglior graph (`gat/nosym` 0.9453), ma
di **0.0020** — dentro il margine che §10.2 dichiarava indecidibile senza test
appaiato, ora possibile; (ii) `hist` sulla geometria prende **17.5%** dello
spazio, cioè quasi nulla: la baseline training-free è cieca alla geometria, come
atteso; (iii) l'asse che separa davvero i sistemi resta la **topologia**
(33.2% di `hist` contro 68.2% di `gcn/tau02`).

**Verifiche di contorno**, tutte passate: lo script accetta
`[all|vision|graph] [test|valid]` e **rifiuta** uno split diverso
(`01_random_floor.sh:53-56`); i due suffissi `_seed0.npz` / `_mean5.txt` sono
prodotti con la ragione scritta nel codice (`:71-74`); `BASE` fissa
2000 query / seed 42 / 5 ranking-seed (`:60`); **nessun modulo Python** legge i
path di `results/random_floor/` (grep su `src/`, `tests/`, `scripts/`), quindi
la rinomina degli 8 file del test non ha rotto nulla — e infatti quei file hanno
ancora l'mtime del 31 lug (`mv`, nessun rilancio).

---

## 21. Fase A.1 chiusa (24 ago) — solo CPU, nessuna run GPU

Chiude i due rilievi mai verificati (B6 e la domanda PNG di A5), rimisura A5 e
sistema i 4 residui di A1. **Due esiti cambiano il quadro, uno lo semplifica.**

### 21.1 A5 è PIENO: tutta la GT geometrica è funzione dell'input del grafo

Misurato su `data_test.mat` (12.110 piante) e `data_valid.mat` (12.108):

    gtBox[-1] == unione(gtBoxNew)[[1,0,3,2]] − [0,0,1,1]   →  100.00%  (24.218/24.218)

Il footprint globale — da cui derivano `footprint_area` e `footprint_aspect` —
è ricostruibile **esattamente** dai `gtBoxNew`, che sono le node feature del
grafo (`graph_builder.py:59-92`). Le due convenzioni: `gtBox` ha gli **assi
scambiati** (riga/colonna) e i **massimi inclusivi** (−1). Entrambe erano già
note e gestite correttamente (`graph_builder.py:22-27`: si usano solo area e
aspect, invarianti allo scambio) → **nessun bug nel codice**, ma il
ridimensionamento di A5 del 30 lug (`§9`) era sbagliato.

**Conseguenza**: tutti e tre i componenti di `geometry_sim` (area, aspect,
`type_area_distribution`) sono funzioni deterministiche delle node feature. «La
GT geometrica è solo in parte ricostruibile dal grafo» è **smentito**, e con essa
«la geometria è l'unico asse alla pari» (vincolo 5 di `CLAUDE.md`, §4 qui).

⚠️ Sfumatura da non perdere: *calcolabile* ≠ *dato in pasto*. Il modello deve
comunque imparare a calcolarlo dal message passing; `footprint_area/aspect` sono
attaccati al `Data` ma **fuori** dal message passing (`graph_builder.py:145`,
`:161-162`) e nessun modello li legge (grep: solo `relevance.py:132-133`). È
esattamente la stessa circolarità di `type_area_distribution`, che l'audit già
considerava piena.

**Fatto nuovo, non nell'audit**: `gtBox[:-1]` **non** è `gtBoxNew` (solo il 5,0%
delle stanze coincide dopo la conversione). Il progetto usa solo `gtBox[-1]`,
quindi nessun effetto — ma il commento `rplan_metadata.py:177` («stesse bbox per
stanza») è impreciso.

### 21.2 Il colore delle PNG codifica `rType`, ma 13 tipi → 6 classi

300 piante, colore per stanza a voto di maggioranza su griglia 7×7 dentro
`gtBoxNew` (scala del render verificata costante, 2.852±0.012 px/unità).

| colore | tipi che ci finiscono | n |
|---|---|---|
| (253,244,171) giallo | SecondRoom 49%, MasterRoom 40%, StudyRoom 9%, ChildRoom 2%, GuestRoom | 748 |
| (205,233,252) azzurro | Bathroom 100% | 361 |
| (208,216,135) oliva | Balcony 100% | 336 |
| (244,242,229) crema | LivingRoom 98%, DiningRoom, Wall-in | 305 |
| (234,216,214) rosa | Kitchen 100% | 284 |
| (249,222,189) pesca | Storage 100% | 9 |

Purezza tipo→colore **100%** su 11 tipi su 12 (LivingRoom 99,67%: l'unica
eccezione è l'artefatto del box su stanza a L). Il colore è quindi una
**funzione deterministica di `rType`**, ma **non iniettiva**. ⚠️ Tipi rari nel
campione (Storage 9, DiningRoom 3, Wall-in 3, GuestRoom 2): per quelli la mappa
è indicativa. `Entrance` non compare mai come stanza.

**Conseguenza — il risultato più importante di A.1**: il ramo vision **ha
`rType` in input**, letto da una legenda a colori. «Il vision deve inferire la
semantica dai pixel» è **falso**. Ma la circolarità non è piena: la GT di
composizione usa i **13** tipi (`type_histogram`), il vision ne vede **6** — le
cinque camere sono indistinguibili fra loro. È un meccanismo candidato per il gap
già misurato sulla composizione (vision 0.83 vs graph 0.97): non solo «vedere è
difficile», ma **l'informazione non c'è nei pixel**.

⚠️ Regime reale: su una planimetria non colorata (CAD, foto, ResPlan) quel
segnale sparisce → i numeri di composizione/topologia del vision su RPLAN sono
**ottimistici**, e va dichiarato (tocca anche O3 / fase E).

**Follow-up a costo basso (fase B)**: misurare il tetto — nDCG di composizione
della baseline `hist` calcolata sulle **6 classi** invece che sui 13 tipi. Dà il
massimo che il vision può prendere su quell'asse leggendo il colore.

### 21.3 B6 è un FALSO ALLARME su entrambe le silenziosità

| Controllo | Misura | Esito |
|---|---|---|
| (a) id di relazione fuori da [0,9] in `rEdge` | **0** su 822.048 archi; range osservato esattamente [0,9] | il `clamp(0,9)` di `graph_builder.py:128-130` è codice morto |
| (b) `edge_attr` non one-hot in `graphs.pt` | **0** su 1.369.101 archi; `sum(edge_attr)` ha **un solo** valore distinto, 1.0 | il `reduce="mean"` non fonde mai relazioni diverse |

Causa: **0 coppie (i,j) ripetute** in tutto il dataset → il caso patologico non
può verificarsi. **Conseguenza**: la diagnosi «GAT è il più capace ed è il
peggiore» (§2) **non ha** una spiegazione alternativa in attributi corrotti;
regge come sta.

**Fatto nuovo**: il contratto `edge_index [2,2E]` di `dataset.md:96-98` è violato
nello **0,503%** dei grafi (339/67.405), che perdono 1-3 archi. Causa trovata:
**397 piante su 80.729 (0,49%) hanno un self-loop** (`i == j`) in `rEdge`;
simmetrizzare un self-loop produce un duplicato che `to_undirected` fonde. Non
sono gli «archi paralleli» previsti dal docstring `graph_builder.py:133-134`:
**entrambe** le affermazioni documentali erano imprecise.

### 21.4 I quattro residui di A1, chiusi

1. **`scripts/vision/{04..07}`**: aggiunto `EVAL_SPLIT="${EVAL_SPLIT:-valid}"`,
   `eval.split=$EVAL_SPLIT`, split stampato nell'intestazione del log. **Default
   `valid`** (decisione dell'utente): per rileggere il test serve
   `EVAL_SPLIT=test sbatch ...`. ⚠️ un rilancio senza flag **non riproduce più** i
   file di luglio — è voluto.
2. **Test letto 3 volte** → dichiarato in `PAPER.md §7 Caveat 5`, con i tre
   numeri e la pre-registrazione del primario.
3. **Oracolo per-asse obsoleto** → `CLAUDE.md` corretto (0.9473 sul valid); §1
   annotato come NON citabile.
4. **Default YAML → `valid`** (decisione dell'utente):
   `configs/vision_retrieval.yaml:55` (era `null`),
   `configs/graph_retrieval.yaml:21` (era `test`) e — **non previsto dal
   residuo** — anche `graph_evaluate.py:332`, che aveva un **secondo** default
   `test` nell'argparse. Ponte YAML→flag verificato: emette `--split valid`.
   Regressione: **28 test passati** (`test_perquery`, `test_metric_diagnostics`,
   `test_loader`).

**Bilancio A.1 — 13 rilievi su 13 ora verificati**: **12 confermati**, **1 falso
allarme** (B6, entrambe le parti), **1 rafforzato** (A5, da parziale a pieno) e
**1 fatto nuovo** (la legenda a colori: la circolarità tocca anche il vision).
Nessun fix dei rilievi confermati: quella è la fase B.

### 21.5 L'ablation di luglio conteneva già l'impronta di A5 — e dice cosa impara la rete

⚠️ **Interpretazione, non misura nuova**: rilettura di `§2f` (28 lug) alla luce
di `§21.1`. Nessun job, nessun numero ricalcolato.

**Fatti (da `§2f`, sonda sul valid: gallery 5000, 500 query — confrontabili fra
loro, NON con la valutazione finale a gallery intera):**

| sistema | comp | topo | geom | media |
|---|---|---|---|---|
| **nessuna rete** (add-pool delle feature grezze) | 0.884 | 0.625 | 0.906 | 0.8050 |
| GNN a **pesi casuali** | — | — | — | 0.799-0.809 |
| GNN **allenata** (GCN) | — | — | — | 0.8615 |

Guadagno del training (pesi casuali → allenata, GCN): C **+0.056**, T **+0.100**,
G **+0.012**.

**Interpretazione 1 — la conferma incrociata di A5.** La geometria arriva a
**0.906 senza nessuna rete**, e il training la muove di **+0.012**. Se fosse un
asse davvero indipendente, un sistema che si limita a sommare le feature dei nodi
non potrebbe prenderla quasi tutta. È l'**impronta empirica** di ciò che §21.1 ha
poi verificato in aritmetica esatta sui `.mat`: la GT geometrica è una funzione
delle node feature. **Due misure indipendenti, stessa conclusione** — una
sperimentale (lug), una analitica (ago).

**Interpretazione 2 — che cosa impara davvero la GNN.** Con i tipi in one-hot,
l'`add`-pool **è** l'istogramma delle stanze → composizione e geometria sono
aritmetica, non apprendimento. Ciò che sommare i nodi **distrugge** sono gli
archi: la struttura relazionale. Coerente col guadagno per asse, T +0.100 contro
G +0.012. **La rete impara a rendere confrontabile la topologia**; il resto lo
porta l'architettura.

⚠️ **Da non confondere con «la rete copia la risposta»**: la GT non entra mai
nella loss (InfoNCE self-supervised, positivo = stessa pianta aumentata). La
prova è già in `§2b`/`§2d`: il GAT ha la loss **migliore** e l'nDCG **peggiore**,
e oltre l'epoca 3 la loss migliora mentre il retrieval peggiora per 20 epoche. Se
stesse copiando, ottimizzare meglio non potrebbe far peggio.

**Interpretazione 3 — la proiezione casuale fa danno.** La GNN a pesi casuali
(0.799-0.809) sta **sotto** il non avere rete affatto (0.8050): l'architettura
non inizializzata *perde* informazione che l'add-pool conservava. Il primo lavoro
del training è **recuperare quel danno**, e solo dopo aggiungere la topologia.

**Conseguenze per il report** (nessuna richiede una run):
1. Il benchmark **non** misura «inferire la struttura da un input grezzo» — non lo
   chiede a nessuno dei due rami (§21.1, §21.2). Misura **quanto bene si comprime
   in uno spazio metrico un'informazione già posseduta**: domanda legittima, ma
   più stretta, e va scritta così.
2. Il **denominatore** da riportare accanto a ogni numero graph è la GNN a pesi
   casuali (già chiesto da `§2f`), non solo `hist`: contro quello il guadagno è
   **+0.056** di media, quasi tutto topologia — vero, ma molto più piccolo di
   quanto suggerisca uno 0.972.
3. Manca il numero gemello per il ramo vision: l'equivalente dell'add-pool
   (encoder non allenato / feature grezze) come pavimento architetturale. Da
   valutare in fase B insieme al tetto a 6 classi di §21.2.


---

## 22. Fase A.4 — sensibilità ai pesi della geometria (24 ago): **il claim di punta NON REGGE**

Prima esecuzione di `src/evaluation/geometry_variants.py` (esisteva dal 30 lug,
mai lanciato). Criterio **fissato il 30 lug nel docstring del modulo, righe
19-26** — precede i dati, non è stato adattato: *il claim «A batte B sulla
geometria» regge se in TUTTE le pesature il segno del delta è invariato E il CI
95% esclude lo zero.* Output completi in `results/geometry_variants/`.

**Pre-registrazione dichiarata prima di guardare**: confronto primario
`ijepa/gem/whiten` vs `graph gat/nosym` sul valid; previsione **INCONCLUSIVO**
(CI larghi). ⚠️ **La previsione era sbagliata per difetto**: l'esito è il più
forte possibile, **NON REGGE** — inversione dimostrata, non incertezza.

**Integrità verificata per prima**: con `(1,1,1)` il ricalcolo riproduce il numero
salvato entro **2·10⁻⁸** su tutti e cinque i file → `ret_rows`, gallery e gain
sono coerenti; ciò che segue non è un artefatto del ricalcolo.

### 22.1 I tre confronti, tutti NON REGGE

nDCG@10 geometria, delta appaiato (A − B), 2000 query del valid, bootstrap
B=10000 seed 0. **In grassetto le pesature che invertono il segno con CI che
esclude lo zero.**

| pesatura (area,aspect,dist) | ijepa − gat/nosym | ijepa − gcn/tau02 | ijepa − tipsv2 |
|---|---|---|---|
| **(1,1,1)** baseline | +0.0020 | +0.0065 | +0.0126 |
| (1,0,0) solo area | +0.0088 | +0.0080 | +0.0129 |
| (0,1,0) solo aspect | +0.0289 | +0.0451 | +0.0375 |
| **(0,0,1) solo distribuzione** | **−0.0327** | **−0.0350** | **−0.0136** |
| (1,1,0) forma pura | +0.0189 | +0.0266 | +0.0252 |
| (1,1,2) distribuzione doppia | **−0.0067** | **−0.0039** | +0.0061 |

Tutti i CI 95% escludono lo zero in tutte e 18 le celle: **non c'è una sola
cella indecisa**. L'instabilità non è rumore, è struttura.

### 22.2 Che cosa è morto e che cosa sopravvive

**Morto — il claim aggregato.** «Il vision batte il graph sulla geometria»
(§4:192) regge **solo perché la formula fa 2 voti a 1**. Il `+0.0020` della
baseline è più piccolo dell'inversione che si ottiene pesando ×2 una delle tre
componenti — una scelta arbitraria esattamente quanto `(1,1,1)`. Il numero non è
una proprietà dei modelli, è **una proprietà della formula** (rilievo **A6**:
pesi hardcoded, `relevance.py:137-146`, nessun punto di iniezione).

**Vivo, e più forte di prima — il claim decomposto.** Sulla **forma del
footprint** `(1,1,0)` il vision vince ovunque e di **+0.0189 / +0.0266**, cioè
~10× il delta aggregato, con CI stretti. Il claim difendibile diventa: *«sulla
forma dell'appartamento (area e proporzioni) il vision batte il graph di un
margine 10 volte più grande di quello che si legge nell'aggregato; sulla
distribuzione di area per tipo perde, ed è la componente che il graph ottiene
gratis»*. È una frase più informativa, non una ritirata.

**Il meccanismo, che chiude il cerchio con §21.1 e §21.5.** L'inversione cade
sempre su `type_area_distribution`, e non è un caso:
- `type_area_distribution` = somma delle aree per tipo / totale → è
  **letteralmente un'operazione di add-pooling** sulle node feature, cioè ciò che
  il ramo graph calcola per costruzione (§21.5). Non deve impararla.
- `footprint_area` / `footprint_aspect` = **min/max** sugli angoli dei box →
  ricostruibili (§21.1, 100%) ma con un'aggregazione che l'add-pool **non**
  fornisce: sommare non dà il minimo.

⇒ Il confine non è «visione vs simboli», è **quale aggregazione l'architettura
implementa nativamente**. Sulla componente che esce per somma vince il graph;
sulle due che richiedono un massimo vince il vision.

⚠️ **Correzione al docstring del modulo** (righe 34-38, scritte il 30 lug): dice
che `(1,1,0)` è «la pesatura meno circolare». Dopo §21.1 va precisato — `(1,1,0)`
**non è non-circolare**, è la pesatura che richiede al grafo *più computazione*
per essere ricostruita dal suo input. La differenza fra le pesature è di **costo
di calcolo**, non di accesso all'informazione.

### 22.3 L'instabilità non è un problema solo cross-ramo

Il terzo confronto è **dentro** il ramo vision: `ijepa` vs `tipsv2` inverte
anch'esso su `(0,0,1)` (−0.0136). Quindi **anche «ijepa è il miglior vision in
geometria» è dipendente dalla pesatura**, e va qualificato allo stesso modo.

Due gradi di robustezza, però, e la differenza conta:
- vision vs graph: invertono **2 pesature su 6**, e l'inversione è **più grande**
  del delta di baseline → l'aggregato è quasi una moneta;
- ijepa vs tipsv2: inverte **1 sola** pesatura (l'estrema `(0,0,1)`), `(1,1,2)`
  regge, e 5 su 6 danno ijepa → claim **fragile ma non capovolto**.

### 22.4 Conseguenze operative

1. **`CLAUDE.md`, `PAPER.md` e §4 non possono più portare il claim aggregato**
   senza la qualificazione. Aggiornati il 24 ago.
2. **Il report riporta la geometria decomposta**, non solo l'aggregato: è la
   lettura che sopravvive al test ed è anche quella che spiega *perché*.
3. **A6 sale di priorità**: finché i pesi restano hardcoded senza punto di
   iniezione, ogni claim geometrico è una scelta implicita non dichiarata. Il
   minimo è **dichiarare `(1,1,1)` come convenzione** e pubblicare la tabella di
   sensibilità accanto — non «scegliere la pesatura che vince», che sarebbe
   selezione sul risultato.
4. **Strumento esteso** (additivo, comportamento invariato senza il flag):
   `geometry_variants.py` ora accetta `--gallery-b` per i confronti cross-ramo,
   che prima `check_gallery` rifiutava con errore duro. L'IDCG resta calcolato
   sulla gallery di ciascun ramo → confronto appaiato sui nomi ma **non
   perfettamente equo fino a B.3**, ed è dichiarato nell'output di ogni run.

---

## 23. Fase A.5 — «migliore» = **più robusto**. Decisione dell'utente, 24 ago

Chiude l'ultimo punto della fase A e sblocca la fase D. **Scritta prima delle
run che la applicheranno**, come richiede il protocollo: da qui in poi non si
tocca per far vincere un candidato.

### 23.1 La definizione operativa

> **Il modello migliore è quello che regge meglio quando la query è incompleta.**

Resa misurabile, sei scelte esplicite:

1. **Misura**: curva di **self-recovery MRR** (ritrovare la pianta sorgente
   partendo dalla sua versione degradata), strategia `random`, sul **valid**.
   *Perché questa e non l'nDCG per-asse in partial*: il self-recovery è l'unica
   metrica del partial il cui protocollo è **già corretto** — il self va incluso
   perché *è* il task (`retrieval.md:118-121`). L'nDCG per-asse in partial usa
   `exclude_self=False` (`evaluate.py:230`), che è il rilievo **B3**: fino al fix
   B.4 non è confrontabile col full e **non entra** nel criterio.
2. **Punti della curva**: f ∈ {0.25, 0.5, 0.75}. **f = 0.0 è escluso**: lì tutti
   gli encoder fanno MRR 0.970, un tetto dei **dati** (duplicati esatti in RPLAN,
   §1) e non dei modelli — includerlo diluirebbe il segnale con una costante.
3. **Statistica di sintesi**: **area sotto la curva** sui tre punti. Una sola
   cifra per configurazione, che pesa i tre livelli di difficoltà allo stesso
   modo.
4. **Assoluto, non relativo.** «Regge meglio» = **punteggio più alto sotto
   masking**, non «cala di meno». ⚠️ È una scelta, e va dichiarata: un modello
   può essere piattissimo perché è mediocre ovunque. Poiché la domanda pratica è
   *«quale uso quando la planimetria è incompleta?»*, vince chi arriva più in
   alto **là**, non chi scende con la pendenza più dolce.
5. **Significatività**: delta **appaiato** con CI 95% bootstrap sui valori
   per-query, stesso strumento del resto del progetto (`significance.py`).
6. **Spareggio, deciso ora** (il roadmap lo chiedeva esplicitamente): se il CI
   del delta contiene lo zero → i due sono **equivalenti** e si passa al criterio
   secondario, cioè l'nDCG@10 **full** sull'asse debole del ramo; se anche lì il
   CI contiene lo zero → vince la configurazione **più economica** (dimensione
   d'embedding minore, poi risoluzione minore), per riproducibilità e costo.

**Guardrail dichiarato**: la configurazione scelta per robustezza **si riporta
sempre accanto al suo punteggio full**, come costo esplicito della scelta. Non si
introduce una soglia arbitraria di perdita massima sul full: si dichiara il
prezzo e lo si lascia leggere.

### 23.2 Le tre conseguenze, verificate a disco

**(a) I tre encoder nuovi non hanno partial → oggi non possono competere.**
`tipsv2/gem448/whiten` è il miglior vision su composizione e topologia (§18), ma
non ha né `head.pt` né run partial. Verificato: `head.pt` esiste **solo** per i 5
storici (`siglip2`, `radio`, `ijepa`, `dinov3`, `dinov2`). ⇒ o si lancia il
partial anche per i nuovi, o il criterio elegge un vincitore fra i 5 vecchi. **È
una run da mettere in coda, non un ostacolo alla decisione.**

**(b) La head rientra in gioco, ed è la conseguenza più grossa.** Sul full «la
head non aiuta»; sul **partial** è un guadagno grande e crescente col masking
(dinov3 a f=0.75: MRR 0.205 → 0.668, ~3×, §1). Con la robustezza come criterio,
il vincitore sarà quasi certamente una configurazione **con head**.
⚠️ Ma quelle head sono selezionate sulla **val-loss InfoNCE**, il criterio che il
ramo graph ha smentito (rilievo **A3**, §2). ⇒ **B.6 sale di priorità**, e cambia
di forma: la probe di selezione della head dovrebbe essere una probe **partial**,
non full — altrimenti si seleziona di nuovo col criterio sbagliato per il task
sbagliato.

**(c) Il ramo graph non ha partial affatto** (è in fase C). Il criterio **non è
applicabile** al graph oggi. Decisione provvisoria da confermare: si congela la
configurazione graph con il criterio **full** (`gcn/tau02`, §16) dichiarandolo
come provvisorio, e si rivede dopo il partial del graph. Altrimenti la fase D
resta bloccata su un ramo solo.

**(d) Mancavano gli strumenti per applicare il criterio — scritti il 24 ago**
col mandato dell'utente. Due pezzi, entrambi **additivi** (nessun numero già
prodotto cambia; 32 test verdi):
- **`scripts/evaluation/05_perquery_vision_partial_valid.sh`** — gemello di `02`
  per il partial. Serviva perché `02` cabla `partial.enabled=false` e
  `scripts/vision/06,07` fanno il partial ma **non** valorizzano
  `eval.perquery_dir`: stampano la MRR nel log e buttano i valori per query.
  Default strategie = **solo `random`** (`semantic`/`topology` triplicano il
  costo e non entrano nel criterio). Porta il `--constraint` anti-Blackwell, che
  `evaluation/{02,03,04}` ancora non hanno.
- **`significance.py --metric self_rr`** — il campo `self_rr` era **già salvato**
  per query (`perquery.py:219-221`) ma non leggibile dal tool: ha forma `[Q]`,
  senza asse e senza k, mentre `METRICS` assume `(asse, k)`. Gestito come caso a
  parte, non aggiunto a quell'elenco. Il ramo dedicato **non** stampa l'endpoint
  primario per-asse: in partial quelle metriche hanno `exclude_self=False`
  (rilievo B3) e affiancarle inviterebbe a leggerle come confrontabili col full.
  `check_compatible` blocca già il confronto fra **frazioni diverse**
  (`partial_label` ∈ `STRICT_META_KEYS`): misurerebbe il livello di masking, non
  la robustezza.

Volume delle run (simulato sui loop): frozen **52 valutazioni → 208 `.npz`**,
head **28 → 112**. Comandi in `COMANDI.md § 05`.

### 23.3 Che cosa sblocca

La fase D aveva come vincolo d'ingresso «una configurazione congelata per ramo +
la definizione di migliore». La definizione ora c'è. Restano da produrre i
**numeri** che la applicano: il partial per-query dei candidati (a) e la
decisione su (c). Fino ad allora la coppia per D.0 resta quella indicata dal
criterio full, e va dichiarata come **provvisoria**.

### 23.4 Catena validata sui primi dati reali (24 ago) — smoke job `dinov3 frozen`

Primo job di `05_perquery_vision_partial_valid.sh`: **24 file** attesi e prodotti
(3 pooling x 2 trasformazioni x 4 frazioni), 24 conferme di scrittura nel log.

**Integrita' del contenuto verificata** su
`vision_dinov3_gem_whiten_partial-random-f0.75_valid.npz`: `self_rr` presente
`[2000]`, `mode=partial`, `partial_label='random f=0.75'`, `split=valid`,
`exclude_self=False` (corretto: nel partial il self **e'** il bersaglio),
`query_seed=42`, gallery sha1 `22d91dcff041` — **la stessa delle run full sul
valid**, quindi appaiabile con esse.

**Primo test appaiato su `self_rr`** (whiten vs raw, dinov3/gem, f=0.75):

    MRR 0.2500 vs 0.0638 — delta +0.1862, CI 95% [+0.1716, +0.2012], p 1.3e-149

⇒ **sotto masking pesante il whitening non e' un ritocco: senza, il modello
collassa** (~4x). Conferma quantitativa dell'osservazione qualitativa di §1
(«con raw puro gli encoder collassano gia' a f=0.25; il solo whitening li
recupera in gran parte»), ora con CI e su per-query.
Il tool si rifiuta correttamente di stampare le metriche **per-asse** del partial
(guardia sul rilievo B3): esce solo l'endpoint `self_rr`.

⚠️ **Terzo pezzo ancora mancante per applicare §23 alla lettera**: l'**AUC** sui
tre punti f ∈ {0.25, 0.5, 0.75} non e' calcolata da nessuno strumento.
`significance.py --metric self_rr` confronta a **un** livello di masking per
volta. L'AUC per-query e' una combinazione lineare dei tre file (allineati per
nome), quindi il test appaiato sull'AUC e' immediato — ma va scritto. **Non
blocca le run**: i file che produce sono la materia prima in entrambi i casi.


---

## 24. A.5 APPLICATA — la griglia partial per-query sul valid (24 ago, 320 run)

Prima applicazione del criterio §23. **320/320 file** a disco
(`results/perquery/vision_partial_valid/`), 80 configurazioni × 4 frazioni.

**Appaiamento verificato su tutti e 320, non a campione**: gallery sha1
`22d91dcff041` unico, `split=valid`, `query_seed=42`, `exclude_self=False`,
2000 query **identiche e nello stesso ordine**, `self_rr` presente ovunque.

**AUC** = media di MRR su f ∈ {0.25, 0.5, 0.75} **per query** (pesi uguali, come
scritto in §23 prima di vedere i dati), poi media sulle query.

### 24.1 Il verdetto: `dinov3/natural/head`

| # | configurazione | AUC | f=.25 | f=.50 | f=.75 | full C/T/G |
|---|---|---|---|---|---|---|
| **1** | **`dinov3/natural/head`** | **0.8157** | 0.9400 | 0.8405 | 0.6665 | 0.8284/0.6416/0.9349 |
| 2 | `dinov3/mean/head` | 0.7557 | 0.9148 | 0.7640 | 0.5883 | 0.8215/0.6177/0.9389 |
| 14 | `pespatial/natural/whiten` ← **miglior frozen** | 0.6360 | 0.9365 | 0.6679 | 0.3035 | 0.8072/0.6113/0.9430 |
| 49 | `tipsv2/gem448/whiten` ← **vincitore col criterio full** | 0.2831 | 0.5980 | 0.1891 | 0.0622 | 0.8433/0.6896/0.9347 |

**Delta appaiati sull'AUC** (bootstrap B=10000, seed 0, n=2000). ⚠️ **Tutti i CI
escludono lo zero: nessun pareggio, la regola di spareggio di §23 non scatta.**

| confronto | delta | CI 95% |
|---|---|---|
| #1 vs #2 | **+0.0600** | [+0.0499, +0.0700] |
| #1 vs miglior frozen | **+0.1797** | [+0.1668, +0.1923] |
| #1 vs vincitore del criterio **full** | **+0.5325** | [+0.5192, +0.5457] |
| quanto compra la **head** (encoder fisso, `dinov3/natural`) | **+0.2840** | [+0.2710, +0.2970] |
| `pespatial` vs `dinov3`, entrambi frozen+whiten | **+0.1043** | [+0.0915, +0.1174] |

**Costo dichiarato sul full** (guardrail §23: si dichiara, non si soglia). Rispetto
a `tipsv2/gem448/whiten`: composizione **−0.0149**, topologia **−0.0480**,
geometria **+0.0002**. La robustezza si paga quasi tutta sulla **topologia**.

### 24.2 Quattro letture

**(a) Il criterio ribalta la classifica, non la sfuma.** `tipsv2/gem448/whiten` —
il migliore su composizione e topologia a immagine intera — è **49° su 80** per
robustezza, a 2.9× di distanza dal primo. Non è un riordino marginale: le due
domande («chi è più bravo?» e «chi regge quando manca metà pianta?») hanno
risposte **diverse**, e §23 ha scelto la seconda.

**(b) Il tetto f=0.0 conferma la scelta fatta prima dei dati.** Su **tutte e 80**
le configurazioni: 0.9705–0.9712, ampiezza **0.0007**. Escluderlo dalla sintesi
non era prudenza, era necessario: quel punto ha informazione **zero** ed è il
tetto dei *dati* (duplicati RPLAN), non dei modelli.

**(c) La head generalizza fuori dal suo range di training.** È allenata con
`mask_fraction: [0.1, 0.5]` (`vision_retrieval.yaml:39`), ma il distacco più
grande è a **f=0.75**, fuori distribuzione: 0.6665 contro 0.2209 dello stesso
encoder frozen (3×). Non sta interpolando ciò che ha visto.

**(d) `pespatial` è il miglior encoder frozen.** Fra le configurazioni senza
head vince uno dei tre nuovi, con **+0.1043** su `dinov3` a parità di
trasformazione. La supervisione spaziale densa serve **alla robustezza**, non
alla geometria del full — dove §19 si aspettava l'effetto. Ipotesi da rifare.

### 24.3 Tre caveat, tutti da dichiarare nel report

1. **Il confronto head vs frozen non misura «quale encoder è migliore».** La head
   è allenata con InfoNCE il cui positivo è **la vista mascherata della stessa
   pianta**: è *esattamente* il task del self-recovery. Il risultato dice «una
   testa allenata per il masking funziona», non «dinov3 è un encoder superiore».
   Il confronto equo fra encoder è quello **dentro** il gruppo frozen (dove vince
   `pespatial`).
2. **Nessuna fuga di dati nei pesi** — verificato: `projection_pairs.py:123-132`
   genera le coppie da `train+valid`, ma `train_projection.py:72-76` **fitta solo
   le righe `train`**; il `valid` serve unicamente alla val-loss di early
   stopping. Resta l'asimmetria normale: il checkpoint della head è **scelto**
   guardando il valid, le config frozen non hanno nulla da scegliere.
   ⚠️ Specularmente il whitening è fittato su **tutta la gallery** (rilievo A2),
   che di valid e test ne contiene: l'asimmetria non è tutta da una parte.
3. **La head vincente è selezionata sulla val-loss InfoNCE**, il criterio
   smentito (rilievo A3). Vince **nonostante** un criterio di selezione
   sbagliato ⇒ **B.6** (riallenamento con probe **partial**) può solo migliorarla.
   Il numero di oggi è quindi un **limite inferiore**.

### 24.4 Config congelata per il ramo vision

**`dinov3/natural/head`** — primo vincolo d'ingresso della fase D soddisfatto.
⚠️ Da rivedere dopo **B.6**. Il ramo graph resta senza partial (fase C): la sua
config (`gcn/tau02`, criterio full) resta **provvisoria**, come da §23.2c.


---

## 25. Fase B.2 — whitening stimato sul solo train (25 ago, decisione + codice)

**Decisione dell'utente** (chiude la Domanda aperta n.2 di `roadmap.md`): le
statistiche del whitening si stimano sul **solo split train**, in entrambi i
rami. Il ramo graph non ha whitening (zero occorrenze in `src/graph/`), quindi
la decisione opera solo sul vision.

**Implementato il 25 ago, nessun numero ancora prodotto.** Il codice:
`whitening.fit_split` in `configs/vision_retrieval.yaml` (default `train`) →
`whitening_fit_rows()` in `evaluate.py` → `prepare_index(fit_rows=...)` in
`retrieval_model.py`, che stima su quelle righe e **indicizza la gallery
intera**. Il `fit_split=all` resta come termine di confronto.

⚠️ **Il tag di run cambia**: `whiten` → `whiten-train` (`config.py:transform_tag`).
Non è cosmetico: i file per-query hanno il tag nel nome, e senza questa
distinzione una run train-only avrebbe **sovrascritto** i **160 file `whiten`**
del partial valid — cioè le run trasduttive su cui poggia §24. Verificato a
disco: 160 file `*whiten*` presenti in `results/perquery/vision_partial_valid/`.

**Che cosa cambia nei numeri (previsione, da falsificare con la run).** Solo le
configurazioni con `whitening.enabled=true`. ⚠️ Il **vincitore di §24**
(`dinov3/natural/head`) **non usa whitening**: il suo AUC 0.8157 resta identico.
Cambiano i suoi **concorrenti** (40 config su 80 hanno `whiten` nel tag), quindi
può cambiare **la classifica**, non il primo posto per come è misurato oggi.
Previsione dichiarata prima della run: la stima su ~1/3 delle righe è più
rumorosa ma non trasduttiva → atteso un calo **piccolo** delle config `whiten`
(l'insieme di stima resta grande in valore assoluto). Se il calo fosse grande, il
guadagno del whitening misurato finora era in parte **leakage**.

**Da misurare (run dell'utente)**: partial valid con `fit_split=train` sulle
config `whiten`, poi delta appaiato con `significance.py --metric self_rr` allo
stesso livello di masking contro i file `whiten` già a disco. Comando in
`experiments.md § B.2`.


---

## 26. Fase B.4 — il self esce dalle metriche per-asse del partial (25 ago, codice)

Chiude il rilievo **B3**. Prima, nel partial, la pianta originale restava in
gallery, fra i rilevanti e nell'IDCG **anche** per le metriche per-asse: veniva
recuperata quasi sempre in cima con gain 1.0, quindi quei numeri erano gonfiati e
non confrontabili col full.

**Il fix**: `partial_rows()` in `evaluate.py` ricava **due liste dalle stesse
risposte FAISS** — `self_rows` (self dentro) per il self-recovery, `axis_rows`
(self fuori) per le metriche per-asse. Entrambe restano lunghe `max_k` perché il
filtro parte dalle `max_k + 1` risposte già richieste a FAISS. Allineati anche il
meta dei `.npz` (`exclude_self: True`), il report stampato e la riga per-asse
delle figure.

**Che cosa NON cambia — e va detto subito**: il `self_rr`, cioè l'endpoint del
criterio A.5, è calcolato sulla lista col self e **non passa** dalle metriche
per-asse. Quindi **§24 resta valido**: AUC, verdetto `dinov3/natural/head` e i
delta appaiati non si toccano. Cambia la lettura per-asse del partial, che nel
report era comunque dichiarata non confrontabile.

**Che cosa cambia**: tutte le tabelle per-asse delle run partial. Attese **più
basse** (spariva un rilevante gratis in cima); di quanto, lo dirà la run.

⚠️ **Trappola operativa**: il nome dei file per-query non dipende da B.4 → una
run nuova con la stessa config **sovrascrive** il `.npz` vecchio (per le config
con whitening no: il tag è cambiato con B.2). Il file nuovo è auto-descrittivo
(`exclude_self=True`) e `check_compatible` rifiuta i confronti misti, quindi non
si può sbagliare *senza accorgersene* — ma i numeri vecchi, se servono, vanno
copiati prima.

**Verificato**: suite 60 passed (CPU); 3 test nuovi sulla separazione delle due
liste + contro-prova quantitativa che il self alzava l'nDCG. **Non verificato**:
nessuna run reale — il delta per-asse è ancora da misurare.


---

## 27. Fase B.3 — la gallery condivisa fra i rami (25 ago, codice)

Chiude il rilievo **B2** (67.453 vs 67.405). Prima, ogni confronto vision↔graph
richiedeva `--allow-gallery-mismatch`: due sistemi valutati su corpus diversi,
con IDCG diversi, e il caveat scritto solo in un `echo` dello script.

**Il fix, e la scelta che conta.** `src/evaluation/gallery_join.py` calcola
l'intersezione **una volta** e la congela in un JSON con provenienza e sha1; i
due rami lo leggono (`eval.gallery_names` nel vision, `--gallery-names` nel
graph) e **restringono E riordinano** la propria gallery sull'ordine canonico
(alfabetico) del file. Il riordino non è cosmetico: `gallery_sha1` dipende
dall'ordine delle righe, quindi è quello a rendere i due rami identici riga per
riga. Conseguenza verificata in test: a parità di seed e split **i due rami
campionano le stesse query**, e `check_compatible` accetta il confronto senza
override.

**Perché un artefatto congelato e non un ricalcolo a ogni run**: due run lanciate
a giorni di distanza userebbero gallery diverse senza che nulla lo dica. Il file
porta lo sha1 e viene riverificato a ogni lettura (un troncamento a mano viene
rifiutato).

⚠️ **Cambia i numeri di entrambi i rami**: ~48 piante escluse dal corpus di
ricerca, e il pool delle query cambia di conseguenza. Il vincolo DURO 2 regge
nello spirito — non si sceglie un sottoinsieme comodo, si toglie ciò che un ramo
non può vedere — ma va dichiarato in ogni tabella prodotta così.

**Stato**: codice + test (66 passed). ⏳ Manca l'artefatto reale e le run sopra.
Comandi in `experiments.md § B.3`.


---

## 28. Fase C.0 — il partial sul ramo graph (25 ago, codice)

Il ramo graph non aveva masking: il criterio A.5 non gli si applicava, la sua
config restava provvisoria (B.7) e la fase D non aveva il vincolo d'ingresso.

**Il codice**: `src/graph/graph_partial_query.py` (`filter_meta` toglie le stanze
e **rimappa** gli archi; `make_partial_graph` ricostruisce il grafo con
`build_graph`) + `evaluate_partial` in `graph_evaluate.py`, con i flag
`--partial*`. Le etichette di run sono **identiche** a quelle del vision
(`random f=0.5`, `semantic`, `topology`), quindi i file per-query hanno nomi
paralleli nei due rami.

**La decisione che rende appaiato il confronto.** La selezione delle stanze non
è reimplementata: si **importa** `select_rooms_to_remove` dal ramo vision. Il
progetto di norma non incrocia i rami, ma qui l'oggetto condiviso è il
*protocollo*: due implementazioni che divergono di una riga farebbero togliere
stanze diverse, e la curva misurerebbe il masking invece dei modelli.

⚠️ **Cade la domanda aperta «stessa frazione di area o di stanze?»**: il masking
del vision seleziona **stanze** (`select_rooms_to_remove`), il riempimento bianco
è solo il modo di renderlo in pixel. Restava aperta per il masking **nuovo** di
O4 (crop e patch), dove l'unità è l'area — lì la domanda vale ancora.

⚠️ **Il pairing dipende da B.3**: il seed per-query è `seed + qi` e solo con la
gallery condivisa (ordine canonico) il `qi` è lo stesso nei due rami. Senza,
stesse frazioni ma piante diverse.

**Altre due scelte dichiarate**: il footprint resta quello della pianta intera
(la tela non cambia, come nell'immagine degradata); le query svuotate del tutto
dalla strategia non sono valutabili e vengono **contate e saltate**.

**Verificato**: 75 passed, incluso uno smoke end-to-end del giro completo
(grafi degradati → encoder → FAISS → due viste → `.npz` con `self_rr`, nome file
`graph_gcn_test_partial-random-f0.25_valid.npz`). Le feature dei nodi superstiti
restano **bit-identiche** a quelle del grafo intero. **Non verificato**: nessuna
run reale, nessun numero.


---

## 29. 10 set — artefatto B.3, AUC promossa a modulo, codice B.6 (nessun numero nuovo)

- **B.3 artefatto**: `results/shared_gallery.json`, 67.405 nomi, sha1
  `0c24cfc05e18`, 48 solo-vision esclusi, 0 solo-graph; verificato che tutti i 26
  `image_paths.json` vision e i 7 `names.json` graph lo coprono per intero.
  `gallery_names` impostato in **entrambi** i YAML.
- **AUC di A.5 = modulo** (`src/evaluation/robustness_auc.py`, chiude il debito
  trasversale n.1): sui 320 file di §24 riproduce **esattamente** 0.8157 /
  0.7557 e i delta +0.0600 [+0.0499, +0.0700], +0.5325 [+0.5192, +0.5457].
- **B.6 codice**: `training.selection=probe_partial` sceglie l'epoca sull'AUC
  di una probe partial (1000 query valid **disgiunte** dalle 2000 di eval, seed
  di masking 1042+riga), checkpoint in `head.file=head_probe.pt` (tag
  `head-probe`); `head.pt` storico intatto. Cross-check su 3 query reali: rank
  del self identici all'evaluate (anche 0.1 e 0.333 a f=0.75). Init della head
  ora con seed (prima non riproducibile; i `head.pt` a disco non cambiano).
- Job lanciati (loop): partial vision protocollo B (dinov3 frozen/head + frozen
  `fit_split=all`), C.0 graph (hist + 6 varianti), B.6 (5 encoder). Traccia in
  `.claude/TODO.md`.


---

## 30. C.0 misurato (10 set, job 103459) — il self-recovery del graph CROLLA sotto masking

Protocollo B: gallery condivisa (n=67.405, sha1 `0c24cfc05e18`), `exclude_self=True`,
valid, 2000 query, 0 query svuotate. AUC con `robustness_auc` (ricetta § 23).

| config | AUC | f=.25 | f=.50 | f=.75 | f=0.0 MRR |
|---|---|---|---|---|---|
| `gat/nosym` | **0.0241** | 0.0681 | 0.0037 | 0.0006 | 0.9765 |
| `sage/nd01` | 0.0129 | 0.0362 | 0.0024 | 0.0001 | 0.9636 |
| `sage/base` | 0.0089 | 0.0244 | 0.0020 | 0.0002 | 0.9635 |
| `gat/base` | 0.0071 | 0.0202 | 0.0010 | 0.0001 | — |
| `gcn/tau02` (provvisoria) | 0.0058 | 0.0153 | 0.0019 | 0.0001 | 0.9645 |
| `gcn/base` | 0.0042 | 0.0103 | 0.0019 | 0.0002 | 0.9662 |
| `hist` | 0.0023 | 0.0044 | 0.0018 | 0.0009 | 0.0259 |

Delta appaiati `gat/nosym` − ciascuno: tutti con CI che esclude lo zero
(vs `sage/nd01` +0.0112 [+0.0088, +0.0137]; vs `gcn/tau02` +0.0183 [+0.0154, +0.0213]).

**Fatti.** (1) A f=0.0 le GNN ritrovano il self (MRR 0.964-0.977, stesso tetto
del vision): pipeline e mapping delle righe sono corretti. (2) A f=0.25 il self
esce dal top-100 nel **91%** delle query (`gcn/tau02`), 98% a f=0.5. (3) Le
metriche per-asse a f=0.25 restano sopra il floor casuale (gcn/tau02 nDCG@10
0.756/0.548/0.893 vs 0.739/0.466/0.869): recupera piante *simili*, non la propria.
(4) `hist` a f=0.0 non ritrova il self (0.026): istogrammi identici fra piante →
parità, atteso.

**Interpretazione (da verificare).** Un grafo con il 25% delle stanze tolte è il
grafo *completo* plausibile di molte altre piante più piccole: sul grafo il self
non ha nulla di speciale, mentre nel vision i pixel delle stanze superstiti sono
identici a quelli della pianta intera. Il criterio A.5 sul ramo graph è quindi
**quasi degenere** (AUC a 0.02, scala 40× sotto il vision): per costruzione
sceglie `gat/nosym`, ma va deciso se il self-recovery sia il metro giusto per un
ramo simbolico → decisione dell'utente prima di congelare (B.7).

### 30.1 B.6 — il criterio di selezione NON è il vincolo: lo è il tetto di 60 epoche (10 set)

Primi `head_probe_history.json` (8 config su 14, job 103475-78). **Fatto**: su
tutte, l'epoca scelta dalla probe è 57-60/60 e l'AUC della probe sale ancora
all'ultima epoca (es. `dinov3/natural` 0.7137 ep10 → 0.8405 ep50 → 0.8466 ep59).
Con init seedato, la val-loss sceglie ep 58-60 e l'AUC della probe lì differisce
di ≤0.006 dalla scelta della probe. ⇒ In questo regime val-loss e probe
**coincidono**: la selezione è censurata dal tetto `training.epochs=60`.
Diagnostico lanciato: `dinov3/natural` con 200 epoche, patience 8 sulla probe →
`head_probe_e200.pt` (job 103486; il 103484 era fallito all'avvio). ⚠️ AUC della probe ≠ AUC di valutazione
(query diverse per costruzione): il confronto con § 24 lo dà solo la run di eval.

### 30.2 B.6 — due censure del protocollo, e il fix (10 set)

- **Tetto epoche** (diagnostico `103486`, dinov3/natural, cap 200, patience 8):
  early stop a ep 89, scelta **ep 81**, probe AUC **0.8619** (cap 60: ep 59, 0.8466).
  Nella stessa run la val-loss avrebbe scelto ep 76 (probe AUC 0.8537).
- **Patience su un criterio non monotono** (ijepa): la probe AUC **scende** nelle
  prime epoche (0.0229 a ep 1) prima di risalire → patience 8 ferma il training a
  ep 9-10 con head peggiori del RAW. Verificato che la probe NON è rotta: su
  `ijepa/natural` dà RAW 0.0720 (eval §24: 0.0700) e `head.pt` 0.4343 (eval 0.4072).
- **Fix**: training ~0,4 s/epoca → `EPOCHS=300 PATIENCE=0`: si allena fino in fondo
  e si sceglie il massimo della probe; nella stessa run si salva anche la scelta
  della val-loss (`head_probe_e300_valloss.pt`) → criterio e numero di epoche si
  separano. Job `103528` (14 head). I `head_probe.pt` (cap 60) restano ma sono
  **censurati**: non vanno usati per B.7.

### 30.3 B.6 a 300 epoche (job 103528, 9/14 config lette): la head era SOTTO-ALLENATA, non scelta male

Probe AUC (1000 query valid disgiunte dall'eval) — scelta probe vs scelta val-loss **nella stessa run**:
dinov3/natural **0.8932** (ep 290) vs 0.8908 (ep 276) · dinov3/mean 0.8667 vs 0.8599 ·
dinov3/gem 0.8003 vs 0.7880 · dinov2/natural 0.8095 vs 0.8039 · siglip2/mean 0.7246 vs 0.7178.
**Fatti**: (1) epoca scelta 278-300 su 300 per tutte → ancora in crescita, ma lenta
(dinov3/natural +0.006 fra ep 250 e 300); (2) val-loss e probe scelgono epoche
vicine, differenza di AUC 0.002-0.012 a favore della probe; (3) tetto 60 → 300:
dinov3/natural 0.8466 → 0.8932. **Interpretazione**: nel training della head la
val-loss InfoNCE *non* è scollegata dal retrieval partial (entrambe monotone): il
limite di § 24 era il **numero di epoche**, il criterio conta poco. Da confermare
sulle query di eval: job `103529` (dinov3, 3 pooling, `head_probe_e300.pt`) e
`103530` (dinov3/natural, `…_valloss.pt`). Diagnostico plateau: `103531` (1000 ep).

### 30.4 B.6 — nessun plateau nemmeno a 1000 epoche (job 103528 completo, 103531)

e300 per le 5 config restanti: radio natural/mean/gem 0.7368/0.7397/0.6641, ijepa
natural/gem 0.6427/0.3888 (ijepa riparte dal dip: ep 50 0.322/0.060); scelta probe
vs val-loss sempre entro 0.009. **e1000 dinov3/natural**: scelta ep 969, probe AUC
**0.9272** (val-loss: ep 990, 0.9262); traiettoria 0.863 (ep100) → 0.892 (300) →
0.914 (700) → 0.923 (1000). **Interpretazione**: con batch 4096 e ~47k coppie di
train sono ~12 step di ottimizzazione per epoca → 60 epoche ≈ 720 step: il
«limite inferiore» di § 24.3 è un problema di **budget di ottimizzazione**, non di
criterio. Plateau cercato con `103535` (dinov3/natural, max 5000 ep, patience 100
→ `head_probe_conv.pt`). ⚠️ Più training = più specializzazione sul self-recovery:
il **costo sul full** (guardrail § 23) va misurato prima di congelare.

### 30.5 B.6 — plateau a ~1600 epoche (job 103535)

dinov3/natural, max 5000 ep, patience 100 sulla probe: early stop a ep 1694, scelta
**ep 1594, probe AUC 0.9461** (val-loss: ep 1690). Curva probe: 0.8466 (cap 60) →
0.8932 (300) → 0.9272 (969) → 0.9461 (1594). **Regola adottata per tutte le head
B.6**: max 5000 ep, patience 100 sulla probe → `head_probe_conv.pt` (+ `_valloss`).
Job: `103542` (dinov2, siglip2, radio, ijepa), `103543` (dinov3 gem/mean); eval
dinov3/natural `103540` (conv) e `103541` (conv_valloss). Le eval e300
(`103529-30`) sono state **cancellate** (superate): 4 file `*head-probe-e300*`
incompleti restano in `vision_partial_valid_B/` e non formano una config completa.

## 31. Protocollo B sul vision — primi numeri (10 set, job 103457 + parziali di 103456)

Gallery condivisa (n=67.405, sha1 `0c24cfc05e18`), `exclude_self=True`, whitening
`fit_split=train`. 11 config dinov3 complete in `results/perquery/vision_partial_valid_B/`.

| config | AUC | f=.25 | f=.50 | f=.75 |
|---|---|---|---|---|
| `dinov3/natural/head` (head.pt storico) | **0.8195** | 0.9406 | 0.8437 | 0.6742 |
| `dinov3/mean/head` | 0.7625 | 0.9224 | 0.7740 | 0.5912 |
| `dinov3/mean/head+whiten-train` | 0.7268 | | | |
| `dinov3/natural/head+whiten-train` | 0.7090 | | | |
| `dinov3/natural/whiten-train` | 0.5357 | 0.8591 | 0.5281 | 0.2199 |
| `dinov3/natural/raw` | 0.2510 | 0.5249 | 0.1713 | 0.0569 |

#1 − #2 = **+0.0569** [+0.0474, +0.0665] (§24: +0.0600): il verdetto regge sul protocollo B.
**Fatti sul confronto con § 24**: stesse 2000 query per nome, ma AUC per-query
identico solo nel 44% dei casi → 0.8157 → 0.8195. Due cause, entrambe attese: la
gallery perde 48 piante, e il seed di masking è `seed + qi` con `qi` che cambia col
riordino canonico ⇒ **vengono tolte stanze diverse**. I due protocolli non sono
appaiabili fra loro (e `robustness_auc` lo rifiuta: `exclude_self` diverso).
B.2 (whiten-train vs trasduttivo) aspetta `103456`/`103458`.

### 30.6 DECISIONE dell'utente su § 30 (10 set): **opzione D**

- **Metro invariato**: A.5 (AUC self-recovery su f ∈ {0.25,0.5,0.75}) resta per entrambi i rami.
- **Una sola variabile**: la forma delle coppie del training graph. Vista A = grafo
  intero (solo flip/rot), vista B = stanze rimosse f~U[0.25, 0.75] (rimosse davvero,
  come il partial di valutazione). Pooling, raw_skip, τ e sonda di selezione
  (nDCG full, topologia) **invariati** — la selezione non è partial, e va dichiarato.
- **Codice**: `pair_mode: symmetric|asym_partial` in `augment.py` (+ `--pair-mode`
  in `train_gnn.py`, chiave in `gcn.yaml`, default `symmetric` = storico bit per
  bit); variante `tau02asym` in `scripts/graph/03` e `04`; `tests/test_graph_asym_pairs.py`
  (5 test: fedeltà al grafo parziale di valutazione, conteggio stanze, default
  bit-identico, forward GCN; mutation check superato).
- **Run**: solo `gcn/tau02` (baseline § 30 AUC 0.0058). Job `103559` training →
  `103560` partial (script 06) → `103561` full valid; baseline full valid `103562`
  (`results/perquery/graph_full_valid/`). Gli altri 2 encoder solo se l'esito è positivo.
- **Previsione (scritta prima delle run)**: AUC partial sale; composizione sul full
  scende; topologia ≈ invariata.
- **Regola di uscita (decisa prima)**: D si adotta se il delta appaiato di AUC vs
  `gcn/tau02` di § 30 ha CI 95% che esclude lo zero. Altrimenti **B**: `gcn/tau02`
  congelata col criterio full e il crollo di § 30 riportato come risultato. Il
  pooling non si prova senza una nuova decisione.
- **Report**: premessa esplicita — senza questo training il partial del graph non
  funzionava (§ 30); e il self-recovery premia in parte l'"impronta esatta" (le
  stanze superstiti sono identiche all'originale).

### 31.1 B.2 CHIUSA — whitening train-only vs trasduttivo (job 103456 vs 103458)

Stesso protocollo B, stesse query e stesso masking (i `raw` delle due cartelle sono
**identici** per-query: controllo di determinismo superato). Delta AUC appaiato
train-only − trasduttivo: natural **+0.0003** [−0.0010, +0.0016] · gem **+0.0020**
[+0.0010, +0.0031] · mean **+0.0010** [−0.0001, +0.0020]. **Previsione (§25)
smentita nel verso, confermata nella sostanza**: attesi un piccolo calo, arriva
parità o un guadagno minimo ⇒ il guadagno del whitening **non era leakage**.
`fit_split: train` resta il default senza costo.

### 31.2 B.4 CHIUSA — quanto pesava il self nelle metriche per-asse del partial

nDCG@10 C/T/G, § 24 (self dentro) → protocollo B (self fuori), non appaiato (masking diverso):
`dinov3/natural/head` f=.25 0.860/0.706/0.945 → **0.822/0.619/0.931** · f=.75
0.826/0.643/0.931 → **0.796/0.574/0.920**; `…/whiten(-train)` f=.75 0.735/0.488/0.883
→ **0.721/0.465/0.881**. Il self gonfiava soprattutto la **topologia** (−0.087 a
f=0.25 per la head). ⚠️ Col floor casuale (0.739/0.466/0.869, § 20) il frozen+whiten
a f=0.75 sta **sotto il caso sulla composizione** (0.721) e al caso sulla topologia.

### 30.7 D — training `gcn/tau02asym` (job 103559): la sonda full sceglie un'epoca PRECOCE

Selezione invariata per decisione (nDCG@10 topologia su retrieval **full**, sonda
del valid): `tau02asym` best **0.7190 a ep 5**, early stop a ep 15; `tau02` era
0.8000 a ep 22. **Fatto**: con le coppie asimmetriche la topologia del *full*
peggiora durante il training, e la sonda ferma il modello presto. **Conseguenza
da dichiarare**: il checkpoint valutato è quello migliore per il full, non per il
partial → se l'AUC partial non sale abbastanza, una causa candidata è il criterio
di selezione (non provato: fuori dalla regola decisa). La previsione «topologia ≈
invariata» è a rischio già dalla sonda; la verifica è la full di `103561` vs `103562`.

### 30.8 D ADOTTATA — esito su `gcn/tau02` (job 103560 partial, 103561/103562 full)

**Regola di uscita (§30.6) soddisfatta**: delta AUC appaiato `tau02asym` − `tau02`
= **+0.0910** [+0.0844, +0.0976], n=2000 → **D si adotta**.

| | AUC | f=.25 | f=.50 | f=.75 | f=0.0 MRR |
|---|---|---|---|---|---|
| `gcn/tau02` (§30) | 0.0058 | 0.0153 | 0.0019 | 0.0001 | 0.9645 |
| `gcn/tau02asym` | **0.0968** | 0.2391 | 0.0447 | 0.0065 | 0.9645 |

**Costo sul full** (valid, gallery condivisa, nDCG@10 appaiato asym − tau02):
composizione **−0.0398** [−0.0419, −0.0377] (0.9343 vs 0.9741) · topologia
**−0.0868** [−0.0906, −0.0831] (0.7430 vs 0.8298) · geometria −0.0010 [−0.0020, −0.0002].

**Previsioni (scritte prima)**: AUC sale ✅ · composizione scende ✅ · topologia ≈
invariata ❌ **smentita** — è il costo più grande. Lettura: il graph compra
robustezza vendendo proprio l'asse su cui era forte (e resta sopra il floor 0.466).
Scala: 16× la baseline, ma ancora ~8× sotto il vision (0.8195, §31); il checkpoint
è scelto a ep 5 dalla sonda full (§30.7). Estensione (esito positivo): `sage/asym`
e `gat/asym` sul YAML `base` (una variabile; baseline §30 sage/base 0.0089, gat/base
0.0071): job 103597→98→99 (+ full base 103600), 103601→02→03 (+ full base 103604).

## 32. 11 set — esiti letti: D su sage/gat, head vision a convergenza (job 103535-43, 103597-604)

Tutti COMPLETED (exit 0). Metro A.5, valid, gallery condivisa `0c24cfc05e18`, n=2000
appaiate ovunque. Calcolo: `robustness_auc.load_auc/compare_auc` + nDCG@10 per-asse
appaiato dai `graph_full_valid/*_full_valid.npz` (bootstrap B=10000, seed 0).

**D estesa — la regola di uscita regge su tutti e 3 gli encoder** (asym − base, AUC):
gcn/tau02 +0.0910 [+0.0844, +0.0976] · sage +0.1016 [+0.0949, +0.1081] (0.1105 vs
0.0089) · **gat +0.2419 [+0.2328, +0.2506] (0.2490 vs 0.0071)**. f=0.0 MRR invariato
(0.965-0.976: pipeline ok). Classifica graph per AUC: gat/asym 0.2490 ≫ sage/asym
0.1105 > gcn/tau02asym 0.0968 ≫ tutte le config simmetriche (≤ 0.0241).

| costo full (asym − base, nDCG@10) | composizione | topologia | geometria |
|---|---|---|---|
| gcn/tau02 | −0.0398 (0.9343) | **−0.0868** (0.7430) | −0.0010 (0.9398) |
| sage | −0.0361 (0.9010) | −0.0434 (0.6870) | −0.0050 (0.9285) |
| gat | **−0.0504** (0.8776) | −0.0240 (0.6757) | **+0.0046** (0.9427) |

Tutti i CI escludono lo zero. **Lettura**: gat/asym è il più robusto (2.6× sage) e
paga meno topologia, ma parte già da una topologia full più bassa (gat/base 0.6997
vs gcn/tau02 0.8298) e ha la composizione full più bassa. Candidato graph per B.7
secondo il metro: **gat/asym** (costo sul full da dichiarare accanto). ⚠️ sage/gat
asym usano il YAML `base` e la sonda full storica: epoche scelte non lette.

**Head vision a convergenza** (max 5000 ep, patience 100 sulla probe): tutte e 14
ferme per patience (nessuna al tetto) → plateau raggiunto. Best ep 565-3091.
Probe AUC (1000 query valid di SELEZIONE, ottimista): dinov3/natural 0.9461 ·
dinov2/natural 0.9016 · dinov3/mean 0.8890 · radio/natural 0.8754 · dinov3/gem 0.8601 ·
dinov2/mean 0.8541 · radio/mean 0.8345 · ijepa/natural 0.8269 · dinov2/gem 0.8060 ·
siglip2/mean 0.7583 · radio/gem 0.7256 · ijepa/gem 0.6441 · siglip2/natural 0.6107 ·
siglip2/gem 0.5483. Checkpoint val-loss della stessa run: probe −0.001/−0.006.

**Protocollo B, dinov3/natural** (2000 query di valutazione, disgiunte dalla probe):
`head-probe-conv` **0.9173** vs `head` storica 0.8195 → **+0.0978 [+0.0906, +0.1053]**.
Criterio: conv − conv_valloss = −0.0003 [−0.0025, +0.0021] (**pari**) → conferma
§30.3-30.5: conta il budget di epoche, non il criterio. Whitening sopra la head
peggiora (conv+whiten 0.8278; storica+whiten 0.7090) → la config resta **senza**
whitening. ⚠️ **Manca il costo full della head nuova** (guardrail §23): nessuna
eval full lanciata; serve prima di congelarla.

### 32.1 Passo 3 della roadmap SUPERATO — controllo di rango §24 ↔ protocollo B (11 set, CPU)

12 config dinov3 (3 pooling × raw/whiten/head/head+whiten; `whiten` ↔ `whiten-train`)
in `vision_partial_valid` (§24, gallery 67.453) e `vision_partial_valid_B` (67.405).
Classifica per AUC **identica** (rango 1-12 uguale); 66 coppie, **0 inversioni**.
Criterio del passo 3 soddisfatto → T2 frozen si riporta da §24 **col caveat
dichiarato**, **nessuna ri-griglia frozen**. Si valutano solo le head `conv`
(job 103957-61 partial, 103955-56 full su `results/perquery/vision_full_valid_B`).

### 32.2 Costo full della head nuova (job 103955 + parte di 103956): NESSUN costo

Full valid, protocollo B (gallery `0c24cfc05e18`, n=2000, stesse query), nDCG@10
dinov3/natural C/T/G: raw 0.8215/0.6409/0.9328 · whiten-train 0.8250/0.6627/0.9347 ·
`head` storica 0.8288/0.6419/0.9354 · **`head-probe-conv` 0.8293/0.6470/0.9397** ·
conv+whiten 0.8306/0.6600/0.9367. Appaiati (bootstrap B=10000):
conv − storica: C +0.0004 [−0.0012, +0.0022] · T **+0.0051** [+0.0028, +0.0074] ·
G **+0.0043** [+0.0037, +0.0049] → guardrail §23 superato (non costa, anzi).
conv − whiten-train (frozen migliore in topologia): T −0.0157 [−0.0193, −0.0122] →
da dichiarare accanto. **Config vision**: `dinov3/natural/head-probe-conv` senza
whitening, **congelata salvo** che una head conv degli altri encoder (103957-61)
la superi in AUC con CI che esclude 0.

## 33. 11 set — config vision CONGELATA + miglior frozen per encoder (CPU, `robustness_auc`)

**Head conv degli altri encoder (job 103956-61, protocollo B)**: nessuna supera la
vincente. Classifica: `dinov3/natural/head-probe-conv` **0.9173** · dinov2/natural 0.8783 ·
dinov3/mean 0.8687 · radio/natural 0.8517 · ijepa/natural 0.7814 · siglip2/mean 0.7460.
#1 − #2 = **+0.0389 [+0.0314, +0.0466]**. `…-valloss` 0.9175 (criterio pari, §32).
⇒ **Config vision congelata: `dinov3/natural/head-probe-conv`, senza whitening**
(il whitening sopra la head peggiora su tutti gli encoder).

**Miglior frozen (raw|whiten) per encoder** — da `vision_partial_valid/` (§24, protocollo
vecchio; legittimo per §32.1 + §31.1, estensione agli altri encoder **assunta**).
Spareggio §23 (CI dell'AUC che contiene 0 → nDCG@10 full di topologia, `vision_valid/`):

| encoder | config | AUC | nota |
|---|---|---|---|
| pespatial | **gem/whiten** | 0.6346 | pari con natural (+0.0013 [−0.0064, +0.0092]) → topologia full 0.6301 vs 0.6113 ⚠️ §24 indicava `natural`: per la regola vince `gem` |
| dinov2 | natural/whiten | 0.5796 | vs gem +0.0140 [+0.0056, +0.0224] |
| radio | natural/whiten | 0.5781 | pari con gem (+0.0004 [−0.0058, +0.0066]) → topologia full natural +0.0032 |
| dinov3 | mean/whiten | 0.5619 | vs gem +0.0085 [+0.0028, +0.0144] |
| ijepa | natural/whiten | 0.4831 | pari con gem (−0.0063 [−0.0127, +0.0000]) → topologia full natural +0.0055 |
| siglip2 | mean/whiten | 0.3411 | vs gem +0.0308 [+0.0247, +0.0372] |
| pecore | mean/whiten | 0.3097 | vs gem +0.0060 [+0.0004, +0.0116] |
| tipsv2 | natural448/whiten | 0.3042 | vs natural224 +0.0111 [+0.0023, +0.0198] |

Servono alle run di crop/patch (roadmap passo 2). Il «miglior frozen in assoluto» resta
`pespatial` (le due pooling sono pari in AUC): cambia l'etichetta, non il claim.

## 34. PRE-REGISTRAZIONE — crop e patch sul vision (11 set, PRIMA di ogni job)

Codice: `src/vision/data/vision_damage.py` (+ `damage_curves.py`, `robustness_auc --strategy`,
campo per-query `area_removed`); suite 152 passed; `vision_partial_query.py` invariato
(sha1 `852d0538`, 21 hash dorati). Cartella: `results/perquery/vision_damage_valid_B/`.
Config: `dinov3/natural/head-probe-conv` + `dinov3/natural/whiten-train` (baseline di H1,
stesso encoder e pooling) + gli 8 miglior frozen di §33.

- **H1 — la head si trasferisce.** Per d ∈ {crop, patch}: Δ_d = AUC_d(head-probe-conv) −
  AUC_d(dinov3/natural/whiten-train). CI 95% con estremo inferiore > 0 → «regge» (se in più
  Δ ≥ 0.02 → «in modo rilevante»); CI che contiene 0 o Δ < 0 → la head è **specifica** del
  danno a stanze (H1 falsificata su d). Attesa: Δ_crop > Δ_patch > 0 (il crop somiglia a un
  blocco di stanze tolte, la griglia di patch no).
- **H2 — a parità di area conta l'informazione o l'artefatto?** Su f=0.75, delta appaiato
  per query self_rr(patch) − self_rr(crop) sugli 8 frozen: ≥ 6/8 > 0 con CI che esclude 0 →
  **informazione** (i buchi sparsi lasciano un pezzo di ogni stanza); ≥ 6/8 < 0 →
  **artefatto** (la griglia domina l'embedding); altrimenti «dipende dall'encoder». Per-asse
  (nDCG@10, self fuori): attese composizione e geometria patch ≥ crop; topologia |Δ| < 0.02.
- **Stanze vs crop/patch**: finestre di area tolta [0.20,0.30], [0.45,0.55], [0.70,0.80],
  **solo descrittive** (non appaiato: le query a stanze si selezionano sull'area ottenuta).
- **Controllo prima dell'ondata 2**: i file `random` dei primi 2 job devono essere
  `np.array_equal` (self_rr, ndcg, ret_rows) a quelli in `vision_partial_valid_B/`; se no, stop.
- Rischi dichiarati: patch appaiata solo a parità di griglia (P14 vs P16, tipsv2 a 448);
  area misurata sull'immagine nativa (stanze, crop) vs ridimensionata (patch).

## 35. SCOPERTA (11 set, verificata) — il «bordo aperto» del vision NON cancella i muri interni

**Fatto.** `render_partial_image` cancella solo i pixel-muro con tutti i canali ≤ `_WALL_MAX = 120`
(`vision_partial_query.py:30,111-112,183`). Nelle PNG RPLAN i muri **esterni** sono (79,79,79), quelli
**interni** (128,128,128) (5/5 PNG controllate in sessione; 100/100 dall'`architect`). ⇒ i muri interni
non vengono **mai** cancellati: la stanza tolta diventa una **cella bianca contornata dai muri grigi**,
con un buco solo nel muro esterno (verificato a vista: `scratchpad/ob_10_rm3.png`, e nei pannelli
«rooms» degli esempi di crop/patch). Si perde il **colore** (= tipo) della stanza, **non** la sua forma né
la sua posizione. Contraddice l'intento di progetto (memoria/`retrieval.md`: «cancellare anche i muri
condivisi con le stanze tenute; la sagoma non trapela»): è un **bug rispetto al design**.

**Cosa NON cambia**: nessun numero a disco (§24, §31-§34); la head è allenata e selezionata con lo
stesso render (`projection_pairs.py`, `retrieval_probe.py`), quindi è coerente con ciò che misura.
**Cosa cambia**: l'**interpretazione**. Il partial del vision misura «stanze **svuotate**» (tipo perso,
geometria visibile), non «stanze tolte». Il graph invece toglie davvero nodo e archi → il confronto
vision↔graph sotto masking (0.9173 vs 0.2490) **non è alla pari**: una parte del divario è protocollo.
Crop e patch (§34) cancellano tutto nel rettangolo/patch, muri inclusi: non hanno il problema.

**Corrispondenze a informazione pari** (proposta, da decidere con l'utente):
vision attuale «stanze svuotate» ↔ graph **nodo-fantasma** (tipo azzerato, geometria e archi tenuti) ·
vision «stanze tolte davvero» (muri interni cancellati, lati aperti) ↔ graph **contatore dei
collegamenti persi** (piano `architect` 11 set, variante `asymlost`) · graph attuale (nessuna traccia)
↔ nessuna delle due.

## 36. Modalità `nowalls` — la stanza tolta perde anche i muri (11 set, codice + CPU)

**Risposta a §35**, lato vision. Nuova famiglia di danno `nowalls_{random,semantic,topology}`
(`src/vision/data/vision_damage.py:100-160`): **stessa** selezione di stanze e **stesso** riempimento
delle omonime storiche, cambia solo la regola del muro nel secondo passo — da «scuro ≤ 120» a
**grigio neutro non-sfondo** (`neutral_wall_mask`: `max-min ≤ 25` e `max ≤ 220`), che prende 79, 128 e
gli anti-alias intermedi e lascia fuori ogni colore-stanza (il più pallido, (244,242,229), è sopra 220).
Raggio di cancellazione **invariato** (`_WALL_ERASE = 9`): una sola variabile cambia.

**Misurato in sessione** (30 piante `snapshot_train`, 207 stanze tolte una alla volta):
celle bianche ancora **chiuse** dai muri 50% (103/207) → **5%** (11/207); area di pianta rimossa
sempre ≥ di quella storica (mediana +0.012). Il 4-5% residuo non è diagnosticato.

**Appaiamento**: `rng = Random(seed + qi)` è per query, non per run ⇒ a parità di `f` la run storica e
la `nowalls` tolgono **le stesse stanze** (test `test_nowalls_and_historical_remove_the_same_rooms`).
Etichette/slug distinti (`partial-nowalls-random-f0.5`) ⇒ nessuna collisione con i file già a disco;
le 21 golden hash del modulo stanze restano verdi (146 test passati, nessun numero a disco toccato).
Curve e AUC: `--strategy nowalls-random` in `robustness_auc.py` e `damage_curves.py`.

**Limite dichiarato, non aggirabile sull'immagine**: la stanza interna resta comunque un **buco bianco
riconoscibile**, perché i colori delle stanze vicine finiscono dove finiva lei. `nowalls` toglie **tipo +
muri**, lascia la **forma del buco** — la stessa informazione che lasciano crop e patch, quindi le tre
modalità diventano confrontabili fra loro. Per togliere anche la forma servirebbe **ridisegnare** la
pianta dal `.mat`: sarebbe una pianta diversa, non una pianta incompleta → scartato.

**Spento di default** nel YAML: acceso, si accoda dopo crop/patch.

## 37. RISULTATI crop + patch + nowalls (11-12 set, job dell'utente) — **H1 FALSIFICATA**, i muri portavano quasi tutta la robustezza

Cartella `results/perquery/vision_damage_valid_B/`, 26 file (13 run × 2 config), `fallback=0`
ovunque, gallery `0c24cfc05e18` su tutte. Solo l'**ondata 1** (config congelata + baseline
frozen dello stesso encoder/pooling); gli 8 miglior frozen (ondata 2) **non** sono stati lanciati.

**AUC self-recovery (f = 0.25/0.5/0.75), le stesse 2000 query ovunque**

| danno | `head-probe-conv` | `whiten-train` | Δ (head − frozen) |
|---|---|---|---|
| stanze **svuotate** (storico) | **0.9169** | 0.5357 | +0.3812 [+0.3687, +0.3938] |
| stanze **tolte** (`nowalls`) | 0.3942 | 0.3040 | +0.0902 [+0.0807, +0.0999] |
| crop | 0.2017 | **0.3846** | **−0.1830** [−0.1919, −0.1741] |
| patch | 0.2785 | **0.4427** | **−0.1641** [−0.1720, −0.1564] |

**H1 («la head si trasferisce») — FALSIFICATA su entrambi i danni.** Il criterio pre-registrato
(§34) chiedeva CI con estremo inferiore > 0: qui il Δ è **negativo** e significativo su crop e
patch → la head è **specifica del danno a stanze**. Sbagliata anche l'attesa d'ordine
(Δ_crop > Δ_patch): è Δ_crop < Δ_patch, entrambi < 0.

**Quanto valevano i muri rimasti** (delta appaiato, stesse query e stesse stanze tolte, unica
differenza i muri; `damage_curves --delta random nowalls-random`, self_rr):
head +0.2167 / +0.6011 / +0.7504 a f=0.25/0.5/0.75 — a f=0.75 il self-recovery passa da 0.840 a
0.036. Frozen: +0.1735 / +0.3361 / ≈+0.185. ⇒ la robustezza del vision sotto masking a stanze
era **quasi tutta artefatto del rendering**, non capacità del modello.

**Confronto coi rami a informazione più vicina** (AUC, stessa gallery, stesso `exclude_self`):
vision head `nowalls` **0.3942** vs graph `gat/asym` **0.2490** (letto in sessione da
`results/perquery/graph_partial_valid`). Il divario scende da +0.67 a **+0.14**. Resta un
residuo di protocollo: il graph perde nodo **e** archi, il `nowalls` lascia la **forma** del buco.

**H2 — non decidibile.** Il livello pre-registrato è f=0.75, dove entrambe le config sono al
pavimento (self_rr 0.003-0.031): patch − crop = −0.0012 [−0.0040, +0.0015] (head) e −0.0222
[−0.0289, −0.0158] (frozen), cioè 0/2 positivi su 2 config invece delle 8 richieste. A danno
moderato il segno è opposto e netto (f=0.25: +0.2054 e +0.1094) → coerente con
«**informazione**», ma la regola pre-registrata resta senza verdetto finché manca l'ondata 2.

⚠️ **Il controllo pre-registrato di bit-identità NON passa alla lettera.** I file `random` della
cartella nuova non sono `np.array_equal` a quelli di `vision_partial_valid_B`. Entità misurata
(f=0.5, head): `names` e `qi` identici, `self_rr` diverso su **6/2000** query (tutte flip 1.0↔0.5,
cioè scambi di rango 1↔2), media −0.0005; `ret_rows` diverso su 154/2000 righe, di cui **149 con
lo stesso insieme di vicini in ordine diverso** e 5 con un solo elemento diverso in coda
(posizione ~99/100); `ndcg` max 0.001. **Interpretazione (ipotesi, non verificata)**: non
determinismo float della forward su GPU che ribalta i pareggi fra piante duplicate — RPLAN ne ha
(tetto f=0.0, §24). Il danno è lo stesso: `partial_label`, `query_seed`, `gallery` e `num_queries`
coincidono. Le conclusioni qui sopra sono 2-3 ordini di grandezza più grandi di questo rumore.
**Decisione dell'utente** se considerarlo passato.

## 38. PRE-REGISTRAZIONE — il metro di robustezza cambia (14 set, decisione utente, PRIMA dell'ondata 2)

**Decisione dell'utente (14 set)**: (0) il controllo di bit-identità di §37 si considera
**rumore di fondo** e passato; (1) **nuovo metro** di robustezza, sotto.

**Il metro.** Per ogni configurazione:

    R(q) = media di AUC_nowalls-random(q), AUC_crop(q), AUC_patch(q)     (pesi uguali)
    R    = media di R(q) sulle query presenti in tutti e tre i danni

dove ogni AUC è quella di §23.1 invariata: self-recovery MRR, f ∈ {0.25, 0.5, 0.75}, pesi
uguali, valid, `exclude_self=True`, gallery condivisa. Strumento:
`python -m src.evaluation.robustness_auc rank|compare --robust` (`ROBUST_STRATEGIES`).

**Perché questi tre**: sono i danni che tolgono davvero informazione. Il `random` storico **esce**
dal metro: lasciava i muri della stanza tolta (§35) e da solo portava quasi tutta la robustezza
misurata (§37). Tre danni diversi, stesso peso: una config non vince sfruttando un danno solo.

**Invariato da §23.1**: assoluto e non relativo (punto 4); delta appaiato con CI 95% bootstrap
(punto 5); **spareggio** (punto 6): CI che contiene lo zero → nDCG@10 **full** sull'asse debole
(topologia) → poi la config più economica. Guardrail: costo sulla pianta intera sempre accanto.
**Aggiunto**: accanto a R si riportano **le tre AUC separate** (la classifica le stampa).

**Dichiarato — fissato DOPO aver visto 2 config** (ondata 1, entrambe dinov3/natural). Col
metro nuovo: `whiten-train` **0.3771** vs `head-probe-conv` **0.2915**, Δ **+0.0857**
[+0.0803, +0.0910] (letto in sessione, `rank --robust` su `vision_damage_valid_B`). ⇒ la config
vision oggi congelata **perde** contro la sua baseline frozen. Nessuna delle config ancora da
misurare era nota quando il metro è stato fissato.

**Limiti dichiarati**: (a) il livello f non è la stessa quantità nei tre danni — per crop/patch è
superficie, per `nowalls` frazione di stanze (area media ottenuta 0.29/0.52/0.71, §37), quindi il
peso uguale è sui **livelli nominali**; (b) la griglia di patch cambia con l'encoder (P14 vs P16,
tipsv2 a 448): il danno patch non è identico fra encoder, `compare --robust` lo segnala.

**Ondata 2 — decisione utente (14 set): griglia frozen COMPLETA, nessuna estrapolazione.**
Scartata la regola «controllo su dinov3 → si estende agli altri»: si valutano tutte le **52**
config frozen (8 encoder × pooling × risoluzione × raw/whiten, `scripts/vision/_common.sh`),
51 nuove + `dinov3/natural/whiten-train` dell'ondata 1. 29 job (`COMANDI.md § Ondata 2 onesta`),
verificati con dry-run a python sostituito: 51 valutazioni uniche, head spenta, `random` spento,
`nowalls`/crop/patch accesi, batch 32 a 448. Costo: ~70 min per valutazione misurato su
dinov3@224 (mtime ondata 1) → ~60 ore-GPU; tipsv2@448 **non misurato**, più lento.

**Cosa decide, fissato ora**:
- **Config vision congelata nuova** = #1 per R (sopra) fra **tutte** le config in
  `vision_damage_valid_B` con i tre danni completi — le 52 frozen + `head-probe-conv`. Parità
  (CI che contiene 0) → spareggio di §23.1 punto 6.
- Il **miglior frozen per encoder** di §33 è **superato**: si rilegge dalla stessa classifica.
- Le head degli altri encoder restano **fuori** (allenate col render vecchio, §37); dichiarato.
- **H2 (§34)** si calcola come pre-registrato, sugli 8 config di §33, tutti inclusi nella griglia.
- Le affermazioni di §33 su raw < whiten e sui pareggi di pooling valgono solo col metro vecchio:
  la griglia le riverifica, non si assumono.

## 39. 14 set — ondata 2 in volo, visualizzazioni sui danni onesti, semantic/topology misurati (CPU)

- **Ondata 2 lanciata** (griglia §38): divisa fra i 3 utenti (squeue ~15:55) — gangelillis
  `106348-50` dinov2 · ggermini 11 job · edimaria 12 job; mancano i 3 di dinov3 (per differenza).
- **Verificato sui file dei job veri**, non solo col dry-run: 8 config nuove hanno scritto 18
  crop + 9 patch e **zero** `random`/`semantic`/`topology`. `partial_runs` esegue random →
  semantic → topology → crop → patch → nowalls: i danni vecchi sono stati saltati.
- **Visualizzazioni** (`retrieval_visualization.py`): il partial passa da `damaged_query(...,
  return_image=True)` → crop, patch e nowalls disegnati con l'immagine effettivamente codificata
  (patch: tela 224). `info["removed"]` per stanze/nowalls; default di `damaged_query` invariato.
  167 test verdi, 21 golden intatte. **Non verificato** il rendering reale (GPU): job 106381-82.
- **semantic / topology** (300 piante `snapshot_train`, `select_rooms_to_remove`): `semantic`
  (tiene Living/Kitchen/Bathroom) toglie in media il **54%** delle stanze [25%, 75%], mai zero;
  `topology` (foglie, grado ≤ 1) il **7%** in media e **nessuna stanza nel 61%** (184/300) → quasi
  sempre query intatta. Senza livelli → fuori dal metro §38. `nowalls_semantic` resta candidato
  per il controllo di trasferimento del graph, solo sulla config vincente.

## 40. PRE-REGISTRAZIONE — il graph regge un danno diverso da quello del training? (14 set, PRIMA dei job)

**Perché.** `asym` si allena togliendo stanze **a caso** (`augment.py:303`, f~U[0.25,0.75]) e si
misura con lo stesso danno → la robustezza di §32 potrebbe essere specifica di quel danno.
Decisione utente (14 set): fase 1 del piano graph, solo valutazione, in parallelo all'ondata 2.

**Danno.** `semantic` di `graph_evaluate` coi default: tiene LivingRoom/Kitchen/Bathroom (tipi 0,2,3),
toglie **tutte** le altre; deterministico, nessun seed. Su 300 piante toglie in media il 54% [25%, 75%]
(§39, selezione condivisa col vision). ⚠️ Resta rimozione di stanze: cambia **quali** (per tipo, non a
caso) e la frazione non è controllata → cambio di distribuzione **moderato**, dichiarato così.

**Run.** `gat/asym` e `gat/base`, valid, stesse 2000 query (seed 42), gallery condivisa,
`exclude_self`, endpoint `self_rr`. Nessun codice nuovo: `GRAPH_EXTRA_FLAGS="--partial-strategies
semantic"` (`04_eval_gnn.sh:170`, argparse fa vincere l'ultimo). Verificato in sessione: dry-run di 06
con python sostituito (checkpoint trovati, flag in coda) + `parse_args` reale → sola run `semantic`.
Output: `results/perquery/graph_partial_valid/graph_gat_{asym,base}_partial-semantic_valid.npz`.

**Regola (decide).** Δ = media per query di self_rr(asym) − self_rr(base) sul `semantic`, appaiato,
CI 95% bootstrap (B=10000, seed 0, come §32). **Si trasferisce** se l'estremo inferiore > 0;
altrimenti la robustezza di `asym` è **specifica del danno allenato** e il report la qualifica così.
**Non decide la config graph** (tutti i candidati si allenano su stanze tolte a caso): cambia solo
come si scrive il claim. Si ripete su `asymlost` se adottata.

**Descrittivo (non decide).** Stesso Δ su `random f=0.5` (file già a disco, query in comune) e
rapporto Δ_semantic / Δ_random; conteggi «senza masking» / «svuotate» dal log. I file `random` vengono
da job più vecchi: rumore GPU dell'ordine di §37, irrilevante per un confronto descrittivo.

**Previsione (prima dei job).** Δ > 0 con CI sopra lo zero (le stanze superstiti restano identiche
all'originale: l'«impronta esatta» aiuta con qualunque scelta delle stanze), ma Δ_semantic <
Δ_random f=0.5: il `semantic` lascia quasi sempre soggiorno + cucina + bagno, un sottoinsieme molto
meno vario di quelli visti in training e comune a quasi tutte le piante.

## 41. PRE-REGISTRAZIONE — graph «bordo aperto» `asymlost` + replica `asymrep` (14 set, PRIMA dei job)

**Decisioni utente (14 set)**: design `architect` approvato; (1) **replica** di `gat/asym` per stimare il
rumore fra training; (2) se `asymlost` non passa **ci si ferma**: `gat/asym` congelata, esito negativo
riportato, **nessuna** altra forma del marcatore (sarebbe selezione a posteriori).

**Variabile unica**: `--lost-marker` = 20ª feature di ogni stanza rimasta, numero di vicini **distinti**
tolti (grezzo), calcolato dalla stessa funzione in training e valutazione; 0 sui grafi interi (gallery,
vista A). Conseguenza necessaria: `in_dim` 19→20. Il resto = `gat/asym` (YAML `gat`, `asym_partial`,
seed, epoche, patience, sonda). Scartate: binaria (meno di `nowalls`: perde quanti lati sono aperti) ·
per tipo di relazione (`to_undirected` copia lo stesso codice sui due versi, letto nel sorgente PyG →
nessuna direzione per nodo; +10 dim = seconda variabile) · normalizzata sul grado (stessa informazione).

**Previsioni (scritte prima)**:
- **P1 (decide)**: AUC self-recovery `asymlost` > `asym`. Meccanismo: col marcatore il grafo parziale di
  una pianta non passa più per il grafo completo di un'altra (§30).
- **P2 (descrittiva)**: guadagno più grande a f=0.25/0.5 che a f=0.75 (poche stanze → poca traccia).
- **P3 (descrittiva, costo full)**: nDCG@10 `asymlost − asym` topologia ≥ 0, composizione ≥ 0, geometria
  ≈ 0 — la vista parziale si separa dalle altre piante senza sacrificare la topologia dei grafi interi
  (§32). Topologia con CI sotto zero ⇒ meccanismo **smentito** anche se P1 passa (guadagno da «impronta»).
- **Controllo pipeline**: MRR a f=0.0 nell'intervallo di §32 (0.965-0.976).

**Regola (roadmap §2 punto 2b, invariata)**: Δ AUC appaiato `asymlost − asym` (f ∈ {0.25,0.5,0.75},
valid, gallery condivisa, `exclude_self`, 2000 query), CI 95% bootstrap (B=10000, seed 0): **adottata** se
l'estremo inferiore > 0; costo nDCG@10 per asse sul full sempre accanto.

**Replica `asymrep` (descrittiva, NON cambia la regola)**: stessi flag di `asym`. Inizializzazione e
shuffle usano l'RNG globale non seedato (verificato: `train_gnn.py:150` senza generator; `torch.rand`
diverso fra due processi) ⇒ due training uguali differiscono, e per il graph non è mai stato misurato.
Si riporta Δ AUC `asymrep − asym`; se |Δ_rep| ≥ Δ_lost il guadagno si dichiara **dentro la variabilità fra
training** anche con CI > 0. ⚠️ Una replica stima l'ordine di grandezza del rumore, non la sua varianza.

**Da dichiarare**: sonda di selezione invariata e **cieca** al marcatore (vede solo grafi interi, §30.7) ·
**circolarità** (vincolo 5): il marcatore deriva da `rEdge` come la GT di topologia → metriche per-asse
del partial favorite per costruzione · il graph non riceve posizione né estensione della stanza tolta →
resta sotto l'informazione di `nowalls`, confronto col vision solo descrittivo · se adottata si ripete
il controllo §40 su `asymlost` · **seconda via del marcatore** (final-reviewer, verificato
`models/base.py:188-191`): `raw_skip` (acceso nel YAML `gat`) somma `data.x` intero, colonna 20 inclusa →
la **somma dei marcatori** (collegamenti tagliati; 0 in gallery) arriva a `proj` senza passare dalla GNN.
Un guadagno di P1 non distingue le due vie (l'ablation che le separa è fuori roadmap): si dichiara.

**Codice (14 set, CPU)**: `--lost-marker` in `transforms.py` / `augment.py` / `graph_partial_query.py` /
`train_gnn.py` / `graph_evaluate.py`, varianti in `03`/`04`; `tests/test_graph_lost_marker.py` 18 test
(mutation 6/6), suite 186 verdi; final-reviewer: approvato, default bit-identico, conteggio
train↔eval verificato su 400 grafi casuali. Non verificato: GPU, `.mat` reali, cache `graphs.pt`.

## 42. RISULTATI §40-§41 (14 set, job 106435-36, 106478-82) — `semantic` ✅ si trasferisce · `asymlost` ❌ FALSIFICATA · rumore fra training misurato

Tutti COMPLETED exit 0; 106435 = `gat/asym`, 106436 = `gat/base` (log). 2000 query appaiate ovunque, meta
compatibili (guardie di `significance`/`robustness_auc`). Calcolo in sessione: `robustness_auc compare`,
`significance --metric self_rr` e `--k 10`, bootstrap B=10000 seed 0.

**§40 — la robustezza di `asym` SI TRASFERISCE al danno `semantic`.** Δ self_rr `asym − base` =
**+0.1100 [+0.0994, +0.1210]** (0.1121 vs 0.0021); query senza masking 0, svuotate 0 (log). Previsioni ✅
Δ > 0 · ✅ Δ_semantic < Δ_random f=0.5 = +0.1500 [+0.1377, +0.1626] (0.1510 vs 0.0010), rapporto 0.73.
⚠️ Frazione tolta dal `semantic` non controllata (54% medio su 300 piante di §39, non su queste query):
il confronto col f=0.5 è indicativo.

**§41 — P1 FALSIFICATA → `asymlost` NON adottata → `gat/asym` CONGELATA** (decisione utente: nessuna altra
forma del marcatore).

| self_rr | AUC | f=0.25 | f=0.5 | f=0.75 | f=0.0 |
|---|---|---|---|---|---|
| `asym` | 0.2490 | 0.5706 | 0.1510 | 0.0254 | 0.9752 |
| `asymlost` | 0.1153 | 0.2728 | 0.0627 | 0.0104 | 0.9781 |
| `asymrep` | 0.2775 | – | – | – | 0.9749 |

Δ AUC `asymlost − asym` **−0.1337 [−0.1422, −0.1250]**; per f: −0.2977 [−0.3157, −0.2791] · −0.0883
[−0.1006, −0.0759] · −0.0150 [−0.0202, −0.0099] → P2 ❌ (a f bassa la **perdita**, non il guadagno, è massima).
**Full** (nDCG@10 `asymlost − asym`): composizione **+0.0540** [+0.0514, +0.0566] (0.9316 vs 0.8776) ·
topologia **+0.0431** [+0.0392, +0.0471] (0.7189 vs 0.6757) · geometria −0.0038 [−0.0048, −0.0028]. P3 ✅ nel
segno, ma senza P1 il meccanismo cade: il marcatore **ricompra il full rivendendo la robustezza**.
**Pipeline**: f=0.0 `asymlost` 0.9781, appena sopra l'intervallo di §32; Δ vs `asym` +0.0029 [−0.0007,
+0.0066], non significativo → ok.

**Confound dichiarato (non si risolve con un'altra run: roadmap §4 regola 3).** La sonda full ha scelto
**epoca 6** per `asymlost` contro 31 (`asym`) e 32 (`asymrep`) (`training_summary.json`; sonda topologia
0.7055 vs 0.6620 / 0.6671): col marcatore la topologia del full sale subito e l'early stopping ferma il
training dopo ~5× meno epoche di coppie parziali. Stesso schema di §30.7. *Ipotesi non verificata*: parte
della perdita è l'epoca scelta. Claim ammesso: «marcatore **+ selezione sul full** → meno robusto», non
«il marcatore peggiora la robustezza».

**Rumore fra training (`asymrep`).** Δ AUC `asymrep − asym` **+0.0285 [+0.0220, +0.0352]**: il CI esclude
lo zero con la **stessa** config ⇒ (1) il bootstrap sulle query **non** contiene la varianza fra training;
(2) |Δ_lost| ≈ 4.7× |Δ_rep| → la perdita di `asymlost` è fuori da questo rumore; (3) fra config graph
allenate, delta di AUC dell'ordine di 0.03 **non sono distinguibili** da un riallenamento (tocca sage/asym vs
gcn/tau02asym, 0.1105 vs 0.0968 di §32; non gat vs gli altri). Una replica = ordine di grandezza, non
varianza. Congelata `asym` come pre-registrato, **non** `asymrep` (sceglierla perché più alta = scegliere sul
rumore).

**Run (utente, dopo i test)**: `asymlost` training → partial valid (06) → full valid
(`04_perquery_graph.sh`, `PERQUERY_OUT=results/perquery/graph_full_valid`), in catena: 06 e 04 riscrivono
`embeddings.npy`/`names.json` della variante. `asymrep`: training → partial valid.

## 43. RISULTATI ondata 2 (15 set) — 41 config su 53, vincitore PROVVISORIO `pespatial/gem/whiten`, H2 → «artefatto»

Cartella `vision_damage_valid_B`, metro §38 (`rank --robust`), gallery `0c24cfc05e18`, n=2000 ovunque.
**Complete 41 su 53** (52 frozen + head; il «54» dato in §39/TODO era un errore di conto: dinov3/natural/
whiten è una delle 52). **Mancano tutte le 12 di tipsv2**: zero file, nessun job in coda, nessun file altrove.
Embedding presenti per tutte e 6 le varianti; nel codice `TIPSv2Encoder` usa `compose_transform` ed
espone `model.config.patch_size` (preset 14) → il contesto patch **non** spiega il fallimento. Causa
**non nota**: servono i log (job id da chiedere agli utenti).

**Classifica (prime 10 di 41)**

| # | config | R | nowalls | crop | patch |
|---|---|---|---|---|---|
| 1 | pespatial/gem/whiten | **0.5245** | 0.3920 | 0.6605 | 0.5209 |
| 2 | radio/natural/whiten | 0.5161 | 0.3821 | 0.6032 | 0.5631 |
| 3 | pespatial/natural/whiten | 0.5148 | 0.3707 | 0.6855 | 0.4881 |
| 4 | radio/gem/whiten | 0.5077 | 0.3809 | 0.6168 | 0.5254 |
| 5 | radio/mean/whiten | 0.4961 | 0.3609 | 0.6056 | 0.5218 |
| 6 | pespatial/mean/whiten | 0.4799 | 0.3382 | 0.6216 | 0.4799 |
| 7 | ijepa/natural/whiten | 0.4259 | 0.2234 | 0.5108 | 0.5435 |
| 8 | ijepa/gem/whiten | 0.4067 | 0.2277 | 0.4907 | 0.5016 |
| 9 | dinov3/mean/whiten | 0.3855 | 0.2798 | 0.4638 | 0.4129 |
| 10 | dinov2/natural/whiten | 0.3810 | 0.2983 | 0.4545 | 0.3900 |

#1 − #2 = **+0.0084 [+0.0030, +0.0138]**; #1 − #3 = +0.0097 [+0.0060, +0.0135]; tutti i delta del #1 fino
all'8° hanno CI che esclude 0 → **nessuno spareggio** fra le 41. Margine piccolo ma netto.
⚠️ Griglia patch diversa fra encoder (P16 pespatial vs P14 ijepa): segnalato dallo strumento, dichiarato in §38.

**Fatti dalla classifica**
- **Vincitori per encoder identici a §33** per tutti e 7 i presenti (pespatial gem, radio natural, ijepa
  natural, dinov3 mean, dinov2 natural, siglip2 mean, pecore mean, tutti whiten). Cambia l'**ordine** fra
  encoder: pespatial > radio > ijepa > dinov3 ≈ dinov2 > siglip2 > pecore (§33: pespatial > dinov2 > radio >
  dinov3 > ijepa > siglip2 > pecore). dinov3 (0.3855) e dinov2 (0.3810) non confrontati con CI.
- **Whitening > raw in tutte le 20 coppie** (encoder × pooling): riverificato col metro nuovo.
- La head congelata `dinov3/natural/head-probe-conv` è **17ª su 41** (0.2915); resta la **migliore su
  `nowalls`** (0.3942), cioè il danno più vicino al suo training (vedi §37).
- Migliore per danno: nowalls → head 0.3942 (pespatial/gem 0.3920 secondo); crop → pespatial/natural
  0.6855; patch → radio/natural 0.5631.

**H2 (§34) — DECISA: «artefatto»**, anche senza tipsv2. patch − crop a f=0.75, appaiato: pespatial/gem
−0.1754 [−0.1908, −0.1605] · dinov2 −0.0256 · radio −0.0619 · dinov3/mean −0.0572 · siglip2 −0.0135 ·
pecore −0.0301 (tutti CI < 0) · ijepa −0.0072 [−0.0150, +0.0005] pari · tipsv2 mancante → **6/8 < 0**, la
soglia è raggiunta qualunque sia tipsv2. ⚠️ A f=0.75 i valori sono vicini al pavimento (patch 0.003-0.029)
e a danno moderato il segno era opposto su dinov3 (§37): il verdetto vale per il livello pre-registrato.

**Aperto**: (a) le 12 tipsv2 — il vincitore resta provvisorio; (b) **costo full** del vincitore: in
`vision_full_valid_B` non c'è nessun file pespatial → job `evaluation/02` da lanciare.

## 44. RISULTATI ondata 2 COMPLETA (15 set, 53/53) — config vision = `pespatial/gem/whiten` (regola §38) · H2 «artefatto» 7/8 · costo full misurato

**Completezza**: 53 config con i 9 file (52 frozen + head), coda vuota. Le 12 tipsv2 rilanciate dall'utente
(106757-65) sono andate a buon fine; la causa del primo fallimento resta **non diagnosticata**.

**tipsv2 (le 12)** — nessuna entra fra le prime 14: migliore `natural224/whiten` **0.3168** (nowalls 0.2046,
crop 0.3919, patch 0.3540), poi `natural448/whiten` 0.3057; tutte le raw sotto 0.157.
`natural224` − `natural448` (whiten) = **+0.0111 [+0.0061, +0.0161]** → per tipsv2 il vincitore di §33
(`natural448`) **cambia** in `natural224`: è l'unico degli 8 encoder in cui la scelta interna cambia.
Whitening > raw anche in tutte le 6 coppie tipsv2 (26/26 in totale).

**Classifica finale**: prime 12 **identiche** a §43 (tipsv2 non entra). #1 `pespatial/gem/whiten` **0.5245**;
delta vs #2-#5 tutti con CI > 0 (vs radio/natural/whiten +0.0084 [+0.0030, +0.0138]) → **nessuno spareggio**.
⇒ per la regola pre-registrata in §38 la **config vision è `pespatial/gem/whiten`** (salvo decisione
dell'utente di rendere candidabile una head nuova, da fare prima di fusione e test).

**H2 (§34) — chiusa, «artefatto»**: tipsv2/natural448/whiten patch − crop a f=0.75 = −0.0140 [−0.0189,
−0.0095] → **7/8 < 0**, 1 pari (ijepa). Stessa riserva di §43: a f=0.75 i valori sono vicini al pavimento.

**Costo sulla pianta intera** (guardrail §23.1; `vision_full_valid_B`, nDCG@10, n=2000, Holm):

| A = pespatial/gem/whiten contro | composizione | topologia | geometria |
|---|---|---|---|
| valori assoluti di A | 0.8130 | 0.6306 | 0.9377 |
| dinov3/natural/head-probe-conv (config congelata prima) | −0.0163 [−0.0191, −0.0135] | −0.0164 [−0.0203, −0.0126] | −0.0020 [−0.0030, −0.0010] |
| dinov3/natural/whiten-train | −0.0120 [−0.0145, −0.0095] | −0.0321 [−0.0357, −0.0287] | +0.0030 [+0.0020, +0.0039] |

Lettura: il più robusto **paga** sulla pianta intera ~1.6 punti di composizione e topologia rispetto alla
config precedente, significativi ma piccoli; geometria praticamente pari. Da riportare accanto, come costo.

## 45. PRE-REGISTRAZIONE — selezione del checkpoint graph sulla ROBUSTEZZA: `asymrob`, `asymlostrob`, replica `asymrobrep` (15 set, PRIMA dei job)

**Decisione utente (15 set)**: la selezione sul full (sonda nDCG@10 topologia, 5000 valid/500 query, patience 10,
cap 150) è disallineata col metro (§30.7, confound di §42) → si corregge il **metodo** per tutto il ramo graph; non è un
salvataggio di `asymlost` (§41-§42 restano riportati come «marcatore + selezione full»). Esito accettato qualunque sia.
Roadmap §4: regola 2 rispettata (nessun criterio nuovo: l'AUC di §23 usata per selezionare, come la head vision §30);
regola 3 derogata esplicitamente dall'utente. Metro graph = AUC `random` (il graph non riceve crop/patch né muri).
Design: `architect` (opzione B: gallery intera + controllo nello stesso training).

**Variabile unica = regola di selezione** (criterio + budget, pacchetto dichiarato). Sonda: 2000 query **valid
disgiunte** dalle 2000 di valutazione (ricalcolate con `sample_query_rows` dai parametri di `graph_retrieval.yaml`,
overlap = 0 nel summary) · gallery = **condivisa intera** (stesso sha1 dei per-query) · `make_partial_graph` `random`
f∈{0.25,0.5,0.75}, rng `seed+qi`, transform del training (marcatore incluso per `asymlostrob`) · RR=0 oltre rango 100 ·
pari → vince il self (ottimista, costante fra epoche) · ogni epoca, **cap 300, patience 0**, soglia 1e-4 (a parità vince
l'epoca precoce). **Budget vincolante** se best_epoch > 270: dichiarato, nessun rilancio.
**Controllo nello stesso training**: ogni run salva anche `<v>_selfull` = checkpoint che la regola vecchia (topologia,
patience 10, cap 150) avrebbe scelto **sulla stessa traiettoria** → Δ appaiato senza rumore fra training.
Init/shuffle non seedati come `asym`/`asymrep`.

**Run**: `asymrob`, `asymlostrob`, `asymrobrep` (training) → 06 su tutte e 6 le varianti → 04 full su `asymrob`,
`asymlostrob`, `asymrob_selfull`, `asymlostrob_selfull`. Statistiche: `robustness_auc compare`, `significance`, bootstrap
B=10000 seed 0, CI 95%.

**Controlli di validità (se uno fallisce → nessuna config nuova, `gat/asym` resta, dichiarato)**: gallery sha1 = §32 ·
2000/2000 appaiate · overlap sonda↔valutazione = 0 · MRR f=0.0 ∈ [0.965, 0.980] per ogni checkpoint nuovo.

**Previsioni (scritte prima)**:
- **P-a1 (claim sul criterio)**: Δ AUC `asymrob − asymrob_selfull` > 0, CI > 0; best epoch rob > best epoch selfull. Se le
  due epoche coincidono Δ≡0 → «il criterio non conta per gat/asym».
- **P-a2 (descr.)**: guadagno assoluto massimo a f=0.25, minimo a f=0.75.
- **P-a3 (costo full, stesso training)**: `asymrob − asymrob_selfull` topologia < 0 (CI < 0), composizione ≤ 0, geometria
  |Δ| ≤ 0.005.
- **P-a4 (descr., fra training)**: `asymrob_selfull − asym` e `asymrobrep_selfull − asym` entro ±0.03 (la shadow replica la
  regola vecchia); `asymrob − asym` > +0.0285 (fuori dal rumore di §42).
- **P-b1 (decide la config)**: previsione `asymlostrob − asymrob` ancora con CI < 0, ma |Δ| < 0.1337 (il divario si riduce).
- **P-b2 (confound §42)**: Δ `asymlostrob − asymlostrob_selfull` > 0, CI > 0, e maggiore dell'analogo di `asymrob`. P-b2 ❌
  ⇒ confound **falsificato**: la perdita è del marcatore, non dell'epoca. `asymlostrob_selfull` ≈ §42 (descrittivo).
- **P-b3 (costo full)**: `asymlostrob − asymrob` composizione > 0, topologia > 0, geometria ≈ 0 (segni di §42).

**Regola di congelamento (decide)**: (1) controlli ok → config graph = **`gat/asymrob`**, anche se non migliora `asym`
(metodo coerente; scegliere fra training il più alto = scegliere sul rumore, §42); (2) **`gat/asymlostrob`** la
sostituisce **solo se** CI inferiore di Δ(`asymlostrob − asymrob`) > 0 **e** Δ > |Δ_rep|, Δ_rep = `asymrobrep − asymrob`;
CI > 0 ma Δ ≤ |Δ_rep| → dentro la variabilità → `asymrob`; (3) `asymrobrep` e i `_selfull` non si congelano mai;
(4) `gat/asym` = riga «regola vecchia» nel report. Costo full per asse sempre accanto. Dopo: `semantic` (§40) ripetuto
sulla congelata, descrittivo.

**Domande dell'architect — applicate le raccomandazioni** (coerenti con «esito accettato qualunque sia»; l'utente può
cambiarle PRIMA del lancio): (Q1) **no** riallenamento di `sage/asym` e `gcn/tau02asym`: gat ≫ gli altri (0.2490 vs 0.1105,
~5× il rumore), ordine degli encoder dichiarato come «regola vecchia»; caveat se l'effetto del criterio misurato nello
stesso training supera ~0.14 · (Q2) se `asymrob` < `asym` oltre il rumore, `gat/asym` **resta esclusa** (rientra solo se
fallisce un controllo di validità) · (Q3) fusione e test graph **aspettano** questi risultati.

**Da dichiarare**: selezione e misura sullo stesso split con query disgiunte (il valore della sonda al best è ottimista,
mai riportato) · la sonda vecchia di `asym` poteva includere query di valutazione (500 righe da 5000 valid), ma su nDCG
full, non self_rr · circolarità (vincolo 5): il self-recovery non usa label; il costo per-asse resta circolare; il
marcatore arriva a `proj` anche via `raw_skip` (§41) · ordine degli encoder (gat ≫ sage/gcn) stabilito con la regola
vecchia · cambia la riga graph di fusione, test e `PAPER.md §6.7` (incl. «0.394 vs 0.249») · RPLAN ha piante
duplicate: una query sonda può essere quasi identica a una di valutazione (piccolo travaso, non eliminabile con
l'esclusione per riga) · P-b2 «maggiore dell'analogo di `asymrob`» è **descrittiva** (nessun test sulla differenza
delle differenze).

**Final-reviewer (15 set)**: approvato con riserve. Fedeltà sonda↔valutazione, traiettoria invariata, shadow = regola
vecchia, default storico: ok (212 test, dry-run, mutazioni). Riserve gestite: test del marcatore via `build` e del seed
(tester) · training salva solo a fine run → `--time=16:00:00` sui 3 training (MaxTime partizione 1 giorno) ·
`train()` in modalità `full` verificato solo leggendo il codice (nessun test lo esegue).

## 46. PRE-REGISTRAZIONE — head nuova su `pespatial/gem` con coppie `nowalls` (15 set, decisione utente, PRIMA dei job)

(§45 è della sessione graph, riapertura della selezione del checkpoint: vedi TODO.)

**Domanda.** Una head allenata su un danno che toglie davvero informazione (stanze tolte **coi
muri**) si trasferisce a danni **mai visti** (crop, patch)? È H1 di §34 rifatta col render corretto,
sul vincitore frozen di §44. Scelte del punto 0 confermate dall'utente il 15 set.

**Setup.**
- Encoder `pespatial/gem`: gli stessi embedding raw del vincitore. Architettura, loss e
  iperparametri del YAML invariati (MLP 1024→256, InfoNCE τ 0.07, batch 4096, lr 1e-3, V=3).
- Coppie: `training.damage=nowalls_random`, frazione di stanze U[0.1, 0.5], flip/rot90; pool
  train+valid, gradiente solo train. Stesse stanze e stesse augmentation delle coppie storiche
  (l'rng si consuma solo nella selezione: codice + test).
- Epoca: probe su **solo** `nowalls_random`, f ∈ {0.25, 0.5, 0.75}, 1000 query valid disgiunte
  dalle 2000 di valutazione → crop e patch non toccano né i pesi né l'epoca.
- Budget come la head conv (§32): `EPOCHS=5000 PATIENCE=100`, checkpoint `head_nowalls_conv.pt`.
- Candidati: `head` e `head+whiten-train` (il whitening può stare solo **dopo** la head).

**Misura.** `vision_damage_valid_B`, stesse 2000 query: nowalls-random + crop + patch a f
0.25/0.5/0.75 (metro §38); full in `vision_full_valid_B`.

**Regole, fissate ora.**
1. Fra i due candidati vince R (`rank --robust`); parità (CI che contiene 0) → `head`. ⚠️ Questa
   scelta a due vie usa R, quindi **anche** crop e patch: dichiarato.
2. **H3 — la head si trasferisce** se Δ = AUC(head scelta) − AUC(`pespatial/gem/whiten-train`) ha
   CI 95% con estremo inferiore > 0 **sia su crop sia su patch**. Uno solo → «parziale»; nessuno →
   **falsificata**.
3. **Adozione**: la head sostituisce `pespatial/gem/whiten` solo se H3 regge **e** R > 0.5245 con CI
   del delta > 0. Altrimenti resta la config di §44.
4. Il Δ su `nowalls` si riporta ma è **in distribuzione** (danno del training): non entra in H3.
5. Costo full (nDCG@10 C/T/G) accanto, sempre; non decide.
6. Un esito negativo si riporta, non si «salva» con un'altra run (`roadmap.md §4`).

**Codice (15 set, nessun numero a disco toccato).** Chiave `training.damage` (default `random` =
storico): `vision_damage.room_damage_image` / `check_head_damage`; `projection_pairs` (render, chiave
della cache solo se ≠ `random` → la cache `b6245e020eed` resta valida, `damage` salvato in `pairs.npz`);
`retrieval_probe` (render + `meta.strategy`); `train_projection` rifiuta coppie, probe e config con danni
diversi (file vecchi senza chiave = `random`). Script `02` (+`POOLS`, `EXTRA`) e `09` (+`EXTRA`), dry-run
ok. `tests/test_head_damage.py` (11) + suite **196 passed**; il test ha trovato un bug prima dei job
(`meta.strategy` restava `random`: il training avrebbe rifiutato la probe). Comandi: `COMANDI.md § Head nuova`.

## 47. RISULTATI §45 (15 set, job 107119-131) — scelta sulla robustezza: AUC `gat` 0.249 → **0.457** · marcatore = **pareggio** · `gat/asymrob` CONGELATA

Tutti COMPLETED exit 0 (training 1:12-1:37 h, 15-20 s/epoca, sonda <1 s). Calcolo in sessione:
`robustness_auc compare`, `significance` (bootstrap B=10000 seed 0); medie per f dai per-query.

**Controlli di validità ✅ tutti**: sha1 `0c24cfc05e18` su ogni file · 2000/2000 · overlap 0 (summary) · MRR f=0.0
0.9781 / 0.9776 / 0.9750 / 0.9785 / 0.9761 / 0.9761 ∈ [0.965, 0.980].
**Budget VINCOLANTE su tutti e tre** (regola §45: dichiarato, nessun rilancio): best epoch `asymrob` 294,
`asymlostrob` 293, `asymrobrep` 300 su 300 → la robustezza saliva ancora, valori = **limite inferiore**. Regola
vecchia in ombra sulle stesse traiettorie: epoca 6 / 7 / 33.

| self_rr | AUC | f=0.25 | f=0.5 | f=0.75 | f=0.0 |
|---|---|---|---|---|---|
| `asym` (§42, regola vecchia) | 0.2490 | 0.5706 | 0.1510 | 0.0254 | 0.9752 |
| **`asymrob`** | **0.4568** | 0.8541 | 0.4279 | 0.0886 | 0.9781 |
| `asymrob_selfull` | 0.1071 | 0.2602 | 0.0531 | 0.0080 | 0.9776 |
| `asymlostrob` | 0.4529 | 0.8249 | 0.4274 | 0.1065 | 0.9750 |
| `asymlostrob_selfull` | 0.1178 | 0.2808 | 0.0629 | 0.0095 | 0.9785 |
| `asymrobrep` | 0.4960 | 0.8759 | 0.4946 | 0.1174 | 0.9761 |
| `asymrobrep_selfull` | 0.2672 | 0.6003 | 0.1740 | 0.0274 | 0.9761 |

**Previsioni**:
- **P-a1 ✅ CONFERMATA**: Δ AUC `asymrob − asymrob_selfull` **+0.3498 [+0.3394, +0.3602]**, epoca 294 > 6; sulla replica
  `asymrobrep − asymrobrep_selfull` +0.2287 [+0.2190, +0.2390].
- **P-a2 ✅**: per f +0.5939 [+0.5760, +0.6119] · +0.3748 [+0.3567, +0.3924] · +0.0806 [+0.0716, +0.0899].
- **P-a3 ❌ in parte** (full, stesso training): composizione −0.0336 [−0.0364, −0.0310] ✅ · topologia **+0.0076**
  [+0.0036, +0.0116] ❌ (attesa < 0) · geometria +0.0145 [+0.0134, +0.0156] ❌ (|Δ| > 0.005) ⇒ allenare a lungo e scegliere
  sulla robustezza **non vende la topologia** (§30.8 era un effetto della selezione precoce): costa composizione,
  guadagna geometria.
- **P-a4**: `asymrob − asym` +0.2079 [+0.1984, +0.2177] ✅ · `asymrobrep_selfull − asym` +0.0182 [+0.0115, +0.0251] ✅ ·
  `asymrob_selfull − asym` **−0.1419** [−0.1499, −0.1341] ❌ ⇒ la regola vecchia è **erratica** fra training identici
  (epoca 6 / 31 / 33 → AUC 0.107 / 0.249 / 0.267).
- **P-b1**: Δ `asymlostrob − asymrob` **−0.0039 [−0.0137, +0.0059]** → CI contiene lo zero: «CI < 0» ❌, |Δ| < 0.1337 ✅.
  **Pareggio.**
- **P-b2**: `asymlostrob − asymlostrob_selfull` +0.3352 [+0.3247, +0.3457] ✅; «maggiore dell'analogo di `asymrob`» ❌
  (descrittiva). `asymlostrob_selfull` riproduce §42 (0.1178 vs 0.1153, epoca 7 vs 6) ⇒ **confound di §42
  CONFERMATO**: a selezione equa la perdita del marcatore (−0.134) sparisce. «Marcatore + selezione full → meno robusto»
  resta vero; «il marcatore peggiora la robustezza» è **falso**.
- **P-b3 ✅** (full, `asymlostrob − asymrob`): composizione +0.0289 [+0.0265, +0.0313] · topologia **+0.0571** [+0.0535,
  +0.0608] · geometria +0.0007 [−0.0001, +0.0014].

**Rumore fra training (regola nuova)**: Δ_rep = `asymrobrep − asymrob` **+0.0391 [+0.0306, +0.0475]** (§42: +0.0285)
→ delta graph sotto ~0.04 non distinguibili da un riallenamento.

**Congelamento (regola §45)**: controlli ok → **`gat/asymrob`**; `asymlostrob` non la sostituisce (estremo
inferiore −0.0137 < 0). ⚠️ **Tensione dichiarata**: lo spareggio generale del metro (§23.1 punto 6: CI con lo zero →
topologia full → più economica) sceglierebbe `asymlostrob` (topologia full +0.057). §45 ha fissato PRIMA dei job una
regola specifica (default `asymrob`; il marcatore entra solo se più robusto) → si applica §45. Cambiarla ora = scelta
dopo aver visto i numeri: decisione utente, da dichiarare.
**Costo full della congelata vs `gat/asym`**: composizione −0.0173 [−0.0198, −0.0149] · topologia +0.0022 [−0.0012,
+0.0056] (pari) · geometria +0.0071 [+0.0062, +0.0079].

**Caveat pre-registrati che scattano**: (Q1 §45) l'effetto del criterio nello stesso training (+0.35 / +0.23) **supera**
~0.14 ⇒ l'ordine gat ≫ sage/gcn, stabilito con la regola vecchia, **non è garantito** con quella nuova (sage/gcn non
riallenati: dichiarato) · budget vincolante → robustezza graph sottostimata · confronto vision↔graph «0.394 vs 0.249»
e `PAPER.md §6.7` da aggiornare con `asymrob`.
**Da fare**: `semantic` su `gat/asymrob` (§40, descrittivo) · fusione e test graph con `gat/asymrob`.

**Addendum 16 set.** (1) **Decisione utente**: congelata `gat/asymrob` come da regola §45 (lo spareggio §23.1 che
sceglierebbe `asymlostrob` NON si applica; dichiarato). (2) **`semantic` sulla congelata** (job 107449, 2000 query,
0 senza masking, 0 svuotate): Δ self_rr `asymrob − base` **+0.3658 [+0.3484, +0.3831]** (0.3678 vs 0.0021) contro
+0.4268 [+0.4086, +0.4450] su `random f=0.5` → rapporto 0.86 (era 0.73 con `asym`, §42): la robustezza **si
trasferisce** anche con la regola nuova.


## 48. RISULTATI §46 (16 set, job 107111 · 107463 · 107476-77 · 107520) — head `nowalls` su `pespatial/gem`: **H3 FALSIFICATA** · config vision **definitiva `pespatial/gem/whiten`**

**Run.** Coppie 107111 (15 set, 3h, danno `nowalls_random`). Training 107225 fallito a `wandb.init` (nessuna
chiave, 0 epoche) → valutazioni 107228-29 e 107442 fallite a cascata (checkpoint assente) → wandb spento nel
YAML dall'utente → 107463: probe riusata (`nowalls_random`), stop a ep 1552, scelta ep **1452**, probe AUC
0.5580 (ep 1: 0.026 · 100: 0.408 · 1000: 0.538). Valutazioni 107476-77 (9 + 9 file), full 107520. Gallery
`0c24cfc05e18`, n=2000 ovunque.

**Regola 1 — candidato**: `head+whiten-train` R **0.4912** vs `head` 0.4328 (head − head+whiten −0.0584
[−0.0628, −0.0541]) → si giudica **head+whiten**. Il whitening dopo la head aiuta: opposto di §33 (dinov3,
render vecchio).

| danno | head+whiten | frozen `pespatial/gem/whiten` | Δ (head+whiten − frozen) | Δ head sola − frozen |
|---|---|---|---|---|
| stanze tolte (in distribuzione) | 0.5392 | 0.3920 | **+0.1472** [+0.1371, +0.1572] | +0.1465 [+0.1351, +0.1580] |
| crop | 0.5191 | 0.6605 | **−0.1415** [−0.1496, −0.1332] | −0.2403 [−0.2496, −0.2311] |
| patch | 0.4154 | 0.5209 | **−0.1055** [−0.1122, −0.0988] | −0.1813 [−0.1889, −0.1739] |
| **R** | 0.4912 | 0.5245 | **−0.0333** [−0.0384, −0.0281] | head sola R 0.4328 |

**Regola 2 — H3 FALSIFICATA**: su crop **e** su patch la head perde, CI < 0 su entrambi.
**Regola 3 — adozione: NO** (R più basso, CI < 0; 6ª su 55 in `rank --robust`) ⇒ **config vision definitiva
`pespatial/gem/whiten`** (§44), passo 1 della roadmap chiuso.
**Regola 5 — costo full** (nDCG@10, Holm), head+whiten − frozen: composizione +0.0002 [−0.0022, +0.0027] (pari)
· topologia −0.0177 [−0.0211, −0.0142] · geometria −0.0094 [−0.0104, −0.0084].

**Lettura.** *Fatto*: stesso segno di §37 (dinov3, render vecchio: −0.183 crop, −0.164 patch, +0.090 stanze),
ora col render corretto e su un altro encoder. *Interpretazione*: la head impara **il danno che vede** (+0.147
sulle stanze tolte) a spese degli altri, non una robustezza generale; i muri rimasti gonfiavano il vantaggio in
distribuzione ma non spiegano la mancata trasferibilità. I Δ di §37 e di qui non sono appaiati (encoder
diversi): nessun confronto di entità.

## 49. PRE-REGISTRAZIONE — late fusion vision + graph (16 set, decisioni utente, PRIMA di codice e job)

**Config**: vision `pespatial/gem/whiten` (§44, §48) · graph `gat/asymrob` (§47). Valid, 2000 query (seed 42),
gallery condivisa `0c24cfc05e18`, `exclude_self`. Riferimenti già letti, AUC stanze tolte: vision 0.3920
(`nowalls-random`, §43) · graph 0.4568 (`random`, §47).

**Domanda (decide)**: la fusione ritrova la pianta danneggiata meglio del miglior ramo singolo?

**Previsione (utente)**: stanze tolte → fusione > entrambi i rami · pianta intera → composizione e topologia
sotto il graph, geometria pari o sopra.

**Metodo (utente)**: concatenazione pesata `[√α·v ; √(1−α)·g]`, ogni ramo L2-normalizzato (vision **dopo** il
whitening fittato su train) ⇒ prodotto scalare = α·sim_V + (1−α)·sim_G. Pianta intera: vettori della gallery a
disco. Stanze tolte: vettori delle query danneggiate **ricalcolati in entrambi i rami**, **stesse stanze** per
query (vision `nowalls_random`, graph `random`), f ∈ {0.25, 0.5, 0.75}. **Nessuna fusione per rango** (utente:
solo vettori). Crop e patch fuori (il graph non li riceve), dichiarato.

**Scelta di α**: griglia {0, 0.1, …, 1} (α=0 = solo graph, α=1 = solo vision). Criterio = AUC self-recovery sulle
stanze tolte (§23.1), sulle **stesse** 2000 query del valid (numero ottimista, dichiarato: quello pulito è il
test). **Parità** = α il cui Δ AUC appaiato contro il migliore ha CI 95% che contiene 0 → si sceglie l'α
**centrale** (mediana dell'insieme dei pari; con due centrali, il più vicino a 0.5; se equidistanti da 0.5, quello
con AUC media più alta — utente, 16 set). Non la topologia full: è
circolare a favore del graph (vincolo 5). Costo sulla pianta intera (nDCG@10 C/T/G) sempre accanto.

**Regola (decide)**: «la fusione aiuta» se α* ∉ {0, 1} **e** Δ AUC(α*) − max(AUC α=0, AUC α=1) ha CI 95% > 0
(bootstrap B=10000, seed 0). max(…) = il ramo con AUC **media** più alta, confrontato query per query — non il
massimo per query (utente, 16 set: decide il ramo migliore, l'oracolo sta accanto). Se Δ < 0.04 → dichiarato
sotto il rumore fra training del graph (§47). Altrimenti «non aiuta», riportato così; la pipeline usa comunque α*.

**Oracolo (descrittivo, NON decide)**: per query e per f il self_rr migliore fra α=0 e α=1, poi AUC come sopra;
pianta intera: max per query dell'nDCG@10 per asse. ⚠️ Correzione (architect, 16 set, prima di ogni numero):
**non è un tetto** — sommando le similarità la fusione può superare entrambi i rami sulla stessa query (self 2° in
entrambi dietro concorrenti diversi → 1° nella somma). Si riportano il distacco dall'oracolo e il numero di query in
cui α* lo supera.

**Controlli di validità** (se uno fallisce ci si ferma e si diagnostica prima di leggere i risultati): α=1 riproduce
i per-query del vision e α=0 quelli del graph **dello stesso job** — soglia per ogni f: ≤ 6/2000 self_rr diversi
(misura di §37), ogni salto di rango ≤ 2 (utente, 16 set) · confronto coi per-query storici solo descrittivo ·
stesse stanze tolte nei due rami, query per query (errore duro) · MRR f=0.0 in [0.965, 0.980] (§47) · sha1 gallery e
2000/2000 appaiate.

**Dopo**: testa congiunta decisa a fusione finita (utente) · test con α* fisso, nello stesso gruppo di job della
pre-registrazione del test (vettori delle query danneggiate del test inclusi). Comandi preparati dall'agente,
lanciati dall'utente.

**Design (architect, 16 set)**: flag opt-in nelle due valutazioni esistenti → artefatto nuovo `qvec/1` (vettori delle
query danneggiate dalla stessa forward del per-query + stanze tolte; vision RAW + parametri del whitening) ·
`src/evaluation/late_fusion.py` (CPU, file fusi nel formato `perquery/1`, etichetta `nowalls-random`, meta
`graph_strategy: random`) · `src/evaluation/fusion_select.py` (controlli, α*, verdetto, oracolo). **Script nuovi**
che fissano da soli le cartelle di output (utente: gli script esistenti non si toccano; 05/06 senza env
sovrascriverebbero i per-query storici).

**Scelte di implementazione e dichiarazioni (16 set, scientific-reviewer, PRIMA dei job; nessun blocco)**:
- Casi limite non coperti sopra (implementer): due α centrali pari anche in AUC → il più piccolo · AUC medie di α=0 e
  α=1 identiche → riferimento = graph · query della pianta intera = quelle unite a f=0.0 · nel controllo α=1 una query
  col self perso ha rango max_k+1 · file fusi etichettati `nowalls-random` anche se il graph riceve `random`.
- **α è un peso nominale**: conta lo spread delle similarità in cima al ranking, non la dimensione (768 vs 128). Se un
  ramo ha spread molto più largo la zona utile di α si schiaccia verso un estremo, dove la griglia ha un punto solo, e
  «0.5 = equilibrato» non vale. Diagnostica **descrittiva** (std delle similarità top-k per ramo): si riporta, non
  cambia griglia né regola.
- **Il verdetto sul valid è ottimista**: α* scelto fra 11 sulle stesse 2000 query su cui si misura il Δ (e i due rami
  scelti anch'essi su queste query). Conferma = test, con α* fisso; sul test il riferimento si ricalcola dalle medie del
  test (conservativo).
- **Insieme dei pari riportato per intero**: può ridursi a {best} (α vicini → ranking simili → CI strette) o non essere
  contiguo.
- **Circolarità**: i Δ per-asse sulla pianta intera (composizione, topologia, e geometria derivata) sono descrittivi.
- **Asimmetria del danno**: il vision tiene la sagoma del buco; il graph perde le stanze e le superstiti restano nella
  posizione assoluta. Va nel verdetto.
- Il verdetto vale per **questo** checkpoint graph: sotto 0.04 non si distingue dal rumore fra training (§47).

### 49.1 17 set — `check` sul valid (job utente 08/09/10): C1-C4 ✅ · **C5 FAIL dal lato ALTO**, diagnosticato (CPU)

**Fatti (output `check` incollato dall'utente)**: C1 stesse stanze 2000/2000 a ogni f · C2 2000/2000, sha1
`0c24cfc05e18` · **C3 esatto**: α=1 vs vision e α=0 vs graph 0/2000 self_rr diversi a ogni f, Δ AUC +0.00000,
stesso job ✅ · C3-full vs storici: Δ nDCG@10 0.00000 su tutti gli assi · C4 vision +0.00000, graph −0.00066
[−0.00205, +0.00070] · **C5**: MRR f=0.0 α=0 0.9768 · 0.1 **0.9807** · 0.2 **0.9816** · 0.3 **0.9804** · 0.4 0.9788 ·
0.5 **0.9812** · 0.6 0.9800 · 0.7 0.9787 · 0.8 0.9791 · 0.9 0.9796 · α=1 0.9707 → sopra 0.980 in 4 α su 11.
`select` **non** lanciato (regola §49: ci si ferma e si diagnostica).

**Diagnosi (solo file f=0.0, nessuna AUC letta; gemello = pianta con similarità al query ≥ quella del self − 1e-4)**:
a f=0.0 la query coincide col suo vettore di gallery in entrambi i rami (similarità min 1.000000). Quando il self
non è 1°, il 1° è **sempre** un gemello nel ramo: vision 104/104 (69 gemelli anche nel graph), graph 82/82 (66
anche nel vision). Nella fusione i fallimenti restano **solo** sui gemelli in entrambi i rami (67-77 per α; 0 casi
gemelli in un ramo solo): la fusione rompe i pareggi che l'altro ramo distingue (recuperate 31-47 query vs vision,
21-32 vs graph). Fra i gemelli doppi l'ordine è un pareggio numerico arbitrario → MRR non monotona in α e 20-32
query peggio del ramo migliore. **Meccanismo che spiega tutti i sintomi**: il tetto a f=0.0 è dei dati (duplicati
RPLAN, §24); la fusione ne rende indistinguibili meno. Nessun difetto di pipeline (C3 esatto). Il limite alto 0.980
era calibrato sui rami singoli (§47) e non ha senso per la fusione. f=0.0 è fuori dall'AUC: il criterio non è toccato.
**Decisione utente in attesa**: considerare C5 superato (limite alto non applicabile alla fusione, dichiarato) → `select`.

### 49.2 17 set — RISULTATI `select` sul valid: **la fusione aiuta** · α* = 0.6 · sopra anche l'oracolo

**Decisione utente**: C5 superato (limite alto non applicabile alla fusione, §49.1) → `select` lanciato. Output
incollato + `results/fusion/select_valid.json` + medie per f dai per-query fusi (lette in sessione).

| AUC stanze tolte | α=0 (graph) | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | **0.6** | 0.7 | 0.8 | 0.9 | α=1 (vision) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| media | 0.4562 | 0.5029 | 0.5436 | 0.5797 | 0.6049 | 0.6213 | **0.6301** | 0.6258 | 0.5922 | 0.5159 | 0.3920 |

- **Insieme dei pari = {0.6}** (0.7 a +0.0043 [+0.0007, +0.0079]) → α* = 0.6; curva unimodale, massimo interno.
- **Regola §49**: riferimento = graph (AUC media più alta fra gli estremi); α* − graph **+0.1740 [+0.1642, +0.1838]**
  → **«aiuta»**; Δ ≈ 4× il rumore fra training (0.04). Contro il vision +0.2381 [+0.2285, +0.2476].
- **Per livello** (MRR self-recovery, graph / vision / fusione): f=0.25 0.8547 / 0.8181 / **0.9553** · f=0.5
  0.4258 / 0.2971 / **0.7201** · f=0.75 0.0880 / 0.0609 / **0.2151**.
- **Oracolo** (descrittivo): 0.5557; fusione **+0.0744** sopra, e lo supera su **1255/2000** query → i due rami non
  si limitano a sbagliare su query diverse: le somiglianze parziali si sommano (meccanismo previsto dall'architect).
- **Spread top-10** (descrittivo): sotto danno stesso ordine nei due rami (vision 0.0551/0.0180/0.0125, graph
  0.0352/0.0223/0.0193 a f=0.25/0.5/0.75) → α non schiacciato a un estremo, il rischio di artefatto di griglia non
  si materializza.
- **Pianta intera** (nDCG@10 C/T/G, descrittivi, circolari): α*=0.6 0.8501/0.6841/0.9536 · graph 0.8603/0.6780/0.9498 ·
  vision 0.8130/0.6306/0.9377. α* − graph: C **−0.0102** [−0.0125, −0.0078] · T **+0.0061** [+0.0028, +0.0094] ·
  G **+0.0039** [+0.0032, +0.0045]. α* − vision: C +0.0371 · T +0.0535 [+0.0502, +0.0569] · G +0.0159.

**Previsione (utente, §49)**: stanze tolte fusione > entrambi ✅ · pianta intera composizione sotto il graph ✅ ·
topologia sotto il graph ❌ **smentita** (sopra, +0.006) · geometria pari o sopra ✅ (sopra entrambi).

**Da dichiarare**: valid ottimista (α* scelto sulle stesse query; margine sul graph ~20× la distanza da α=0.7) →
conferma = test con α* = 0.6 fisso · asimmetria del danno (il vision tiene la sagoma del buco) · verdetto valido per
questo checkpoint graph · Δ per-asse circolari.

### 49.3 17 set — verifica del meccanismo «sopra l'oracolo» + solidità (CPU, per-query α=0/0.6/1)

**Fatto**: query in cui α*=0.6 mette il self 1° e nessun ramo lo fa: f=0.25 **58** · f=0.5 **423** · f=0.75 **171**.
Rango del ramo migliore in questi casi ≤10: 53/58 · 349/423 · 120/171 (mai oltre 100); rango del ramo **peggiore**
≤10: solo 30/58 · 76/423 · 8/171. Inverso (fusione non 1a, un ramo sì): 55 · 102 · 57.
⇒ **Correzione** a §49.2/recap («entrambi la danno vicina»): di norma **un** ramo tiene il self vicino alla cima e
l'altro abbassa i concorrenti che lo precedevano.
**Solidità (valutazione del coordinatore)**: «la fusione aiuta sulle stanze tolte» regge sul valid (pre-registrato,
appaiato, pipeline esatta, Δ 0.174 ≫ CI ±0.01 ≫ ottimismo della scelta di α). Da ridimensionare: rumore fra training
da **una** replica (ordine di grandezza) · valore assoluto 0.630 ottimista (anche rami scelti su queste query) · Δ
sulla pianta intera piccoli e circolari. **Non dimostrato**: che il guadagno venga dalla complementarità vision↔graph
e non da un semplice effetto d'insieme (manca il controllo «due modelli dello stesso ramo fusi») · danno unico
(stanze a caso) · graph dal `.mat` esatto (regime A).

## 50. PRE-REGISTRAZIONE — controllo «effetto d'insieme»: fusione graph + graph (17 set, decisioni utente, PRIMA di codice e job)

**Domanda**: il guadagno di vision + graph (§49.2) è più grande di quello che dà fondere **due training dello stesso
graph**? Distingue la **complementarità** (i pixel portano informazione in più) dall'**effetto d'insieme** (due modelli
con la stessa informazione sbagliano in modo diverso).

**Setup**: fusione di controllo C = `gat/asymrob` + `gat/asymrobrep` (replica identica, §41/§47: 0.4568 vs 0.4960),
stesso metodo di §49: vettori L2, concatenazione `[√β·asymrob ; √(1−β)·asymrobrep]` (β=1 = asymrob, β=0 = replica),
β ∈ {0, 0.1, …, 1}, stesse 2000 query del valid, stesse stanze tolte (`random`, f ∈ {0.25, 0.5, 0.75} + 0.0 di
controllo), stessa gallery. β* con la **regola di α** (§49: best, pari = CI che contiene 0, α centrale, equidistanti →
AUC media più alta). La fusione vera F = vision + asymrob resta **fissa** a α*=0.6.

**Regola (decide)**: per query q, AUC come §23.1. G_F(q) = AUC_F(q) − AUC del componente di F con AUC media più alta
(oggi il graph, α=0); G_C(q) = AUC_C(β*)(q) − AUC del componente di C con AUC media più alta. D = media di
G_F(q) − G_C(q), appaiata per nome, CI 95% bootstrap (B=10000, seed 0):
- CI > 0 → **complementarità**: vision porta più dell'effetto d'insieme;
- CI contiene 0 → **non distinguibile** dall'effetto d'insieme;
- CI < 0 → guadagno di vision + graph **spiegato dall'effetto d'insieme** (complementarità smentita).
Guadagni e non valori assoluti: i componenti di C sono più forti di quelli di F.

**Descrittivi**: G_F e G_C con CI · AUC_F(α*) − AUC_C(β*) con CI · quota G_C/G_F · nDCG@10 C/T/G sulla pianta intera di
C(β*) vs il training migliore (circolari).

**Previsione (proposta del coordinatore, approvata dall'utente)**: (1) G_C > 0 con CI > 0 · (2) D > 0 con CI > 0 e
G_C < 0.5·G_F · (3) β* ∈ [0.4, 0.6] · (4) pianta intera: C(β*) entro ±0.01 dal training migliore su tutti e tre gli assi.

**Controlli di validità** (se uno fallisce ci si ferma): stesse stanze tolte nei due training, query per query · β=1
e β=0 riproducono i per-query dello **stesso job** (≤6/2000 self_rr diversi per f, salto ≤2) · 2000/2000 appaiate,
sha1 gallery · MRR f=0.0 ≥ 0.965 (**solo limite basso**: quello alto non vale per una fusione, §49.1 — fissato ora).

**Da dichiarare**: ottimismo simmetrico (β*, come α*, scelto fra 11 sulle stesse query) · controllo **severo**: due
training identici = effetto d'insieme più favorevole; modelli diversi dello stesso tipo non provati · solo stanze
tolte · **solo valid** (utente: controllo interpretativo, il test non si consuma).

### 50.1 17 set — RISULTATI del controllo graph + graph (job utente 11, 12 + check/select/complementarity): **complementarità** · controllo MINIMO (correzione)

**Controlli ✅ tutti**: C1 stesse stanze 2000/2000 · C2 2000/2000, sha1 `0c24cfc05e18` · C3 β=1 e β=0 = i due training
**esatti** (0/2000 a ogni f, stesso job) · C4 asymrob −0.00066 [−0.00205, +0.00070], replica +0.00028 [−0.00131, +0.00190]
· C5 MRR f=0.0 0.9747-0.9779 (≥ 0.965).

| AUC stanze tolte, β | 0 (replica) | 0.1 | 0.2 | 0.3 | **0.4** | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1 (asymrob) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| media | 0.4963 | 0.5103 | 0.5220 | 0.5285 | **0.5299** | 0.5306 | 0.5247 | 0.5169 | 0.5023 | 0.4862 | 0.4562 |

Pari {0.3, 0.4, 0.5} → β* = 0.4 (mediana). β* − replica +0.0336 [+0.0286, +0.0387] (< 0.04). Oracolo del controllo 0.5467:
C **sotto** (−0.0168), lo supera su 680/2000 (F: sopra +0.0744, 1255/2000).

**Decisione (§50)**: G_F **+0.1740** [+0.1642, +0.1838] · G_C **+0.0336** [+0.0286, +0.0387] · **D = +0.1403 [+0.1287, +0.1519]**
→ **complementarità** · AUC_F − AUC_C +0.1003 [+0.0900, +0.1104] (F vince pur con componenti più deboli) · quota G_C/G_F
0.193. Pianta intera (descrittivo): C(0.4) 0.8538/0.6736/0.9520 vs replica 0.8451/0.6620/0.9506 → +0.0088/**+0.0116**/+0.0015
(asymrob 0.8603/0.6780/0.9498).

**Previsioni**: (1) G_C > 0 ✅ · (2) D > 0 e G_C < 0.5·G_F ✅ (0.193) · (3) β* ∈ [0.4, 0.6] ✅ (0.4, al bordo) · (4) pianta intera
entro ±0.01 dal training migliore ❌ **smentita di poco** sulla topologia (+0.0116; composizione e geometria entro).

⚠️ **Correzione a §50 (coordinatore, dopo i numeri, non cambia regola né verdetto)**: «controllo severo: due training
identici = effetto d'insieme più favorevole» è **sbagliato**. Due training identici hanno errori **più correlati** di due
modelli diversi: sono l'insieme **meno** diverso, quindi il controllo è **minimo**. Escluso: che il guadagno di vision + graph
sia effetto d'insieme fra training ripetuti. **Non escluso**: effetto d'insieme fra modelli diversi con la stessa informazione
(es. gat + sage) — non misurato. Resta vero che la replica è più forte (vale per i valori assoluti, gestito coi guadagni).
Note corrette in `fusion_select.py` (GRAPHGRAPH_NOTES, COMPLEMENTARITY_NOTES, testo del verdetto); `select` graph-graph e
`complementarity` rilanciati dal coordinatore su CPU → json rigenerati, **numeri identici** all'output dell'utente
(determinismo verificato); `tests/test_fusion_graphgraph.py` 17 passed.

## 51. PRE-REGISTRAZIONE — controllo «modelli diversi, stessa informazione»: fusione vision + vision (17 set, decisioni utente, PRIMA di codice e job)

**Perché**: §50.1 esclude solo l'effetto d'insieme fra training ripetuti (insieme minimo). Vision + graph differisce in
**informazione e modello** insieme; serve una coppia con **stessa informazione e modelli diversi**. Scopo dichiarato
dall'utente: una conclusione sulla complementarità con prova **sia in caso positivo sia negativo**.

**Scala a tre gradini** (guadagno sul componente migliore): (1) due training identici del graph, stessa info e stesso
modello: +0.0336 (§50.1) · (2) **`pespatial/gem/whiten` + `radio/natural/whiten`**, stessa immagine, modelli diversi:
da misurare · (3) vision + graph, info e modelli diversi: +0.1740 (§49.2).

**Setup**: V = `[√γ·pespatial ; √(1−γ)·radio]`, ciascuno whitenato col fit su train e L2; γ ∈ {0, 0.1, …, 1} (γ=1 =
pespatial); stesse 2000 query del valid, stesse stanze tolte (`nowalls_random`, f ∈ {0.25, 0.5, 0.75} + 0.0 di
controllo), stessa gallery; γ* con la regola di α (§49). `radio/natural/whiten` = #2 della griglia robusta e migliore
config di un encoder ≠ pespatial (§43-§44: R 0.5161, nowalls 0.3821), scelta con criterio fissato prima della fusione.
F = vision + graph resta fissa a α*=0.6.

**Regola (decide)**: D₂ = media per query di G_F(q) − G_V(q) (guadagni sul componente con AUC media più alta, come §50),
appaiata per nome, CI 95% bootstrap (B=10000, seed 0). Quattro esiti:
1. CI > 0 → **complementarità** (vision + graph guadagna più di due modelli diversi con la stessa informazione);
2. CI < 0 → **complementarità smentita**;
3. CI contiene 0 **e** tutto dentro [−0.04, +0.04] → **equivalenti**: la diversità dei modelli basta (conclusione
   negativa con prova);
4. CI contiene 0 e più largo di ±0.04 → **non conclusivo**.
Margine 0.04 = soglia di differenza trascurabile di robustezza già usata nel progetto (§47).

**Descrittivi**: scala G_C / G_V / G_F con CI · AUC_F − AUC_V con CI · correlazione per query (Spearman) delle AUC dei
due componenti per le tre coppie · nDCG@10 C/T/G sulla pianta intera di V(γ*) vs pespatial (circolari).

**Previsione (coordinatore, approvata dall'utente)**: (1) G_V > 0 con CI > 0 · (2) G_V < 0.5·G_F ed esito 1 · (3) γ* ∈
[0.4, 0.6] · (4) pianta intera: V(γ*) fra 0 e +0.02 sopra pespatial su ciascun asse.

**Controlli di validità** (se uno fallisce ci si ferma): stesse stanze nei due encoder, query per query · γ=1 e γ=0
riproducono i per-query dello **stesso job** (≤6/2000 self_rr diversi per f, salto ≤2) · 2000/2000, sha1 gallery ·
whitening fittato su train per entrambi · MRR f=0.0 ≥ 0.965 (solo limite basso, come §50).

**Da dichiarare**: due ViT sono meno diversi fra loro di un ViT e una GNN → anche l'esito 1 lascia un residuo sulla
**quantità** di diversità (limite strutturale: informazione e modello non si separano del tutto) · il componente
migliore del controllo è più debole (≈0.392 vs 0.456) → più margine di guadagno, il controllo parte favorito · ottimismo
simmetrico (γ* fra 11 sulle stesse query) · solo valid, solo stanze tolte.

### 51.0 17 set — review finale del codice §51 (PRIMA dei job): approvato · regressione reale su §49-§50 ok · pareggi FAISS

- **Regressione su dati reali (final-reviewer, CPU)**: `select` vision-graph (α*=0.6, +0.1740 [+0.1642, +0.1838], oracolo 0.5557),
  `select` graph-graph (pari {0.3,0.4,0.5}, β*=0.4, +0.0336 [+0.0286, +0.0387]), `complementarity` (G_F +0.1740, G_C +0.0336,
  D +0.1403 [+0.1287, +0.1519], quota 0.193) **identici**; json in `results/` invariati (md5). Test 297 passed.
- **Refactor `_run_same_kind`**: file graph-graph rigenerati (β=0 f=0.0 e f=0.25; full β=0.4) → meta identici; self_rr diversi
  37/2000 e 38/2000, **tutti pareggi esatti di similarità** (75/75 ranghi dentro l'intervallo del pareggio); full nDCG
  differenze ≤ 4e-5. *Ipotesi (non verificata)*: FAISS spezza i pareggi in modo diverso col numero di thread (1 in review, 8
  nel job). ⇒ **Da dichiarare**: rilanciare 10/12/14 non riproduce i file bit per bit; i numeri di §49-§50 restano quelli dei job.
- **Regola di contingenza (proposta dal coordinatore, CONFERMATA dall'utente il 17 set, PRIMA dei job)**: se C3 di §51 fallisce, si
  verifica se **tutte** le query diverse sono pareggi esatti di similarità; se sì → C3 superato e dichiarato; altrimenti stop.
- **C5 per radio (γ=0)**: rischio basso — a f=0.0 la query non è danneggiata e §24(b) misurò MRR 0.9705-0.9712 su tutte le 80
  config (radio incluso), sopra 0.965.
- Minori: note «768 vs 128» e help di `--vision-qvec` rese neutre rispetto alla coppia (coordinatore; 71 test fusione passed).

### 51.1 17 set — RISULTATI del controllo vision + vision (job 13, 14 + check/select/complementarity): **complementarità**, ma il modello pesa più del previsto

**Controlli ✅ tutti**: C1 stesse stanze 2000/2000 · C2 sha1 `0c24cfc05e18` · C3 γ=1 e γ=0 **esatti** (0/2000 a ogni f,
stesso job) · C4 entrambi +0.00000 · C5 MRR f=0.0 **0.9707 identico per tutti gli 11 γ** (contingenza sui pareggi non
servita). ⚠️ Fatto notevole: a f=0.0 la fusione vision+vision **non** rompe nessun pareggio (i duplicati RPLAN hanno
immagini identiche, quindi nessun encoder li distingue), mentre vision+graph saliva a 0.982 (§49.1). Indizio a favore
dell'informazione diversa; fuori dal metro (f=0 escluso dall'AUC).

| AUC stanze tolte, γ | 0 (radio) | 0.1 | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 | 0.8 | 0.9 | 1 (pespatial) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| media | 0.3821 | 0.4116 | 0.4391 | 0.4643 | 0.4832 | **0.4902** | 0.4842 | 0.4652 | 0.4443 | 0.4174 | 0.3920 |

Pari = {0.5} → γ* = 0.5; riferimento pespatial (γ=1); γ* − riferimento +0.0982 [+0.0916, +0.1047]. Oracolo del controllo
0.4540: V lo **supera** (+0.0361, su 910/2000).

**Decisione (§51)**: **D₂ = +0.0758 [+0.0639, +0.0876]** → **esito 1: complementarità**. Scala dei guadagni:
G_C **+0.0336** (stesso modello) · G_V **+0.0982** (modelli diversi, stessa informazione) · G_F **+0.1740** (informazione
diversa). AUC_F − AUC_V +0.1399 [+0.1305, +0.1491]. **Quota G_V/G_F = 0.564**.
**Correlazione fra i componenti (Spearman per query)**: vision/graph **0.092** · pespatial/radio 0.622 · graph/replica 0.649
→ la coppia vision+graph è di gran lunga la meno correlata.
**Pianta intera** (descrittivo): V(0.5) 0.8200/0.6476/0.9435 vs pespatial 0.8130/0.6306/0.9377 → +0.0070/+0.0171/+0.0058.

**Previsioni**: (1) G_V > 0 ✅ · (2) esito «complementarità» ✅ **ma** «G_V < 0.5·G_F» ❌ (0.564: il modello pesa più del
previsto) · (3) γ* ∈ [0.4, 0.6] ✅ (0.5) · (4) pianta intera fra 0 e +0.02 su ogni asse ✅.

**Lettura (da riportare così)**. *Fatti*: fondere due modelli **diversi** con la **stessa** informazione dà +0.098; passare a
un modello con **informazione diversa** aggiunge altri +0.076, oltre il margine di equivalenza ±0.04 e con CI che esclude lo
zero. *Interpretazione*: il guadagno della late fusion è **in parte** effetto d'insieme fra modelli diversi (~56%) e **in
parte** informazione complementare (~44%); non è né solo l'uno né solo l'altro. Il controllo parte favorito (componente
migliore più debole: 0.392 vs 0.456) e resta il residuo strutturale (due ViT < ViT vs GNN in diversità).

## 52. PRE-REGISTRAZIONE DEL TEST — approvata dall'utente il 18 set, PRIMA di ogni job del test

**Il test si legge UNA VOLTA.** Nessuna scelta, nessuna config nuova, nessun iperparametro: tutto è congelato sul valid.

**Sistemi**: vision **`pespatial/gem/whiten`** (§44, §48) · graph **`gat/asymrob`** (§47) · **fusione** dei due con
**α = 0.6 fisso** (§49.2, via `--fixed-alpha-from results/fusion/select_valid.json`) · baseline training-free `hist` ·
pavimento = ranking casuale sul test, già a disco (`results/random_floor/random_{vision,graph}_test_*`, §10, §20).

**Misure** (split test, 2000 query seed 42, gallery condivisa `0c24cfc05e18`, `exclude_self`):
- **Pianta intera**: nDCG@10 per asse (+ Recall/mAP@10 come nel resto del progetto) per vision, graph, `hist`, fusione.
- **Robustezza** (AUC self-recovery, f ∈ {0.25, 0.5, 0.75}): vision col metro a tre danni (`nowalls` + crop + patch,
  §38); graph su stanze tolte (`random`); fusione su stanze tolte con α fisso. f=0.0 solo come controllo.
- **Confronti appaiati** con CI 95% bootstrap (B=10000, seed 0) sulle stesse query.

**Punto aperto CHIUSO qui (era in `COMANDI.md § Test finale`)**: il confronto **vision↔graph sotto danno** si legge dai
**file della fusione**, α=1 (= vision) e α=0 (= graph): stessa etichetta, stesse query, stesso file → `compare_auc`
appaiato senza override. Sul valid α=1/α=0 riproducono i rami **esattamente** (§49.1, §51.1 C3).

**Previsioni (scritte prima)**: (1) i due rami sul test entro ±0.01 dai valori del valid (drift misurato ad agosto
±0.006, §14.1) · (2) la fusione resta sopra il ramo migliore con Δ AUC > +0.12 (valid +0.1740: l'α è scelto sul valid,
quindi atteso più piccolo) · (3) ordine sotto danno fusione > graph > vision · (4) pianta intera: graph > vision su
composizione e topologia; fusione fra i due sulla composizione e sopra entrambi sulla geometria.

**Controlli di validità sul test** (se uno fallisce: stop e diagnosi prima di leggere i risultati): sha1 gallery e
2000/2000 appaiate · α=1 e α=0 riproducono i per-query dello stesso job (≤6/2000 self_rr diversi per f, salto ≤2; se
falliscono solo per pareggi esatti → superato, contingenza confermata dall'utente il 17 set, §51.0) · MRR f=0.0 ≥ 0.965
(solo limite basso, §50) · `fallback=0` e 0 query svuotate nei log.

**Correzioni della review scientifica (18 set, PRIMA dei job; 4 punti bloccanti risolti a costo zero)**:
1. **C5 sul test = solo limite basso** (≥ 0.965) anche per la fusione: `pair_spec(pair, split)` in `fusion_select.py`
   ora toglie il tetto 0.980 quando lo split è `test` (sul valid resta come in §49, dove la deroga fu decisa a
   posteriori e dichiarata, §49.1). Deciso **prima** di vedere i numeri del test.
2. **Ordine dei job vincolante**: `07 gat asymrob` **prima** di `09_queryvec_graph.sh test`, mai in parallelo —
   entrambi riscrivono `embeddings/graph/gat/asymrob/{embeddings.npy,names.json}` e la fusione verifica lo sha1.
3. **Comando del danno vision, letterale** (altrimenti esce il `random` coi muri e il metro a tre danni non è
   calcolabile): `STRAT=damage EXTRA="partial.strategies.random.enabled=false partial.strategies.nowalls_random.enabled=true"`.
4. **Riferimenti della previsione (1), fissati ora**: vision = AUC di α=1 contro **0.3920**, graph = AUC di α=0 contro
   **0.4562** (entrambi dai file fusi del valid); metro a tre danni del vision contro **0.5245** (§44).
5. **Baseline `hist` anche sotto danno**: si lancia `07 hist - both` (non solo `full`), altrimenti il claim «più
   robusti della baseline senza training» non sarebbe sostenibile senza riaprire il test.

**Altre dichiarazioni (review, PRIMA dei job)**: il floor vision sul test ha la gallery vecchia (sha1 `22d91d`,
67.453) mentre quello graph ha la condivisa → **si usa il floor graph** come pavimento per entrambi, dichiarato (le
2000 query dei due file sono identiche, verificato dalla review) · `hist` resta **oracolo per costruzione** sulla
composizione: non è denominatore su quell'asse · le stanze tolte del vision escono da due forward indipendenti
(`03 … partial` e `08 test`): **controllo CPU aggiunto** — `self_rr` identici fra le due, altrimenti stop ·
**controllo aggiunto**: i nomi delle 2000 query del test coincidono con i `names` del floor · `fallback=0` si legge
lato vision; per il graph «0 query svuotate» si legge dal log · **claim che il test NON potrà sostenere**: circolarità
dei tre assi, asimmetria del danno, ottimismo residuo (anche i due rami sono stati scelti su queste query del valid),
ordine gat ≫ sage/gcn, la quota «~56% diversità di modello» (è del valid), qualunque cosa fuori da RPLAN.

**Job, tutti insieme dopo questa pre-registrazione** (`COMANDI.md § Test finale` + § Late fusion):
`03 … full` · `03 … partial` col comando di cui sopra · **`07 gat asymrob` (full + partial) PRIMA di 09** ·
`07 hist - both` ·
`08_queryvec_vision.sh test` · `09_queryvec_graph.sh test` · poi, su CPU, `10_late_fusion.sh test` con
`ALPHAS_FROM=results/fusion/select_valid.json`, `fusion_select check` e `select --fixed-alpha-from`.
⚠️ Sovrapposizione dichiarata: le stanze tolte del vision escono sia da `03 … partial` (cartella `vision_test_B`) sia da
`08 test` (cartella `fusion_branches_test`); i confronti appaiati coi rami usano i file della famiglia fusa.

**Fuori dal test, deciso ora**: **testa congiunta** — se si farà, sarà una **seconda tornata pre-registrata a parte**,
allenata e selezionata **solo sul valid**, con l'impegno a riportarne l'esito **qualunque sia**; il report dichiarerà che
il test è stato letto in due momenti distinti (come già fa per le tre varianti vision di agosto, Caveat 5 di `PAPER.md`).

## 53. RISULTATI DEL TEST (18 set) — letto UNA VOLTA, come pre-registrato in §52. **La fusione aiuta anche sul test**

**Controlli ✅ tutti** (`check --split test`): C1 stesse stanze 2000/2000 a ogni f · C2 sha1 `0c24cfc05e18`,
2000/2000 · C3 α=1 = vision e α=0 = graph **esatti** (0/2000 a ogni f, stesso job; Δ AUC +0.00000) · C5 MRR f=0.0
0.9768 / 0.9789 / 0.9701 (≥ 0.965, solo limite basso come pre-registrato). Controllo aggiuntivo di §52 (doppio
forward del danno vision, `03 … partial` vs `08 test`): self_rr diversi **0 / 1 / 0** su 2000 a f=0.25/0.5/0.75,
medie identiche alla quarta cifra → pareggi, superato. Query del test = query del floor (verificato).

**Robustezza, stanze tolte (AUC self-recovery, test)**: graph (α=0) **0.4700** · vision (α=1) **0.3965** ·
**fusione α=0.6 fisso dal valid 0.6424**. α* − graph (riferimento, ricalcolato sulle medie del test)
**+0.1725 [+0.1625, +0.1826]** → **«la fusione aiuta»**, praticamente identico al valid (+0.1740).
Per livello (graph / vision / fusione): f=0.25 0.8620 / 0.8216 / **0.9580** · f=0.5 0.4471 / 0.3037 / **0.7363** ·
f=0.75 0.1008 / 0.0642 / **0.2331**. **Oracolo** 0.5693: la fusione lo supera di **+0.0731**, su **1218/2000** query.

**Vision col metro a tre danni (test)**: R **0.5210** (nowalls 0.3965 · crop 0.6543 · patch 0.5123); sul valid 0.5245.

**Pianta intera (test, nDCG@10 C/T/G)**: graph 0.8585/0.6748/0.9494 · vision 0.8144/0.6328/0.9372 ·
fusione 0.8512/0.6856/0.9541. Appaiati: fusione − graph composizione **−0.0073** [−0.0096, −0.0049] · topologia
**+0.0107** [+0.0074, +0.0141] · geometria **+0.0047** [+0.0040, +0.0053]; fusione − vision +0.0368 / +0.0528 /
+0.0169. Oracolo per asse 0.8702 / 0.6963 / 0.9537 (fusione sopra la geometria dell'oracolo su 1049/2000 query).
⚠️ Descrittivi e circolari (§21), come dichiarato.

**Previsioni di §52**: (1) rami entro ±0.01 dal valid → vision +0.0045 ✅, **graph +0.0138 ❌** (0.4700 vs 0.4562:
fuori di poco, **in meglio**), metro a tre danni −0.0035 ✅ · (2) Δ fusione > +0.12 ✅ (+0.1725: **nessun calo**
rispetto al valid, quindi l'ottimismo della scelta di α era trascurabile) · (3) ordine fusione > graph > vision ✅ ·
(4) graph > vision su composizione e topologia ✅; fusione fra i due sulla composizione ✅ e sopra entrambi sulla
geometria ✅ (anche sulla topologia, come sul valid).

**Baseline training-free `hist` (test, letta in sessione dai per-query)**: pianta intera 0.9998/0.6428/0.8936
(composizione = oracolo per costruzione, non è denominatore su quell'asse) · stanze tolte **AUC 0.0013**
(0.0026/0.0011/0.0003 per f) → sotto danno la baseline non ritrova nulla: fusione 0.6424 e graph 0.4700 sono
~500× e ~360× la baseline.

**Conclusione**: il risultato del valid **generalizza**. Restano i limiti già dichiarati (§52): circolarità degli assi
sulla pianta intera, asimmetria del danno fra i rami, verdetto legato a questo checkpoint del graph, la quota
«~56% diversità di modello» misurata sul valid, e niente fuori da RPLAN.

## 54. FIGURE (18 set) — le prime due, nessun numero nuovo

Report a **due colonne** (deciso dall'utente): le figure si disegnano alla larghezza finale
(`src/figures/style.py`: 3.25″ una colonna, 6.875″ due; font 8/7 pt, palette Okabe-Ito + tratteggi e
marker diversi, così reggono anche il bianco e nero). Ogni figura scrive in `figures/` PDF + PNG +
`*.sources.txt` (file letti, comando, numeri disegnati): una figura non tracciabile ai per-query non è
rifacibile.

- **F7 `figures/f_damage_test.pdf`** (`src/figures/damage_test.py`): self-recovery vs frazione di stanze
  tolte sul **test**, quattro sistemi. Riletta dai per-query, riproduce §53 alla quarta cifra: AUC
  baseline 0.0013 · vision 0.3965 · graph 0.4700 · fusione 0.6424; per f=0.25/0.5/0.75 vision
  0.8216/0.3037/0.0642, graph 0.8620/0.4471/0.1008, fusione 0.9580/0.7363/0.2331; f=0.0
  0.9701/0.9768/0.9789 (disegnato e marcato «tetto dei dati», fuori dall'AUC). 2000 query appaiate su
  tutti e quattro i sistemi; bande = CI 95% bootstrap (B=10000, seed 0) della media.
- **F8 `figures/f_alpha_valid.pdf`** (`src/figures/alpha_valid.py`): AUC di robustezza vs α sul **valid**,
  le tre coppie sovrapposte. α scelto e guadagno sul **miglior estremo**: vision+graph α*=0.6 0.6301 vs
  0.4562 → **+0.1740** · vision+vision α*=0.5 0.4902 vs 0.3920 → **+0.0982** · graph+graph α*=0.4 0.5299 vs
  0.4963 → **+0.0336**; linea dell'oracolo vision+graph 0.5557. Coincide con §49.2, §50.1, §51.1.
- **Guardia**: `alpha_valid` ricalcola ogni media dai per-query e **si ferma** (`ValueError`) se differisce
  da quella del json di `fusion_select` oltre 1e-6 — le 33 medie hanno superato il controllo.
- **F1 `figures/f_teaser_valid.pdf`** (`src/figures/teaser_valid.py`): teaser, **valid**, danno
  `nowalls-random` f=0.5 — una query danneggiata e i primi 5 risultati di vision / graph / fusione,
  con l'originale incorniciato. **Senza job**: `ret_rows` è la classifica SENZA la query e `self_rr` dice
  a che posto stava, quindi reinserendola al suo rango la classifica torna esatta; il danno è ridisegnato
  con `vision_damage.render_wiped_image` dalle stanze salvate nel `qvec` (le stesse tolte in valutazione).
  Query scelta con **regola dichiarata**: prima in ordine alfabetico fra le **423** in cui la fusione mette
  l'originale al 1° posto e nessuno dei due rami lo fa → `10340` (4 stanze tolte; vision >100° · graph 3° ·
  fusione 1°). ⚠️ **Contenuto diverso dal piano di `PAPER.md §10.b`**: non i top-5 per asse (impostazione
  per-asse = circolare, §21) ma il risultato del report.
- **F3 `figures/f_damage_kinds_valid.pdf`** (`src/figures/damage_kinds_valid.py`), due pannelli, valid.
  Sinistra, `pespatial/gem/whiten-train` sui tre danni: stanze tolte **0.3920** (f: 0.8181/0.2971/0.0609) ·
  crop **0.6605** (0.9600/0.8312/0.1904) · patch **0.5209** (0.9623/0.5856/0.0150) → **R 0.5245**, identico a §44.
  Destra, l'artefatto dei muri su `dinov3/natural/whiten-train` (l'unica config con entrambe le curve):
  `random` **0.5357** vs `nowalls-random` **0.3040**, a f=0.5 0.5281 vs 0.1920 (**+0.336**). Aree davvero
  tolte a f=0.5: crop 0.498 · patch 0.500 · nowalls 0.522 → i tre danni sono confrontabili a parità di f.
- **F2 `figures/f_pipeline.pdf`** (`src/figures/pipeline.py`): schema dei due rami + fusione + misura
  condivisa. I numeri scritti nel diagramma (vision **768-d**, graph **128-d**, fusa **896-d**, α=0.6,
  gallery 67.405) sono **letti a ogni disegno** dai qvec, da `select_valid.json` e da `shared_gallery.json`.
- **F5 `figures/f_classes_valid.pdf`** (`src/figures/classes_valid.py`): dimensione delle classi di
  equivalenza dal campo `num_relevant` (valid, 2000 query, self escluso). Composizione mediana **5007**
  (media 5839, max 15251, singleton **6**) · topologia mediana **19** (media 198, max 2633, singleton
  **311**) → saturazione da una parte, query saltate dall'altra. Il conteggio non dipende dal sistema:
  la figura lo verifica su due file (α=0.6 e α=0) e si ferma se non coincide.
- **F6 `figures/f_ablation_valid.pdf`** (`src/figures/ablation_valid.py`): R di **55** config
  (52 frozen + 3 head) per encoder, box + punti. Mediane: pecore 0.1490 · siglip2 0.1619 · tipsv2 0.2096 ·
  dinov2 0.2585 · ijepa 0.2657 · dinov3 0.2764 · radio 0.3351 · **pespatial 0.3864**. Migliori per encoder:
  pespatial/gem/whiten **0.5245** (la scelta) e radio/natural/whiten **0.5161** → il primo posto non è
  speciale; dentro un encoder lo spread è grande (pespatial va da 0.1950 a 0.5245). Cache dei valori in
  `figures/f_ablation_valid.data.json` (`--refresh` ricalcola, ~90 s).
- `tests/test_figures.py` (19 test: appaiamento della curva, AUC senza f=0, guardia json↔file,
  provenienza, larghezze, ricostruzione della classifica del teaser, regola di scelta della query,
  appaiamento fra danni, guardia sui conteggi delle classi, raggruppamento delle ablation).
  Suite completa senza `test_vision_retrieval.py`: **316 passed**.

Resta **solo F4** (scatter loss↔retrieval) di `PAPER.md §10.b`: dipende da
`notebooks/vision/vision_encoders_results.csv`, incompleto (11 serie su 15 attese).
