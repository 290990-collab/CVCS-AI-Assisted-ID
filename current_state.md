# current_state.md — audit metodologico della codebase

**Data:** 30 luglio 2026 · **Tipo:** diagnosi, nessuna modifica al codice
**Metodo:** 3 ricognizioni a basso costo (estratti `file:riga`) → 2 revisioni
indipendenti in parallelo (validità scientifica · coerenza codice/config/doc) →
riconciliazione. Ogni rilievo è ancorato a `file:riga` letta in sessione.

> ⚠️ **Assunzione dichiarata.** Gli "obiettivi aggiuntivi" annunciati non sono
> ancora stati forniti: questo audit misura la codebase contro gli obiettivi
> **documentati** (`CLAUDE.md`, `README.md`, `PAPER.md`) e contro i vincoli DURI
> che il progetto si è dato. Se gli obiettivi cambiano, cambia la gravità di
> alcuni rilievi, non la loro esistenza.

**Come si legge.** I **Fatti** sono righe di codice lette (o comandi eseguiti in
sessione dagli agenti). Le **Interpretazioni** sono deduzioni, marcate come tali.
Le stime numeriche di impatto sono ordini di grandezza, non misure.

---

## 0. Il quadro in una pagina

Il codice **funziona** e la pipeline è coerente: nessun bug che produca numeri
sbagliati in silenzio, i due rami condividono davvero il codice metrico, le 2000
query sono davvero le stesse. Il problema non è l'implementazione: è che **il
protocollo di misura concede più di quanto il progetto dichiari**, e i due
risultati di punta sono i più esposti.

I quattro punti che, se non chiusi, un revisore esterno userebbe per demolire il
report:

1. **La selezione delle configurazioni avviene sul test** (nessuno script valuta
   sul valid nel ramo vision) → i massimi per-asse, incluso il **0.948**, sono
   max su ~60 configurazioni lette sul test.
2. **Il whitening del vision è fittato su tutta la gallery**, che contiene valid
   e test → il vincolo "statistiche dal solo train" è letteralmente falso per
   metà del progetto, e i due rami usano protocolli di normalizzazione diversi.
3. **Nessuna stima di incertezza esiste**, e i valori per-query non vengono
   nemmeno salvati → il claim di punta ha margine **0.002** e non è difendibile.
4. **La circolarità è più estesa di quanto dichiarato**: anche la ground truth
   geometrica è funzione dei campi `.mat` che il grafo riceve in input → la frase
   "la geometria è l'unico asse alla pari" non è supportata dal codice.

Nessuno dei quattro richiede di ri-allenare qualcosa: tre sono ri-valutazioni,
uno è una riscrittura di come si racconta il risultato.

---

## 1. Rilievi ad alta gravità

### A1 — Le configurazioni sono scelte sul test (viola il vincolo DURO 1)

**Fatto.** Nessuno script di valutazione usa lo split `valid`: tutti passano
`eval.split=test` — [scripts/vision/04_eval_frozen_full.sh:33](scripts/vision/04_eval_frozen_full.sh#L33),
`05:33`, `06:33`, `07:34`; il default in
[configs/vision_retrieval.yaml:55](configs/vision_retrieval.yaml#L55) è `null`.
Nel ramo vision **non esiste alcun meccanismo di selezione sul valid**: la
scelta fra encoder × pooling × {raw, whiten, head, head+whiten} (≈60
combinazioni) è fatta leggendo il test.

**Perché conta.** Il vincolo DURO 1 dice "il test non sceglie nulla". Qui il test
sceglie l'encoder, il pooling e la trasformazione. La riga "miglior config per
asse" (**C 0.830 · T 0.662 · G 0.948**) è inoltre un **oracolo per-asse**: i tre
numeri vengono da tre configurazioni diverse, ciascuna massimizzata sul test.

**Effetto sui numeri.** Winner's curse: il massimo di ~60 misure rumorose
sovrastima il valore atteso della configurazione scelta. La distanza fra prima e
seconda dentro lo stesso encoder (0.948 vs 0.943) è dello stesso ordine del
rumore di selezione. **Interpretazione:** l'entità è ignota perché non c'è una
stima di varianza (→ A4).

**Asimmetria fra i rami.** Il graph è in regola sul punto principale: la variante
si scegle con la `RetrievalProbe` sul **valid** dentro il training
([src/graph/training/train_gnn.py:174-181](src/graph/training/train_gnn.py#L174-L181),
`:213-215`) e il test si legge per le finaliste. Resta che il *numero* di
geometria attribuito al graph (0.946 di `gat/nosym`) è anch'esso il **massimo
per-asse fra varianti diverse** — 0.946 da gat, 0.943 da sage, 0.938 da gcn.

**Come si chiude (nessun retraining).** Ri-lanciare la griglia con
`eval.split=valid`, scegliere lì, e leggere il test **una volta** per la
configurazione vincente. Costo: solo valutazioni.

**Nota.** Un corollario da correggere nel report: il ranking sul test è stato
usato come prova che la probe "misura la cosa giusta". È una decisione
metodologica validata sul test.

---

### A2 — Il whitening del vision è trasduttivo, il vincolo dice il contrario

**Fatto.** `prepare_index` fitta media e PCA sulla gallery **intera**
([src/vision/models/retrieval_model.py:135-138](src/vision/models/retrieval_model.py#L135-L138)
→ `_fit_whitening`, `:157-173`), e la gallery è tutto `snapshot_train/`
([src/vision/data/loader.py:44](src/vision/data/loader.py#L44), 67.453 PNG) che
per ammissione dello stesso `CLAUDE.md` **mescola i tre split ufficiali**. Non
esiste un'opzione per fittare sul solo train.
Il ramo graph invece rispetta il vincolo: statistiche da
`RplanGraphDataset(split="train")`
([src/graph/training/train_gnn.py:135-139](src/graph/training/train_gnn.py#L135-L139)),
salvate e **ricaricate** in eval senza ricalcolo
([src/graph/evaluation/graph_evaluate.py:99-112](src/graph/evaluation/graph_evaluate.py#L99-L112)).

**Perché conta.** Due ragioni, di peso diverso. La prima è formale ma letale in
review: il vincolo DURO 1 come è scritto è **falso** per metà del progetto, ed è
la prima cosa che si trova confrontando `CLAUDE.md` col codice. La seconda è
sostanziale: **i due rami non usano lo stesso protocollo di normalizzazione**, e
il confronto cross-ramo mette a paragone un ramo trasduttivo con uno induttivo.

**Effetto sui numeri.** Tutte le righe `whiten` — incluso il **0.948** — sono
ottimistiche di una quantità **non quantificata**. *Interpretazione:* trattandosi
di statistiche del secondo ordine su 67k campioni di cui il test è ~3%,
l'inflazione attesa è piccola; il whitening non usa label ed è difendibile in
letteratura come scelta trasduttiva. Ma va **dichiarata**, non subita.

**Come si chiude.** O si rifitta sul solo train (una ri-valutazione) e si misura
la differenza, o si riscrive il vincolo come "whitening trasduttivo dichiarato" e
si allinea il graph. La cosa che non si può fare è tenere il vincolo così com'è.

---

### A3 — La head del vision è selezionata con il criterio che il progetto ha smentito

**Fatto.** Il checkpoint della projection head è scelto per **val-loss InfoNCE
minima** con early stopping
([src/vision/training/train_projection.py:95-116](src/vision/training/train_projection.py#L95-L116),
`patience: 8` in `configs/vision_retrieval.yaml:45`). Il ramo graph ha
**abbandonato esattamente questo criterio** — documentato nel codice a
[src/graph/training/train_gnn.py:20-33](src/graph/training/train_gnn.py#L20-L33)
— perché la classifica per val-loss è l'**inverso** di quella per retrieval, e lo
ha sostituito con una probe su nDCG.

**Perché conta.** È la scoperta metodologica più importante del progetto,
applicata a un ramo solo. La causa individuata (InfoNCE è *instance
discrimination*: in un batch tratta da negativi le piante che la ground truth
considera rilevanti) vale identica per la head vision, che usa la stessa loss.

**Effetto sui numeri.** *Interpretazione, asse-dipendente:*
- sulle metriche **per-asse** la head è plausibilmente **sotto-stimata** → la
  conclusione «sul full la projection head non aiuta, frozen+whiten vince» è
  **confusa dal criterio di selezione** e non è difendibile così com'è;
- sul **self-recovery partial** i guadagni restano credibili (MRR 0.205 → 0.668),
  perché lì il positivo dell'InfoNCE *è* la vista mascherata, cioè quasi il task
  valutato.

**Come si chiude.** Riallenare una head con selezione su una probe di retrieval
sul valid e confrontarla **sul valid** con quella attuale. È l'unico rilievo alto
che richiede un training (breve: è solo la testa).

---

### A4 — Non esiste nessuna stima di incertezza, e i per-query non si salvano

**Fatto.** In `src/evaluation/` e in entrambi i rami non esiste alcun calcolo di
std, intervallo di confidenza, bootstrap o test statistico: le medie vengono
**solo stampate**
([src/vision/evaluation/evaluate.py:277-295](src/vision/evaluation/evaluate.py#L277-L295),
[src/graph/evaluation/axis_metrics.py:96-116](src/graph/evaluation/axis_metrics.py#L96-L116)).
I valori **per-query non vengono mai persistiti**: l'unico `np.save` in
valutazione sono gli embedding
([graph_evaluate.py:150-151](src/graph/evaluation/graph_evaluate.py#L150-L151)).

**Perché conta.** Non è solo "manca un intervallo di confidenza": non si può fare
nemmeno un test **post-hoc**, perché il dato per-query è già stato buttato. Ogni
verifica di significatività richiede di ri-lanciare.

**Conclusioni oggi a rischio:** (a) geometria vision 0.948 vs graph 0.946
(Δ **0.002**); (b) «GCN +0.001» presentato come guadagno; (c) l'ordine del
cluster di testa vision (0.8077 vs 0.8067 vs 0.8067) deciso alla **quarta
cifra**; (d) GCN > SAGE > GAT (margini più larghi, ma senza CI).
C'è anche un'**auto-contraddizione** nei documenti: ±0.003 è dichiarato "rumore"
in un punto, e 0.005 è venduto come "il risultato più significativo della
tabella" in un altro.

**Come si chiude, a costo zero di GPU.** I confronti sono su **query fisse**,
quindi sono appaiati: salvare il vettore per-query di nDCG@10 e fare un
**Wilcoxon signed-rank** (o un bootstrap appaiato) sulle differenze. Per gli assi
discreti, appaiare solo le query sopravvissute allo skip singleton in **entrambe**
le run.

---

### A5 — La circolarità è più estesa di quanto il progetto dichiari

**Fatto.** La ground truth geometrica è funzione di `footprint` (`rec.gtBox`),
`boxes` (`rec.gtBoxNew`) e `room_types` (`rec.rType`) —
[src/evaluation/relevance.py:137-146](src/evaluation/relevance.py#L137-L146) e
[src/data/rplan_metadata.py:113-134](src/data/rplan_metadata.py#L113-L134). Le
feature dei nodi del grafo sono `one-hot(rType)` ⊕ `[cx, cy, w, h, area, aspect]`
calcolate **da `meta.boxes`** —
[src/graph/graph_builder.py:59-92](src/graph/graph_builder.py#L59-L92).

**Interpretazione (solida, non misurata).** `type_area_distribution` è
letteralmente «somma delle aree per tipo / totale», cioè una funzione delle
feature dei nodi (tipo × area, entrambe in input); il bbox del footprint è
l'unione dei box delle stanze. Il ramo graph riceve dunque in input tutto ciò che
serve a **calcolare** la ground truth geometrica con poche operazioni
aritmetiche. La differenza rispetto a composizione/topologia è di **grado** (un
calcolo invece di una lettura diretta), non di natura.

**Effetto sui claim.** «La geometria è l'unico asse dove il confronto è alla
pari» non è supportato dal codice, e va riscritto anche l'argomento sulla fusione
che vi si appoggia. Paradossalmente il risultato *migliora* nel raccontarlo: il
vision batte sulla geometria un modello che ha gli ingredienti in input. Ma la
**motivazione dichiarata è sbagliata**.

**Domanda aperta da chiudere prima del report.** Le PNG di `snapshot_train/` sono
renderizzate dagli stessi `.mat`. Se il **colore** delle stanze codifica `rType`
(non verificato in questa sessione), allora anche il vision ha `rType` in input e
la circolarità tocca anche lui su composizione e topologia. Va verificato: cambia
il senso di tutta la sezione.

**Terreno davvero neutro (due opzioni).** (a) Un asse che il grafo non può
strutturalmente rappresentare — forma non rettangolare, muri, aperture; (b) una
ablation «graph senza feature geometriche» (solo `rType`/`rEdge`) confrontata
sulla geometria.

---

### A6 — L'asse che porta il claim è una formula fatta a mano, mai validata

**Fatto.** La geometria è la media aritmetica di tre componenti con costanti
scelte a mano ([relevance.py:137-146](src/evaluation/relevance.py#L137-L146)):
`1-|Δarea|`, rapporto `min/max` sull'aspect, `1 - 0.5·L1` sulla distribuzione di
area per tipo, poi `/3`. Le tre componenti **sono** in [0,1] (verificato:
`footprint_area` normalizzata su 256², `footprint_aspect ≥ 1`,
`type_area_distribution` somma a 1).

**Ma non sono commensurabili** (interpretazione con esempio): l'area usa una
**differenza** su una quantità concentrata in una banda stretta, l'aspect un
**rapporto** molto più punitivo (aspect 2× → 0.5; area 2× → ~0.8), la
distribuzione una L1 su 13 bin. I pesi 1/3-1/3-1/3 sono arbitrari, senza analisi
di sensibilità.

**Evidenza di compressione della scala.** La baseline `hist`, il cui embedding è
il solo istogramma dei tipi e che quindi **non vede geometria**
([graph_evaluate.py:131-132](src/graph/evaluation/graph_evaluate.py#L131-L132)),
prende **0.894** di geometria. La scala utile dell'asse è quindi ≈[0.894, 1.0] e
il margine 0.002 è ~2% di quell'intervallo, sopra un descrittore cieco alla
geometria. **Il floor vero — nDCG di un ranking casuale — non è mai stato
misurato**: la domanda "quanto prende un sistema stupido?" non ha risposta nel
repo.

**Conseguenza.** Un ordinamento a 0.002 su un asse definito da una media a pesi
arbitrari di tre termini a scala diversa è **ribaltabile** cambiando i pesi — in
particolare togliendo `dist_sim`, che è il termine correlato alla composizione.

**Come si chiude (nessuna GPU).** Misurare l'nDCG geometria di un ranking casuale
e ricalcolare l'ordine vision/graph con 2-3 pesature alternative e con le tre
componenti prese singolarmente. Se l'ordine regge, il claim diventa robusto; se
si ribalta, va abbandonato.

---

### A7 — Zero dei quattro contratti critici è coperto da un test

**Fatto.** `tests/` contiene tre file: `test_loader.py`, `test_vision_encoder.py`
e `test_vision_retrieval.py`. Eseguito in sessione: `grep -c assert tests/*.py` →
**0, 0, 0**. I due "veri test" fanno solo `print`: passerebbero anche con shape
sbagliate. `test_vision_encoder.py:10` scarica pesi da HF, contro la regola
"nessun download" della guida di testing. Il terzo è l'entrypoint di
indicizzazione, già dichiarato tale.

Copertura dei contratti che il progetto dichiara critici: **0 su 4**.

| Contratto | Coperto | Oggi garantito da |
|---|---|---|
| riga↔nome fra `embeddings.npy` e `names.json` | no | commenti; nessun `load()` verifica nemmeno `len(emb)==len(names)` |
| disgiunzione degli split | no | i `.mat` |
| correttezza di nDCG/Recall/mAP | no | nulla — ed è il codice da cui escono **tutti** i numeri del report |
| ricaricabilità del checkpoint al variare di `raw_skip` | no | un `RuntimeError` di torch |

**Perché conta.** È la causa strutturale per cui i rilievi B4, B5 e B6 restano
latenti invece di essere intercettati.

**I quattro smoke test CPU che mancano** (nome · invariante):
1. `test_gallery_alignment` — riga `i` ↔ nome `i` dopo save→load, e `len` uguali.
2. `test_checkpoint_shape_contract` — salvare con `raw_skip=True` e ricaricare con
   `False` **deve** fallire; con lo stesso flag deve riuscire.
3. `test_metrics_reference_values` — nDCG/Recall/mAP su un ranking a mano con
   valori calcolati a penna, incluso il caso classe singleton.
4. `test_split_disjointness_and_stats_from_train` — i tre split non condividono
   nomi; le statistiche fittate sul train, applicate al valid, danno media
   **vicina ma non esattamente** 0 (la firma dell'assenza di leakage).

---

## 2. Rilievi a gravità media

### B1 — Le config su disco non riproducono le varianti migliori misurate

**Fatto.** Nessuno dei tre YAML contiene la propria configurazione vincente:
`temperature: 0.3` in [configs/graph_models/gcn.yaml:58](configs/graph_models/gcn.yaml#L58)
(misurata migliore: 0.2), `node_drop: 0.2` in `graph_sage.yaml:36` (migliore:
0.1), `flip_prob/rot_prob: 0.5` in `gat.yaml:38-39` (migliore: 0). Peggio: il
commento a `gcn.yaml:58-62` **argomenta a favore** di 0.3 con un'ipotesi che la
misura ha poi smentito.

**Impatto (verificato eseguendo il ponte YAML→flag).** Chi lancia
`03_train_gnn.sh gcn` senza secondo argomento allena `base`, cioè la
configurazione **peggiore**: media 0.897 invece di 0.913. Il numero di punta è
raggiungibile **solo** con `03_train_gnn.sh gcn tau02`. Se ne accorge chiunque
riparta dai config per il report. Già tracciato in `.claude/TODO.md`, quindi noto
e aperto.

### B2 — Le due gallery non sono identiche (67.453 vs 67.405)

**Fatto.** Il vision indicizza 67.453 PNG, il graph 67.405 grafi: le 48 piante di
differenza sono quelle senza record `.mat`
([graph_builder.py:182](src/graph/graph_builder.py#L182) le scarta).
Sull'**IDCG** l'effetto è **nullo** (hanno `valid=False`, similarità 0 su tutti
gli assi, e l'IDCG prende i top-K). Sul **DCG** no: solo il vision può recuperarle
e quando lo fa incassa gain **0** su tutti gli assi.

**Interpretazione (stima a spanne).** A tasso casuale, ≈10·48/67453 ≈ 0,007
righe-a-gain-zero per query nel top-10; una riga zero a metà classifica costa
~7% dell'nDCG di quella query → perdita attesa ≈ **0,0005**, cioè ~1/4 del
margine 0,002, **contro il vision**. Non ribalta il claim, ma è dello stesso
ordine del margine: **un margine di 0.002 non è difendibile con gallery diverse**.

**Come si chiude.** Ri-valutare la riga geometria del vision restringendo la
gallery ai 67.405 nomi — è lo stesso inner join che la late fusion richiede già.

### B3 — Nel partial le esclusioni sono diverse dal full

**Fatto.** [evaluate.py:259](src/vision/evaluation/evaluate.py#L259) costruisce le
righe con `exclude_self=False`, e **le stesse righe** alimentano anche le metriche
per-asse (`_accumulate_axes(..., exclude_self=False)`, `:137-171`), non solo il
self-recovery. Il full usa `exclude_self=True` (`:191`), come il graph.

Internamente il partial è **coerente** (il self sta su entrambi i lati, quindi
l'IDCG non è deflazionato), ma l'nDCG per-asse del partial **non è confrontabile**
con quello del full: contiene un hit a rank 1 con gain 1.0. Qualunque curva
"degrado full→partial" **sottostima il degrado**, e a f=0.0 il partial non
riproduce la tabella full.

**Nessun numero pubblicato è invalidato**: dei partial si riportano MRR e Recall
di self-recovery, che con il self dentro sono legittimi per costruzione.

### B4 — Il ponte YAML→flag scarta i booleani in silenzio, e la logica è duplicata

**Fatto (dimostrato per esecuzione).** In
[scripts/graph/_common.sh:40-41](scripts/graph/_common.sh#L40-L41) c'è
`if isinstance(v, bool): continue`: aggiungendo `amp: true` e `foo_flag: false` a
una copia del YAML, l'output del ponte è **byte-identico** e l'exit code 0.
Una chiave **non** booleana inesistente, invece, fa morire il training con
"unrecognized arguments". L'asimmetria è proprio sul caso più silenzioso.

Il dict `special` esiste in **due copie** (`_common.sh:28-33` e
[04_eval_gnn.sh:57-61](scripts/graph/04_eval_gnn.sh#L57-L61)) e sui booleani non
registrati si comportano in modo **opposto**: train li scarta, eval emette
`--flag True`, che su uno `store_true` è un errore argparse.
Verificato che **oggi** i flag architetturali coincidono fra training ed eval →
i checkpoint si ricaricano; il difetto è latente, non attuale.

**Contrasto interno.** Gli stessi script hanno una guardia difensiva con `exit 1`
sulle varianti non definite (`03:97-102`, `04:114-119`), messa lì dopo che una
variante saltata era già costata un job. La stessa logica non è stata applicata
al ponte, che è il punto di guasto più silenzioso dei quattro livelli.

### B5 — `raw_skip`: nessun controllo, errore criptico

**Fatto (riprodotto).** `raw_skip` aggiunge `in_dim` alla larghezza di `proj`
([src/graph/models/base.py:138-144](src/graph/models/base.py#L138-L144)): misurato
`proj.weight [128,147]` con il flag, `[128,128]` senza. In eval il caricamento è
`encoder.load_state_dict(...)` a nudo
([graph_evaluate.py:90](src/graph/evaluation/graph_evaluate.py#L90)) e l'utente
ottiene un `RuntimeError: size mismatch for proj.weight`. Muto sulla causa
(`raw_skip` derivato dal YAML, non dal checkpoint), mentre lo stesso
`load_encoder` ha messaggi curati per checkpoint e statistiche mancanti.

**Impatto.** Fallisce **rumorosamente**: nessun numero sbagliato in silenzio.
Scenario concreto: chi cambia `raw_skip: false` in un YAML vede morire tutte le
valutazioni sui checkpoint già allenati.

### B6 — Costruzione dei grafi: due silenziosità

**Fatto (riprodotto su tensori sintetici).**
(a) [graph_builder.py:128-130](src/graph/graph_builder.py#L128-L130) fa
`clamp(0, 9)` sugli id di relazione, che sono copiati **verbatim** da `rEdge`
senza validazione (`rplan_metadata.py:164-169`): un id invalido (`-1`, `12`)
diventa una **classe valida** senza errore né conteggio.
(b) `:135-137` fa `to_undirected(..., reduce="mean")` su `edge_attr` one-hot: se
la stessa coppia (i,j) compare con **due relazioni diverse**, l'attributo diventa
`0.5/0.5`, cioè non più one-hot.

**Incoerenza documentale collegata.** `.claude/shared/dataset.md:96-98` afferma
`edge_index [2, 2E]` e `edge_attr [2E, 10]` — vero **solo** se nessuna coppia
collassa; il docstring di `graph_builder.py:133-134` prevede invece
esplicitamente archi paralleli. Le due affermazioni non possono essere entrambe
vere.

**Impatto.** `edge_attr` è consumato **solo da GAT**: un attributo mescolato non
fa crashare nulla e produce numeri leggermente diversi senza traccia. Poiché il
progetto ha già scritto che «GAT è il più capace ed è il peggiore», una
spiegazione alternativa a quella diagnosi **non è esclusa**.

**Chiude il dubbio in ~1 minuto di CPU** (non eseguito: legge `graphs.pt`):
```bash
python -c "from src.graph.graph_dataset import RplanGraphDataset as D; ds=D(); \
bad=[d.name for d in ds if d.edge_attr.numel() and ((d.edge_attr.sum(1)-1).abs()>1e-6).any()]; \
print(len(bad), bad[:5])"
```

---

## 3. Rilievi minori (robustezza e documentazione)

- **C1 — MRR partial contaminato.** Se nessuna stanza viene rimossa (f=0.0, o
  `semantic`/`topology` su piante senza stanze rimovibili) la query è la pianta
  completa e contribuisce MRR 1.0. `n_empty` viene **stampato ma non escluso**
  dalla media ([evaluate.py:255-268](src/vision/evaluation/evaluate.py#L255-L268),
  stampa a `:300`) → l'MRR di `semantic`/`topology` è un upper bound.
- **C2 — Due liste di varianti mantenute a mano.** `ALL_VARIANTS`
  (`03_train_gnn.sh:52`) e `KNOWN_VARIANTS` (`04_eval_gnn.sh:101`): le guardie
  proteggono ciascuno script al suo interno, ma nessuna confronta le due liste.
  Questa esatta desincronizzazione è già costata una variante (`selgeom`).
- **C3 — Un summary troncato sparisce dalla tabella di selezione.**
  `scripts/graph/03_train_gnn.sh:149-150` ha `except Exception: continue` sul
  `json.load` dei `training_summary.json`: la tabella di riepilogo con cui si
  scegle la variante vincente può avere una riga in meno senza che nessuno lo
  sappia.
- **C4 — `RoomMeta.footprint` è una trappola armata.** `dataset.md:37` avverte
  che `gtBox` ha gli **assi scambiati**; il modulo che lo legge non lo dice e
  chiama il campo `(xmin, ymin, xmax, ymax)`
  ([rplan_metadata.py:178-180](src/data/rplan_metadata.py#L178-L180)). Oggi è
  innocuo (i soli consumatori sono area e aspect, invarianti allo scambio), ma il
  primo che ne usi le coordinate sbaglia.
- **C5 — Un `except Exception` senza log.** `retrieval_visualization.py:118-122`:
  un PNG corrotto diventa un riquadro grigio in una figura del report **senza
  traccia del path**.
- **C6 — Due cose non documentate.** `scripts/vision/08_visualize.sh` non è
  citato in nessun `.md` (`COMANDI.md` documenta al suo posto il comando python
  nudo); e il vision ha `wandb.enabled: true` + `mode: online` come default
  committato senza fallback (`train_projection.py:124-144`) → su un nodo senza
  rete il job muore all'avvio, e `COMANDI.md` dichiara il contrario (ma solo per
  il ramo graph).

---

## 4. Verificato e **a posto** (contro-evidenza, va detta)

Non tutto è un problema: questi punti erano sospetti e sono risultati corretti.

- **Le 2000 query sono davvero le stesse fra i due rami.** Verificato per
  costruzione *e* empiricamente sui due json: `names.json` (67.405) ⊂
  `image_paths.json` (67.453), solo-vision = 48, solo-graph = 0, e filtrando la
  lista vision sull'insieme graph si riottiene **esattamente** l'ordine graph.
  Entrambi i rami usano `sorted()` sulla stessa cartella piatta, filtrano il pool
  con lo stesso predicato e campionano con `random.Random(42).sample(pool, 2000)`
  → stesse posizioni su pool identici. Il claim "stesse 2000 query" è **onesto**.
  ⚠️ Ma niente lo garantisce a runtime: nessun assert.
- **IDCG e self sono coerenti.** `gallery_gains` esclude `qi` **esattamente
  quando** lo esclude anche la lista recuperata, in entrambi i rami e in entrambe
  le modalità (`evaluate.py:144-154`, `axis_metrics.py:66-76`): nessuna
  deflazione sistematica dell'nDCG.
- **Esclusioni singleton identiche fra i rami.** `composition_relevant` e
  `topology_relevant` fanno AND con `valid` (`relevance.py:160,165`): le 48
  piante extra del vision non possono mai essere rilevanti, quindi i conteggi di
  query escluse coincidono per costruzione (corrobora il 10/313 dei log).
- **Il ramo graph rispetta il vincolo train-only** sulle statistiche, e la probe
  di selezione usa **letteralmente lo stesso codice metrico** della valutazione
  finale (`axis_metrics.py` condiviso) → sonda e test misurano la stessa cosa.
- **Nessun leakage nel training della head**: le coppie usano solo gli split
  `train`+`valid` (`projection_pairs.py:123-134`), il test è fuori.
- **Il masking partial è appaiato fra encoder**: `random.Random(seed + qi)` con
  `qi` = riga di gallery → stesse immagini degradate per tutti.
- **Whitening applicato in modo simmetrico** a gallery e query, e la query del
  full è ri-encodata dal PNG con lo stesso transform → il self-match è genuino.
- **`compileall src tests` → rc=0**; import dei moduli graph/data OK;
  `NUM_RELATION_TYPES=10`, `NODE_FEATURE_DIM=19` come documentato.
- **Le dipendenze fra stage in `COMANDI.md` corrispondono** agli entrypoint reali
  di tutti gli script dei due rami (verificate una per una).
- **Nessun `except` nudo in `src/`** (solo i due discussi in C5).

---

## 5. Contraddizioni interne alla documentazione (da sanare a parte)

1. `CLAUDE.md` vincolo 1 «statistiche dal solo train» **vs** whitening vision
   trasduttivo → A2.
2. `CLAUDE.md` vincolo 5 «l'unico asse alla pari è la geometria» **vs** GT
   geometrica derivabile dagli input del grafo → A5.
3. `dataset.md:96-98` (`edge_attr [2E, 10]`) **vs** docstring di
   `graph_builder.py:133-134` (archi paralleli previsti) → B6.
4. «±0.003 è rumore» **vs** «0.005 è il risultato più significativo della
   tabella» → A4.
5. `COMANDI.md` «i job girano senza rete» **vs** `wandb online` nel vision → C6.

---

## 6. Limiti di questo audit (cosa **non** è stato verificato)

- Nessun `.mat`, `graphs.pt`, `.npy` di embedding o log è stato letto (vincolo di
  progetto): quindi **non** è verificato se su RPLAN esistano davvero id di
  relazione fuori range o coppie con relazioni diverse (B6), né l'entità reale di
  `n_empty` nel partial (C1).
- L'entità numerica del leakage di whitening (A2), del winner's curse (A1) e
  della penalità da gallery diversa (B2) sono **stime**, non misure.
- Nessuna run è stata lanciata: nessuna delle conclusioni "il numero cambierebbe
  di X" è misurata.
- Un'affermazione di un reviewer **non corroborata** e quindi scartata: che il
  «graph 0.946» venisse da un checkpoint selezionato su val-loss. `status.md:156`
  lo attribuisce a `gat/nosym` dell'ablation OFAT, che è selezionata con la
  probe. Il problema reale su quel numero è un altro: è un **massimo per-asse fra
  varianti** (→ A1).
- **Duplicati o quasi-duplicati in RPLAN** non sono stati indagati: se esistono,
  gonfiano tutti gli assi. Punto aperto.

---

## 7. Ordine di intervento suggerito (quando si passerà ai fix)

Criterio: prima ciò che costa poco e cambia il *significato* dei numeri.

**Senza GPU, subito**
1. Misurare il **floor**: nDCG per-asse di un ranking casuale (A6). Dà una scala
   a tutti i numeri del progetto.
2. Salvare i **per-query** e fare il test appaiato di Wilcoxon (A4). Dice quali
   conclusioni sopravvivono.
3. Analisi di **sensibilità ai pesi** della geometria (A6).

**Solo ri-valutazioni (nessun retraining)**
4. Griglia vision su `eval.split=valid`, poi test una volta sola (A1).
5. Whitening fittato sul solo train, e misura della differenza (A2).
6. Gallery ristretta all'inner join di 67.405 nomi (B2).
7. Partial con `exclude_self=True` sulle metriche per-asse, per avere la curva di
   degrado confrontabile col full (B3).

**Con training breve**
8. Head vision selezionata con una probe di retrieval sul valid (A3).

**Igiene, in parallelo e a costo quasi nullo**
9. I quattro smoke test di A7 (proteggono tutto il resto).
10. Guardia sui booleani non registrati nel ponte + de-duplicazione del dict
    `special` (B4); messaggio esplicito sul mismatch `raw_skip` (B5).
11. Allineare i tre YAML alle varianti vincenti e correggere il commento smentito
    (B1).
12. Riscrivere i due vincoli contraddetti in `CLAUDE.md` (A2, A5) e le altre
    contraddizioni della §5.

**Da decidere prima di scrivere il report**
13. La verifica sul colore delle stanze nelle PNG (A5): se il render codifica
    `rType`, la sezione sulla circolarità va riscritta per **entrambi** i rami.
