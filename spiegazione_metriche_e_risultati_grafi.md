# RAMO GRAPH

## Spiega bene le metriche utilizzate:

### - cosa misurano esattamente

**Il setup, prima delle metriche.** La *gallery* è l'insieme di tutte le piante indicizzate: 67.405 grafi RPLAN. Una *query* è una pianta di cui chiediamo "trovami quelle che le somigliano": ne usiamo 2000, prese dallo split ufficiale di **test**. Il sistema trasforma ogni pianta in un vettore (*embedding*), ordina l'intera gallery per similarità con la query e restituisce le prime K. Le metriche confrontano quest'ordine con una "verità" che **non viene dagli embedding** ma dai metadati `.mat` di RPLAN: quali stanze ci sono, come sono collegate, che forma ha l'appartamento.

Questo punto è cruciale: il modello propone un ordine, i `.mat` dicono quale sarebbe stato l'ordine giusto, e la metrica misura la distanza tra i due.

La "verità" (*rilevanza*) esiste in due forme, e ogni metrica ne usa una:

- **graduata**: un numero in [0,1] che dice *quanto* due piante si somigliano su un asse (0.87 = molto simili, 0.20 = poco);
- **binaria**: la *classe di equivalenza esatta*, cioè l'insieme delle piante che hanno **esattamente** la stessa composizione (o la stessa topologia) della query. O ci sei o non ci sei.

---

#### nDCG@K — "quanto è buono l'ordine dei primi K, rispetto al migliore ordine possibile"

Si costruisce in tre passi.

**1) DCG (Discounted Cumulative Gain).** Si sommano i *gain* dei primi K risultati, dove il gain è la similarità graduata (in [0,1]) tra quel risultato e la query. Ogni gain viene però **scontato in base alla posizione**, con peso `1/log₂(rank+1)`:

| posizione | peso |
|---|---|
| 1ª | 1.00 |
| 2ª | 0.63 |
| 5ª | 0.39 |
| 10ª | 0.29 |

*Perché lo sconto:* in un motore di ricerca l'utente guarda dall'alto. Un risultato perfetto in prima posizione vale molto più dello stesso risultato in decima. Lo sconto logaritmico è la convenzione standard: decresce, ma dolcemente.

**2) IDCG (Ideal DCG).** Lo stesso identico calcolo, ma sull'ordinamento **ideale**: si prendono i K gain più alti presenti nell'**intera gallery** e li si mette in ordine decrescente. È il massimo teoricamente raggiungibile per quella query.

**3) nDCG = DCG / IDCG**, quindi sempre in [0,1]. Vale 1.0 solo se il modello ha restituito esattamente le K piante migliori, nell'ordine migliore.

> **Dettaglio che rende la metrica onesta:** l'IDCG si calcola sull'**intera gallery** (67.405 piante), non sui risultati che il modello ha restituito. Il denominatore quindi non dipende da quanto il modello è stato bravo, e la metrica **può davvero fallire**. Nella prima versione del progetto le metriche venivano calcolate solo sui top-k recuperati: era un ragionamento circolare (il modello veniva giudicato sul materiale che aveva scelto lui) e infatti erano sature, davano ~1.0 anche a un retrieval casuale.

#### Recall@K — "delle posizioni disponibili, quante ne ho riempite con roba giusta"

Serve l'insieme binario dei rilevanti = la classe di equivalenza esatta.

```
Recall@K = |top-K ∩ rilevanti| / min(K, numero_totale_di_rilevanti)
```

*Perché quel `min(K, ·)` al denominatore:* se una query ha 5.000 piante equivalenti nella gallery e K=10, il massimo ottenibile con la formula classica sarebbe 10/5000 = 0.002. La metrica sarebbe schiacciata sullo zero e illeggibile, non per colpa del modello ma per la dimensione della classe. Con `min(K, ·)` il denominatore diventa 10, e "ho riempito tutti e 10 gli slot con piante equivalenti" vale correttamente 1.0.

Recall **non guarda l'ordine** dentro i K: 3 rilevanti in posizione 1,2,3 o in posizione 8,9,10 danno lo stesso valore.

#### mAP@K — "e li ho messi anche in cima?"

`AP@K` (Average Precision) media le *precision* calcolate nelle posizioni dove compare un rilevante. Con un esempio a K=10 e 3 rilevanti:

- rilevanti in posizione **1, 2, 3** → (1/1 + 2/2 + 3/3) / 3 = **1.00**
- rilevanti in posizione **8, 9, 10** → (1/8 + 2/9 + 3/10) / 3 = **0.21**

Stesso Recall (3 trovati su 3), mAP quasi cinque volte diverso. Il mAP è quindi la metrica **sensibile all'ordine dentro l'insieme dei rilevanti**. La "m" sta per *mean*: la media su tutte le query.

#### Due accorgimenti che valgono per tutte

- **Self-match escluso.** La query è essa stessa una riga della gallery, quindi il primo risultato sarebbe sempre sé stessa con similarità 1.0. Viene esclusa dai risultati, dai rilevanti e dall'IDCG, altrimenti tutti i punteggi sarebbero gonfiati.
- **Query singleton escluse (solo da Recall/mAP).** Se una query non ha *nessun* rilevante esatto (è l'unica pianta con quella composizione o quella topologia), Recall e mAP non sono definite. Quelle query vengono **escluse dalla media, non contate come zero**: contarle zero significherebbe punire il modello per un compito impossibile. Nelle run: 10 query su 2000 sulla composizione, **313 su 2000 sulla topologia**.

---

### - perché vengono usate

**Perché tre metriche e non una.** Misurano tre proprietà diverse di una stessa lista di risultati, e insieme raccontano la storia completa:

| metrica | domanda a cui risponde | tipo di rilevanza |
|---|---|---|
| **nDCG@K** | l'ordine complessivo è vicino a quello ideale? | graduata |
| **Recall@K** | quante piante *esattamente* equivalenti ho pescato? | binaria |
| **mAP@K** | le ho anche messe in alto? | binaria + posizione |

L'**nDCG è la metrica primaria** per tre motivi: usa tutta l'informazione disponibile (non solo "uguale/diverso" ma "quanto simile"), non richiede di scegliere una soglia arbitraria, ed è l'unica che funziona anche sull'asse **geometria**, che essendo continuo non ha classi di equivalenza.

Recall e mAP restano perché sono **molto più leggibili**: "il 50% delle volte il primo risultato ha esattamente lo stesso schema di adiacenze" comunica qualcosa che un nDCG di 0.789 non comunica.

**Perché NON un unico punteggio fuso.** La prima versione del progetto fondeva i tre assi in un solo numero con pesi e una soglia di rilevanza a 0.5. Era satura e inutile: **il 96% delle coppie casuali di piante superava quella soglia**, quindi Recall@5 dava ≈1.0 anche a un retrieval completamente casuale. Una metrica che premia allo stesso modo un buon modello e il caso non misura nulla. Da qui la riscrittura: niente fusione, niente pesi, niente soglia, rilevanza **scomposta per asse** e valutazione **contro l'intera gallery**.

C'è anche un motivo diagnostico: con un numero solo sapresti che il modello A è meglio del modello B, ma non *perché*. Con tre assi separati scopri che GCN vince sulla topologia mentre SAGE vince sulla geometria — informazione che orienta le decisioni successive.

**Perché queste e non altre.** nDCG, Recall e mAP sono lo standard dell'*information retrieval*: rendono i numeri confrontabili con la letteratura e, cosa altrettanto importante, con il **ramo vision** di questo stesso progetto, che usa le identiche funzioni dal core condiviso [metrics.py](src/evaluation/metrics.py). Se un giorno si fondono i due rami, i numeri sono già sulla stessa scala.

**Perché K ∈ {1, 5, 10, 100}.** K=1 è il caso più severo ("il primo risultato è azzeccato?"), K=10 corrisponde a una schermata di risultati, K=100 mostra il comportamento in profondità. Guardarli insieme rivela cose che un singolo K nasconde (vedi il trade-off su K più avanti).

---

### - cosa indicano le tre tipologie: composizione, topologia e geometria

**L'idea di fondo:** "due piante si somigliano" non è una domanda sola. Un architetto può cercare per *quali ambienti ci sono*, per *come sono collegati*, o per *che forma ha l'appartamento*. Sono tre domande indipendenti — due piante possono avere le stesse identiche stanze disposte in modo completamente diverso — quindi vanno misurate separatamente.

#### 1. COMPOSIZIONE — "ci sono le stesse stanze?"

- **Feature:** l'istogramma dei tipi di stanza, un vettore di **13 conteggi** (i 13 tipi RPLAN: LivingRoom, MasterRoom, Kitchen, Bathroom, DiningRoom, ChildRoom, StudyRoom, SecondRoom, GuestRoom, Balcony, Entrance, Storage, Wall-in). Esempio: `[1 soggiorno, 1 cucina, 2 camere, 1 bagno, ...]`.
- **Similarità:** *Weighted Jaccard* = somma dei minimi / somma dei massimi.
  Esempio concreto: A = 2 camere + 1 bagno, B = 1 camera + 1 bagno.
  minimi = 1+1 = 2, massimi = 2+1 = 3 → **similarità 0.67**.
  *Perché questa e non la distanza euclidea:* gestisce naturalmente i conteggi (un *multiset*, cioè un insieme in cui un elemento può comparire più volte), sta sempre in [0,1] senza normalizzazioni arbitrarie, e vale esattamente 1.0 per due piante identiche.
- **Classe di equivalenza:** le piante con istogramma **identico**. Sono classi **grandi**: solo lo 0,5% delle query è singleton, cioè quasi ogni pianta ha molte "sorelle" con la stessa composizione. Questo dettaglio tornerà — è la causa del problema principale delle ultime run.
- **Cosa cattura:** il *programma funzionale* dell'appartamento — "trilocale con doppio bagno e balcone". Ignora completamente **dove** stanno le stanze.

#### 2. TOPOLOGIA — "sono collegate allo stesso modo?"

- **Feature:** l'adiacenza **tipizzata**, un vettore di **91 conteggi**. 91 = tutte le coppie non ordinate di 13 tipi (13·14/2). Ogni collegamento tra due stanze viene tradotto nella coppia dei loro *tipi*, ordinata: `(Cucina, Soggiorno)` è la stessa voce in qualunque pianta.
- **Perché tipizzata e non per indice di stanza:** rende la misura **invariante alla permutazione**. Se rinumero le stanze di una pianta, la pianta è la stessa; ma un vettore basato sugli indici darebbe due descrizioni diverse per lo stesso appartamento. Usando i tipi, il problema sparisce.
- **Similarità:** stesso Weighted Jaccard.
- **Classe di equivalenza:** adiacenza identica. Qui le classi sono **molto più piccole**: il **15,7% delle query è singleton** (313 su 2000). Il match topologico esatto è raro, quindi su questo asse Recall e mAP sono strutturalmente bassi per tutti e ci si appoggia soprattutto all'nDCG graduato.
- **Cosa cattura:** la struttura relazionale — il grafo vero e proprio. Cucina attaccata al soggiorno, bagno accessibile dal disimpegno. È l'informazione più vicina a come un architetto ragiona sulla distribuzione, ed è quella che un semplice elenco di stanze non contiene.

#### 3. GEOMETRIA — "hanno la stessa forma e le stesse proporzioni?"

Media di tre componenti, ciascuna in [0,1]:

1. **Area del footprint:** `1 − |area_A − area_B|`, con le aree normalizzate sulla griglia 256×256 → sta in [0,1].
2. **Aspect ratio:** `min/max` dei due aspect (lato lungo / lato corto, sempre ≥ 1). Vale 1.0 se le due piante hanno le stesse proporzioni, indipendentemente dalla scala.
3. **Distribuzione dell'area per tipo:** `1 − 0.5·Σ|distA − distB|`. Le due distribuzioni sommano a 1, quindi la somma degli scarti assoluti sta in [0,2]: il fattore 0.5 la riporta in [0,1]. (È la *total variation distance* trasformata in similarità.) Cattura "quanta parte della casa è dedicata alle camere rispetto ai servizi".

- **È un asse continuo:** due piante non hanno mai *esattamente* la stessa area, quindi la classe di equivalenza esatta non esiste. Di conseguenza **si calcola solo l'nDCG**, e nelle tabelle Recall e mAP compaiono come `—`. Non è una dimenticanza: è che quelle metriche lì non sono definite.
- **Cosa cattura:** scala e proporzioni — appartamento grande e allungato contro piccolo e compatto.

#### Perché proprio questi tre

Coprono i tre livelli con cui si descrive una pianta: **cosa contiene** (semantica), **come è connessa** (struttura), **che forma ha** (metrica). Sono anche gli stessi tre assi usati dal ramo vision, il che rende i due rami direttamente confrontabili.

> ⚠️ **Caveat di circolarità, da dichiarare sempre.** Composizione e topologia derivano da `rType` e `rEdge`, che sono **esattamente** ciò che il ramo graph riceve in ingresso (le feature dei nodi e gli archi). Il grafo riceve in input ciò che la ground truth misura in output: su quei due assi i suoi risultati vanno letti come **upper bound**, non come un confronto alla pari con il ramo vision, che quelle informazioni deve inferirle dai pixel. Sull'asse geometria il problema è molto più debole (le feature geometriche dei nodi non sono la stessa cosa dell'area del footprint). È una questione aperta del progetto: le tre uscite possibili sono dichiararlo apertamente, valutare il graph solo sulla geometria, oppure estrarre il grafo dall'immagine.

---

### - cosa indica ogni metrica di ogni tipologia

Le combinazioni utili sono sette (la geometria non ha Recall/mAP). Numeri dall'ultima run, **gp_04 74184**, encoder GCN salvo dove indicato:

| asse | metrica | valore | come si legge in italiano |
|---|---|---|---|
| **composizione** | nDCG@10 | 0.934 | i primi 10 risultati hanno un mix di stanze molto vicino al meglio ottenibile |
| composizione | Recall@1 | 0.793 | nel **79% dei casi il primo risultato ha esattamente lo stesso istogramma** di stanze |
| composizione | mAP@10 | 0.632 | le piante con composizione identica non solo compaiono, ma stanno mediamente in alto |
| **topologia** | nDCG@10 | 0.789 | i primi 10 hanno schemi di collegamento molto simili (baseline: 0.643) |
| topologia | Recall@1 | 0.507 | **una volta su due il primo risultato ha esattamente lo stesso schema di adiacenze** (baseline: 0.027) |
| topologia | mAP@10 | 0.299 | più basso perché i rilevanti esatti sono pochi (15,7% di query singleton) e difficili da mettere tutti in cima |
| **geometria** | nDCG@10 | 0.938 | forma e proporzioni dei primi 10 sono molto vicine all'ideale |
| geometria | Recall/mAP | — | non definite: asse continuo, nessuna classe di equivalenza |

**Come leggere i valori assoluti.** Non vanno confrontati *tra assi*: un nDCG di 0.938 sulla geometria non è "meglio" di 0.789 sulla topologia, perché i due assi hanno difficoltà intrinseche diverse. La geometria parte alta anche per un sistema mediocre (tante piante hanno area e proporzioni simili, quindi anche un ordinamento così così raccoglie gain alti); la topologia è severa. Il confronto sensato è **tra modelli sullo stesso asse**, oppure **contro la baseline**.

**Il ruolo della baseline `hist`.** È un sistema *training-free*: usa direttamente l'istogramma dei tipi (13 dimensioni, L2-normalizzato) come embedding, senza alcuna rete. Sulla composizione ottiene **1.000 su tutte le metriche** — ma non perché sia un buon modello: è l'**oracolo per costruzione**, perché ordina la gallery usando esattamente la feature con cui viene giudicato. Serve a due cose: verificare che l'idraulica (FAISS + metriche) funzioni, e fornire un termine di paragone onesto sugli **altri due assi**, dove non ha vantaggi. Ed è lì che si vede il valore delle GNN:

- topologia: **0.789 vs 0.643** di nDCG, ma soprattutto **Recall@1 0.507 vs 0.027 — 19 volte meglio**;
- geometria: **0.938 vs 0.894**.

Il messaggio è netto: sapere *quali* stanze ci sono (l'istogramma) non dice quasi nulla su *come sono collegate*. Il message passing della GNN, che propaga informazione lungo gli archi, cattura struttura che un semplice conteggio non può avere.

---

## Spiega i risultati e le metriche delle ultime run:

### - cosa è stato cambiato e perché

**Che cosa è cambiato, in una riga:** il criterio con cui si sceglie il modello finale tra tutte le epoche di training. Prima era la *val-loss*, ora è l'*nDCG di retrieval*.

**Come funzionava prima.** L'encoder si allena con **InfoNCE**, una loss *contrastiva self-supervised*: per ogni pianta si generano due viste leggermente perturbate (togliendo nodi, archi o feature), il modello deve riconoscerle come vicine (*positivi*) e allontanarle da tutte le altre piante del batch (*negativi*). Alla fine si teneva il checkpoint con la **val-loss minima**, con early stopping sulla stessa. È la prassi standard nel machine learning.

**Il problema scoperto.** La classifica dei modelli secondo la val-loss era l'**esatto contrario** di quella secondo il retrieval:

| encoder | val-loss (più bassa = meglio) | nDCG@10 (più alto = meglio) |
|---|---|---|
| GAT | **2.08** ← la migliore | **0.856** ← la peggiore |
| SAGE | 2.22 | 0.865 |
| GCN | **2.73** ← la peggiore | **0.886** ← la migliore |

**Perché succede.** InfoNCE fa *instance discrimination*: insegna a distinguere **ogni singola pianta da tutte le altre**. Ma in un batch da 256 piante ce ne sono molte con la stessa composizione o la stessa topologia della query — lo sappiamo dai dati: sulla composizione **solo lo 0,5% delle piante è singleton**, cioè le classi di equivalenza sono grandi. La loss tratta quelle piante come **negativi** e le allontana attivamente: sta spingendo via proprio ciò che la valutazione considera **rilevante**.

Conseguenza controintuitiva ma logica: **migliorare la loss può peggiorare il retrieval**. E selezionare i pesi su quella loss premia sistematicamente il modello sbagliato — quello più bravo a separare piante che dovrebbero stare vicine.

> **Un'analogia.** È come addestrare qualcuno al riconoscimento facciale dicendogli "queste due foto sono la stessa persona, tutte le altre no" — ma nel mazzo ci sono gemelli e fratelli. Più diventa bravo a distinguere i gemelli, più il suo giudizio diventa inutile se il compito vero è "trovami persone che si somigliano".

**Un tentativo che non ha funzionato, prima di capirlo.** Nel primo giro nessun encoder attivava l'early stopping in 50 epoche, quindi sembravano sotto-allenati: le epoche sono state alzate a 150–200. Risultato: GCN +0.003, GAT invariato, **SAGE peggiorato** (composizione 0.912 → 0.904). Allenare tre volte più a lungo non ha dato nulla, perché il problema non era la quantità di training ma **l'obiettivo su cui si stava ottimizzando**.

**La soluzione: la sonda di retrieval** ([retrieval_probe.py](src/graph/evaluation/retrieval_probe.py)). È un mini-retrieval eseguito ogni epoca sul set di **validazione**, con lo **stesso protocollo della valutazione finale**: gallery campionata di 5.000 piante, 500 query prese dalle righe della gallery, self escluso, stesse metriche per-asse. Il suo nDCG guida best-checkpoint ed early stopping al posto della val-loss.

Tre scelte di design che la rendono praticabile e corretta:

1. **Gallery e ground truth fisse, calcolate una volta sola.** Il costo vero non è il forward della rete ma la costruzione della rilevanza dai `.mat`; essendo la gallery fissa, si paga una volta (30 s) e poi ogni sonda costa **2,6 s**, circa il 20% di overhead per epoca. Come effetto collaterale, misurare sempre sulle **stesse** query rende il confronto tra epoche *appaiato*, quindi molto più sensibile.
2. **Query sempre dal valid, mai dal test.** Il test non deve scegliere nulla, altrimenti i numeri finali sarebbero ottimisti.
3. **Stesso codice della valutazione finale.** Gli helper di accumulo sono stati estratti in [axis_metrics.py](src/graph/evaluation/axis_metrics.py) e sono condivisi tra sonda e `graph_evaluate.py`: la metrica che **sceglie** i pesi e quella che li **giudica** sono letteralmente lo stesso codice. Se divergessero, selezionare sulla sonda non avrebbe senso.

La val-loss continua a essere calcolata e stampata, ma solo come **diagnostica** (dice se il contrastivo sta convergendo); non decide più nulla.

---

### - cosa ha comportato questo cambiamento

**1. Ha reso visibile la patologia.** Con la sonda attiva si vede quello che prima era invisibile: i modelli raggiungono il massimo di retrieval **molto presto** e poi peggiorano, mentre la loss continua tranquillamente a scendere. Dal giro pulito **gp_03 74121**:

| encoder | picco nDCG (sonda) | early stop | val-loss al picco → a fine corsa |
|---|---|---|---|
| GAT | **epoch 3** (0.8432) | 23 | 2.699 → 2.276 (continua a migliorare) |
| GCN | epoch 12 (0.8619) | 32 | 2.905 → 2.758 (idem) |
| SAGE | epoch 40 (0.8459) | 60 | 2.259 → 2.204 (idem) |

GAT tocca il massimo dopo **3 epoche su 200**. Nel giro precedente, selezionato sulla val-loss, si era allenato per ~147 epoche **allontanandosi** dal buon retrieval — e la val-loss lo premiava per questo.

**2. Ha confermato il meccanismo, asse per asse.** Il degrado dopo il picco non è uniforme, e la sua forma è la prova che la diagnosi è giusta. Dal picco all'ultima epoca:

| encoder | composizione | topologia | geometria |
|---|---|---|---|
| GAT | 0.891 → 0.859 (**−0.032**) | 0.713 → 0.694 (**−0.019**) | 0.926 → 0.931 (**+0.005**) |
| GCN | 0.910 → 0.900 (−0.010) | 0.748 → 0.740 (−0.008) | 0.928 → 0.930 (+0.002) |
| SAGE | 0.878 → 0.867 (−0.011) | 0.725 → 0.716 (−0.009) | 0.934 → 0.934 (0.000) |

Tutti e tre gli encoder mostrano lo stesso schema: **composizione e topologia calano, la geometria no**. È esattamente ciò che il meccanismo prevede — composizione e topologia hanno grandi classi di equivalenza, quindi tanti falsi negativi che la loss allontana; la geometria è continua, non ha classi, quindi non ha falsi negativi strutturali e continua a migliorare.

**3. L'early stopping ora scatta presto — ma non è prematuro.** Lo stop cade sempre a `picco + 20` (la `patience`). Verifica: in tutti e tre i casi **nessuna delle 20 epoche successive al picco lo supera** (0/20). Se fosse rumore su una curva ancora in salita, ci aspetteremmo circa metà dei valori sopra il picco. GAT degrada davvero (−0.015, circa 4 volte la deviazione standard della curva); GCN va in **plateau** (oscilla entro ±0.002 dopo l'epoca 7); SAGE cala lievemente.

**4. Validazione indipendente.** Il ranking della sonda — GCN 0.8619 > SAGE 0.8459 > GAT 0.8432 — riproduce il ranking sul test set, cioè **quello che la val-loss invertiva**. È la prova che la sonda misura la cosa giusta.

---

### - come migliorano/peggiorano le metriche e perché

Confronto **appaiato** (identiche 2000 query, `seed 42`, stessa gallery): cambia solo il criterio con cui è stato scelto il checkpoint.

#### Il quadro d'insieme (nDCG@10, media dei tre assi)

| encoder | selezione su val-loss | selezione su nDCG | Δ | comportamento della curva |
|---|---|---|---|---|
| GCN | 0.8860 | 0.8870 | **+0.001** | plateau → non c'era nulla da recuperare |
| SAGE | 0.8647 | 0.8750 | **+0.010** | calo lieve dopo il picco |
| GAT | 0.8557 | 0.8713 | **+0.016** | degrado netto dopo il picco |

**Il punto più importante: il guadagno è proporzionale al danno.** GCN, che andava in plateau, non guadagna nulla — il checkpoint scelto sulla val-loss (epoca 118) stava già sullo stesso altopiano. GAT, che degradava davvero, recupera di più. Questa proporzionalità è ciò che trasforma il risultato da coincidenza a **conferma**: la sonda aveva predetto *chi* aveva qualcosa da recuperare, e il test lo ha confermato.

Per dare la scala: il passaggio da 50 a 150 epoche (stesso criterio, solo più training) valeva ±0.003, cioè rumore. Qui GAT guadagna 0.016 di media e 0.030 sulla topologia — un ordine di grandezza sopra.

#### Il dettaglio per asse (GAT, il caso più chiaro)

| asse | prima | dopo | Δ |
|---|---|---|---|
| composizione | 0.890 | 0.915 | **+0.025** |
| topologia | 0.731 | 0.761 | **+0.030** |
| geometria | 0.946 | 0.938 | **−0.008** |

#### Le metriche di ranking mostrano l'effetto molto più chiaramente

| encoder | metrica | prima | dopo | variazione relativa |
|---|---|---|---|---|
| GAT | composizione mAP@10 | 0.401 | **0.525** | **+31%** |
| GAT | composizione Recall@1 | 0.643 | 0.722 | +12% |
| GAT | topologia mAP@10 | 0.225 | **0.273** | **+21%** |
| SAGE | composizione mAP@10 | 0.449 | 0.486 | +8% |

**Perché il mAP guadagna molto più dell'nDCG — ed è la parte più istruttiva.** Il mAP è sensibile all'**ordine dentro l'insieme dei rilevanti**, che è precisamente ciò che InfoNCE stava scompaginando: spingendo via l'una dall'altra piante della stessa classe di equivalenza, le sparpagliava lungo il ranking. L'nDCG invece usa la similarità graduata ed è più "morbido": anche se i rilevanti esatti scivolano un po' più in basso, i quasi-rilevanti che restano in cima tengono su il punteggio. Il mAP no, è binario e posizionale: se il rilevante scende dalla posizione 2 alla 8, se ne accorge subito.

Quindi il mAP è la metrica che **vede meglio questo tipo di danno**, e il fatto che sia proprio lei a guadagnare di più (+31%) è un'ulteriore conferma che il problema era esattamente quello ipotizzato e non qualcos'altro.

#### Il ranking finale

**GCN (0.8870) > SAGE (0.8750) > GAT (0.8713).** L'ordine non cambia, ma GAT dimezza il distacco da GCN (da −0.030 a −0.016).

Che l'encoder più semplice resti il migliore è coerente con tutta la diagnosi: GAT è il più *capace* — usa l'attenzione ed è l'unico che consuma il tipo di relazione sugli archi — e proprio per questo è il più bravo nel gioco dell'instance discrimination, che qui è in parte controproducente. Correggere la selezione recupera gran parte del danno, ma non tutto: il suo problema non era **solo** la selezione, era anche l'obiettivo. Per questo i prossimi interventi in programma (temperatura più alta, testa di proiezione stile SimCLR) puntano a cambiare la loss, non a scegliere meglio dentro di essa.

---

### - qual è il trade-off tra le metriche e perché si verifica

#### Trade-off principale: composizione/topologia **contro** geometria

Fermando il training al picco dell'nDCG medio si guadagna sugli assi discreti e si perde un po' sulla geometria: **GAT −0.008, GCN −0.002, SAGE −0.001**.

**Perché si verifica.** I due gruppi di assi rispondono in modo **opposto** alla stessa cosa — la quantità di allenamento contrastivo:

- **Composizione e topologia** hanno grandi classi di equivalenza (0,5% e 15,7% di singleton, cioè moltissime piante hanno "sorelle"). Ogni batch è pieno di falsi negativi. Più alleni, più il modello separa piante che dovrebbero stare vicine → **peggiorano**.
- **La geometria è continua**: non esistono classi di equivalenza, quindi non esistono falsi negativi strutturali. Il modello semplicemente impara a codificare meglio area e proporzioni → **continua a migliorare**.

Sono due obiettivi che tirano in direzioni opposte lungo lo stesso asse ("quanto alleno"), e questo rende un compromesso **inevitabile**: non esiste un'epoca che sia ottima per entrambi. Con `select_criterion: mean` si sceglie il punto che massimizza la media dei tre, e il bilancio è nettamente favorevole — si cede **0.008 di geometria per guadagnare 0.030 di topologia**. Se un domani la geometria diventasse l'asse prioritario, basta impostare `select_criterion: geometry` per far scegliere alla sonda un checkpoint più tardo: il meccanismo è già configurabile.

#### Trade-off secondario: Recall **contro** mAP (dentro lo stesso asse)

Recall@K conta *quanti* rilevanti stanno nei primi K; mAP@K conta *dove* stanno. Un modello può alzare il Recall abbassando il mAP (trova più rilevanti ma li mette in fondo) o viceversa (ne trova pochi ma perfettamente in cima). Guardarli insieme dice se il modello "pesca nel mucchio giusto" **e** "ordina bene dentro il mucchio".

Nel nostro caso migliorano **insieme** (GAT: Recall@1 +12%, mAP@10 +31%), che è il segnale più pulito possibile — non stiamo spostando qualità da un posto all'altro, ne stiamo aggiungendo.

#### Trade-off su K: cima **contro** profondità

Media dei tre assi al variare di K:

| | K=1 | K=10 | K=100 |
|---|---|---|---|
| GCN | **0.911** | 0.887 | 0.864 |
| SAGE | 0.904 | 0.875 | 0.848 |
| GAT | 0.897 | 0.871 | 0.845 |
| baseline hist | 0.836 | 0.846 | **0.858** |

Le GNN **calano** al crescere di K, la baseline **sale**: a K=100 il margine si è quasi annullato (0.864 contro 0.858).

**Perché.** Al crescere di K la composizione — l'asse su cui la baseline è oracolo — pesa sempre di più nella media, e lì le GNN scendono (0.907 a K=100 contro il ~1.0 della baseline). Le GNN sono molto forti **in cima al ranking**, che è ciò che conta davvero nel retrieval (l'utente guarda i primi risultati), ma il vantaggio si diluisce in profondità. Sulla topologia però il divario resta netto anche a K=100 (0.752 contro 0.673): è lì che il valore aggiunto è reale a ogni profondità.

È onesto dichiarare che il margine non è uniforme: il confronto a K piccolo è quello rilevante per l'applicazione, ma non è l'unico modo di guardare i dati.

#### Trade-off di metodo: oracolo **contro** modello

La baseline ha composizione 1.000 perché usa esattamente la feature con cui viene giudicata: non è un modello migliore, è un promemoria che **su quell'asse la metrica è satura per costruzione**. Lo stesso caveat, in forma più debole, vale per l'intero ramo graph (composizione e topologia derivano da `rType`/`rEdge`, che sono l'input del grafo).

**Ma questo trade-off suggerisce anche la mossa successiva.** Se la baseline è perfetta sulla composizione a costo zero, e le GNN sono nettamente superiori su topologia e geometria, la **late fusion** dei due — combinando i punteggi dei due indici — prenderebbe il meglio da entrambi: composizione dall'oracolo, struttura e forma dalla GNN. Gli embedding allineati (`embeddings.npy` + `names.json`) di tutti e quattro i sistemi sono già stati salvati dall'ultima run, quindi è il prossimo passo con il miglior rapporto tra valore e sforzo.

---
---

# FUSIONE VISION + GRAPH

> *In futuro vogliamo fondere la parte di vision con quella di graph.*

**Premessa metodologica:** i numeri qui sotto sono direttamente confrontabili. Il ramo vision (job `vx_04`, 5 encoder × 3 pooling × 2 trasformazioni) e il ramo graph (job `gp_04 74184`) hanno usato **le stesse identiche 2000 query** dallo split test (`seed 42`), la stessa gallery intera e le stesse metriche dal core condiviso. La prova: entrambi riportano esattamente **10 query singleton sulla composizione e 313 sulla topologia**, cioè stanno valutando le stesse piante. Il confronto è quindi *appaiato*, non un accostamento approssimativo.

---

## - in che cosa è più forte la parte di vision?

### Il quadro numerico (nDCG@10, stesse 2000 query)

| sistema | composizione | topologia | geometria | media |
|---|---|---|---|---|
| **Vision** — miglior config per asse | 0.830 | 0.662 | **0.948** | — |
| &nbsp;&nbsp;*(ijepa/natural/whiten)* | 0.826 | 0.654 | 0.943 | 0.8077 |
| &nbsp;&nbsp;*(dinov3/natural/whiten)* | 0.824 | **0.662** | 0.934 | 0.8067 |
| &nbsp;&nbsp;*(ijepa/gem/whiten)* | 0.824 | 0.648 | **0.948** | 0.8067 |
| **Graph** — GCN | **0.934** | **0.789** | 0.938 | **0.8870** |
| **Graph** — SAGE | 0.916 | 0.766 | 0.943 | 0.8750 |
| baseline hist | 1.000\* | 0.643 | 0.894 | 0.8457 |

\* oracolo per costruzione.

### 1. La geometria — l'unico asse dove il confronto è davvero alla pari, e il vision vince

**0.948 contro 0.943.** Sembra un dettaglio, ma è il risultato più significativo della tabella, per un motivo preciso: **è l'unico asse su cui il ramo graph non ha vantaggi ingiusti**. Su composizione e topologia il grafo riceve in input (`rType`, `rEdge`) esattamente ciò che la ground truth misura in output — quei numeri sono un *upper bound*, non una prestazione. Sulla geometria no: la ground truth è area del footprint, aspect ratio e distribuzione di area, che nessuno dei due riceve già pronta.

E su quel terreno neutro **il vision è davanti**, il che ha perfettamente senso: forma e proporzioni sono *letteralmente ciò che si vede*. Un encoder visivo legge il contorno dell'appartamento e la sua compattezza direttamente dai pixel, senza doverli ricostruire.

### 2. Vede cose che il grafo non può nemmeno rappresentare

Questo è il punto più importante e il meno visibile nelle metriche attuali. Il ramo graph descrive ogni stanza con un **bounding box** (`cx, cy, w, h, area, aspect`). Quindi:

- un soggiorno a **L** e uno rettangolare con lo stesso bounding box sono **identici** per il grafo, e ovviamente diversissimi per il vision;
- **spessore dei muri**, posizione e ampiezza di **porte e finestre**, forma esatta del **contorno esterno** (rientranze, balconi incassati): il grafo non li ha proprio, il vision li vede;
- la **texture del disegno** e le convenzioni grafiche.

Nessuno dei tre assi di valutazione attuali misura queste cose, quindi questo vantaggio è **invisibile ai numeri di oggi** — ma è informazione reale, e in una fusione è esattamente il tipo di contributo che non si sovrappone all'altro ramo.

### 3. Funziona partendo da un'immagine e basta

È il vantaggio decisivo sul piano pratico. Il ramo vision ha bisogno solo di un **PNG**. Il ramo graph ha bisogno di un **grafo strutturato**, che in RPLAN arriva gratis dai `.mat` ma **nel mondo reale non esiste**: un progettista che carica una scansione, un render o uno schizzo non porta con sé `rType` e `rEdge`.

Quindi oggi, fuori dal dataset, il ramo vision è l'unico dei due che sia **direttamente utilizzabile**.

### 4. I suoi numeri sono onesti

Corollario del punto 3: il vision deve **inferire** composizione e topologia dai pixel. Il suo 0.826 di composizione e 0.662 di topologia sono prestazioni vere, non riflessi dell'input. Il confronto 0.826 contro 0.934 non è "il grafo è meglio del 13%": è "il grafo ha la risposta in tasca, il vision no".

### 5. Il partial retrieval è implementato, validato e funziona molto bene

È il caso d'uso realistico (il progettista parte da una pianta incompleta) ed è dove il ramo vision ha il suo risultato più forte. Con la *projection head* addestrata sul masking, l'MRR di self-recovery (ritrovare la pianta completa da cui la query degradata proviene):

| frazione di stanze rimosse | frozen | con head | guadagno |
|---|---|---|---|
| f = 0.5 (dinov3) | 0.535 | **0.835** | +56% |
| f = 0.75 (dinov3) | 0.223 | **0.662** | **~3×** |
| f = 0.75 (dinov2) | 0.258 | 0.490 | +90% |

Il guadagno **cresce con il livello di masking**: più la pianta è incompleta, più la head serve. Il ramo graph una modalità partial non ce l'ha ancora nemmeno implementata.

---

## - in che cosa è più forte la parte di graph?

### 1. La topologia — con distacco enorme, ma da leggere con attenzione

**0.789 contro 0.662** di nDCG, e soprattutto **Recall@1 0.507 contro 0.027** della baseline. Il *message passing* — il meccanismo con cui ogni nodo aggiorna la propria rappresentazione scambiando informazione con i vicini nel grafo — lavora **direttamente sulla struttura delle adiacenze**, che è precisamente ciò che l'asse topologia misura.

Un dato che aiuta a capire quanto sia difficile questo asse per il vision: il migliore encoder visivo fa **0.662**, la baseline che usa solo l'istogramma dei tipi fa **0.643**. Cioè il vision, sulla topologia, supera di pochissimo un sistema che le adiacenze **non le guarda affatto**. Ha senso: la differenza tra "muro pieno" e "passaggio" sono pochi pixel, e due piante visivamente somiglianti possono avere connettività diverse.

⚠️ Va però ripetuto che parte di quel 0.789 è **circolarità**: il grafo riceve `rEdge` in ingresso.

### 2. La composizione (0.934 contro 0.826), con lo stesso caveat

Il conteggio delle stanze per tipo è nelle feature dei nodi, e il pooling `add` lo preserva per costruzione. Il vision deve invece contare le stanze guardando l'immagine, che è un compito genuinamente difficile.

### 3. Ragionamento relazionale esplicito e manipolabile

Nel grafo l'adiacenza è un **oggetto di prima classe**, non una proprietà da inferire. Questo apre possibilità che nel vision non esistono:

- imporre **vincoli espliciti** ("deve avere 2 bagni", "cucina adiacente al soggiorno") direttamente sulla rappresentazione;
- modificare la query in modo strutturato (aggiungi un nodo, togli un arco) e ri-cercare;
- spiegare *perché* due piante sono simili in termini architettonici.

È anche il motivo per cui questo ramo è il ponte naturale verso la **Fase 2** del progetto, la generazione *constraint-aware*: quella fase ragiona su vincoli strutturali, ed è il grafo a parlare quella lingua.

### 4. Costa pochissimo

Embedding a 128 dimensioni contro i 768 del vision, nessun peso pre-addestrato da scaricare, training da zero in **pochi minuti** su una GPU modesta, nessuna rete richiesta sul nodo di calcolo. Il vision ha bisogno di backbone pre-addestrati da centinaia di milioni di parametri.

### 5. È invariante allo stile del disegno

Due rappresentazioni grafiche diverse della stessa pianta (colori, spessori, convenzioni) danno **lo stesso grafo** ma immagini diverse. Se un giorno si indicizzano piante da fonti eterogenee (Maticad, ResPlan, scansioni), questa invarianza diventa preziosa: il vision andrebbe ri-tarato, il grafo no.

---

## - come facciamo a fondere in modo complementare le due parti?

### Passo 0 — Prima di costruire: misurare quanto sono davvero complementari

Questa è la cosa più importante e costa mezza giornata, perché **gli embedding sono già tutti salvati** (`embeddings.npy` + `names.json` allineati per hist/gcn/gat/sage, e le cartelle `embeddings/vision/<modello>/<variante>/`).

Va fatto un **inner join sul nome della pianta** (attenzione: il graph ha 67.405 piante, il vision 67.453 — 48 PNG non hanno record `.mat` e sono stati saltati nella costruzione dei grafi), e poi si calcolano tre diagnostiche sulle stesse 2000 query:

1. **Sovrapposizione dei top-10** tra la miglior config vision e il miglior encoder graph. Se è bassa (diciamo sotto il 30%) mentre entrambi hanno nDCG decente, vuol dire che i due rami **guardano cose diverse** → la fusione ha molto da guadagnare. Se è alta, la fusione aggiungerà poco.
2. **Correlazione dell'nDCG per-query.** Se le due serie sono poco correlate, i due rami **sbagliano su query diverse**: è esattamente la condizione in cui fondere paga.
3. **Upper bound "oracolo"**: per ogni query prendere il migliore dei due nDCG. È il tetto massimo raggiungibile scegliendo, query per query, il ramo giusto. Se questo tetto è molto sopra entrambi, c'è spazio; se è appena sopra, la fusione non è la priorità.

> **Un'aspettativa realistica, da mettere in conto subito.** Se si prende il meglio per asse tra tutti i sistemi disponibili — composizione 1.000 (hist), topologia 0.789 (GCN), geometria 0.948 (vision) — si ottiene una media di **0.912** contro lo **0.887** del solo GCN. Il margine complessivo è quindi **+0.025, e viene quasi tutto dalla composizione della baseline**: il contributo specifico del vision sulla geometria vale circa **+0.005÷0.010**.
>
> Detto chiaramente: **sulle metriche attuali, aggiungere il vision al graph sposta poco.** Non perché il vision sia debole, ma perché (a) il graph vince su due assi grazie alla circolarità e (b) i tre assi attuali **non misurano** ciò che solo il vision vede (stanze non rettangolari, muri, aperture). Il valore della fusione va cercato altrove — robustezza, applicabilità reale, partial — e questo va detto esplicitamente nel report, invece di promettere un salto di nDCG che non arriverà.

### Metodo 1 — Concatenazione pesata = un solo indice FAISS *(la scelta consigliata)*

È la forma più pulita di *late fusion score-level*, ed è più semplice di come di solito la si descrive. Con gli embedding di entrambi i rami già L2-normalizzati, si costruisce per **ogni pianta** (e per la query) il vettore concatenato:

```
e = [ √α · v ; √(1−α) · g ]
```

dove `v` è l'embedding vision, `g` quello graph, e α ∈ [0,1] è il peso. Il prodotto scalare tra due vettori così costruiti è esattamente:

```
α · (v_q · v_g)  +  (1−α) · (g_q · g_g)
```

cioè **la somma pesata delle due similarità**. Il vantaggio pratico è grosso: si ottiene la fusione con **un solo indice e una sola ricerca**, senza dover interrogare due indici, unire i candidati e ri-ordinare. Ed è compatibile con tutta l'infrastruttura FAISS esistente.

L'unica accortezza è la **scala**: i due rami producono distribuzioni di similarità con dispersione diversa (il vision dopo whitening, il graph addestrato da zero). Conviene quindi standardizzare ciascun ramo — per esempio con uno z-score dei punteggi stimato su un campione di coppie — **prima** di applicare α, altrimenti α finisce per compensare una differenza di scala invece di esprimere una preferenza.

### Metodo 2 — Fusione per rango (RRF), come termine di paragone

*Reciprocal Rank Fusion*: si ignorano i punteggi e si usano solo le posizioni.

```
score(pianta) = Σ_rami  1 / (k₀ + rango_nel_ramo)      con k₀ ≈ 60
```

È robusta per definizione al problema della scala (non guarda i punteggi) e ha **un solo iperparametro poco sensibile**. Nella letteratura di information retrieval è una baseline notoriamente difficile da battere. Vale la pena implementarla per prima come riferimento: se la fusione pesata non batte RRF, il problema è nella normalizzazione dei punteggi, non nell'idea.

### Metodo 3 — Fusione a tre vie, includendo la baseline hist

Da non dimenticare, perché è quasi gratis: la baseline `hist` è **perfetta sulla composizione** e costa 13 dimensioni e zero training. Una fusione a tre — `hist` + `graph` + `vision` — assegna a ogni asse il sistema che lo domina:

| asse | chi lo copre | valore |
|---|---|---|
| composizione | hist (oracolo) | 1.000 |
| topologia | GNN | 0.789 |
| geometria | vision | 0.948 |

È il modo più diretto di trasformare il trade-off tra i sistemi in un vantaggio.

### Metodo 4 — α dipendente dalla query: il trade-off diventa una funzionalità

Invece di cercare **un** α ottimo globale, si può lasciare che dipenda da cosa l'utente sta cercando. In uno strumento per progettisti è naturale:

- *"trovami piante con questa struttura distributiva"* → α verso il **graph**;
- *"trovami piante con questa forma e queste proporzioni"* → α verso il **vision**;
- *"trovami piante simili"* → α bilanciato, tarato sul valid.

Così la complementarità smette di essere un compromesso da subire e diventa un controllo esposto all'utente. È anche molto più difendibile in sede di report: si mostra che i due rami sono forti su cose diverse *e* che il sistema sa sfruttarlo.

### Metodo 5 — Fusione appresa (dopo, non prima)

Una piccola MLP sopra `[v ; g]`, addestrata con lo stesso protocollo InfoNCE + `RetrievalProbe` già costruito per il ramo graph. Più potente, ma da affrontare **solo dopo** aver misurato i metodi non addestrati, per tre motivi: rischio di overfitting, la circolarità verrebbe incorporata nei pesi, e soprattutto perché — lezione già imparata in questo progetto — un modello addestrato va selezionato sulla **metrica di retrieval**, mai sulla loss.

### Dove la fusione conta davvero: il partial retrieval

È qui che vale la pena investire, perché i due rami hanno **modalità di fallimento diverse** proprio dove il compito è più difficile:

- il **vision** degrada gradualmente con il masking, e la head recupera moltissimo (MRR 0.223 → 0.662 a f=0.75) — ma resta legato a *quanti pixel* restano;
- il **graph** è nativamente robusto alla rimozione di nodi (è addestrato con `node_drop` come augmentation): togliere una stanza è un'operazione naturale sul grafo, non una degradazione dell'input.

Una query incompleta è il caso d'uso reale del progetto, ed è lo scenario in cui è più plausibile che i due rami si coprano a vicenda. Prerequisito: implementare la **modalità partial in `graph_evaluate`**, gemella di quella vision — è già nella lista dei prossimi passi.

### I due regimi della fusione — da dichiarare esplicitamente

Questa distinzione va messa nel report, perché cambia il significato dei numeri:

**Regime A — su RPLAN, con i `.mat` disponibili.** È lo scenario attuale. La fusione funziona e si misura, ma su composizione e topologia il contributo del graph è **parzialmente auto-avverante** (riceve in input ciò che la metrica valuta). I numeri sono validi come *upper bound*, non come stima di prestazione reale.

**Regime B — nel mondo reale, con solo un'immagine.** Non esiste alcun `.mat`. Per avere il ramo graph bisogna **estrarre il grafo dall'immagine** (l'approccio di Graph2Plan). A quel punto succedono due cose importanti: la circolarità **sparisce** (il grafo è stimato, non fornito), e la fusione diventa **davvero complementare** — un ramo legge i pixel, l'altro legge la struttura stimata dagli stessi pixel, e i loro errori sono di natura diversa. È anche l'unico regime in cui il sistema è utilizzabile da un progettista.

Il Regime B è fuori dallo scope della Fase 1, ma è **la direzione in cui la fusione ha senso pieno**, ed è utile dirlo apertamente: mostra che la scelta architetturale è consapevole e non un artefatto del dataset.

### Un'ultima raccomandazione sulla valutazione

Due regole, entrambe già imparate a caro prezzo in questo progetto:

1. **α si tara sul valid, mai sul test.** Altrimenti i numeri finali sono ottimisti e non lo si scopre.
2. **α si sceglie sulla metrica di retrieval**, non su una loss surrogata — è esattamente l'errore che la `RetrievalProbe` è stata costruita per correggere.

E una terza, specifica per la fusione: se si vuole che i numeri riflettano il vantaggio reale del vision, **serve un asse di valutazione che misuri ciò che solo il vision vede** (forma non rettangolare delle stanze, posizione delle aperture). Con i tre assi attuali, una parte del contributo del vision rimane per costruzione invisibile.
