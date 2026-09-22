# Recap dei risultati — ramo vision e ramo graph

*Aggiornato al 18 set 2026 (test finale letto). Fonte di ogni numero: `.claude/shared/status.md`.*

Le configurazioni definitive sono scelte: **vision `pespatial/gem/whiten`** e **graph `gat/asymrob`**.
La **late fusion** dei due rami **aiuta**: con α=0.6 ritrova le piante danneggiate molto meglio di entrambi i
rami, sul valid e **confermato sul test** (letto una volta sola il 18 set). Sette figure su otto sono
pronte: resta da scrivere il testo del report.

Da metà settembre «migliore» vuol dire **più robusto quando la planimetria è incompleta**. Il punteggio sulla pianta intera si riporta accanto, come prezzo
della scelta.

## Come leggere i numeri

- **Pianta intera**: nDCG@10 (qualità dei primi 10 risultati) su tre assi. Sono composizione (quali
  stanze ci sono), topologia (come sono collegate) e geometria (forma e aree). Un ranking casuale prende
  già 0.739 / 0.466 / 0.869, quindi conta solo il margine sopra questi valori.
- **Robustezza**: si danneggia la query e si misura se il sistema ritrova la pianta originale. La sigla è
  AUC: la media su tre livelli di danno (25, 50 e 75%), da 0 a 1.
  - Nel vision è anche la media di tre tipi di danno: stanze tolte, ritaglio (crop) e riquadri
    cancellati (patch).
  - Nel graph il danno è uno solo: stanze tolte.

## Ramo vision

### Pianta intera (split di validazione, agosto)

- I migliori per asse sono `tipsv2/gem448/whiten` (composizione 0.843, topologia 0.690) e
  `ijepa/gem/whiten` (geometria 0.947).
- Il whitening (decorrelare e normalizzare le feature) migliora la topologia in 14 combinazioni su 14,
  di +0.03 in media.
- Sul test (una volta sola) `siglip2/mean/whiten` fa 0.826 / 0.654 / 0.936. Le due alternative (dinov2,
  dinov3) sono equivalenti entro il rumore: a pianta intera non c'è un vincitore.
- Gli encoder più recenti o più grandi non vincono: pesa di più la distanza fra planimetrie colorate e
  foto naturali.

### Robustezza, com'è andata

1. La head è una piccola rete allenata sopra l'encoder congelato per ritrovare la pianta danneggiata.
   Con la head, `dinov3/natural` sembrava dominare: AUC 0.82, e 0.917 allenandola a convergenza, contro
   0.536 senza head.
2. **Scoperta dell'11 settembre**: la stanza «tolta» perdeva il colore ma teneva i muri interni, quindi
   forma e posizione restavano visibili. Togliendo anche i muri, la head scende da 0.917 a 0.394; col 75%
   di danno il ritrovamento crolla da 0.840 a 0.036. La robustezza era quasi tutta un artefatto del
   disegno.
3. Su crop e patch la head **perde** contro lo stesso encoder senza head: −0.183 e −0.164.

### Griglia completa col metro nuovo (53 configurazioni)

| # | config | media | stanze tolte | crop | patch |
|---|---|---|---|---|---|
| 1 | pespatial/gem/whiten | **0.525** | 0.392 | 0.661 | 0.521 |
| 2 | radio/natural/whiten | 0.516 | 0.382 | 0.603 | 0.563 |
| 3 | pespatial/natural/whiten | 0.515 | 0.371 | 0.686 | 0.488 |

- Il distacco fra il 1° e il 2° è piccolo (+0.008) ma significativo.
- Il whitening batte l'assenza di whitening in 26 coppie su 26. La vecchia head dinov3 scende al 17°
  posto.
- **Costo a pianta intera** della vincente: 0.813 / 0.631 / 0.938. Rispetto alla head di prima sono circa
  −0.016 su composizione e topologia; la geometria resta quasi pari.

### Head rifatta coi muri tolti (16 settembre)

- Sopra `pespatial` guadagna +0.147 sulle stanze tolte, il danno che ha visto in training.
- Perde −0.142 sul crop e −0.106 sulle patch, e la media scende a 0.491, sotto lo 0.525 del vincitore.
  **Non adottata**: la head impara solo il danno che vede.

**Config vision definitiva: `pespatial/gem/whiten`** (nessun training).

## Ramo graph

### Pianta intera

- **La loss di training non predice il retrieval, anzi va al contrario.** GAT dà il miglior retrieval
  all'epoca 3 e poi peggiora per 20 epoche mentre la loss migliora. Il motivo: la loss allontana le
  piante simili finite nello stesso batch. Da allora il checkpoint si sceglie con un piccolo retrieval
  sul valid.
- **Molto viene dall'architettura.** Sommare le feature delle stanze, senza nessuna rete, dà già 0.805 di
  media, contro 0.862 della GCN allenata (valutazione ridotta). Il training compra soprattutto la
  topologia (+0.100).
- **A pianta intera la migliore è `gcn/tau02`**: 0.974 / 0.830 / 0.941 sul valid. Era la scelta col
  criterio di agosto; col criterio della robustezza vince gat (sotto).
  - Contro la baseline senza training (istogramma dei tipi di stanza) guadagna +0.187 di topologia.
  - Il mAP@10 di topologia è 0.351 contro 0.015, cioè 24 volte tanto.

### Robustezza, com'è andata

1. **Allenato su coppie di piante intere, il graph crolla.** L'AUC resta sotto 0.025, e con appena il
   25% di stanze tolte la pianta giusta esce dai primi 100 nel 91% dei casi. Il perché: un grafo con meno
   stanze sembra il grafo intero di tante altre piante.
2. **Coppie «pianta intera ↔ versione con stanze tolte».** L'AUC sale per tutti: gcn 0.006→0.097, sage
   0.009→0.111, gat 0.007→0.249.
3. **Checkpoint scelto sulla robustezza** invece che sulla pianta intera: gat passa da 0.249 a **0.457**.
   È un limite inferiore, perché a 300 epoche stava ancora salendo. La regola vecchia era erratica: su
   training identici sceglieva le epoche 6, 31 e 33, con AUC 0.107, 0.249 e 0.267.
4. **Costo a pianta intera** contro la gat precedente: composizione −0.017, topologia pari, geometria
   +0.007. La forte perdita di topologia vista prima dipendeva dall'epoca scelta. Contro `gcn/tau02`
   il prezzo è più alto: la gat precedente faceva 0.878 / 0.676 / 0.943, quindi la nuova fa
   0.860 / 0.678 / 0.950 (misurato il 17 set nel job della fusione).
5. **Feature «vicini persi»** (conta i collegamenti tagliati): a parità di regola di scelta finisce
   **pari** (−0.004, non significativo). Il −0.134 misurato prima era colpa dell'epoca scelta.
6. **Regge anche un danno mai visto** (tenere solo soggiorno, cucina e bagno): 0.368 contro 0.002 del
   modello allenato senza stanze tolte.
7. **Rumore**: due training identici differiscono di circa 0.04 di AUC, quindi differenze più piccole
   non si leggono. L'ordine gat ≫ sage/gcn viene dalla regola vecchia e non è stato riverificato.

**Config graph definitiva: `gat/asymrob`**.

## Fra i due rami

- **Nessun ramo è «pulito».** A pianta intera il graph domina composizione e topologia, ma per
  costruzione: le etichette derivano dal suo stesso input. Anche la geometria si ricostruisce al 100% dai
  box delle stanze. Il vision, dal canto suo, legge il tipo di stanza dal colore, con 13 tipi ridotti a 6
  colori.
- **Geometria**: il vantaggio del vision (+0.002) dipende dai pesi della formula. Sulla forma
  dell'appartamento vince il vision (+0.019 / +0.027); sulla distribuzione delle aree per tipo vince il
  graph.
- **I due rami trovano piante quasi diverse**: nel 68% delle query non hanno nessuna pianta in comune nei
  primi 10.
- **Primo esperimento di fusione (agosto)**: per rango, esplorativo, sul test, solo geometria, con le
  config di allora: 0.944, contro 0.936 del vision e 0.941 del graph.
- **Robustezza sulle stanze tolte, config definitive**: vision 0.392, graph 0.456.

## Late fusion (17 settembre, valid)

### Come funziona

- Ogni ramo trasforma la pianta (intera o danneggiata) in un vettore; i due vettori si normalizzano e si
  **concatenano con un peso**: `[√α · vision ; √(1−α) · graph]`. Cercare con il vettore concatenato equivale a
  sommare le due somiglianze, pesate α e 1−α.
- Per le piante danneggiate si tolgono **le stesse stanze** in entrambi i rami (verificato query per query). Il
  vision cancella anche i muri. Crop e patch restano fuori: il graph non può riceverli.
- **α si sceglie sul valid** fra 0, 0.1, …, 1 (α=0 = solo graph, α=1 = solo vision), tenendo quello più robusto;
  a parità si prende quello centrale. Regole scritte prima di vedere i numeri.

### Controlli

- Con α=1 e α=0 la fusione ridà **esattamente** il vision e il graph (0 differenze su 2000 query).
- Unico controllo fallito: senza danno la fusione ritrova la pianta **troppo bene** (fino a 0.982, oltre il
  limite di 0.980 preso dai rami singoli). Non è un bug: RPLAN contiene piante quasi identiche, e ciò che un ramo
  confonde l'altro spesso lo distingue. Considerato superato (decisione del gruppo).

### Risultati (robustezza sulle stanze tolte)

| | graph | vision | **fusione (α=0.6)** |
|---|---|---|---|
| media sui tre livelli | 0.456 | 0.392 | **0.630** |
| 25% di stanze tolte | 0.855 | 0.818 | **0.955** |
| 50% | 0.426 | 0.297 | **0.720** |
| 75% | 0.088 | 0.061 | **0.215** |

- **Contro il graph**, il ramo migliore: +0.174 (intervallo di confidenza [+0.164, +0.184]) → per la regola
  fissata prima, la fusione **aiuta**. Il guadagno è circa 4 volte il rumore fra due training identici del graph.
- **α=0.6 è l'unico migliore**: la robustezza sale fino a 0.6 e poi scende, senza salti.
- **Supera anche l'oracolo** (scegliere per ogni query il ramo giusto: 0.556), su 1255 query su 2000. Nelle query
  in cui la fusione mette la pianta giusta al 1° posto e nessun ramo lo fa, quasi sempre **un** ramo la teneva
  vicina alla cima (nei primi 10) mentre l'altro spesso no: il secondo ramo abbassa le piante che la precedevano.
- **α=0.6 non è sbilanciato «di nome»**: sotto danno i punteggi dei due rami sono sparsi in misura simile.

**Pianta intera** (solo descrittivo, per la circolarità):

| | composizione | topologia | geometria |
|---|---|---|---|
| graph | 0.860 | 0.678 | 0.950 |
| vision | 0.813 | 0.631 | 0.938 |
| fusione (α=0.6) | 0.850 | 0.684 | 0.954 |

**La previsione scritta prima**: stanze tolte, fusione migliore di entrambi ✅ · pianta intera: composizione
sotto il graph ✅, topologia sotto il graph ❌ (è leggermente sopra), geometria pari o sopra ✅ (sopra entrambi).

### Cosa tenere a mente

- **Il numero sul valid è un po' ottimista**: α è stato scelto sulle stesse query su cui si misura. La conferma
  arriva dal test con α=0.6 fisso.
- **Il danno non è identico fra i rami**: al vision resta la sagoma del buco, al graph no.
- **Il verdetto vale per questo checkpoint del graph.**

### Controllo: complementarità o effetto d'insieme? (17 settembre, solo valid)

**La domanda.** Due modelli che sbagliano in modo diverso migliorano quando li si fonde anche se sanno le stesse
cose (**effetto d'insieme**). Il guadagno di vision + graph viene da lì, o dal fatto che i pixel portano
informazione che il grafo non ha (**complementarità**)?

**Il controllo.** Si fondono con lo stesso metodo **due training identici del graph** (`asymrob` + la replica, che
differiscono solo per il caso) e si confronta il loro guadagno con quello di vision + graph. Regola e previsione
scritte prima.

| | componente migliore | fusione | guadagno |
|---|---|---|---|
| vision + graph (α=0.6) | 0.456 (graph) | **0.630** | **+0.174** |
| graph + graph (β=0.4) | 0.496 (replica) | 0.530 | +0.034 |

- **Differenza fra i guadagni: +0.140** (intervallo [+0.129, +0.152]) → per la regola, **complementarità**.
- Graph + graph guadagna circa **un quinto** di vision + graph; vision + graph vince anche nel confronto diretto
  (0.630 contro 0.530) pur partendo da componenti più deboli.
- Previsione: guadagno di graph + graph sopra zero ✅ · meno della metà di vision + graph ✅ · β fra 0.4 e 0.6 ✅ ·
  pianta intera entro ±0.01 dal training migliore ❌ (topologia +0.012).

**Cosa dimostra e cosa no.** Due training identici sono l'insieme **meno** diverso possibile (sbagliano in modo
molto simile). Il controllo esclude che il guadagno sia solo «rifare lo stesso modello due volte», ma non basta per
i **modelli diversi** che vedono la stessa informazione: serve il secondo controllo, qui sotto.

### Secondo controllo: due encoder vision diversi (17 settembre, solo valid)

**L'idea.** `pespatial` e `radio` guardano **la stessa immagine** ma sono modelli molto diversi. Fondendoli si misura
quanto vale la sola diversità dei modelli, senza informazione nuova.

| gradino | coppia | informazione | modelli | guadagno |
|---|---|---|---|---|
| 1 | due training identici del graph | uguale | uguali | +0.034 |
| 2 | `pespatial` + `radio` (γ=0.5) | **uguale** | diversi | **+0.098** |
| 3 | vision + graph (α=0.6) | diversa | diversi | **+0.174** |

- **Differenza fra il gradino 3 e il 2: +0.076**, intervallo [+0.064, +0.088], fuori dal margine di equivalenza
  ±0.04 → per la regola fissata prima, **complementarità**.
- **Ma il gradino 2 vale già il 56% del gradino 3**: più di quanto avessimo previsto (prevedevamo meno della metà).
- **Quanto si somigliano gli errori** delle due coppie (correlazione query per query): `pespatial` e `radio` 0.62,
  i due training del graph 0.65, **vision e graph solo 0.09**. Vision e graph sbagliano su query quasi diverse.
- **Senza danno** la fusione di due encoder vision non recupera **nessuna** pianta in più (0.9707 per ogni γ, come
  `pespatial` da solo): i duplicati di RPLAN hanno immagini identiche, quindi nessun encoder li distingue, mentre il
  grafo sì. Indizio a favore dell'informazione diversa, fuori dalla misura di robustezza.
- Previsione: guadagno sopra zero ✅ · complementarità ✅ · meno della metà di vision + graph ❌ · γ fra 0.4 e 0.6 ✅ ·
  pianta intera fra 0 e +0.02 su ogni asse ✅.

**La conclusione onesta.** Il guadagno della fusione è **in parte** effetto d'insieme fra modelli diversi (circa il
56%) e **in parte** informazione complementare (circa il 44%). Non è né solo l'uno né solo l'altro. Resta un limite
strutturale: due encoder vision sono comunque meno diversi fra loro di quanto lo siano un encoder di immagini e una
rete su grafo, e informazione e modello non si separano del tutto.

## Il test finale (18 settembre, letto una volta sola)

Regole scritte prima nella pre-registrazione: sistemi congelati, **α fisso a 0.6** preso dal valid, nessuna scelta
fatta guardando il test.

| stanze tolte | baseline senza training | vision | graph | **fusione (α=0.6)** |
|---|---|---|---|---|
| media sui tre livelli | 0.0013 | 0.3965 | 0.4700 | **0.6424** |
| 25% di stanze tolte | 0.0026 | 0.8216 | 0.8620 | **0.9580** |
| 50% | 0.0011 | 0.3037 | 0.4471 | **0.7363** |
| 75% | 0.0003 | 0.0642 | 0.1008 | **0.2331** |

- **Guadagno sul ramo migliore: +0.1725**, intervallo [+0.1625, +0.1826]. Sul valid era +0.1740, quindi il
  vantaggio di aver scelto α sul valid era trascurabile.
- **Supera l'oracolo** (0.5693) di +0.0731, su 1218 query su 2000.
- **Vision sul metro a tre danni**: 0.5210 (0.5245 sul valid).
- **Sotto danno la baseline senza training non ritrova quasi nulla** (0.0013).

**Pianta intera** (solo descrittivo, per la circolarità): graph 0.859 / 0.675 / 0.949 · vision 0.814 / 0.633 /
0.937 · **fusione 0.851 / 0.686 / 0.954**. La fusione perde 0.007 di composizione sul graph e guadagna 0.011 di
topologia e 0.005 di geometria.

**Le previsioni scritte prima**: rami entro 0.01 dal valid ✅ per il vision e per il metro a tre danni, ❌ per il
graph (+0.014, fuori di poco e in meglio) · guadagno sopra +0.12 ✅ · ordine fusione, graph, vision ✅ · pianta
intera come previsto ✅.

**I controlli**: tutti passati. I due estremi della fusione riproducono esattamente i due rami, le query e le
stanze tolte coincidono, e il ritrovamento senza danno resta nell'intervallo atteso.

⚠️ I due controlli sulla complementarità (la scala a tre gradini) restano misurati **sul valid**: non sono stati
ripetuti sul test.

## Le figure (18 settembre): sette su otto

Il report sarà **a due colonne**, come nelle conferenze: le figure si disegnano già alla larghezza definitiva,
così il testo dentro la figura ha la stessa dimensione di quello del report. Stanno in `figures/`, in PDF (per il
report) e in PNG (per guardarle al volo); accanto a ognuna c'è un file di testo che dice da quali file vengono i
numeri, con quale comando è stata fatta e quali valori ha disegnato — serve a poterla rifare identica fra mesi.

1. **`figures/f_damage_test.pdf` — quanto regge ogni sistema mentre la pianta si svuota (test).**
   Sull'asse orizzontale la frazione di stanze tolte alla query, su quello verticale quanto in alto la pianta
   ritrova sé stessa. Quattro linee: baseline senza training, vision, graph e fusione. La linea della fusione
   sta sopra le altre a ogni livello di danno, e il divario cresce col danno: a metà stanze tolte 0.74 contro
   0.45 del graph e 0.30 del vision. Il punto a zero stanze tolte è disegnato ma segnato come «tetto dei dati»:
   lì arrivano tutti, perché RPLAN contiene piante duplicate, ed è il motivo per cui non entra nel punteggio.
2. **`figures/f_alpha_valid.pdf` — quanto guadagna ogni tipo di fusione (valid).**
   Sull'asse orizzontale il peso dato al primo dei due modelli, da «solo il secondo» a «solo il primo». Le tre
   curve sono le tre coppie provate: stesso modello allenato due volte, due modelli diversi che guardano la
   stessa cosa, e vision + graph. Tutte e tre sono una collina: mescolare batte sempre i due estremi. Quello che
   cambia è **quanto** sale la collina sopra il suo estremo migliore, ed è la freccia annotata su ognuna:
   +0.034, +0.098 e +0.174. È la scala dei guadagni di cui parla il report, in una figura sola.

3. **`figures/f_teaser_valid.pdf` — la figura di apertura: una query rotta e i primi risultati dei tre sistemi.**
   In alto una pianta intera e la stessa pianta con metà delle stanze cancellate, muri compresi: è esattamente
   quello che i sistemi hanno ricevuto. Sotto, tre righe — vision, graph, fusione — coi primi cinque risultati
   per quella query, e la pianta originale incorniciata in verde dove rientra. Il vision la perde (oltre il
   centesimo posto), il graph la mette terza, la fusione prima. È il risultato del report in una figura.
   La query non è scelta a occhio: è la prima in ordine alfabetico fra le 423 in cui la fusione ritrova
   l'originale al primo posto e nessuno dei due rami ci riesce; la regola è scritta accanto alla figura.
   Usiamo il valid, non il test: una figura qualitativa non ha bisogno del test, che resta un numero letto
   una volta sola.

4. **`figures/f_damage_kinds_valid.pdf` — lo stesso encoder non è «robusto» in generale.**
   A sinistra, il ramo vision sotto i tre danni: togliere stanze lo mette in crisi (0.392), un ritaglio
   rettangolare molto meno (0.661), le toppe sparse stanno in mezzo (0.521). La media dei tre è il 0.524
   con cui abbiamo scelto la configurazione. A destra la prova del bug di settembre: le stesse stanze
   tolte, una volta lasciando i loro muri sull'immagine e una volta cancellandoli. La distanza fra le due
   curve (+0.336 a metà stanze) è quanto valeva leggere il contorno di una stanza che doveva essere sparita.
5. **`figures/f_pipeline.pdf` — lo schema del sistema.** I due rami in parallelo, la fusione e, sotto,
   la barra che dice cosa hanno in comune: stessa gallery di 67.405 piante, stesse query, stessa misura.
   I numeri scritti nello schema (768 e 128 dimensioni, 896 dopo la fusione, α=0.6) non sono scritti a
   mano: la figura li rilegge dai file ogni volta che viene disegnata.
6. **`figures/f_classes_valid.pdf` — perché due dei tre assi vanno letti con prudenza.**
   Quante piante contano come «giuste» per una query? In composizione la metà delle query ne ha più di
   **5000** su 67.405: con così tanti bersagli il punteggio si satura e distingue poco. In topologia la
   mediana è **19**, e 311 query su 2000 non hanno nessun'altra pianta uguale: quelle vengono saltate dal
   conteggio, ed è il motivo per cui lo dichiariamo ogni volta.
7. **`figures/f_ablation_valid.pdf` — il primo posto non è speciale.**
   Un riquadro per encoder con dentro tutte le sue configurazioni provate (52 in tutto). Si vede che
   contano più l'encoder della combinazione, e che la configurazione scelta (la stella, 0.5245) è
   praticamente appaiata alla migliore di un altro encoder (0.5161): la classifica top-1, da sola,
   sarebbe stata una lettura fuorviante.

Le figure leggono i risultati già calcolati, non rifanno nessun esperimento; e più d'una si rifiuta di
disegnare se un numero non coincide con quello già registrato, così non possono mostrare cifre diverse dal
report. Nessuna ha richiesto un job: per il teaser le classifiche erano già salvate (con la posizione della
pianta originale), e il danno è stato ridisegnato dalle stanze che la valutazione aveva tolto davvero.

## Cosa manca

1. **Il testo del report: lo scrivete voi.** Da qui in poi i numeri, le figure e i controlli sono pronti e
   allineati; se serve un numero, una figura rifatta o una verifica, si chiede.
2. Una sola figura non fatta: quella che confronta la loss finale col risultato di ricerca. I dati che
   abbiamo sono incompleti (11 serie su 15), quindi va decisa: farla dichiarando il buco, o lasciarla fuori.
3. Facoltativo: la testa congiunta (una piccola rete allenata sopra i due vettori concatenati), che sarebbe una
   seconda tornata pre-registrata a parte.
