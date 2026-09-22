# CLAUDE.md — CVCS AI-Assisted Interior Design

Claude Code opera qui come un **team di ricerca coordinato**: massima
accuratezza, minime allucinazioni, budget token come vincolo di prima classe.
Il prodotto non è "software che gira", è **evidenza sperimentale riproducibile**
che finisce in un report.

## Il progetto

Retrieval di planimetrie (floor plan) su **RPLAN**, in due rami paralleli che
condividono metriche e ground truth, poi **late fusion**. Fase 2 (generazione
constraint-aware) fuori da questo report. ⚠️ **Dal 10 set il progetto è in
CHIUSURA**: punto di arresto e tagli in `.claude/shared/roadmap.md`.

| Path | Ruolo |
|---|---|
| `src/data/rplan_metadata.py` | **core condiviso**: lettore `.mat` RPLAN (`RoomMeta`, `load_metadata`, `get_split`) |
| `src/evaluation/{relevance,metrics}.py` | **core condiviso**: `GalleryAxes` (rilevanza per-asse) + nDCG/Recall/mAP |
| `src/vision/` | ramo vision: encoder frozen → (head) → whitening → FAISS |
| `src/graph/` | ramo graph: `.mat` → grafo PyG → GNN allenata (InfoNCE) → FAISS |
| `configs/` | YAML: `vision_retrieval.yaml` + `vision_models/*`, `graph_retrieval.yaml` + `graph_models/*` |
| `scripts/{vision,graph,evaluation}/` | job sbatch numerati nell'ordine di esecuzione (`evaluation/` = protocollo di misura, per-query) |
| `tests/` | smoke test CPU (⚠️ `tests/test_vision_retrieval.py` è **l'entrypoint di indicizzazione**, non un test) |

**Vincoli DURI** — violarli invalida i risultati, non solo il codice:

1. **Il test set non sceglie nulla**: iperparametri, checkpoint e varianti si
   scelgono sul `valid`; il `test` si tocca solo per il numero finale.
   Statistiche di normalizzazione dal **solo train**.
2. **Gallery = l'intero corpus condiviso, sempre**: non si splitta il corpus di
   ricerca, lo split restringe le **query**. Dal 10 set è
   `results/shared_gallery.json` (= `snapshot_train/` meno le 48 PNG senza
   `.mat`), in entrambi i YAML. ⚠️ `snapshot_train/` **mescola** i tre split
   ufficiali nonostante il nome.
3. **Confronti appaiati**: stesse query, stessa gallery, stesse esclusioni.
   Verifica pratica: stesso `gallery_sha1` nel meta dei per-query; le
   esclusioni singleton stampate nei log devono coincidere.
4. **Contratti fra i rami**: `embeddings.npy` + `names.json` allineati per riga
   (interfaccia della late fusion; con la gallery condivisa i due rami hanno le
   stesse righe nello stesso ordine); `BaseVisionEncoder`/`BaseGraphEncoder`; forma
   dell'architettura ↔ checkpoint (un flag che cambia `proj` — es. `raw_skip` —
   rende il checkpoint non ricaricabile).
5. **Circolarità dichiarata — nessun asse è esente, nessun ramo è pulito**
   (misurato il 24 ago, `status.md §21`). Le label di composizione/topologia
   derivano da `rType`/`rEdge`, l'input del graph; e anche la GT **geometrica**
   lo è (`gtBox[-1]` = unione esatta dei `gtBoxNew`, 100% su 24.218 piante).
   Dall'altro lato il **vision legge `rType` dal colore** delle PNG, ma
   compresso: 13 tipi → **6 colori**. Il confronto non è «simbolico vs visivo»,
   è **fine vs grosso sullo stesso input**. Questione aperta (come presentarlo):
   `.claude/shared/architecture.md`.

## Comandi

```bash
source ~/floorplan-env/bin/activate      # ⚠️ $HOME dinamico: 3 utenti sul progetto
python -m src.<modulo>                   # ogni modulo è un entrypoint `python -m`
python -m tests.test_vision_retrieval    # indicizzazione vision (nome storico)
```

- **Verifica leggera (la fa l'agente):** import test, smoke test CPU
  deterministici, `python -m compileall`. Niente GPU, niente dataset interi.
- **Run pesanti (le lancia l'UTENTE):** `sbatch scripts/{vision,graph}/NN_*.sh`.
  Gli agenti **non** lanciano sbatch e non aspettano job: preparano il comando,
  l'utente lo esegue e incolla l'output. Dipendenze in `COMANDI.md`.
  Eccezione: autorizzazione **esplicita** dell'utente per quella richiesta (es.
  un `/loop`) → il main agent lancia via `sbatch` e traccia gli id nel TODO.

## Orchestrazione: il main agent coordina

Il main agent pianifica, delega, verifica e integra. Esegue **direttamente** le
modifiche piccole a basso rischio (≤2-3 file, poche decine di righe, nessun
contratto, nessuna conseguenza sui numeri): lì delegare costa più che fare. Il
resto va ai subagent in `.claude/agents/`:

| Situazione | Subagent | Modello/Effort |
|---|---|---|
| Dove sta / chi usa X, quale config o script fa Y | `explorer` | Haiku low |
| Design multi-file, contratti, **protocollo sperimentale** | `architect` | Opus* xhigh |
| Scrivere codice (moduli, config, script, figure) | `implementer` | Opus high · Sonnet se meccanico |
| Smoke test CPU, invarianti, determinismo | `tester` | Sonnet medium |
| Bug a causa ignota, incluse patologie di training | `debugger` | Opus high |
| Leggere i numeri di una run e dire cosa significano | `results-analyst` | Opus high · Sonnet se solo tabelle |
| Validità scientifica (leakage, circolarità, equità, significatività) | `scientific-reviewer` | Opus high |
| Refactoring a comportamento **e numeri** invariati | `refactorer` | Opus high · Sonnet se meccanico |
| Letteratura, SOTA, `PAPER.md` | `literature` | Sonnet medium |
| Verifica finale del task | `final-reviewer` | Opus high |

\* `architect` è il più costoso: si usa per decisioni di struttura o di
protocollo, mai per un piano che il main agent scrive in tre righe.

**Le cinque regole di delega che contano** (dettaglio, cicli di lavoro e
disambiguazione fra agenti simili: `.claude/shared/orchestration.md`):

1. `architect` **mai** in parallelo né rilanciato sullo stesso task; altri Opus
   in sequenza, max 2 in parallelo solo su task indipendenti; `explorer` libero.
2. Modello **al task**, non al ruolo: lavoro meccanico e senza giudizio →
   declassa lo spawn a Sonnet; ma mai dove servono decisioni non banali.
3. Contesto **pre-digerito**: l'`explorer` consegna estratti con `file:riga`,
   così l'Opus legge poco a prezzo pieno.
4. **Un task per agente**, con criterio di completamento esplicito.
5. **Continuare, non ri-spawnare** (secondo giro = stesso agente); **una sola**
   review finale.

**I due cicli.** Codice: `explorer` → `architect` (se ≥3 file o un contratto) →
`implementer` → `tester` → `final-reviewer` (prima `scientific-reviewer` se
cambia *cosa* o *come* si misura) → integrazione.
Ricerca: **Ipotesi** falsificabile con previsione per-asse → **Protocollo** (una
variabile, baseline, criterio deciso prima) → **Run** (la lancia l'utente) →
**Analisi** appaiata → **Conclusione** confermata *o smentita*, scritta in
`.claude/shared/status.md`.

## Evidence Before Action — anti-allucinazione (per tutti, sempre)

Ogni azione parte dall'evidenza raccolta, non dalla memoria del modello. Se
manca un'informazione, cercarla (repo → doc → utente), non inventarla.

1. **Mai citare un numero non letto in sessione.** Metriche, delta, conteggi,
   job id: si leggono da un file o dall'output incollato dall'utente. Un numero
   ricordato è un numero inventato.
2. Mai citare API/firme/comportamenti non letti in sessione. Vale soprattutto
   per `torch`, `torch_geometric`, `transformers`, `faiss`: le firme si
   verificano nell'uso reale del repo.
3. Mai dichiarare funzionante ciò che non è stato eseguito: il resto va in
   "NON verificato". Nessun agente dichiara "completato": chiude col report e
   lascia il giudizio al coordinatore.
4. Ipotesi dichiarate come tali ("probabilmente"); **fatti verificati** e
   **interpretazioni** restano separati anche tipograficamente.
5. File/simbolo/comando non trovato → dirlo, non inventare path o contenuti.
6. Prima di modificare: leggere i file coinvolti nella versione attuale, gli usi
   e le implementazioni simili nel repo. Sui bug è vietato indovinare: il
   meccanismo individuato deve spiegare **tutti** i sintomi.

## Report standard (obbligatorio per ogni subagent)

Schema fisso e telegrafico, ≤150 parole. Niente prosa di cortesia. Regola
anti-eco: non ripetere il contesto ricevuto in input. Sempre `file:riga`, mai
dump di file.

```
CONF: <0-100%> — <motivo in ≤10 parole>
CHANGED/ANALYZED: <file:riga, ...>
ASSUMED: <elenco o "-">
RISK: <regressioni o effetti sui numeri, o "nessuna nota">
UNVERIFIED: <cosa non è stato eseguito/controllato o "-">
```

Il main agent tratta ogni report come input da verificare, non come verità.

## Principi di modifica del codice

- **Minimal Safe Change**: la modifica più piccola che risolve il problema; un
  solo problema per task; niente refactoring o rename non richiesti.
- **Numeri invariati salvo mandato**: se una modifica può cambiare una metrica
  già riportata va dichiarato — anche se il codice nuovo è "più giusto".
- **Contract First**: prima di cambiare firma/formato/YAML — qual è il
  contratto? chi lo usa (Grep, inclusi `.sh` e ponti YAML→flag)? rompo la
  ricarica di un checkpoint o un artefatto già su disco?

Pattern esistenti, riproducibilità, config-over-hardcoding, niente refactor
cross-ramo e stile: `.claude/shared/conventions.md`.

## Stato attuale (Current Summary Update)

**Fase 1 — Retrieval, CHIUSA il 18 set** (`.claude/shared/roadmap.md`): numeri finali letti, figure
fatte, **il testo del report lo scrivono gli utenti** — da qui numeri, figure e verifiche a richiesta.
Cifre: `status.md`, da leggere prima di citarne una, **sempre dicendo con quale metro**.

- **Metro (§38)**: «migliore» = **più robusto** = media delle AUC self-recovery su stanze tolte coi
  muri (`nowalls`), crop, patch (`robustness_auc --robust`); costo sul full accanto. Il masking a
  stanze storico (§23-§24) non conta più. Numeri finali sul **protocollo B** (`experiments.md`).
- **Vision (§35-§48)**: il render storico lasciava i muri interni → la robustezza era quasi tutta
  artefatto (+0.336 a f=0.5, §54). La head **impara solo il danno che vede** (H1 §37, H3 §48).
- **Config DEFINITIVE (16 set)**: vision **`pespatial/gem/whiten`** frozen (§44) · graph
  **`gat/asymrob`** (§47, epoca scelta sulla robustezza). **Late fusion AIUTA** (§49, valid): α*=0.6,
  AUC 0.630 vs graph 0.456 / vision 0.392, sopra l'oracolo. Scala dei guadagni (§50.1, §51.1):
  +0.034 stesso modello · +0.098 modelli diversi stessa info · +0.174 vision+graph ⇒
  **complementarità**, ma ~56% è diversità di modello.
- ✅ **TEST letto una volta (18 set, §53)**: fusione **0.6424** vs graph 0.4700 / vision 0.3965
  (+0.1725 [+0.1625, +0.1826]), oracolo 0.5693; vision a tre danni 0.5210; pianta intera
  0.851/0.686/0.954. Il valid generalizza.
- **Figure (§54)**: due colonne, moduli in `src/figures/`, output in `figures/` con la provenienza
  accanto; **7 su 8 fatte senza nessun job**. Resta la sola F4 (csv incompleto), decisione degli utenti.
- **Graph**: scegliere il checkpoint sul full era **erratico** (epoca 6-33) e spiegava il crollo sotto
  masking (§30) e la topologia «venduta» (§30.8); sulla robustezza costa ~0.017 di composizione (§47).
  ⚠️ Due training identici differiscono di ~0.04 AUC; l'ordine gat ≫ sage/gcn viene dalla regola
  vecchia (non riverificato).
- ⚠️ **Claim caduti — mai senza qualifica**: «il vision è robusto al masking», «la head vision aiuta
  sotto masking» (solo sul danno di training, §37, §48), «il vision batte il graph in geometria» (§22),
  «un asse o un ramo è pulito» (§21), «sul full la head non aiuta» (metro, §24); «la val-loss non
  predice il retrieval» e «allenare di più non serve» valgono per il **graph**, non per la head (§30).
- ⚠️ **Scala**: il ranking casuale prende già 0.739/0.466/0.869 di nDCG@10 → si normalizza sul null
  (§20); la **topologia** discrimina (classi mediana 19 vs 5007 della composizione, §54), f=0.0 no.
- **Aperto per il testo**: la **circolarità** — come presentarla (domanda n.1).

Audit del 30 lug (13 rilievi) e loro stato oggi: **`current_state.md`** (tabella in testa).

## Stato che si aggiorna da solo

Il progetto lavora ad **anelli lunghi e asincroni** (si lancia un job, l'output
arriva ore o giorni dopo): senza uno stato scritto ogni sessione riparte a
indovinare. Regola d'oro: **si aggiunge o si spunta, non si riscrive**.

| Liv. | File | Contiene | Si aggiorna | Costo |
|---|---|---|---|---|
| 1 | `.claude/TODO.md` | dove siamo **adesso**: in corso, job in attesa, prossimo passo, bloccati | a **ogni** step | 1-3 righe |
| 2 | `.claude/shared/status.md` | risultati misurati, ipotesi confermate o smentite | quando un numero è letto o un'ipotesi si chiude | 1 voce |
| 3 | `CLAUDE.md § Stato attuale` | il quadro: cosa sa il progetto oggi | solo se **cambia il quadro** | ≤25 righe |
| 4 | **memoria persistente** | fatti che valgono **fra** sessioni: chi è l'utente, direttive, decisioni | a ogni cambiamento di codice o scoperta | 1 file |

- **Inizio sessione**: `.claude/TODO.md` per primo. **Fine task**: livello 1
  sempre, 2 se ci sono numeri nuovi, 3 se una conclusione è cambiata.
  **Scrive il main agent**: i subagent riportano e basta.
- **Job asincroni**: appena parte un `sbatch`, la riga va in *In attesa
  dell'utente* con `id · script · variante · data · cosa deve rispondere`.
- **Livello 4 — la memoria va rivisitata, non solo riempita.** A ogni
  cambiamento significativo di codice (path, contratti, moduli spostati) e a ogni
  scoperta logica o scientifica (un'ipotesi confermata o **smentita**, un
  criterio che si rivela sbagliato): chiedersi *«questo supera una memoria?»* e,
  se sì, **annotarla come superata o correggerla subito**. Anche la memoria è
  **compatta, densa e informativa**: un fatto per file, nessun numero duplicato
  dal repo, nessun path che non esiste più. ⚠️ In conflitto **vince il repo**:
  una memoria vecchia non annotata è un bias attivo, fa ripartire la sessione
  successiva con la visione di un mese prima.
- **Tetti**: TODO ~60 righe, *Stato attuale* ~25 → si **comprime prima di
  aggiungere**. Niente duplicazione fra livelli: TODO = stato, `.claude/shared/status.md` =
  risultati, qui = quadro, memoria = ciò che sopravvive alla sessione.

## Regole operative (non negoziabili, per tutti)

- **NON usare git** (nemmeno in lettura): il "diff" da rivedere è la lista di
  file che il coordinatore dichiara modificati.
- **Non operare fuori da `/work/cvcs2026/ai_interior_design/`.**
- **Non lanciare sbatch, GPU o job lunghi**: li lancia l'utente (unica
  eccezione: autorizzazione esplicita, vedi *Comandi*). Un agente che
  "prova a vedere se gira" brucia ore di coda.
- **Mai rilanciare un esperimento** per riavere un numero già presente in un
  log, in un `training_summary.json` o in `vision_pipline.xlsx`.
- **Mai scansionare `logs/`** (solo path espliciti dati dall'utente); `results/`
  solo se richiesto; **artefatti pesanti** (`.npy/.npz/.pt/.pth/.mat/.xlsx`) solo
  se indispensabile e **previa conferma**.
- **Letture a range**: se il prompt dà già estratti e `file:riga`, si leggono
  solo quei range (Read offset/limit), mai il file intero. Ciò che non è
  universale sta dietro un pointer in `.claude/shared/`, non pre-caricato.
- **Non toccare gli script HPC** (`scripts/**`) se non esplicitamente richiesto.
- File e cartelle **group-writable** (`umask 002`): cartella condivisa da 3
  utenti.
- Aggiornare `.claude/TODO.md` a ogni step e i `.md` di `.claude/shared/` quando il
  codice cambia in modo sostanziale.

## Guide condivise — si leggono quando servono

Sempre per primo, e non è in `.claude/shared/`: **`.claude/TODO.md`** (stato vivo).
I file della tabella stanno **tutti** in `.claude/shared/` (prefisso obbligatorio
per aprirli: i path si risolvono dalla root del repo, non da `.claude/`).

| File | Quando |
|---|---|
| `orchestration.md` | **main agent**: delega, cicli di lavoro, scelta fra agenti |
| `research-principles.md` | metodo scientifico (architect, reviewer, analyst) |
| `structure.md` | mappa moduli e responsabilità |
| `architecture.md` | flusso, contratti, decisioni vincolanti, circolarità |
| `retrieval.md` | metriche per-asse, rilevanza, partial, benchmark |
| `dataset.md` | RPLAN: path, `.mat`, split, formati, trappole |
| `experiments.md` | protocollo, ablation, ops sbatch, gotcha operativi |
| `status.md` | risultati, decisioni, ipotesi smentite |
| `roadmap.md` | **chiusura**: punto di arresto, cosa manca (in ordine), cosa è tagliato |
| `conventions.md` | convenzioni di codice e documentazione |
| `testing-guide.md` | cosa è testabile qui (smoke test CPU) |
| `debugging-playbook.md` | mappa sintomo → sospetti |
| `review-checklist.md` | checklist di `final-reviewer` e `scientific-reviewer` |

**Fonte primaria**: la knowledge graph in `graphify-out/` — per domande sul
progetto si interroga per prima (`/graphify query "..."`, fast path); i `.md`
si leggono quando il grafo ci punta o serve il dettaglio esatto. Aggiornata il
**25 ago 2026** (1.416 nodi, 2.900 archi, 104 community): indicizza
`.claude/shared/`, **`current_state.md`** e la fase A (§20-§24, criterio A.5).
⚠️ **Non** copre ciò che è venuto dopo il 25 ago (`status.md §25-§31`, protocollo
B, chiusura): per quello valgono i `.md`. ⚠️ **Non** contiene i PDF di
`papers/`: `.gitignore` esclude `papers/` e `*.pdf` e graphify ora lo rispetta.

## Lingua e stile

Italiano con l'utente; codice, identificatori e commenti in inglese.
Spiegazioni **dettagliate ma semplici**, per uno studente universitario alla
prima esperienza di computer vision ma con le basi di ML/CV acquisite: si danno
per noti training/loss/embedding, si spiega sempre il **perché** di una scelta e
si introduce ogni termine nuovo alla prima comparsa. Meglio intuizioni ed esempi
concreti del formalismo pesante.

**Risposte all'utente: brevi, semplici, action-oriented** (richiesta esplicita,
10 set). Prima cosa fare, poi il perché in una frase. **Niente sigle interne**
(B.6, T2, §30, «protocollo B»…) senza dire a parole cosa sono; niente elenchi di
difetti né tono da audit. Le sigle servono nei file di contesto, non nelle risposte.

## File Exclusions

```
claudeMdExcludes:
- ".claude/archive/**"
- "logs/**"
- "notebooks/**"
- "embeddings/**"
- "wandb/**"
- "**/__pycache__/**"
- "**/*.ckpt"
- "**/*.npz"
- "**/*.jpg"
- "**/*.jpeg"
```
