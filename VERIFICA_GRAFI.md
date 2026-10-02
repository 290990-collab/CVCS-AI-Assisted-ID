# Verifica del ramo graph: GAT è davvero meglio di GraphSAGE?

> **Esito (2 ott 2026): SAGE è più robusto di GAT a parità di regola** (+0.026 di AUC sul valid, 4/4
> repliche). La fusione resta con GAT (analisi aggiuntiva). Dettagli nella sezione 5.

*Scritto il 1 ott 2026. Nessun job lanciato e nessun numero nuovo: è un piano. I numeri citati
vengono da `.claude/shared/status.md` (§ indicati) e dai documenti di riepilogo.*

## 1. Perché la config finale è GAT

### «Migliore» dipende dal metro

**Sulla pianta intera GAT è il peggiore dei tre.** Media nDCG@10 (vedi
`spiegazione_metriche_e_risultati_grafi.md`): GCN 0.8870 > SAGE 0.8750 > GAT 0.8713. La migliore a
pianta intera resta `gcn/tau02` (`recap.md`).

**GAT vince perché il 24 agosto si è deciso che «migliore» significa «più robusto»** (§23): conta
l'AUC del self-recovery quando si tolgono stanze dalla query. Con le coppie «pianta intera ↔ stanze
tolte» (§32):

| encoder | AUC robustezza | guadagno asym − base |
|---|---|---|
| **gat/asym** | **0.249** | +0.242 |
| sage/asym | 0.111 | +0.102 |
| gcn/tau02asym | 0.097 | +0.091 |

Su questo metro GAT fa circa 2.6 volte SAGE. Rispetto alla versione simmetrica perde meno topologia
(−0.024 contro −0.043 di SAGE) e guadagna un po' di geometria (+0.005). In cambio perde più
composizione (−0.050 contro −0.036).

### Perché GAT dovrebbe reggere meglio (ipotesi, non verificata)

GAT è l'unico dei tre che usa `edge_attr`, cioè il tipo di relazione RPLAN fra stanze
(`configs/graph_models/gat.yaml`), e pesa i vicini con un'attenzione dinamica. Quando mancano
delle stanze, le relazioni rimaste e la possibilità di ripesare i vicini superstiti potrebbero dare
più informazione della media fissa dei vicini di SAGE. Nel progetto questo non è stato testato.

### Il vantaggio su SAGE non è dimostrato

Lo dice il progetto stesso (`recap.md`, §47 «Caveat Q1»):

1. **L'ordine GAT ≫ SAGE/GCN viene dalla regola vecchia** di scelta del checkpoint (topologia a
   pianta intera). Quella regola era erratica: su tre training identici di GAT ha scelto le epoche 6,
   31 e 33, con AUC 0.107, 0.249 e 0.267. Lo 0.249 contro 0.111 dipende in parte da quale epoca è
   capitata. `sage/asym` in particolare è stato scelto all'**epoca 3**: il suo 0.111 sottostima quasi
   certamente quanto può fare SAGE.
2. **Con la regola nuova (`asymrob`) è stato riallenato solo GAT**, che è salito a 0.457. SAGE e GCN
   non sono stati riallenati con la stessa regola.
3. **Cambiare regola vale +0.35 di AUC**, più del distacco fra GAT e SAGE (~0.14): con la regola
   nuova SAGE potrebbe ridurre il distacco o anche ribaltarlo.
4. **Rumore fra training.** §47 lo stimava in ~0.04 di AUC; le repliche multi-seed di §57 danno una
   deviazione standard di circa 0.007 sul valid (0.0035 sul test). Bastano quindi differenze molto più
   piccole di quanto si pensava per separare GAT da SAGE.

**In sintesi:** GAT è la config finale (`gat/asymrob`) perché una regola fissata prima dei risultati
(§45) l'ha scelta sul metro di robustezza. Non è stato dimostrato che sia migliore di GraphSAGE a
parità di condizioni.

## 2. Piano di verifica, in ordine

### Passo 0 — Pre-registrazione (`status.md §58`), prima di ogni job

Tre decisioni dell'utente da scrivere prima:

- **Ricetta.** Proposta: la stessa di `gat/asymrob` per tutti (YAML base, τ=0.3, selezione sulla
  robustezza, tetto 300 epoche). GCN prima usava τ=0.2 (`tau02asym`): se si vuole anche quella,
  serve una variante nuova in `scripts/graph/03_train_gnn.sh` e `scripts/graph/04_eval_gnn.sh`
  (due righe).
- **Repliche.** Proposta: seed 42 + 100042, 200042, 300042, cioè 4 per encoder come GAT (§57). Il
  confronto diventa fra medie con la loro deviazione standard.
- **Regola di verdetto.** Esempio: «GAT è confermato se Δ AUC GAT − SAGE ha CI > 0 sulla replica 0 e
  la media di GAT è più alta su 4 repliche». Va scritto anche cosa succede se vince SAGE (passo 3).

### Passo 1 — Training (nessuna modifica al codice)

Le varianti `asymrob` e `asymrob_s<S>` funzionano già per qualunque encoder:

```bash
sbatch scripts/graph/03_train_gnn.sh graph_sage asymrob
sbatch scripts/graph/03_train_gnn.sh graph_sage asymrob_s100042   # e così s200042, s300042
sbatch scripts/graph/03_train_gnn.sh gcn asymrob                  # e così i 3 seed
```

Sono 8 job indipendenti, da lanciare in parallelo. Un training GAT da 300 epoche ha richiesto
1h12–1h37 (§47); SAGE e GCN dovrebbero metterci al massimo altrettanto. GAT era arrivato al tetto
delle 300 epoche e quasi certamente succederà anche a SAGE e GCN: va dichiarato, non si rilancia.

### Passo 2 — Valutazione della robustezza sul valid

**Replica 0 (nessuna modifica al codice):**

```bash
sbatch scripts/evaluation/06_perquery_graph_partial_valid.sh graph_sage asymrob
sbatch scripts/evaluation/06_perquery_graph_partial_valid.sh gcn asymrob
python -m src.evaluation.robustness_auc compare --a <gat/asymrob> --b <sage/asymrob>
```

Il confronto è appaiato con i numeri di `gat/asymrob` già su disco (§47): stesse query e stesse
stanze tolte.

> ⚠️ **Non** rilanciare 06 su `gat asymrob`: riscriverebbe
> `embeddings/graph/gat/asymrob/embeddings.npy`, il cui sha1 è fissato dai query vector della fusione.

**Repliche con seed (piccola modifica):** per confrontarle con le repliche GAT bisogna togliere le
stesse stanze, cioè usare `--partial-seed S`.

- `06_perquery_graph_partial_valid.sh` non lo passa.
- `09_queryvec_graph.sh` lo passa, ma ha `TARGET=gat` fisso e scrive nelle cartelle della fusione.
- Strada consigliata: aggiungere a 06 un seed opzionale che scriva in
  `results/perquery/seeds/s<S>/graph_partial_valid/`. Circa 15 righe di bash; il Python
  (`graph_evaluate`) supporta già `--partial-seed`.

**Facoltativo:** costo a pianta intera di `sage/asymrob` con lo stage 04 (nessuna modifica).

### Passo 3 — Verdetto e conseguenze sulla late fusion

**Se GAT è confermato:** la late fusion non cambia e non c'è niente da rifare. Si aggiornano
`status.md` e `PAPER.md` togliendo il caveat Q1 di §47, e la scelta di GAT diventa difendibile.

**Se SAGE pareggia o vince:** c'è lavoro di codice e un problema di metodo.

- *Codice:* `scripts/evaluation/09_queryvec_graph.sh`, `10_late_fusion.sh`,
  `15_seed_fusion_select.sh` e `src/evaluation/seed_summary.py` hanno `gat/asymrob` fisso nel codice
  (anche export CSV e figure, da verificare). Vanno parametrizzati. Poi bisogna rifare tutto: query
  vector, controlli C1–C5, scelta di α sul valid, test, 4 repliche.
- *Metodo:* il test è già stato letto (§53, §57). Cambiare adesso la config graph vorrebbe dire
  sceglierla dopo aver visto il test. Il pulito è dichiararlo e riportare la fusione con SAGE come
  analisi aggiuntiva, non come sostituto dei numeri principali. Questa decisione va scritta nel §58
  **prima** di vedere i risultati.

## 3. Stima dei tempi

| | lavoro manuale | GPU, tempo reale |
|---|---|---|
| Pre-registrazione | ~30 min | — |
| Seed in 06 + test | ~1 h | — |
| Training (8 job in parallelo) | lancio | ~2 h, coda permettendo |
| Valutazione con 06 | lancio | ~1–2 h (stimato, non misurato) |
| Analisi + aggiornare `status.md` e `PAPER.md` | ~1–2 h | — |
| **Totale se GAT è confermato** | **~mezza giornata** | **~1 giorno** compresa la coda |
| In più se vince SAGE (fusione rifatta) | +1 giorno di codice e controlli | +1 giorno |

## 4. Piano minimo di esecuzione (1 ott 2026)

**Obiettivo:** confrontare SAGE e GAT a parità di ricetta e di regola di selezione, con 4 repliche ciascuno,
**senza modificare codice o config e senza sovrascrivere nulla** di esistente. Se si decide di abbandonare
la verifica, si cancellano due gruppi di cartelle nuove e il progetto torna esattamente com'era.

**Perimetro (minimo indispensabile):**
- **Sì:** SAGE con la ricetta di `gat/asymrob`, 4 repliche (seed 42 + 100042, 200042, 300042), valutazione
  sul **valid** con la curva di robustezza `random`.
- **No:** GCN, varianti di τ, tuning, test, fusione. GCN resta fuori e va dichiarato come limitazione.
- **GAT non si riallena e non si rivaluta:** si usano i file già su disco.

### 4.1 Cosa viene scritto e dove (verificato sul codice)

| Chi | Scrive | Esiste già? |
|---|---|---|
| `03_train_gnn.sh graph_sage <V>` | `embeddings/graph/sage/<V>/` e `embeddings/graph/sage/<V>_selfull/` | No (in `sage/` ci sono solo `asym`, `base`, `nd01`, `noaug`, `nojitter`, `noskip`, `nosym`, `selmean`, `tau02`, `tau05`) |
| `04_eval_gnn.sh graph_sage <V>` | `embeddings.npy` + `names.json` dentro `embeddings/graph/sage/<V>/` | No: è la cartella appena creata dal training |
| `04_eval_gnn.sh` con `PERQUERY_OUT` | `results/perquery/verifica_grafi/s<S>/graph_sage_<V>_partial-random-f*_valid.npz` | No: cartella nuova |
| entrambi | `logs/gp_03_train_gnn_<job>.*`, `logs/<nome job 04>_<job>.*` | No: nomi con l'id del job |

Non vengono toccati config, script, `results/queryvec/`, `results/fusion/`, `results/perquery/seeds/`,
`results/perquery/graph_partial_valid/` né nulla sotto `embeddings/graph/gat/`. I file di GAT usati per il
confronto vengono solo **letti**.

> ⚠️ Non usare `06_perquery_graph_partial_valid.sh`: senza argomenti rilancia le sue run di default (anche
> GAT) e, anche con argomenti, scrive in `results/perquery/graph_partial_valid/`. Si chiama 04 direttamente,
> con le variabili d'ambiente che 04 già supporta. È lo stesso meccanismo usato da 06 e 09.

**Ritorno indietro** (se si decide di interrompere):
```bash
rm -r embeddings/graph/sage/asymrob embeddings/graph/sage/asymrob_selfull \
      embeddings/graph/sage/asymrob_s{100042,200042,300042} \
      embeddings/graph/sage/asymrob_s{100042,200042,300042}_selfull \
      results/perquery/verifica_grafi
```
I log restano e non danno fastidio.

### 4.2 Stasera — passo 0: pre-registrazione in `status.md §58` (PRIMA dei job)

Va scritta prima del lancio, perché domattina il `training_summary.json` mostrerà già la sonda di robustezza
di SAGE. Testo proposto, da incollare e completare:

> **§58. PRE-REGISTRAZIONE — SAGE vs GAT a parità di regola di selezione (1 ott, PRIMA dei job)**
>
> **Domanda:** l'ordine gat ≫ sage (§32, regola vecchia) regge con la regola di §45? Chiude il caveat Q1 di §47.
> **Ricetta:** `configs/graph_models/graph_sage.yaml` invariato (τ=0.3, `aggr: mean`) + variante `asymrob`
> (coppie `asym_partial`, sonda di robustezza, tetto 300, patience 0, ombra `_selfull`), identica a `gat/asymrob`.
> **Repliche:** `sage/asymrob` (seed 42) + `sage/asymrob_s{100042,200042,300042}`; danno di valutazione
> = default per la replica 0 e `--partial-seed S` per le altre. Così ogni replica SAGE è appaiata alla
> replica GAT corrispondente (§47 e §57): stesse 2000 query, stessa gallery, stesse stanze tolte.
> **Metro:** AUC self_rr `random` sul valid (§23), `robustness_auc compare`, bootstrap B=10000.
> **Controlli di validità** (se uno fallisce → nessun verdetto, si diagnostica): gallery sha1 `0c24cfc05e18` ·
> 2000/2000 appaiate · MRR a f=0.0 ∈ [0.965, 0.980] · nessun avviso di danno diverso da `compare`.
> **Budget:** tetto 300 epoche; se best_epoch > 270 → dichiarato, nessun rilancio.
> **Regola di verdetto** (Δ = SAGE − GAT):
> - **GAT confermato** se Δ della replica 0 ha CI interamente < 0 **e** la media dei 4 Δ appaiati è < 0.
> - **SAGE migliore** se Δ della replica 0 ha CI interamente > 0 **e** la media dei 4 Δ è > 0.
> - **Altrimenti pareggio → resta GAT** (default; lo spareggio §23.1 **non** si applica).
> **Se SAGE è migliore:** [SCEGLIERE ORA: (a) analisi aggiuntiva, `gat/asymrob` resta la config del paper ·
> (b) sostituzione dichiarata: SAGE diventa la config graph, fusione rifatta per intero (§49-§57) con il test
> letto una sola volta e i numeri GAT riportati]. Nel caso (b) il costo a pianta intera di SAGE (guardrail
> §23) si misura prima della fusione.
> **Previsione:** P1: Δ < 0 (GAT più robusto, come in §32).
> **Fuori perimetro (dichiarato):** GCN, τ e `aggr` non ottimizzati per la regola nuova; il verdetto vale
> per «questa ricetta», non per l'architettura in generale.

### 4.3 Stasera — passi 1 e 2: training + valutazione a cascata

Un solo blocco lancia 4 training e, per ognuno, la valutazione che parte **solo se il training termina
con successo** (`afterok`). Tempi attesi: training ~1h30–1h50 (le repliche GAT di oggi ci hanno messo
~1h50), valutazione ~7–10 min. Le 4 catene girano in parallelo: domattina dovrebbe essere tutto pronto.

```bash
cd /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID

# Guardia: nessuna destinazione deve esistere già.
OK=1
for V in asymrob asymrob_s100042 asymrob_s200042 asymrob_s300042; do
  for D in embeddings/graph/sage/$V embeddings/graph/sage/${V}_selfull; do
    [ -e "$D" ] && { echo "!! ESISTE GIA': $D"; OK=0; }
  done
done
[ -e results/perquery/verifica_grafi ] && { echo "!! ESISTE GIA': results/perquery/verifica_grafi"; OK=0; }

if [ $OK = 1 ]; then
  for S in 0 100042 200042 300042; do
    if [ $S = 0 ]; then V=asymrob; PSEED=""; else V=asymrob_s$S; PSEED="--partial-seed $S"; fi
    JT=$(sbatch --parsable scripts/graph/03_train_gnn.sh graph_sage $V)
    JE=$(PERQUERY_OUT=results/perquery/verifica_grafi/s$S \
         GRAPH_PARTIAL_FLAGS="--partial --partial-strategies random --partial-fractions 0.0 0.25 0.5 0.75 $PSEED" \
         GRAPH_EXTRA_FLAGS="--split valid" \
         sbatch --parsable --export=ALL --dependency=afterok:$JT --time=02:00:00 \
                scripts/graph/04_eval_gnn.sh graph_sage $V)
    echo "replica s$S ($V): training $JT -> eval $JE"
  done
fi
```

Annotare gli 8 id dei job in `status.md §58`.

### 4.4 Domani — passo 3: controlli e lettura

**a) Job completati e percorsi giusti**
```bash
sacct -j <gli 8 id> --format=JobID,JobName,State,ExitCode,Elapsed
grep -h "valori per-query salvati in" logs/*_<id eval>.log   # devono puntare a results/perquery/verifica_grafi/s<S>/
ls results/perquery/verifica_grafi/s*/                        # 4 cartelle × 4 file (f0.0 … f0.75)
```
Se un training fallisce per motivi di infrastruttura, si rilancia lo stesso seed (regola di §57).

**b) Budget e controlli di validità**
```bash
for V in asymrob asymrob_s100042 asymrob_s200042 asymrob_s300042; do
  python -c "import json;d=json.load(open('embeddings/graph/sage/$V/training_summary.json'));print('$V',d['best_epoch'],d['epochs_max'],d['budget_binding'])"
done
python - <<'PY'
import glob, numpy as np
for f in sorted(glob.glob("results/perquery/verifica_grafi/s*/*_partial-random-f0.0_valid.npz")):
    z = np.load(f, allow_pickle=True)
    print(f, "MRR f=0.0 = %.4f" % np.nanmean(z["self_rr"]), "n =", len(z["names"]))   # atteso 0.965–0.980, n = 2000
PY
```

**c) Confronto appaiato SAGE − GAT, replica per replica**
```bash
python -m src.evaluation.robustness_auc compare \
  --a results/perquery/verifica_grafi/s0/graph_sage_asymrob \
  --b results/perquery/graph_partial_valid/graph_gat_asymrob
for S in 100042 200042 300042; do
  python -m src.evaluation.robustness_auc compare \
    --a results/perquery/verifica_grafi/s$S/graph_sage_asymrob_s$S \
    --b results/perquery/seeds/s$S/fusion_branches_valid/graph_gat_asymrob_s$S
done
```
Ogni riga stampa Δ con CI, AUC SAGE vs GAT e n. `compare` verifica da solo che split, query, gallery ed
esclusione del self coincidano, e avvisa se il danno è diverso. La media dei 4 Δ si calcola a mano dalle 4
righe. Riferimento GAT sul valid: 0.4568 (replica 0), media su 4 repliche 0.4661 ± 0.0069 (§57).

**d) Verdetto**
Si applica la regola del §58, si scrivono i numeri in `status.md §58.1` e si aggiorna questo file.
- **GAT confermato o pareggio:** fine. In `PAPER.md` si chiude il caveat Q1 di §47 e la fusione non si tocca.
- **SAGE migliore:** si segue la scelta (a) o (b) già scritta nel §58. Solo nel caso (b) parte il lavoro
  descritto al passo 3 della sezione 2 (parametrizzare 09/10/11/15 e `seed_summary`, rifare fusione e
  controlli §50-§51), con output in cartelle nuove per non sovrascrivere quelle di GAT.

### 4.5 Tempi

| Quando | Cosa | Tempo |
|---|---|---|
| Stasera | Scrivere §58 + lanciare il blocco | ~30–45 min |
| Notte | 4 catene training → eval in parallelo | ~2 h di GPU, più la coda |
| Domani | Controlli a)–c) + verdetto + §58.1 | ~1 h |

Il push su GitHub è indipendente (problema di permessi su `.git/objects`) e non blocca nulla di questo
piano. Le cartelle nuove stanno in `embeddings/` e `results/`, che sono già in `.gitignore`.

---

## 5. Esito (2 ott 2026)

**Scelta registrata prima di leggere i numeri** (`status.md §58`): **(a) analisi aggiuntiva** — GAT resta
nella fusione qualunque sia l'esito, SAGE non va sul test.

**Job**: 127996–128003 tutti COMPLETED (exit 0); risultati per query in `results/perquery/verifica_grafi/s<S>/`
(i log lo confermano). Nessun file esistente toccato.

**Controlli**
- gallery `0c24cfc05e18` in 16/16 file, split valid, self escluso ✅
- 2000/2000 query appaiate in 4/4 repliche, nessun avviso di danno diverso ✅
- MRR a pianta intera (f=0) ∈ [0.965, 0.980]: ❌ in 2 repliche su 4 — SAGE 0.9672 / 0.9655 / 0.9644 / 0.9628,
  GAT sulle stesse query 0.9781 / 0.9765 / 0.9789 / 0.9765. Diagnosi: non è un guasto (con un guasto
  l'MRR sarebbe vicino a 0); la soglia era calibrata su GAT. **Deviazione accettata dall'utente e
  dichiarata** (`PAPER.md` §6.6 punto 7).
- Budget: epoca scelta 295 / 299 / 297 / 299 su 300 → tetto vincolante, dichiarato, nessun rilancio.

**Confronto** (valid, AUC di self-recovery con stanze tolte a caso, Δ = SAGE − GAT, CI 95%)

| replica | SAGE | GAT | Δ [CI] |
|---|---|---|---|
| 0 (quella della fusione) | 0.4779 | 0.4568 | +0.0210 [+0.0123, +0.0298] |
| s100042 | 0.4979 | 0.4691 | +0.0288 [+0.0201, +0.0371] |
| s200042 | 0.4882 | 0.4719 | +0.0163 [+0.0078, +0.0250] |
| s300042 | 0.5030 | 0.4670 | +0.0360 [+0.0275, +0.0445] |
| media | 0.4918 | 0.4662 | **+0.0255** |

**Verdetto**: **SAGE più robusto** (CI della replica 0 > 0 e media dei 4 Δ > 0). La previsione «GAT più
robusto» è smentita: il vantaggio di GAT del 10 settembre era un effetto della regola di selezione vecchia.

**Conseguenze (caso a)**
- La fusione non si tocca: il guadagno è misurato con un ramo graph che non è il più robusto
  disponibile, quindi se mai è conservativo. La tesi «la fusione batte i singoli rami» non cambia.
- `PAPER.md` aggiornato (§5.6, §6.6 punto 7, §7.5, §7.8, §7.9, §8, §9, §10, §11.3).
- Il verdetto vale per questa ricetta: τ e `aggr` non ottimizzati, GCN non riprovato, solo valid.
- Rollback (se si decide di abbandonare la strada): il comando `rm` della sezione 4.1.
