# TODO — stato operativo vivo

**Ultimo aggiornamento:** 18 set 2026 — **test letto** (§53, fusione 0.6424 vs graph 0.4700 / vision
0.3965) e **figure 7 su 8** (§54). Config definitive: vision `pespatial/gem/whiten` (§44) · graph
`gat/asymrob` (§47). ⚠️ **Il testo del report lo scrivono gli utenti** (decisione 18 set): dalla nostra
parte non resta lavoro obbligato, solo le voci in *Bloccato*.

> **Primo** file a inizio sessione, **ultimo** a fine task: dice dove siamo
> *adesso*. Regole in `CLAUDE.md § Stato che si aggiorna da solo`. Tetto ~60
> righe: si comprime prima di aggiungere, la traccia va in `status.md`.

## In corso

Piano: `roadmap.md` (chiusura). Storia e numeri: `status.md` — fasi A-C e loop 1-7 (§20-§33),
quadro nuovo da §35. **Fase 1 chiusa**: restano solo voci opzionali.

- [x] **Vision (§35-§44)**: muri interni nel render storico → `nowalls`; H1 falsificata; metro §38;
      53 config → **`pespatial/gem/whiten`** R 0.5245.
- [x] **Head nuova (§46-§48)**: `training.damage` + 11 test; candidato R 0.4912 (CI < 0) →
      **H3 falsificata, non adottata**.
- [x] **Graph (§42, §45-§47)**: checkpoint scelto sulla robustezza → **`gat/asymrob` congelata**
      (AUC 0.249 → 0.457) · rumore fra training identici ~0.04 · `semantic` +0.366.
      ⚠️ ordine gat ≫ sage/gcn non riverificato (caveat).
- [x] **Fusione + 2 controlli (§49-§51.1)**: `query_vectors`/`late_fusion`/`fusion_select`,
      script `evaluation/08-14`. Valid: α*=0.6, 0.630 vs graph 0.456 / vision 0.392, sopra l'oracolo ·
      scala dei guadagni +0.034 / +0.098 / +0.174 → **complementarità**, ma ~56% diversità di modello.
- [x] **Test finale (§52 pre-reg, §53 risultati, letto una volta)**: controlli PASS · fusione
      **0.6424** vs graph 0.4700 / vision 0.3965 (+0.1725 [+0.1625, +0.1826]), oracolo 0.5693 ·
      vision 3 danni 0.5210 · previsioni 3/4 (graph +0.0138 fuori ±0.01, in meglio).
- [x] **Figure (§54, `PAPER.md §10.b`)**: stile a **due colonne** in `src/figures/style.py`, output in
      `figures/` (PDF + PNG + `*.sources.txt`), 19 test nuovi (**316 passed**). Fatte **7 su 8**, tutte
      **senza job**: F1 teaser · F2 pipeline · F3 tre danni + artefatto dei muri · F5 classi di
      equivalenza · F6 box delle ablation · F7 danno sul test · F8 AUC vs α. ⚠️ F1 e F3 hanno
      un'inquadratura diversa dal piano (dichiarato in `PAPER.md §10.b`). **F4** → *Bloccato*.
- [ ] **Visualizzazioni** dinov3/gem raw+whiten (`106381-82`): esito mai controllato. Non serve più
      a nessuna figura (F1 è fatta senza job): si guarda solo se qualcuno le vuole.

## In attesa dell'utente (job sbatch)

- (nessun job in coda: head nuova chiusa il 16 set, §48)
- ⏳ Viz `106381-82` (dinov3/gem raw, whiten) → `results/visualizations/dinov3_gem_*` (facoltativa).

## Prossimo passo

**Nulla di obbligato da questa parte**: numeri, figure e documentazione sono allineati; il testo del
report è in mano agli utenti (18 set). Opzionali, solo se lo chiedono: F4, la testa congiunta
(seconda tornata pre-registrata) e l'aggiornamento della knowledge graph.

## Bloccato / in attesa di decisione

- **Domande aperte** (`roadmap.md §6`): testa congiunta · circolarità nel testo · O3.
- [ ] **F4** (scatter loss finale ↔ retrieval): farla dichiarando il buco o lasciarla fuori?
      `notebooks/vision/vision_encoders_results.csv` ha **11 serie su 15** (manca dinov2, manca
      `ijepa_mean`). Decisione degli utenti.
- Note minori: 4 file `*e300*` incompleti in `vision_partial_valid_B` · causa del primo
  fallimento tipsv2 non diagnosticata.
- [ ] **Committare luglio** (solo l'utente): `.claude/archive/CLAUDE-2026-07-29-full.md`
      è l'unica copia della cronologia di Fase 1; dopo il commit si elimina.
- [ ] Knowledge graph ferma al **25 ago** → `/graphify --update` quando i doc si stabilizzano.
- [x] Script del test pronti (11 set) · comandi alle config finali (16 set).
