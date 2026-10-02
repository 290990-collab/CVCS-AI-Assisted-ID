# TODO — stato operativo vivo

**Ultimo aggiornamento:** 2 ott 2026 (SAGE vs GAT: risultati in §58.1) · 1 ott 2026 (verifica SAGE vs GAT lanciata, §58) · 28 set 2026 — numeri **ri-verificati** e riversati in **13 CSV** (§55), più
il **notebook di analisi** riorganizzato e due correzioni ai CSV (§56). Nessun numero del report cambia.
Config definitive: vision `pespatial/gem/whiten` (§44) · graph `gat/asymrob` (§47). ⚠️ **Il testo del
report lo scrivono gli utenti** (18 set): non resta lavoro obbligato, solo le voci in *Bloccato*.

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
- [x] **Figure (§54, `PAPER.md §10.b`)**: stile a **due colonne** (`src/figures/style.py`), output in
      `figures/` (PDF + PNG + provenienza). Fatte **7 su 8** senza job; F1 e F3 con inquadratura
      diversa dal piano (dichiarato). **F4** → *Bloccato*.
- [x] **Ri-verifica + CSV + notebook (28 set, §55-§56)**: tutto ricalcolato dai per-query, coincide
      con §53 · null `random` sul test (0.7403/0.4663/0.8700) · rumore di *valutazione* graph
      +0.00039 · `src/evaluation/export_csv.py` → 13 CSV in `results/csv/` ·
      `analisi_risultati.ipynb` riordinato ed eseguito · ⚠️ per-query ora **senza arrotondamento**
      (i pareggi contavano come vittorie: 1394 vs **1218** sull'oracolo).
- [x] **`PAPER.md` riscritto (1 ott)**: autosufficiente per il paper CVPR (numeri finali, repliche, competitor,
      storia del progetto §10 senza bug, previsioni smentite, claim vietati, caveat `gem`). Copia vecchia nello
      scratchpad della sessione. Aperti per gli utenti: circolarità (§7.1), tenere o no l'artefatto dei muri (§10).
- [x] **Figure in inglese (1 ott)**: `figures/english/` = stesse 7 figure di `figures/italian/` con
      `--lang en --out-dir figures/english`; numeri identici. Ritocchi solo all'inglese: separatore
      migliaia 67,405 (classes, pipeline), etichette accorciate (ablation, damage_test, pipeline).
- [ ] **Competitor (1 ott, richiesta dottorandi)**: `COMPETITORS.md` = 9 lavori scelti, tutti con PDF in
      `papers/comp/` (SHRAG e Uda&Ozaki tolti: non open access). Implementabile solo LayoutGKN (+ LayoutGMN,
      costoso). Prossimo: clonare il repo LayoutGKN e verificare dati, mining e id delle piante.
- [x] **Verifica SAGE vs GAT (1 ott)**: riallenare `sage/asymrob` × 4 repliche con la stessa regola di
      `gat/asymrob` e confrontarle appaiate sul valid. Piano, comandi e regola di verdetto in
      `VERIFICA_GRAFI.md §4`. Nessun file esistente sovrascritto. §58 scritto, job lanciati (vedi sotto).
      ✅ Scelta **(a) analisi aggiuntiva** scritta in §58 (2 ott) prima di leggere i numeri.
      **Esito (§58.1)**: SAGE più robusto in 4/4 repliche, Δ AUC medio +0.0255 (CI tutti > 0); P1 smentita.
      Deviazione sul controllo MRR f=0.0 (2/4 sotto 0.965) accettata e dichiarata → verdetto **SAGE più robusto**.
      Fusione invariata con GAT. PAPER.md e VERIFICA_GRAFI.md aggiornati (2 ott). ✅ chiuso.
- [ ] **Visualizzazioni** dinov3/gem raw+whiten (`106381-82`): esito mai controllato. Non serve più
      a nessuna figura (F1 è fatta senza job): si guarda solo se qualcuno le vuole.

## In attesa dell'utente (job sbatch)

- ✅ **Repliche multi-seed chiuse (§57-§57.3, 1 ott)**: 3 repliche + storica, controlli PASS, test n=4 fusione
  0.6401 ± 0.0031 · graph 0.4697 ± 0.0035 · vision 0.3916 ± 0.0045; P2-P5 ✅, P1 ❌ sul test. `results/fusion/seeds/summary.json`.
  Figura `f_damage_test_seeds` (it + en, `src/figures/damage_test_seeds.py`): curva test, media ± sd fra repliche.
  Ricetta per seed nuovi (proposta 400042/500042/600042) in `COMANDI.md` § Repliche multi-seed.

- ✅ **SAGE vs GAT (§58)**: 127996–128003 tutti COMPLETED, controlli e confronti fatti il 2 ott → §58.1.
- ⏳ Viz `106381-82` (dinov3/gem raw, whiten) → `results/visualizations/dinov3_gem_*` (facoltativa).

## Prossimo passo

**Nulla di obbligato da questa parte**: numeri verificati, figure e documentazione allineate, CSV in
`results/csv/` pronti per i notebook. Opzionali, solo se lo chiedono: F4, la testa congiunta
(seconda tornata pre-registrata) e l'aggiornamento della knowledge graph.

## Bloccato / in attesa di decisione

- **Domande aperte** (`roadmap.md §6`): testa congiunta · circolarità nel testo · O3.
- [ ] **F4** (scatter loss finale ↔ retrieval): farla dichiarando il buco o lasciarla fuori?
      `notebooks/vision/vision_encoders_results.csv` ha **11 serie su 15** (manca dinov2, manca
      `ijepa_mean`). Decisione degli utenti.
- Note minori: 4 file `*e300*` incompleti in `vision_partial_valid_B` · causa del primo
  fallimento tipsv2 non diagnosticata.
- [ ] **Push su GitHub (1 ott)**: bloccato, 104 cartelle di `.git/objects` di gangelillis senza scrittura
      di gruppo → `chmod -R g+w .git/objects`. Fatti: `paper-skills/` in `.gitignore`, `core.sharedRepository=group`.
- [ ] **Committare luglio** (solo l'utente): `.claude/archive/CLAUDE-2026-07-29-full.md`
      è l'unica copia della cronologia di Fase 1; dopo il commit si elimina.
- [ ] Knowledge graph ferma al **25 ago** → `/graphify --update` quando i doc si stabilizzano.
- [x] Script del test pronti (11 set) · comandi alle config finali (16 set).
