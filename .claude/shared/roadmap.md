# Roadmap — chiusura del progetto (dal 10 set 2026)

Piano, non stato: lo stato vivo è `.claude/TODO.md`, i numeri
`.claude/shared/status.md`. La storia delle fasi A-C è in `status.md §9-§32`.

**Scadenza: progetto + report entro fine settembre / primissimi di ottobre
2026** (decisa dall'utente l'11 set: ~20 giorni). Le date di §2 sono una **guida**,
non un vincolo duro (utente, 11 set). Da qui si lancia solo ciò che
riempie una cella del report; ogni idea nuova va in `PAPER.md §9`, non in un job.

## 1. Il punto di arresto (definition of done)

Compiuto quando queste tabelle e figure hanno i numeri sul **protocollo B**
(`experiments.md`) e il test è stato letto **una volta**.

| Output | Contenuto | Fonte | Stato |
|---|---|---|---|
| **T1** setup | split, gallery condivisa, query, esclusioni, floor | §20, §29 | ✅ |
| **T2** vision | 53 config (52 frozen + head) col metro a tre danni + full C/T/G | §43-§44 (metro §38); ~~§24 + head conv (§33)~~ superati | ✅ numeri esistono (full letto per vincitore e confronti) |
| **T3** rami + fusione | vision · graph · late fusion · oracolo, per-asse + AUC | §49-§51.1 | ✅ valid (α*=0.6, +0.174 sul ramo migliore) + 2 controlli (effetto d'insieme) |
| **T4** ablation | OFAT graph (§3) · pooling×trasformazione vision · pesi geometria (§22) | esistono | ✅ |
| **T5** tipi di danno | stanze tolte vs crop vs patch, stessa frazione, config finali | §37, §43-§44 | ✅ numeri esistono |
| **F1** teaser | query intera/mascherata con top-5 per asse | `08_visualize` (disegna tutti i danni, §39) | ❌ 1 run sulla config finale |
| **F2** pipeline | i due rami + GT condivisa | disegno | ❌ |
| **F3** curva masking | MRR self-recovery vs f, bande CI, vision vs graph (stanze = `nowalls`) | per-query a disco | CPU |
| **F4** loss ≠ retrieval | scatter loss finale vs nDCG, un punto per run | csv esistente, incompleto. ⚠️ il claim vale **fra** varianti (graph), non dentro il training della head | CPU |
| **F5** classi di equivalenza | distribuzione composizione vs topologia | `GalleryAxes` | CPU |
| **F6** robustezza ablation | boxplot per encoder | notebook `§ 5` | CPU |

Grafici (decisione utente 11 set): si producono i più significativi per il
report, poi si scelgono; niente produzione a tappeto.

## 2. Cosa manca, in ordine — con date indicative

0. ✅ **Fatto l'11 set** (§32): head vision a convergenza, `dinov3/natural/head-probe-conv`
   senza whitening, nessun costo sul full (§32.2) · graph: D regge su tutti e 3,
   candidato **`gat/asym`** · controllo di rango superato (§32.1) → niente ri-griglia.
1. **Congelare le due config.** ✅ **Vision congelata** (§33): `dinov3/natural/head-probe-conv`
   senza whitening; miglior frozen per encoder in §33 (servono al passo 2).
   Graph: `gat/asym` candidato, **congelato dopo il passo 2b** (bordo aperto).
   ✅ **14 set (`status.md §42`)**: 2b negativo → **`gat/asym` CONGELATA**.
   ✅ **15 set (§45-§47)**: checkpoint scelto sulla robustezza → **`gat/asymrob` CONGELATA** (AUC 0.457);
   il marcatore pareggia, il 2b negativo di §42 era la selezione.
   ⚠️ **14 set (`status.md §37-§38`)**: la config vision **non è più congelata** — si
   riscieglie dalla griglia frozen completa (ondata 2, in volo) col metro nuovo.
   ✅ **15 set (§44)**: scelta **`pespatial/gem/whiten`** (#1 su 53, costo full dichiarato).
   ✅ **16 set (§48)**: head nuova su pespatial con coppie `nowalls` non adottata → vision **definitiva**.
2. **Nuovi tipi di danno, solo vision** (12-16 set) — decisione utente 11 set.
   - **Crop** (✅ deciso 11 set): rettangolo **bianco** pieno (= sfondo RPLAN e
     stesso riempimento del masking a stanze, `vision_partial_query.py:172,184`),
     bordi **netti**: niente dilatazione né cancellazione di muri attorno.
     Lati casuali e indipendenti, posizione casuale (seed fisso); il rettangolo si
     **accetta** solo se copre la frazione voluta dei **pixel della pianta** (non
     del canvas, non del bbox) ±5%: 25/50/75%. Così non copre mai tutto e il danno
     è confrontabile. Asse x delle curve per tutti e tre i tipi di danno =
     frazione di **area della pianta** effettivamente tolta.
   - **Patch** (✅ deciso 11 set: **sostituire**): patch casuali allineati alla
     griglia dell'encoder, **sbiancati** dopo il resize, stessa frazione di area.
     Il bianco in RPLAN è lo sfondo («qui non c'è niente»): è la sostituzione più
     vicina al togliere senza toccare i modelli. Numero di token invariato
     (pooling invariato), niente chirurgia su modelli con RoPE.
   - Si valutano le config finali + il miglior frozen per encoder, sul valid.
     Domanda: la head (allenata sul masking a stanze) regge anche crop e patch?
     → **No** (§37, H1 falsificata). Aggiunto `nowalls` (§36); ondata 2 = griglia
     frozen completa sui tre danni (§38-§39).
   - Il graph non ha pixel: resta sul masking a stanze (ma vedi 2b).
2b. **Graph «bordo aperto»** (✅ deciso dall'utente l'11 set): alternativa di masking
   del graph, confrontata in robustezza con quella attuale. A ogni stanza rimasta
   si aggiunge l'informazione dei collegamenti **persi** verso stanze tolte: è
   l'equivalente dei muri aperti del vision (oggi il grafo parziale non ha traccia
   di ciò che manca → asimmetria che spiega parte del crollo di §30). Serve
   riallenare `gat/asym` con il marcatore (una variabile). Design in corso
   (`architect`). Adottato se il delta AUC appaiato ha CI che esclude 0; costo sul
   full dichiarato. ❌ **Chiuso 14 set (§42): non adottato** (−0.134 AUC; sonda full ferma a
   epoca 6, confound dichiarato); la robustezza di `asym` si trasferisce al danno `semantic`.
3. ✅ **FATTA (17 set, `status.md §49-§51.1`)** — **Late fusion + oracolo** (CPU): un peso α scelto sul
   valid (punteggi dei due rami combinati) + oracolo per query. Il full usa gli
   `embeddings.npy`; il partial la fusione dei ranking sui `ret_rows` (top-100,
   stesse query e stesse stanze tolte nei due rami). Controllo di correttezza:
   α=1 riproduce il vision, α=0 il graph. ✅ Confermata (11 set). Si misura su
   pianta intera e stanze tolte, gli unici danni che **entrambi** i rami possono
   ricevere (crop e patch non esistono per il graph) → non dipende dal passo 2.
   ⚠️ 14 set: lato vision le «stanze tolte» sono le run `nowalls-random` (§37), non il
   `random` storico, e la config vision è quella nuova dell'ondata 2.
3b. **Testa congiunta** (✅ scelta dall'utente l'11 set; distillazione ancora
   indecisa): MLP sulla concatenazione [vettore vision ; vettore graph], encoder
   congelati, InfoNCE con positivo = stessa pianta con **le stesse stanze tolte**
   in entrambi i rami, selezione dell'epoca sulla robustezza nel valid (come la
   head). Confronto: testa congiunta vs late fusion vs singoli rami. ~4-5 giorni;
   da progettare con l'`architect` (coppie appaiate fra i rami = contratto nuovo).
   ⚠️ 14 set: lato vision le coppie vanno fatte col render `nowalls`.
4. ✅ **FATTO (18 set, `status.md §52` pre-reg, §53 risultati)** — **Pre-registrazione + test, una volta sola**. Prima si scrive in
   `status.md` cosa si misura (config congelate, full + stanze + crop + patch,
   floor); poi **un** job per ramo. La fusione sul test si calcola dopo, su CPU,
   dai file del test, con l'α già scelto sul valid. ✅ Script pronti (11 set):
   `evaluation/03` (vision, full|partial) e `evaluation/07` (graph), comandi in
   `COMANDI.md § Test finale`. ⚠️ Dal 14 set: «stanze» = `nowalls` (via `EXTRA`), metro
   §38, config vision nuova; le date slittano a dopo l'ondata 2.
5. **Circolarità nel testo** (proposta in `architecture.md`): il criterio
   principale (ritrovare la pianta originale) **non** usa le etichette → immune;
   i confronti dentro un ramo sono equi; solo il confronto vision↔graph per-asse
   ne è toccato e si presenta come «informazione esatta vs grossolana».
6. **Code freeze ~19 set**, poi figure e report (~19-30 set). ▶️ **Figure iniziate il 18 set**
   (`status.md §54`): report a **due colonne**, infrastruttura in `src/figures/` (stile condiviso,
   output PDF+PNG+provenienza in `figures/`); fatte **F7** (curva del danno sul test, quattro
   sistemi), **F8** (AUC vs α, tre coppie), **F1** (teaser), **F3** (tre danni + artefatto dei
   muri), **F2** (schema della pipeline), **F5** (classi di equivalenza) e **F6** (box delle
   ablation per encoder) — tutte su CPU, nessun job. Resta la sola **F4** (decisione aperta).
   ⚠️ **Il testo lo scrivono gli utenti** (18 set): da qui numeri, figure e verifiche a richiesta.

## 3. Stato delle voci fuori dal percorso principale (decisioni utente, 11 set)

| Voce | Stato | Nota |
|---|---|---|
| **O3** ResPlan / CubiCasa5K | ⏸ **stand-by** — da discutere coi dottorandi, non eliminato | serve una conversione dati verso i campi `.mat` |
| **O4** crop + masking a patch | ✅ **dentro** (passo 2) | patch: togliere vs sostituire → una sola variante |
| **O5 b** testa congiunta | ✅ **dentro** (passo 3b) | scelta dall'utente l'11 set |
| **O5 c-d** distillazione, cross-attention | ⏸ distillazione indecisa (complessità) · cross-attention non sta in 20 giorni | |
| altri allenamenti (epoche, τ, LoRA, varianti graph) | ✂️ fuori, **salvo** quelli che correggono un difetto scoperto (es. 2b) | decisione utente 11 set: niente training inutili |
| GED topologica, validazione umana del proxy | ✂️ fuori | lunghe → future work |
| `semantic`/`topology` sul protocollo B, M1/M2/M4a-b | ✂️ fuori | non entrano in tabella (`pespatial` full `.npz`: ✅ fatto 15 set, §44) |
| Fase 2 generativa | ✂️ fuori | altro progetto |

**Fusioni intermedie, in ordine di fattibilità** (per la discussione):
(b) **testa congiunta** — MLP allenata con InfoNCE sulla concatenazione dei vettori
già calcolati dei due rami, encoder congelati: riusa codice e dati, ~4-5 giorni,
confronto diretto con la late fusion (stessi input). · (c) **distillazione** — un
ramo impara a imitare l'altro con una loss in più (es. il graph verso il vision,
molto più robusto): riallenare la GNN + un iperparametro; se è il vision a
imparare dal graph importa la circolarità. · (d) **cross-attention** patch↔nodi —
la più potente, ma servono i token di patch (non salvati: decine di GB o backbone
nel loop) e un'architettura nuova: non sta in 20 giorni.

**Da dichiarare, non da correggere** (nessun effetto sui numeri, verificato il
10 set): B1 (YAML graph ≠ variante: variante = YAML + flag di `03_train_gnn.sh`) ·
B4 (booleani oggi tutti mappati) · B5 (errore generico su `raw_skip`) · label
`Recall` in `axis_metrics.py:100`.

## 4. Regole di chiusura (per ogni sessione)

1. Una run si lancia **solo** se riempie una cella di §1.
2. Nessun criterio nuovo: si applicano quelli già scritti (§23, §30.6).
3. Un esito negativo non si «salva» con un'altra run: si riporta.
4. Il test si legge una volta, dopo la pre-registrazione.
5. Codice solo se cambia un numero in tabella o sblocca un passo di §2.

## 5. Obiettivi dell'utente — dove sono finiti

| ID | Obiettivo | Esito |
|---|---|---|
| O1 | Incoerenze metodologiche | ✅ fasi A-B (§20-§32); resta congelare le config |
| O2 | TIPS-v2, PE Core, PE Spatial | ✅ misurati (§17-§19, §24) |
| O3 | ResPlan + CubiCasa5K | ⏸ stand-by |
| O4 | Crop + masking a patch | ✅ fatto (§37, §43-§44) + `nowalls` (§36) |
| O5 | Pipeline che comunicano | late fusion (passo 3) + testa congiunta (3b); distillazione ⏸ |
| O6 | Report 10 pagine + 2, stile CVPR | ~19-30 set (`PAPER.md §10`) |

## 6. Domande aperte

1. Circolarità: ok alla presentazione del passo 5?
2. Distillazione: sì/no (oggi indecisa).
3. O3: esito della discussione coi dottorandi.
4. ~~Head vision nuova sì/no~~ → **sì** (utente, 15 set), pre-registrata in `status.md §46` →
   ✅ **chiusa 16 set (§48)**: H3 falsificata, non adottata; config vision definitiva `pespatial/gem/whiten`.
