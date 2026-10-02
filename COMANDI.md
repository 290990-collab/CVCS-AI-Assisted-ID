# COMANDI.md — pipeline sbatch (vision + graph retrieval)

Prima parte: ramo **vision** (`scripts/vision/`). In fondo: ramo **graph**
(`scripts/graph/`).

I job vision stanno in **`scripts/vision/`**, numerati nell'ordine di esecuzione.
Coprono la **griglia completa** per ogni modello:

```
pooling (natural | gem | mean)  →  trasformazione (raw/L2 | whiten | head | head+whiten)  →  L2  →  FAISS
```

**Principio:** l'unico artefatto costoso è l'estrazione degli embedding **RAW**, e
dipende solo dalla *feature* (encoder × pooling). Whitening e head sono
trasformazioni applicate **al volo** in valutazione → le 4 combinazioni di
trasformazione **riusano lo stesso RAW** senza ri-estrarre.

- **Feature** (cambia il RAW) = `model.variant=<pooling>` + `model.kwargs.pooling=<pooling>`.
  Una sottocartella RAW per pooling: `embeddings/vision/<modello>/<pooling>/`.
- **Trasformazione** (non cambia il RAW) = flag `head.enabled` / `whitening.enabled`.
  Si distingue nei risultati col `transform_tag` (`raw`/`whiten`/`head`/`head+whiten`).

> **Da zero** (niente `embeddings/`, niente `results/`): il formato è cambiato
> (`embeddings.npy` ora è **RAW**). Cancella `embeddings/` e `results/` vecchi e
> riparti dallo Stage 01.

## Mappa delle dipendenze

```
01 extract_raw ──┬─> 04 eval_frozen_full     (raw + whiten, FULL)
                 ├─> 06 eval_frozen_partial   (raw + whiten, PARTIAL)
                 └─> 02 build_pairs ─> 03 train_head ──┬─> 05 eval_head_full    (head + head+whiten, FULL)
                                                       └─> 07 eval_head_partial (head + head+whiten, PARTIAL)
```

## Gli script

| # | Script | Stage | Dipende da | Peso | Lancio consigliato |
|---|---|---|---|---|---|
| 01 | `01_extract_raw.sh` | A: RAW | — | GPU, medio | 1 job (o per-modello) |
| 02 | `02_build_pairs.sh` | C: coppie | 01 | GPU, **pesante** | **per-modello** (parallelo) |
| 03 | `03_train_head.sh` | D: head | 02 | GPU, leggero | 1 job |
| 04 | `04_eval_frozen_full.sh` | eval raw+whiten FULL | 01 | leggero | 1 job |
| 05 | `05_eval_head_full.sh` | eval head FULL | 03 | leggero | 1 job |
| 06 | `06_eval_frozen_partial.sh` | eval raw+whiten PARTIAL | 01 | **pesante** | **per-modello** |
| 07 | `07_eval_head_partial.sh` | eval head PARTIAL | 03 | **pesante** | **per-modello** |

Ogni script: **senza argomenti** processa tutti i modelli (in sequenza);
**con un argomento** ne processa uno solo, per parallelizzare; il **secondo
argomento** restringe la risoluzione (serve solo a `tipsv2`, vedi sotto):
```bash
sbatch scripts/vision/02_build_pairs.sh           # tutti (lungo)
sbatch scripts/vision/02_build_pairs.sh dinov2    # solo dinov2 (1 GPU)
sbatch scripts/vision/01_extract_raw.sh tipsv2 448  # solo tipsv2 a 448px
```
I pooling **validi per encoder** sono processati dentro lo script: `natural gem mean`
per tutti, ma `natural gem` per **I-JEPA** (che non ha CLS → `mean` coinciderebbe con
`natural`). La logica è centralizzata in `scripts/vision/_common.sh` (`poolings_for`).

**Risoluzione (`resolutions_for`, stesso file).** Sette encoder girano a una
sola risoluzione, quella del preset YAML: per loro la variante resta il solo
pooling (`embeddings/vision/dinov2/natural/`) e **i comandi non sono cambiati**.
Per `pecore` la 224 non è un default ma un **vincolo**: il checkpoint è a
risoluzione fissa (RoPE su griglia 14x14) e l'encoder solleva `ValueError` su
qualunque altro lato — passargli un `$2` diverso da `native` fa fallire il job.
Per `pespatial` la 224 è invece una **scelta**: il checkpoint è nativo a 512 e
timm ricampiona il pos-embed, ma la griglia lo usa a 224 per restare allineato
agli altri (la 512 è deliberatamente fuori da `resolutions_for`). Il vincolo
qui è solo «multiplo di 16».
`tipsv2` è l'unico con nativa 448 e gira **entrambe** le risoluzioni: la
risoluzione entra nel nome della variante (`tipsv2/natural448`, `tipsv2/natural224`),
altrimenti le due si sovrascriverebbero. A 448 il batch scende automaticamente a
32: senza xformers l'attenzione materializza `[B, heads, N, N]` e col 256 del
config si va in OOM.

## Ordine di esecuzione (cosa in parallelo, cosa aspettare)

1. **`01_extract_raw`** — PRIMA di tutto. (per parallelizzare: un job per
   modello; per `tipsv2`, un job per risoluzione)
   → *aspetta che finisca.*
2. Appena finito 01, in **parallelo** (dipendono solo da 01):
   - `02_build_pairs`  ← vedi nota **cache viste** qui sotto per l'ordine ottimale
   - `04_eval_frozen_full`  ← risultati frozen FULL (la tabella di confronto)
   - `06_eval_frozen_partial`  ← **per-modello**, frozen PARTIAL
3. Quando **02** è finito → **`03_train_head`** (aspetta 02).
4. Quando **03** è finito, in **parallelo**:
   - `05_eval_head_full`
   - `07_eval_head_partial`  ← **per-modello**

In breve: `01` → (`02` ∥ `04` ∥ `06`) → `03` → (`05` ∥ `07`).

> **Cache viste renderizzate (Stage 02).** Il rendering di una vista degradata
> dipende solo da `(pianta, vista, config masking)`, **non** dall'encoder: `02`
> cacha le viste in `embeddings/vision/_render_cache/<hash>/` e le **riusa** per
> tutti i (modello × pooling). Il rendering (flood-fill + dilatazione) è il collo
> di bottiglia → si paga **una volta sola** invece di 14.
> **Per sfruttarla al massimo: lancia PRIMA un solo job `02` (un modello qualsiasi)**
> per riempire la cache, **poi** gli altri in parallelo — saranno cache-hit
> (solo encoding GPU, veloce). Se li lanci tutti insieme a cache fredda, ognuno
> renderizza la sua fetta (comunque safe: scrittura atomica), ma perdi parte del
> risparmio. La cache è condivisa anche tra i pooling dello stesso modello.

## Combinazioni valutate (per ogni modello × pooling)

| Tag | Override | Richiede |
|---|---|---|
| `raw` | `head.enabled=false whitening.enabled=false` | 01 |
| `whiten` | `head.enabled=false whitening.enabled=true` | 01 |
| `head` | `head.enabled=true whitening.enabled=false` | 01+02+03 |
| `head+whiten` | `head.enabled=true whitening.enabled=true` | 01+02+03 |

Riduzione PCA a dimensione comune (per togliere il confondente della larghezza
embedding di RADIO/I-JEPA): aggiungi `whitening.dim=768` a un contributo con whitening.

## Visualizzazioni qualitative (opzionali)

```bash
python -m src.vision.utils.retrieval_visualization \
    model.name=dinov2 model.variant=natural model.kwargs.pooling=natural \
    head.enabled=true whitening.enabled=true partial.enabled=false
# output: results/visualizations/dinov2_natural_head+whiten_full/
```

## Note

- **Pooling per encoder** in `scripts/vision/_common.sh` (`poolings_for`): I-JEPA
  salta `mean` (= `natural`, no CLS) **automaticamente**. Per cambiare la griglia
  (es. solo `natural`, o niente `mean` ovunque) modifichi **solo** quella funzione.
- **gem/mean** di **SigLIP2/RADIO/I-JEPA** usano path di codice nuovi (`natural`
  passa da teste/uscite dedicate, mean/gem dai token grezzi): lo Stage 01 è anche
  il loro primo test reale su GPU → guarda quei log.
- **Split**: la head si allena su **train+valid** (`training.split_pool`), test
  tenuto fuori. La gallery di ricerca resta **intera**. Nessun leakage.
- **Iperparametri head** (`configs/vision_retrieval.yaml`, blocco `training`):
  `temperature` (τ) e `views_per_plan` (V) i due da ablare; anche `epochs`, `lr`,
  `batch_size`, `mask_fraction`, `augment`.
- **Strategia pratica**: gira prima i FULL (04, 05: veloci) per scegliere le
  combinazioni promettenti, poi i PARTIAL (06, 07: pesanti) solo su quelle.


# PER LANCIARE I JOB IN PARALLELO
for m in dinov2 dinov3 siglip2 radio ijepa tipsv2 pecore pespatial; do sbatch scripts/vision/04_eval_frozen_full.sh $m; done

---

# Ramo graph (`scripts/graph/`)

Job numerati nell'ordine di esecuzione. A differenza del vision, **nessun
download da Hugging Face** (le GNN si allenano da zero, dati locali) e nessun
log esterno (`wandb: false` nei YAML): i job girano senza rete. `03` e `04`
hanno la **constraint anti-Blackwell** (i nodi sm_120 crashano ogni forward
torch con il PyTorch di floorplan-env).

## Mappa delle dipendenze

```
(cache grafi: embeddings/graph/rplan/processed/graphs.pt — già presente,
 si rigenera da sola al primo uso se cancellata)
        │
        └─> 03 train_gnn ─> 04 eval_gnn   (04 include anche la baseline hist,
                                           che NON dipende da 03)
```

## Gli script

| # | Script | Stage | Dipende da | Config |
|---|---|---|---|---|
| 01 | `01_graph_builder.sh` | smoke builder | — | — |
| 02 | `02_graph_dataset.sh` | cache grafi | — | — |
| 03 | `03_train_gnn.sh` | training InfoNCE | cache | `configs/graph_models/<name>.yaml` |
| 04 | `04_eval_gnn.sh` | eval retrieval + baseline | 03 (baseline: no) | idem + `configs/graph_retrieval.yaml` |

- **01/02 di norma non servono**: la cache dei grafi esiste già; `RplanGraphDataset`
  la rigenera comunque da sé se manca (è `03`/`04` stesso a pagarla, ~3 min).
- **03**: per ogni encoder legge la ricetta YAML → flag argparse → training con
  early stopping sul valid; salva `encoder.pt` + `geom_stats.npz` +
  `training_summary.json` in `embeddings/graph/<encoder>/<variant>/`.
  I pesi migliori si scelgono sull'**nDCG di una sonda di retrieval sul valid**
  (`probe_*`/`select_criterion` nel YAML), **non** sulla val-loss InfoNCE: le due
  divergono (nel run del 17 lug la classifica per loss era l'inverso di quella per
  retrieval). `training_summary.json` riporta criterio, best score ed epoca →
  serve a confrontare le configurazioni nello sweep OFAT senza rileggere i log.
  Con `probe_every: 0` si torna al vecchio criterio (solo per confronto).
- **04**: valuta baseline `hist` (se `baseline_hist: true` nel config condiviso)
  + ogni encoder addestrato. Architettura dal **medesimo YAML del training**
  (il checkpoint si ricarica per costruzione), parametri di valutazione da
  `configs/graph_retrieval.yaml` (query dal test split, gallery intera, metriche
  per-asse). Salva anche `embeddings.npy` + `names.json` per la late fusion.
- **Secondo argomento = variante di ablation**, in entrambi gli script: senza,
  si usa `base` (comportamento storico). In `04` le varianti che hanno cambiato
  solo il training (τ, augmentation, criterio) non cambiano la rete → basta
  `--variant`; l'unica che cambia architettura è `noskip`, gestita dallo script.
  ⚠️ `04 <enc> ablation` valuta le varianti **presenti su disco** (cartelle con
  `encoder.pt`), quindi salta da sé quelle non allenate.

```bash
# sequenziale (tutti e tre gli encoder in un job)
sbatch scripts/graph/03_train_gnn.sh
sbatch scripts/graph/04_eval_gnn.sh          # dopo che 03 è finito

# parallelo (un job per encoder; i nomi sono i basename dei YAML)
for e in gcn gat graph_sage; do sbatch scripts/graph/03_train_gnn.sh $e; done
for e in gcn gat graph_sage; do sbatch scripts/graph/04_eval_gnn.sh $e; done
sbatch scripts/graph/04_eval_gnn.sh hist     # solo baseline training-free

# ablation (⚠️ un encoder per job: 03 fa 10 training di fila, 04 fino a 10 eval)
sbatch scripts/graph/03_train_gnn.sh gcn ablation   # allena tutte le varianti
sbatch scripts/graph/04_eval_gnn.sh gcn tau02       # valuta UNA variante (la vincente)
sbatch scripts/graph/04_eval_gnn.sh gcn ablation    # valuta tutte quelle allenate
```

⚠️ **Le ablation si confrontano sulla sonda (valid)**, cioè sul `best_score` dei
`training_summary.json` che `03` riepiloga a fine job: `04` misura sul **test**,
che va toccato solo per la variante vincente (o alla fine, per il report).

---

# Protocollo di misura (`scripts/evaluation/`)

Script del **core condiviso**: non appartengono a nessuno dei due rami, servono a
rendere leggibili e confrontabili i numeri di entrambi. Nessuna dipendenza fra
loro; `02→03` è l'unica sequenza obbligata.

```bash
# 01 — il floor: quanto prende chi NON guarda la query (CPU, niente GPU)
sbatch scripts/evaluation/01_random_floor.sh            # 4 run sul TEST: {vision,graph} x {casuale,costante}
sbatch scripts/evaluation/01_random_floor.sh all valid  # le stesse 4 sul VALID (dove si SCEGLIE)
# ⚠️ sul login node muore: ulimit -t = 600 s di CPU. Output: results/random_floor/
#   <null>_<ramo>_<split>_seed0.npz = per-query di UN seed | ..._mean5.txt = media dei 5.
#   Denominatore delle percentuali = null `random` (deciso, status.md §20).

# 02+03 — vision: si sceglie sul VALID, il test si legge UNA volta sola
sbatch scripts/evaluation/02_perquery_vision_valid.sh          # griglia 104 (56 pre-registrate + tipsv2 + pecore + pespatial) -> RISCRIVE
sbatch scripts/evaluation/02_perquery_vision_valid.sh "" frozen  # solo raw+whiten (52)
sbatch scripts/evaluation/02_perquery_vision_valid.sh pespatial frozen  # un encoder solo: 6 run, niente riscritture
# ⚠️ `frozen` e' obbligatorio per i 3 encoder nuovi: non hanno head.pt, il gruppo `head` fallirebbe.
sbatch scripts/evaluation/03_perquery_vision_test.sh dinov3 gem whiten

# 04 — graph: le varianti sono gia' scelte sul valid dalla probe -> bastano i per-query
sbatch scripts/evaluation/04_perquery_graph.sh          # hist + 3 vincenti + 3 base
```

### Curva di degrado sotto masking — applica il criterio A.5 (`status.md §23`)

Dal 24 ago «migliore» = **piu' robusto**: AUC della curva self-recovery MRR su
f in {0.25, 0.5, 0.75}. Questi job la producono. ⚠️ Da oggi 06/07 valutano sul
**valid** per default (`EVAL_SPLIT`); per il test serve `EVAL_SPLIT=test`.

```bash
# FROZEN (raw + whiten) — un modello per job: il partial e' molto piu' pesante del full
sbatch scripts/vision/06_eval_frozen_partial.sh dinov2      # idem dinov3 siglip2 radio ijepa pecore pespatial
sbatch scripts/vision/06_eval_frozen_partial.sh tipsv2 448  # $2 = risoluzione (tipsv2 ne ha due)
sbatch scripts/vision/06_eval_frozen_partial.sh tipsv2 224

# HEAD (head + head+whiten) — SOLO i 5 con head.pt: dinov2 dinov3 siglip2 radio ijepa
sbatch scripts/vision/07_eval_head_partial.sh dinov3
# ⚠️ tipsv2/pecore/pespatial NON hanno head.pt -> il gruppo head fallirebbe.
```

⚠️ 06/07 **non** salvano i valori per query: stampano la MRR nel log e basta.
Per il **delta appaiato** di § 23 serve `05` qui sotto.

### 05 — partial per-query sul VALID: applica il criterio A.5 per intero

✅ **Validato il 24 ago** con `... dinov3 frozen`: 24 `.npz` attesi e prodotti,
`self_rr [2000]` presente, gallery sha1 `22d91dcff041` = quella delle run full
sul valid (quindi appaiabile). Primo numero: whiten vs raw a f=0.75, MRR
0.2500 vs 0.0638, **+0.186** CI [+0.172, +0.202] (`status.md §23.4`).
**Si lanciano tutti in parallelo**: la eval e' in sola lettura su `embeddings/`
(`load_pipeline` non chiama mai `_save`) e i nomi dei `.npz` sono unici per
(encoder, pooling, risoluzione, trasformazione, frazione) — `variant_for` separa
`gem448` da `gem224`. ⚠️ La render-cache di `experiments.md:110-113` e' di
`projection_pairs.py` (training della head), **non** del partial: niente da
«scaldare».

```bash
# FROZEN (raw + whiten) — tutti e 8, un job per encoder. $2 = gruppo, $3 = strategie.
for M in dinov2 dinov3 siglip2 radio ijepa pecore pespatial; do
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh "$M" frozen
done
RES=448 sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh tipsv2 frozen
RES=224 sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh tipsv2 frozen

# HEAD (head + head+whiten) — SOLO i 5 con head.pt
for M in dinov2 dinov3 siglip2 radio ijepa; do
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh "$M" head
done
```

Default `$3 = random`: `semantic` e `topology` sono **disattivate**, perche' il
criterio § 23 usa solo la curva a frazione crescente e le altre due triplicano il
costo. Per averle tutte: `... <enc> all all`.

Volume (simulato sui loop, non stimato): **frozen 52 valutazioni -> 208 `.npz`**;
**head 28 -> 112**. Un file per (config x frazione), nomi
`vision_<enc>_<pool>_<trasf>_partial-random-f{0.0,0.25,0.5,0.75}_valid.npz`.

Poi, su CPU, il confronto appaiato **allo stesso livello di masking**:

```bash
D=results/perquery/vision_partial_valid
python -m src.evaluation.significance --metric self_rr \
  --a $D/vision_dinov3_natural_head_partial-random-f0.75_valid.npz \
  --b $D/vision_ijepa_gem_whiten_partial-random-f0.75_valid.npz
```

⚠️ `f=0.0` **non entra** nell'AUC di § 23: li' tutti fanno MRR 0.970, tetto dei
**dati** (duplicati RPLAN), non dei modelli. Confrontare due frazioni diverse e'
bloccato da `check_compatible` (`partial_label` e' in `STRICT_META_KEYS`):
misurerebbe il livello di masking, non la robustezza.

L'AUC di § 23 (dal 10 set) la calcola `robustness_auc` (CPU, secondi):

```bash
python -m src.evaluation.robustness_auc rank --dir results/perquery/vision_partial_valid_B --top 10
python -m src.evaluation.robustness_auc compare \
  --a results/perquery/vision_partial_valid_B/vision_dinov3_natural_head \
  --b results/perquery/graph_partial_valid/graph_gcn_tau02
```

### Protocollo B (dal 10 set) — il metro dei numeri finali

Definizione e trappole: `.claude/shared/experiments.md § Protocollo B`. La
gallery condivisa si genera **una volta** (fatto il 10 set, `status.md §29`) e si
punta in **entrambi** i YAML (`gallery_names`, gia' impostato):

```bash
python -m src.evaluation.gallery_join \
    --vision embeddings/vision/<modello>/<pooling>/image_paths.json \
    --graph  embeddings/graph/<run>/names.json \
    --out    results/shared_gallery.json
```

```bash
# VISION partial per-query -> cartella separata (i nomi dei .npz non dipendono dal protocollo)
PERQUERY_DIR=results/perquery/vision_partial_valid_B \
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 frozen
# head B.6 (checkpoint alternativo), un solo pooling:
PERQUERY_DIR=results/perquery/vision_partial_valid_B POOLS=natural EXTRA="head.file=head_probe_conv.pt" \
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 head

# GRAPH partial per-query (C.0) — split e gallery dal YAML
sbatch scripts/evaluation/06_perquery_graph_partial_valid.sh            # hist + 6 varianti
sbatch scripts/evaluation/06_perquery_graph_partial_valid.sh gcn tau02  # una coppia

# HEAD B.6 — training con selezione sulla probe partial (solo i 5 encoder con pairs.npz)
sbatch scripts/vision/09_train_head_probe.sh dinov3
# regola adottata (status.md § 30.5): max 5000 epoche, patience 100 sulla probe
EPOCHS=5000 PATIENCE=100 HEAD_FILE=head_probe_conv.pt REUSE_PROBE=1 \
  sbatch scripts/vision/09_train_head_probe.sh dinov3 natural
```

### Crop e patch sul vision (11 set) — pre-registrazione in `status.md §34`

`$3` di `05`: `random | all | crop | patch | damage` (damage = stanze random + crop +
patch); `TRANSFORMS` filtra dentro il gruppo. Un job = una config × 10 run.

```bash
D=results/perquery/vision_damage_valid_B
# ONDATA 1 — la config congelata e la sua baseline frozen (H1)
PERQUERY_DIR=$D POOLS=natural TRANSFORMS=head EXTRA="head.file=head_probe_conv.pt" \
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 head damage
PERQUERY_DIR=$D POOLS=natural TRANSFORMS=whiten \
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 frozen damage
# CONTROLLO (CPU) prima dell'ondata 2: i file random di D devono essere identici a
# quelli in results/perquery/vision_partial_valid_B (self_rr, ndcg, ret_rows)

# ONDATA 2 — miglior frozen per encoder (status.md §33)
for S in "dinov2 natural" "dinov3 mean" "siglip2 mean" "radio natural" "ijepa natural" "pecore mean" "pespatial gem"; do
  set -- $S
  PERQUERY_DIR=$D POOLS=$2 TRANSFORMS=whiten sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh $1 frozen damage
done
PERQUERY_DIR=$D POOLS=natural TRANSFORMS=whiten RES=448 \
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh tipsv2 frozen damage

# poi, su CPU
python -m src.evaluation.robustness_auc rank --dir $D --strategy crop
python -m src.evaluation.robustness_auc rank --dir $D --strategy patch
python -m src.evaluation.damage_curves --prefix $D/vision_dinov3_natural_head-probe-conv --out <csv>
```

### `nowalls` — stanza tolta DAVVERO (11 set, `status.md §36`)

Il masking a stanze storico lascia i muri interni (§35): `nowalls_random` toglie
le **stesse** stanze ma cancella anche i muri. Non c'e' un gruppo `$3` dedicato:
si accende via `EXTRA`, cosi' i comandi esistenti restano identici. 3 file in piu'
per config (`partial-nowalls-random-f0.25/0.5/0.75`, niente f=0.0).
⚠️ **Stessa cartella** dei file `random`, o il delta appaiato non e' calcolabile.

```bash
D=results/perquery/vision_damage_valid_B
NW="partial.strategies.nowalls_random.enabled=true"

# CASO 1 — l'ondata crop+patch NON e' ancora partita: un job per config, 13 run
#   (random 4 + crop 3 + patch 3 + nowalls 3). Sostituisce i due comandi dell'ondata 1.
PERQUERY_DIR=$D POOLS=natural TRANSFORMS=head \
  EXTRA="head.file=head_probe_conv.pt $NW" \
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 head damage
PERQUERY_DIR=$D POOLS=natural TRANSFORMS=whiten EXTRA="$NW" \
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 frozen damage

# CASO 2 — crop+patch gia' lanciati/finiti: solo le 3 run nowalls ($3=random spegne
#   gia' semantic/topology/crop/patch, EXTRA spegne anche random).
NWONLY="partial.strategies.random.enabled=false $NW"
PERQUERY_DIR=$D POOLS=natural TRANSFORMS=head EXTRA="head.file=head_probe_conv.pt $NWONLY" \
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 head random
PERQUERY_DIR=$D POOLS=natural TRANSFORMS=whiten EXTRA="$NWONLY" \
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh dinov3 frozen random

# ANALISI (CPU). Il numero che risponde alla domanda e' il delta appaiato:
#   stesse query, stesse stanze tolte, unica differenza i muri rimasti.
H=$D/vision_dinov3_natural_head-probe-conv
python -m src.evaluation.damage_curves --prefix $H --delta random nowalls-random
python -m src.evaluation.robustness_auc rank --dir $D --strategy nowalls-random
python -m src.evaluation.robustness_auc compare --strategy nowalls-random \
  --a $H --b $D/vision_dinov3_natural_whiten-train
```

⚠️ `significance` **rifiuta** il confronto `random` vs `nowalls-random` (i due file
hanno `partial_label` diverso): e' voluto: il delta fra strategie si legge con
`damage_curves --delta`, che appaia per nome query e riporta il CI bootstrap.

### Ondata 2 onesta (14 set) — griglia frozen COMPLETA, metro `--robust` (status.md §38)

Decisione utente 14 set: tutte le 52 config frozen, nessuna estrapolazione da dinov3.
51 valutazioni nuove (dinov3/natural/whiten e' gia' nell'ondata 1) in **29 job**; 9 run per
valutazione (3 danni x f=0.25/0.5/0.75), ~70 min l'una su dinov3@224. Verificato con dry-run a
python sostituito (29 job con exit 0, flag, cartella, batch 32 a 448, nessun doppione).

```bash
D=results/perquery/vision_damage_valid_B
HONEST="partial.strategies.random.enabled=false partial.strategies.nowalls_random.enabled=true"
S=scripts/evaluation/05_perquery_vision_partial_valid.sh

# ---- dinov2 (3 job) ----
PERQUERY_DIR=$D POOLS=natural EXTRA="$HONEST" sbatch $S dinov2 frozen damage   # natural, con e senza whitening
PERQUERY_DIR=$D POOLS=gem EXTRA="$HONEST" sbatch $S dinov2 frozen damage   # gem, con e senza whitening
PERQUERY_DIR=$D POOLS=mean EXTRA="$HONEST" sbatch $S dinov2 frozen damage   # mean, con e senza whitening

# ---- dinov3 (3 job) ----
PERQUERY_DIR=$D POOLS=natural TRANSFORMS=raw EXTRA="$HONEST" sbatch $S dinov3 frozen damage   # natural senza whitening (con whitening: già fatto)
PERQUERY_DIR=$D POOLS=gem EXTRA="$HONEST" sbatch $S dinov3 frozen damage   # gem, con e senza whitening
PERQUERY_DIR=$D POOLS=mean EXTRA="$HONEST" sbatch $S dinov3 frozen damage   # mean, con e senza whitening

# ---- siglip2 (3 job) ----
PERQUERY_DIR=$D POOLS=natural EXTRA="$HONEST" sbatch $S siglip2 frozen damage   # natural, con e senza whitening
PERQUERY_DIR=$D POOLS=gem EXTRA="$HONEST" sbatch $S siglip2 frozen damage   # gem, con e senza whitening
PERQUERY_DIR=$D POOLS=mean EXTRA="$HONEST" sbatch $S siglip2 frozen damage   # mean, con e senza whitening

# ---- radio (3 job) ----
PERQUERY_DIR=$D POOLS=natural EXTRA="$HONEST" sbatch $S radio frozen damage   # natural, con e senza whitening
PERQUERY_DIR=$D POOLS=gem EXTRA="$HONEST" sbatch $S radio frozen damage   # gem, con e senza whitening
PERQUERY_DIR=$D POOLS=mean EXTRA="$HONEST" sbatch $S radio frozen damage   # mean, con e senza whitening

# ---- ijepa (2 job) ----
PERQUERY_DIR=$D POOLS=natural EXTRA="$HONEST" sbatch $S ijepa frozen damage   # natural, con e senza whitening
PERQUERY_DIR=$D POOLS=gem EXTRA="$HONEST" sbatch $S ijepa frozen damage   # gem, con e senza whitening

# ---- tipsv2 (9 job) ----
PERQUERY_DIR=$D POOLS=natural RES=224 EXTRA="$HONEST" sbatch $S tipsv2 frozen damage   # natural a 224, con e senza whitening
PERQUERY_DIR=$D POOLS=gem RES=224 EXTRA="$HONEST" sbatch $S tipsv2 frozen damage   # gem a 224, con e senza whitening
PERQUERY_DIR=$D POOLS=mean RES=224 EXTRA="$HONEST" sbatch $S tipsv2 frozen damage   # mean a 224, con e senza whitening
PERQUERY_DIR=$D POOLS=natural RES=448 TRANSFORMS=raw EXTRA="$HONEST" sbatch $S tipsv2 frozen damage   # natural a 448, senza whitening
PERQUERY_DIR=$D POOLS=natural RES=448 TRANSFORMS=whiten EXTRA="$HONEST" sbatch $S tipsv2 frozen damage   # natural a 448, con whitening
PERQUERY_DIR=$D POOLS=gem RES=448 TRANSFORMS=raw EXTRA="$HONEST" sbatch $S tipsv2 frozen damage   # gem a 448, senza whitening
PERQUERY_DIR=$D POOLS=gem RES=448 TRANSFORMS=whiten EXTRA="$HONEST" sbatch $S tipsv2 frozen damage   # gem a 448, con whitening
PERQUERY_DIR=$D POOLS=mean RES=448 TRANSFORMS=raw EXTRA="$HONEST" sbatch $S tipsv2 frozen damage   # mean a 448, senza whitening
PERQUERY_DIR=$D POOLS=mean RES=448 TRANSFORMS=whiten EXTRA="$HONEST" sbatch $S tipsv2 frozen damage   # mean a 448, con whitening

# ---- pecore (3 job) ----
PERQUERY_DIR=$D POOLS=natural EXTRA="$HONEST" sbatch $S pecore frozen damage   # natural, con e senza whitening
PERQUERY_DIR=$D POOLS=gem EXTRA="$HONEST" sbatch $S pecore frozen damage   # gem, con e senza whitening
PERQUERY_DIR=$D POOLS=mean EXTRA="$HONEST" sbatch $S pecore frozen damage   # mean, con e senza whitening

# ---- pespatial (3 job) ----
PERQUERY_DIR=$D POOLS=natural EXTRA="$HONEST" sbatch $S pespatial frozen damage   # natural, con e senza whitening
PERQUERY_DIR=$D POOLS=gem EXTRA="$HONEST" sbatch $S pespatial frozen damage   # gem, con e senza whitening
PERQUERY_DIR=$D POOLS=mean EXTRA="$HONEST" sbatch $S pespatial frozen damage   # mean, con e senza whitening

# LETTURA (CPU), a job finiti
python -m src.evaluation.robustness_auc rank --dir $D --robust --compare-top 5
ls $D | grep -c "nowalls-random-f0.75"      # atteso: 53 (52 frozen + la head dell'ondata 1)
```

### Visualizzazioni dei danni onesti (14 set) — pannelli query + top-5

Da 14 set `retrieval_visualization` passa da `damaged_query` (stessa funzione della valutazione):
disegna anche crop, patch e `nowalls`. Output: `results/visualizations/<modello>_<pool>_<contributo>_{full,partial}/`
con una sottocartella per danno e livello. `--time` sopra l'ora del default: il partial ha 9 run.

```bash
VIZ="partial.strategies.random.enabled=false partial.strategies.semantic.enabled=false partial.strategies.topology.enabled=false partial.strategies.crop.enabled=true partial.strategies.patch.enabled=true partial.strategies.nowalls_random.enabled=true"
POOL=gem CONTRIB=raw    MODE=both EXTRA="$VIZ" sbatch --time=02:00:00 scripts/vision/08_visualize.sh dinov3
POOL=gem CONTRIB=whiten MODE=both EXTRA="$VIZ" sbatch --time=02:00:00 scripts/vision/08_visualize.sh dinov3
```

### Graph (14 set) — controllo `semantic` + «bordo aperto» (`status.md §40-§41`)

Il ramo graph **non** aspetta l'ondata 2. Fase 1 = solo valutazione sui checkpoint esistenti;
fase 2 = variante `asymlost` (marcatore «vicini persi», `--lost-marker`, in_dim 20) + replica
`asymrep` di `asym` (stima del rumore fra training: init e shuffle non seedati).

```bash
# FASE 1 — il danno semantico sostituisce il random (GRAPH_EXTRA_FLAGS e' appeso per ultimo: vince)
GRAPH_EXTRA_FLAGS="--partial-strategies semantic" sbatch scripts/evaluation/06_perquery_graph_partial_valid.sh gat asym
GRAPH_EXTRA_FLAGS="--partial-strategies semantic" sbatch scripts/evaluation/06_perquery_graph_partial_valid.sh gat base

# FASE 2 — asymlost: training -> partial valid -> full valid, IN CATENA
# (06 e 04 riscrivono embeddings.npy/names.json della variante: mai in parallelo fra loro)
J1=$(sbatch --parsable scripts/graph/03_train_gnn.sh gat asymlost)
J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/evaluation/06_perquery_graph_partial_valid.sh gat asymlost)
PERQUERY_OUT=results/perquery/graph_full_valid sbatch --dependency=afterok:$J2 scripts/evaluation/04_perquery_graph.sh gat asymlost
# replica: training -> partial valid
R1=$(sbatch --parsable scripts/graph/03_train_gnn.sh gat asymrep)
sbatch --dependency=afterok:$R1 scripts/evaluation/06_perquery_graph_partial_valid.sh gat asymrep
echo "$J1 $J2 $R1"      # da annotare nel TODO
```

⚠️ `afterok` **non** protegge: `03_train_gnn.sh` esce 0 anche se il training fallisce, e `04_eval_gnn.sh`
salta un checkpoint mancante («checkpoint mancante — salto») ed esce 0. Un training fallito produce solo
valutazioni vuote, non numeri sbagliati: nel log di J1/R1 cercare `salvato …/encoder.pt`.

```bash
# LETTURA (CPU), a job finiti
python -m src.evaluation.robustness_auc compare \
  --a results/perquery/graph_partial_valid/graph_gat_asymlost --b results/perquery/graph_partial_valid/graph_gat_asym
python -m src.evaluation.robustness_auc compare \
  --a results/perquery/graph_partial_valid/graph_gat_asymrep  --b results/perquery/graph_partial_valid/graph_gat_asym
python -m src.evaluation.significance --k 10 \
  --a results/perquery/graph_full_valid/graph_gat_asymlost_full_valid.npz \
  --b results/perquery/graph_full_valid/graph_gat_asym_full_valid.npz
```

Se `asymlost` è adottata: controllo `semantic` anche su `gat asymlost`, e al test `07 ... gat asymlost`
(il flag arriva da `04_eval_gnn.sh`). Se non passa: `gat/asym` congelata, nessuna altra forma del marcatore.
✅ **Esito 14 set (`status.md §42`)**: `asymlost` non adottata → `gat/asym` congelata; niente da rilanciare.

### Graph (15 set) — checkpoint scelto sulla ROBUSTEZZA (`status.md §45`, regole scritte prima)

`asymrob` / `asymlostrob` / `asymrobrep` = `asym` / `asymlost` / replica con `--selection-probe partial`
(AUC self-recovery su 2000 query valid disgiunte, gallery intera, cap 300, patience 0). Ogni training salva
anche `<v>_selfull` = checkpoint della regola vecchia **sulla stessa traiettoria**. Tempo stimato ~1,5-3 h a
training (non misurato): nel log della prima epoca guardare `AUC ... (Ns)` e `overlap: 0`.

```bash
squeue -u gangelillis,edimaria,ggermini          # prima: la quota GPU dell'account e' condivisa
# --time=16:00:00 sovrascrive le 08:00:00 dello script (MaxTime partizione = 1 giorno): il training
# salva solo alla fine, un timeout perderebbe anche `_selfull`
T1=$(sbatch --parsable --time=16:00:00 scripts/graph/03_train_gnn.sh gat asymrob)
T2=$(sbatch --parsable --time=16:00:00 scripts/graph/03_train_gnn.sh gat asymlostrob)
T3=$(sbatch --parsable --time=16:00:00 scripts/graph/03_train_gnn.sh gat asymrobrep)
# 06 e 04 della STESSA variante in catena (riscrivono embeddings.npy/names.json della variante)
for V in asymrob asymrob_selfull; do
  P=$(sbatch --parsable --dependency=afterok:$T1 scripts/evaluation/06_perquery_graph_partial_valid.sh gat $V)
  PERQUERY_OUT=results/perquery/graph_full_valid sbatch --dependency=afterok:$P scripts/evaluation/04_perquery_graph.sh gat $V
done
for V in asymlostrob asymlostrob_selfull; do
  P=$(sbatch --parsable --dependency=afterok:$T2 scripts/evaluation/06_perquery_graph_partial_valid.sh gat $V)
  PERQUERY_OUT=results/perquery/graph_full_valid sbatch --dependency=afterok:$P scripts/evaluation/04_perquery_graph.sh gat $V
done
for V in asymrobrep asymrobrep_selfull; do
  sbatch --dependency=afterok:$T3 scripts/evaluation/06_perquery_graph_partial_valid.sh gat $V
done
echo "$T1 $T2 $T3"
```

⚠️ `afterok` non protegge (03 esce 0 anche se il training fallisce): nei log dei training cercare
`salvato …/encoder.pt` **e** la cartella `…_selfull/encoder.pt`.
Tempo reale (dopo ~20 min): `grep -c "^\[train\] epoch" logs/gp_03_train_gnn_$T1.log` e `sacct -j $T1 -o Elapsed`
→ tempo/epoca × 300 deve stare sotto le 16 h (altrimenti `scancel` e dirlo, non rilanciare a occhio).

```bash
# LETTURA (CPU), a job finiti — regole di status.md §45
D=results/perquery/graph_partial_valid/graph_gat_ ; F=results/perquery/graph_full_valid/graph_gat_
for AB in "asymrob asymrob_selfull" "asymlostrob asymlostrob_selfull" "asymlostrob asymrob" \
          "asymrobrep asymrob" "asymrobrep_selfull asymrobrep" "asymrob asym" "asymrob_selfull asym" \
          "asymrobrep_selfull asym"; do
  set -- $AB; python -m src.evaluation.robustness_auc compare --a $D$1 --b $D$2
done
python -m src.evaluation.significance --k 10 --a ${F}asymrob_full_valid.npz --b ${F}asymrob_selfull_full_valid.npz
python -m src.evaluation.significance --k 10 --a ${F}asymlostrob_full_valid.npz --b ${F}asymrob_full_valid.npz
# controllo f=0.0 e per livello: significance --metric self_rr sui file ${D}<v>_partial-random-f*_valid.npz
```

### Costo sulla pianta intera del vincitore vision (15 set, status.md §43)

```bash
PERQUERY_DIR=results/perquery/vision_full_valid_B POOLS=gem \
  sbatch scripts/evaluation/02_perquery_vision_valid.sh pespatial frozen   # gem raw + gem whiten, pianta intera
```

### Head nuova su `pespatial/gem` con coppie `nowalls` (15 set, pre-registrata in status.md §46)

✅ **Eseguita e chiusa il 16 set** (job 107111 · 107463 · 107476-77 · 107520): non adottata, `status.md §48`.
I comandi restano come ricetta.

Stesso danno (`training.damage=nowalls_random`) per coppie **e** probe: il training rifiuta i due
diversi. ⚠️ `02` esce con 0 anche se python fallisce e `09` **salta in silenzio** se manca
`pairs.npz`: il job 2 si lancia **dopo** aver controllato il job 1, non con `afterok`.

```bash
# 1. coppie (il più lungo: render di tutte le piante train+valid × 3 viste, cache nuova)
POOLS=gem EXTRA="training.damage=nowalls_random" \
  sbatch scripts/vision/02_build_pairs.sh pespatial
#    controllo: nel log «[pairs] salvato ... | danno nowalls_random»
ls -l embeddings/vision/pespatial/gem/pairs.npz

# 2. probe nowalls + training fino a convergenza. wandb: dal 16 set `enabled: false` nel YAML
#    (il job 107225 era fallito a `wandb.init` senza chiave). REUSE_PROBE=1 solo se
#    probe_partial.npz e' gia' `nowalls_random` (il training lo controlla).
EPOCHS=5000 PATIENCE=100 HEAD_FILE=head_nowalls_conv.pt REUSE_PROBE=1 \
  EXTRA="training.damage=nowalls_random" \
  sbatch scripts/vision/09_train_head_probe.sh pespatial gem
#    controllo: nel log «[train] danno di coppie e probe: nowalls_random» e «fallimenti: 0»
ls -l embeddings/vision/pespatial/gem/head_nowalls_conv.pt

# 3. valutazione: stanze tolte + crop + patch, due candidati (9 file ciascuno)
D=results/perquery/vision_damage_valid_B
HONEST="partial.strategies.random.enabled=false partial.strategies.nowalls_random.enabled=true"
PERQUERY_DIR=$D POOLS=gem TRANSFORMS=head EXTRA="head.file=head_nowalls_conv.pt $HONEST" \
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh pespatial head damage
PERQUERY_DIR=$D POOLS=gem TRANSFORMS=head+whiten EXTRA="head.file=head_nowalls_conv.pt $HONEST" \
  sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh pespatial head damage
#    controllo: atteso 18 (9 head + 9 head+whiten-train)
ls $D | grep -c "vision_pespatial_gem_head-nowalls-conv"

# 4. pianta intera (costo): head e head+whiten-train
PERQUERY_DIR=results/perquery/vision_full_valid_B POOLS=gem EXTRA="head.file=head_nowalls_conv.pt" \
  sbatch scripts/evaluation/02_perquery_vision_valid.sh pespatial head
```

Analisi (CPU, regole di §46): `robustness_auc rank --dir $D --robust` (scelta fra i due candidati e
R contro 0.5245) · `robustness_auc compare --strategy crop` e `--strategy patch` contro
`$D/vision_pespatial_gem_whiten-train` (H3) · `significance` sul full.

### Late fusion vision + graph (16 set, pre-registrata in status.md §49)

Config fisse negli script (niente argomenti di modello, niente env di cartella): vision
`pespatial/gem/whiten` frozen, graph `gat/asymrob`, stanze tolte f = 0.0 0.25 0.5 0.75
(vision `nowalls_random`, graph `random`: stesse stanze). Cartelle **fisse e nuove**, i
per-query storici non si toccano: `results/perquery/fusion_branches_<split>/` (per-query dei due
job), `results/queryvec/<split>/` (vettori delle query, `qvec/1`), `results/perquery/fusion_<split>/`
(file fusi). Gli script rifiutano di riscrivere file già presenti senza `FORCE=1`.
⚠️ Fra `09` e `10` **nessun'altra valutazione di gat/asymrob**: riscrive `embeddings.npy` e la
fusione si ferma (sha1). Dry-run a python sostituito dei tre script: ok (16 set).

```bash
# VALID — 1. vettori delle query (GPU, i due job in parallelo)
sbatch scripts/evaluation/08_queryvec_vision.sh valid
sbatch scripts/evaluation/09_queryvec_graph.sh valid
#    controllo: 4 + 4 file per ramo, i log finiscono con «file mancanti: 0»
ls results/queryvec/valid | wc -l                      # atteso: 8

# 2. fusione, griglia alpha 0..1 passo 0.1 (CPU, niente GPU; 11 alpha x 5 passate)
sbatch scripts/evaluation/10_late_fusion.sh valid

# 3. PRIMA i controlli (CPU, login node ok): se uno e' FAIL ci si ferma e si diagnostica
python -m src.evaluation.fusion_select check --split valid --fusion-dir results/perquery/fusion_valid \
    --vision-qvec results/queryvec/valid/vision_pespatial_gem_whiten-train \
    --graph-qvec  results/queryvec/valid/graph_gat_asymrob \
    --vision-perquery results/perquery/fusion_branches_valid/vision_pespatial_gem_whiten-train \
    --graph-perquery  results/perquery/fusion_branches_valid/graph_gat_asymrob \
    --vision-hist results/perquery/vision_damage_valid_B/vision_pespatial_gem_whiten-train \
    --graph-hist  results/perquery/graph_partial_valid/graph_gat_asymrob \
    --vision-full-hist results/perquery/vision_full_valid_B/vision_pespatial_gem_whiten-train_full_valid.npz \
    --graph-full-hist  results/perquery/graph_full_valid/graph_gat_asymrob_full_valid.npz

# 4. solo se l'esito e' PASS: alpha*, verdetto, oracolo, costo sulla pianta intera
python -m src.evaluation.fusion_select select --split valid --fusion-dir results/perquery/fusion_valid
#    -> results/fusion/select_valid.json

# TEST — solo nel gruppo di job pre-registrato, alpha* fissato dal valid
sbatch scripts/evaluation/08_queryvec_vision.sh test
sbatch scripts/evaluation/09_queryvec_graph.sh test
ALPHAS_FROM=results/fusion/select_valid.json sbatch scripts/evaluation/10_late_fusion.sh test
A=$(python -c "import json; print(json.load(open('results/fusion/select_valid.json'))['alpha_star'])")
python -m src.evaluation.fusion_select check --split test --fusion-dir results/perquery/fusion_test \
    --alphas 0 $A 1 \
    --vision-qvec results/queryvec/test/vision_pespatial_gem_whiten-train \
    --graph-qvec  results/queryvec/test/graph_gat_asymrob \
    --vision-perquery results/perquery/fusion_branches_test/vision_pespatial_gem_whiten-train \
    --graph-perquery  results/perquery/fusion_branches_test/graph_gat_asymrob
python -m src.evaluation.fusion_select select --split test --fusion-dir results/perquery/fusion_test \
    --fixed-alpha-from results/fusion/select_valid.json
```

### Controllo effetto d'insieme (graph + graph) (17 set, pre-registrato in status.md §50)

Fonde due training identici del graph (`gat/asymrob` + la replica `gat/asymrobrep`) con lo stesso
metodo della fusione vera, per capire se il guadagno di vision + graph è complementarità o solo
«due modelli sbagliano in modo diverso». **Solo valid**, cartelle fisse (`fusion_branches_valid`,
`queryvec/valid`, `fusion_graphgraph_valid`); i vettori di asymrob sono quelli già scritti da `09`.
⚠️ Fra `11` e `12` nessun'altra valutazione di gat/asymrobrep (riscrive `embeddings.npy`: sha1).
Dry-run a python sostituito di 11 e 12, anche con env ostili: ok (17 set).

```bash
# 1. vettori delle query della replica (GPU)
sbatch scripts/evaluation/11_queryvec_graph_replica.sh
#    controllo: il log finisce con «file mancanti: 0»
# 2. fusione graph + graph, griglia beta 0..1 (CPU)
sbatch scripts/evaluation/12_late_fusion_graphgraph.sh
# 3. PRIMA i controlli (C5 qui ha solo il limite basso 0.965)
python -m src.evaluation.fusion_select check --pair graph-graph --split valid \
    --fusion-dir results/perquery/fusion_graphgraph_valid \
    --graph-qvec  results/queryvec/valid/graph_gat_asymrob \
    --graph2-qvec results/queryvec/valid/graph_gat_asymrobrep \
    --graph-perquery  results/perquery/fusion_branches_valid/graph_gat_asymrob \
    --graph2-perquery results/perquery/fusion_branches_valid/graph_gat_asymrobrep \
    --graph-hist  results/perquery/graph_partial_valid/graph_gat_asymrob \
    --graph2-hist results/perquery/graph_partial_valid/graph_gat_asymrobrep
# 4. solo se PASS: beta* (stessa regola di alpha) -> results/fusion/select_graphgraph_valid.json
python -m src.evaluation.fusion_select select --pair graph-graph --split valid \
    --fusion-dir results/perquery/fusion_graphgraph_valid
# 5. la decisione: guadagno della fusione vera vs guadagno del controllo
python -m src.evaluation.fusion_select complementarity --split valid
#    (default: select_valid.json + fusion_valid, select_graphgraph_valid.json + fusion_graphgraph_valid)
#    -> results/fusion/complementarity_valid.json
```

### Controllo modelli diversi, stessa informazione (vision + vision) (17 set, pre-registrato in status.md §51)

Fonde due encoder vision diversi sulla stessa immagine (`pespatial/gem/whiten` + `radio/natural/whiten`,
ciascuno col whitening del proprio job) con lo stesso metodo della fusione vera: se vision + graph
guadagna più di così, la differenza viene dall'informazione e non solo dal modello. **Solo valid**,
cartelle fisse (`fusion_branches_valid`, `queryvec/valid`, `fusion_visionvision_valid`); i vettori di
pespatial sono quelli già scritti da `08`. Dry-run a python sostituito di 13 e 14, anche con env ostili:
ok (17 set).

```bash
# 1. vettori delle query di radio (GPU; stesse chiavi della run radio dell'ondata 2)
sbatch scripts/evaluation/13_queryvec_vision_radio.sh
#    controllo: il log finisce con «file mancanti: 0»
# 2. fusione vision + vision, griglia gamma 0..1 (CPU)
sbatch scripts/evaluation/14_late_fusion_visionvision.sh
# 3. PRIMA i controlli (C5 solo limite basso; C4 e C3-full descrittivi: radio non ha il full storico)
python -m src.evaluation.fusion_select check --pair vision-vision --split valid \
    --fusion-dir results/perquery/fusion_visionvision_valid \
    --vision-qvec  results/queryvec/valid/vision_pespatial_gem_whiten-train \
    --vision2-qvec results/queryvec/valid/vision_radio_natural_whiten-train \
    --vision-perquery  results/perquery/fusion_branches_valid/vision_pespatial_gem_whiten-train \
    --vision2-perquery results/perquery/fusion_branches_valid/vision_radio_natural_whiten-train \
    --vision-hist  results/perquery/vision_damage_valid_B/vision_pespatial_gem_whiten-train \
    --vision2-hist results/perquery/vision_damage_valid_B/vision_radio_natural_whiten-train
# 4. solo se PASS: gamma* (stessa regola di alpha) -> results/fusion/select_visionvision_valid.json
python -m src.evaluation.fusion_select select --pair vision-vision --split valid \
    --fusion-dir results/perquery/fusion_visionvision_valid
# 5. la decisione a quattro esiti (G_C letto da results/fusion/complementarity_valid.json)
python -m src.evaluation.fusion_select complementarity --control-pair vision-vision --split valid
#    -> results/fusion/complementarity_visionvision_valid.json
```

### Test finale — pre-registrato in `status.md §52` (18 set), UNA VOLTA SOLA

Ordine vincolante: **`07 gat asymrob` prima di `09`** (entrambi riscrivono `embeddings/graph/gat/asymrob/`,
e la fusione verifica lo sha1); `10` solo quando 03/07/08/09 sono finiti. Output in cartelle nuove
(`vision_test_B/`, `graph_test_B/`, `fusion_branches_test/`, `queryvec/test/`, `fusion_test/`).

```bash
# 1. GRAPH per primo (config §47) — full + stanze tolte, poi la baseline training-free
sbatch scripts/evaluation/07_perquery_graph_test.sh gat asymrob        # full + partial
sbatch scripts/evaluation/07_perquery_graph_test.sh hist - both        # baseline, anche sotto danno (§52)

# 2. VISION (config §44/§48) — pianta intera + i tre danni del metro (§38)
sbatch scripts/evaluation/03_perquery_vision_test.sh pespatial gem whiten full
STRAT=damage EXTRA="partial.strategies.random.enabled=false partial.strategies.nowalls_random.enabled=true" \
  sbatch scripts/evaluation/03_perquery_vision_test.sh pespatial gem whiten partial

# 3. vettori delle query danneggiate (per la fusione) — il graph SOLO dopo il punto 1
sbatch scripts/evaluation/08_queryvec_vision.sh test
sbatch scripts/evaluation/09_queryvec_graph.sh test

# 4. fusione con alpha FISSO dal valid (CPU)
ALPHAS_FROM=results/fusion/select_valid.json sbatch scripts/evaluation/10_late_fusion.sh test

# 5. controlli (CPU) — se uno e' FAIL: stop e diagnosi prima di leggere i risultati
A=$(python -c "import json; print(json.load(open('results/fusion/select_valid.json'))['alpha_star'])")
python -m src.evaluation.fusion_select check --split test --fusion-dir results/perquery/fusion_test \
    --alphas 0 $A 1 \
    --vision-qvec results/queryvec/test/vision_pespatial_gem_whiten-train \
    --graph-qvec  results/queryvec/test/graph_gat_asymrob \
    --vision-perquery results/perquery/fusion_branches_test/vision_pespatial_gem_whiten-train \
    --graph-perquery  results/perquery/fusion_branches_test/graph_gat_asymrob

# 6. numeri finali (CPU)
python -m src.evaluation.fusion_select select --split test --fusion-dir results/perquery/fusion_test \
    --fixed-alpha-from results/fusion/select_valid.json
python -m src.evaluation.robustness_auc rank --split test --dir results/perquery/vision_test_B --robust
```

Confronto **vision↔graph sotto danno**: si legge dai file fusi, α=1 (= vision) e α=0 (= graph), stessa
etichetta e stesse query (§52); sul valid i due estremi riproducono i rami esattamente.
Pavimento: si usa `results/random_floor/random_graph_test_*` per **entrambi** i rami (il gemello vision ha la
gallery pre-B.3; le 2000 query coincidono), dichiarato in §52.

`04` non duplica il ponte YAML→flag: esporta `PERQUERY_OUT` e delega a
`scripts/graph/04_eval_gnn.sh`, che senza quella variabile si comporta
**esattamente** come prima.

Poi, su CPU e senza rilanciare nulla:

```bash
python -m src.evaluation.significance --a A.npz --b B.npz --k 10   # e' un guadagno o rumore?
python -m src.evaluation.geometry_variants --gallery <json> --perquery A.npz B.npz
```

### Figure del report (18 set) — CPU, nessun job

Il report è a **due colonne** (stile conferenza): le figure si disegnano già alla larghezza
finale (`src/figures/style.py`, 3.25″ una colonna / 6.875″ due) e non si riscalano in LaTeX.
Ogni figura scrive in `figures/` il PDF (quello che va nel report), il PNG e un
`*.sources.txt` con i file letti, il comando e i numeri disegnati.

```bash
# curva del danno sul TEST, quattro sistemi (baseline hist, vision, graph, fusione α=0.6)
python -m src.figures.damage_test                       # ~10 s
# AUC in funzione di α sul VALID, le tre coppie fuse (bande bootstrap: ~45 s)
python -m src.figures.alpha_valid
python -m src.figures.alpha_valid --no-ci               # veloce, senza bande
```

```bash
# teaser: una query danneggiata e i primi 5 risultati dei tre sistemi (valid, f=0.5)
python -m src.figures.teaser_valid                      # ~15 s, nessun job
python -m src.figures.teaser_valid --query 10340        # per fissare un'altra query
# i tre danni + l'artefatto dei muri (due pannelli, valid)
python -m src.figures.damage_kinds_valid                # ~20 s
# schema della pipeline (numeri letti dagli artefatti)
python -m src.figures.pipeline                          # istantaneo
# quante piante sono rilevanti per una query (composizione vs topologia)
python -m src.figures.classes_valid                     # istantaneo
# tutte le ablation per encoder (box + punti): la prima volta ~90 s, poi cache
python -m src.figures.ablation_valid
python -m src.figures.ablation_valid --refresh          # ricalcola figures/f_ablation_valid.data.json
```

```bash
# curva del danno sul TEST, media sulle 4 repliche (§57), banda ± 1 sd FRA repliche
python -m src.figures.damage_test_seeds --out-dir figures/italian
python -m src.figures.damage_test_seeds --lang en --out-dir figures/english
```

Etichette in inglese con `--lang en`; `--name`/`--out-dir` per scriverle altrove.
`alpha_valid` **si ferma** se una media ricalcolata dai per-query non coincide con quella
già nel json di `fusion_select`: la figura deve mostrare i numeri del report, non altri.
Il teaser non ricalcola niente: la classifica viene da `ret_rows` (che esclude la query) più
`self_rr` (che dice a che posto stava), e il danno è ridisegnato dalle stanze salvate nel `qvec`.
Senza `--query` sceglie da sé con una regola dichiarata, scritta nel `*.sources.txt`.

### Repliche multi-seed — protocollo `status.md §57`, ricetta per QUALSIASI seed nuovo

Una replica = un seed `S`: training graph `gat/asymrob_s<S>` (`--seed S`) + danno `partial.seed=S` su
entrambi i rami (vision: nowalls + crop + patch nello stesso job 08; graph: `--partial-seed S`, stesse
stanze). Stesse 2000 query (seed 42), stessa gallery. α* scelto sul **valid** della replica, il test lo usa
fisso. Senza il secondo argomento 08/09/10 fanno **esattamente** il comando storico. Output solo sotto
`results/{perquery,queryvec,fusion}/seeds/s<S>/` ed `embeddings/graph/gat/asymrob_s<S>{,_selfull}/`.

**Già fatte (1 ott, §57.1-§57.3)**: `100042 200042 300042` (+ la storica = replica 0). Non riusarle.

**Scegliere i seed nuovi**: solo cifre, mai già usati, distanti fra loro (e da 42, 100042, 200042,
300042) **almeno 67.405** = n. di piante: il danno usa l'rng `seed + qi` con `qi` fino a 67.405, quindi
seed vicini riuserebbero gli stessi stream. Proposta: `400042 500042 600042`.

**Prima di lanciare** (regola del progetto: si dichiara prima, non dopo): aggiungere in `status.md` una
riga sotto §57 con i seed nuovi e la data — protocollo, controlli e previsioni restano quelli di §57.

⚠️ Mentre una replica è in corso **non** lanciare `04_eval_gnn.sh gat ablation` né rivalutare `asymrob*`:
riscrive `embeddings.npy` della variante e la fusione si ferma (sha1).

```bash
cd /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID
SEEDS="400042 500042 600042"                     # <- i seed nuovi
squeue -u gangelillis,edimaria,ggermini          # la quota GPU dell'account è condivisa

# FASE 1 — GPU: training + valid (+ vision sul test, che non dipende dal graph). Repliche in parallelo.
for S in $SEEDS; do
  T=$(sbatch --parsable --time=16:00:00 scripts/graph/03_train_gnn.sh gat asymrob_s$S)
  VV=$(sbatch --parsable scripts/evaluation/08_queryvec_vision.sh valid $S)
  VT=$(sbatch --parsable scripts/evaluation/08_queryvec_vision.sh test $S)
  GV=$(sbatch --parsable --dependency=afterok:$T scripts/evaluation/09_queryvec_graph.sh valid $S)
  FV=$(sbatch --parsable --dependency=afterok:$VV:$GV scripts/evaluation/10_late_fusion.sh valid $S)
  echo "S=$S train=$T vision_valid=$VV vision_test=$VT graph_valid=$GV fusion_valid=$FV"
done
# -> annotare gli id in .claude/TODO.md (In attesa dell'utente)

# controllo a fase 1 finita: tutti COMPLETED + un encoder.pt per seed (03 esce 0 anche se fallisce)
sacct -u $USER -S today -o JobID,JobName%22,State,Elapsed | grep -v "\."
for S in $SEEDS; do grep -H "salvato .*asymrob_s$S/encoder.pt" logs/gp_03_train_gnn_*.log; done

# FASE 2 — CPU, login node (secondi): controlli, poi alpha* della replica
for S in $SEEDS; do
  bash scripts/evaluation/15_seed_fusion_select.sh check  valid $S && \
  bash scripts/evaluation/15_seed_fusion_select.sh select valid $S
done
#    «ESITO REPLICA: FAIL» -> quella replica si ferma (le altre proseguono): vedi «Se C3 fallisce» sotto

# FASE 3 — GPU: test con alpha* della replica (09 rifiuta di partire senza select_valid.json)
for S in $SEEDS; do
  GT=$(sbatch --parsable scripts/evaluation/09_queryvec_graph.sh test $S)
  FT=$(sbatch --parsable --dependency=afterok:$GT scripts/evaluation/10_late_fusion.sh test $S)
  echo "S=$S graph_test=$GT fusion_test=$FT"
done

# FASE 4 — CPU: controlli e numeri del test
for S in $SEEDS; do
  bash scripts/evaluation/15_seed_fusion_select.sh check  test $S && \
  bash scripts/evaluation/15_seed_fusion_select.sh select test $S
done

# FASE 5 — CPU: tabella fra TUTTE le repliche (trova da sé le cartelle seeds/s<S>) + figura
python -m src.evaluation.seed_summary                                   # -> results/fusion/seeds/summary.json
python -m src.figures.damage_test_seeds --out-dir figures/italian       # repliche finite trovate da sé
python -m src.figures.damage_test_seeds --lang en --out-dir figures/english
```

**Se C3 fallisce** (la fusione con α=0 non riproduce il graph, o con α=1 il vision): stop per quella
replica e diagnosi. Precedente (§57.2): differenze tutte **pareggi alla precisione float32** di FAISS →
accettato dall'utente con un file di deroga. Diagnosi per il caso graph (α=0), con `S`, `SPLIT`, `F` presi
dalla riga FAIL del check. ⚠️ Va fatta **subito**, prima della fase 3 della stessa replica: ogni
valutazione del graph (09 test compreso) riscrive `embeddings.npy` e, per il non determinismo della GPU, la
gallery cambia di qualche ulp (visto il 1 ott: s300042 valid 624e… → test ef6b…). Lo script si ferma se la
gallery su disco non è quella del check:

```bash
S=400042 SPLIT=valid F=0.75 python - <<'PY'
import json, os, numpy as np
from src.evaluation.query_vectors import load_qvec
S, SP, F = os.environ["S"], os.environ["SPLIT"], os.environ["F"]
fa = np.load(f"results/perquery/seeds/s{S}/fusion_{SP}/fusion_a0_partial-nowalls-random-f{F}_{SP}.npz")
gb = np.load(f"results/perquery/seeds/s{S}/fusion_branches_{SP}/graph_gat_asymrob_s{S}_partial-random-f{F}_{SP}.npz")
q = load_qvec(f"results/queryvec/seeds/s{S}/{SP}/graph_gat_asymrob_s{S}_partial-random-f{F}_{SP}.npz")
from src.evaluation.query_vectors import array_sha1
G = np.load(f"embeddings/graph/gat/asymrob_s{S}/embeddings.npy")
assert array_sha1(G) == q.meta["gallery_vectors"]["sha1"], "gallery riscritta dopo il check: diagnosi non valida"
G = G.astype(np.float64)
idx = {n: i for i, n in enumerate(json.load(open(f"embeddings/graph/gat/asymrob_s{S}/names.json")))}
diff = np.where(fa["self_rr"] != gb["self_rr"])[0]; ok = 0
for i in diff:
    row = idx[str(fa["names"][i])]; s = G @ q.vectors[i].astype(np.float64); x = s[row]
    ulp = float(np.spacing(np.float32(x))); o = np.delete(s, row)
    lo, hi = int((o > x + ulp).sum()) + 1, int((o >= x - ulp).sum()) + 1
    ok += all(lo <= round(1 / rr) <= hi for rr in (fa["self_rr"][i], gb["self_rr"][i]))
print(f"{ok}/{len(diff)} differenze spiegate da pareggi entro ±1 ulp float32")
PY
```

Solo se stampa **N/N** (tutte) e **l'utente approva**, si scrive la deroga e si riprende da `select`
(`15_seed_fusion_select.sh` la legge e la stampa); poi una riga in `status.md` sotto §57. Modello del
file `results/fusion/seeds/s<S>/c3_waiver_<split>.json` (precedente: `seeds/s300042/c3_waiver_valid.json`):

```json
{"accepted": true, "date": "<data>", "approved_by": "<utente>",
 "rule": "status.md §50/§57.2: pareggi alla precisione float32",
 "reason": "C3 alpha=0 vs graph f=<F>: <k>/2000 diversi, <k>/<k> entro ±1 ulp float32"}
```

```bash
bash scripts/evaluation/15_seed_fusion_select.sh select $SPLIT $S       # dopo aver scritto la deroga
```

Se **non** sono tutte spiegate, o il FAIL è su α=1 (vision) o su C1/C2: stop e diagnosi prima di andare
avanti. **Mai scartare** una replica completata; una replica fallita per l'infrastruttura (job morto,
timeout) si rilancia **con lo stesso S** (`FORCE=1` davanti allo `sbatch` dello step da rifare).

Durata indicativa per replica (`sacct` del 1 ott): training 0:45-1:48 · 08 ~1:00-1:30 per split · 09 ~7 min
per split · 10 valid ~1:20, test ~0:23 (CPU). ≈ 4-5 GPU-h a replica.
