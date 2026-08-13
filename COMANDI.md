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
sbatch scripts/evaluation/01_random_floor.sh            # 4 run: {vision,graph} x {casuale,costante}
# ⚠️ sul login node muore: ulimit -t = 600 s di CPU. Output: results/random_floor/

# 02+03 — vision: si sceglie sul VALID, il test si legge UNA volta sola
sbatch scripts/evaluation/02_perquery_vision_valid.sh          # griglia 104 (56 pre-registrate + tipsv2 + pecore + pespatial) -> RISCRIVE
sbatch scripts/evaluation/02_perquery_vision_valid.sh "" frozen  # solo raw+whiten (52)
sbatch scripts/evaluation/02_perquery_vision_valid.sh pespatial frozen  # un encoder solo: 6 run, niente riscritture
# ⚠️ `frozen` e' obbligatorio per i 3 encoder nuovi: non hanno head.pt, il gruppo `head` fallirebbe.
sbatch scripts/evaluation/03_perquery_vision_test.sh dinov3 gem whiten

# 04 — graph: le varianti sono gia' scelte sul valid dalla probe -> bastano i per-query
sbatch scripts/evaluation/04_perquery_graph.sh          # hist + 3 vincenti + 3 base
```

`04` non duplica il ponte YAML→flag: esporta `PERQUERY_OUT` e delega a
`scripts/graph/04_eval_gnn.sh`, che senza quella variabile si comporta
**esattamente** come prima.

Poi, su CPU e senza rilanciare nulla:

```bash
python -m src.evaluation.significance --a A.npz --b B.npz --k 10   # e' un guadagno o rumore?
python -m src.evaluation.geometry_variants --gallery <json> --perquery A.npz B.npz
```
