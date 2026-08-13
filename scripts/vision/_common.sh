# scripts/vision/_common.sh
# Impostazioni condivise dai job vision (da "source"-are in ogni script).
# Centralizza la lista modelli, i pooling VALIDI per encoder e le risoluzioni:
# la griglia resta coerente e si modifica in un solo posto.

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"

# Modelli da processare: l'argomento $1 (un solo modello) oppure tutti e 8.
# ATTENZIONE: la usa anche scripts/evaluation/02_perquery_vision_valid.sh, che
# senza argomenti gira sull'INTERA lista. Aggiungere un modello qui allarga
# quella griglia (56 run con i 5 encoder storici, 80 con tipsv2, 92 con pecore,
# 104 con pespatial; feature RAW: 14 -> 20 -> 23 -> 26):
# per valutare solo il nuovo encoder passarlo esplicitamente come argomento.
select_models() {
  if [ -n "$1" ]; then echo "$1"; else echo "dinov2 dinov3 siglip2 radio ijepa tipsv2 pecore pespatial"; fi
}

# Pooling validi per encoder. I-JEPA non ha il token CLS: il suo pooling naturale
# E' GIA' la media dei patch, quindi "mean" coinciderebbe con "natural" -> escluso.
poolings_for() {
  case "$1" in
    ijepa) echo "natural gem" ;;
    *)     echo "natural gem mean" ;;
  esac
}

# Risoluzioni da processare per encoder.
#   "native" = quella del preset configs/vision_models/<modello>.yaml, cioe' il
#              comportamento storico: UNA risoluzione e variante = pooling.
# TIPSv2 e' l'unico encoder con risoluzione nativa != 224 (448, ViT-B/14): il
# confronto con gli altri cinque richiede entrambe, quindi la risoluzione entra
# nel NOME DELLA VARIANTE. Senza suffisso le due scriverebbero nella stessa
# cartella RAW (embeddings/vision/tipsv2/<variante>/) sovrascrivendosi.
resolutions_for() {
  case "$1" in
    tipsv2) echo "448 224" ;;
    *)      echo "native" ;;
  esac
}

# Risoluzioni effettive: l'argomento esplicito (per lanciare un job per
# risoluzione, in parallelo) oppure tutte quelle previste per il modello.
#   $1 = modello, $2 = risoluzione richiesta (opzionale)
select_resolutions() {
  if [ -n "${2:-}" ]; then echo "$2"; else resolutions_for "$1"; fi
}

# Nome della variante = sottocartella di embeddings/vision/<modello>/.
# Con risoluzione nativa resta il solo pooling: i RAW gia' su disco dei cinque
# encoder del benchmark NON cambiano nome.
#   $1 = pooling, $2 = risoluzione
variant_for() {
  if [ "$2" = "native" ]; then echo "$1"; else echo "$1$2"; fi
}

# Override da passare a python per una risoluzione non nativa (stringa vuota se
# nativa: i comandi dei cinque encoder restano identici a prima).
# Il batch scala con la risoluzione: senza xformers l'attenzione materializza
# [B, heads, N, N], quindi la memoria cresce con B*N^2 e non con B*N. A 448 i
# token sono 1026 contro i 258 di 224 (~16x la matrice di attenzione a parita'
# di batch): col batch 256 del config si va in OOM su una scheda da 11G.
#   $1 = risoluzione
res_flags() {
  if [ "$1" = "native" ]; then
    echo ""
  elif [ "$1" -ge 384 ] 2>/dev/null; then
    echo "model.kwargs.image_size=$1 retrieval.batch_size=32"
  else
    echo "model.kwargs.image_size=$1"
  fi
}

# Etichetta per i log: "dinov2/natural" oppure "tipsv2/natural@448".
#   $1 = modello, $2 = pooling, $3 = risoluzione
label_for() {
  if [ "$3" = "native" ]; then echo "$1/$2"; else echo "$1/$2@$3"; fi
}
