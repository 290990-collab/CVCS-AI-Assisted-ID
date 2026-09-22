# src/vision/models/retrieval_model.py

import torch
import numpy as np
import faiss
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.vision.models.vision_encoders import BaseVisionEncoder
from src.vision.data.preprocess import get_transform, load_image
from src.vision.data.loader import RPLANDataset


class VisionRetrievalPipeline:
    """
    Pipeline completa: vision-based retrieval, agnostica rispetto all'encoder.

    Fasi:
      1. extract_embeddings() — passa tutto il dataset nell'encoder
      2. fit_whitening()      — PCA whitening sugli embedding
      3. build_index()        — costruisce l'indice FAISS
      4. query()              — dato un'immagine query, restituisce i top-k simili
    """

    def __init__(
        self,
        encoder: BaseVisionEncoder,
        transform=None,
        device: str = "cpu"
    ):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.device = device

        # preprocessing dell'encoder
        self.transform = transform if transform is not None else get_transform()

        self.raw_embeddings = None   # np.ndarray [N, D] embedding RAW dell'encoder
        self.embeddings  = None   # np.ndarray [N, D'] gallery indicizzata (post-trasformazioni)
        self.image_paths = []     # lista di path, indice i → path i
        self.index       = None   # indice FAISS

        # trasformazioni applicate sopra il raw (frozen): head opzionale + whitening
        self.head          = None   # ProjectionHead o None
        self.whiten_mean   = None   # np.ndarray [1, D]
        self.whiten_matrix = None   # np.ndarray [D, D']

    # ------------------------------------------------------------------
    # FASE 1 — estrazione embedding
    # ------------------------------------------------------------------

    def extract_embeddings(
        self,
        data_dir: str | None = None,
        batch_size: int = 128,
        save_path: str | None = None,
        image_paths: list | None = None,
    ) -> np.ndarray:
        """
        Estrae gli embedding per le immagini specificate.

        Args:
            data_dir:    cartella con i PNG RPLAN
            batch_size:  immagini per batch
            save_path:   se specificato, salva embeddings e paths su disco
            image_paths: lista esplicita di path da indicizzare (alternativa a
                         data_dir)

        Returns:
            embeddings: array numpy [N, embedding_dim]
        """
        dataset = RPLANDataset(
            data_dir=data_dir,
            image_paths=image_paths,
            transform=self.transform,
        )
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,        
            num_workers=6,
            pin_memory=(self.device != "cpu")
        )

        all_embeddings = []
        all_paths      = []

        print(f"[Retrieval] Estrazione embedding su {len(dataset)} immagini...")

        with torch.no_grad():
            for images, paths in tqdm(loader, desc="Encoding"): 
                images = images.to(self.device)
                embs   = self.encoder(images)               

                # ordine preservato tra embeddings e paths
                all_embeddings.append(embs.cpu().numpy())
                all_paths.extend(paths)

        self.embeddings     = np.vstack(all_embeddings).astype("float32")  # [N, D] (D = encoder.embedding_dim)
        self.raw_embeddings = self.embeddings                               # in indicizzazione salviamo il RAW
        self.image_paths    = all_paths

        print(f"[Retrieval] Embedding estratti: {self.embeddings.shape}")

        if save_path is not None:
            self._save(save_path)

        return self.embeddings

    # ------------------------------------------------------------------
    # FASE 1.5 — whitening (centering + PCA whitening)
    # ------------------------------------------------------------------

    def prepare_index(
        self,
        head=None,
        whiten: bool = False,
        eps: float = 1e-6,
        whiten_dim: int | None = None,
        fit_rows: list[int] | np.ndarray | None = None,
    ):
        """
        Prepara la gallery per la ricerca partendo dagli embedding RAW: applica
        (in ordine) la head opzionale, poi il whitening opzionale, infine L2; poi
        costruisce l'indice FAISS. Le stesse trasformazioni si applicano alla query.

        Tutte le combinazioni (raw / whiten / head / head+whiten) riusano lo stesso
        `raw_embeddings` → si cambia contributo senza ri-estrarre nulla.

        Args:
            fit_rows: righe su cui STIMARE il whitening. None = tutta la gallery
                (protocollo trasduttivo, quello di tutte le run fino al 24 ago
                2026). Con le righe del train si rispetta il vincolo DURO 1
                («statistiche dal solo train»): la gallery indicizzata resta
                comunque INTERA, cambia solo l'insieme di stima.
        """
        assert self.raw_embeddings is not None, \
            "Chiama extract_embeddings() o load() prima di prepare_index()"

        self.head = head
        gallery = self._apply_head(self.raw_embeddings)

        if whiten:
            fit_on = gallery
            if fit_rows is not None:
                rows = np.asarray(fit_rows, dtype=np.int64)
                if rows.size == 0:
                    raise ValueError(
                        "fit_rows è vuoto: il whitening non ha righe su cui stimare. "
                        "Controlla whitening.fit_split e la presenza dei metadati .mat."
                    )
                fit_on = gallery[rows]
            print(f"[Retrieval] Whitening stimato su {len(fit_on)}/{len(gallery)} "
                  f"righe della gallery")
            self._fit_whitening(fit_on, eps=eps, dim=whiten_dim)
        else:
            self.whiten_mean = self.whiten_matrix = None

        self.embeddings = self._apply_whitening(gallery)
        self.build_index()

    def _apply_head(self, embs: np.ndarray) -> np.ndarray:
        """Proietta gli embedding con la head (a batch, no_grad). Identità se head=None."""
        if self.head is None:
            return embs.astype("float32")
        self.head.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(embs), 8192):
                x = torch.from_numpy(embs[i:i + 8192]).float().to(self.device)
                out.append(self.head(x).cpu().numpy())
        return np.vstack(out).astype("float32")

    def _fit_whitening(self, embs: np.ndarray, eps: float = 1e-6, dim: int | None = None):
        """
        Calcola i parametri di PCA whitening (mean + matrice) sugli embedding dati.
        Equalizza la varianza lungo le direzioni principali → ranking discriminativo.

        Args:
            embs: embedding su cui fittare (gallery, già post-head).
            eps:  regolarizzazione sulle direzioni a varianza quasi nulla.
            dim:  se specificato, riduce alle top-`dim` componenti principali.
        """
        # 1. centering — sottrai la media del dataset
        self.whiten_mean = embs.mean(axis=0, keepdims=True).astype("float32")
        centered = embs - self.whiten_mean

        # 2. eigendecomposition della covarianza (D × D, veloce)
        N = centered.shape[0]
        cov = (centered.T @ centered) / max(N - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 3. ordina per varianza decrescente (eigh la restituisce crescente) e,
        #    se richiesto, tieni solo le top-`dim` componenti principali
        order = np.argsort(eigvals)[::-1]
        if dim is not None:
            order = order[:int(dim)]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # 4. matrice di whitening: proietta sulle componenti principali
        #    e scala ciascuna per 1/σ → varianza unitaria su tutte le direzioni
        self.whiten_matrix = (eigvecs / np.sqrt(eigvals + eps)).astype("float32")
        print(f"[Retrieval] Whitening fittato: matrix {self.whiten_matrix.shape}")

    def _apply_whitening(self, embs: np.ndarray) -> np.ndarray:
        """
        Applica centering + PCA whitening + L2 normalize a un batch di embedding.
        Se whiten_mean/matrix non sono settati, ritorna gli embedding L2-normalizzati senza modifiche.
        """
        embs = embs.astype("float32")
        if self.whiten_mean is not None and self.whiten_matrix is not None:
            embs = (embs - self.whiten_mean) @ self.whiten_matrix
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return (embs / np.maximum(norms, 1e-12)).astype("float32")

    # ------------------------------------------------------------------
    # FASE 2 — costruzione indice FAISS
    # ------------------------------------------------------------------

    def build_index(self) -> faiss.Index:
        """
        Costruisce un indice FAISS IndexFlatIP (inner product).
        """
        assert self.embeddings is not None, \
            "Chiama extract_embeddings() prima di build_index()"

        dim         = self.embeddings.shape[1]
        self.index  = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        print(f"[Retrieval] Indice FAISS costruito: {self.index.ntotal} vettori, dim={dim}")
        return self.index

    # ------------------------------------------------------------------
    # FASE 3 — query
    # ------------------------------------------------------------------

    def load_query(self, path: str) -> torch.Tensor:
        """
        Carica un'immagine di query applicando il transform.

        Args:
            path: percorso al PNG di query.

        Returns:
            tensore [3, H, W] pronto per query().
        """
        return load_image(path, self.transform)

    def query(
        self,
        query_image: torch.Tensor,
        top_k: int = 5,
        return_embeddings: bool = False,
    ) -> list[dict]:
        """
        Dato un tensore immagine, restituisce i top-k floor plan più simili.

        Args:
            query_image: tensore [3, 224, 224] o [1, 3, 224, 224]
            top_k:       numero di risultati da restituire
            return_embeddings: if True (late fusion, 16 Sep 2026) also return the
                         query vectors of THIS forward: (results, query_raw [1, D],
                         query_final [1, D']). False = historical return, unchanged.

        Returns:
            lista di dizionari con 'path' e 'score' (cosine similarity)
        """
        assert self.index is not None, \
            "Chiama build_index() prima di query()"

        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)

        with torch.no_grad():
            query_emb = self.encoder(query_image.to(self.device))
            query_np  = query_emb.cpu().numpy().astype("float32")

        query_raw = query_np   # [1, D] RAW encoder output (the head/whitening below copy)

        # stesse trasformazioni della gallery: head opzionale, poi whitening + L2
        query_np = self._apply_head(query_np)
        query_np = self._apply_whitening(query_np)

        # cosine similarity tra query e gallery calcolata sull'indice
        scores, indices = self.index.search(query_np, top_k)     

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "path":  self.image_paths[idx],
                "score": float(score)       
            })

        if return_embeddings:
            return results, query_raw, query_np
        return results

    # ------------------------------------------------------------------
    # salvataggio e caricamento su disco
    # ------------------------------------------------------------------

    def _save(self, save_dir: str):
        """
        Salva gli embedding RAW e i path. Whitening e head NON sono persistiti qui:
        sono trasformazioni applicate al volo da prepare_index (whitening) o
        allenate a parte (head.pt) → lo stesso raw alimenta tutte le combinazioni.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / "embeddings.npy", self.raw_embeddings)
        with open(save_dir / "image_paths.json", "w") as f:
            json.dump(self.image_paths, f)

        print(f"[Retrieval] Salvato in {save_dir}")


    def load(self, save_dir: str):
        """
        Carica gli embedding RAW e i path. NON costruisce l'indice: il chiamante
        decide le trasformazioni con prepare_index(head=..., whiten=...).
        """
        save_dir = Path(save_dir)

        self.raw_embeddings = np.load(save_dir / "embeddings.npy")
        self.embeddings     = self.raw_embeddings
        with open(save_dir / "image_paths.json") as f:
            self.image_paths = json.load(f)

        print(f"[Retrieval] Caricato da {save_dir}: {len(self.image_paths)} floor plan "
              f"(raw {self.raw_embeddings.shape})")
